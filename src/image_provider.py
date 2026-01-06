"""
Copyright (C) 2024 Michael Piazza

This file is part of Smart Notes.

Smart Notes is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Smart Notes is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Smart Notes.  If not, see <https://www.gnu.org/licenses/>.
"""

import asyncio
import base64
import json
from typing import Any, Optional

import aiohttp

from .config import config
from .constants import (
    GOOGLE_IMAGE_BASE_URL,
    IMAGE_PROVIDER_TIMEOUT_SEC,
    MAX_RETRIES,
    RETRY_BASE_SECONDS,
)
from .logger import logger
from .models import ImageAspectRatio, ImageModels, ImageProviders, ImageResolution
from .rate_limiter import (
    estimate_tokens,
    extract_google_token_usage,
    extract_rate_limit_headers,
    get_rate_limiter,
    parse_retry_after,
)

REPLICATE_API_BASE = "https://api.replicate.com/v1"

MODEL_MAP = {
    "flux-dev": "black-forest-labs/flux-dev",
    "flux-schnell": "black-forest-labs/flux-schnell",
}


class ImageProvider:
    async def async_get_image_response(
        self,
        prompt: str,
        model: ImageModels,
        provider: ImageProviders,
        note_id: int,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        resolution: Optional[ImageResolution] = None,
        output_format: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> bytes:
        # Check custom providers
        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        if custom_provider:
            return await self._get_openai_image(
                prompt, model, aspect_ratio, resolution, custom_provider
            )

        # Safeguard: Ensure provider matches model to avoid config mismatch errors
        if model in MODEL_MAP:
            provider = "replicate"
        elif "gemini" in model:
            provider = "google"
        elif "gpt-image" in model or "dall-e" in model:
            provider = "openai"

        if provider == "replicate":
            return await self._get_replicate_image(
                prompt,
                model,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                quality=quality,
            )
        if provider == "google":
            return await self._get_google_image(
                prompt, model, aspect_ratio=aspect_ratio, resolution=resolution
            )
        if provider == "openai":
            return await self._get_openai_image(
                prompt, model, aspect_ratio=aspect_ratio, resolution=resolution
            )
        raise ValueError(f"Unknown image provider: {provider}")

    async def _execute_request(
        self,
        url: str,
        headers: Optional[dict],
        json_payload: Optional[dict],
        timeout_sec: int,
        provider: str,
        model: str = "",
        retry_count: int = 0,
        estimated_tokens: int = 0,
        **kwargs,
    ) -> bytes:
        # Get per-model rate limiter
        limiter = get_rate_limiter(provider, model)

        try:
            # Acquire rate limit slot with token estimate
            await limiter.acquire(estimated_tokens)

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url,
                    headers=headers,
                    json=json_payload,
                    timeout=timeout_sec,
                    **kwargs,
                ) as response,
            ):
                # Extract headers for rate limit learning
                response_headers = extract_rate_limit_headers(response.headers)

                if response.status == 429:
                    retry_after = parse_retry_after(response.headers)
                    await limiter.report_failure(response_headers, retry_after)

                    if retry_count < MAX_RETRIES:
                        wait_time = retry_after or (2**retry_count) * RETRY_BASE_SECONDS
                        logger.debug(
                            f"{provider} 429: Waiting {wait_time}s before retry {retry_count + 1}"
                        )
                        await asyncio.sleep(wait_time)
                        return await self._execute_request(
                            url,
                            headers,
                            json_payload,
                            timeout_sec,
                            provider,
                            model,
                            retry_count + 1,
                            estimated_tokens,
                            **kwargs,
                        )

                    response.raise_for_status()

                response.raise_for_status()
                response_bytes = await response.read()

                # For Google providers, extract actual token usage from response body
                actual_tokens = estimated_tokens
                if "google" in provider:
                    actual_tokens = extract_google_token_usage(
                        response_bytes, estimated_tokens
                    )

                await limiter.report_success(actual_tokens, response_headers)
                return response_bytes

        except asyncio.TimeoutError:
            logger.warning(f"{provider} request timed out")
            await limiter.report_timeout()

            if retry_count < MAX_RETRIES:
                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                await asyncio.sleep(wait_time)
                return await self._execute_request(
                    url,
                    headers,
                    json_payload,
                    timeout_sec,
                    provider,
                    model,
                    retry_count + 1,
                    estimated_tokens,
                    **kwargs,
                )
            raise

        except aiohttp.ClientResponseError as e:
            if e.status != 429:
                logger.warning(f"{provider} returned status {e.status}: {e.message}")

            # Only report failure to rate limiter for rate-limit related errors
            # 429 = Too Many Requests, 5xx = Server errors (might indicate overload)
            # Don't penalize rate limits for client errors like 400, 401, 403, 404
            if e.status == 429 or e.status >= 500:
                await limiter.report_failure()
            raise

        except Exception:
            # Don't report generic exceptions to rate limiter - they're not rate limit issues
            raise

    async def _get_replicate_image(
        self,
        prompt: str,
        model: ImageModels,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        output_format: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> bytes:
        api_key = config.replicate_api_key
        if not api_key:
            raise Exception("Replicate API key not found.")

        model_path = MODEL_MAP.get(model)
        if not model_path:
            raise ValueError(f"Unknown Replicate model: {model}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }

        # Handle Replicate output format
        # Flux supports: webp, jpg, png.
        # If user requests avif, we should ask for png (lossless) and convert later.
        repl_format = "webp"
        if output_format:
            if output_format in ["webp", "jpg", "png"]:
                repl_format = output_format
            elif output_format == "jpeg":
                repl_format = "jpg"
            elif output_format == "avif":
                repl_format = "png"  # Get lossless then convert

        # Handle Replicate quality
        # Flux supports 0-100.
        repl_quality = 80
        if quality is not None and quality > 0:
            repl_quality = quality

        url = f"{REPLICATE_API_BASE}/models/{model_path}/predictions"
        payload = {
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio or "1:1",
                "output_format": repl_format,
                "output_quality": repl_quality,
            }
        }

        # NOTE: Replicate is complex because of Polling.
        # Retry logic for creation is easy, but polling retry logic is harder.
        # For now, let's implement basic retry for the CREATION request.
        # If creation succeeds but polling fails/times out, we don't retry currently.

        # Estimate tokens for rate limiting (prompt only for image gen)
        estimated_tokens = estimate_tokens(prompt)

        return await self._get_replicate_image_with_retry(
            url, headers, payload, "replicate", model, estimated_tokens=estimated_tokens
        )

    async def _get_replicate_image_with_retry(
        self,
        url: str,
        headers: dict,
        payload: dict,
        provider: str,
        model: str,
        retry_count: int = 0,
        estimated_tokens: int = 0,
    ) -> bytes:
        # Get per-model rate limiter
        limiter = get_rate_limiter(provider, model)

        try:
            await limiter.acquire(estimated_tokens)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=IMAGE_PROVIDER_TIMEOUT_SEC,
                ) as response:
                    response_headers = self._extract_rate_limit_headers(
                        response.headers
                    )

                    if response.status == 429:
                        retry_after = self._parse_retry_after(response.headers)
                        await limiter.report_failure(response_headers, retry_after)

                        if retry_count < MAX_RETRIES:
                            wait_time = (
                                retry_after or (2**retry_count) * RETRY_BASE_SECONDS
                            )
                            await asyncio.sleep(wait_time)
                            return await self._get_replicate_image_with_retry(
                                url,
                                headers,
                                payload,
                                provider,
                                model,
                                retry_count + 1,
                                estimated_tokens,
                            )
                        response.raise_for_status()

                    response.raise_for_status()
                    prediction = await response.json()

                # Check prediction status immediately if 'Prefer: wait' worked
                if prediction.get("status") == "succeeded":
                    await limiter.report_success(estimated_tokens, response_headers)
                    output_url = prediction["output"][0]
                    async with aiohttp.ClientSession() as dl_session:
                        return await self._download_image(dl_session, output_url)

                get_url = prediction["urls"]["get"]
                start_time = asyncio.get_event_loop().time()

                await limiter.report_success(estimated_tokens, response_headers)

                async with aiohttp.ClientSession() as session:
                    while True:
                        if (
                            asyncio.get_event_loop().time() - start_time
                            > IMAGE_PROVIDER_TIMEOUT_SEC
                        ):
                            raise TimeoutError("Replicate prediction timed out.")

                        await asyncio.sleep(1)

                        async with session.get(get_url, headers=headers) as response:
                            response.raise_for_status()
                            prediction = await response.json()

                        status = prediction.get("status")
                        if status == "succeeded":
                            output_url = prediction["output"][0]
                            return await self._download_image(session, output_url)
                        if status == "failed":
                            raise Exception(
                                f"Replicate prediction failed: {prediction.get('error')}"
                            )
                        if status == "canceled":
                            raise Exception("Replicate prediction canceled.")

        except aiohttp.ClientResponseError as e:
            # Only report failure for rate limit related errors (429, 5xx)
            if e.status == 429 or e.status >= 500:
                await limiter.report_failure()
            raise

        except Exception:
            # Don't report generic exceptions to rate limiter
            raise

    async def _download_image(self, session: aiohttp.ClientSession, url: str) -> bytes:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    async def _get_google_image(
        self,
        prompt: str,
        model: ImageModels,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        resolution: Optional[ImageResolution] = None,
    ) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        # Handle legacy model name if passed in somehow
        if model == "gemini-nano-banana-pro":
            model = "gemini-3-pro-image-preview"

        url = f"{GOOGLE_IMAGE_BASE_URL}/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        image_config: dict[str, Any] = {
            "aspectRatio": aspect_ratio or "1:1",
        }

        if resolution and resolution == "2048x2048":
            image_config["imageSize"] = "2K"
        elif resolution and resolution == "4096x4096":
            image_config["imageSize"] = "4K"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"imageConfig": image_config},
        }

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(prompt)

        # Use specific limiter for Gemini Image generation
        data_bytes = await self._execute_request(
            url,
            headers,
            payload,
            IMAGE_PROVIDER_TIMEOUT_SEC,
            "google_image",
            model,
            estimated_tokens=estimated_tokens,
        )

        data = json.loads(data_bytes)

        # Check for error in response body (Google sometimes returns 200 with error)
        if "error" in data:
            error_msg = data.get("error", {})
            if isinstance(error_msg, dict):
                error_text = error_msg.get("message", str(error_msg))
                error_code = error_msg.get("code", "unknown")
                error_status = error_msg.get("status", "")
            else:
                error_text = str(error_msg)
                error_code = "unknown"
                error_status = ""
            logger.error(
                f"Google Image API returned error: [{error_code}] {error_status}: {error_text}"
            )
            raise Exception(f"Google Image API error: {error_text}")

        image_bytes = self._extract_google_image_bytes(data)
        if not image_bytes:
            logger.error(f"No image bytes in response: {data}")
            raise Exception("No image bytes returned from Google.")
        return base64.b64decode(image_bytes)

    def _extract_google_image_bytes(self, data: dict[str, Any]) -> Optional[str]:
        if not data:
            return None

        # New API format (gemini-3-pro-image-preview)
        # { "candidates": [ { "content": { "parts": [ { "inlineData": { ... } } ] } } ] }
        if (
            (candidates := data.get("candidates"))
            and isinstance(candidates, list)
            and candidates
        ):
            first_candidate = candidates[0]
            if (content := first_candidate.get("content")) and (
                parts := content.get("parts")
            ):
                for part in parts:
                    if (inline_data := part.get("inlineData")) and (
                        image_bytes := inline_data.get("data")
                    ):
                        return image_bytes

        # Legacy formats
        if (image := data.get("image")) and (image_bytes := image.get("imageBytes")):
            return image_bytes
        if (images := data.get("images")) and isinstance(images, list) and images:
            first = images[0]
            if image_bytes := first.get("imageBytes"):
                return image_bytes
        if image_bytes := data.get("imageBytes"):
            return image_bytes
        if (outputs := data.get("outputs")) and isinstance(outputs, list) and outputs:
            first = outputs[0]
            if image_bytes := first.get("imageBytes"):
                return image_bytes
        return None

    async def _get_openai_image(
        self,
        prompt: str,
        model: ImageModels,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        resolution: Optional[ImageResolution] = None,
        provider_config: Optional[dict] = None,
    ) -> bytes:
        if provider_config:
            api_key = provider_config["api_key"]
            base_url = provider_config["base_url"]
            # Assume base_url is root or v1. If it ends with v1, we append images/generations
            if base_url.endswith("/v1"):
                url = f"{base_url}/images/generations"
            else:
                url = f"{base_url}/v1/images/generations"
            provider_name = provider_config["name"]
        else:
            api_key = config.openai_api_key
            if not api_key:
                raise Exception(
                    "OpenAI API key not found. Please set it in the settings."
                )
            url = f"{config.openai_endpoint or 'https://api.openai.com'}/v1/images/generations"
            provider_name = "openai"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Construct payload
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }

        if "gpt-image" in model:
            if aspect_ratio == "1:1":
                payload["size"] = "1024x1024"
            elif aspect_ratio == "16:9" or aspect_ratio == "4:3":  # Approx
                payload["size"] = "1536x1024"
            elif aspect_ratio == "9:16" or aspect_ratio == "3:4":  # Approx
                payload["size"] = "1024x1536"
            else:
                payload["size"] = "1024x1024"  # Default
        else:
            if aspect_ratio == "1:1":
                payload["size"] = "1024x1024"
            elif aspect_ratio == "16:9":
                payload["size"] = "1792x1024"
            elif aspect_ratio == "9:16":
                payload["size"] = "1024x1792"
            else:
                payload["size"] = "1024x1024"

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(prompt)

        data_bytes = await self._execute_request(
            url,
            headers,
            payload,
            IMAGE_PROVIDER_TIMEOUT_SEC,
            provider_name,
            model,
            estimated_tokens=estimated_tokens,
        )

        data = json.loads(data_bytes)

        try:
            b64_json = data["data"][0]["b64_json"]
            return base64.b64decode(b64_json)
        except (KeyError, IndexError) as e:
            raise Exception("Invalid response format from OpenAI Image API") from e


image_provider = ImageProvider()
