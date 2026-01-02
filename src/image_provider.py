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
from typing import Any, Optional

import aiohttp

from .config import config
from .constants import (
    GOOGLE_IMAGE_BASE_URL,
    IMAGE_PROVIDER_TIMEOUT_SEC,
    MAX_RETRIES,
    RETRY_BASE_SECONDS,
)
from .models import ImageAspectRatio, ImageModels, ImageProviders, ImageResolution
from .rate_limiter import get_rate_limiter

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
    ) -> bytes:
        # Safeguard: Ensure provider matches model to avoid config mismatch errors
        if model in MODEL_MAP:
            provider = "replicate"
        elif "gemini" in model:
            provider = "google"
        elif "gpt-image" in model or "dall-e" in model:
            provider = "openai"

        if provider == "replicate":
            return await self._get_replicate_image(
                prompt, model, aspect_ratio=aspect_ratio
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
        retry_count: int = 0,
        **kwargs
    ) -> bytes:
        limiter = get_rate_limiter(provider)
        
        try:
            async with limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, 
                        headers=headers, 
                        json=json_payload, 
                        timeout=timeout_sec,
                        **kwargs
                    ) as response:
                        if response.status == 429:
                            await limiter.report_failure()
                            
                            if retry_count < MAX_RETRIES:
                                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                                logger.debug(f"{provider} 429: Waiting {wait_time}s before retry {retry_count + 1}")
                                await asyncio.sleep(wait_time)
                                return await self._execute_request(
                                    url, headers, json_payload, timeout_sec, provider, retry_count + 1, **kwargs
                                )
                            
                            response.raise_for_status()
                        
                        response.raise_for_status()
                        await limiter.report_success()
                        return await response.read()

        except asyncio.TimeoutError:
            from .logger import logger
            logger.warning(f"{provider} request timed out")
            await limiter.report_failure()
            
            if retry_count < MAX_RETRIES:
                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                await asyncio.sleep(wait_time)
                return await self._execute_request(
                    url, headers, json_payload, timeout_sec, provider, retry_count + 1, **kwargs
                )
            raise

        except Exception:
            await limiter.report_failure()
            raise

    async def _get_replicate_image(
        self,
        prompt: str,
        model: ImageModels,
        aspect_ratio: Optional[ImageAspectRatio] = None,
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

        url = f"{REPLICATE_API_BASE}/models/{model_path}/predictions"
        payload = {
            "input": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio or "1:1",
                "output_format": "webp",
            }
        }

        # NOTE: Replicate is complex because of Polling.
        # Retry logic for creation is easy, but polling retry logic is harder.
        # For now, let's implement basic retry for the CREATION request.
        # If creation succeeds but polling fails/times out, we don't retry currently.
        
        return await self._get_replicate_image_with_retry(
            url, headers, payload, "replicate"
        )
        
    async def _get_replicate_image_with_retry(
        self, url, headers, payload, provider, retry_count=0
    ) -> bytes:
        limiter = get_rate_limiter(provider)
        
        try:
            async with limiter:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, headers=headers, json=payload, timeout=IMAGE_PROVIDER_TIMEOUT_SEC
                    ) as response:
                        if response.status == 429:
                            await limiter.report_failure()
                            if retry_count < MAX_RETRIES:
                                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                                await asyncio.sleep(wait_time)
                                return await self._get_replicate_image_with_retry(
                                    url, headers, payload, provider, retry_count + 1
                                )
                            response.raise_for_status()

                        response.raise_for_status()
                        prediction = await response.json()

                    # Check prediction status immediately if 'Prefer: wait' worked
                    if prediction.get("status") == "succeeded":
                        await limiter.report_success()
                        output_url = prediction["output"][0]
                        async with aiohttp.ClientSession() as dl_session:
                            return await self._download_image(dl_session, output_url)

                    get_url = prediction["urls"]["get"]
                    start_time = asyncio.get_event_loop().time()
                    
                    await limiter.report_success()
                    
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
                            
        except Exception:
            # If we retry, we handle it above. If we get here, it's a hard fail.
            # Only catch if we want to report failure to limiter.
            # Limiter report failure was called if 429.
            # If other exception, report failure too.
            await limiter.report_failure()
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
             image_config["imageSize"] = "4K"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"imageConfig": image_config},
        }

        # Use specific limiter for Gemini Image generation
        data_bytes = await self._execute_request(
            url, headers, payload, IMAGE_PROVIDER_TIMEOUT_SEC, "google_image"
        )
        
        import json
        data = json.loads(data_bytes)

        image_bytes = self._extract_google_image_bytes(data)
        if not image_bytes:
            raise Exception("No image bytes returned from Google.")
        return base64.b64decode(image_bytes)

    def _extract_google_image_bytes(self, data: dict[str, Any]) -> Optional[str]:
        if not data:
            return None

        # New API format (gemini-3-pro-image-preview)
        # { "candidates": [ { "content": { "parts": [ { "inlineData": { ... } } ] } } ] }
        if candidates := data.get("candidates"):
            if isinstance(candidates, list) and candidates:
                first_candidate = candidates[0]
                if content := first_candidate.get("content"):
                    if parts := content.get("parts"):
                        for part in parts:
                            if inline_data := part.get("inlineData"):
                                if image_bytes := inline_data.get("data"):
                                    return image_bytes

        # Legacy formats
        if image := data.get("image"):
            if image_bytes := image.get("imageBytes"):
                return image_bytes
        if images := data.get("images"):
            if isinstance(images, list) and images:
                first = images[0]
                if image_bytes := first.get("imageBytes"):
                    return image_bytes
        if image_bytes := data.get("imageBytes"):
            return image_bytes
        if outputs := data.get("outputs"):
            if isinstance(outputs, list) and outputs:
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
    ) -> bytes:
        api_key = config.openai_api_key
        if not api_key:
            raise Exception("OpenAI API key not found. Please set it in the settings.")
        
        # ... logic for size ...
        
        url = f"{config.openai_endpoint or 'https://api.openai.com'}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Handle size
        size = "1024x1024"
        if aspect_ratio:
            if aspect_ratio == "1:1":
                size = "1024x1024"
            elif aspect_ratio == "16:9":
                size = "1792x1024"  # Closest to 16:9 for DALL-E 3/GPT-Image
            elif aspect_ratio == "9:16":
                size = "1024x1792"
            # For other ratios, default to square or closest available if model supports it.
            # DALL-E 3 standard sizes are 1024x1024, 1024x1792, 1792x1024.
            # GPT-Image models might support "auto" or these sizes.
            # Using 1024x1024 as safe default if ratio isn't standard supported by model.
        
        # Use resolution parameter if provided and valid for standard square
        if resolution == "2048x2048" and model == "dall-e-3":
             pass

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
             elif aspect_ratio == "16:9" or aspect_ratio == "4:3": # Approx
                 payload["size"] = "1536x1024"
             elif aspect_ratio == "9:16" or aspect_ratio == "3:4": # Approx
                 payload["size"] = "1024x1536"
             else:
                 payload["size"] = "1024x1024" # Default
        else:
             if aspect_ratio == "1:1":
                 payload["size"] = "1024x1024"
             elif aspect_ratio == "16:9":
                 payload["size"] = "1792x1024"
             elif aspect_ratio == "9:16":
                 payload["size"] = "1024x1792"
             else:
                 payload["size"] = "1024x1024"

        data_bytes = await self._execute_request(
            url, headers, payload, IMAGE_PROVIDER_TIMEOUT_SEC, "openai"
        )
        
        import json
        data = json.loads(data_bytes)
        
        try:
            b64_json = data["data"][0]["b64_json"]
            return base64.b64decode(b64_json)
        except (KeyError, IndexError):
            raise Exception("Invalid response format from OpenAI Image API")


image_provider = ImageProvider()
