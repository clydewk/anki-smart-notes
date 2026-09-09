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

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, Literal

from .chat_provider import build_v1_endpoint
from .config import config
from .constants import GOOGLE_IMAGE_BASE_URL, MAX_RETRIES
from .logger import logger
from .models import image_generation_qualities
from .provider_runtime import (
    RequestTimeouts,
    provider_runtime,
)

if TYPE_CHECKING:
    from .models import (
        CustomProvider,
        ImageAspectRatio,
        ImageGenerationQuality,
        ImageModels,
        ImageProviders,
        ImageResolution,
    )

REPLICATE_API_BASE = "https://api.replicate.com/v1"
LegacyGoogleImageModel = Literal["gemini-nano-banana-pro"]

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
        aspect_ratio: ImageAspectRatio | None = None,
        resolution: ImageResolution | None = None,
        output_format: str | None = None,
        quality: int | None = None,
        generation_quality: ImageGenerationQuality | None = None,
    ) -> bytes:
        del note_id

        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        if custom_provider:
            return await self._get_openai_image(
                prompt,
                model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                provider_config=custom_provider,
                generation_quality=generation_quality,
            )

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
                prompt,
                model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
        if provider == "openai":
            return await self._get_openai_image(
                prompt,
                model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                generation_quality=generation_quality,
            )
        raise ValueError(f"Unknown image provider: {provider}")

    async def _execute_request(
        self,
        *,
        url: str,
        provider: str,
        model: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: object | None = None,
    ) -> bytes:
        return await provider_runtime.request_bytes(
            key=f"image:{provider}:{model}",
            initial_window=1,
            method="POST",
            url=url,
            headers=headers,
            json_payload=json_payload,
            data=data,
            provider=provider,
            model=model,
            timeouts=RequestTimeouts(
                connect_timeout_sec=10.0,
                sock_read_timeout_sec=180.0
                if model.startswith("gpt-image-2.5-")
                else 60.0,
            ),
            max_retries=MAX_RETRIES,
        )

    async def _get_replicate_image(
        self,
        prompt: str,
        model: ImageModels,
        aspect_ratio: ImageAspectRatio | None = None,
        output_format: str | None = None,
        quality: int | None = None,
    ) -> bytes:
        api_key = config.replicate_api_key
        if not api_key:
            raise Exception("Replicate API key not found.")

        model_path = MODEL_MAP.get(model)
        if not model_path:
            raise ValueError(f"Unknown Replicate model: {model}")

        repl_format = "webp"
        if output_format:
            if output_format in ["webp", "jpg", "png"]:
                repl_format = output_format
            elif output_format == "jpeg":
                repl_format = "jpg"
            elif output_format == "avif":
                repl_format = "png"

        repl_quality = 80
        if quality is not None and quality > 0:
            repl_quality = quality

        return await self._get_replicate_image_with_retry(
            url=f"{REPLICATE_API_BASE}/models/{model_path}/predictions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            payload={
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio or "1:1",
                    "output_format": repl_format,
                    "output_quality": repl_quality,
                }
            },
            provider="replicate",
            model=model,
        )

    async def _get_replicate_image_with_retry(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        provider: str,
        model: str,
    ) -> bytes:
        prediction = await provider_runtime.request_json(
            key=f"image:{provider}:{model}",
            initial_window=1,
            method="POST",
            url=url,
            headers=headers,
            json_payload=payload,
            provider=provider,
            model=model,
            timeouts=RequestTimeouts(
                connect_timeout_sec=10.0,
                sock_read_timeout_sec=60.0,
            ),
            max_retries=MAX_RETRIES,
        )
        if not isinstance(prediction, dict):
            raise Exception("Replicate returned an invalid prediction payload.")

        if prediction.get("status") == "succeeded":
            output = prediction.get("output", [])
            if not isinstance(output, list) or not output:
                raise Exception("Replicate response did not include an output image.")
            return await self._download_image(str(output[0]), provider, model)

        get_url = prediction.get("urls", {}).get("get")
        if not isinstance(get_url, str):
            raise Exception("Replicate response did not include a polling URL.")

        start_time = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start_time > 900:
                raise TimeoutError("Replicate prediction timed out.")

            await asyncio.sleep(1)
            polled = await provider_runtime.request_json(
                key=f"image-poll:{provider}:{model}",
                initial_window=1,
                method="GET",
                url=get_url,
                headers=headers,
                provider=provider,
                model=model,
                timeouts=RequestTimeouts(
                    connect_timeout_sec=10.0,
                    sock_read_timeout_sec=30.0,
                ),
                max_retries=2,
            )
            if not isinstance(polled, dict):
                raise Exception("Replicate polling returned an invalid payload.")

            status = polled.get("status")
            if status == "succeeded":
                output = polled.get("output", [])
                if not isinstance(output, list) or not output:
                    raise Exception("Replicate polling response had no output.")
                return await self._download_image(str(output[0]), provider, model)
            if status == "failed":
                raise Exception(f"Replicate prediction failed: {polled.get('error')}")
            if status == "canceled":
                raise Exception("Replicate prediction canceled.")

    async def _download_image(self, url: str, provider: str, model: str) -> bytes:
        return await provider_runtime.request_bytes(
            key=f"image-download:{provider}:{model}",
            initial_window=1,
            method="GET",
            url=url,
            provider=provider,
            model=model,
            timeouts=RequestTimeouts(
                connect_timeout_sec=10.0,
                sock_read_timeout_sec=60.0,
            ),
            max_retries=MAX_RETRIES,
        )

    async def _get_google_image(
        self,
        prompt: str,
        model: ImageModels | LegacyGoogleImageModel,
        aspect_ratio: ImageAspectRatio | None = None,
        resolution: ImageResolution | None = None,
    ) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        if model == "gemini-nano-banana-pro":
            model = "gemini-3-pro-image-preview"

        data_bytes = await self._execute_request(
            url=f"{GOOGLE_IMAGE_BASE_URL}/{model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json_payload={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "imageConfig": self._google_image_config(
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    )
                },
            },
            provider="google_image",
            model=model,
        )

        data = json.loads(data_bytes)

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

    def _extract_google_image_bytes(self, data: dict[str, Any]) -> str | None:
        if not data:
            return None

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
        aspect_ratio: ImageAspectRatio | None = None,
        resolution: ImageResolution | None = None,
        output_format: str | None = None,
        provider_config: CustomProvider | None = None,
        generation_quality: ImageGenerationQuality | None = None,
    ) -> bytes:
        headers = {"Content-Type": "application/json"}
        provider_name = "openai"
        base_url = config.openai_endpoint or "https://api.openai.com"
        if provider_config is None:
            api_key = config.openai_api_key
            if not api_key:
                raise Exception(
                    "OpenAI API key not found. Please set it in the settings."
                )
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            provider_name = provider_config["name"]
            base_url = provider_config["base_url"]
            if provider_config["api_key"]:
                headers["Authorization"] = f"Bearer {provider_config['api_key']}"

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": self._openai_image_size(model, aspect_ratio, resolution),
        }
        if model.startswith("gpt-image-"):
            if generation_quality:
                if generation_quality not in image_generation_qualities(model):
                    raise ValueError(
                        f"{model} does not support {generation_quality} image quality."
                    )
                payload["quality"] = generation_quality
        else:
            payload["response_format"] = "b64_json"
        if output_format in {"png", "jpeg", "webp"}:
            payload["output_format"] = output_format
        elif output_format == "jpg":
            payload["output_format"] = "jpeg"

        data_bytes = await self._execute_request(
            url=build_v1_endpoint(base_url, "/images/generations"),
            headers=headers,
            json_payload=payload,
            provider=provider_name,
            model=model,
        )

        data = json.loads(data_bytes)
        try:
            b64_json = data["data"][0]["b64_json"]
            return base64.b64decode(b64_json)
        except (KeyError, IndexError) as exc:
            raise Exception("Invalid response format from OpenAI Image API") from exc

    def _google_image_config(
        self,
        *,
        aspect_ratio: ImageAspectRatio | None,
        resolution: ImageResolution | None,
    ) -> dict[str, Any]:
        config_payload: dict[str, Any] = {"aspectRatio": aspect_ratio or "1:1"}
        if resolution == "2048x2048":
            config_payload["imageSize"] = "2K"
        elif resolution == "4096x4096":
            config_payload["imageSize"] = "4K"
        return config_payload

    def _openai_image_size(
        self,
        model: str,
        aspect_ratio: ImageAspectRatio | None,
        resolution: ImageResolution | None,
    ) -> str:
        if model == "gpt-image-2" or model.startswith("gpt-image-2.5-"):
            return self._openai_image_2_size(aspect_ratio, resolution)

        if "gpt-image" in model:
            if aspect_ratio == "16:9" or aspect_ratio == "4:3":
                return "1536x1024"
            if aspect_ratio == "9:16" or aspect_ratio == "3:4":
                return "1024x1536"
            return "1024x1024"

        if aspect_ratio == "16:9":
            return "1792x1024"
        if aspect_ratio == "9:16":
            return "1024x1792"
        return "1024x1024"

    def _openai_image_2_size(
        self,
        aspect_ratio: ImageAspectRatio | None,
        resolution: ImageResolution | None,
    ) -> str:
        if resolution == "4096x4096":
            if aspect_ratio == "16:9":
                return "3840x2160"
            if aspect_ratio == "4:3":
                return "2736x2048"
            if aspect_ratio == "9:16":
                return "2160x3840"
            if aspect_ratio == "3:4":
                return "2048x2736"
            return "2048x2048"

        if resolution == "2048x2048":
            if aspect_ratio == "16:9":
                return "2048x1152"
            if aspect_ratio == "4:3":
                return "2048x1536"
            if aspect_ratio == "9:16":
                return "1152x2048"
            if aspect_ratio == "3:4":
                return "1536x2048"
            return "2048x2048"

        if aspect_ratio == "16:9" or aspect_ratio == "4:3":
            return "1536x1024"
        if aspect_ratio == "9:16" or aspect_ratio == "3:4":
            return "1024x1536"
        return "1024x1024"


image_provider = ImageProvider()
