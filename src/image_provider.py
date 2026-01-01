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
)
from .models import ImageAspectRatio, ImageModels, ImageProviders, ImageResolution

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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=IMAGE_PROVIDER_TIMEOUT_SEC
            ) as response:
                response.raise_for_status()
                prediction = await response.json()

            if prediction.get("status") == "succeeded":
                output_url = prediction["output"][0]
                return await self._download_image(session, output_url)

            get_url = prediction["urls"]["get"]
            start_time = asyncio.get_event_loop().time()
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
        # Default or "1024x1024" doesn't need explicit imageSize or can be omitted if it's default
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"imageConfig": image_config},
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=IMAGE_PROVIDER_TIMEOUT_SEC)
        ) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

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
             # DALL-E 3 supports 1024x1024 HD? No, standard/hd quality.
             # The docs say "Size: Image dimensions (e.g., 1024x1024, 1024x1536)"
             # Let's stick to size parameter.
             pass

        # Construct payload
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
        
        # Add size if it's a known supported size or if using a model that supports specific sizes
        # For GPT-Image 1.5, we can use "size": "auto" according to docs?
        # Docs say: "Available sizes: 1024x1024 (square) - 1536x1024 (landscape) - 1024x1536 (portrait) auto (default)"
        # So if we map aspects to those sizes:
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
             # DALL-E 3 logic (1024x1024, 1024x1792, 1792x1024)
             if aspect_ratio == "1:1":
                 payload["size"] = "1024x1024"
             elif aspect_ratio == "16:9":
                 payload["size"] = "1792x1024"
             elif aspect_ratio == "9:16":
                 payload["size"] = "1024x1792"
             else:
                 payload["size"] = "1024x1024"

        # Quality
        # GPT-Image models support "quality": "high" etc.
        # DALL-E 3 supports "quality": "standard" or "hd".
        # We can default to standard/auto unless we want to expose quality in UI later.
        # For now, stick to default (which is usually standard/auto).

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=IMAGE_PROVIDER_TIMEOUT_SEC)
        ) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    try:
                        error_resp = await response.json()
                        error_msg = error_resp.get("error", {}).get("message", str(response.status))
                    except:
                        error_msg = str(response.status)
                    raise Exception(f"OpenAI Image Error: {error_msg}")
                
                data = await response.json()

        try:
            b64_json = data["data"][0]["b64_json"]
            return base64.b64decode(b64_json)
        except (KeyError, IndexError):
            raise Exception("Invalid response format from OpenAI Image API")


image_provider = ImageProvider()
