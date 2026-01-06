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
import io
import json
import wave
from typing import Optional

import aiohttp
from anki.utils import strip_html as anki_strip_html

from .config import config
from .constants import MAX_RETRIES, RETRY_BASE_SECONDS, TTS_PROVIDER_TIMEOUT_SEC
from .logger import logger
from .models import TTSModels, TTSProviders
from .rate_limiter import (
    estimate_tokens,
    extract_google_token_usage,
    extract_rate_limit_headers,
    get_rate_limiter,
    parse_retry_after,
)


class TTSProvider:
    async def async_get_tts_response(
        self,
        input: str,
        model: TTSModels,
        provider: TTSProviders,
        voice: str,
        strip_html: bool,
        note_id: int = -1,
    ) -> bytes:
        text = input
        if strip_html:
            text = anki_strip_html(input)

        # Check custom providers
        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        if custom_provider:
            return await self._get_openai_tts(text, model, voice, custom_provider)

        if provider == "openai":
            return await self._get_openai_tts(text, model, voice)
        elif provider == "elevenLabs":
            return await self._get_elevenlabs_tts(text, model, voice)
        elif provider == "google":
            # Check if using new Gemini models or old ones
            if "gemini" in model:
                return await self._get_google_gemini_tts(text, model, voice)
            else:
                return await self._get_google_tts(text, model, voice)
        elif provider == "azure":
            raise NotImplementedError(
                "Azure TTS is not currently supported in BYOK mode."
            )
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

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
            # Don't report generic exceptions to rate limiter
            raise

    async def _get_openai_tts(
        self, text: str, model: str, voice: str, provider_config: Optional[dict] = None
    ) -> bytes:
        if provider_config:
            api_key = provider_config["api_key"]
            base_url = provider_config["base_url"]
            # Assume base_url is root or v1. If it ends with v1, we append audio/speech
            if base_url.endswith("/v1"):
                url = f"{base_url}/audio/speech"
            else:
                url = f"{base_url}/v1/audio/speech"
            provider_name = provider_config["name"]
        else:
            api_key = config.openai_api_key
            if not api_key:
                raise Exception("OpenAI API key not found.")
            url = (
                f"{config.openai_endpoint or 'https://api.openai.com'}/v1/audio/speech"
            )
            provider_name = "openai"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
        }

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(text)

        return await self._execute_request(
            url,
            headers,
            payload,
            TTS_PROVIDER_TIMEOUT_SEC,
            provider_name,
            model,
            estimated_tokens=estimated_tokens,
        )

    async def _get_elevenlabs_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.elevenlabs_api_key
        if not api_key:
            raise Exception("ElevenLabs API key not found.")

        # Voice ID is usually passed as 'voice' in the config
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": model,
        }

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(text)

        return await self._execute_request(
            url,
            headers,
            payload,
            TTS_PROVIDER_TIMEOUT_SEC,
            "elevenLabs",
            model,
            estimated_tokens=estimated_tokens,
        )

    async def _get_google_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google Cloud API key not found.")

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"

        # Parse voice name to get language code. Usually format is "en-US-Standard-A"
        # We need languageCode="en-US", name="en-US-Standard-A"
        language_code = "-".join(voice.split("-")[:2])

        payload = {
            "input": {"text": text},
            "voice": {"languageCode": language_code, "name": voice},
            "audioConfig": {"audioEncoding": "MP3"},
        }

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(text)

        # Google Cloud TTS has high limits, but usually uses standard cloud quotas
        # Note: Google TTS returns JSON with content, not raw bytes. _execute_request returns raw bytes.
        # We need to handle this.

        data_bytes = await self._execute_request(
            url,
            None,
            payload,
            TTS_PROVIDER_TIMEOUT_SEC,
            "google",
            model,
            estimated_tokens=estimated_tokens,
        )

        data = json.loads(data_bytes)
        return base64.b64decode(data["audioContent"])

    async def _get_google_gemini_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }

        headers = {"Content-Type": "application/json"}

        # Estimate tokens for rate limiting
        estimated_tokens = estimate_tokens(text)

        # Use specific limiter for Gemini TTS which has restrictive quotas (10 RPM)
        data_bytes = await self._execute_request(
            url,
            headers,
            payload,
            TTS_PROVIDER_TIMEOUT_SEC,
            "google_tts",
            model,
            estimated_tokens=estimated_tokens,
        )

        data = json.loads(data_bytes)

        # Extract audio data
        # Structure: candidates[0].content.parts[0].inlineData.data
        try:
            inline_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            audio_b64 = inline_data["data"]
            pcm_data = base64.b64decode(audio_b64)

            # Convert raw PCM to WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(pcm_data)

            return wav_buffer.getvalue()

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Google Gemini TTS: {data}")
            raise Exception(
                "Failed to extract audio from Google Gemini response"
            ) from e


tts_provider = TTSProvider()
