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
import wave
from typing import Optional

import aiohttp
from anki.utils import strip_html as anki_strip_html

from .config import config
from .constants import MAX_RETRIES, RETRY_BASE_SECONDS, TTS_PROVIDER_TIMEOUT_SEC
from .logger import logger
from .models import TTSModels, TTSProviders
from .rate_limiter import get_rate_limiter


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
            raise NotImplementedError("Azure TTS is not currently supported in BYOK mode.")
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

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

    async def _get_openai_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.openai_api_key
        if not api_key:
            raise Exception("OpenAI API key not found.")

        url = f"{config.openai_endpoint or 'https://api.openai.com'}/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
        }

        return await self._execute_request(
            url, headers, payload, TTS_PROVIDER_TIMEOUT_SEC, "openai"
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

        return await self._execute_request(
            url, headers, payload, TTS_PROVIDER_TIMEOUT_SEC, "elevenLabs"
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

        # Google Cloud TTS has high limits, but usually uses standard cloud quotas
        # Note: Google TTS returns JSON with content, not raw bytes. _execute_request returns raw bytes.
        # We need to handle this.
        
        data_bytes = await self._execute_request(
            url, None, payload, TTS_PROVIDER_TIMEOUT_SEC, "google"
        )
        
        import json
        data = json.loads(data_bytes)
        return base64.b64decode(data["audioContent"])

    async def _get_google_gemini_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts":[{
                    "text": text
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            }
        }
        
        headers = {"Content-Type": "application/json"}

        # Use specific limiter for Gemini TTS which has restrictive quotas (10 RPM)
        data_bytes = await self._execute_request(
            url, headers, payload, TTS_PROVIDER_TIMEOUT_SEC, "google_tts"
        )
        
        import json
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
            
        except (KeyError, IndexError):
            logger.error(f"Unexpected response format from Google Gemini TTS: {data}")
            raise Exception("Failed to extract audio from Google Gemini response")

tts_provider = TTSProvider()
