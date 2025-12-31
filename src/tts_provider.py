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
from typing import Optional

import aiohttp
from anki.utils import strip_html as anki_strip_html

from .config import config
from .constants import TTS_PROVIDER_TIMEOUT_SEC
from .logger import logger
from .models import TTSModels, TTSProviders


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
            return await self._get_google_tts(text, model, voice)
        elif provider == "azure":
            raise NotImplementedError("Azure TTS is not currently supported in BYOK mode.")
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=TTS_PROVIDER_TIMEOUT_SEC
            ) as response:
                response.raise_for_status()
                return await response.read()

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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=TTS_PROVIDER_TIMEOUT_SEC
            ) as response:
                response.raise_for_status()
                return await response.read()

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

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, timeout=TTS_PROVIDER_TIMEOUT_SEC
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return base64.b64decode(data["audioContent"])

tts_provider = TTSProvider()
