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

import base64
import io
import json
import wave
from typing import TYPE_CHECKING, Any

from anki.utils import strip_html as anki_strip_html

from .chat_provider import build_v1_endpoint
from .config import config
from .constants import MAX_RETRIES
from .logger import logger
from .provider_runtime import (
    RequestTimeouts,
    provider_runtime,
)

if TYPE_CHECKING:
    from .models import CustomProvider, TTSModels, TTSProviders


class TTSProvider:
    async def async_get_tts_response(
        self,
        input: str,
        model: TTSModels,
        provider: TTSProviders,
        voice: str,
        strip_html: bool,
        note_id: int = -1,
        instructions: str | None = None,
    ) -> bytes:
        del note_id

        text = input
        if strip_html:
            text = anki_strip_html(input)

        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        if custom_provider:
            return await self._get_openai_tts(
                text,
                model,
                voice,
                instructions,
                provider_config=custom_provider,
            )

        if provider == "openai":
            return await self._get_openai_tts(text, model, voice, instructions)
        if provider == "elevenLabs":
            return await self._get_elevenlabs_tts(text, model, voice)
        if provider == "google":
            if "gemini" in model:
                return await self._get_google_gemini_tts(text, model, voice)
            return await self._get_google_tts(text, model, voice)
        if provider == "azure":
            raise NotImplementedError(
                "Azure TTS is not currently supported in BYOK mode."
            )
        raise ValueError(f"Unknown TTS provider: {provider}")

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
            key=f"tts:{provider}:{model}",
            initial_window=2,
            method="POST",
            url=url,
            headers=headers,
            json_payload=json_payload,
            data=data,
            provider=provider,
            model=model,
            timeouts=RequestTimeouts(
                connect_timeout_sec=10.0,
                sock_read_timeout_sec=60.0,
            ),
            max_retries=MAX_RETRIES,
        )

    async def _get_openai_tts(
        self,
        text: str,
        model: str,
        voice: str,
        instructions: str | None = None,
        provider_config: CustomProvider | None = None,
    ) -> bytes:
        headers = {"Content-Type": "application/json"}
        provider_name = "openai"
        base_url = config.openai_endpoint or "https://api.openai.com"
        if provider_config is None:
            api_key = config.openai_api_key
            if not api_key:
                raise Exception("OpenAI API key not found.")
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            provider_name = provider_config["name"]
            base_url = provider_config["base_url"]
            if provider_config["api_key"]:
                headers["Authorization"] = f"Bearer {provider_config['api_key']}"

        payload: dict[str, str] = {
            "model": model,
            "input": text,
            "voice": voice,
        }
        if instructions and model == "gpt-4o-mini-tts":
            payload["instructions"] = instructions

        return await self._execute_request(
            url=build_v1_endpoint(base_url, "/audio/speech"),
            headers=headers,
            json_payload=payload,
            provider=provider_name,
            model=model,
        )

    async def _get_elevenlabs_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.elevenlabs_api_key
        if not api_key:
            raise Exception("ElevenLabs API key not found.")

        return await self._execute_request(
            url=f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json_payload={
                "text": text,
                "model_id": model,
            },
            provider="elevenLabs",
            model=model,
        )

    async def _get_google_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google Cloud API key not found.")

        language_code = "-".join(voice.split("-")[:2])
        data_bytes = await self._execute_request(
            url=f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}",
            headers=None,
            json_payload={
                "input": {"text": text},
                "voice": {"languageCode": language_code, "name": voice},
                "audioConfig": {"audioEncoding": "MP3"},
            },
            provider="google",
            model=model,
        )

        data = json.loads(data_bytes)
        return base64.b64decode(data["audioContent"])

    async def _get_google_gemini_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        data_bytes = await self._execute_request(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json_payload={
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                    },
                },
            },
            provider="google_tts",
            model=model,
        )

        data = json.loads(data_bytes)

        try:
            inline_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            audio_b64 = inline_data["data"]
            pcm_data = base64.b64decode(audio_b64)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(pcm_data)

            return wav_buffer.getvalue()
        except (KeyError, IndexError) as exc:
            logger.error(f"Unexpected response format from Google Gemini TTS: {data}")
            raise Exception(
                "Failed to extract audio from Google Gemini response"
            ) from exc


tts_provider = TTSProvider()
