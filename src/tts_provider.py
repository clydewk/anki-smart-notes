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
        if provider == "fish":
            return await self._get_fish_tts(text, model, voice)
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

    async def _get_fish_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.fish_api_key
        if not api_key:
            raise Exception("Fish Audio API key not found.")

        reference_id = voice.strip()
        if not reference_id:
            raise Exception("Fish Audio voice model ID not found.")

        return await self._execute_request(
            url="https://api.fish.audio/v1/tts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "model": model,
            },
            json_payload={
                "text": text,
                "reference_id": reference_id,
                "format": "mp3",
            },
            provider="fish",
            model=model,
        )

    async def _get_google_tts(self, text: str, model: str, voice: str) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google Cloud API key not found.")

        language_code = "-".join(voice.split("-")[:2])
        data_bytes = await self._execute_request(
            url="https://texttospeech.googleapis.com/v1/text:synthesize",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
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
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
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
            return self._extract_google_gemini_audio_bytes(data)
        except (KeyError, IndexError) as exc:
            logger.error(f"Unexpected response format from Google Gemini TTS: {data}")
            raise Exception(
                "Failed to extract audio from Google Gemini response"
            ) from exc

    def _extract_google_gemini_audio_bytes(self, data: Any) -> bytes:
        if not isinstance(data, dict):
            raise KeyError("response")

        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            raise KeyError("candidates")

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline_data = part.get("inlineData")
                if not isinstance(inline_data, dict):
                    continue

                audio_b64 = inline_data.get("data")
                if not isinstance(audio_b64, str):
                    continue

                mime_type = inline_data.get("mimeType")
                normalized_mime_type = mime_type if isinstance(mime_type, str) else None
                if normalized_mime_type and not normalized_mime_type.lower().startswith(
                    "audio/"
                ):
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                return self._google_gemini_audio_to_wav(
                    audio_bytes, normalized_mime_type
                )

        raise KeyError("inlineData")

    def _google_gemini_audio_to_wav(
        self, audio_bytes: bytes, mime_type: str | None
    ) -> bytes:
        if audio_bytes.startswith(b"RIFF"):
            return audio_bytes

        sample_rate = 24000
        channels = 1

        if mime_type:
            for parameter in mime_type.split(";")[1:]:
                key, _, value = parameter.partition("=")
                normalized_key = key.strip().lower()
                normalized_value = value.strip()

                if normalized_key == "rate" and normalized_value.isdigit():
                    sample_rate = int(normalized_value)
                elif normalized_key == "channels" and normalized_value.isdigit():
                    channels = int(normalized_value)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        return wav_buffer.getvalue()


tts_provider = TTSProvider()
