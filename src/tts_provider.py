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
import binascii
import io
import json
import re
import wave
from html import escape
from typing import TYPE_CHECKING, Any

from anki.utils import strip_html as anki_strip_html

from .chat_provider import build_v1_endpoint
from .config import config
from .constants import MAX_RETRIES
from .models import GEMINI_METADATA_TTS_MODELS
from .provider_runtime import (
    RequestTimeouts,
    ResponseFormatError,
    provider_runtime,
)

if TYPE_CHECKING:
    from .models import CustomProvider, TTSModels, TTSProviders


MAX_GEMINI_AUDIO_BYTES = 64 * 1024 * 1024
GEMINI_VOCAL_TAGS = (
    "argh",
    "breath",
    "heavy breath",
    "exhales",
    "cackle",
    "cheer",
    "chuckle",
    "chuckles",
    "cough",
    "cry",
    "gasp",
    "giggle",
    "groan",
    "growl",
    "grunt",
    "grr",
    "hiss",
    "laugh",
    "laughter",
    "moan",
    "pant",
    "pff",
    "phew",
    "scream",
    "shout",
    "shriek",
    "sigh",
    "sighs",
    "sneeze",
    "snicker",
    "snort",
    "sob",
    "throat-clearing",
    "tsk",
    "whimper",
    "whispers",
    "whispering",
    "yawn",
    "short pause",
    "long pause",
)
GEMINI_VOCAL_TAG_PATTERN = re.compile(
    "<(?:" + "|".join(re.escape(tag) for tag in GEMINI_VOCAL_TAGS) + ")>",
    re.IGNORECASE,
)


def prepare_speech_text(text: str, strip_html: bool) -> str:
    if not strip_html:
        return text
    # Anki decodes entities after stripping HTML, retaining vocal tags as text.
    return anki_strip_html(
        GEMINI_VOCAL_TAG_PATTERN.sub(lambda match: escape(match.group()), text)
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
        instructions: str | None = None,
    ) -> bytes:
        del note_id

        text = prepare_speech_text(input, strip_html)

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
                return await self._get_google_gemini_tts(
                    text, model, voice, instructions
                )
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

    async def _get_google_gemini_tts(
        self, text: str, model: str, voice: str, instructions: str | None = None
    ) -> bytes:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found.")

        part: dict[str, Any] = {"text": text}
        voice_config: dict[str, Any] = {"prebuiltVoiceConfig": {"voiceName": voice}}
        if model in GEMINI_METADATA_TTS_MODELS:
            voice_config = {"voice": voice}
            if instructions:
                part["speech_metadata"] = {"style": instructions}
        elif instructions:
            part["text"] = f"{instructions} {text}"

        data_bytes = await self._execute_request(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json_payload={
                "contents": [{"role": "user", "parts": [part]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": voice_config},
                },
            },
            provider="google_tts",
            model=model,
        )

        try:
            data = json.loads(data_bytes)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ResponseFormatError(
                "Google Gemini TTS returned an invalid JSON response."
            ) from exc
        return self._extract_google_gemini_audio_bytes(data)

    def _extract_google_gemini_audio_bytes(self, data: object) -> bytes:
        if not isinstance(data, dict):
            raise ResponseFormatError("Google Gemini TTS returned an invalid response.")
        feedback = data.get("promptFeedback")
        if data.get("error") or (
            isinstance(feedback, dict) and feedback.get("blockReason")
        ):
            raise ResponseFormatError("Google Gemini TTS did not complete the request.")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ResponseFormatError(
                "Google Gemini TTS returned no unique audio result."
            )
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get("finishReason") != "STOP":
            raise ResponseFormatError("Google Gemini TTS returned incomplete audio.")
        ratings = candidate.get("safetyRatings", [])
        if isinstance(ratings, list) and any(
            isinstance(rating, dict) and rating.get("blocked") for rating in ratings
        ):
            raise ResponseFormatError("Google Gemini TTS blocked the audio response.")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise ResponseFormatError("Google Gemini TTS returned no audio content.")
        inline_parts = [
            part["inlineData"]
            for part in parts
            if isinstance(part, dict) and "inlineData" in part
        ]
        if len(inline_parts) != 1 or not isinstance(inline_parts[0], dict):
            raise ResponseFormatError(
                "Google Gemini TTS returned no unique audio data."
            )
        audio_b64 = inline_parts[0].get("data")
        mime_type = inline_parts[0].get("mimeType")
        if not isinstance(audio_b64, str) or not isinstance(mime_type, str):
            raise ResponseFormatError("Google Gemini TTS returned invalid audio data.")
        if len(audio_b64) > 4 * ((MAX_GEMINI_AUDIO_BYTES + 2) // 3):
            raise ResponseFormatError("Google Gemini TTS audio exceeds the size limit.")
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ResponseFormatError(
                "Google Gemini TTS returned invalid base64 audio."
            ) from exc
        return self._google_gemini_audio_to_wav(audio_bytes, mime_type)

    def _google_gemini_audio_to_wav(self, audio_bytes: bytes, mime_type: str) -> bytes:
        if not audio_bytes or len(audio_bytes) > MAX_GEMINI_AUDIO_BYTES:
            raise ResponseFormatError(
                "Google Gemini TTS returned empty or oversized audio."
            )
        media_type, *parameters = mime_type.lower().split(";")
        if media_type.strip() in {"audio/wav", "audio/x-wav", "audio/wave"}:
            try:
                if (
                    audio_bytes[:4] != b"RIFF"
                    or audio_bytes[8:12] != b"WAVE"
                    or int.from_bytes(audio_bytes[4:8], "little")
                    != len(audio_bytes) - 8
                ):
                    raise ValueError("Invalid RIFF size or format")
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    frames = wav_file.getnframes()
                    if (
                        wav_file.getnchannels() not in {1, 2}
                        or wav_file.getsampwidth() != 2
                        or not 8000 <= wav_file.getframerate() <= 96000
                        or frames == 0
                        or len(wav_file.readframes(frames + 1))
                        != frames * wav_file.getnchannels() * 2
                    ):
                        raise ValueError("Invalid PCM frames")
            except (EOFError, ValueError, wave.Error) as exc:
                raise ResponseFormatError(
                    "Google Gemini TTS returned invalid WAV audio."
                ) from exc
            return audio_bytes
        if media_type.strip() != "audio/l16":
            raise ResponseFormatError(
                "Google Gemini TTS returned an unsupported audio format."
            )
        sample_rate = 24000
        channels = 1
        try:
            for parameter in parameters:
                key, separator, value = parameter.strip().partition("=")
                if not separator:
                    raise ValueError("Invalid audio parameter")
                value = value.strip().strip('"')
                if key.strip() == "rate":
                    sample_rate = int(value)
                elif key.strip() == "channels":
                    channels = int(value)
                elif key.strip() == "codec" and value != "pcm":
                    raise ValueError("Unsupported audio codec")
            if (
                not 8000 <= sample_rate <= 96000
                or channels not in {1, 2}
                or len(audio_bytes) % (channels * 2)
            ):
                raise ValueError("Invalid PCM format or frame size")
        except ValueError as exc:
            raise ResponseFormatError(
                "Google Gemini TTS returned invalid PCM audio."
            ) from exc
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

        return wav_buffer.getvalue()


tts_provider = TTSProvider()
