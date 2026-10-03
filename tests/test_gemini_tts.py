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

import base64
import io
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.models import GEMINI_METADATA_TTS_MODELS, TTSModels
from src.provider_runtime import ResponseFormatError
from src.tts_provider import TTSProvider


def gemini_audio_response(audio: bytes, mime: str = "audio/wav") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime,
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        }
                    ]
                },
            }
        ]
    }


async def synthesize_response(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> bytes:
    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(google_api_key="test-key", custom_providers=[]),
    )

    async def fake_request_bytes(**_kwargs: Any) -> bytes:
        return json.dumps(response).encode()

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes", fake_request_bytes
    )
    return await TTSProvider().async_get_tts_response(
        "Hello!", "gemini-3.8-flash-tts", "google", "Kore", False
    )


@pytest.fixture
def wav_audio() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00\x02\x00")
    return output.getvalue()


@pytest.mark.parametrize("model", GEMINI_METADATA_TTS_MODELS)
@pytest.mark.parametrize("style", [None, "Warm, with a measured pace."])
async def test_gemini_38_keeps_transcript_verbatim_and_separates_style(
    monkeypatch: pytest.MonkeyPatch,
    wav_audio: bytes,
    model: TTSModels,
    style: str | None,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(google_api_key="test-key", custom_providers=[]),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured.update(kwargs)
        return json.dumps(gemini_audio_response(wav_audio)).encode()

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes", fake_request_bytes
    )
    transcript = "Wait... <short pause> did you hear that? <sigh>"
    result = await TTSProvider().async_get_tts_response(
        transcript, model, "google", "Kore", False, instructions=style
    )
    assert result == wav_audio
    assert captured["url"].endswith(f"/{model}:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert "key=" not in captured["url"]
    part: dict[str, Any] = {"text": transcript}
    if style:
        part["speech_metadata"] = {"style": style}
    assert captured["json_payload"] == {
        "contents": [{"role": "user", "parts": [part]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"voice": "Kore"}},
        },
    }


async def test_legacy_gemini_preserves_prompted_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(google_api_key="test-key", custom_providers=[]),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured.update(kwargs)
        return json.dumps(
            gemini_audio_response(b"\x00\x01", "audio/L16;codec=pcm;rate=24000")
        ).encode()

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes", fake_request_bytes
    )
    result = await TTSProvider().async_get_tts_response(
        "Hello!",
        "gemini-3.1-flash-tts-preview",
        "google",
        "Kore",
        False,
        instructions="Say warmly:",
    )
    assert captured["json_payload"]["contents"][0]["parts"] == [
        {"text": "Say warmly: Hello!"}
    ]
    assert captured["json_payload"]["generationConfig"]["speechConfig"] == {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
    }
    with wave.open(io.BytesIO(result), "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.readframes(1) == b"\x00\x01"


@pytest.mark.parametrize("finish_reason", [None, "MAX_TOKENS", "SAFETY", "OTHER"])
async def test_gemini_rejects_unfinished_audio(
    monkeypatch: pytest.MonkeyPatch, wav_audio: bytes, finish_reason: str | None
) -> None:
    response = gemini_audio_response(wav_audio)
    response["candidates"][0]["finishReason"] = finish_reason
    with pytest.raises(ResponseFormatError, match="incomplete"):
        await synthesize_response(monkeypatch, response)


@pytest.mark.parametrize("block", ["prompt", "candidate"])
async def test_gemini_rejects_blocked_audio(
    monkeypatch: pytest.MonkeyPatch, wav_audio: bytes, block: str
) -> None:
    response = gemini_audio_response(wav_audio)
    if block == "prompt":
        response["promptFeedback"] = {"blockReason": "SAFETY"}
    else:
        response["candidates"][0]["safetyRatings"] = [{"blocked": True}]
    with pytest.raises(ResponseFormatError):
        await synthesize_response(monkeypatch, response)


@pytest.mark.parametrize("encoded", ["!garbage!", "é", "AAAA===garbage", ""])
async def test_gemini_rejects_invalid_base64(
    monkeypatch: pytest.MonkeyPatch, encoded: str
) -> None:
    response = gemini_audio_response(b"ok")
    response["candidates"][0]["content"]["parts"][0]["inlineData"]["data"] = encoded
    with pytest.raises(ResponseFormatError):
        await synthesize_response(monkeypatch, response)


@pytest.mark.parametrize("mime", [None, "text/plain", "audio/mpeg", "audio/ogg"])
async def test_gemini_rejects_unrecognized_audio(
    monkeypatch: pytest.MonkeyPatch, wav_audio: bytes, mime: str | None
) -> None:
    response = gemini_audio_response(wav_audio)
    response["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"] = mime
    with pytest.raises(ResponseFormatError):
        await synthesize_response(monkeypatch, response)


@pytest.mark.parametrize(
    "mime",
    [
        "audio/L16;rate=0",
        "audio/L16;rate=word",
        "audio/L16;channels=0",
        "audio/L16;channels=999",
        "audio/L16;codec=mp3",
        "audio/L16;invalid",
    ],
)
async def test_gemini_rejects_invalid_pcm_format(
    monkeypatch: pytest.MonkeyPatch, mime: str
) -> None:
    with pytest.raises(ResponseFormatError, match="PCM"):
        await synthesize_response(monkeypatch, gemini_audio_response(b"\x01\x00", mime))


@pytest.mark.parametrize("audio", [b"", b"\x01", b"\x01\x00\x02"])
async def test_gemini_rejects_empty_or_partial_pcm(
    monkeypatch: pytest.MonkeyPatch, audio: bytes
) -> None:
    with pytest.raises(ResponseFormatError):
        await synthesize_response(
            monkeypatch, gemini_audio_response(audio, "audio/L16")
        )


@pytest.mark.parametrize("corruption", ["truncated", "header", "format", "frame"])
async def test_gemini_rejects_malformed_wav(
    monkeypatch: pytest.MonkeyPatch, wav_audio: bytes, corruption: str
) -> None:
    if corruption == "truncated":
        audio = wav_audio[:-2]
    elif corruption == "header":
        audio = b"RIFF" + b"\x00" * 40
    elif corruption == "format":
        audio = wav_audio[:8] + b"OOPS" + wav_audio[12:]
    else:
        audio = wav_audio[:-1]
        audio = audio[:4] + (len(audio) - 8).to_bytes(4, "little") + audio[8:]
    with pytest.raises(ResponseFormatError, match="WAV"):
        await synthesize_response(monkeypatch, gemini_audio_response(audio))


async def test_gemini_checks_size_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.tts_provider.MAX_GEMINI_AUDIO_BYTES", 4)
    with pytest.raises(ResponseFormatError, match="size limit"):
        await synthesize_response(
            monkeypatch, gemini_audio_response(b"longer than bound")
        )


async def test_gemini_rejects_ambiguous_audio(
    monkeypatch: pytest.MonkeyPatch, wav_audio: bytes
) -> None:
    response = gemini_audio_response(wav_audio)
    response["candidates"][0]["content"]["parts"] *= 2
    with pytest.raises(ResponseFormatError, match="unique"):
        await synthesize_response(monkeypatch, response)


def test_speech_html_preparation_uses_real_anki_and_preserves_vocal_tags() -> None:
    script = """
from anki.lang import set_lang
from src.tts_provider import GEMINI_VOCAL_TAGS, prepare_speech_text
set_lang("en_US")
for tag in GEMINI_VOCAL_TAGS:
    assert prepare_speech_text(f"<b>Wait</b> <{tag}> &amp; listen", True) == f"Wait <{tag}> & listen"
assert prepare_speech_text("<b>Hello</b> &lt;sigh&gt;", True) == "Hello <sigh>"
assert prepare_speech_text("<b>Hello</b> &amp;lt;sigh&amp;gt;", True) == "Hello &lt;sigh&gt;"
assert prepare_speech_text("<script>bad <sigh></script><b>OK</b>", True) == "OK"
assert prepare_speech_text('<sigh onclick="bad">OK</sigh>', True) == "OK"
assert prepare_speech_text("<b>Hello</b> <sigh>", False) == "<b>Hello</b> <sigh>"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "IS_TEST": "True"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
