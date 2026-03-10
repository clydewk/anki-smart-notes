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
import json
from types import SimpleNamespace
from typing import Any

import pytest

from src import models
from src.chat_provider import (
    ChatProvider,
    TextGenerationRequest,
    filter_openai_text_models,
    prompt_cache_key_for_request,
)
from src.image_provider import ImageProvider
from src.tts_provider import TTSProvider


def test_gpt_5_4_present_in_openai_models() -> None:
    assert "gpt-5.4" in models.openai_chat_models
    assert "gpt-5.4" in models.provider_model_map["openai"]


def test_gpt_5_3_chat_latest_present() -> None:
    assert "gpt-5.3-chat-latest" in models.openai_chat_models
    assert "gpt-5.3-chat-latest" in models.provider_model_map["openai"]


def test_gpt_5_4_reasoning_efforts() -> None:
    efforts_54 = models.openai_reasoning_efforts_for_model("gpt-5.4")
    efforts_52 = models.openai_reasoning_efforts_for_model("gpt-5.2")

    assert efforts_54 == efforts_52
    assert "xhigh" in efforts_54

    efforts_chat = models.openai_reasoning_efforts_for_model("gpt-5.3-chat-latest")
    assert efforts_chat == list(models.OPENAI_DEFAULT_REASONING_EFFORTS)


def test_filter_openai_text_models_excludes_non_text_models() -> None:
    models_to_filter = [
        "gpt-5.4",
        "gpt-5-mini",
        "gpt-image-1",
        "gpt-4o-mini-tts",
        "text-embedding-3-large",
    ]

    filtered = filter_openai_text_models(models_to_filter)
    assert filtered == ["gpt-5-mini", "gpt-5.4"]


def test_prompt_cache_key_is_stable() -> None:
    key1 = prompt_cache_key_for_request("gpt-5.4", "note-type:Basic:Front:template")
    key2 = prompt_cache_key_for_request("gpt-5.4", "note-type:Basic:Front:template")
    assert key1 == key2


@pytest.mark.asyncio
async def test_openai_payload_with_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
            custom_providers=[],
        ),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        yield {"type": "response.created", "response": {"id": "resp_123"}}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "usage": {"total_tokens": 9},
                "output_text": "hello",
            },
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort="high",
            prompt_cache_key="smart-notes:test",
        )
    )

    assert result.text == "hello"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-5.4"
    assert captured["payload"]["reasoning"] == {"effort": "high"}
    assert captured["payload"]["store"] is False
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["prompt_cache_key"] == "smart-notes:test"


@pytest.mark.asyncio
async def test_custom_provider_without_api_key_omits_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            custom_providers=[
                {
                    "name": "Local",
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "capabilities": ["chat"],
                    "models": ["gpt-oss"],
                    "chat_models": ["gpt-oss"],
                    "tts_models": [],
                    "image_models": [],
                    "chat_api_mode": "responses",
                    "streaming_mode": "enabled",
                }
            ]
        ),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        yield {"type": "response.created", "response": {"id": "resp_local"}}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_local",
                "output_text": "hello from local",
            },
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-oss",
            provider="Local",
            temperature=0.5,
            reasoning_effort=None,
        )
    )

    assert result.text == "hello from local"
    assert captured["headers"] == {"Content-Type": "application/json"}
    assert captured["payload"]["stream"] is True


@pytest.mark.asyncio
async def test_custom_provider_chat_completions_mode_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            custom_providers=[
                {
                    "name": "Compat",
                    "base_url": "http://localhost:1234",
                    "api_key": "test-key",
                    "capabilities": ["chat"],
                    "models": ["compat-model"],
                    "chat_models": ["compat-model"],
                    "tts_models": [],
                    "image_models": [],
                    "chat_api_mode": "chat_completions",
                    "streaming_mode": "disabled",
                }
            ]
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        captured["url"] = kwargs["url"]
        captured["payload"] = kwargs["json_payload"]
        captured["headers"] = kwargs["headers"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "chat completions result",
                    }
                }
            ]
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hello",
            model="compat-model",
            provider="Compat",
            temperature=0.25,
            reasoning_effort=None,
        )
    )

    assert result.text == "chat completions result"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_fetch_openai_chat_models_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        return {
            "data": [
                {"id": "gpt-5.4"},
                {"id": "gpt-5-mini"},
                {"id": "gpt-image-1"},
                {"id": "gpt-4o-mini-tts"},
            ]
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    models = await cp.fetch_openai_chat_models()

    assert models == ["gpt-5-mini", "gpt-5.4"]
    assert cp.get_cached_openai_chat_models() == ["gpt-5-mini", "gpt-5.4"]
    assert captured["url"] == "https://api.openai.com/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_openai_text_uses_responses_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
            custom_providers=[],
        ),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        yield {"type": "response.created", "response": {"id": "resp_fallback"}}
        yield {"type": "response.output_text.delta", "delta": "fallback text"}
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_fallback",
                "usage": {"total_tokens": 7},
            },
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort="high",
            prompt_cache_key="smart-notes:test",
        )
    )

    assert result.text == "fallback text"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["prompt_cache_key"] == "smart-notes:test"
    assert captured["payload"]["stream"] is True


@pytest.mark.asyncio
async def test_fetch_openai_chat_models_uses_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        captured["headers"] = kwargs["headers"]
        return {
            "data": [
                {"id": "gpt-5.4"},
                {"id": "gpt-image-1"},
                {"id": "gpt-4o-mini-tts"},
            ]
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    models = await cp.fetch_openai_chat_models()

    assert models == ["gpt-5.4"]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_openai_tts_uses_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TTSProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
            custom_providers=[],
            elevenlabs_api_key=None,
            google_api_key=None,
        ),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        return b"audio-bytes"

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes",
        fake_request_bytes,
    )

    data = await provider.async_get_tts_response(
        input="hello",
        model="tts-1",
        provider="openai",
        voice="alloy",
        strip_html=False,
    )

    assert data == b"audio-bytes"
    assert captured["url"].endswith("/v1/audio/speech")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "tts-1"


@pytest.mark.asyncio
async def test_openai_image_uses_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ImageProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.image_provider.config",
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
            custom_providers=[],
            replicate_api_key=None,
            google_api_key=None,
        ),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        return json.dumps(
            {
                "data": [
                    {
                        "b64_json": base64.b64encode(b"image-bytes").decode("ascii"),
                    }
                ]
            }
        ).encode("utf-8")

    monkeypatch.setattr(
        "src.image_provider.provider_runtime.request_bytes",
        fake_request_bytes,
    )

    data = await provider.async_get_image_response(
        prompt="cat",
        model="gpt-image-1",
        provider="openai",
        note_id=1,
    )

    assert data == b"image-bytes"
    assert captured["url"].endswith("/v1/images/generations")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gpt-image-1"
