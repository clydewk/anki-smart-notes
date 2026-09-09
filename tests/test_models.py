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
import wave
from types import SimpleNamespace
from typing import Any, Optional, get_args

import pytest

from src import models
from src.chat_provider import (
    MAX_TOOL_TURNS,
    ChatProvider,
    TextGenerationRequest,
    TextToolDefinition,
    choose_reasoning_effort,
    filter_openai_text_models,
    prompt_cache_key_for_request,
)
from src.constants import DEFAULT_CHAT_MODEL
from src.image_provider import ImageProvider
from src.provider_runtime import ProviderHTTPError, StreamingNotSupportedError
from src.tts_provider import TTSProvider


def test_curated_openai_chat_models_are_ordered_and_labeled() -> None:
    assert DEFAULT_CHAT_MODEL == "gpt-5.6-sol"
    assert models.openai_chat_models == [
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
    ]
    assert models.provider_model_map["openai"] == models.openai_chat_models
    assert models.openai_model_label("gpt-6-astra") == "GPT-6 Astra"
    assert filter_openai_text_models(["gpt-6-astra", "gpt-image-2.5-flare"]) == [
        "gpt-6-astra"
    ]
    assert models.openai_model_label("gpt-5.6-sol") == "GPT-5.6 Sol"
    assert models.openai_model_label("gpt-5.6-terra") == "GPT-5.6 Terra"
    assert models.openai_model_label("gpt-5.6-luna") == "GPT-5.6 Luna"
    assert models.openai_model_label("gpt-5.5-pro") == "GPT-5.5 Pro"


def test_default_extras_include_chat_use_tools() -> None:
    assert "chat_use_tools" in models.DEFAULT_EXTRAS
    assert models.DEFAULT_EXTRAS["chat_use_tools"] is None


def test_google_tts_models_include_gemini_3_1_flash_preview() -> None:
    assert "gemini-3.1-flash-tts-preview" in get_args(models.GoogleModels)


def test_fish_tts_models_match_api() -> None:
    assert get_args(models.FishTTSModels) == (
        "s2.1-pro-free",
        "s2.1-pro",
        "s2-pro",
        "s1",
    )


def test_normalize_field_extras_migrates_only_active_legacy_tts_override() -> None:
    legacy = {
        "use_custom_model": True,
        "tts_provider": "fish",
        "tts_model": "s2.1-pro-free",
        "tts_voice": "voice-id",
    }
    normalized = models.normalize_field_extras(legacy)
    assert normalized["tts_voice_pool"] == [
        {
            "provider": "fish",
            "model": "s2.1-pro-free",
            "voice": "voice-id",
            "language": None,
            "enabled": True,
        }
    ]

    legacy["use_custom_model"] = False
    assert models.normalize_field_extras(legacy)["tts_voice_pool"] is None
    assert (
        models.normalize_field_extras({"tts_voice_pool": None})["tts_voice_pool"]
        is None
    )


def test_normalize_tts_voice_pool_skips_invalid_entries() -> None:
    assert models.normalize_tts_voice_pool(
        [
            {
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy",
                "language": " en-US ",
                "enabled": False,
            },
            {"provider": "fish", "model": "s2.1-pro-free", "voice": ""},
            "bad",
        ]
    ) == [
        {
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "language": "en-US",
            "enabled": False,
        }
    ]


def test_openai_image_models_include_gpt_image_2() -> None:
    assert "gpt-image-2" in get_args(models.OpenAIImageModels)


def test_openai_reasoning_efforts_match_model_family() -> None:
    efforts_56 = models.openai_reasoning_efforts_for_model("gpt-5.6-sol")
    efforts_56_alias = models.openai_reasoning_efforts_for_model("gpt-5.6")
    efforts_55 = models.openai_reasoning_efforts_for_model("gpt-5.5")

    assert efforts_56 == list(models.OPENAI_REASONING_EFFORTS_WITH_MAX)
    assert efforts_56_alias == efforts_56
    assert "max" in efforts_56
    assert efforts_55 == list(models.OPENAI_REASONING_EFFORTS_WITH_NONE_AND_XHIGH)
    assert "max" not in efforts_55


def test_filter_openai_text_models_excludes_non_text_models() -> None:
    models_to_filter = [
        "gpt-5.4",
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.6-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5-2026-04-23",
        "gpt-5-codex",
        "gpt-5-search-api",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o",
        "gpt-image-1",
        "gpt-4o-mini-tts",
        "text-embedding-3-large",
        "computer-use-preview",
    ]

    filtered = filter_openai_text_models(models_to_filter)
    assert filtered == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.6",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ]


def test_openai_chat_models_for_display_merges_available_models() -> None:
    available_models = ["gpt-5.5-pro", "gpt-5.4-mini", "gpt-3.5-turbo"]

    assert models.openai_chat_models_for_display(available_models) == [
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
    ]
    assert models.openai_chat_models_for_display(
        available_models,
        include_available=True,
    ) == [
        "gpt-6-astra",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4-mini",
    ]
    assert (
        models.openai_chat_models_for_display(
            available_models,
            current_model="custom-account-model",
        )[-1]
        == "custom-account-model"
    )


def test_prompt_cache_key_is_stable() -> None:
    key1 = prompt_cache_key_for_request("gpt-5.4", "note-type:Basic:Front:template")
    key2 = prompt_cache_key_for_request("gpt-5.4", "note-type:Basic:Front:template")
    assert key1 == key2


def openai_test_config(*, fast_mode: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        openai_api_key="sk-test",
        openai_endpoint=None,
        openai_fast_mode_enabled=fast_mode,
        custom_providers=[],
        openai_daily_token_budget_enabled=False,
        openai_daily_token_budget=1_000_000,
    )


def custom_provider_test_config(
    *,
    name: str = "Compat",
    base_url: str = "http://localhost:1234",
    api_key: str = "test-key",
    model: str = "compat-model",
    chat_api_mode: str = "responses",
    streaming_mode: str = "disabled",
) -> SimpleNamespace:
    return SimpleNamespace(
        custom_providers=[
            {
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "capabilities": ["chat"],
                "models": [model],
                "chat_models": [model],
                "tts_models": [],
                "image_models": [],
                "chat_api_mode": chat_api_mode,
                "streaming_mode": streaming_mode,
            }
        ],
        openai_fast_mode_enabled=False,
        openai_daily_token_budget_enabled=False,
        openai_daily_token_budget=1_000_000,
    )


def response_created_event(response_id: str) -> dict[str, Any]:
    return {"type": "response.created", "response": {"id": response_id}}


def response_completed_event(
    response_id: str,
    *,
    text: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
    output: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {"id": response_id}
    if text is not None:
        response["output_text"] = text
    if usage is not None:
        response["usage"] = usage
    if output is not None:
        response["output"] = output
    return {"type": "response.completed", "response": response}


def function_call_item(
    *,
    call_id: str,
    name: str,
    arguments: str,
    item_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id or call_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def function_call_added_event(
    response_id: str,
    *,
    output_index: int,
    call_id: str,
    name: str,
    item_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": output_index,
        "item": function_call_item(
            call_id=call_id,
            name=name,
            arguments="",
            item_id=item_id,
        ),
    }


def function_call_delta_event(
    response_id: str,
    *,
    output_index: int,
    item_id: str,
    delta: str,
) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "delta": delta,
    }


def function_call_done_event(
    response_id: str,
    *,
    output_index: int,
    call_id: str,
    name: Optional[str],
    arguments: str,
    item_id: Optional[str] = None,
    item: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "item_id": item_id or call_id,
        "output_index": output_index,
        "call_id": call_id,
        "arguments": arguments,
    }
    if name is not None:
        event["name"] = name
    if item is not None:
        event["item"] = item
    return event


def make_stream_sse_json(
    turns: list[list[dict[str, Any]]],
    *,
    payloads: Optional[list[dict[str, Any]]] = None,
):
    turn_index = 0

    async def fake_stream_sse_json(**kwargs: Any):
        nonlocal turn_index
        if payloads is not None:
            payloads.append(kwargs["json_payload"])
        events = turns[turn_index]
        turn_index += 1
        for event in events:
            yield event

    return fake_stream_sse_json


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-5.6-sol", "max", "max"),
        ("gpt-6-astra", None, "low"),
        ("gpt-6-astra", "none", "low"),
        ("gpt-6-astra", "minimal", "low"),
        ("gpt-6-astra", "max", "max"),
    ],
)
async def test_openai_payload_with_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    effort: Optional[models.OpenAIReasoningEffort],
    expected: str,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
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
            model=model,
            provider="openai",
            temperature=0.5,
            reasoning_effort=effort,
            prompt_cache_key="smart-notes:test",
        )
    )

    assert result.text == "hello"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["model"] == model
    assert captured["payload"]["reasoning"] == {"effort": expected}
    assert "temperature" not in captured["payload"]
    assert choose_reasoning_effort(model, effort) == expected
    assert captured["payload"]["store"] is False
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["prompt_cache_key"] == "smart-notes:test"
    assert "service_tier" not in captured["payload"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "endpoint", "fast"),
    [
        ("gpt-5.6-sol", None, True),
        ("gpt-6-astra", None, True),
        ("gpt-6-astra", "https://eu.api.openai.com/v1", False),
    ],
)
async def test_openai_fast_mode_sets_service_tier(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    endpoint: Optional[str],
    fast: bool,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}
    test_config = openai_test_config(fast_mode=True)
    test_config.openai_endpoint = endpoint
    monkeypatch.setattr(
        "src.chat_provider.config",
        test_config,
    )

    async def fake_stream_sse_json(**kwargs: Any):
        captured["payload"] = kwargs["json_payload"]
        yield response_created_event("resp_fast")
        yield response_completed_event("resp_fast", text="hello")

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model=model,
            provider="openai",
            temperature=0.5,
            reasoning_effort="none",
        )
    )

    assert result.text == "hello"
    assert captured["payload"].get("service_tier") == ("fast" if fast else None)


@pytest.mark.asyncio
async def test_reasoning_trace_logging_when_debug_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    logged_messages: list[str] = []
    test_config = openai_test_config()
    test_config.debug = True
    monkeypatch.setattr("src.chat_provider.config", test_config)

    def fake_debug(message: str, *args: Any) -> None:
        logged_messages.append(message % args)

    monkeypatch.setattr("src.chat_provider.logger.debug", fake_debug)
    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    {
                        "type": "response.output_item.added",
                        "response_id": "resp_1",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "Checking local examples.",
                                }
                            ],
                            "encrypted_content": "opaque-secret",
                        },
                    },
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "response_id": "resp_1",
                        "delta": "Using bold-tag convention.",
                    },
                    response_completed_event("resp_1", text="final answer"),
                ]
            ],
        ),
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
        )
    )

    assert result.text == "final answer"
    joined_logs = "\n".join(logged_messages)
    assert "Checking local examples." in joined_logs
    assert "Using bold-tag convention." in joined_logs
    assert "opaque-secret" not in joined_logs


@pytest.mark.asyncio
async def test_reasoning_trace_logging_skipped_when_debug_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    logged_messages: list[str] = []
    test_config = openai_test_config()
    test_config.debug = False
    monkeypatch.setattr("src.chat_provider.config", test_config)

    def fake_debug(message: str, *args: Any) -> None:
        logged_messages.append(message % args)

    monkeypatch.setattr("src.chat_provider.logger.debug", fake_debug)
    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "response_id": "resp_1",
                        "delta": "Hidden trace.",
                    },
                    response_completed_event("resp_1", text="final answer"),
                ]
            ],
        ),
    )

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
        )
    )

    assert result.text == "final answer"
    assert not any("Hidden trace." in message for message in logged_messages)


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
                {"id": "gpt-5.6-luna"},
                {"id": "gpt-5.6-sol"},
                {"id": "gpt-5.6-terra"},
                {"id": "gpt-5.4"},
                {"id": "gpt-5.5"},
                {"id": "gpt-5.5-pro"},
                {"id": "gpt-5.4-mini"},
                {"id": "gpt-5.4-mini-2026-03-17"},
                {"id": "gpt-5-codex"},
                {"id": "gpt-3.5-turbo"},
                {"id": "gpt-image-1"},
                {"id": "gpt-4o-mini-tts"},
            ]
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    models = await cp.fetch_openai_chat_models()

    assert models == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
    ]
    assert cp.get_cached_openai_chat_models() == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
    ]
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
        openai_test_config(),
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
            model="gpt-5.6-terra",
            provider="openai",
            temperature=0.5,
            reasoning_effort="none",
            prompt_cache_key="smart-notes:test",
        )
    )

    assert result.text == "fallback text"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["payload"]["reasoning"] == {"effort": "none"}
    assert captured["payload"]["temperature"] == 0.5
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
                {"id": "gpt-5.6-sol"},
                {"id": "gpt-5.4"},
                {"id": "gpt-5.5"},
                {"id": "gpt-5.5-pro"},
                {"id": "gpt-image-1"},
                {"id": "gpt-4o-mini-tts"},
            ]
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    models = await cp.fetch_openai_chat_models()

    assert models == ["gpt-5.6-sol", "gpt-5.5", "gpt-5.5-pro", "gpt-5.4"]
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
async def test_fish_tts_uses_model_header_and_reference_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TTSProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(
            custom_providers=[],
            fish_api_key="fish-test-key",
        ),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        return b"fish-audio"

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes",
        fake_request_bytes,
    )

    data = await provider.async_get_tts_response(
        input="hello",
        model="s2.1-pro-free",
        provider="fish",
        voice="  voice-model-id  ",
        strip_html=False,
    )

    assert data == b"fish-audio"
    assert captured["url"] == "https://api.fish.audio/v1/tts"
    assert captured["headers"]["Authorization"] == "Bearer fish-test-key"
    assert captured["headers"]["model"] == "s2.1-pro-free"
    assert captured["payload"] == {
        "text": "hello",
        "reference_id": "voice-model-id",
        "format": "mp3",
    }


@pytest.mark.asyncio
async def test_google_gemini_tts_uses_header_auth_and_wraps_pcm_as_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TTSProvider()
    captured: dict[str, Any] = {}
    pcm_bytes = b"\x01\x00\x02\x00"

    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(
            openai_api_key=None,
            openai_endpoint=None,
            custom_providers=[],
            elevenlabs_api_key=None,
            google_api_key="google-test-key",
        ),
    )

    async def fake_request_bytes(**kwargs: Any) -> bytes:
        captured["url"] = kwargs["url"]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json_payload"]
        return json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "metadata"},
                                {
                                    "inlineData": {
                                        "mimeType": "audio/L16;rate=24000",
                                        "data": base64.b64encode(pcm_bytes).decode(
                                            "ascii"
                                        ),
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

    monkeypatch.setattr(
        "src.tts_provider.provider_runtime.request_bytes",
        fake_request_bytes,
    )

    data = await provider.async_get_tts_response(
        input="hello",
        model="gemini-3.1-flash-tts-preview",
        provider="google",
        voice="Kore",
        strip_html=False,
    )

    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "google-test-key"
    assert "key=" not in captured["url"]
    assert captured["payload"]["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert (
        captured["payload"]["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Kore"
    )

    with wave.open(io.BytesIO(data), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == pcm_bytes


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]
)
async def test_openai_image_2_uses_supported_high_resolution_size(
    monkeypatch: pytest.MonkeyPatch,
    model: models.ImageModels,
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
        model=model,
        provider="openai",
        note_id=1,
        aspect_ratio="16:9",
        resolution="4096x4096",
        generation_quality="max" if model.startswith("gpt-image-2.5-") else "high",
    )

    assert data == b"image-bytes"
    assert captured["payload"]["model"] == model
    assert captured["payload"]["quality"] == (
        "max" if model.startswith("gpt-image-2.5-") else "high"
    )
    assert "response_format" not in captured["payload"]
    assert captured["payload"]["size"] == "3840x2160"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-6-astra"])
async def test_openai_responses_tool_loop_executes_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []
    request_calls = 0

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        nonlocal request_calls
        request_calls += 1
        raise AssertionError(
            "Official OpenAI tool calls should use streamed responses."
        )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )
    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    function_call_added_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name="mcp_test_echo",
                        item_id="fc_1",
                    ),
                    function_call_delta_event(
                        "resp_1",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"query":"hi"}',
                    ),
                    function_call_done_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name="mcp_test_echo",
                        item_id="fc_1",
                        arguments='{"query":"hi"}',
                    ),
                    response_completed_event(
                        "resp_1",
                        usage={"input_tokens": 2},
                        output=[
                            function_call_item(
                                call_id="call_1",
                                name="mcp_test_echo",
                                item_id="fc_1",
                                arguments='{"query":"hi"}',
                            )
                        ],
                    ),
                ],
                [
                    response_created_event("resp_2"),
                    response_completed_event(
                        "resp_2",
                        text="tool-backed answer",
                        usage={"output_tokens": 3},
                    ),
                ],
            ],
            payloads=payloads,
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        assert name == "mcp_test_echo"
        return f"tool:{arguments['query']}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model=model,
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "tool-backed answer"
    assert result.usage == {"input_tokens": 2, "output_tokens": 3}
    if model == "gpt-6-astra":
        for payload in payloads:
            assert payload["reasoning"] == {"effort": "low"}
            assert "temperature" not in payload
    assert request_calls == 0
    assert payloads[0]["tools"][0]["name"] == "mcp_test_echo"
    assert payloads[0]["store"] is True
    assert payloads[0]["stream"] is True
    assert "prompt_cache_key" not in payloads[0]
    assert payloads[1]["store"] is True
    assert payloads[1]["stream"] is True
    assert payloads[1]["previous_response_id"] == "resp_1"
    assert payloads[1]["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "tool:hi"}
    ]


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_uses_reasoning_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        captured["timeouts"] = kwargs["timeouts"]
        yield response_created_event("resp_1")
        yield response_completed_event(
            "resp_1",
            text="tool-backed answer",
            usage={"output_tokens": 3},
        )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )

    async def tool_executor(_: str, __: dict[str, Any]) -> str:
        raise AssertionError("Tool executor should not be called without tool calls.")

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.6-sol",
            provider="openai",
            temperature=0.5,
            reasoning_effort="max",
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "tool-backed answer"
    assert captured["timeouts"].first_event_timeout_sec == 90.0
    assert captured["timeouts"].stream_idle_timeout_sec == 90.0


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_handles_message_output_items_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    {
                        "type": "response.output_item.added",
                        "response_id": "resp_1",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_1",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_1",
                        "delta": "No tool needed",
                    },
                    response_completed_event(
                        "resp_1",
                        usage={"output_tokens": 3},
                    ),
                ]
            ],
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        del name, arguments
        raise AssertionError("Tool executor should not be called.")

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="anki_search_notes",
                    description="Search notes",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "No tool needed"
    assert result.usage == {"output_tokens": 3}


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_streams_multiple_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []
    seen_tool_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_1",
                        "delta": "Looking at similar notes...",
                    },
                    function_call_added_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name="anki_search_notes",
                        item_id="fc_1",
                    ),
                    function_call_delta_event(
                        "resp_1",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"query":"漢字"}',
                    ),
                    function_call_done_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name="anki_search_notes",
                        item_id="fc_1",
                        arguments='{"query":"漢字"}',
                    ),
                    function_call_added_event(
                        "resp_1",
                        output_index=1,
                        call_id="call_2",
                        name="anki_get_deck_overview",
                        item_id="fc_2",
                    ),
                    function_call_delta_event(
                        "resp_1",
                        output_index=1,
                        item_id="fc_2",
                        delta='{"deck_name":"',
                    ),
                    function_call_delta_event(
                        "resp_1",
                        output_index=1,
                        item_id="fc_2",
                        delta='Japanese"}',
                    ),
                    function_call_done_event(
                        "resp_1",
                        output_index=1,
                        call_id="call_2",
                        name="anki_get_deck_overview",
                        item_id="fc_2",
                        arguments='{"deck_name":"Japanese"}',
                    ),
                    response_completed_event(
                        "resp_1",
                        usage={"input_tokens": 4, "output_tokens": 2},
                    ),
                ],
                [
                    response_created_event("resp_2"),
                    response_completed_event(
                        "resp_2",
                        text="Done",
                        usage={"output_tokens": 1},
                    ),
                ],
            ],
            payloads=payloads,
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        seen_tool_calls.append((name, arguments))
        return f"tool:{name}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="anki_search_notes",
                    description="Search notes",
                    input_schema={"type": "object"},
                ),
                TextToolDefinition(
                    name="anki_get_deck_overview",
                    description="Get deck overview",
                    input_schema={"type": "object"},
                ),
            ],
        ),
        tool_executor=tool_executor,
    )

    assert payloads[0]["stream"] is True
    assert payloads[1]["stream"] is True
    assert result.response_id == "resp_2"
    assert result.text == "Done"
    assert result.usage == {"input_tokens": 4, "output_tokens": 3}
    assert seen_tool_calls == [
        ("anki_search_notes", {"query": "漢字"}),
        ("anki_get_deck_overview", {"deck_name": "Japanese"}),
    ]


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_uses_completed_response_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    seen_tool_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    function_call_added_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name="mcp_test_echo",
                        item_id="fc_1",
                    ),
                    function_call_delta_event(
                        "resp_1",
                        output_index=0,
                        item_id="fc_1",
                        delta='{"query":"hi"}',
                    ),
                    response_completed_event(
                        "resp_1",
                        usage={"input_tokens": 2},
                        output=[
                            function_call_item(
                                call_id="call_1",
                                name="mcp_test_echo",
                                item_id="fc_1",
                                arguments='{"query":"hi"}',
                            )
                        ],
                    ),
                ],
                [
                    response_created_event("resp_2"),
                    response_completed_event(
                        "resp_2",
                        text="fallback complete",
                        usage={"output_tokens": 1},
                    ),
                ],
            ],
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        seen_tool_calls.append((name, arguments))
        return "tool:hi"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert seen_tool_calls == [("mcp_test_echo", {"query": "hi"})]
    assert result.text == "fallback complete"
    assert result.usage == {"input_tokens": 2, "output_tokens": 1}


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_uses_done_event_item_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    seen_tool_calls: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event("resp_1"),
                    function_call_done_event(
                        "resp_1",
                        output_index=0,
                        call_id="call_1",
                        name=None,
                        item_id="fc_1",
                        arguments='{"query":"hi"}',
                        item=function_call_item(
                            call_id="call_1",
                            name="mcp_test_echo",
                            item_id="fc_1",
                            arguments='{"query":"hi"}',
                        ),
                    ),
                    response_completed_event(
                        "resp_1",
                        usage={"input_tokens": 2},
                    ),
                ],
                [
                    response_created_event("resp_2"),
                    response_completed_event(
                        "resp_2",
                        text="tool-backed answer",
                        usage={"output_tokens": 1},
                    ),
                ],
            ],
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        seen_tool_calls.append((name, arguments))
        return "tool:hi"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert seen_tool_calls == [("mcp_test_echo", {"query": "hi"})]
    assert result.text == "tool-backed answer"
    assert result.usage == {"input_tokens": 2, "output_tokens": 1}


@pytest.mark.asyncio
async def test_custom_responses_tool_loop_executes_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(
        "src.chat_provider.config",
        custom_provider_test_config(),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        payloads.append(kwargs["json_payload"])
        if len(payloads) == 1:
            return {
                "id": "resp_custom_1",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "mcp_test_echo",
                        "arguments": '{"query":"hi"}',
                    }
                ],
                "usage": {"input_tokens": 2},
            }

        return {
            "id": "resp_custom_2",
            "output_text": "tool-backed answer",
            "usage": {"output_tokens": 3},
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        assert name == "mcp_test_echo"
        return f"tool:{arguments['query']}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="compat-model",
            provider="Compat",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "tool-backed answer"
    assert result.usage == {"input_tokens": 2, "output_tokens": 3}
    assert payloads[0]["store"] is True
    assert payloads[1]["store"] is True
    assert payloads[1]["previous_response_id"] == "resp_custom_1"
    assert payloads[1]["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "tool:hi"}
    ]


@pytest.mark.asyncio
async def test_custom_responses_tool_loop_streaming_falls_back_to_non_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []
    stream_calls = 0

    monkeypatch.setattr(
        "src.chat_provider.config",
        custom_provider_test_config(streaming_mode="enabled"),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        del kwargs
        nonlocal stream_calls
        stream_calls += 1
        raise StreamingNotSupportedError("streaming not supported")
        yield {}

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        payloads.append(kwargs["json_payload"])
        if len(payloads) == 1:
            return {
                "id": "resp_custom_1",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "mcp_test_echo",
                        "arguments": '{"query":"hi"}',
                    }
                ],
                "usage": {"input_tokens": 2},
            }

        return {
            "id": "resp_custom_2",
            "output_text": "tool-backed answer",
            "usage": {"output_tokens": 3},
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )
    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        assert name == "mcp_test_echo"
        return f"tool:{arguments['query']}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="compat-model",
            provider="Compat",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "tool-backed answer"
    assert stream_calls == 1
    assert len(payloads) == 2
    assert "stream" not in payloads[0]
    assert payloads[1]["previous_response_id"] == "resp_custom_1"


@pytest.mark.asyncio
async def test_custom_provider_responses_not_implemented_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()

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
                    "chat_api_mode": "responses",
                    "streaming_mode": "disabled",
                }
            ]
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise ProviderHTTPError(
            500,
            "Internal Server Error",
            '{"error":{"message":"not implemented","type":"rix_api_error","code":"convert_request_failed"}}',
            "http://localhost:1234/v1/responses",
        )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    with pytest.raises(Exception) as exc_info:
        await cp.generate_text(
            TextGenerationRequest(
                prompt="hello",
                model="compat-model",
                provider="Compat",
                temperature=0.25,
                reasoning_effort=None,
            )
        )

    assert "does not support the Responses API" in str(exc_info.value)
    assert "Chat Completions" in str(exc_info.value)


@pytest.mark.asyncio
async def test_custom_provider_responses_with_tools_not_implemented_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()

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
                    "chat_api_mode": "responses",
                    "streaming_mode": "disabled",
                }
            ]
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise ProviderHTTPError(
            500,
            "Internal Server Error",
            '{"error":{"message":"not implemented","type":"rix_api_error","code":"convert_request_failed"}}',
            "http://localhost:1234/v1/responses",
        )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        del name, arguments
        return "unused"

    with pytest.raises(Exception) as exc_info:
        await cp.generate_text(
            TextGenerationRequest(
                prompt="hello",
                model="compat-model",
                provider="Compat",
                temperature=0.25,
                reasoning_effort=None,
                tools=[
                    TextToolDefinition(
                        name="mcp_test_echo",
                        description="Echo input",
                        input_schema={"type": "object"},
                    )
                ],
            ),
            tool_executor=tool_executor,
        )

    assert "does not support the Responses API" in str(exc_info.value)
    assert "tool calling cannot be used" in str(exc_info.value)


@pytest.mark.asyncio
async def test_custom_provider_streaming_falls_back_to_non_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    stream_calls = 0
    request_calls = 0

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
                    "chat_api_mode": "responses",
                    "streaming_mode": "enabled",
                }
            ]
        ),
    )

    async def fake_stream_sse_json(**kwargs: Any):
        del kwargs
        nonlocal stream_calls
        stream_calls += 1
        raise StreamingNotSupportedError("streaming not supported")
        yield {}

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        nonlocal request_calls
        request_calls += 1
        return {"id": "resp_1", "output_text": "fallback text"}

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        fake_stream_sse_json,
    )
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

    assert result.text == "fallback text"
    assert stream_calls == 1
    assert request_calls == 1


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_forces_final_answer_after_max_tool_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []
    request_calls = 0

    monkeypatch.setattr(
        "src.chat_provider.config",
        openai_test_config(),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        nonlocal request_calls
        request_calls += 1
        raise AssertionError(
            "Official OpenAI tool calls should use streamed responses."
        )

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )
    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.stream_sse_json",
        make_stream_sse_json(
            [
                [
                    response_created_event(f"resp_{turn}"),
                    function_call_done_event(
                        f"resp_{turn}",
                        output_index=0,
                        call_id=f"call_{turn}",
                        name="mcp_test_echo",
                        item_id=f"fc_{turn}",
                        arguments=json.dumps({"query": f"hi {turn}"}),
                    ),
                    response_completed_event(
                        f"resp_{turn}",
                        usage={"input_tokens": turn},
                    ),
                ]
                for turn in range(1, MAX_TOOL_TURNS + 1)
            ]
            + [
                [
                    response_created_event("resp_final"),
                    response_completed_event(
                        "resp_final",
                        text="final answer",
                        usage={"output_tokens": 3},
                    ),
                ]
            ],
            payloads=payloads,
        ),
    )

    async def tool_executor(name: str, arguments: dict[str, Any]) -> str:
        assert name == "mcp_test_echo"
        return f"tool:{arguments['query']}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "final answer"
    assert request_calls == 0
    assert len(payloads) == MAX_TOOL_TURNS + 1
    assert all(payload["store"] is True for payload in payloads)
    assert all(payload["stream"] is True for payload in payloads)
    assert payloads[-1]["tool_choice"] == "none"
    assert "tools" not in payloads[-1]
    assert payloads[-1]["previous_response_id"] == f"resp_{MAX_TOOL_TURNS}"
    assert payloads[-1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": f"call_{MAX_TOOL_TURNS}",
            "output": f"tool:hi {MAX_TOOL_TURNS}",
        }
    ]


@pytest.mark.asyncio
async def test_custom_chat_completions_tool_loop_executes_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    payloads: list[dict[str, Any]] = []

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
        payloads.append(kwargs["json_payload"])
        if len(payloads) == 1:
            return {
                "id": "chat_1",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "mcp_test_echo",
                                        "arguments": '{"query":"hi"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2},
            }

        return {
            "id": "chat_2",
            "choices": [{"message": {"content": "done"}}],
            "usage": {"completion_tokens": 1},
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    async def tool_executor(_: str, arguments: dict[str, Any]) -> str:
        return f"tool:{arguments['query']}"

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hello",
            model="compat-model",
            provider="Compat",
            temperature=0.25,
            reasoning_effort=None,
            tools=[
                TextToolDefinition(
                    name="mcp_test_echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        ),
        tool_executor=tool_executor,
    )

    assert result.text == "done"
    assert payloads[0]["tools"][0]["function"]["name"] == "mcp_test_echo"
    assert (
        payloads[1]["messages"][1]["tool_calls"][0]["function"]["name"]
        == "mcp_test_echo"
    )
    assert payloads[1]["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool:hi",
    }
