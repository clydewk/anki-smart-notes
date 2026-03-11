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
from typing import Any, Optional

import pytest

from src import models
from src.chat_provider import (
    MAX_TOOL_TURNS,
    ChatProvider,
    TextGenerationRequest,
    TextToolDefinition,
    filter_openai_text_models,
    prompt_cache_key_for_request,
)
from src.image_provider import ImageProvider
from src.provider_runtime import ProviderHTTPError, StreamingNotSupportedError
from src.tts_provider import TTSProvider


def test_gpt_5_4_present_in_openai_models() -> None:
    assert "gpt-5.4" in models.openai_chat_models
    assert "gpt-5.4" in models.provider_model_map["openai"]


def test_default_extras_include_chat_use_tools() -> None:
    assert "chat_use_tools" in models.DEFAULT_EXTRAS
    assert models.DEFAULT_EXTRAS["chat_use_tools"] is None


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


def openai_test_config() -> SimpleNamespace:
    return SimpleNamespace(
        openai_api_key="sk-test",
        openai_endpoint=None,
        custom_providers=[],
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
        ]
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
async def test_openai_payload_with_reasoning(
    monkeypatch: pytest.MonkeyPatch,
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


@pytest.mark.asyncio
async def test_openai_responses_tool_loop_executes_tool_calls(
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

    assert result.text == "tool-backed answer"
    assert result.usage == {"input_tokens": 2, "output_tokens": 3}
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
        SimpleNamespace(
            openai_api_key="sk-test",
            openai_endpoint=None,
            custom_providers=[],
        ),
    )

    async def fake_request_json(**kwargs: Any) -> dict[str, Any]:
        captured["timeouts"] = kwargs["timeouts"]
        return {
            "id": "resp_1",
            "output_text": "tool-backed answer",
            "usage": {"output_tokens": 3},
        }

    monkeypatch.setattr(
        "src.chat_provider.provider_runtime.request_json",
        fake_request_json,
    )

    async def tool_executor(_: str, __: dict[str, Any]) -> str:
        raise AssertionError("Tool executor should not be called without tool calls.")

    result = await cp.generate_text(
        TextGenerationRequest(
            prompt="hi",
            model="gpt-5.4",
            provider="openai",
            temperature=0.5,
            reasoning_effort="xhigh",
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
    assert captured["timeouts"].sock_read_timeout_sec == 90.0


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
