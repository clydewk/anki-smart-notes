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

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.built_in_tools import BuiltInToolContext
from src.chat_provider import ChatProvider, TextGenerationRequest, TextToolDefinition
from src.chatgpt_auth import chatgpt_auth
from src.chatgpt_transport import chatgpt_transport
from src.mcp_runtime import McpRuntime
from src.provider_runtime import ResponseFormatError, request_accounting
from src.tool_registry import ToolRegistry
from tests.test_mcp import make_mcp_config, make_stdio_server_script
from tests.test_tools import FakeCollection, FakeNote, use_collection

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator
    from pathlib import Path


@pytest.fixture
def native_provider(monkeypatch: pytest.MonkeyPatch) -> ChatProvider:
    @asynccontextmanager
    async def session() -> AsyncGenerator[None]:
        yield

    monkeypatch.setattr(chatgpt_auth, "session", session)
    monkeypatch.setattr(
        chatgpt_auth,
        "headers",
        AsyncMock(return_value={"Authorization": "Bearer synthetic"}),
    )
    monkeypatch.setattr(chatgpt_transport, "get_cached_models", lambda: [])
    monkeypatch.setattr(
        "src.chat_provider.chat_usage_tracker.finalize_request", MagicMock()
    )
    return ChatProvider()


def request(*, tools: bool = False) -> TextGenerationRequest:
    return TextGenerationRequest(
        prompt="Define a word.",
        model="opaque-model",
        provider="chatgpt",
        temperature=0.2,
        reasoning_effort="ultra",
        tools=[TextToolDefinition("lookup", "Read a definition", {"type": "object"})]
        if tools
        else None,
    )


def answer(text: str = "Definition", *, phase: str = "final_answer") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": phase,
        "content": [{"type": "output_text", "text": text}],
    }


def completion(
    output: list[dict[str, Any]],
    *,
    response_id: str = "r1",
    status: str = "completed",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": status,
            "output": output,
            "usage": usage,
        },
    }


def tool(
    call_id: str = "c1", *, name: str = "lookup", arguments: str = '{"term":"word"}'
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "namespace": "smart_notes",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def install_peer(
    monkeypatch: pytest.MonkeyPatch, rounds: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def stream(
        payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        index = len(sent)
        sent.append({"payload": deepcopy(payload), "headers": dict(headers)})
        accounting = request_accounting.get()
        if accounting:
            accounting.start_request()
        for event in rounds[index]:
            if accounting:
                accounting.observe(event)
            yield event

    monkeypatch.setattr(chatgpt_transport, "stream_response", stream)
    return sent


async def test_native_payload_and_final_answer_only(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = install_peer(
        monkeypatch,
        [
            [
                {"type": "response.output_text.delta", "delta": "partial commentary"},
                completion(
                    [answer("private commentary", phase="commentary"), answer()],
                    usage={"input_tokens": 5, "output_tokens": 2},
                ),
            ]
        ],
    )
    result = await native_provider.generate_text(request())
    assert result.text == "Definition"
    assert result.usage == {"input_tokens": 5, "output_tokens": 2}
    payload = sent[0]["payload"]
    assert payload["model"] == "opaque-model"
    assert payload["reasoning"] == {"effort": "ultra"}
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["tools"] == []
    assert "temperature" not in payload and "max_output_tokens" not in payload
    assert not set(payload) & {
        "background",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "moderation",
        "multi_agent",
        "prompt",
        "prompt_cache_retention",
        "safety_identifier",
        "temperature",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
        "previous_response_id",
        "service_tier",
    }
    assert isinstance(payload["input"], list)


async def test_tool_history_retains_opaque_reasoning_and_deduplicates_calls(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    opaque = {
        "type": "reasoning",
        "id": "rs1",
        "encrypted_content": "synthetic-opaque",
        "summary": [],
    }
    sent = install_peer(
        monkeypatch,
        [
            [completion([opaque, tool()])],
            [completion([tool()], response_id="r2")],
            [completion([answer()], response_id="r3")],
        ],
    )
    executor = AsyncMock(return_value="found")
    result = await native_provider.generate_text(request(tools=True), executor)
    assert result.text == "Definition"
    executor.assert_awaited_once_with("lookup", {"term": "word"})
    assert opaque in sent[1]["payload"]["input"]
    assert {"type": "function_call_output", "call_id": "c1", "output": "found"} in sent[
        1
    ]["payload"]["input"]
    namespace = sent[0]["payload"]["tools"][0]
    assert namespace["type"] == "namespace"
    assert namespace["name"] == "smart_notes"
    assert namespace["tools"][0]["name"] == "lookup"
    assert "session-id" not in sent[0]["headers"]


@pytest.mark.parametrize(
    "bad_tool",
    [
        tool(name="shell"),
        {**tool(), "namespace": "other"},
        {**tool(), "namespace": None},
        tool(arguments="[1]"),
        tool(arguments="bad"),
        {"type": "computer_call"},
    ],
)
async def test_rejects_undeclared_or_malformed_tools(
    native_provider: ChatProvider,
    monkeypatch: pytest.MonkeyPatch,
    bad_tool: dict[str, Any],
) -> None:
    install_peer(monkeypatch, [[completion([bad_tool])]])
    executor = AsyncMock()
    with pytest.raises(ResponseFormatError):
        await native_provider.generate_text(request(tools=True), executor)
    executor.assert_not_awaited()


async def test_tool_limit_rejects_excess_before_execution(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(monkeypatch, [[completion([tool(f"c{i}") for i in range(13)])]])
    executor = AsyncMock()
    with pytest.raises(ResponseFormatError, match="12 tool calls"):
        await native_provider.generate_text(request(tools=True), executor)
    executor.assert_not_awaited()


async def test_reused_call_id_cannot_change_arguments(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(
        monkeypatch,
        [
            [completion([tool()])],
            [completion([tool(arguments="{}")], response_id="r2")],
        ],
    )
    executor = AsyncMock(return_value="found")
    with pytest.raises(ResponseFormatError, match="different arguments"):
        await native_provider.generate_text(request(tools=True), executor)
    assert executor.await_count == 1


@pytest.mark.parametrize(
    "ending",
    [
        [],
        [{"type": "response.incomplete", "response": {"status": "incomplete"}}],
        [completion([answer()], status="in_progress")],
        [
            completion(
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "refusal": "no"}],
                    }
                ]
            )
        ],
        [completion([answer("commentary", phase="commentary")])],
    ],
)
async def test_incomplete_or_nonfinal_text_is_unsavable(
    native_provider: ChatProvider,
    monkeypatch: pytest.MonkeyPatch,
    ending: list[dict[str, Any]],
) -> None:
    sent = install_peer(
        monkeypatch,
        [[{"type": "response.output_text.delta", "delta": "partial"}, *ending]],
    )
    with pytest.raises(ResponseFormatError):
        await native_provider.generate_text(request())
    assert len(sent) == 1


async def test_cumulative_usage_is_replaced_and_rounds_are_summed(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(
        monkeypatch,
        [
            [
                {
                    "type": "response.usage",
                    "response": {"usage": {"input_tokens": 8, "output_tokens": 1}},
                },
                completion([tool()], usage={"input_tokens": 8, "output_tokens": 2}),
            ],
            [
                completion(
                    [answer()],
                    response_id="r2",
                    usage={"input_tokens": 10, "output_tokens": 3},
                )
            ],
        ],
    )
    result = await native_provider.generate_text(
        request(tools=True), AsyncMock(return_value="found")
    )
    assert result.usage == {"input_tokens": 18, "output_tokens": 5}


async def test_failed_response_records_reported_usage_once(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = MagicMock()
    monkeypatch.setattr(
        "src.chat_provider.chat_usage_tracker.finalize_request", finalizer
    )
    install_peer(
        monkeypatch,
        [
            [
                {
                    "type": "response.failed",
                    "response": {"usage": {"input_tokens": 8, "output_tokens": 1}},
                }
            ]
        ],
    )
    with pytest.raises(ResponseFormatError):
        await native_provider.generate_text(request())
    finalizer.assert_called_once()
    assert finalizer.call_args.kwargs["raw_usage"] == {
        "input_tokens": 8,
        "output_tokens": 1,
    }
    assert finalizer.call_args.kwargs["record_openai_daily_usage"] is False


async def test_unknown_usage_stays_unknown(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(monkeypatch, [[completion([answer()])]])
    assert (await native_provider.generate_text(request())).usage is None


async def test_concurrent_fields_have_separate_conversations(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = install_peer(
        monkeypatch, [[completion([answer()], response_id=f"r{i}")] for i in range(2)]
    )
    await asyncio.gather(*(native_provider.generate_text(request()) for _ in range(2)))
    assert sent[0]["payload"]["input"] is not sent[1]["payload"]["input"]
    assert all("session-id" not in item["headers"] for item in sent)
    assert all(len(item["payload"]["input"]) == 1 for item in sent)


async def test_cancellation_closes_stream_without_replay(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, closed = asyncio.Event(), asyncio.Event()
    calls = 0

    async def stream(
        payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[dict[str, Any]]:
        nonlocal calls
        calls += 1
        try:
            yield {"type": "response.output_text.delta", "delta": "partial"}
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(chatgpt_transport, "stream_response", stream)
    task = asyncio.create_task(native_provider.generate_text(request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and calls == 1
    install_peer(monkeypatch, [[completion([answer()])]])
    assert (await native_provider.generate_text(request())).text == "Definition"


async def test_native_generation_reuses_builtin_and_independent_mcp_peer(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = McpRuntime()
    monkeypatch.setattr("src.mcp_manager.mcp_runtime", runtime)
    monkeypatch.setattr("src.mcp_runtime.get_version", lambda: "test")
    peer = tmp_path / "peer.py"
    make_stdio_server_script(peer)
    collection = FakeCollection(
        notes=[FakeNote(2, "Basic", {"Front": "hello"})],
        note_queries=[("hello", [2])],
    )
    use_collection(monkeypatch, collection)
    registry = ToolRegistry(
        context=BuiltInToolContext(1, 10, "Deck", "Basic", ("Front", "Back"), "Back"),
        built_in_tools={"anki_search_notes": True, "anki_get_deck_overview": False},
        mcp_servers=[make_mcp_config(sys.executable, [str(peer)])],
    )
    try:
        definitions, warnings = await registry.build_tool_registry()
        assert not warnings
        sent = install_peer(
            monkeypatch,
            [
                [
                    completion(
                        [
                            tool(
                                "builtin",
                                name="anki_search_notes",
                                arguments='{"query":"hello"}',
                            ),
                            tool(
                                "mcp",
                                name="mcp_test_echo",
                                arguments='{"text":"independent peer"}',
                            ),
                        ]
                    )
                ],
                [completion([answer()], response_id="r2")],
            ],
        )
        result = await native_provider.generate_text(
            replace(request(), tools=definitions), registry.execute_tool_call
        )
        assert result.text == "Definition"
        outputs = {
            item["call_id"]: item["output"]
            for item in sent[1]["payload"]["input"]
            if item.get("type") == "function_call_output"
        }
        assert json.loads(outputs["builtin"])["results"][0]["note_id"] == 2
        assert outputs["mcp"] == "independent peer"
    finally:
        await runtime.close_current_session()


async def test_stateless_history_omits_unencrypted_reasoning(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {"type": "reasoning", "id": "summary-only", "summary": []}
    sent = install_peer(
        monkeypatch,
        [[completion([summary, tool()])], [completion([answer()], response_id="r2")]],
    )
    await native_provider.generate_text(
        request(tools=True), AsyncMock(return_value="found")
    )
    assert summary not in sent[1]["payload"]["input"]
    assert sent[0]["payload"]["tools"][0]["tools"][0]["strict"] is False


async def test_conflicting_duplicate_in_same_round_never_executes(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(monkeypatch, [[completion([tool(), tool(arguments="{}")])]])
    executor = AsyncMock()
    with pytest.raises(ResponseFormatError, match="different arguments"):
        await native_provider.generate_text(request(tools=True), executor)
    executor.assert_not_awaited()


async def test_text_bound_applies_across_tool_rounds(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    delta = {"type": "response.output_text.delta", "delta": "x" * (600 * 1024)}
    install_peer(
        monkeypatch,
        [
            [delta, completion([tool()])],
            [delta, completion([answer()], response_id="r2")],
        ],
    )
    with pytest.raises(ResponseFormatError, match="1 MiB"):
        await native_provider.generate_text(
            request(tools=True), AsyncMock(return_value="found")
        )


async def test_refusal_event_cannot_be_overridden_by_completed_text(
    native_provider: ChatProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_peer(
        monkeypatch,
        [[{"type": "response.refusal.delta", "delta": "no"}, completion([answer()])]],
    )
    with pytest.raises(ResponseFormatError):
        await native_provider.generate_text(request())
