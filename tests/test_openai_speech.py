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
import base64
import io
import json
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.server import serve

from src.models import TTSModels, openai_tts_voices
from src.provider_runtime import (
    RequestAccounting,
    ResponseFormatError,
    request_accounting,
)
from src.tts_provider import TTSProvider, provider_runtime
from src.tts_routing import tts_audio_extension

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from websockets.asyncio.server import ServerConnection
    from websockets.http11 import Request, Response


def audio_events(pcm: bytes = b"\x01\x00\x02\x00") -> list[dict[str, Any]]:
    audio_identity = {
        "response_id": "response-1",
        "item_id": "audio-1",
        "output_index": 0,
        "content_index": 0,
    }
    return [
        {"type": "response.created", "response": {"id": "response-1"}},
        {
            "type": "response.output_audio.delta",
            **audio_identity,
            "delta": base64.b64encode(pcm).decode(),
        },
        {"type": "response.output_audio.done", **audio_identity},
        {
            "type": "response.done",
            "response": {
                "id": "response-1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "output": [
                    {
                        "id": "audio-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_audio", "transcript": "Hello!"}],
                    }
                ],
            },
        },
    ]


@dataclass
class SpeechPeer:
    events: list[dict[str, Any] | str]
    stall: bool = False
    requests: list[dict[str, Any]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    connections: int = 0
    endpoint: str = ""

    async def handle(self, connection: ServerConnection) -> None:
        self.connections += 1
        assert connection.request is not None
        assert connection.request.path == "/v1/realtime?model=gpt-realtime-2.1-mini"
        assert connection.request.headers["Authorization"] == "Bearer synthetic"
        try:
            await connection.send(json.dumps({"type": "session.created"}))
            self.requests.append(json.loads(await connection.recv()))
            await connection.send(json.dumps({"type": "session.updated"}))
            self.requests.append(json.loads(await connection.recv()))
            self.started.set()
            for event in self.events:
                await connection.send(
                    event if isinstance(event, str) else json.dumps(event)
                )
            if self.stall:
                await connection.wait_closed()
        finally:
            self.closed.set()


@asynccontextmanager
async def realtime_peer(
    monkeypatch: pytest.MonkeyPatch, peer: SpeechPeer
) -> AsyncGenerator[SpeechPeer]:
    async with serve(peer.handle, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        peer.endpoint = f"http://127.0.0.1:{port}"
        monkeypatch.setattr(
            "src.tts_provider.config",
            SimpleNamespace(
                custom_providers=[],
                openai_api_key="synthetic",
                openai_endpoint=peer.endpoint,
            ),
        )
        try:
            yield peer
        finally:
            await provider_runtime.close_current_session()


async def realtime_speech(
    text: str = "Hello!", instructions: str | None = None
) -> bytes:
    return await TTSProvider().async_get_tts_response(
        text,
        "gpt-realtime-2.1-mini",
        "openai",
        "cedar",
        False,
        instructions=instructions,
    )


async def test_realtime_uses_isolated_verbatim_input_and_only_commits_completed_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = SpeechPeer(list(audio_events()))
    usage = RequestAccounting()
    token = request_accounting.set(usage)
    try:
        async with realtime_peer(monkeypatch, peer):
            result = await realtime_speech(
                "Hello! What is 2+2?", "Warm and deliberate."
            )
    finally:
        request_accounting.reset(token)
    session = peer.requests[0]["session"]
    assert session["tools"] == [] and session["tool_choice"] == "none"
    assert session["audio"] == {
        "input": {"turn_detection": None},
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "cedar"},
    }
    response = peer.requests[1]["response"]
    assert response["conversation"] == "none"
    assert response["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello! What is 2+2?"}],
        }
    ]
    assert "Warm and deliberate." in response["instructions"]
    assert response["tools"] == [] and response["tool_choice"] == "none"
    assert peer.connections == 1
    assert usage.usage == {"input_tokens": 10, "output_tokens": 20}
    with wave.open(io.BytesIO(result)) as audio:
        assert audio.getframerate() == 24000
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.readframes(2) == b"\x01\x00\x02\x00"


@pytest.mark.parametrize(
    "failure",
    [
        "eof",
        "incomplete",
        "failed",
        "cancelled",
        "refusal",
        "tool",
        "empty",
        "odd",
        "invalid_base64",
        "missing_audio_done",
        "wrong_response",
        "malformed",
        "error",
    ],
)
async def test_realtime_discards_bad_audio_without_replay(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    events: list[dict[str, Any] | str] = list(audio_events())
    if failure == "eof":
        events.pop()
    elif failure in {"incomplete", "failed", "cancelled"}:
        assert isinstance(events[-1], dict)
        events[-1]["response"]["status"] = failure
    elif failure in {"refusal", "tool"}:
        assert isinstance(events[-1], dict)
        item = events[-1]["response"]["output"][0]
        if failure == "refusal":
            item["content"] = [{"type": "refusal", "refusal": "No"}]
        else:
            item["type"] = "function_call"
    elif failure in {"empty", "odd"}:
        events = list(audio_events(b"" if failure == "empty" else b"\x01"))
    elif failure == "invalid_base64":
        assert isinstance(events[1], dict)
        events[1]["delta"] = "!not base64!"
    elif failure == "missing_audio_done":
        events.pop(2)
    elif failure == "wrong_response":
        assert isinstance(events[1], dict)
        events[1]["response_id"] = "different"
    elif failure == "malformed":
        events = ["{invalid json"]
    else:
        events = [{"type": "error", "error": {"message": "secret-key synthetic"}}]
    peer = SpeechPeer(events)
    async with realtime_peer(monkeypatch, peer):
        with pytest.raises(ResponseFormatError) as error:
            await realtime_speech()
        assert "synthetic" not in str(error.value)
        assert peer.connections == 1


async def test_realtime_cancellation_closes_connection_and_next_request_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = SpeechPeer(list(audio_events()[:2]), stall=True)
    async with realtime_peer(monkeypatch, peer):
        task = asyncio.create_task(realtime_speech())
        await asyncio.wait_for(peer.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        await asyncio.wait_for(peer.closed.wait(), 2)
        peer.events = list(audio_events())
        peer.stall = False
        assert (await realtime_speech()).startswith(b"RIFF")
        assert peer.connections == 2


async def test_realtime_rejects_redirects_without_forwarding_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = SpeechPeer(list(audio_events()))
    async with realtime_peer(monkeypatch, peer):
        target = (
            peer.endpoint.replace("http://", "ws://")
            + "/v1/realtime?model=gpt-realtime-2.1-mini"
        )

        def redirect(connection: ServerConnection, request: Request) -> Response:
            response = connection.respond(HTTPStatus.TEMPORARY_REDIRECT, "Moved")
            response.headers["Location"] = target
            return response

        async with serve(
            peer.handle, "127.0.0.1", 0, process_request=redirect
        ) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            monkeypatch.setattr(
                "src.tts_provider.config.openai_endpoint", f"http://127.0.0.1:{port}"
            )
            with pytest.raises(ResponseFormatError, match="connection failed"):
                await realtime_speech()
            assert peer.connections == 0


async def test_realtime_observes_usage_from_failed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = audio_events()
    events[-1]["response"]["status"] = "incomplete"
    accounting = RequestAccounting()
    token = request_accounting.set(accounting)
    try:
        async with realtime_peer(monkeypatch, SpeechPeer(list(events))):
            with pytest.raises(ResponseFormatError, match="incomplete"):
                await realtime_speech()
    finally:
        request_accounting.reset(token)
    assert accounting.dispatched
    assert accounting.usage == {"input_tokens": 10, "output_tokens": 20}
    assert accounting.previous_usages == []


@pytest.mark.parametrize(
    "limit", ["REALTIME_TIMEOUT_SECONDS", "REALTIME_EVENT_TIMEOUT_SECONDS"]
)
async def test_realtime_timeouts_close_connection(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    monkeypatch.setattr(f"src.tts_provider.{limit}", 0.03)
    peer = SpeechPeer([], stall=True)
    async with realtime_peer(monkeypatch, peer):
        with pytest.raises(ResponseFormatError, match="timed out"):
            await realtime_speech()
        await asyncio.wait_for(peer.closed.wait(), 2)


@pytest.mark.parametrize(
    "limit", ["MAX_REALTIME_AUDIO_BYTES", "MAX_REALTIME_MESSAGE_BYTES"]
)
async def test_realtime_bounds_audio_and_message_sizes(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    events: list[dict[str, Any] | str] = list(audio_events())
    if limit == "MAX_REALTIME_AUDIO_BYTES":
        monkeypatch.setattr(f"src.tts_provider.{limit}", 2)
    else:
        monkeypatch.setattr(f"src.tts_provider.{limit}", 2048)
        events = [{"type": "ignored", "data": "a" * 4096}]
    async with realtime_peer(monkeypatch, SpeechPeer(events)):
        with pytest.raises(ResponseFormatError):
            await realtime_speech()


@pytest.mark.parametrize(
    "model", ["tts-1", "gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"]
)
@pytest.mark.parametrize("part", ["text", "instructions"])
async def test_speech_limits_before_network_and_supports_snapshot_instructions(
    monkeypatch: pytest.MonkeyPatch, model: TTSModels, part: str
) -> None:
    monkeypatch.setattr(
        "src.tts_provider.config",
        SimpleNamespace(
            custom_providers=[], openai_api_key="synthetic", openai_endpoint=None
        ),
    )
    request = AsyncMock(return_value=b"audio")
    monkeypatch.setattr(provider_runtime, "request_bytes", request)
    args = {"input": "x" * 4096, "instructions": "x" * 4096}
    await TTSProvider().async_get_tts_response(
        model=model,
        provider="openai",
        voice="alloy",
        strip_html=False,
        input=args["input"],
        instructions=args["instructions"],
    )
    payload = request.call_args.kwargs["json_payload"]
    if model.startswith("gpt-4o-mini-tts"):
        assert payload["instructions"] == args["instructions"]
    args["input" if part == "text" else "instructions"] += "x"
    with pytest.raises(ValueError, match="4,096"):
        await TTSProvider().async_get_tts_response(
            model=model,
            provider="openai",
            voice="alloy",
            strip_html=False,
            input=args["input"],
            instructions=args["instructions"],
        )
    request.assert_awaited_once()


async def test_realtime_rejects_incompatible_saved_voice_without_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.tts_provider.config", SimpleNamespace(custom_providers=[]))
    with pytest.raises(ValueError, match="Choose another voice"):
        await TTSProvider().async_get_tts_response(
            "Hello", "gpt-realtime-2.1-mini", "openai", "nova", False
        )
    assert len(openai_tts_voices("gpt-realtime-2.1-mini")) == 10
    assert len(openai_tts_voices("gpt-4o-mini-tts")) == 13
    assert (
        tts_audio_extension(
            {
                "provider": "openai",
                "model": "gpt-realtime-2.1-mini",
                "voice": "cedar",
                "enabled": True,
                "language": None,
            }
        )
        == "wav"
    )
