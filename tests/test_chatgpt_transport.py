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
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.chatgpt_transport import (
    CATALOG_CLIENT_VERSION,
    ChatGPTTransport,
    ChatGPTTransportError,
    format_limits,
    parse_models,
)
from src.provider_runtime import (
    ProviderRuntime,
    RequestAccounting,
    ResponseFormatError,
    request_accounting,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


class TestAccount:
    __test__ = False
    generation = 1

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None]:
        yield

    async def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test-token", "ChatGPT-Account-Id": "account-a"}


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> TestAccount:
    value = TestAccount()
    monkeypatch.setattr("src.chatgpt_transport.chatgpt_auth", value)
    return value


@pytest.fixture
def http_client(monkeypatch: pytest.MonkeyPatch) -> Callable[[httpx.AsyncClient], None]:
    runtime = ProviderRuntime()
    monkeypatch.setattr("src.chatgpt_transport.provider_runtime", runtime)
    return lambda client: monkeypatch.setattr(
        runtime, "get_http_client", AsyncMock(return_value=client)
    )


async def collect(transport: ChatGPTTransport) -> list[dict[str, Any]]:
    return [event async for event in transport.stream_response({"model": "test"}, {})]


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
async def test_sse_chunk_framing_and_usage_replacement(
    http_client: Callable[[httpx.AsyncClient], None], ending: bytes
) -> None:
    events = [
        {"type": "response.in_progress", "response": {"usage": {"input_tokens": 3}}},
        {
            "type": "response.failed",
            "response": {"usage": {"input_tokens": 7, "output_tokens": 2}},
        },
    ]
    data = ending.join(
        [
            b": heartbeat",
            b"",
            b"event: response",
            b"data: " + json.dumps(events[0]).encode(),
            b"",
            b"data: " + json.dumps(events[1]).encode(),
            b"",
            b"",
        ]
    )
    accounting = RequestAccounting()
    token = request_accounting.set(accounting)
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream; charset=utf-8"},
                    stream=ChunkStream(
                        [data[i : i + 3] for i in range(0, len(data), 3)]
                    ),
                )
            )
        ) as client:
            http_client(client)
            assert await collect(ChatGPTTransport()) == events
            assert accounting.usage == {"input_tokens": 7, "output_tokens": 2}
            assert accounting.previous_usages == []
            await collect(ChatGPTTransport())
            assert accounting.previous_usages == [
                {"input_tokens": 7, "output_tokens": 2}
            ]
    finally:
        request_accounting.reset(token)


@pytest.mark.parametrize(
    "data", [b"data: secret-not-json\n\n", b"data: []\n\n", b'data: {"type":"partial"}']
)
async def test_invalid_or_truncated_event_has_no_payload_in_error(
    http_client: Callable[[httpx.AsyncClient], None], data: bytes
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=data
            )
        )
    ) as client:
        http_client(client)
        with pytest.raises(ResponseFormatError) as caught:
            await collect(ChatGPTTransport())
        assert "secret" not in str(caught.value)
        assert "partial" not in str(caught.value)


async def test_event_bound_counts_split_lines(
    monkeypatch: pytest.MonkeyPatch, http_client: Callable[[httpx.AsyncClient], None]
) -> None:
    monkeypatch.setattr("src.chatgpt_transport.MAX_MESSAGE_BYTES", 32)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ChunkStream(
                    [b":" + b"x" * 20 + b"\n", b":" + b"y" * 20 + b"\n"]
                ),
            )
        )
    ) as client:
        http_client(client)
        with pytest.raises(ResponseFormatError, match="size limit"):
            await collect(ChatGPTTransport())


@pytest.mark.parametrize("status", [302, 401, 403, 429, 503])
async def test_failures_do_not_follow_redirects_retry_or_expose_body(
    http_client: Callable[[httpx.AsyncClient], None], status: int
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            headers={"location": "https://other.test/?token=secret"},
            json={"error": "secret", "usage": {"output_tokens": 9}},
        )

    accounting = RequestAccounting()
    token = request_accounting.set(accounting)
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        ) as client:
            http_client(client)
            with pytest.raises(ChatGPTTransportError) as caught:
                await collect(ChatGPTTransport())
            assert "secret" not in str(caught.value)
            assert len(requests) == 1
            assert accounting.usage == {"output_tokens": 9}
    finally:
        request_accounting.reset(token)


async def test_ambiguous_network_failure_is_not_replayed(
    http_client: Callable[[httpx.AsyncClient], None],
) -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise httpx.ReadError("token=secret", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        with pytest.raises(ChatGPTTransportError) as caught:
            await collect(ChatGPTTransport())
        assert "secret" not in str(caught.value)
        assert requests == 1


async def test_cancellation_releases_controller_and_next_request_works(
    http_client: Callable[[httpx.AsyncClient], None],
) -> None:
    started = asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            started.set()
            await asyncio.Event().wait()
            yield b""

    requests = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=WaitingStream()
            if requests == 1
            else ChunkStream([b"data: [DONE]\n\n"]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        transport = ChatGPTTransport()
        task = asyncio.create_task(collect(transport))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await collect(transport) == [{"type": "done"}]


async def test_catalog_is_authenticated_preserves_reasoning_and_invalidates_cache(
    account: TestAccount, http_client: Callable[[httpx.AsyncClient], None]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/backend-api/codex/models"
        assert request.url.params["client_version"] == CATALOG_CLIENT_VERSION
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "new-model",
                        "display_name": "New model",
                        "visibility": "list",
                        "supported_reasoning_levels": [{"effort": "future"}],
                        "default_reasoning_level": "future",
                    },
                    {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        transport = ChatGPTTransport()
        models = await transport.fetch_models()
        assert [model.model for model in models] == ["new-model"]
        assert models[0].reasoning_efforts == ("future",)
        assert models[0].default_reasoning_effort == "future"
        assert transport.get_cached_models() == models
        models.clear()
        assert transport.get_cached_models()
        account.generation += 1
        assert transport.get_cached_models() == []


@pytest.mark.parametrize(
    "payload", [{}, {"models": None}, {"models": [{"visibility": "list", "slug": "x"}]}]
)
def test_malformed_catalog_never_becomes_empty_success(payload: dict[str, Any]) -> None:
    with pytest.raises(ResponseFormatError):
        parse_models(payload)


def test_subscription_limits_distinguish_missing_and_zero() -> None:
    assert format_limits({}) == "Subscription limits are unavailable."
    assert (
        format_limits(
            {
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 0,
                        "limit_window_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 100,
                        "limit_window_seconds": 604800,
                    },
                }
            }
        )
        == "5-hour window: 100% remaining; 7-day window: 0% remaining"
    )


async def test_control_request_bound_and_account_change(
    account: TestAccount,
    http_client: Callable[[httpx.AsyncClient], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed(_: httpx.Request) -> httpx.Response:
        account.generation += 1
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(changed)) as client:
        http_client(client)
        with pytest.raises(ChatGPTTransportError, match="account changed"):
            await ChatGPTTransport().fetch_models()
    monkeypatch.setattr("src.chatgpt_transport.MAX_MESSAGE_BYTES", 16)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 17))
    ) as client:
        http_client(client)
        with pytest.raises(ResponseFormatError, match="size limit"):
            await ChatGPTTransport().fetch_models()


async def test_control_timeout_is_bounded_and_sanitized(
    account: TestAccount,
    http_client: Callable[[httpx.AsyncClient], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.chatgpt_transport.CONTROL_TIMEOUT", 0.01)

    async def respond(_: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        with pytest.raises(ChatGPTTransportError, match="could not complete"):
            await ChatGPTTransport().fetch_models()


async def test_non_stream_response_is_unsuccessful_but_usage_is_retained(
    http_client: Callable[[httpx.AsyncClient], None],
) -> None:
    accounting = RequestAccounting()
    token = request_accounting.set(accounting)
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"usage": {"output_tokens": 9}})
            )
        ) as client:
            http_client(client)
            with pytest.raises(ResponseFormatError, match="streaming response"):
                await collect(ChatGPTTransport())
            assert accounting.usage == {"output_tokens": 9}
    finally:
        request_accounting.reset(token)


async def test_limits_use_separate_subscription_endpoint(
    account: TestAccount, http_client: Callable[[httpx.AsyncClient], None]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/backend-api/wham/usage"
        assert request.headers["ChatGPT-Account-Id"] == "account-a"
        return httpx.Response(
            200,
            json={
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 40,
                        "limit_window_seconds": 18000,
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        assert await ChatGPTTransport().fetch_limits() == "5-hour window: 60% remaining"


async def test_outgoing_message_bound_precedes_dispatch(
    monkeypatch: pytest.MonkeyPatch, http_client: Callable[[httpx.AsyncClient], None]
) -> None:
    monkeypatch.setattr("src.chatgpt_transport.MAX_MESSAGE_BYTES", 8)
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        with pytest.raises(ResponseFormatError, match="request exceeded"):
            await collect(ChatGPTTransport())
        assert calls == []


async def test_stream_request_content_type_and_accept_headers(
    http_client: Callable[[httpx.AsyncClient], None],
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "text/event-stream"
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(request.content) == {"model": "test"}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        http_client(client)
        assert await collect(ChatGPTTransport()) == [{"type": "done"}]
