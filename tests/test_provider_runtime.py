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

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.provider_runtime import (
    ProviderRuntime,
    ProviderTimeoutError,
    RequestTimeouts,
    TrafficController,
)


def test_traffic_controller_ramps_after_success_streak() -> None:
    controller = TrafficController(initial_window=4)

    for _ in range(20):
        controller.note_success()

    metrics = controller.snapshot()
    assert metrics["window"] == 5


def test_traffic_controller_throttle_reduces_window_and_sets_cooldown() -> None:
    controller = TrafficController(initial_window=4)
    controller.note_throttle(
        retry_after_sec=3.0,
        reset_after_sec=None,
        shared_capacity=False,
    )

    metrics = controller.snapshot()
    assert metrics["window"] == 2
    assert metrics["cooldown_remaining"] > 0


def test_traffic_controller_shared_capacity_blocks_ramp() -> None:
    controller = TrafficController(initial_window=4)
    before = time.time()
    controller.note_throttle(
        retry_after_sec=None,
        reset_after_sec=None,
        shared_capacity=True,
    )

    metrics = controller.snapshot()
    assert metrics["window"] == 2
    assert 0 < metrics["ramp_blocked_remaining"] <= 900
    assert metrics["ramp_blocked_remaining"] >= 899 - (time.time() - before)


def test_traffic_controller_timeout_reduces_window_by_one() -> None:
    controller = TrafficController(initial_window=2)
    controller.note_timeout()

    metrics = controller.snapshot()
    assert metrics["window"] == 1
    assert metrics["timeouts"] == 1


@pytest.mark.asyncio
async def test_provider_runtime_closes_current_loop_session() -> None:
    runtime = ProviderRuntime()
    loop = asyncio.get_running_loop()
    client = AsyncMock()
    client.is_closed = False
    runtime._clients[loop] = client

    await runtime.close_current_session()

    client.aclose.assert_awaited_once()
    assert loop not in runtime._clients


@pytest.mark.asyncio
async def test_stream_idle_timeout_reports_last_event_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntime()

    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"Content-Type": "text/event-stream"}
            self.url = "https://api.openai.com/v1/responses"
            self.text = ""

        async def _line_generator(self):
            yield 'data: {"type":"response.created"}'
            yield ""
            await asyncio.sleep(0.05)

        def aiter_lines(self):
            return self._line_generator()

    class FakeStreamContext:
        def __init__(self, response: FakeResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeResponse:
            return self._response

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            return None

    class FakeClient:
        def stream(self, *args, **kwargs) -> FakeStreamContext:
            del args, kwargs
            return FakeStreamContext(FakeResponse())

    monkeypatch.setattr(
        runtime,
        "get_http_client",
        AsyncMock(return_value=FakeClient()),
    )

    with pytest.raises(ProviderTimeoutError) as exc_info:
        async for _ in runtime.stream_sse_json(
            key="text:openai:gpt-5.4",
            initial_window=1,
            method="POST",
            url="https://api.openai.com/v1/responses",
            provider="openai",
            model="gpt-5.4",
            timeouts=RequestTimeouts(
                connect_timeout_sec=1.0,
                first_event_timeout_sec=1.0,
                stream_idle_timeout_sec=0.01,
            ),
            max_retries=0,
        ):
            pass

    assert exc_info.value.phase == "stream_idle"
    assert exc_info.value.last_event_type == "response.created"
