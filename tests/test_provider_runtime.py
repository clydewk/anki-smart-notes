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
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.provider_runtime import (
    ProviderHTTPError,
    ProviderRuntime,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RequestTimeouts,
    TrafficController,
    extract_provider_error_detail,
    format_provider_http_error_for_log,
    parse_google_retry_delay,
)


def json_body(value: object) -> str:
    return json.dumps(value)


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


class ProviderRuntimeHarness(ProviderRuntime):
    def set_loop_client(
        self, loop: asyncio.AbstractEventLoop, client: AsyncMock
    ) -> None:
        self._clients[loop] = client

    def has_loop_client(self, loop: asyncio.AbstractEventLoop) -> bool:
        return loop in self._clients

    async def handle_http_error(
        self,
        *,
        controller: TrafficController,
        provider: str,
        model: str,
        attempt: int,
        max_retries: int,
        error: ProviderHTTPError,
    ) -> bool:
        return await self._handle_http_error(
            controller=controller,
            provider=provider,
            model=model,
            attempt=attempt,
            max_retries=max_retries,
            error=error,
        )

    async def handle_timeout_error(
        self,
        *,
        controller: TrafficController,
        attempt: int,
        max_retries: int,
        error: ProviderTimeoutError,
        allow_retry: bool,
    ) -> bool:
        return await self._handle_timeout_error(
            controller=controller,
            attempt=attempt,
            max_retries=max_retries,
            error=error,
            allow_retry=allow_retry,
        )


@pytest.mark.asyncio
async def test_provider_runtime_closes_current_loop_session() -> None:
    runtime = ProviderRuntimeHarness()
    loop = asyncio.get_running_loop()
    client = AsyncMock()
    client.is_closed = False
    runtime.set_loop_client(loop, client)

    await runtime.close_current_session()

    client.aclose.assert_awaited_once()
    assert not runtime.has_loop_client(loop)


def test_provider_runtime_scopes_controllers_per_event_loop() -> None:
    runtime = ProviderRuntime()

    async def exercise_controller() -> TrafficController:
        controller = runtime.controller("text:openai:gpt-5.4", initial_window=1)
        assert controller is runtime.controller("text:openai:gpt-5.4", initial_window=1)

        await controller.acquire()
        waiter = asyncio.create_task(controller.acquire())
        await asyncio.sleep(0)
        await controller.release()
        await waiter
        await controller.release()

        assert "text:openai:gpt-5.4" in runtime.get_metrics_summary()

        await runtime.close_current_session()
        return controller

    first_controller = asyncio.run(exercise_controller())
    second_controller = asyncio.run(exercise_controller())

    assert first_controller is not second_controller


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

        async def _line_generator(self) -> AsyncIterator[str]:
            yield 'data: {"type":"response.created"}'
            yield ""
            await asyncio.sleep(0.05)

        def aiter_lines(self) -> AsyncIterator[str]:
            return self._line_generator()

    class FakeStreamContext:
        def __init__(self, response: FakeResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeResponse:
            return self._response

        async def __aexit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc: Optional[BaseException],
            tb: Optional[Any],
        ) -> None:
            del exc_type, exc, tb
            return None

    class FakeClient:
        def stream(self, *args: Any, **kwargs: Any) -> FakeStreamContext:
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


def test_extract_provider_error_detail_prefers_structured_fields() -> None:
    body = """
    {
        "error": {
            "message": "Model gemini-3-pro-preview is not available.",
            "type": "invalid_request_error",
            "code": "model_not_found"
        }
    }
    """

    detail = extract_provider_error_detail(body)

    assert "message=Model gemini-3-pro-preview is not available." in detail
    assert "type=invalid_request_error" in detail
    assert "code=model_not_found" in detail


def test_format_provider_http_error_for_log_includes_trace_id() -> None:
    error = ProviderHTTPError(
        500,
        "Internal Server Error",
        '{"error":{"message":"Upstream timeout","type":"server_error"}}',
        "https://example.com/v1/responses",
        {"x-request-id": "req_123"},
    )

    formatted = format_provider_http_error_for_log(error)

    assert "status=500" in formatted
    assert "trace_id=req_123" in formatted
    assert "message=Upstream timeout" in formatted


def test_parse_google_retry_delay_from_retry_info() -> None:
    body = """
    {
        "error": {
            "code": 429,
            "message": "Quota exceeded. Please retry in 21.5s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "21s"
                }
            ]
        }
    }
    """

    assert parse_google_retry_delay(body) == 21.0


@pytest.mark.asyncio
async def test_handle_http_error_logs_non_retryable_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    fake_logger = MagicMock()
    monkeypatch.setattr("src.provider_runtime.logger", fake_logger)

    error = ProviderHTTPError(
        400,
        "Bad Request",
        '{"error":{"message":"Unsupported field","type":"invalid_request_error"}}',
        "https://example.com/v1/responses",
        {"x-request-id": "req_non_retry"},
    )

    with pytest.raises(ProviderHTTPError):
        await runtime.handle_http_error(
            controller=TrafficController(initial_window=1),
            provider="custom",
            model="gemini-3-pro-preview",
            attempt=0,
            max_retries=2,
            error=error,
        )

    log_call = fake_logger.error.call_args
    assert log_call is not None
    assert log_call.args[1:6] == (
        "custom",
        "gemini-3-pro-preview",
        1,
        3,
        "no",
    )
    assert "status=400" in log_call.args[6]
    assert "trace_id=req_non_retry" in log_call.args[6]


@pytest.mark.asyncio
async def test_handle_http_error_logs_retryable_failures_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    fake_logger = MagicMock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.logger", fake_logger)
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)

    should_retry = await runtime.handle_http_error(
        controller=TrafficController(initial_window=1),
        provider="openai",
        model="gpt-5.4",
        attempt=0,
        max_retries=2,
        error=ProviderHTTPError(
            500,
            "Internal Server Error",
            '{"error":{"message":"Transient upstream error","type":"server_error"}}',
            "https://api.openai.com/v1/responses",
            {"x-request-id": "req_retry"},
        ),
    )

    assert should_retry is True
    sleep_mock.assert_awaited_once()

    log_call = fake_logger.warning.call_args
    assert log_call is not None
    assert log_call.args[1:6] == ("openai", "gpt-5.4", 1, 3, "yes")
    assert "status=500" in log_call.args[6]
    assert "trace_id=req_retry" in log_call.args[6]


@pytest.mark.asyncio
async def test_handle_google_retry_info_quota_error_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)

    should_retry = await runtime.handle_http_error(
        controller=TrafficController(initial_window=1),
        provider="google_tts",
        model="gemini-3.1-flash-tts-preview",
        attempt=0,
        max_retries=2,
        error=ProviderHTTPError(
            429,
            "Too Many Requests",
            json_body(
                {
                    "error": {
                        "code": 429,
                        "message": "Quota exceeded. Please retry in 3s.",
                        "status": "RESOURCE_EXHAUSTED",
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                "violations": [
                                    {
                                        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                    }
                                ],
                            },
                            {
                                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                "retryDelay": "3s",
                            },
                        ],
                    }
                }
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent",
            {},
        ),
    )

    assert should_retry is True
    sleep_mock.assert_awaited_once_with(3.0)


@pytest.mark.asyncio
async def test_handle_google_daily_quota_error_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)
    controller = TrafficController(initial_window=1)

    with pytest.raises(ProviderUnavailableError):
        await runtime.handle_http_error(
            controller=controller,
            provider="google_tts",
            model="gemini-3.1-flash-tts-preview",
            attempt=0,
            max_retries=2,
            error=ProviderHTTPError(
                429,
                "Too Many Requests",
                json_body(
                    {
                        "error": {
                            "code": 429,
                            "message": "You exceeded your current quota.",
                            "status": "RESOURCE_EXHAUSTED",
                            "details": [
                                {
                                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [
                                        {
                                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent",
                {},
            ),
        )

    sleep_mock.assert_not_awaited()
    with pytest.raises(ProviderUnavailableError):
        await controller.acquire()


@pytest.mark.asyncio
async def test_handle_http_error_does_not_retry_known_custom_provider_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    fake_logger = MagicMock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.logger", fake_logger)
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)

    error = ProviderHTTPError(
        500,
        "Internal Server Error",
        '{"error":{"message":"not implemented","type":"rix_api_error","code":"convert_request_failed"}}',
        "https://example.com/v1/responses",
        {"x-request-id": "req_no_retry"},
    )

    with pytest.raises(ProviderHTTPError):
        await runtime.handle_http_error(
            controller=TrafficController(initial_window=1),
            provider="Compat",
            model="compat-model",
            attempt=0,
            max_retries=2,
            error=error,
        )

    sleep_mock.assert_not_awaited()
    log_call = fake_logger.error.call_args
    assert log_call is not None
    assert log_call.args[1:6] == ("Compat", "compat-model", 1, 3, "no")
    assert "convert_request_failed" in log_call.args[6]


@pytest.mark.asyncio
async def test_handle_timeout_error_logs_retryable_timeouts_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    fake_logger = MagicMock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.logger", fake_logger)
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)

    should_retry = await runtime.handle_timeout_error(
        controller=TrafficController(initial_window=1),
        attempt=0,
        max_retries=2,
        error=ProviderTimeoutError(
            provider="openai",
            model="gpt-5.4",
            url="https://api.openai.com/v1/responses",
            phase="read_idle",
            timeout_sec=30.0,
            attempt=1,
        ),
        allow_retry=True,
    )

    assert should_retry is True
    sleep_mock.assert_awaited_once()

    log_call = fake_logger.warning.call_args
    assert log_call is not None
    assert log_call.args[1:6] == ("openai", "gpt-5.4", 1, 3, "yes")
    assert "phase=read_idle" in log_call.args[6]
    assert "timeout=30s" in log_call.args[6]


@pytest.mark.asyncio
async def test_handle_timeout_error_does_not_retry_custom_read_idle_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ProviderRuntimeHarness()
    fake_logger = MagicMock()
    sleep_mock = AsyncMock()
    monkeypatch.setattr("src.provider_runtime.logger", fake_logger)
    monkeypatch.setattr("src.provider_runtime.asyncio.sleep", sleep_mock)

    should_retry = await runtime.handle_timeout_error(
        controller=TrafficController(initial_window=1),
        attempt=0,
        max_retries=2,
        error=ProviderTimeoutError(
            provider="Compat",
            model="compat-model",
            url="https://example.com/v1/chat/completions",
            phase="read_idle",
            timeout_sec=30.0,
            attempt=1,
        ),
        allow_retry=True,
    )

    assert should_retry is False
    sleep_mock.assert_not_awaited()

    log_call = fake_logger.error.call_args
    assert log_call is not None
    assert log_call.args[1:6] == ("Compat", "compat-model", 1, 3, "no")
    assert "phase=read_idle" in log_call.args[6]
    assert "timeout=30s" in log_call.args[6]
