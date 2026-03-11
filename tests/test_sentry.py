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

import sys
from unittest.mock import MagicMock

import pytest

sys.modules["aqt.operations"] = MagicMock()

from src.provider_runtime import ProviderHTTPError, ProviderTimeoutError
from src.sentry import log_async_failure


def test_log_async_failure_logs_provider_http_error_with_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr("src.sentry.logger", fake_logger)

    try:
        raise ProviderHTTPError(
            500,
            "Internal Server Error",
            '{"error":{"message":"Upstream timeout","type":"server_error"}}',
            "https://example.com/v1/responses",
            {"x-request-id": "req_123"},
        )
    except ProviderHTTPError as exc:
        log_async_failure(exc)

    log_call = fake_logger.error.call_args
    assert log_call is not None
    assert "provider HTTP error" in log_call.args[0]
    assert "status=500" in log_call.args[1]
    assert log_call.kwargs["exc_info"][1].status == 500


def test_log_async_failure_logs_provider_timeout_with_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr("src.sentry.logger", fake_logger)

    try:
        raise ProviderTimeoutError(
            provider="openai",
            model="gpt-5.4",
            url="https://api.openai.com/v1/responses",
            phase="first_event",
            timeout_sec=45.0,
            attempt=2,
            last_event_type="response.created",
        )
    except ProviderTimeoutError as exc:
        log_async_failure(exc)

    log_call = fake_logger.error.call_args
    assert log_call is not None
    assert "provider timeout" in log_call.args[0]
    assert log_call.args[1:7] == (
        "openai",
        "gpt-5.4",
        "first_event",
        45.0,
        2,
        "response.created",
    )
    assert log_call.kwargs["exc_info"][1].phase == "first_event"
