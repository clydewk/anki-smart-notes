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

from collections.abc import Callable
from typing import Any

from aqt import mw
from aqt.qt import QWidget

from .job_context import JobCancelledError
from .logger import logger
from .provider_runtime import (
    ProviderHTTPError,
    ProviderTimeoutError,
    format_provider_http_error_for_log,
)
from .tasks import JobHandle, run_async_in_background
from .ui.ui_utils import show_message_box


def with_sentry(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.exception("Error caught in wrapper")
            raise

    return wrapper


def log_async_failure(error: Exception) -> None:
    exc_info = (type(error), error, error.__traceback__)
    if isinstance(error, ProviderHTTPError):
        logger.error(
            "Async operation failed with provider HTTP error: %s",
            format_provider_http_error_for_log(error),
            exc_info=exc_info,
        )
        return

    if isinstance(error, ProviderTimeoutError):
        logger.error(
            "Async operation failed with provider timeout: provider=%s model=%s phase=%s timeout=%s attempt=%s last_event=%s url=%s",
            error.provider,
            error.model,
            error.phase,
            error.timeout_sec,
            error.attempt,
            error.last_event_type or "n/a",
            error.url,
            exc_info=exc_info,
        )
        return

    logger.error("Async operation failed: %s", error, exc_info=exc_info)


def run_async_in_background_with_sentry(
    op: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_failure: Callable[[Exception], None] | None = None,
    *,
    owner_alive: Callable[[], bool] | None = None,
    owner: QWidget | None = None,
    on_finished: Callable[[], None] | None = None,
    show_error: bool = True,
) -> JobHandle:
    """Run collection-independent async work with consistent error handling."""

    if not mw:
        raise Exception("Error: mw not found in run_async_in_background")

    def wrapped_on_failure(e: Exception) -> None:
        if isinstance(e, JobCancelledError):
            if on_failure:
                on_failure(e)
            return
        log_async_failure(e)
        if show_error:
            show_message_box("Smart Notes Error", str(e))
        if on_failure:
            on_failure(e)

    return run_async_in_background(
        op,
        on_success,
        wrapped_on_failure,
        owner_alive=owner_alive,
        owner=owner,
        on_finished=on_finished,
    )


def pinger(event: str) -> Callable[[], Any]:
    # No-op pinger
    async def ping() -> None:
        pass

    return ping


sentry = None
