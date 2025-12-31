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

from typing import Any, Callable, Optional

from aqt import mw

from .logger import logger
from .tasks import run_async_in_background
from .ui.ui_utils import show_message_box


def with_sentry(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error caught in wrapper: {e}")
            raise e

    return wrapper


def run_async_in_background_with_sentry(
    op: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_failure: Optional[Callable[[Exception], None]] = None,
    with_progress: bool = False,
    use_collection: bool = True,
):
    "Runs an async operation in the background and calls on_success when done."

    if not mw:
        raise Exception("Error: mw not found in run_async_in_background")

    def wrapped_on_failure(e: Exception) -> None:
        logger.error(f"Async operation failed: {e}")
        show_message_box("Smart Notes Error", str(e))
        if on_failure:
            on_failure(e)

    run_async_in_background(
        op, on_success, wrapped_on_failure, with_progress, use_collection=use_collection
    )


def pinger(event: str) -> Callable[[], Any]:
    # No-op pinger
    async def ping() -> None:
        pass
    return ping

sentry = None
