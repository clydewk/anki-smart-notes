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
from collections.abc import Callable
from typing import Any, Optional

from aqt import mw
from aqt.operations import QueryOp

from .mcp_runtime import mcp_runtime
from .provider_runtime import provider_runtime


def run_async_in_background(
    op: Callable[[], Any],
    on_success: Callable[[Any], None] = lambda _: None,
    on_failure: Optional[Callable[[Exception], None]] = None,
) -> None:
    """Run collection-independent async work in the background."""

    if not mw:
        raise Exception("Error: mw not found in run_async_in_background")

    async def run_op_async() -> Any:
        try:
            return await op()
        finally:
            await mcp_runtime.close_current_session()
            await provider_runtime.close_current_session()

    def run_op(_):
        """
        Wrapper to avoid capturing FrameLocalsProxy objects in closures.
        Sentry's error tracking can create frame proxy objects that can't be pickled,
        causing "Cannot pickle 'FrameLocalsProxy' object" errors when QueryOp
        tries to serialize the operation for background execution.
        """
        return asyncio.run(run_op_async())

    query_op = QueryOp(
        parent=mw,
        op=run_op,
        success=on_success,
    ).without_collection()

    if on_failure:
        query_op.failure(on_failure)

    query_op.run_in_background()
