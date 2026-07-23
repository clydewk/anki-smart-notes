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

# pyright: reportMissingImports=false

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from aqt import mw
from aqt.operations import CollectionOp, QueryOp

T = TypeVar("T")
StartOperation = Callable[
    [Callable[[T], None], Callable[[Exception], None]],
    None,
]


async def _await_collection_operation(start: StartOperation[T]) -> T:
    if not mw:
        raise RuntimeError("Anki main window is not available")

    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()

    def resolve(result: T) -> None:
        def set_result() -> None:
            if not future.done():
                future.set_result(result)

        loop.call_soon_threadsafe(set_result)

    def reject(error: Exception) -> None:
        def set_exception() -> None:
            if not future.done():
                future.set_exception(error)

        loop.call_soon_threadsafe(set_exception)

    def schedule() -> None:
        try:
            start(resolve, reject)
        except Exception as error:
            reject(error)

    try:
        mw.taskman.run_on_main(schedule)
    except Exception as error:
        reject(error)

    return await future


async def query_collection(operation: Callable[[Any], T]) -> T:
    """Run a short read operation on Anki's serialized collection worker."""

    def start(
        on_success: Callable[[T], None],
        on_failure: Callable[[Exception], None],
    ) -> None:
        QueryOp(parent=mw, op=operation, success=on_success).failure(
            on_failure
        ).run_in_background()

    return await _await_collection_operation(start)


async def mutate_collection(operation: Callable[[Any], T]) -> T:
    """Run a short undoable operation on Anki's serialized collection worker.

    The operation must return an Anki changes object, or an object with a
    ``changes`` attribute, so Anki can update undo state and refresh the UI.
    """

    def start(
        on_success: Callable[[T], None],
        on_failure: Callable[[Exception], None],
    ) -> None:
        CollectionOp(parent=mw, op=operation).success(on_success).failure(
            on_failure
        ).run_in_background()

    return await _await_collection_operation(start)
