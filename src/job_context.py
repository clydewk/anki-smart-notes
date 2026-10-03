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

from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from aqt.qt import QWidget


NO_COLLECTION = object()


class JobCancelledError(Exception):
    """The job or its originating profile is no longer active."""


@dataclass(eq=False)
class JobContext:
    collection: Any
    generation: int
    active: Event
    cancelled: Event = field(default_factory=Event)
    owner_alive: Callable[[], bool] | None = None
    connection_valid: Callable[[], bool] | None = None
    _bridges: set[asyncio.Future[Any]] = field(default_factory=set, repr=False)
    _bridge_lock: Lock = field(default_factory=Lock, repr=False)

    def track_bridge(self, future: asyncio.Future[Any]) -> None:
        with self._bridge_lock:
            self._bridges.add(future)
        future.add_done_callback(self._forget_bridge)
        if not self.valid():
            future.cancel()

    def _forget_bridge(self, future: asyncio.Future[Any]) -> None:
        with self._bridge_lock:
            self._bridges.discard(future)

    def cancel(self) -> None:
        self.cancelled.set()
        with self._bridge_lock:
            pending = tuple(self._bridges)
        for future in pending:
            deliver_future(future.get_loop(), future.cancel)

    def valid(self) -> bool:
        return (
            self.active.is_set()
            and not self.cancelled.is_set()
            and (self.connection_valid is None or self.connection_valid())
        )

    def check(self, collection: Any = NO_COLLECTION) -> None:
        if not self.valid() or (
            collection is not NO_COLLECTION and collection is not self.collection
        ):
            raise JobCancelledError(
                "Generation belongs to a closed or changed profile."
            )

    def ui_valid(self) -> bool:
        return self.valid() and self.owner_valid()

    def owner_valid(self) -> bool:
        try:
            return self.owner_alive is None or self.owner_alive()
        except RuntimeError:
            # Qt raises when a Python wrapper outlives its C++ widget.
            return False


current_job: ContextVar[JobContext | None] = ContextVar("smart_notes_job", default=None)


def require_job() -> JobContext:
    context = current_job.get()
    if context is None:
        raise RuntimeError("Collection work requires an owned Smart Notes job.")
    context.check()
    return context


def deliver_future(
    loop: asyncio.AbstractEventLoop, action: Callable[[], object]
) -> None:
    if not loop.is_closed():
        # The worker can finish between the check and scheduling.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(action)


ui_job_owner: ContextVar[QWidget | None] = ContextVar(
    "smart_notes_ui_job_owner", default=None
)
