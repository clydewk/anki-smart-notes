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
import concurrent.futures
import sys
import threading
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from aqt import mw
from PyQt6 import sip

from .config import config, config_context
from .job_context import (
    JobCancelledError,
    JobContext,
    current_job,
    deliver_future,
    ui_job_owner,
)
from .logger import logger
from .mcp_runtime import mcp_runtime
from .provider_runtime import provider_runtime

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from aqt.qt import QWidget

T = TypeVar("T")
SHUTDOWN_TIMEOUT_SEC = 4.5
RESOURCE_CLOSE_TIMEOUT_SEC = 2.0


class JobHandle:
    def __init__(
        self,
        future: concurrent.futures.Future[Any],
        context: JobContext,
        loop: asyncio.AbstractEventLoop,
        on_finished: Callable[[], None] | None,
    ) -> None:
        self.future = future
        self.context = context
        self.loop = loop
        self._cleanup = [on_finished] if on_finished is not None else []
        self._finished = False

    def cancel(self) -> None:
        if self.context.cancelled.is_set():
            return
        self.context.cancel()
        self.future.cancel()
        deliver_future(self.loop, self._cancel_owned_tasks)

    def _cancel_owned_tasks(self) -> None:
        for task in asyncio.all_tasks(self.loop):
            if (
                task.get_context().get(current_job) is self.context
                and not task.cancelling()
            ):
                task.cancel()

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        if self._finished:
            callback()
        else:
            self._cleanup.append(callback)

    def finish(self) -> None:
        """Release bookkeeping on the main thread, including when the owner closed."""
        if self._finished:
            return
        self._finished = True
        callbacks, self._cleanup = self._cleanup, []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Job cleanup callback failed.")


class ProfileWorker:
    def __init__(self, collection: Any, generation: int) -> None:
        self.collection = collection
        self.generation = generation
        self.active = threading.Event()
        self.active.set()
        self.loop = (
            asyncio.ProactorEventLoop()
            if sys.platform == "win32"
            else asyncio.new_event_loop()
        )
        self.jobs: set[JobHandle] = set()
        self.thread = threading.Thread(
            target=self._run, name="Smart Notes", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.wait(pending, timeout=0.1))
            self.loop.close()

    def submit(
        self,
        op: Callable[[], Awaitable[T]],
        on_success: Callable[[T], None],
        on_failure: Callable[[Exception], None] | None,
        owner_alive: Callable[[], bool] | None,
        on_finished: Callable[[], None] | None = None,
    ) -> JobHandle:
        if not mw or mw.col is not self.collection or not self.active.is_set():
            raise JobCancelledError("Smart Notes profile is closing.")
        context = JobContext(
            self.collection, self.generation, self.active, owner_alive=owner_alive
        )
        if not context.ui_valid():
            raise JobCancelledError("The originating window is closed.")
        settings = config.snapshot()
        taskman = mw.taskman

        async def run() -> T:
            job_token = current_job.set(context)
            config_token = config_context.set(settings)
            try:
                context.check()
                return await op()
            finally:
                config_context.reset(config_token)
                current_job.reset(job_token)

        future = asyncio.run_coroutine_threadsafe(run(), self.loop)
        handle = JobHandle(future, context, self.loop, on_finished)
        self.jobs.add(handle)

        def completed(result: concurrent.futures.Future[T]) -> None:
            def dispatch() -> None:
                try:
                    if (
                        not self.active.is_set()
                        or not mw
                        or mw.col is not self.collection
                        or not context.owner_valid()
                    ):
                        return
                    try:
                        value = result.result()
                        context.check(mw.col)
                        on_success(value)
                    except concurrent.futures.CancelledError:
                        if on_failure:
                            on_failure(JobCancelledError("Generation cancelled."))
                    except Exception as error:
                        if on_failure:
                            on_failure(error)
                        else:
                            logger.error("Background operation failed: %s", error)
                finally:
                    self.jobs.discard(handle)
                    handle.finish()

            if self.active.is_set():
                try:
                    taskman.run_on_main(dispatch)
                except RuntimeError:
                    # Profile close synchronously finalizes jobs if Qt has stopped.
                    logger.debug(
                        "Qt stopped before background completion could be delivered."
                    )

        future.add_done_callback(completed)
        return handle

    def close(self) -> None:
        if not self.active.is_set():
            return
        self.active.clear()
        for handle in tuple(self.jobs):
            handle.cancel()
            handle.finish()
        self.jobs.clear()

        async def shutdown() -> None:
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            ]
            for task in pending:
                if not task.cancelling():
                    task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=2.0)
            async with asyncio.timeout(RESOURCE_CLOSE_TIMEOUT_SEC):
                try:
                    # MCP HTTP sessions need the shared client for their final DELETE.
                    await mcp_runtime.close_current_session()
                finally:
                    await provider_runtime.close_current_session()

        future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
        try:
            future.result(timeout=SHUTDOWN_TIMEOUT_SEC)
        except Exception:
            future.cancel()
            logger.warning("Profile cleanup exceeded its deadline or failed.")
        finally:
            with suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=0.5)


worker: ProfileWorker | None = None
profile_generation = 0


def open_profile_worker() -> None:
    global worker, profile_generation
    close_profile_worker()
    if mw and mw.col:
        profile_generation += 1
        worker = ProfileWorker(mw.col, profile_generation)


def close_profile_worker() -> None:
    global worker, profile_generation
    previous, worker = worker, None
    if previous:
        profile_generation += 1
        previous.close()


def run_async_in_background(
    op: Callable[[], Awaitable[T]],
    on_success: Callable[[T], None] = lambda _: None,
    on_failure: Callable[[Exception], None] | None = None,
    *,
    owner_alive: Callable[[], bool] | None = None,
    owner: QWidget | None = None,
    on_finished: Callable[[], None] | None = None,
) -> JobHandle:
    if not mw or not mw.col:
        raise RuntimeError("Anki main window or collection is not available.")
    if worker is None or worker.collection is not mw.col:
        raise JobCancelledError("Smart Notes profile is not open.")
    owner = owner if owner is not None else ui_job_owner.get()
    if owner_alive is None and owner is not None:

        def widget_alive() -> bool:
            if sip.isdeleted(owner):
                return False
            window = owner.window()
            return window is not None and window.isVisible()

        owner_alive = widget_alive

    handle = worker.submit(op, on_success, on_failure, owner_alive, on_finished)
    if owner is not None:

        def cancel_owner(*_: object) -> None:
            handle.cancel()

        signals = [owner.destroyed]
        window = owner.window()
        finished = getattr(window, "finished", None)
        if finished is not None:
            signals.append(finished)
        for signal in signals:
            signal.connect(cancel_owner)
            handle.add_cleanup(partial(signal.disconnect, cancel_owner))
    return handle
