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
import concurrent.futures
import queue
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import collection_ops, tasks
from src.job_context import current_job
from src.mcp_runtime import McpRuntime
from src.provider_runtime import ProviderRuntime
from tests.test_mcp import make_mcp_config, make_stdio_server_script


@pytest.fixture
def profile_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[tasks.ProfileWorker, queue.Queue[Callable[[], None]], Any]]:
    callbacks: queue.Queue[Callable[[], None]] = queue.Queue()
    window = SimpleNamespace(
        col=object(), taskman=SimpleNamespace(run_on_main=callbacks.put)
    )
    monkeypatch.setattr(tasks, "mw", window)
    monkeypatch.setattr(tasks.config, "snapshot", lambda: {})
    monkeypatch.setattr(tasks.mcp_runtime, "close_current_session", AsyncMock())
    monkeypatch.setattr(tasks.provider_runtime, "close_current_session", AsyncMock())
    tasks.open_profile_worker()
    assert tasks.worker is not None
    yield tasks.worker, callbacks, window
    tasks.close_profile_worker()


def test_jobs_share_loop_and_snapshot_origin(profile_worker: Any) -> None:
    worker, callbacks, window = profile_worker
    loops: list[asyncio.AbstractEventLoop] = []
    outputs: list[str] = []

    async def operation() -> str:
        context = current_job.get()
        assert context is not None and context.collection is window.col
        loops.append(asyncio.get_running_loop())
        return "done"

    first = tasks.run_async_in_background(operation, outputs.append)
    assert first.future.result(timeout=1) == "done"
    callbacks.get(timeout=1)()
    second = tasks.run_async_in_background(operation, outputs.append)
    assert second.future.result(timeout=1) == "done"
    callbacks.get(timeout=1)()
    assert loops == [worker.loop, worker.loop]
    assert outputs == ["done", "done"]


def test_late_callback_is_discarded_after_profile_close(profile_worker: Any) -> None:
    _, callbacks, window = profile_worker
    outputs: list[str] = []

    async def operation() -> str:
        return "old profile"

    handle = tasks.run_async_in_background(operation, outputs.append)
    assert handle.future.result(timeout=1) == "old profile"
    callback = callbacks.get(timeout=1)
    tasks.close_profile_worker()
    window.col = object()
    callback()
    assert outputs == []


def test_shutdown_cancels_without_pumping_ui(profile_worker: Any) -> None:
    worker, _, _ = profile_worker
    started = threading.Event()
    finished = threading.Event()

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()

    handle = tasks.run_async_in_background(operation)
    assert started.wait(1)
    tasks.close_profile_worker()
    assert finished.wait(1)
    assert not worker.thread.is_alive()
    assert handle.future.cancelled()


def test_cancelled_job_releases_and_next_job_runs(profile_worker: Any) -> None:
    async def operation() -> None:
        await asyncio.Future()

    handle = tasks.run_async_in_background(operation)
    handle.cancel()
    with pytest.raises(concurrent.futures.CancelledError):
        handle.future.result(timeout=1)

    async def next_operation() -> int:
        return 42

    assert tasks.run_async_in_background(next_operation).future.result(timeout=1) == 42


def test_profile_reopen_with_matching_note_ids_rejects_old_completion(
    profile_worker: Any,
) -> None:
    old_worker, callbacks, window = profile_worker
    outputs: list[str] = []
    finalized: list[int] = []

    async def operation() -> str:
        return "old result"

    handle = tasks.run_async_in_background(
        operation,
        outputs.append,
        on_finished=lambda: finalized.append(threading.get_ident()),
    )
    assert handle.future.result(timeout=1) == "old result"
    late_callback = callbacks.get(timeout=1)
    tasks.close_profile_worker()
    window.col = SimpleNamespace(notes={1: "profile B"})
    tasks.open_profile_worker()
    assert tasks.worker is not None
    assert tasks.worker.generation != old_worker.generation
    late_callback()
    assert outputs == []
    assert finalized == [threading.get_ident()]
    assert window.col.notes == {1: "profile B"}


def test_owner_close_cancels_work_and_finalizes_once(profile_worker: Any) -> None:
    worker, callbacks, _ = profile_worker
    owner = MagicMock()
    destroyed: list[Callable[..., None]] = []
    finished: list[Callable[..., None]] = []
    owner.destroyed.connect.side_effect = destroyed.append
    owner.destroyed.disconnect.side_effect = destroyed.remove
    owner.window().finished.connect.side_effect = finished.append
    owner.window().finished.disconnect.side_effect = finished.remove
    alive = True
    started = threading.Event()
    stopped = threading.Event()
    finalized: list[int] = []
    results: list[None] = []

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    handle = tasks.run_async_in_background(
        operation,
        results.append,
        owner=owner,
        owner_alive=lambda: alive,
        on_finished=lambda: finalized.append(threading.get_ident()),
    )
    assert started.wait(1)
    alive = False
    for callback in tuple(finished):
        callback(0)
    assert stopped.wait(1)
    callbacks.get(timeout=1)()
    assert handle.future.cancelled()
    assert results == []
    assert finalized == [threading.get_ident()]
    assert not destroyed and not finished
    assert not worker.jobs
    tasks.close_profile_worker()
    assert finalized == [threading.get_ident()]


def test_shutdown_cancels_pending_collection_bridge_without_qt(
    profile_worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, callbacks, window = profile_worker
    monkeypatch.setattr(collection_ops, "mw", window)
    started = threading.Event()
    stopped = threading.Event()
    writes: list[str] = []
    finalized: list[str] = []

    async def operation() -> None:
        started.set()
        try:
            await collection_ops.query_collection(lambda _: writes.append("unexpected"))
        finally:
            stopped.set()

    handle = tasks.run_async_in_background(
        operation, on_finished=lambda: finalized.append("finished")
    )
    assert started.wait(1)
    pending_schedule = callbacks.get(timeout=1)
    tasks.close_profile_worker()
    assert stopped.wait(1)
    pending_schedule()
    assert handle.future.cancelled()
    assert not worker.thread.is_alive()
    assert writes == []
    assert finalized == ["finished"]


def test_finalizer_runs_when_completion_callback_raises(profile_worker: Any) -> None:
    worker, callbacks, _ = profile_worker
    finalized: list[str] = []
    errors: list[Exception] = []

    async def operation() -> str:
        return "done"

    def failed_callback(_: str) -> None:
        raise ValueError("callback failed")

    handle = tasks.run_async_in_background(
        operation,
        failed_callback,
        errors.append,
        on_finished=lambda: finalized.append("finished"),
    )
    assert handle.future.result(timeout=1) == "done"
    callbacks.get(timeout=1)()
    assert len(errors) == 1 and str(errors[0]) == "callback failed"
    assert not worker.jobs
    assert finalized == ["finished"]


def test_cancellation_stops_owned_children_and_preserves_sibling_job(
    profile_worker: Any,
) -> None:
    child_started = threading.Event()
    child_stopped = threading.Event()

    async def child() -> None:
        child_started.set()
        try:
            await asyncio.Future()
        finally:
            child_stopped.set()

    async def parent() -> None:
        asyncio.create_task(child())
        await asyncio.Future()

    async def sibling() -> int:
        await asyncio.sleep(0.05)
        return 42

    first = tasks.run_async_in_background(parent)
    second = tasks.run_async_in_background(sibling)
    assert child_started.wait(1)
    first.cancel()
    assert child_stopped.wait(1)
    assert first.future.cancelled()
    assert second.future.result(timeout=1) == 42


def test_shutdown_bounds_resource_cleanup(
    profile_worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, _, _ = profile_worker
    cancelled = threading.Event()

    async def stalled_close() -> None:
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(tasks.mcp_runtime, "close_current_session", stalled_close)
    monkeypatch.setattr(tasks, "RESOURCE_CLOSE_TIMEOUT_SEC", 0.02)
    tasks.close_profile_worker()
    assert cancelled.wait(1)
    assert not worker.thread.is_alive()
    assert tasks.profile_generation > worker.generation


def test_shutdown_closes_mcp_before_its_shared_http_client(
    profile_worker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, callbacks, _ = profile_worker
    runtime = ProviderRuntime()
    clients = []

    async def use_client() -> None:
        clients.append(await runtime.get_http_client())

    async def close_mcp() -> None:
        # MCP closes its individual sessions in child tasks, each needing HTTP.
        await asyncio.gather(use_client())

    monkeypatch.setattr(tasks, "provider_runtime", runtime)
    monkeypatch.setattr(tasks.mcp_runtime, "close_current_session", close_mcp)
    tasks.run_async_in_background(use_client).future.result(timeout=1)
    callbacks.get(timeout=1)()
    tasks.close_profile_worker()
    assert len(clients) == 2 and clients[0] is clients[1]
    assert all(client.is_closed for client in clients)


def test_cancelled_caller_does_not_stop_shared_mcp_stderr_drain(
    profile_worker: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "stdio_server.py"
    make_stdio_server_script(script)
    server = make_mcp_config(sys.executable, [str(script)])
    runtime = McpRuntime()
    monkeypatch.setattr(tasks, "mcp_runtime", runtime)
    monkeypatch.setattr("src.mcp_runtime.get_version", lambda: "test-version")
    monkeypatch.setattr("src.mcp_runtime.MCP_DEFAULT_READ_TIMEOUT_SEC", 0.5)
    first_call_finished = threading.Event()

    async def first_job() -> None:
        await runtime.call_tool(server, "echo", {"text": "first"})
        first_call_finished.set()
        await asyncio.Future()

    async def second_job() -> str:
        result = await runtime.call_tool(
            server, "echo", {"text": "second", "noisy": True}
        )
        return result.text

    first = tasks.run_async_in_background(first_job)
    assert first_call_finished.wait(1)
    first.cancel()
    second = tasks.run_async_in_background(second_job)
    assert second.future.result(timeout=2) == "second"
