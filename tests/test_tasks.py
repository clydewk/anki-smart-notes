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
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import collection_ops, tasks


def test_background_helper_is_always_collection_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = MagicMock()
    query.without_collection.return_value = query
    query_type = MagicMock(return_value=query)

    monkeypatch.setattr(tasks, "mw", MagicMock())
    monkeypatch.setattr(tasks, "QueryOp", query_type)

    async def operation() -> str:
        return "done"

    on_success = MagicMock()
    on_failure = MagicMock()
    tasks.run_async_in_background(operation, on_success, on_failure)

    query.without_collection.assert_called_once_with()
    query.failure.assert_called_once_with(on_failure)
    query.run_in_background.assert_called_once_with()

    parameters = inspect.signature(tasks.run_async_in_background).parameters
    assert "use_collection" not in parameters
    assert "with_progress" not in parameters


@pytest.mark.parametrize("should_fail", [False, True])
def test_background_helper_closes_loop_scoped_sessions(
    monkeypatch: pytest.MonkeyPatch,
    should_fail: bool,
) -> None:
    query = MagicMock()
    query.without_collection.return_value = query
    query_type = MagicMock(return_value=query)
    close_mcp = AsyncMock()
    close_provider = AsyncMock()

    monkeypatch.setattr(tasks, "mw", MagicMock())
    monkeypatch.setattr(tasks, "QueryOp", query_type)
    monkeypatch.setattr(tasks.mcp_runtime, "close_current_session", close_mcp)
    monkeypatch.setattr(tasks.provider_runtime, "close_current_session", close_provider)

    async def operation() -> str:
        if should_fail:
            raise RuntimeError("failed")
        return "done"

    tasks.run_async_in_background(operation)
    background_op = query_type.call_args.kwargs["op"]

    if should_fail:
        with pytest.raises(RuntimeError, match="failed"):
            background_op(None)
    else:
        assert background_op(None) == "done"

    close_mcp.assert_awaited_once_with()
    close_provider.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_collection_query_runs_while_provider_job_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_started = threading.Event()
    provider_finished = threading.Event()
    release_provider = threading.Event()
    provider_thread_id: Optional[int] = None
    collection_thread_id: Optional[int] = None

    with (
        ThreadPoolExecutor(max_workers=1) as detached_executor,
        ThreadPoolExecutor(max_workers=1) as collection_executor,
    ):

        class TaskManager:
            def run_on_main(self, operation: Any) -> None:
                operation()

        class ExecutorQueryOp:
            collection = object()

            def __init__(self, *, parent: Any, op: Any, success: Any) -> None:
                self.operation = op
                self.on_success = success
                self.on_failure = None
                self.detached = False

            def without_collection(self) -> "ExecutorQueryOp":
                self.detached = True
                return self

            def failure(self, callback: Any) -> "ExecutorQueryOp":
                self.on_failure = callback
                return self

            def run_in_background(self) -> None:
                executor = detached_executor if self.detached else collection_executor
                argument = None if self.detached else self.collection
                future = executor.submit(self.operation, argument)

                def done(completed: Any) -> None:
                    try:
                        result = completed.result()
                    except Exception as error:
                        assert self.on_failure is not None
                        self.on_failure(error)
                    else:
                        self.on_success(result)

                future.add_done_callback(done)

        main_window = SimpleNamespace(taskman=TaskManager())
        monkeypatch.setattr(tasks, "mw", main_window)
        monkeypatch.setattr(collection_ops, "mw", main_window)
        monkeypatch.setattr(tasks, "QueryOp", ExecutorQueryOp)
        monkeypatch.setattr(collection_ops, "QueryOp", ExecutorQueryOp)
        monkeypatch.setattr(
            tasks.mcp_runtime,
            "close_current_session",
            AsyncMock(),
        )
        monkeypatch.setattr(
            tasks.provider_runtime,
            "close_current_session",
            AsyncMock(),
        )

        async def slow_provider_operation() -> str:
            nonlocal provider_thread_id
            provider_thread_id = threading.get_ident()
            provider_started.set()
            while not release_provider.is_set():
                await asyncio.sleep(0.01)
            provider_finished.set()
            return "provider complete"

        def backup_query(_: object) -> str:
            nonlocal collection_thread_id
            collection_thread_id = threading.get_ident()
            assert not provider_finished.is_set()
            return "backup complete"

        try:
            tasks.run_async_in_background(slow_provider_operation)
            assert await asyncio.to_thread(provider_started.wait, 1)

            assert (
                await asyncio.wait_for(
                    collection_ops.query_collection(backup_query),
                    timeout=1,
                )
                == "backup complete"
            )
            assert provider_thread_id != collection_thread_id
            assert not provider_finished.is_set()
        finally:
            release_provider.set()

        assert await asyncio.to_thread(provider_finished.wait, 1)


def test_background_helper_requires_anki_main_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "mw", None)

    async def operation() -> None:
        return None

    with pytest.raises(Exception, match="mw not found"):
        tasks.run_async_in_background(operation)
