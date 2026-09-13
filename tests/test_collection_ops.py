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
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src import collection_ops
from src.job_context import JobCancelledError, JobContext, current_job

if TYPE_CHECKING:
    from collections.abc import Callable

    from anki.collection import OpChanges


class ImmediateTaskManager:
    def __init__(self) -> None:
        self.main_thread_calls = 0

    def run_on_main(self, operation: Any) -> None:
        self.main_thread_calls += 1
        operation()


class ImmediateQueryOp:
    collection = object()

    def __init__(self, *, parent: Any, op: Any, success: Any) -> None:
        self.operation = op
        self.on_success = success
        self.on_failure = None

    def failure(self, callback: Any) -> ImmediateQueryOp:
        self.on_failure = callback
        return self

    def run_in_background(self) -> None:
        try:
            self.on_success(self.operation(self.collection))
        except Exception as error:
            assert self.on_failure is not None
            self.on_failure(error)


class ImmediateCollectionOp:
    collection = object()

    def __init__(self, *, parent: Any, op: Any) -> None:
        self.operation = op
        self.on_success = None
        self.on_failure = None

    def success(self, callback: Any) -> ImmediateCollectionOp:
        self.on_success = callback
        return self

    def failure(self, callback: Any) -> ImmediateCollectionOp:
        self.on_failure = callback
        return self

    def run_in_background(self) -> None:
        try:
            assert self.on_success is not None
            self.on_success(self.operation(self.collection))
        except Exception as error:
            assert self.on_failure is not None
            self.on_failure(error)


@dataclass
class MutationResult:
    value: str
    changes: OpChanges


@pytest.fixture
def immediate_anki(monkeypatch: pytest.MonkeyPatch) -> Any:
    taskman = ImmediateTaskManager()
    monkeypatch.setattr(
        collection_ops,
        "mw",
        SimpleNamespace(taskman=taskman, col=ImmediateQueryOp.collection),
    )
    monkeypatch.setattr(collection_ops, "QueryOp", ImmediateQueryOp)
    monkeypatch.setattr(collection_ops, "CollectionOp", ImmediateCollectionOp)
    active = Event()
    active.set()
    context = JobContext(ImmediateQueryOp.collection, 1, active)
    ImmediateCollectionOp.collection = ImmediateQueryOp.collection
    token = current_job.set(context)
    yield taskman
    current_job.reset(token)


@pytest.mark.asyncio
async def test_query_collection_runs_on_collection_worker(
    immediate_anki: ImmediateTaskManager,
) -> None:
    observed_collection = None

    def operation(collection: object) -> str:
        nonlocal observed_collection
        observed_collection = collection
        return "result"

    assert await collection_ops.query_collection(operation) == "result"
    assert observed_collection is ImmediateQueryOp.collection
    assert immediate_anki.main_thread_calls == 1


@pytest.mark.asyncio
async def test_query_collection_propagates_failure(
    immediate_anki: ImmediateTaskManager,
) -> None:
    def operation(_: object) -> str:
        raise ValueError("query failed")

    with pytest.raises(ValueError, match="query failed"):
        await collection_ops.query_collection(operation)


@pytest.mark.asyncio
async def test_mutate_collection_runs_collection_op(
    immediate_anki: ImmediateTaskManager,
) -> None:
    result = MutationResult(value="saved", changes=cast("OpChanges", object()))
    observed_collection = None

    def operation(collection: object) -> MutationResult:
        nonlocal observed_collection
        observed_collection = collection
        return result

    assert await collection_ops.mutate_collection(operation) is result
    assert observed_collection is ImmediateCollectionOp.collection
    assert immediate_anki.main_thread_calls == 1


@pytest.mark.asyncio
async def test_mutate_collection_propagates_failure(
    immediate_anki: ImmediateTaskManager,
) -> None:
    def operation(_: object) -> MutationResult:
        raise RuntimeError("mutation failed")

    with pytest.raises(RuntimeError, match="mutation failed"):
        await collection_ops.mutate_collection(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["query_collection", "mutate_collection"])
async def test_collection_operation_requires_main_window(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    monkeypatch.setattr(collection_ops, "mw", None)
    operation = getattr(collection_ops, operation_name)

    with pytest.raises(RuntimeError, match="main window"):
        await operation(lambda _: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_name", ["query_collection", "mutate_collection"])
async def test_collection_operation_propagates_scheduling_failure(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    class BrokenTaskManager:
        def run_on_main(self, _: Any) -> None:
            raise RuntimeError("scheduling failed")

    monkeypatch.setattr(
        collection_ops,
        "mw",
        SimpleNamespace(taskman=BrokenTaskManager()),
    )
    operation = getattr(collection_ops, operation_name)
    active = Event()
    active.set()
    token = current_job.set(JobContext(object(), 1, active))

    with pytest.raises(RuntimeError, match="scheduling failed"):
        await operation(lambda _: None)
    current_job.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["cancel", "profile", "owner", "collection_closed"])
async def test_queued_bridge_cannot_start_after_origin_changes(
    immediate_anki: ImmediateTaskManager, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    pending: list[Callable[[], None]] = []
    monkeypatch.setattr(immediate_anki, "run_on_main", pending.append)
    context = current_job.get()
    assert context is not None
    calls: list[str] = []
    task = asyncio.create_task(
        collection_ops.query_collection(lambda _: calls.append("query"))
    )
    await asyncio.sleep(0)
    assert len(pending) == 1
    if reason == "cancel":
        context.cancel()
    elif reason == "owner":
        context.owner_alive = lambda: False
    elif reason == "collection_closed":
        monkeypatch.setattr(collection_ops.mw, "col", None)
    else:
        monkeypatch.setattr(
            collection_ops.mw, "col", SimpleNamespace(notes={1: "another profile"})
        )
    pending.pop()()
    with pytest.raises((asyncio.CancelledError, JobCancelledError)):
        await task
    assert calls == []


@pytest.mark.asyncio
async def test_bridge_rejects_late_result_after_owner_closes(
    immediate_anki: ImmediateTaskManager,
) -> None:
    successes: list[Callable[[str], None]] = []
    context = current_job.get()
    assert context is not None

    def start(
        success: Callable[[str], None], failure: Callable[[Exception], None]
    ) -> None:
        del failure
        successes.append(success)

    task = asyncio.create_task(collection_ops.await_collection_operation(start))
    await asyncio.sleep(0)
    context.owner_alive = lambda: False
    successes[0]("stale result")
    with pytest.raises(asyncio.CancelledError):
        await task
