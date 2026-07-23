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

import inspect
from unittest.mock import MagicMock

import pytest

from src import tasks


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


def test_background_helper_requires_anki_main_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "mw", None)

    async def operation() -> None:
        return None

    with pytest.raises(Exception, match="mw not found"):
        tasks.run_async_in_background(operation)
