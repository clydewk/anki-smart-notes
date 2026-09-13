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

from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast

from aqt.qt import QWidget

from .config import config, config_context
from .job_context import ui_job_owner
from .models import (
    overridable_chat_options,
    overridable_image_options,
    overridable_tts_options,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

P = ParamSpec("P")
R = TypeVar("R")
W = TypeVar("W", bound=QWidget)


def resolve_field_settings(
    defaults: Mapping[str, Any], extras: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve a detached set of field options; None means inherit."""
    values = deepcopy(dict(defaults))
    inherited = (
        set(
            overridable_chat_options
            + overridable_image_options
            + overridable_tts_options
        )
        if extras.get("use_custom_model") is False
        else set()
    )
    values.update(
        {
            key: deepcopy(value)
            for key, value in extras.items()
            if value is not None and key not in inherited
        }
    )
    return values


def widget_settings(widget: QWidget) -> dict[str, Any]:
    parent: Any = widget
    while parent is not None:
        draft = getattr(parent, "draft_settings", None)
        if callable(draft):
            return cast("Callable[[], dict[str, Any]]", draft)()
        parent = parent.parentWidget()
    return config.snapshot()


def with_widget_settings(
    function: Callable[Concatenate[W, P], R],
) -> Callable[Concatenate[W, P], R]:
    """Snapshot the containing settings draft while a UI action builds a job."""

    @wraps(function)
    def wrapped(self: W, *args: P.args, **kwargs: P.kwargs) -> R:
        token = config_context.set(widget_settings(self))
        owner_token = ui_job_owner.set(self)
        try:
            return function(self, *args, **kwargs)
        finally:
            ui_job_owner.reset(owner_token)
            config_context.reset(token)

    return wrapped
