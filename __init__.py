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
import os
import platform
import sys
from importlib import import_module
from pathlib import Path

try:
    from .runtime_dependencies import add_runtime_dependency_path
except ImportError:
    from runtime_dependencies import add_runtime_dependency_path


def setup_platform_specific_functionality() -> None:
    # https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
    # https://github.com/piazzatron/anki-smart-notes/issues/5
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]


def init_addon() -> None:
    addon_dir = Path(__file__).resolve().parent
    add_runtime_dependency_path(addon_dir, sys.path)

    dotenv_module = import_module("dotenv")
    dotenv_module.load_dotenv(dotenv_path=addon_dir / ".env")

    setup_platform_specific_functionality()

    main_module = import_module(".src.main", __name__)
    main_module.main()


if not os.getenv("IS_TEST"):
    init_addon()
