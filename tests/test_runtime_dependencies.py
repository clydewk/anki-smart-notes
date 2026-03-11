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

from typing import TYPE_CHECKING

import pytest

from runtime_dependencies import (
    REQUIRED_RUNTIME_IMPORTS,
    add_runtime_dependency_path,
    resolve_dependency_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


def write_runtime_modules(
    package_dir: Path, *, missing: set[str] | None = None
) -> None:
    missing_imports = missing or set()
    package_dir.mkdir(parents=True, exist_ok=True)

    for module_name in REQUIRED_RUNTIME_IMPORTS:
        if module_name in missing_imports:
            continue

        module_path = package_dir / f"{module_name}.py"
        module_path.write_text("VALUE = 1\n", encoding="utf-8")


def test_resolve_dependency_dir_prefers_dist_vendor(tmp_path: Path) -> None:
    dist_vendor = tmp_path / "dist" / "vendor"
    venv_site_packages = tmp_path / ".venv" / "Lib" / "site-packages"

    write_runtime_modules(dist_vendor)
    write_runtime_modules(venv_site_packages)

    assert resolve_dependency_dir(tmp_path) == dist_vendor.resolve()


def test_resolve_dependency_dir_uses_venv_before_incomplete_vendor(
    tmp_path: Path,
) -> None:
    venv_site_packages = tmp_path / ".venv" / "Lib" / "site-packages"
    stale_vendor = tmp_path / "vendor"

    write_runtime_modules(venv_site_packages)
    write_runtime_modules(stale_vendor, missing={"httpx"})

    assert resolve_dependency_dir(tmp_path) == venv_site_packages.resolve()


def test_resolve_dependency_dir_supports_unix_site_packages(tmp_path: Path) -> None:
    unix_site_packages = tmp_path / ".venv" / "lib" / "python3.13" / "site-packages"
    write_runtime_modules(unix_site_packages)

    assert resolve_dependency_dir(tmp_path) == unix_site_packages.resolve()


def test_add_runtime_dependency_path_raises_clear_error_for_incomplete_dirs(
    tmp_path: Path,
) -> None:
    stale_vendor = tmp_path / "vendor"
    write_runtime_modules(stale_vendor, missing={"httpx", "httpcore"})

    with pytest.raises(ModuleNotFoundError) as exc_info:
        add_runtime_dependency_path(tmp_path, [])

    error_message = str(exc_info.value)
    assert str(stale_vendor.resolve()) in error_message
    assert "missing httpcore, httpx" in error_message
    assert ".venv" in error_message
