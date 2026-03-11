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

from dataclasses import dataclass
from importlib.machinery import PathFinder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

REQUIRED_RUNTIME_IMPORTS = (
    "anyio",
    "certifi",
    "dotenv",
    "h11",
    "httpcore",
    "httpx",
    "idna",
    "sniffio",
    "typing_extensions",
)

VIRTUAL_ENV_DIR_NAMES = (".venv", "venv", "env")


@dataclass(frozen=True)
class DependencyCandidate:
    path: Path
    missing_imports: tuple[str, ...]


def candidate_dependency_dirs(addon_dir: Path) -> tuple[Path, ...]:
    candidates: list[Path] = [addon_dir / "dist" / "vendor"]

    for env_name in VIRTUAL_ENV_DIR_NAMES:
        env_dir = addon_dir / env_name
        candidates.append(env_dir / "Lib" / "site-packages")

        lib_dir = env_dir / "lib"
        if lib_dir.is_dir():
            candidates.extend(sorted(lib_dir.glob("python*/site-packages")))

    candidates.append(addon_dir / "vendor")

    unique_candidates: list[Path] = []
    seen_paths: set[Path] = set()

    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_paths:
            continue
        seen_paths.add(resolved_candidate)
        unique_candidates.append(resolved_candidate)

    return tuple(unique_candidates)


def missing_runtime_imports(package_dir: Path) -> tuple[str, ...]:
    return tuple(
        module_name
        for module_name in REQUIRED_RUNTIME_IMPORTS
        if PathFinder.find_spec(module_name, [str(package_dir)]) is None
    )


def probe_dependency_dirs(addon_dir: Path) -> tuple[DependencyCandidate, ...]:
    candidates = []

    for candidate_dir in candidate_dependency_dirs(addon_dir):
        if not candidate_dir.is_dir():
            continue
        candidates.append(
            DependencyCandidate(
                path=candidate_dir,
                missing_imports=missing_runtime_imports(candidate_dir),
            )
        )

    return tuple(candidates)


def resolve_dependency_dir(addon_dir: Path) -> Path | None:
    for candidate in probe_dependency_dirs(addon_dir):
        if not candidate.missing_imports:
            return candidate.path

    return None


def runtime_dependency_error_message(addon_dir: Path) -> str:
    probes = probe_dependency_dirs(addon_dir)

    if probes:
        details = "; ".join(
            f"{probe.path} missing {', '.join(probe.missing_imports)}"
            for probe in probes
        )
        return (
            "Smart Notes could not locate a complete runtime dependency bundle. "
            f"Found incomplete dependency directories: {details}. "
            "Run scripts/build.sh build for a packaged add-on or install "
            "requirements.txt into a local .venv for source checkouts."
        )

    searched_paths = ", ".join(
        str(path) for path in candidate_dependency_dirs(addon_dir)
    )
    return (
        "Smart Notes could not locate its runtime dependencies. "
        f"Searched: {searched_paths}. "
        "Run scripts/build.sh build for a packaged add-on or install "
        "requirements.txt into a local .venv for source checkouts."
    )


def add_runtime_dependency_path(addon_dir: Path, sys_path: list[str]) -> Path:
    dependency_dir = resolve_dependency_dir(addon_dir)
    if dependency_dir is None:
        raise ModuleNotFoundError(runtime_dependency_error_message(addon_dir))

    dependency_dir_str = str(dependency_dir)
    if dependency_dir_str not in sys_path:
        sys_path.insert(0, dependency_dir_str)

    return dependency_dir
