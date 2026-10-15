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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from runtime_dependencies import (
    REQUIRED_RUNTIME_IMPORTS,
    add_runtime_dependency_path,
    resolve_dependency_dir,
)


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


def copy_build_project(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(source / "scripts/build.sh", project / "scripts/build.sh")
    for name in (
        "runtime_dependencies.py",
        "__init__.py",
        "manifest.json",
        "license",
        "changelog.md",
        "eleven_voices.json",
        "google_voices.json",
        "azure_voices.json",
    ):
        shutil.copy2(source / name, project / name)
    (project / "src").mkdir()
    (project / "src/env.py").write_text('environment: str = "DEV"\n', encoding="utf-8")
    return project


def run_build(project: Path, command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(project / "scripts/build.sh"), command],
        cwd=project.parent,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Creating symlinks needs Windows developer privileges",
)
def test_clean_refuses_staging_symlinks(tmp_path: Path) -> None:
    project = copy_build_project(tmp_path)
    installed = tmp_path / "installed-addon"
    installed.mkdir()
    sentinel = installed / "keep.txt"
    sentinel.write_text("installed settings", encoding="utf-8")
    (project / "dist").symlink_to(installed, target_is_directory=True)
    result = run_build(project, "clean")
    assert result.returncode != 0
    assert "Refusing to clean" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "installed settings"


def test_build_excludes_runtime_state_and_bundles_rendering_dependencies(
    tmp_path: Path,
) -> None:
    project = copy_build_project(tmp_path)
    defaults = {
        "openai_api_key": "",
        "chat_model": "example-model",
        "prompts_map": {"note_types": {}},
    }
    (project / "config.json").write_text(json.dumps(defaults), encoding="utf-8")
    for args in (
        ["init", "--quiet"],
        ["add", "config.json"],
        [
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@example.com",
            "commit",
            "--quiet",
            "-m",
            "defaults",
        ],
    ):
        subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)
    (project / "config.json").write_text(
        '{"openai_api_key": "private-key", "prompts_map": {"secret": "prompt"}}',
        encoding="utf-8",
    )
    for name in (
        "meta.json",
        "config_backup.json",
        "chat_usage.json",
        "debug.log",
        ".env",
        "src/chat_usage.json",
        "src/debug.log",
        "chatgpt.json",
        "chatgpt-host.json",
        "chatgpt-auth.json",
    ):
        (project / name).write_text("private-state", encoding="utf-8")
    result = run_build(project, "build")
    assert result.returncode == 0, result.stderr
    with ZipFile(project / "dist/smart-notes.ankiaddon") as archive:
        names = archive.namelist()
        assert "vendor/markdown_it/__init__.py" in names
        assert "vendor/mdurl/__init__.py" in names
        assert "vendor/websockets/asyncio/client.py" in names
        assert "vendor/jose/jwt.py" in names
        assert "vendor/rsa/pkcs1.py" in names
        assert not any(
            "private-state" in archive.read(name).decode("utf-8", errors="ignore")
            for name in names
        )
        assert not any(
            name.endswith((".log", ".pyc", ".so", ".pyd", ".ankiaddon"))
            or "__pycache__" in name
            for name in names
        )
        packaged_defaults = json.loads(archive.read("config.json"))
        assert not packaged_defaults["openai_api_key"]
        assert packaged_defaults["prompts_map"] == {"note_types": {}}
        assert packaged_defaults["chat_model"] == "example-model"
        assert '"PROD"' in archive.read("src/env.py").decode()
    isolated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[1])
import httpx
import rsa
from jose import jwt
from jose.exceptions import JWTError
from websockets.asyncio.client import connect
from websockets.frames import apply_mask
from markdown_it import MarkdownIt
assert apply_mask(b'hello', b'1234') == b'YW_X^'
assert '<strong>safe</strong>' in MarkdownIt('commonmark', {'html': False}).render('**safe**')
public, private = rsa.newkeys(2048)
token = jwt.encode({'sub': 'test', 'iss': 'issuer', 'aud': 'client', 'exp': 4102444800}, private.save_pkcs1(), algorithm='RS256')
assert jwt.decode(token, public.save_pkcs1(), algorithms=['RS256'], issuer='issuer', audience='client')['sub'] == 'test'
header, body, signature = token.split('.')
altered = f"{header}.{body}.{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"
try:
    jwt.decode(altered, public.save_pkcs1(), algorithms=['RS256'], issuer='issuer', audience='client')
except JWTError:
    pass
else:
    raise AssertionError('Invalid signature accepted')
""",
            str(project / "dist/vendor"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert isolated.returncode == 0, isolated.stderr
