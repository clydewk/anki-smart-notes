#!/bin/bash
set -eu

# Copyright (C) 2024 Michael Piazza
#
# This file is part of Smart Notes.
#
# Smart Notes is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Smart Notes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Smart Notes.  If not, see <https://www.gnu.org/licenses/>.

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$PROJECT_DIR"

resolve_python () {
  for executable in .venv/Scripts/python.exe .venv/bin/python; do
    if [ -x "$executable" ]; then
      echo "$executable"
      return
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo python
  fi
}

PYTHON_CMD=$(resolve_python)

clean () {
  "$PYTHON_CMD" - <<'PYTHON'
from pathlib import Path
import shutil

stage = Path("dist")
if stage.is_symlink():
    raise SystemExit("Refusing to clean a symlinked staging directory: dist")
if stage.exists():
    shutil.rmtree(stage)
PYTHON
}

build () {
  clean
  "$PYTHON_CMD" - <<'PYTHON'
import json
import re
import shutil
import subprocess
import sys
from importlib.metadata import distribution, packages_distributions
from importlib.util import find_spec
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from runtime_dependencies import REQUIRED_RUNTIME_IMPORTS

if sys.version_info < (3, 13):
    raise SystemExit("Building requires Python 3.13 or newer.")

stage = Path("dist")
vendor = stage / "vendor"
vendor.mkdir(parents=True)

# Only code and declared assets belong in the package, never runtime state.
assets = (
    "__init__.py", "runtime_dependencies.py", "manifest.json", "license",
    "changelog.md", "eleven_voices.json", "google_voices.json", "azure_voices.json",
)
for source in [*(Path(name) for name in assets), *Path("src").rglob("*.py")]:
    if source.is_symlink():
        raise SystemExit(f"Refusing to package a symlink: {source}")
    destination = stage / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

# Anki may modify config.json in a source-linked install. Read committed defaults.
defaults = json.loads(subprocess.check_output(["git", "show", "HEAD:config.json"]))
for key in defaults:
    if key.endswith("_api_key") or key in {"auth_token", "uuid", "openai_endpoint"}:
        defaults[key] = None
for key in ("mcp_servers", "custom_providers"):
    defaults[key] = []
defaults["provider_settings"] = {}
defaults["prompts_map"] = {"note_types": {}}
(stage / "config.json").write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")

environment = stage / "src" / "env.py"
environment.write_text(
    re.sub(r'environment(?:: str)? = "(?:DEV|PROD)"', 'environment: str = "PROD"', environment.read_text(encoding="utf-8")),
    encoding="utf-8",
)

package_names = packages_distributions()
for name in REQUIRED_RUNTIME_IMPORTS:
    spec = find_spec(name)
    if spec is None:
        raise SystemExit(f"Missing runtime dependency: {name}. Install requirements.txt first.")
    source = Path(next(iter(spec.submodule_search_locations))) if spec.submodule_search_locations else Path(spec.origin or "")
    if source.is_dir():
        shutil.copytree(source, vendor / source.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "_tests", "pytest_plugin.py"))
    else:
        shutil.copy2(source, vendor / source.name)
    # Keep package metadata and license notices with the vendored modules.
    for package in package_names.get(name, []):
        metadata = distribution(package)
        for entry in metadata.files or []:
            if ".dist-info/" in str(entry) and (entry.name in {"METADATA", "LICENSE", "COPYING"} or "/licenses/" in str(entry)):
                destination = vendor / entry
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata.locate_file(entry), destination)

archive = stage / "smart-notes.ankiaddon"
with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=9) as output:
    for source in sorted(stage.rglob("*")):
        if source.is_file() and source != archive:
            output.write(source, source.relative_to(stage))
print(f"Built {archive.resolve()}")
PYTHON
}

case "${1:-}" in
  build) build ;;
  clean) clean ;;
  format) "$PYTHON_CMD" -m ruff format . ;;
  lint) "$PYTHON_CMD" -m ruff check . ;;
  typecheck) "$PYTHON_CMD" -m pyright . ;;
  check)
    "$PYTHON_CMD" -m ruff format . --check
    "$PYTHON_CMD" -m ruff check .
    "$PYTHON_CMD" -m pyright .
    ;;
  fix)
    "$PYTHON_CMD" -m ruff check . --fix
    "$PYTHON_CMD" -m ruff format .
    ;;
  *)
    echo "Usage: $0 {build|clean|format|lint|typecheck|check|fix}" >&2
    exit 1
    ;;
esac
