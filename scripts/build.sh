#!/bin/bash
set -e

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

resolve_python () {
  if [ -x ".venv/Scripts/python.exe" ]; then
    echo ".venv/Scripts/python.exe"
    return
  fi

  if [ -x ".venv/bin/python" ]; then
    echo ".venv/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi

  echo "Python executable not found" >&2
  exit 1
}

PYTHON_CMD=$(resolve_python)

vendor_deps () {
  "$PYTHON_CMD" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import shutil

from runtime_dependencies import REQUIRED_RUNTIME_IMPORTS

target = Path("dist/vendor")
required = list(REQUIRED_RUNTIME_IMPORTS)

def copy_import(name: str) -> None:
    spec = find_spec(name)
    if spec is None:
        raise SystemExit(f"Missing runtime dependency for vendoring: {name}")

    if spec.submodule_search_locations:
        source = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        source = Path(spec.origin)
    else:
        raise SystemExit(f"Could not resolve vendored dependency: {name}")

    destination = target / source.name
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)

for dependency in required:
    copy_import(dependency)
PY
}

strip_dist_noise () {
  find dist -name "__pycache__" -type d -prune -exec rm -rf {} +
  find dist/vendor \( -name "_tests" -o -name "tests" \) -type d -prune -exec rm -rf {} +
  find dist -name "*.pyc" -delete
  find dist/vendor -name "pytest_plugin.py" -delete
}

build () {
  echo "Building..."
  rm -rf dist
  mkdir -p dist/vendor

  cp *.py dist/
  cp manifest.json dist/
  cp config.json dist/
  cp -r src dist/
  cp license dist/
  cp changelog.md dist/
  echo "environment = \"PROD\"" > dist/src/env.py

  vendor_deps
  strip_dist_noise

  # Voices
  cp -r eleven_voices.json dist/
  cp -r google_voices.json dist/
  cp -r azure_voices.json dist/

  # Zip it
  cd dist
  zip -9 -r smart-notes.ankiaddon .
  cd ..
}

clean () {
  echo "Cleaning..."
  rm -rf dist
  rm -rf ~/Library/Application\ Support/Anki2/addons21/smart-notes
  rm -rf ~/development/win_shared/smart-notes
}

link-dev () {
  # Link for Mac Local Dev
  ln -s $(pwd) ~/Library/Application\ Support/Anki2/addons21/smart-notes
}

win-dist () {
  clean
  build
  rm -rf ~/development/win_shared/smart-notes
  mkdir ~/development/win_shared/smart-notes
  # Link for Windows dev thru shared folder
  cp -r $(pwd)/dist/* ~/development/win_shared/smart-notes
}

# Tests a production build by symlinking dist folder
link-dist () {
  ln -s $(pwd)/dist ~/Library/Application\ Support/Anki2/addons21/smart-notes
}

anki () {
   /Applications/Anki.app/Contents/MacOS/launcher
}

test-dev () {
  clean
  link-dev
  anki
}

test-build () {
  clean
  build
  rm -rf dist/meta.json
  link-dist
  # cp meta.json dist/
  # copy the current meta to make testing easier
  # jq '.config.auth_token = null' dist/meta.json > dist/temp.json && mv dist/temp.json dist/meta.json
  anki
}

sentry-release () {
  # Write some jq
  version=$(jq '.human_version' manifest.json)
  echo $version
  # sentry-cli releases --org michael-piazza new --finalize ${version}
}

format () {
  echo "Formatting code..."
  "$PYTHON_CMD" -m ruff format .
}

lint () {
  echo "Linting code..."
  "$PYTHON_CMD" -m ruff check .
}

typecheck () {
  echo "Type checking..."
  "$PYTHON_CMD" -m pyright .
}

check () {
  echo "Running all checks..."
  "$PYTHON_CMD" -m ruff format . --check && "$PYTHON_CMD" -m ruff check . && "$PYTHON_CMD" -m pyright .
}

fix () {
  echo "Fixing code issues..."
  "$PYTHON_CMD" -m ruff format . && "$PYTHON_CMD" -m ruff check . --fix
}

version () {
  if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    exit 1
  fi

  VERSION=$1

  jq --arg version "$VERSION" '.human_version = $version' manifest.json > manifest.tmp && mv manifest.tmp manifest.json

  # Commit the changes
  git add manifest.json
  git add changelog.md
  git commit -m "v$VERSION"

  # Create a tag
  git tag "v$VERSION"

  # Push the commit and the tag
  git push
  git push origin "v$VERSION"
}


if [ "$1" == "build" ]; then
  clean
  build
elif [ "$1" == "clean" ]; then
  clean
elif [ "$1" == "win" ]; then
  win-dist
elif [ "$1" == "test-dev" ]; then
  test-dev
elif [ "$1" == "test-build" ]; then
  test-build
elif [ "$1" == "sentry-release" ]; then
  sentry-release
elif [ "$1" == "version" ]; then
  version $2
elif [ "$1" == "format" ]; then
  format
elif [ "$1" == "lint" ]; then
  lint
elif [ "$1" == "typecheck" ]; then
  typecheck
elif [ "$1" == "check" ]; then
  check
elif [ "$1" == "fix" ]; then
  fix
else
  echo "Invalid argument: $1"
  echo "Available commands: build, clean, win, test-dev, test-build, sentry-release, version, format, lint, typecheck, check, fix"
  exit 1
fi

# Check if the last command succeeded
if [ $? -eq 0 ]; then
  echo "Done"
fi
