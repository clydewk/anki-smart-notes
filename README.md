# Anki Smart Notes (BYOK)

Generate text, speech, and images in Anki fields using your own provider API keys. This fork retains the native Qt interface, dependent smart fields, prompt import/export, voice routing, built-in Anki tools, and MCP servers. It does not require a Smart Notes subscription; providers charge your own account for API use.

Supports **Anki 26.5 or newer on Windows and macOS**. Development and tests use Python 3.13.

## Install or update

Build this repository and install `dist/smart-notes.ankiaddon` through **Tools → Add-ons → Install from file**, then restart Anki. The package keeps the existing add-on ID so an existing installation can migrate in place. The BYOK fork and upstream share this ID: install only one, and do not replace the fork with upstream when updating.

Back up your Anki profile before upgrading. On first opening version 3.0, every existing profile receives an independent copy of the legacy prompts and generation behavior. New profiles start with no prompts. Credentials, provider defaults, MCP settings, and the estimated token guard remain shared across the installation. Profile settings live in `smart-notes.json` inside each profile folder; they are local settings and are not synced by Anki. Existing fields are not rewritten by migration.

## Generate fields

1. Open **Tools → Smart Notes**, enter your provider API keys, and choose models.
2. Associate a prompt with a note type, deck, and target field. Reference source fields using `{{Field Name}}`; dependent smart fields run in dependency order.
3. Save the settings. Generate from the editor's sparkle button or field context menu, select notes in the browser for batch generation, or enable generation during review.

Prompt previews use the current settings draft. **Save** commits settings; **Cancel** discards the draft, including edits made in nested dialogs. Unsaved notes receive results only in their originating editor. Cancelling a job discards uncommitted results; completed batch chunks remain undoable.

Enable only the built-in tools and MCP servers you intend providers to call. MCP supports stdio and Streamable HTTP for protocol version `2025-03-26`; stdio servers inherit essential OS variables plus explicitly selected environment variables and overrides. Existing Azure configurations are retained, but Azure choices are unavailable until a supported direct integration exists.

The OpenAI token setting is an **estimated token guard**, not an exact billing ceiling. Actual usage may remain unknown when a dispatched request fails or is cancelled. Check provider billing for charges.

## Development

Create a Python 3.13 virtual environment at `.venv`, activate it, and install the development requirements:

```sh
python -m pip install -r requirements.txt
./scripts/build.sh check
python -m pytest
./scripts/build.sh build
```

The script detects `.venv/bin/python` on macOS and `.venv/Scripts/python.exe` on Windows. Use Git Bash to run the shell script on Windows. `fix` applies formatting and lint fixes; `clean` deletes only the repository's `dist` staging directory and refuses a symlinked staging directory.

Builds bundle the selected pure Python runtime dependencies and their license notices. They package committed `config.json` defaults, never a source checkout's modified runtime configuration. Commit intentional default changes before a release build. Backups, usage ledgers, logs, environment files, tests, and installed Anki folders are excluded. Building or cleaning never installs the add-on or changes an Anki profile.

For independent MCP integration tests, install `mcp==1.12.4` in a separate Python environment and set `MCP_TEST_PYTHON` to its interpreter before running `python -m pytest tests/test_mcp.py`. These tests start local subprocess/HTTP servers. The SDK is a test dependency only and is never bundled. CI runs checks, unit tests, SDK integration tests, and packaging on Windows and macOS.

Before release, verify the built add-on in disposable Windows and macOS profiles at an 800×650 logical viewport: profile switching, active shutdown, unsaved editors, undo, sync, settings migration, and media updates. Paid-provider smoke tests require separate authorization.

## Project

See [changelog.md](changelog.md) for version history and [the fork's issue tracker](https://github.com/clydewk/anki-smart-notes/issues) for reports. Original project: [piazzatron/anki-smart-notes](https://github.com/piazzatron/anki-smart-notes). Licensed under GPLv3 or later; see [license](license).
