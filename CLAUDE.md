# About

- This repo is an extension for the Anki app called Anki Smart Notes. 
- Anki Smart Notes adds functionality to Anki to allow fields to be AI generated. This works for both text, audio, and images.


# Code Style and Practices
- Follow existing practices and conventions in the codebase.
- Use modern Python (3.9+). Make sure your code is fully typed.
- **Async/Sync**: Anki uses Qt (synchronous UI). The plugin uses `asyncio` for network requests. 
    - Use `src/utils.py:run_on_main` for UI updates from background threads.
    - Use `src/sentry.py:run_async_in_background_with_sentry` for launching async background tasks.
- Always run format, lint, typecheck, and tests as the last thing you do. This is super important.
- Always put imports at the top of files, never inside functions or methods 
- Don't prefix top level defs with an `_` ever. Save those for private methods on classes.

# Important Commands
- Use `./scripts/build.sh check` to run all code quality checks (format, lint, typecheck)
- Use `./scripts/build.sh typecheck` to run only type checking
- Use `./scripts/build.sh fix` to automatically fix formatting and linting issues
- Use `python -m pytest` to run tests.

# Testing
- Always check for existing test patterns and follow them consistently.
- Always run tests after writing code and before submitting PRs to ensure you didn't break anything. 
- Always reuse existing test fixtures and mocks instead of creating new ones. 

# PRs
- Prefix PRs with one of either `feat:` (new feature), `fix:` (fix a bug), `refactor:` (refactor an existing feature), or `chore:` (fixing lint, upgrading deps, tests, etc). Keep titles concise. For example: `feat: implement new MegaSqueek TTS model`.
- When you are told to fix PR comments, you should mark them as resolved when they are done. 
- You should be very careful that you actually look at every PR comment. Don't let them slip through.
- We use `Linear` for issue tracking: if you see text like [ANK-123] somewhere, like in your prompt or a Github issue title , the PR should include that text `[ANK-123]` at the end of the title. Example: `fix: double sign out bug [ANK-1337]`
- Branches should contain the Linear tag if you are provided one as well. Names should be concise, for example: `fix-sign-out-bug-ank-1337`

# Permissions
- If you're in accept edits mode, don't ask for permission to git commit or git push. Just do it.

# Code Structure
- This is a Python3 codebase which takes full advantage of strong typing for static analysis.
- UI specific code lives in `src/ui`
- The code entry point is `main.py` → `hooks.py`, which sets up the Anki UI and initializes key classes.

## Key Architecture Components

### Entry Point & Flow
- **`main.py`**: Entry point that creates `NoteProcessor` and calls `setup_hooks()` 
- **`hooks.py`**: Sets up all Anki UI hooks, menu items, buttons, and context menus. Initializes logging, config, and migrations. Contains the main UI integration logic.

### Core Processing Classes
- **`NoteProcessor`** (`note_proccessor.py`): Main orchestrator for processing notes/cards. 
    - Handles batch processing with granular, cancellable progress bars.
    - Manages the DAG execution flow for field dependencies.
    - Handles retry logic and error reporting.
- **`FieldProcessor`** (`field_processor.py`): Handles individual field processing based on type (chat, TTS, image). Routes to appropriate providers and handles response formatting.
- **`AppStateManager`** (`app_state.py`): Manages subscription state, capacity limits, and app unlock status. Handles transitions between subscription states.

### Field Processing System  
- **`dag.py`**: Creates Directed Acyclic Graph for field dependencies. Handles topological sorting and cycle detection for smart fields that reference other fields.
- **`nodes.py`**: `FieldNode` class representing individual fields in the DAG with dependencies, processing state, and metadata.
- **`prompts.py`**: Prompt interpolation, field extraction from prompts, and prompt management per note type/deck.

### Data Models & Configuration
- **`models.py`**: Type definitions for:
  - Chat/TTS/Image providers and models 
  - `SmartFieldType` ("chat", "tts", "image")
  - `FieldExtras` configuration per field
  - Provider/model mappings
- **`config.py`**: Global configuration management with persistence.
- **`constants.py`**: Error messages, limits, and static values (e.g. timeouts).
- **`env.py`**: Environment detection (PROD vs DEV) for API endpoints.

### Provider Classes
- **`chat_provider.py`**: Handles OpenAI, Anthropic, DeepSeek chat completions
- **`tts_provider.py`**: Text-to-speech via OpenAI, ElevenLabs, Google (BYOK only)
- **`image_provider.py`**: Image generation via Replicate (Flux models) and Google Gemini Nano Banana Pro
- **`open_ai_client.py`**: Legacy OpenAI client for users with API keys

### UI System (`src/ui/`)
- **Reactive Components**: Base classes (`ReactiveWidget`, `ReactiveState`) for UI widgets that update automatically when underlying state changes.
- **`addon_options_dialog.py`**: Main settings dialog, heavily uses reactive components.
- **`field_menu.py`**: Context menu additions for individual fields in the editor.
- **`custom_prompt.py`**: "Prompt Palette" for testing/running prompts on a single note.

### Utilities
- **`utils.py`**: General utilities, async helpers (`run_on_main`), file loading.
- **`sentry.py`**: Error reporting and background task wrapping (`run_async_in_background_with_sentry`).
- **`notes.py`**: Note type utilities, field validation.
- **`decks.py`**: Deck-related utilities and caching.

# Creating New Files
- All new files must have the following license information at the very top.

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