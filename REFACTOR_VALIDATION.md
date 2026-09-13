<!--
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
-->

# Reliability refactor validation

The fork is retained. The implementation introduces one profile-owned asyncio worker and one effective-settings resolver, with repairs at existing provider, collection, MCP, and Qt boundaries. No provider framework, database, web UI, event bus, billing system, or prefetch feature was added.

## Review coverage

| Review groups | Implemented behavior | Regression coverage |
| --- | --- | --- |
| F01, F02, F07, F08 | Job origin and editor ownership; cancellable Qt collection bridges; bounded shutdown; conflict/schema checks before media; missing-note isolation; memory-only unsaved notes; undoable commits. | `test_tasks.py`, `test_collection_ops.py`, `test_processor.py`, `test_anki_collection.py` |
| F06, F18 | TopologicalSorter validates the complete graph before target closure; failures and blocked fields propagate; stale targets warn; batch overwrite is isolated by generation mode. | `test_processor.py` |
| F12, F14, F15 | Detached effective settings; nested draft Save/Cancel; restartable per-profile migration with backup; atomic save rollback; retained custom/unavailable IDs; preview/signature parity. | `test_config.py`, `test_settings.py`, `test_migrations.py`, `test_ui_widgets.py` |
| F04, F05, F11, F13, F19 | Cancellation-safe reservations; reported usage retained on failures and retries; conservative unknown OpenAI usage distinguished from measured usage; terminal-state validation; no replay of delivered streams/tool calls; capability-aware controls; existing GPT Image corrections retained; search/media bounds. | `test_models.py`, `test_chat_usage.py`, `test_provider_runtime.py`, `test_tools.py`, `test_media_utils.py` |
| F09, F10 | Newline stdio, bounded stderr, streaming HTTP/SSE, request IDs, notifications/errors, tool pagination, restricted environment, config invalidation, isolated draft probes, bounded shutdown. | `test_mcp.py`, independent MCP SDK stdio/HTTP peers, `test_tasks.py` |
| F03, F16, F17, F20 | Redaction and masked secrets; Google auth headers; safe Markdown/plain text; selection/undo/zero-value fixes; scrollable native UI; native image previews; Anki 26.5/Python 3.13 checks; staging-only packaging without runtime state. | `test_content_safety.py`, `test_ui_widgets.py`, `test_runtime_dependencies.py` |

## Verification

Run from the repository with its Python 3.13 `.venv`:

```sh
./scripts/build.sh check
.venv/bin/python -m pytest
./scripts/build.sh build
```

Strict typechecking uses the actual Anki/aqt 26.5 environment. Tests include native Qt subprocesses and real disposable Anki collections, including copied collections with matching note IDs. The ordinary suite skips the two optional independent MCP SDK tests; run them with a separate SDK interpreter:

```sh
MCP_TEST_PYTHON=/tmp/anki-smart-notes-mcp-sdk/bin/python .venv/bin/python -m pytest tests/test_mcp.py tests/test_tasks.py tests/test_anki_collection.py -q
```

Verified on 2026-09-12: format/lint/strict typechecking passed with zero errors or warnings; the full suite passed **297 tests**, with **2 optional SDK tests skipped**. The separate SDK-enabled MCP, lifecycle, and real-Anki run passed **30 tests**, including both independent SDK peers.

The package is `dist/smart-notes.ankiaddon`, BYOK version 3.0.0, preserving add-on ID 1531888719 for in-place migration. Packaging uses committed default configuration and excludes profile settings, credentials/runtime configuration, logs, backups, ledgers, tests, and the MCP SDK.

Packaged macOS verification passed using `/Applications/Anki.app` 26.05 (bundle 26.5) on macOS 26.6.2 arm64, with two disposable profiles and no provider keys. Native Qt automation verified:

- Save/Cancel and reachable controls/footer at 800×650 logical pixels.
- Independent A/B profile settings and unchanged original note fields.
- Cancellation during A → B → A, discarded late callbacks, a new request after reopening, and shutdown with an active job.
- The initial refactor package passed the same harness after its final rebuild. Profile A close dispatch completed in 0.139 seconds; the final Anki process exited with status 0. The subsequent AVIF correction was verified in the live source-linked installation below and covered by native regression tests.

Local evidence includes the native smoke report and an 800×650 settings screenshot, retained outside the repository. Accessibility automation timed out during an earlier attempt; native Qt automation subsequently verified the application behavior above.

Actual Windows packaged-install testing and a controlled AnkiWeb sync round trip remain release gates. Concurrent backend updates are covered locally; that does not certify a sync-account round trip. Paid-provider smoke tests were subsequently authorized and run as described below.

## Live main-profile provider tests

On 2026-09-12, the user authorized testing the actual installed add-on with real providers in the main Anki profile. The installation is linked to this checkout. A private collection/configuration backup was taken before launch. Tests used synthetic content in a temporary Basic note, the installed profile worker, the production field processor, and guarded collection operations. Existing prompts and provider selections were retained; test settings were captured as temporary overrides.

| Provider and modality | Observed result |
| --- | --- |
| OpenAI text, `gpt-5.6-sol` | Stream completed successfully; generated field saved and undo restored the original. Reported usage: 19 input + 10 output tokens. |
| Fish Audio, `s2.1-pro-free` | One request produced a valid 1.698-second MP3, 44.1 kHz mono. Save, media reference, file bytes, and undo passed using the same returned audio. |
| OpenAI image, `gpt-image-1.5` | The first response exposed the AVIF encoder defect below. After correction and Anki restart, a second bounded request passed AVIF conversion, 1024×1024 native preview, media write, and undo. Reported usage: 418 and 430 total tokens. |
| Google text | The saved `gemini-3-pro-preview` selection was correctly rejected as retired before dispatch. Read-only discovery advertised `gemini-3.1-pro-preview`; one temporary probe returned HTTP 429 quota exhaustion. The saved selection was not changed. |
| Custom provider A | Generation and separate read-only model discovery both hit the 10-second connection timeout. No generation retry was made. |
| Custom provider B | Generation returned HTTP 503: no available channel for the saved model. Model discovery returned an empty list. No generation retry was made. |

The AVIF failure came from forcing `libaom-av1` although this machine's FFmpeg provides `libsvtav1`. Encoding now lets FFmpeg select its installed encoder. Native previews decode unsupported formats to PNG in the worker while preserving original AVIF bytes for Save. Conversion failures now propagate as errors instead of appearing as blocked fields. Three additional regressions cover the actual encoder, both native preview paths, original-format preservation, and conversion errors.

One harness assertion initially overlooked Anki's filename normalization. The assertion was corrected to inspect the actual media reference and verify bytes; this required no additional speech request. Its leftover temporary audio was hash-verified and moved to Anki's media trash.

All probes disabled automatic retries and tools, capped text output at 1,024 tokens, and used 90-second text/speech or 180-second image deadlines. Image probes requested one low-quality 1024×1024 image. Seven generation attempts were made in total, including failed endpoints; the image verification after the fix was a separate probe. The providers reported 877 OpenAI tokens across successful responses; speech billing and unsuccessful provider usage were not reported, so these figures are not an exact cost statement. Actual usage ledger entries were preserved.

Final checks confirmed that all existing notes were unchanged, saved settings and profile prompts were unchanged, temporary notes/decks/media were removed, and no background jobs or token reservations remained. The one-use instrumentation and diagnostic windows were removed. Anki was left open on the main profile. Automatic sync was observed during the restart; no controlled cross-device sync test was performed.

Private local evidence includes the live smoke, follow-up, discovery, and final-state reports, retained outside the repository. These reports contain no credentials or study-note contents. The private backup is also excluded from publication and packaging.

## Live settings regression

Opening settings with a configured OpenAI key exposed an initialization error missed by the earlier disposable-profile checks: model discovery submitted a job before its containing dialog was visible. Discovery now starts when the Text page is shown, captures its settings draft, and remains valid when another tab is selected. Closing the containing dialog still cancels its jobs and clears the loading state. Discovered models and later model-list toggles use the draft's credentials.

A native Qt regression covers construction with a configured key, unsaved key changes, hidden-page completion, model-list toggles, cancellation, reopening, and rejected submissions. After restarting the actual Anki application, Tools → Smart Notes opened successfully in the main profile; OpenAI discovery returned 21 models; tab switching, model selection, Cancel, and reopening passed. The original model and options reappeared after Cancel, saved installation/profile settings matched their before-test hashes, and no background jobs remained after the first close. No generation requests were made during this check. Anki was left open on the main profile.

The live settings report is retained privately outside the repository.

## Provider contracts checked

Existing settings are preserved; catalog updates do not migrate saved model selections. Relevant official references include [OpenAI capabilities](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra), [OpenAI speech](https://developers.openai.com/api/docs/guides/text-to-speech), [DeepSeek completions](https://api-docs.deepseek.com/api/create-chat-completion/), [Google model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), [Gemini thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures), [Gemini speech](https://ai.google.dev/gemini-api/docs/speech-generation), and [Anthropic deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations). Azure selections are retained but disabled because this fork has no supported direct Azure integration.

## Configuration UI revamp

The follow-up configuration audit was implemented on 2026-09-12 using the existing native Qt widgets and settings draft. No UI framework or runtime dependency was added.

- Smart Fields: readable type/behavior columns, full-value tooltips, sensible field widths, stable table items, search counts and empty states, a single Add menu, and repairable stale targets.
- Providers: grouped credentials/request options/token guard/custom providers, aligned masked keys and contextual reveal controls, honest reported/estimated usage context and UTC reset, and selection-aware actions.
- Tools: named built-in checkboxes, explicit availability versus field permission, native MCP server selector with connection details, and isolated cancellable probes with inline feedback.
- Text/Images: expanding forms, capability explanations, discovery loading/cache/empty/error/retry states, preserved unavailable choices, correct effective image dimensions and distinct rendering/compression quality.
- Speech: compact voice pool, distinguishable custom Fish voice IDs, wide searchable catalog, explicit filtered selection, language editing, inline cancellable previews and temporary playback files outside collection media.
- Generation: direct descriptions of review/batch/blank-source behavior and a separate troubleshooting group.
- Nested editors: draft-scoped Done/Cancel, inherited summaries plus Customize, modality-specific copy, sample/provider disclosure before generation, scrollable results, correct MCP executable/directory guidance, optional environment/header groups, inline validation, and immediate removal of obsolete row widgets.
- Shared layout: native font sizes, growing/wrapping forms, preserved QLabel height-for-width sizing, bounded/restored window geometry, and an always-reachable footer.

Regression validation: **309 passed, 2 optional SDK tests skipped**; format/lint/strict typechecking passed with zero errors or warnings. Native subprocess tests cover every main page at **800×650 and 640×470**, using **13pt and 18pt** fonts, label and checkbox sizing, horizontal overflow, footer placement including save errors, actual Add/Edit Qt signals, stable selection/filter identities, nested drafts/repair, model capability parity, probe failure/cancellation/retry and preview-media ownership.

Live verification used the source-linked installation in the main profile, Anki 26.05 / Python 3.13.11 / Qt 6.11.0 on macOS arm64 in dark mode. All seven main pages, all three field editor types and their inheritance pages, preview preflight, all three custom-provider model tabs, and MCP HTTP/stdio/environment forms were inspected. Text was also inspected at a 640×470 client size. The live pass found and corrected an Add-action signal argument mismatch and overlapping deferred-deletion MCP argument rows; both have focused regression coverage. OpenAI read-only model discovery returned 21 models. No generation or connection-test requests were made. Installation configuration and profile settings matched their pre-inspection hashes after the drafts were closed.

The Tools page initially reproduced the native Qt/macOS accessibility crash with a selected server table. Removing implicit selection prevented its opening trigger, but the upstream ownership defect can affect later selected-table scans. The final Tools page therefore uses a QComboBox with plain selectable connection details; it no longer uses the failing table path. Live accessibility enumeration of the final page and its server popup passed. Server selection, reordering, removal, and probe-state behavior are covered by native Qt regressions. This is a local avoidance of the affected widget path, not a patch to Anki's Qt runtime. Background: [Anki development report](https://forums.ankiweb.net/t/macos-accessibility-scan-can-crash-browse-when-a-row-is-selected/70882), [Qt Cocoa ownership change](https://github.com/qt/qtbase/commit/b1ed5f656f064e553b33752f8e87d2f5b9553e38).

The add-on package was rebuilt. Windows packaged UI testing, a complete screen-reader pass, and a controlled sync round trip remain release checks; the offscreen layout tests do not certify those environments.
