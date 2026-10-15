# Anki Smart Notes

Generate text, speech, and images in Anki fields using your own provider API keys, or sign in with ChatGPT for text generation through your subscription. This fork retains the native Qt interface, dependent smart fields, prompt import/export, voice routing, built-in Anki tools, and MCP servers. It does not require a Smart Notes subscription.

Supports **Anki 26.5 or newer on Windows and macOS**. Development and tests use Python 3.13.

## Install or update

Build this repository and install `dist/smart-notes.ankiaddon` through **Tools → Add-ons → Install from file**, then restart Anki. The package keeps the existing add-on ID so an existing installation can migrate in place. The BYOK fork and upstream share this ID: install only one, and do not replace the fork with upstream when updating.

Back up your Anki profile before upgrading. On first opening version 3.0, every existing profile receives an independent copy of the legacy prompts and generation behavior. New profiles start with no prompts. Credentials, provider defaults, MCP settings, and the estimated token guard remain shared across the installation. Profile settings live in `smart-notes.json` inside each profile folder; they are local settings and are not synced by Anki. Existing fields are not rewritten by migration.

## Generate fields

1. Open **Tools → Smart Notes → Providers**, enter your provider API keys or sign in with ChatGPT, and choose a model on the Text page.
2. Associate a prompt with a note type, deck, and target field. Reference source fields using `{{Field Name}}`; dependent smart fields run in dependency order.
3. Save the settings. Generate from the editor's sparkle button or field context menu, select notes in the browser for batch generation, or enable generation during review.

Prompt previews use the current settings draft. **Save** commits settings; **Cancel** discards the draft, including edits made in nested dialogs. Unsaved notes receive results only in their originating editor. Cancelling a job discards uncommitted results; completed batch chunks remain undoable.

Enable only the built-in tools and MCP servers you intend providers to call. MCP supports stdio and Streamable HTTP for protocol version `2025-03-26`; stdio servers inherit essential OS variables plus explicitly selected environment variables and overrides. Existing Azure configurations are retained, but Azure choices are unavailable until a supported direct integration exists.

The OpenAI token setting is an **estimated token guard**, not an exact billing ceiling. Actual usage may remain unknown when a dispatched request fails or is cancelled. Check provider billing for charges.

## OpenAI models and processing

The Text page includes GPT-6 Astra, GPT-6.1 Sol, GPT-6 Sol, and GPT-6 Luna alongside supported older models. Reasoning, temperature, and tool controls follow each model's capabilities. Existing selections are preserved. Models with announced shutdown dates display warnings; retired selections remain visible for correction and cannot generate.

Choose processing speed on **Providers → OpenAI text requests**. **Project default** follows the API project's setting, **Standard** explicitly requests standard processing, and **Fast** and **Ultrafast** request premium processing. Ultrafast supports Astra with global/US processing and 6.1 Sol with global/US/EU processing. Unsupported combinations fail before generation rather than switching tiers. These controls apply to API-key text requests; the native ChatGPT connection retains its account-provided models and options. See [OpenAI's speed documentation](https://developers.openai.com/api/docs/guides/ultrafast-mode) for availability and pricing.

GPT Image 2.5 Sunburst and Flare support all six rendering qualities, including `xhigh` and `max`. Image size choices show the requested dimensions and aspect ratio; large sizes are experimental. GPT Image requests allow three minutes for a response. A network failure after possible submission is not automatically replayed, since the original generation may have succeeded.

## Speech delivery

Gemini 3.8 Flash TTS and Flash-Lite TTS are available alongside the existing Google voices. Add a voice on the Speech page and choose a delivery mode in its field prompt:

- **Natural** uses the voice's normal delivery and makes no text-model request.
- **Custom** uses your direction, including `{{Field}}` references. Existing saved styles remain Custom.
- **Automatic** asks your default text provider and model for a short delivery direction before synthesis. It requires a Gemini 3.8 voice, adds one text-generation request, and leaves the transcript and configured voice unchanged. Tools are disabled. Invalid output, timeout, or cancellation stops generation.

Gemini 3.8 receives directions separately from the transcript using Google's [speech metadata contract](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation). Recognized inline vocal tags such as `<sigh>` and `<short pause>` survive HTML stripping. Older Gemini models retain their existing prompting format. Test previews use the same preparation as generation and show the chosen voice, transcript, and direction before replaying the audio.

Automatic delivery has a 60-second deadline, a 4,096-character response bound, and a 1,024-output-token limit where the text provider supports it. Directions are limited to 400 characters. Reported text usage is recorded through the existing usage tracker; these bounds do not guarantee a billing ceiling. Automatic delivery is off by default. No model or voice is automatically upgraded or selected by the text model.

OpenAI's **GPT Realtime 2.1 Mini** is available in the Speech voice catalog for Natural or Custom delivery. It reads the supplied text in an isolated audio response and produces a WAV file only after successful completion. Requests are bounded to five minutes, cancellation closes the connection, and interrupted requests are not automatically replayed. Existing Speech API models remain available until their announced shutdown dates; text and instructions are each limited to 4,096 characters after field substitution.

OpenAI schedules `tts-1`, `tts-1-hd`, and the dated mini-TTS snapshots to shut down on **January 6, 2027**. The undated mini-TTS alias has no separately announced shutdown date, but its current snapshot is affected. Realtime voices differ: saved `fable`, `nova`, and `onyx` selections require an explicit replacement if changing to Realtime. See the [deprecation notice](https://developers.openai.com/api/docs/deprecations#2026-10-01-text-to-speech-models). No saved voice is automatically replaced.

## ChatGPT connection

Choose **Continue with ChatGPT** on the Providers page, finish sign-in in your browser, and allow Smart Notes to use your ChatGPT plan. Then select **ChatGPT** on the Text page and refresh its model list. The connection follows OpenAI's [native Sign in with ChatGPT reference](https://developers.openai.com/cookbook/articles/sign-in-with-chatgpt) and uses the public Responses API. No API key, Codex executable, or other application is required. Speech and image generation still use their existing providers.

Older Smart Notes ChatGPT connections require a new sign-in. Their credentials cannot be reused for the new app registration. Saved prompts and model choices are preserved. Saved ChatGPT accounts can be selected independently of Anki profiles; each registration retains its own identity and credentials, even when email addresses match. Account actions apply immediately across Anki profiles. Save and Cancel continue to govern ordinary settings and prompts.

**Manage usage** opens ChatGPT's usage settings, where you can review plan usage and set Smart Notes' weekly app limit. A cap does not reserve a portion of your allowance. When OpenAI reports a subscription limit, Smart Notes stops starting new ChatGPT requests; after reviewing or changing the limit, explicitly resume requests in Providers. The add-on does not infer remaining percentages or reset times, switch to API billing, or count ChatGPT usage against the OpenAI API estimated token guard. See [OpenAI's usage guidance](https://learn.chatgpt.com/docs/sign-in-with-chatgpt#see-and-manage-usage).

Credentials and the installation's host ID are stored in private application data, outside Anki profiles and exports. ID tokens are verified before account identity is accepted. Signing out cancels active work and invalidates pending note writes, attempts to revoke the renewable session, and clears that account's local tokens while retaining its registration for later sign-in. If remote revocation cannot be confirmed, disconnect the app in ChatGPT settings.

ChatGPT uses its subscription limits, separately from the OpenAI API estimated token guard. Model discovery preserves saved unavailable model IDs and reasoning settings. Temperature and output-token limits are not sent. Only completed final answers can be saved; interrupted or incomplete streams are discarded without automatic replay. Existing built-in and MCP tools run through Smart Notes, with a maximum of 12 calls per generated field. Each field has a separate conversation and a five-minute timeout.

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

For independent MCP integration tests, install `mcp==1.12.4 pydantic==2.11.10` in a separate Python environment and set `MCP_TEST_PYTHON` to its interpreter before running `python -m pytest tests/test_mcp.py`. The Pydantic pin preserves compatibility with this SDK version. These tests start local subprocess/HTTP servers. The SDK is a test dependency only and is never bundled. CI runs checks, unit tests, SDK integration tests, and packaging on Windows and macOS.

Before release, verify the built add-on in disposable Windows and macOS profiles at an 800×650 logical viewport: profile switching, active shutdown, unsaved editors, undo, sync, settings migration, and media updates. Paid-provider smoke tests require separate authorization.

## Project

See [changelog.md](changelog.md) for version history and [the fork's issue tracker](https://github.com/clydewk/anki-smart-notes/issues) for reports. Original project: [piazzatron/anki-smart-notes](https://github.com/piazzatron/anki-smart-notes). Licensed under GPLv3 or later; see [license](license).
