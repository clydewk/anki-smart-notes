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

import os
import subprocess
import sys
from pathlib import Path


def test_nested_editors_preserve_drafts_and_make_preview_data_explicit(
    tmp_path: Path,
) -> None:
    script = r"""
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QPlainTextEdit, QScrollArea, QTabWidget
from anki.collection import Collection
from src.config import config_context
from src.constants import GLOBAL_DECK_ID
from src.models import DEFAULT_EXTRAS
import src.decks as decks_module
import src.notes as notes_module
import src.utils as utils_module
import src.ui.prompt_dialog as prompt_module
import src.ui.custom_prompt as palette_module
from src.ui.chat_options import ChatOptions
from src.ui.image_options import ImageOptions
from src.ui.tts_options import TTSOptions

app = QApplication([])
settings = json.loads(Path("config.json").read_text())
settings.update(openai_api_key="test-key", chat_reasoning_effort="xhigh", prompts_map={"note_types": {}})
config_context.set(settings)
collection = Collection(str(Path(sys.argv[1]) / "nested.anki2"))
for module in [decks_module, notes_module, utils_module, prompt_module, palette_module]:
    module.mw = SimpleNamespace(col=collection)
messages = []
prompt_module.show_message_box = lambda *args, **kwargs: messages.append(args)
note = collection.new_note(collection.models.by_name("Basic"))
note["Front"] = "Sample source <script>literal data</script>"
collection.add_note(note, GLOBAL_DECK_ID)
prompt_module.get_random_note = lambda *args, **kwargs: note
calls = []
def launch(*args, **kwargs):
    calls.append((args, kwargs))
prompt_module.run_async_in_background_with_sentry = launch
processor = SimpleNamespace(field_processor=SimpleNamespace())
for kind in ("chat", "image", "tts"):
    dialog = prompt_module.PromptDialog(settings["prompts_map"], processor, lambda value: None, field_type=kind, deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back", prompt="Describe {{Front}}")
    assert dialog.windowTitle().startswith("Edit ")
    assert dialog.enabled_box.text() == "Generate this field automatically"
    assert dialog.model_options.isHidden()
    assert "defaults" in dialog.effective_summary.text()
    if kind == "chat":
        assert "Temperature: not used" in dialog.effective_summary.text()
        assert "Reasoning: xhigh" in dialog.effective_summary.text()
        dialog.state.update({"use_custom_model": True})
        dialog.chat_options.state.update({"chat_provider": "anthropic", "chat_model": "claude-sonnet-4-6", "chat_temperature": 0, "chat_reasoning_effort": "xhigh"})
        assert "Temperature: 0" in dialog.effective_summary.text()
        assert "Reasoning:" not in dialog.effective_summary.text()
        dialog.state.update({"use_custom_model": False})
        assert "Temperature: not used" in dialog.effective_summary.text()
    if kind == "image":
        inherited_summary = dialog.effective_summary.text()
        dialog.state.update({"use_custom_model": True})
        dialog.image_options.state.update({"image_aspect_ratio": "9:16"})
        assert "9:16" in dialog.effective_summary.text()
        dialog.state.update({"use_custom_model": False})
        assert dialog.effective_summary.text() == inherited_summary
    assert len(dialog.findChildren(ChatOptions)) + len(dialog.findChildren(TTSOptions)) + len(dialog.findChildren(ImageOptions)) == 1
    dialog.state.update({"use_custom_model": True})
    assert not dialog.model_options.isHidden()
    dialog.state.update({"use_custom_model": False})
    assert dialog.model_options.isHidden()
    assert not dialog.test_button.isEnabled()
    dialog.on_test()
    assert not calls
    dialog.load_sample()
    assert dialog.test_button.isEnabled()
    assert "Sample source" in dialog.sample_fields.toPlainText()
    assert "Provider:" in dialog.sample_description.text()
    if kind == "chat":
        dialog.on_test()
        assert len(calls) == 1
        calls[-1][1]["on_finished"]()
        calls.clear()
    dialog.reject()

speech_extras = deepcopy(DEFAULT_EXTRAS)
speech_extras.update(type="tts", tts_style="Warm {{Front}}", tts_voice_pool=[{
    "provider": "google", "model": "gemini-3.8-flash-tts", "voice": "Kore",
    "language": None, "enabled": True,
}], use_custom_model=True)
speech_map = {"note_types": {"Basic": {str(GLOBAL_DECK_ID): {"fields": {"Back": "{{Front}}"}, "extras": {"Back": speech_extras}}}}}
speech_original = deepcopy(speech_map)
speech_saved = []
speech = prompt_module.PromptDialog(speech_map, processor, speech_saved.append, field_type="tts", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back", prompt="{{Front}}")
speech.resize(800, 650)
speech.show()
app.processEvents()
assert speech.state.s["tts_delivery"] == "custom"
assert speech.tts_delivery_combo.currentText() == "Custom"
assert not speech.tts_style_box.isHidden()
speech.load_sample()
assert "Sample source" in speech.sample_fields.toPlainText()
speech.tts_delivery_combo.setCurrentIndex(speech.tts_delivery_combo.findData("automatic"))
assert speech.state.s["tts_delivery"] == "automatic"
assert speech.tts_style_box.isHidden()
assert "one text request" in speech.tts_delivery_help.text()
assert "Gemini 3.8" in speech.tts_delivery_help.text()
assert speech.state.s["tts_style"] == "Warm {{Front}}"
for index in range(3):
    speech.findChild(QTabWidget).setCurrentIndex(index)
    app.processEvents()
    page = speech.findChild(QTabWidget).currentWidget()
    assert isinstance(page, QScrollArea)
    assert page.horizontalScrollBar().maximum() == 0
    assert speech.standard_buttons.geometry().bottom() < speech.height()
speech.findChild(QTabWidget).setCurrentIndex(0)
app.processEvents()
assert speech.grab().save(str(Path(sys.argv[1]) / "speech-delivery-800x650.png"))
speech.on_accept()
assert speech_saved[0]["note_types"]["Basic"][str(GLOBAL_DECK_ID)]["extras"]["Back"]["tts_delivery"] == "automatic"
assert speech_map == speech_original
cancelled = prompt_module.PromptDialog(speech_map, processor, speech_saved.append, field_type="tts", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back", prompt="{{Front}}")
cancelled.state.update({"tts_delivery": "natural", "tts_style": "Changed draft"})
assert cancelled.tts_style_box.isHidden()
cancelled.reject()
assert len(speech_saved) == 1
assert speech_map == speech_original
preview = prompt_module.SpeechTestDialog(prompt_module.SpeechResult(b"audio", speech_extras["tts_voice_pool"][0], "Calm <sigh> delivery", "Text <sigh> stays literal."))
preview.resize(800, 650)
preview.show()
app.processEvents()
preview_text = preview.findChild(QPlainTextEdit).toPlainText()
assert "Kore" in preview_text and "gemini-3.8-flash-tts" in preview_text
assert "Direction: Calm <sigh> delivery" in preview_text
assert "Text <sigh> stays literal." in preview_text
assert preview.findChild(QPlainTextEdit).isReadOnly()
assert preview.grab().save(str(Path(sys.argv[1]) / "speech-result-800x650.png"))
preview.close()
settings["prompts_map"] = speech_map
palette_requests = []
async def generate_speech(**kwargs):
    palette_requests.append(kwargs)
    return prompt_module.SpeechResult(b"audio", speech_extras["tts_voice_pool"][0], "Warm sample", "Sample source")
palette_module.field_processor.generate_speech = generate_speech
palette_jobs = []
def launch_palette(operation, **callbacks):
    palette_jobs.append((operation, callbacks))
palette_module.run_async_in_background_with_sentry = launch_palette
palette_module.play_audio = lambda data: None
palette = palette_module.CustomTTSPrompt(note, GLOBAL_DECK_ID, "Back", lambda result: None)
assert palette.tts_options.state.s["tts_voice_pool"] == speech_extras["tts_voice_pool"]
palette._on_generate()
operation, callbacks = palette_jobs[-1]
result = asyncio.run(operation())
callbacks["on_success"](result)
callbacks["on_finished"]()
assert palette_requests[-1]["extras"]["tts_style"] == "Warm {{Front}}"
assert palette_requests[-1]["field"] == "Back"
assert palette.audio == b"audio" and palette.audio_extension == "wav"
assert "Kore" in palette.speech_details.text()
assert "Warm sample" in palette.speech_details.text()
assert palette._save_button.isEnabled()
assert note["Back"] == ""
settings["prompts_map"] = {"note_types": {}}

extras = deepcopy(DEFAULT_EXTRAS)
extras.update(type="chat", use_custom_model=True, chat_provider="openai", chat_model="custom-long-model-id", chat_temperature=0, regenerate_when_batching=True)
legacy = {"note_types": {"Deleted note type": {str(GLOBAL_DECK_ID): {"fields": {"Missing target": "Explain {{Front}}"}, "extras": {"Missing target": extras}}}}}
original = deepcopy(legacy)
accepted = []
repair = prompt_module.PromptDialog(legacy, processor, accepted.append, field_type="chat", deck_id=GLOBAL_DECK_ID, card_type="Deleted note type", field="Missing target", prompt="Explain {{Front}}")
assert repair.repairing
assert repair.note_combo_box.isEnabled()
assert repair.field_combo_box.isEnabled()
repair._on_new_card_type_selected("Basic")
repair.on_target_field_changed("Back")
assert repair.state.s["prompt"] == "Explain {{Front}}"
assert repair.state.s["regenerate_when_batching"]
assert repair.chat_options.state.s["chat_model"] == "custom-long-model-id"
assert repair.chat_options.state.s["chat_temperature"] == 0
repair.on_accept()
assert legacy == original
assert accepted and "Deleted note type" not in accepted[0]["note_types"]
assert accepted[0]["note_types"]["Basic"][str(GLOBAL_DECK_ID)]["fields"]["Back"] == "Explain {{Front}}"
assert accepted[0]["note_types"]["Basic"][str(GLOBAL_DECK_ID)]["extras"]["Back"]["chat_model"] == "custom-long-model-id"
result = prompt_module.TextTestDialog("Long response\n" * 500)
assert result.findChild(QPlainTextEdit).isReadOnly()
assert result.findChild(QPlainTextEdit).toPlainText().count("Long response") == 500
assert result.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close) is not None
assert not messages
collection.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_provider_and_mcp_editors_validate_inline_and_cancel_probes() -> None:
    script = r"""
import json
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QFormLayout, QLineEdit, QPushButton
from src.config import config_context
from src.job_context import JobCancelledError
import src.ui.custom_provider_dialog as provider_module
import src.ui.mcp_server_dialog as mcp_module

app = QApplication([])
config_context.set(json.loads(Path("config.json").read_text()))
requests = []
def launch(operation, success, failure, **kwargs):
    job = SimpleNamespace(cancel=lambda: failure(JobCancelledError("cancelled")))
    requests.append(SimpleNamespace(operation=operation, success=success, failure=failure, options=kwargs, job=job))
    return job
provider_module.run_async_in_background_with_sentry = launch
mcp_module.run_async_in_background_with_sentry = launch
provider = provider_module.CustomProviderDialog()
provider.accept()
assert "name" in provider.status_label.text().lower()
provider.name_edit.setText("My provider")
provider.url_edit.setText("https://example.test/v1")
provider.chat_check.setChecked(False)
provider.accept()
assert "at least one" in provider.status_label.text()
provider.image_check.setChecked(True)
assert provider.models_tabs.isTabEnabled(2)
assert not provider.models_tabs.isTabEnabled(0)
assert provider.models_tabs.currentIndex() == 2
assert not provider.api_mode_combo.isEnabled()
provider.image_models_edit.setPlainText("custom-image-id\nother-image")
assert provider.get_provider()["capabilities"] == ["image"]
assert provider.get_provider()["image_models"] == ["custom-image-id", "other-image"]
provider.on_fetch_models()
assert not provider.fetch_btn.isEnabled()
assert requests[-1].options["show_error"] is False
provider._cancel_fetch()
assert provider.fetch_btn.isEnabled()
provider.on_fetch_models()
requests[-1].success(SimpleNamespace(models=[], chat_api_mode="responses"))
assert "no model IDs" in provider.status_label.text()
assert provider.image_models_edit.toPlainText() == "custom-image-id\nother-image"
provider.resize(650, 470)
provider.show()
app.processEvents()
assert provider.url_edit.width() >= 300
assert provider.key_edit.echoMode() == QLineEdit.EchoMode.Password
assert any(button.accessibleName() == "Show custom provider API key" for button in provider.findChildren(provider_module.QCheckBox))
assert provider.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).text() == "Done"
provider.reject()

mcp = mcp_module.McpServerDialog()
mcp.name_edit.setText("Local tools")
mcp.command_edit.setText("npx")
mcp.cwd_edit.setText("~/project")
mcp._on_test_connection()
assert "absolute" in mcp.status_label.text()
assert len(requests) == 2
mcp.cwd_edit.setText("")
original_argument_row = mcp.args_editor._rows[0][0]
mcp.args_editor.set_values(["-y", "example-server"])
assert original_argument_row.isHidden(), "Deferred row deletion must not leave ghost inputs visible"
removed_argument_row = mcp.args_editor._rows[-1][0]
mcp.args_editor.remove_row(removed_argument_row)
assert removed_argument_row.isHidden()
mcp.args_editor.set_values(["-y", "example-server"])
assert mcp.cwd_edit.accessibleName() == "MCP working directory"
mcp.env_editor.set_pairs([{"key": "TOKEN", "value": "private-value"}])
assert mcp.get_server()["args"] == ["-y", "example-server"]
assert mcp.get_server()["env"] == [{"key": "TOKEN", "value": "private-value"}]
assert all(edit.minimumWidth() < 360 for edit in mcp.findChildren(QLineEdit))
assert any(button.accessibleName().startswith("Remove environment variables") for button in mcp.env_editor.findChildren(QPushButton))
assert any(button.accessibleName().startswith("Show environment variables") for button in mcp.env_editor.findChildren(QPushButton))
assert mcp.env_editor.findChild(QFormLayout) is not None
mcp._on_test_connection()
assert not mcp.test_button.isEnabled()
assert requests[-1].options["show_error"] is False
mcp._cancel_probe()
assert mcp.test_button.isEnabled()
mcp._set_transport("streamable_http")
mcp.url_edit.setText("file:///tmp/data")
mcp.accept()
assert "HTTP or HTTPS" in mcp.status_label.text()
mcp.url_edit.setText("https://example.test/mcp")
mcp._on_test_connection()
requests[-1].success(SimpleNamespace(server_info=SimpleNamespace(name="Peer", version="1"), tools=[]))
assert "connected successfully" in mcp.status_label.text()
assert mcp.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).text() == "Done"
mcp.reject()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tools_probe_keeps_selection_safe_and_reports_cancellation(
    tmp_path: Path,
) -> None:
    script = r"""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication
from src.job_context import JobCancelledError
import src.config as config_module
import src.ui.addon_options_dialog as options_module
import src.ui.chatgpt_account as account_module

app = QApplication([])
account_module.chatgpt_auth.status = lambda: None
defaults = json.loads(Path("config.json").read_text())
defaults["mcp_servers"] = [
    {"id": name, "name": name, "enabled": True, "transport": "stdio", "command": "unused-test-command", "args": [], "cwd": "", "env": [], "env_passthrough": [], "url": "", "headers": [], "header_env_vars": []}
    for name in ("First server", "Second server")
]
writes = []
config_module.mw = SimpleNamespace(
    addonManager=SimpleNamespace(getConfig=lambda name: deepcopy(defaults), writeConfig=lambda *args: writes.append(args)),
    pm=SimpleNamespace(name="Test", profileFolder=lambda: sys.argv[1]),
)
options_module.get_version = lambda: "test"
options_module.chat_usage_tracker.get_openai_daily_usage = lambda: SimpleNamespace(used_total_tokens=100, estimated_total_tokens=30)
opened = []
options_module.McpServerDialog.exec = lambda dialog: opened.append(True) or 0
requests = []
def launch(operation, success, failure, **kwargs):
    request = SimpleNamespace(operation=operation, success=success, failure=failure, options=kwargs, settings=config_module.config.snapshot(), cancelled=False)
    def cancel():
        request.cancelled = True
    request.job = SimpleNamespace(cancel=cancel)
    requests.append(request)
    return request.job
options_module.run_async_in_background_with_sentry = launch
options = options_module.AddonOptionsDialog(MagicMock())
options.show()
app.processEvents()
assert "100" in options.usage_label.text() and "30" in options.usage_label.text()
assert "100 reported +" not in options.usage_label.text()
options.state.update({"openai_api_key": "draft-only-key"})
options.mcp_selector.setCurrentIndex(0)
options._on_test_selected_mcp_server()
assert len(requests) == 1
assert requests[0].settings["openai_api_key"] == "draft-only-key"
assert requests[0].options["owner"] is options
assert requests[0].options["show_error"] is False
assert not options.mcp_test_button.isEnabled()
options.mcp_selector.setCurrentIndex(1)
assert not options.mcp_test_button.isEnabled()
assert not options.mcp_edit_button.isEnabled()
options._on_edit_selected_mcp_server()
assert not opened
options._on_test_selected_mcp_server()
assert len(requests) == 1
options._cancel_mcp_test()
assert requests[0].cancelled
requests[0].failure(JobCancelledError("cancelled"))
requests[0].options["on_finished"]()
assert "cancel" in options.mcp_status.text().lower()
assert "fail" not in options.mcp_status.text().lower()
assert options.mcp_test_button.isEnabled()
assert options._selected_mcp_server()["id"] == "Second server"
options._on_test_selected_mcp_server()
assert len(requests) == 2
requests[1].success(SimpleNamespace(server_info=SimpleNamespace(name="Second server"), tools=[]))
requests[1].options["on_finished"]()
assert "Second server" in options.mcp_status.text()
assert "0 tools" in options.mcp_status.text()
assert options._mcp_job is None
options.on_reject()
assert not writes
assert config_module.config.openai_api_key == defaults["openai_api_key"]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
