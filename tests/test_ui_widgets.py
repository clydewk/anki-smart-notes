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

import pytest


def test_native_widgets_keep_undo_zero_values_and_image_aspect_ratio() -> None:
    script = """
import json
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PyQt6.QtGui import QImage
from src.config import config_context
from src.ui.chat_options import ChatOptions
from src.ui.custom_provider_dialog import validate_custom_provider
from src.ui.image_options import ImageOptions
from src.ui.image_displayer import ImageDisplayer
from src.ui.reactive_edit_text import ReactiveEditText
from src.ui.reactive_spin_box import ReactiveSpinBox
from src.ui.state_manager import StateManager
from src.ui.tts_options import TTSOptions, base_providers, base_voices

app = QApplication([])
state = StateManager({"prompt": "initial", "count": 5})
edit = ReactiveEditText(state, "prompt")
edit.on_change.connect(lambda value: state.update({"prompt": value}))
edit.insertPlainText("typed ")
assert edit.document().isUndoAvailable()
edit.undo()
assert edit.toPlainText() == "initial"
state.update({"prompt": "x"})
assert edit.textCursor().position() <= 1
spinner = ReactiveSpinBox(state, "count")
state.update({"count": 0})
assert spinner.value() == 0
try:
    state.update({"missing": True})
except AssertionError:
    pass
assert not state.updating

image = QImage(200, 100, QImage.Format.Format_RGB32)
image.fill(Qt.GlobalColor.red)
array = QByteArray()
buffer = QBuffer(array)
buffer.open(QIODevice.OpenModeFlag.WriteOnly)
assert image.save(buffer, "PNG")
display = ImageDisplayer(array.data(), width=300, height=300)
display.show()
app.processEvents()
pixmap = display.label.pixmap()
assert pixmap.width() == 2 * pixmap.height()
assert not hasattr(display, "webview")

validate_custom_provider("Local", "http://localhost:11434")
for name, url, existing in [
    ("openai", "https://example.com", ()),
    ("duplicate", "https://example.com", ("DUPLICATE",)),
    ("", "https://example.com", ()),
    ("Local", "file:///tmp/key", ()),
    ("Local", "https://user:password@example.com", ()),
    ("Local", "https://example.com?key=secret", ()),
    ("Local", "http://localhost:bad", ()),
]:
    try:
        validate_custom_provider(name, url, existing)
    except ValueError:
        pass
    else:
        raise AssertionError((name, url))

assert "azure" not in base_providers
assert all(voice["tts_provider"] != "azure" for voice in base_voices)
config_context.set(json.loads(Path("config.json").read_text()))
options = TTSOptions()
options.state.update({"catalog_provider": "azure", "catalog_model": "neural", "catalog_voice": "saved"})
options.state.update({"search_text": "no-matching-voice"})
assert options.voices_models.get_data() == []
assert not options.test_button.isEnabled()
assert "unavailable" in options.selected_voice_label.text()
chat = ChatOptions({"chat_provider": "anthropic", "chat_model": "claude-sonnet-4-0"})
assert chat.state.s["chat_model"] == "claude-sonnet-4-0"
assert "unavailable" in chat.model_warning.text()
assert not chat.reasoning_effort.isEnabled()
assert chat.temperature.isEnabled()
astra = ChatOptions({"chat_provider": "openai", "chat_model": "gpt-6-astra"})
assert not astra.temperature.isEnabled()
images = ImageOptions({"image_provider": "google", "image_model": "gemini-2.0-flash-preview-image-generation", "image_quality": 0})
assert images.state.s["image_model"] == "gemini-2.0-flash-preview-image-generation"
assert images.quality_spinner.value() == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_settings_dialog_drafts_save_cancel_and_footer(tmp_path: Path) -> None:
    script = """
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QWidget
from anki.collection import Collection
from anki.decks import DeckId
from threading import Event
from src.note_processor import commit_processed_notes
from src.ui.field_menu import FieldMenu
import src.ui.field_menu as field_menu_module
import src.ui.custom_prompt as palette_module
from src.constants import GLOBAL_DECK_ID
import src.decks as decks_module
import src.notes as notes_module
import src.utils as utils_module
import src.ui.prompt_dialog as prompt_module
import src.config as config_module
import src.ui.addon_options_dialog as options_module

app = QApplication([])
defaults = json.loads(Path("config.json").read_text())
class Manager:
    def __init__(self):
        self.values = deepcopy(defaults)
        self.writes = 0
        self.fail = False
    def getConfig(self, name):
        return deepcopy(self.values)
    def writeConfig(self, name, value):
        if self.fail:
            raise OSError("simulated storage failure")
        self.values = deepcopy(value)
        self.writes += 1
manager = Manager()
config_module.mw = SimpleNamespace(addonManager=manager, pm=SimpleNamespace(name="Test", profileFolder=lambda: sys.argv[1]))
config_module.Config.defaults = lambda self: deepcopy(defaults)
messages = []
options_module.show_message_box = lambda *args, **kwargs: messages.append(args)
options_module.get_version = lambda: "test"
dialog = options_module.AddonOptionsDialog(MagicMock())
dialog.resize(800, 650)
dialog.show()
app.processEvents()
buttons = dialog.findChild(QDialogButtonBox)
for standard in [QDialogButtonBox.StandardButton.Save, QDialogButtonBox.StandardButton.Cancel]:
    button = buttons.button(standard)
    origin = button.mapTo(dialog, button.rect().topLeft())
    assert 0 <= origin.y() and origin.y() + button.height() <= dialog.height()
    assert 0 <= origin.x() and origin.x() + button.width() <= dialog.width()
dialog.state.update({"openai_api_key": "draft-key", "generate_at_review": False})
dialog.chat_options.state.update({"chat_temperature": 0})
assert manager.writes == 0
dialog.on_reject()
assert manager.values == defaults
assert manager.writes == 0

saved = options_module.AddonOptionsDialog(MagicMock())
saved.state.update({"openai_api_key": "draft-key", "generate_at_review": False})
saved.chat_options.state.update({"chat_temperature": 0})
assert saved.write_config()
assert manager.writes == 1
assert config_module.config.openai_api_key == "draft-key"
assert config_module.config.generate_at_review is False
assert config_module.config.chat_temperature == 0
manager.fail = True
saved.state.update({"openai_api_key": "unsaved-key"})
assert not saved.write_config()
assert saved.state.s["openai_api_key"] == "unsaved-key"
assert config_module.config.openai_api_key == "draft-key"
assert "could not be saved" in saved.save_status.text()
assert "simulated storage failure" in saved.save_status.text()
manager.fail = False
manager.values["openai_api_key"] = ""
collection = Collection(str(Path(sys.argv[1]) / "collection.anki2"))
collection_window = SimpleNamespace(col=collection)
for module in [decks_module, notes_module, utils_module, prompt_module]:
    module.mw = collection_window
prompt_module.show_message_box = lambda *args, **kwargs: messages.append(args)
parent = options_module.AddonOptionsDialog(MagicMock())
before_prompts = deepcopy(config_module.config.prompts_map)
child = prompt_module.PromptDialog(
    parent.state.s["prompts_map"], MagicMock(), parent.on_update_prompts,
    field_type="chat", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back",
    prompt="Explain {{Front}}", parent=parent,
)
child.state.update({"prompt": "Cancelled {{Front}}"})
child.reject()
assert parent.state.s["prompts_map"] == before_prompts
child = prompt_module.PromptDialog(
    parent.state.s["prompts_map"], MagicMock(), parent.on_update_prompts,
    field_type="chat", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back",
    prompt="Explain {{Front}}", parent=parent,
)
child.on_accept()
assert parent.state.s["prompts_map"] != before_prompts
assert config_module.config.prompts_map == before_prompts
parent.on_reject()
assert config_module.config.prompts_map == before_prompts
active = Event()
active.set()
palette_module.tasks.worker = SimpleNamespace(collection=collection, generation=1, active=active)
palette_module.mw = collection_window
field_menu_module.mw = collection_window
palette_module.show_message_box = lambda *args, **kwargs: messages.append(args)
callbacks = {}
def fake_runner(operation, success, failure=None, **kwargs):
    def complete(callback, value):
        try:
            callback(value)
        finally:
            if kwargs.get("on_finished"):
                kwargs["on_finished"]()
    callbacks.update(success=lambda value: complete(success, value), failure=lambda error: complete(failure, error), **kwargs)
palette_module.run_async_in_background_with_sentry = fake_runner
note = collection.new_note(collection.models.by_name("Basic"))
note["Front"] = "source"
editor = SimpleNamespace(note=note, web=QWidget(), loadNote=MagicMock())
menu = object.__new__(FieldMenu)
menu.editor = editor
menu.card = SimpleNamespace(note=lambda: note)
menu.field_upper = "Back"
original_count = collection.note_count()
palette = palette_module.CustomTextPrompt(note=note, deck_id=GLOBAL_DECK_ID, field_upper="Back", on_success=menu._make_custom_field_success())
palette._prompt_window.setPlainText("Explain {{Front}}")
palette._response_edit.setPlainText("previous complete result")
palette._on_generate()
assert palette._response_edit.toPlainText() == ""
assert not palette._save_button.isEnabled()
callbacks["failure"](ValueError("incomplete response"))
assert not palette._save_button.isEnabled()
palette._on_generate()
callbacks["success"]("<script>unsafe</script> **result**")
assert palette._save_button.isEnabled()
palette._on_save_result()
assert note.id == 0
assert collection.note_count() == original_count
assert "<script>" not in note["Back"]
assert "result" in note["Back"]
editor.loadNote.assert_called_once()

persisted = collection.new_note(collection.models.by_name("Basic"))
persisted["Front"] = "persisted"
collection.add_note(persisted, DeckId(1))
results = []
palette = palette_module.CustomTextPrompt(note=persisted, deck_id=DeckId(99999), field_upper="Back", on_success=results.append)
assert palette._snapshot().deck_name is None
palette._response_edit.setPlainText("generated result")
palette._on_save_result()
assert collection.get_note(persisted.id)["Back"] == ""
commit_processed_notes(collection, results)
assert "generated result" in collection.get_note(persisted.id)["Back"]
collection.undo()
assert collection.get_note(persisted.id)["Back"] == ""
deleted_results = []
palette = palette_module.CustomTextPrompt(note=collection.get_note(persisted.id), deck_id=DeckId(1), field_upper="Back", on_success=deleted_results.append)
palette._response_edit.setPlainText("late result")
collection.remove_notes([persisted.id])
palette._on_save_result()
assert deleted_results == []
assert any("deleted" in str(message).lower() for message in messages)
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


def test_prompt_previews_capture_drafts_and_finalize_after_dialog_close(
    tmp_path: Path,
) -> None:
    script = """
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from PyQt6 import sip
from PyQt6.QtWidgets import QApplication
from anki.collection import Collection
from src.chat_usage import chat_usage_tracker
from src.constants import GLOBAL_DECK_ID
import src.config as config_module
import src.decks as decks_module
import src.notes as notes_module
import src.utils as utils_module
import src.ui.addon_options_dialog as options_module
import src.ui.prompt_dialog as prompt_module
from src.ui.chat_options import ChatOptions
from src.ui.custom_prompt import CustomTextPrompt

app = QApplication([])
ChatOptions._refresh_openai_models_if_needed = lambda *args: None
defaults = json.loads(Path("config.json").read_text())
defaults.update(openai_api_key="persisted-key", tts_strip_html=True)
config_module.mw = SimpleNamespace(
    addonManager=SimpleNamespace(getConfig=lambda name: deepcopy(defaults)),
    pm=SimpleNamespace(name="Test", profileFolder=lambda: sys.argv[1]),
)
config_module.Config.defaults = lambda self: deepcopy(defaults)
collection = Collection(str(Path(sys.argv[1]) / "preview.anki2"))
for module in [decks_module, notes_module, utils_module, prompt_module]:
    module.mw = SimpleNamespace(col=collection)
note = collection.new_note(collection.models.by_name("Basic"))
note["Front"] = "captured source"
collection.add_note(note, GLOBAL_DECK_ID)
prompt_module.get_random_note = lambda note_type, deck_id: note
messages = []
prompt_module.show_message_box = lambda *args, **kwargs: messages.append(args)
prompt_module.play_audio = lambda data: None
prompt_module.TextTestDialog.exec = lambda dialog: messages.append((dialog.findChild(prompt_module.QPlainTextEdit).toPlainText(),)) or 0
options_module.show_message_box = lambda *args, **kwargs: messages.append(args)
options_module.get_version = lambda: "test"
captured = {}
def launch(operation, success, failure=None, **kwargs):
    captured.clear()
    captured.update(operation=operation, success=success, failure=failure,
                    settings=config_module.config.snapshot(), **kwargs)
prompt_module.run_async_in_background_with_sentry = launch
requests = []
async def get_chat_response(**kwargs):
    requests.append((kwargs, config_module.config.openai_api_key))
    return "complete response"
async def get_tts_response(**kwargs):
    requests.append((kwargs, config_module.config.openai_api_key))
    return b"audio"
processor = SimpleNamespace(field_processor=SimpleNamespace(
    get_chat_response=get_chat_response, get_tts_response=get_tts_response,
))
parent = options_module.AddonOptionsDialog(processor)
parent.state.update({"openai_api_key": "draft-key"})
parent.tts_options.state.update({"tts_voice_pool": [{
    "provider": "openai", "model": "tts-1", "voice": "alloy",
    "enabled": True, "language": None,
}]})
parent.chat_options.state.update({"chat_model": "gpt-5.6-sol"})
child = prompt_module.PromptDialog(
    parent.state.s["prompts_map"], processor, parent.on_update_prompts,
    field_type="chat", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back",
    prompt="Explain {{Front}}", parent=parent,
)
assert child.get_current_chat_settings()[1] == "gpt-5.6-sol"
initial_signature = child.get_current_prompt_signature()
parent.chat_options.state.update({"chat_model": "gpt-5.6-terra"})
assert child.get_current_prompt_signature() != initial_signature
before_scopes = set(chat_usage_tracker._scopes)
child.load_sample()
child.on_test()
assert child.state.s["is_loading_prompt"]
assert set(chat_usage_tracker._scopes) != before_scopes
parent.state.update({"openai_api_key": "later-key"})
child.state.update({"prompt": ""})
note["Front"] = "later source"
token = config_module.config_context.set(captured["settings"])
try:
    result = asyncio.run(captured["operation"]())
finally:
    config_module.config_context.reset(token)
assert requests[-1][0]["prompt"] == "Explain {{Front}}"
assert requests[-1][0]["values"]["front"] == "captured source"
assert requests[-1][0]["model"] == "gpt-5.6-terra"
assert requests[-1][1] == "draft-key"
try:
    captured["success"](result)
finally:
    captured["on_finished"]()
assert not child.state.s["is_loading_prompt"]
assert set(chat_usage_tracker._scopes) == before_scopes
assert any("captured source" in str(message) for message in messages)

child.state.update({"prompt": "Again {{Front}}"})
child.on_test()
assert set(chat_usage_tracker._scopes) != before_scopes
child.reject()
captured["on_finished"]()
assert set(chat_usage_tracker._scopes) == before_scopes
assert not child.state.s["is_loading_prompt"]

child.on_test()
assert set(chat_usage_tracker._scopes) != before_scopes
sip.delete(child)
captured["on_finished"]()
assert set(chat_usage_tracker._scopes) == before_scopes
palette = CustomTextPrompt(
    note=note, deck_id=GLOBAL_DECK_ID, field_upper="Back", on_success=lambda result: None,
)
palette._loading = True
sip.delete(palette)
palette._finish_generation()
assert not palette._loading

tts = prompt_module.PromptDialog(
    parent.state.s["prompts_map"], processor, parent.on_update_prompts,
    field_type="tts", deck_id=GLOBAL_DECK_ID, card_type="Basic", field="Back",
    prompt="{{Front}}", parent=parent,
)
tts.state.update({"use_custom_model": True})
tts.tts_options.state.update({"tts_strip_html": False})
tts.state.update({"use_custom_model": False})
tts.load_sample()
tts.on_test()
token = config_module.config_context.set(captured["settings"])
try:
    asyncio.run(captured["operation"]())
finally:
    config_module.config_context.reset(token)
    captured["on_finished"]()
assert requests[-1][0]["strip_html"] is True
assert not tts.state.s["is_loading_prompt"]

def reject_launch(*args, **kwargs):
    raise RuntimeError("profile closing")
prompt_module.run_async_in_background_with_sentry = reject_launch
tts.on_test()
assert not tts.state.s["is_loading_prompt"]
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


def test_native_avif_previews_keep_original_media_for_save(tmp_path: Path) -> None:
    script = """
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from anki.collection import Collection
from src.config import config_context
from src.constants import GLOBAL_DECK_ID
from src.media_utils import convert_image_data, find_executable, image_preview_data
from src.ui.image_displayer import ImageDisplayer
import src.decks as decks_module
import src.notes as notes_module
import src.utils as utils_module
import src.ui.custom_prompt as palette_module
import src.ui.prompt_dialog as prompt_module

if not find_executable("ffmpeg"):
    raise SystemExit(77)
app = QApplication([])
main_thread = get_ident()
source = QImage(128, 128, QImage.Format.Format_RGB32)
source.fill(0x3377CC)
encoded = QByteArray()
buffer = QBuffer(encoded)
buffer.open(QIODevice.OpenModeFlag.WriteOnly)
assert source.save(buffer, "PNG")
with ThreadPoolExecutor(max_workers=1) as executor:
    avif = executor.submit(copy_context().run, lambda: asyncio.run(convert_image_data(encoded.data(), "avif"))).result()
assert b"ftypavif" in avif[:32]

defaults = json.loads(Path("config.json").read_text())
defaults.update(image_output_format="avif", image_provider="google", image_model="gemini-3-pro-image", google_api_key="test-key")
config_context.set(defaults)
collection = Collection(str(Path(sys.argv[1]) / "preview.anki2"))
window = SimpleNamespace(col=collection)
for module in [palette_module, prompt_module, decks_module, notes_module, utils_module]:
    module.mw = window
active = Event()
active.set()
palette_module.tasks.worker = SimpleNamespace(collection=collection, generation=1, active=active)
note = collection.new_note(collection.models.by_name("Basic"))
note["Front"] = "a blue square"
prompt_module.get_random_note = lambda note_type, deck_id: note
messages = []
for module in [palette_module, prompt_module]:
    module.show_message_box = lambda *args, **kwargs: messages.append(args)
captured = {}
def launch(operation, on_success, on_failure=None, **kwargs):
    captured.clear()
    captured.update(operation=operation, success=on_success, failure=on_failure, **kwargs)
for module in [palette_module, prompt_module]:
    module.run_async_in_background_with_sentry = launch

async def get_image_response(**kwargs):
    assert get_ident() != main_thread
    return avif

async def prepare_preview(data):
    assert get_ident() != main_thread
    return await image_preview_data(data)

for module in [palette_module, prompt_module]:
    module.image_preview_data = prepare_preview
palette_module.field_processor = SimpleNamespace(get_image_response=get_image_response)
saved = []
palette = palette_module.CustomImagePrompt(
    note=note, deck_id=GLOBAL_DECK_ID, field_upper="Back", on_success=saved.append,
)
palette._prompt_window.setPlainText("Draw {{Front}}")
palette._on_generate()
assert not palette._save_button.isEnabled()
with ThreadPoolExecutor(max_workers=1) as executor:
    result = executor.submit(copy_context().run, lambda: asyncio.run(captured["operation"]())).result()
try:
    captured["success"](result)
finally:
    captured["on_finished"]()
assert palette.raw_image == avif
assert palette._save_button.isEnabled()
assert not palette.response_image.label.pixmap().isNull()
palette._on_save_result()
assert saved[0].media[0].data == avif
assert saved[0].media[0].filename.endswith(".avif")
assert '.avif"' in saved[0].updates["Back"]
assert note.id == 0

seen = []
def show_preview(dialog):
    image = dialog.findChild(ImageDisplayer)
    assert image is not None and not image.label.pixmap().isNull()
    seen.append(True)
    return 0
prompt_module.ImageTestDialog.exec = show_preview
prompt = prompt_module.PromptDialog(
    defaults["prompts_map"], SimpleNamespace(field_processor=palette_module.field_processor),
    lambda value: None, field_type="image", deck_id=GLOBAL_DECK_ID,
    card_type="Basic", field="Back", prompt="Draw {{Front}}",
)
prompt.load_sample()
prompt.on_test()
with ThreadPoolExecutor(max_workers=1) as executor:
    result = executor.submit(copy_context().run, lambda: asyncio.run(captured["operation"]())).result()
try:
    captured["success"](result)
finally:
    captured["on_finished"]()
assert seen == [True]
assert not prompt.state.s["is_loading_prompt"]
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
    if completed.returncode == 77:
        pytest.skip("AVIF preview integration requires ffmpeg")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_settings_model_discovery_waits_for_visible_draft_and_survives_tab_changes(
    tmp_path: Path,
) -> None:
    script = """
import asyncio
import json
import queue
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication, QTabWidget
import aqt
import src.config as config_module
import src.sentry as sentry_module
import src.tasks as tasks
import src.ui.addon_options_dialog as options_module
import src.ui.chat_options as chat_module
from src.job_context import JobCancelledError

app = QApplication([])
defaults = json.loads(Path("config.json").read_text())
defaults.update(openai_api_key="configured-test-key", chat_provider="openai", chat_model="gpt-5.6-sol")
class Manager:
    def __init__(self):
        self.writes = 0
    def getConfig(self, name):
        return deepcopy(defaults)
    def writeConfig(self, name, values):
        self.writes += 1
manager = Manager()
callbacks = queue.Queue()
window = SimpleNamespace(
    col=object(),
    taskman=SimpleNamespace(run_on_main=callbacks.put),
    addonManager=manager,
    pm=SimpleNamespace(name="Test", profileFolder=lambda: sys.argv[1]),
)
config_module.mw = window
config_module.Config.defaults = lambda self: deepcopy(defaults)
tasks.mw = window
sentry_module.mw = window
messages = []
sentry_module.show_message_box = lambda *args, **kwargs: messages.append(args)
options_module.show_message_box = lambda *args, **kwargs: messages.append(args)
options_module.get_version = lambda: "test"
requests = []
sentinel = "gpt-5.99-mini"
async def fetch_models():
    request = {"key": config_module.config.openai_api_key, "gate": asyncio.Event(), "cancelled": False}
    requests.append(request)
    try:
        await request["gate"].wait()
        request["completion_key"] = config_module.config.openai_api_key
        chat_module.chat_provider._openai_models_cache = [sentinel]
        chat_module.chat_provider._openai_models_cache_key = chat_module.chat_provider._openai_model_cache_key()
        return [sentinel]
    except asyncio.CancelledError:
        request["cancelled"] = True
        raise
chat_module.chat_provider.fetch_openai_chat_models = fetch_models

def pump_until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        app.processEvents()
        while not callbacks.empty():
            callbacks.get_nowait()()
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("Timed out waiting for the native worker")

tasks.open_profile_worker()
worker = tasks.worker
assert worker is not None
try:
    dialog = options_module.AddonOptionsDialog(MagicMock())
    assert not requests and not worker.jobs
    dialog.show()
    app.processEvents()
    tabs = dialog.findChild(QTabWidget)
    text_tab = next(index for index in range(tabs.count()) if tabs.tabText(index) == "Text")
    assert tabs.currentIndex() == 0
    assert not requests and not worker.jobs
    chat = dialog.chat_options
    dialog.state.update({"openai_api_key": "unsaved-draft-key"})
    chat.state.update({"show_available_openai_models": True})
    chat._refresh_openai_models_if_needed("openai")
    assert not requests and not worker.jobs

    tabs.setCurrentIndex(text_tab)
    pump_until(lambda: len(requests) == 1)
    assert requests[0]["key"] == "unsaved-draft-key"
    assert chat._openai_refresh_inflight
    dialog.state.update({"openai_api_key": "edited-after-submit"})
    tabs.setCurrentIndex(0)
    app.processEvents()
    assert not chat.isVisible() and dialog.isVisible()
    worker.loop.call_soon_threadsafe(requests[0]["gate"].set)
    pump_until(lambda: not worker.jobs)
    assert not chat._openai_refresh_inflight
    assert requests[0]["completion_key"] == "unsaved-draft-key"
    assert sentinel not in chat.state.s["chat_models"]
    assert "Connection changed" in chat.discovery_status.text()

    dialog.state.update({"openai_api_key": "unsaved-draft-key"})
    chat.state.update({"show_available_openai_models": False})
    chat._on_show_available_openai_models_change(False)
    assert sentinel not in chat.state.s["chat_models"]
    chat.state.update({"show_available_openai_models": True})
    chat._on_show_available_openai_models_change(True)
    assert sentinel in chat.state.s["chat_models"]
    assert len(requests) == 1
    dialog.state.update({"openai_api_key": "edited-after-submit"})

    tabs.setCurrentIndex(text_tab)
    pump_until(lambda: len(requests) == 2)
    assert requests[1]["key"] == "edited-after-submit"
    before_close = chat.state.s
    dialog.reject()
    pump_until(lambda: not worker.jobs and requests[1]["cancelled"])
    assert requests[1]["cancelled"]
    assert not chat._openai_refresh_inflight
    assert chat.state.s == before_close
    chat._refresh_openai_models_if_needed("openai")
    assert len(requests) == 2 and not worker.jobs

    dialog.show()
    pump_until(lambda: len(requests) == 3)
    worker.loop.call_soon_threadsafe(requests[2]["gate"].set)
    pump_until(lambda: not worker.jobs)
    assert not chat._openai_refresh_inflight
    assert sentinel in chat.state.s["chat_models"]

    worker.active.clear()
    try:
        try:
            chat._refresh_openai_models_if_needed("openai")
        except JobCancelledError:
            pass
        else:
            raise AssertionError("An inactive profile accepted model discovery")
        assert not chat._openai_refresh_inflight
    finally:
        worker.active.set()
    dialog.reject()
    assert not messages
    assert manager.writes == 0
    assert config_module.config.openai_api_key == "configured-test-key"
finally:
    tasks.close_profile_worker()
assert not worker.thread.is_alive()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
