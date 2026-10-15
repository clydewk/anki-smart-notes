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


def test_speech_catalog_selection_pool_editing_and_preview_lifecycle() -> None:
    script = """
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLineEdit, QScrollArea
from src.config import config_context
from src.job_context import JobCancelledError
import src.ui.tts_options as speech
import src.models as models
from datetime import date

class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 13)
models.date = FrozenDate

app = QApplication([])
config_context.set(json.loads(Path("config.json").read_text()))
gemini = speech.get_gemini_voices()
assert {voice["model"] for voice in gemini} == {
    "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts",
}
assert len(gemini) == 150
assert len([voice for voice in speech.generate_openai_voices() if voice["model"] == "gpt-realtime-2.1-mini"]) == 10
assert {voice["voice"] for voice in gemini if voice["model"] == "gemini-3.8-flash-tts"} == {voice["voice"] for voice in gemini if voice["model"] == "gemini-2.5-pro-preview-tts"}
options = speech.TTSOptions({"tts_voice_pool": [
    {"provider": "openai", "model": "gpt-4o-mini-tts", "voice": "marin", "enabled": True, "language": None},
]})
scroll = QScrollArea()
scroll.setWidgetResizable(True)
scroll.setWidget(options)
scroll.resize(650, 470)
scroll.show()
app.processEvents()
assert isinstance(options.preview_text, QLineEdit)
assert options.selected_voice_label.hasHeightForWidth()
assert options.minimumSizeHint().width() < 600
assert options.pool_table.height() < 100
assert "marin" in options.pool_table.item(0, 1).text().lower()
assert "Voice ID: marin" in options.pool_table.item(0, 1).toolTip()
assert "2027-01-06" in options.selected_voice_label.text()
assert not options.remove_voice_button.isEnabled()
options.pool_table.selectRow(0)
assert options.remove_voice_button.isEnabled()
options.pool_table.item(0, 2).setText("ja")
assert options.state.s["tts_voice_pool"][0]["language"] == "ja"
options.pool_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
assert not options.state.s["tts_voice_pool"][0]["enabled"]

options.state.update({"search_text": "no-such-voice-123"})
assert not options.voices_models.get_data()
assert "No matching voices" in options.catalog_status.text()
assert "outside current filters" in options.selected_voice_label.text().lower()
assert "marin" in options.selected_voice_label.text().lower()
assert options.test_button.isEnabled()
options.clear_filters_button.click()
assert options.voices_models.get_data()
assert "outside current filters" not in options.selected_voice_label.text().lower()
assert not options.clear_filters_button.isEnabled()
options.preview_text.clear()
assert not options.test_button.isEnabled()
options.preview_text.setText("A brief preview")
assert options.test_button.isEnabled()

requests = []
def run(operation, **callbacks):
    handle = SimpleNamespace(cancel=Mock())
    requests.append({"operation": operation, "handle": handle, **callbacks})
    return handle
speech.run_async_in_background_with_sentry = run
speech.play_audio = Mock()
options.test_button.click()
assert len(requests) == 1
request = requests[-1]
assert request["show_error"] is False
assert not options.test_button.isEnabled()
assert "Generating" in options.preview_status.text()
assert options.cancel_preview_button.isVisible()
options.cancel_preview()
request["handle"].cancel.assert_called_once()
request["on_failure"](JobCancelledError("Generation cancelled."))
request["on_finished"]()
assert options._preview_job is None
assert options.test_button.isEnabled()
assert "cancelled" in options.preview_status.text()
assert not options.cancel_preview_button.isVisible()
options.test_and_play()
requests[-1]["on_failure"](ValueError("simulated failure"))
requests[-1]["on_finished"]()
assert "simulated failure" in options.preview_status.text()
assert options.test_button.isEnabled()
options.test_and_play()
requests[-1]["on_success"](b"audio")
requests[-1]["on_finished"]()
speech.play_audio.assert_called_once_with(b"audio")
assert "Playing Marin" in options.preview_status.text()
assert options.test_button.isEnabled()

def reject(operation, **callbacks):
    raise RuntimeError("profile closed")
speech.run_async_in_background_with_sentry = reject
options.test_and_play()
assert "profile closed" in options.preview_status.text()
assert options._preview_job is None
assert options.test_button.isEnabled()
assert not options.cancel_preview_button.isVisible()

options.state.update({"catalog_provider": "azure", "catalog_model": "neural", "catalog_voice": "saved"})
assert "unavailable" in options.selected_voice_label.text().lower()
assert not options.test_button.isEnabled()
assert not options.add_voice_button.isEnabled()
options.state.update({"catalog_provider": "openai", "catalog_model": "gpt-realtime-2.1-mini", "catalog_voice": "nova"})
assert "Choose a supported voice" in options.selected_voice_label.text()
assert not options.test_button.isEnabled()
assert not options.add_voice_button.isEnabled()
options.remove_selected_voice()
assert options.state.s["tts_voice_pool"] == []
assert "No voices yet" in options.pool_status.text()
assert not options.pool_table.isVisible()
assert not options.remove_voice_button.isEnabled()
scroll.close()

class RetiredDate(date):
    @classmethod
    def today(cls):
        return cls(2027, 1, 6)
models.date = RetiredDate
assert all(voice["model"] not in {"tts-1", "tts-1-hd"} for voice in speech.generate_openai_voices())
saved = {"provider": "openai", "model": "tts-1", "voice": "alloy", "enabled": True, "language": None}
retired = speech.TTSOptions({"tts_voice_pool": [saved]})
assert retired.state.s["tts_voice_pool"] == [saved]
assert "Choose a supported voice" in retired.selected_voice_label.text()
assert "unavailable" in retired.pool_table.item(0, 1).toolTip()
assert all(voice["model"] != "tts-1" for voice in retired.get_visible_voice_filters())
assert not retired.test_button.isEnabled() and not retired.add_voice_button.isEnabled()

voice_ids = ["a1b2c3d4aaaaaaaaffff", "d4c3b2a1bbbbbbbbaaaa"]
fish = speech.TTSOptions({"tts_voice_pool": [
    {"provider": "fish", "model": "s2.1-pro-free", "voice": voice, "enabled": True, "language": "Japanese"}
    for voice in voice_ids
]})
items = [[fish.pool_table.item(row, column) for column in range(3)] for row in range(2)]
assert items[0][1].text() != items[1][1].text()
for row, voice_id in enumerate(voice_ids):
    assert voice_id[:8] in items[row][1].text()
    assert "Custom voice (S2.1 Pro Free)" in items[row][1].text()
    assert voice_id in items[row][1].toolTip()
fish.pool_table.selectRow(0)
items[0][2].setText("ja")
items[0][0].setCheckState(Qt.CheckState.Unchecked)
assert fish.state.s["tts_voice_pool"][0]["language"] == "ja"
assert not fish.state.s["tts_voice_pool"][0]["enabled"]
for row in range(2):
    for column in range(3):
        assert fish.pool_table.item(row, column) is items[row][column]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_audio_preview_uses_temporary_files_and_only_cleans_its_own_playback(
    tmp_path: Path,
) -> None:
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from aqt import gui_hooks
from anki.sound import SoundOrVideoTag
import src.tts_utils as audio

folder = Path(sys.argv[1])
paths = []
def temp_file(prefix, suffix):
    path = folder / f"{prefix}{len(paths)}{suffix}"
    paths.append(path)
    return str(path)
audio.tmpfile = temp_file
player = SimpleNamespace(play_file_with_caller=Mock(), stop_and_clear_queue_if_caller=Mock())
audio.sound.av_player = player

audio.play_audio(b"first")
first = paths[-1]
assert first.read_bytes() == b"first"
player.play_file_with_caller.assert_called_with(str(first), first)
audio.audio_preview_did_begin(None, SoundOrVideoTag("unrelated.mp3"))
audio.audio_preview_did_end(None)
assert first.exists()
audio.audio_preview_did_begin(None, SoundOrVideoTag(str(first)))
audio.audio_preview_did_end(None)
assert not first.exists()
assert audio.preview_path is None
player.stop_and_clear_queue_if_caller.assert_not_called()

audio.play_audio(b"second")
second = paths[-1]
audio.audio_preview_did_begin(None, SoundOrVideoTag(str(second)))
audio.audio_preview_did_begin(None, SoundOrVideoTag("new-reviewer-audio.mp3"))
assert not second.exists()
player.stop_and_clear_queue_if_caller.assert_not_called()

audio.play_audio(b"third")
third = paths[-1]
audio.play_audio(b"RIFF0000WAVEfourth")
fourth = paths[-1]
assert fourth.suffix == ".wav"
assert not third.exists()
player.stop_and_clear_queue_if_caller.assert_called_with(third)
gui_hooks.profile_will_close()
assert not fourth.exists()
assert audio.preview_path is None
player.stop_and_clear_queue_if_caller.assert_called_with(fourth)
audio.audio_preview_did_begin(None, SoundOrVideoTag(str(fourth)))
audio.audio_preview_did_end(None)
assert audio.preview_path is None
assert not list(folder.iterdir())

player.play_file_with_caller.side_effect = RuntimeError("playback unavailable")
try:
    audio.play_audio(b"failed playback")
except RuntimeError:
    pass
else:
    raise AssertionError("Expected playback failure")
assert audio.preview_path is None
assert not list(folder.iterdir())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
