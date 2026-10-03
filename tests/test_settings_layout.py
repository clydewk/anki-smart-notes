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

SETTINGS_SETUP = """
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from PyQt6.QtCore import QPersistentModelIndex, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QLabel, QPushButton, QScrollArea
from anki.collection import Collection
from src.chatgpt_auth import ChatGPTAuth
from src.config import config, config_context
from src.constants import GLOBAL_DECK_ID
from src.models import DEFAULT_EXTRAS
import src.decks as decks_module
import src.notes as notes_module
import src.utils as utils_module
import src.ui.addon_options_dialog as options_module
import src.ui.chatgpt_account as account_module

app = QApplication([])
account_module.chatgpt_auth = ChatGPTAuth(Path(sys.argv[1]) / "chatgpt.json")
collection = Collection(str(Path(sys.argv[1]) / "layout.anki2"))
collection_window = SimpleNamespace(col=collection)
for module in [decks_module, notes_module, utils_module]:
    module.mw = collection_window
options_module.get_version = lambda: "test"
options_module.show_message_box = lambda *args, **kwargs: False
settings = json.loads(Path("config.json").read_text())
for key in settings:
    if key.endswith("api_key"):
        settings[key] = ""
settings.update(prompts_map={"note_types": {}}, custom_providers=[], mcp_servers=[])
config_context.set(settings)
def settle():
    for _ in range(3):
        app.processEvents()
"""


@pytest.mark.parametrize("size", [(800, 650), (640, 470)])
@pytest.mark.parametrize("point_size", [13, 18])
def test_every_settings_page_fits_native_viewport(
    tmp_path: Path, size: tuple[int, int], point_size: int
) -> None:
    script = (
        SETTINGS_SETUP
        + """
font = QFont(app.font())
font.setPointSize(int(sys.argv[4]))
app.setFont(font)
dialog = options_module.AddonOptionsDialog(MagicMock())
width, height = int(sys.argv[2]), int(sys.argv[3])
dialog.resize(width, height)
dialog.show()
settle()
failures = []
assert [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())] == [
    "Smart Fields", "Providers", "Tools", "Text", "Speech", "Images", "Generation"
]
buttons = dialog.findChild(QDialogButtonBox)
assert buttons is not None
for index in range(dialog.tabs.count()):
    dialog.tabs.setCurrentIndex(index)
    settle()
    name = dialog.tabs.tabText(index)
    if dialog.width() != width or dialog.height() != height:
        failures.append(f"{name}: requested {width}×{height}, got {dialog.width()}×{dialog.height()}")
    scroll = dialog.tabs.widget(index)
    assert isinstance(scroll, QScrollArea)
    if scroll.horizontalScrollBar().maximum() != 0:
        failures.append(f"{name}: horizontal page overflow {scroll.horizontalScrollBar().maximum()}px")
    for label in scroll.widget().findChildren(QLabel):
        if label.isVisible() and label.wordWrap() and label.height() < label.heightForWidth(label.width()):
            failures.append(f"{name}: clipped label {label.text()[:80]!r}, height {label.height()} needs {label.heightForWidth(label.width())}")
    for checkbox in scroll.widget().findChildren(QCheckBox):
        if checkbox.isVisible() and checkbox.width() < checkbox.minimumSizeHint().width():
            failures.append(f"{name}: clipped checkbox {checkbox.text()!r}, width {checkbox.width()} needs {checkbox.minimumSizeHint().width()}")
    for button in buttons.buttons():
        if not button.isVisible():
            continue
        corner = button.mapTo(dialog, button.rect().topLeft())
        if (corner.x() < 0 or corner.y() < 0 or corner.x() + button.width() > dialog.width()
                or corner.y() + button.height() > dialog.height()):
            failures.append(f"{name}: footer {button.text()!r} outside dialog")
        if button.width() < button.minimumSizeHint().width():
            failures.append(f"{name}: footer {button.text()!r} clipped")

assert dialog.table.isHidden()
assert "No Smart Fields yet" in dialog.table_empty.text()
assert dialog.table_count.text() == "0 Smart Fields"
assert dialog.mcp_selector.isHidden()
assert dialog.mcp_selector.currentIndex() == -1
assert not dialog.mcp_edit_button.isEnabled()
assert not dialog.mcp_remove_button.isEnabled()
assert not dialog.mcp_test_button.isEnabled()
assert dialog.custom_table.isHidden()
assert not dialog.custom_edit.isEnabled()
assert not dialog.custom_remove.isEnabled()
dialog.save_status.setText("Could not save settings. The draft is preserved; correct the reported problem and try again.")
dialog.save_status.show()
settle()
for button in buttons.buttons():
    corner = button.mapTo(dialog, button.rect().topLeft())
    if corner.y() + button.height() > dialog.height():
        failures.append(f"Save error: footer {button.text()!r} outside dialog")
dialog.reject()
collection.close()
assert not failures, "\\n".join(failures)
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), *map(str, size), str(point_size)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_settings_preserve_field_items_server_selection_and_filtered_targets(
    tmp_path: Path,
) -> None:
    script = (
        SETTINGS_SETUP
        + """
settings["prompts_map"] = {"note_types": {"Basic": {str(GLOBAL_DECK_ID): {
    "fields": {"Front": "alpha-token prompt", "Back": "beta-token prompt", "RemovedTarget": "removed-token prompt"},
    "extras": {field: deepcopy(DEFAULT_EXTRAS) for field in ("Front", "Back", "RemovedTarget")},
}}}}
settings["mcp_servers"] = [{
    "id": "native-test-server", "name": "Native test server", "enabled": True,
    "transport": "stdio", "command": "example-command", "args": [], "env": [],
    "env_passthrough": [], "cwd": "", "url": "", "headers": [], "header_env_vars": [],
}]
dialog = options_module.AddonOptionsDialog(MagicMock())
dialog.show()
settle()
assert dialog.table.columnCount() == 6
assert dialog.table.rowCount() == 3
assert dialog.mcp_selector.count() == 1
assert dialog.mcp_selector.currentIndex() == -1
assert not dialog.table.isHidden()
assert not dialog.mcp_selector.isHidden()
assert dialog.table_empty.isHidden()
assert dialog.mcp_empty_label.isHidden()
dialog.mcp_selector.setCurrentIndex(0)
settle()
assert dialog._selected_mcp_server_id() == "native-test-server"
field_items = [[dialog.table.item(row, column) for column in range(dialog.table.columnCount())]
               for row in range(dialog.table.rowCount())]
server_index = QPersistentModelIndex(dialog.mcp_selector.model().index(0, 0))
assert server_index.data(Qt.ItemDataRole.UserRole) == "native-test-server"
for row in field_items:
    assert all(item is not None and item.toolTip() for item in row)
for update in ({"debug": True}, {"openai_daily_token_budget": 123456}, {"openai_api_key": "draft-only-key"}):
    dialog.state.update(update)
    settle()
    for row_index, row in enumerate(field_items):
        for column, item in enumerate(row):
            assert dialog.table.item(row_index, column) is item, "Unrelated settings destroyed a field-table item"
    assert server_index.isValid(), "Unrelated settings replaced a server-selector item"
    assert server_index.data(Qt.ItemDataRole.UserRole) == "native-test-server"
    assert dialog._selected_mcp_server_id() == "native-test-server"
    assert "example-command" in dialog.mcp_details.text()

first_server = deepcopy(dialog.state.s["mcp_servers"][0])
second_server = {**first_server, "id": "second-server", "name": "Second server", "command": "another-command"}
dialog.state.update({"mcp_servers": [second_server, first_server]})
settle()
assert dialog.mcp_selector.currentIndex() == 1
assert dialog._selected_mcp_server_id() == "native-test-server"
assert "example-command" in dialog.mcp_details.text()

back_row = next(row for row in range(dialog.table.rowCount()) if dialog.table.item(row, 2).text() == "Back")
dialog.state.update({"search_text": "beta-token", "selected_row": None})
settle()
assert dialog.table.rowCount() == 3
assert [row for row in range(3) if not dialog.table.isRowHidden(row)] == [back_row]
assert dialog.table_count.text() == "1 of 3 Smart Fields"
assert dialog.table.item(back_row, 2) is field_items[back_row][2]
dialog.table.selectRow(back_row)
settle()
assert dialog.state.s["selected_row"] == back_row
assert dialog.edit_button.isEnabled() and dialog.remove_button.isEnabled()
opened = []
errors = []
sys.excepthook = lambda _kind, error, _traceback: errors.append(error)
def prompt_dialog(*args, **kwargs):
    opened.append({**kwargs, "draft_key": config.openai_api_key})
    return SimpleNamespace(exec=lambda: QDialog.DialogCode.Rejected)
options_module.PromptDialog = prompt_dialog
add_button = next(button for button in dialog.findChildren(QPushButton) if button.accessibleName() == "Add Smart Field")
actions = add_button.menu().actions()
assert [action.text() for action in actions] == ["Text", "Speech", "Image"]
for action, field_type in zip(actions, ("chat", "tts", "image"), strict=True):
    previous_count = len(opened)
    action.trigger()
    settle()
    assert not errors, errors
    assert len(opened) == previous_count + 1
    assert opened[-1]["field_type"] == field_type
    assert opened[-1]["deck_id"] == GLOBAL_DECK_ID
    assert opened[-1]["parent"] is dialog
    assert opened[-1]["draft_key"] == "draft-only-key"
dialog.edit_button.click()
settle()
assert not errors, errors
assert opened[-1]["field"] == "Back"
assert opened[-1]["prompt"] == "beta-token prompt"
assert opened[-1]["deck_id"] == GLOBAL_DECK_ID
assert dialog.table.item(back_row, 2) is field_items[back_row][2]

dialog.state.update({"search_text": "no-matching-record", "selected_row": None})
settle()
assert dialog.table.isHidden()
assert "matching" in dialog.table_empty.text().lower()
assert dialog.table_count.text() == "0 of 3 Smart Fields"
assert not dialog.edit_button.isEnabled() and not dialog.remove_button.isEnabled()
assert dialog.state.s["selected_row"] is None
dialog.state.update({"search_text": "", "selected_row": None})
settle()
assert not dialog.table.isHidden()
assert all(not dialog.table.isRowHidden(row) for row in range(3))
assert dialog.table_empty.isHidden()
assert dialog.table.item(back_row, 2) is field_items[back_row][2]
missing_row = next(row for row in range(3) if dialog.table.item(row, 2).text() == "RemovedTarget")
assert "repair" in dialog.table.item(missing_row, 5).text().lower()
dialog.state.update({"mcp_servers": [second_server]})
settle()
assert dialog.mcp_selector.currentIndex() == -1
assert dialog._selected_mcp_server_id() is None
assert not dialog.mcp_edit_button.isEnabled()
assert not dialog.mcp_test_button.isEnabled()
dialog.reject()
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
