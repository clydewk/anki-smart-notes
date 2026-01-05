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

import html
import re
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse

from aqt import (
    QAction,
    QApplication,
    QDesktopServices,
    QDialog,
    QDialogButtonBox,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPoint,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QUrl,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from ..config import config
from ..constants import GLOBAL_DECK_ID
from ..decks import deck_id_to_name_map, deck_name_to_id_map
from ..logger import logger
from ..models import (
    CustomProvider,
    PromptMap,
    SmartFieldType,
)
from ..note_proccessor import NoteProcessor
from ..prompts import get_all_prompts, get_extras, get_prompts_for_note, remove_prompt
from ..utils import get_fields, get_version
from .chat_options import ChatOptions
from .custom_provider_dialog import CustomProviderDialog
from .image_options import ImageOptions
from .prompt_dialog import PromptDialog
from .reactive_check_box import ReactiveCheckBox
from .reactive_line_edit import ReactiveLineEdit
from .state_manager import StateManager
from .tts_options import TTSOptions
from .ui_utils import default_form_layout, font_large, font_small, show_message_box

OPTIONS_MIN_WIDTH = 875
TTS_PROMPT_STUB_VALUE = "🔈"


def get_item_text(item: QTableWidgetItem) -> str:
    val = item.data(Qt.ItemDataRole.UserRole)
    if val:
        return str(val)
    return item.text()


def highlight_match(text: str, query: str) -> str:
    if not query:
        return html.escape(text)

    pattern = re.compile(f"({re.escape(query)})", re.IGNORECASE)
    parts = pattern.split(text)

    result = []
    for i, part in enumerate(parts):
        escaped_part = html.escape(part)
        if i % 2 == 1:
            result.append(
                f'<span style="background-color: rgba(255, 213, 79, 0.4); font-weight: bold;">{escaped_part}</span>'
            )
        else:
            result.append(escaped_part)

    return "".join(result)


class State(TypedDict):
    prompts_map: PromptMap
    selected_row: Optional[int]
    generate_at_review: bool
    regenerate_notes_when_batching: bool
    openai_endpoint: Optional[str]
    allow_empty_fields: bool
    debug: bool

    # API Keys
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    deepseek_api_key: Optional[str]
    google_api_key: Optional[str]
    elevenlabs_api_key: Optional[str]
    replicate_api_key: Optional[str]

    custom_providers: list[CustomProvider]
    search_text: str


class AddonOptionsDialog(QDialog):
    table_buttons: QHBoxLayout
    remove_button: QPushButton
    table: QTableWidget
    restore_defaults: QPushButton
    edit_button: QPushButton
    state: StateManager[State]
    save_timer: QTimer

    def __init__(self, processor: NoteProcessor):
        super().__init__()
        self.processor = processor
        self.state = StateManager[State](self.make_initial_state())
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(500)
        self.save_timer.timeout.connect(lambda: self.write_config(silent=True))
        self.setup_ui()

    def setup_ui(self) -> None:
        self.setWindowTitle("Smart Notes ✨")
        self.setMinimumWidth(OPTIONS_MIN_WIDTH)

        standard_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.restore_defaults = QPushButton("Restore Defaults")
        standard_buttons.addButton(
            self.restore_defaults, QDialogButtonBox.ButtonRole.ResetRole
        )
        self.restore_defaults.clicked.connect(self.on_restore_defaults)

        standard_buttons.accepted.connect(self.on_accept)
        standard_buttons.rejected.connect(self.on_reject)

        # Set up layout

        tabs = QTabWidget()

        tabs.addTab(self.render_general_tab(), "General")
        tabs.addTab(self.render_providers_tab(), "Providers")
        tabs.addTab(self.render_chat_tab(), "Text")
        self.tts_tab = self.render_tts_tab()
        tabs.addTab(self.tts_tab, "TTS")
        self.images_tab = self.render_images_tab()
        tabs.addTab(self.images_tab, "Images")
        tabs.addTab(self.render_plugin_tab(), "Advanced")

        tab_layout = QVBoxLayout()

        if not config.did_click_rate_link:
            rate_box = QWidget()
            rate_layout = QHBoxLayout()
            rate_box.setLayout(rate_layout)
            rate_label = QLabel(
                'Enjoying Smart Notes? Please consider <a href="https://ankiweb.net/shared/info/1531888719">leaving a review.</a>'
            )
            rate_label.setContentsMargins(0, 12, 0, 18)
            rate_font = rate_label.font()
            rate_font.setItalic(True)
            rate_label.setFont(rate_font)
            rate_layout.addStretch()
            rate_layout.addWidget(rate_label)
            rate_layout.addStretch()

            def on_rate_click(url: str):
                QDesktopServices.openUrl(QUrl(url))
                config.did_click_rate_link = True

            rate_label.linkActivated.connect(on_rate_click)
            tab_layout.addWidget(rate_box)
        tab_layout.addWidget(tabs)

        # Version Box

        version_box = QWidget()
        version_box_layout = QHBoxLayout()
        version_box_layout.setContentsMargins(0, 0, 12, 0)
        version_box.setLayout(version_box_layout)
        support_label = QLabel(
            "Found a bug or have a feature request? <a href='https://github.com/piazzatron/anki-smart-notes/issues'>Create an issue on Github</a> or email <a href='mailto:support@smart-notes.xyz'>support@smart-notes.xyz</a>."
        )
        support_label.setFont(font_small)
        support_label.setOpenExternalLinks(True)
        version_label = QLabel(f"Smart Notes v{get_version()}")
        version_label.setFont(font_small)
        version_box_layout.addWidget(support_label)
        version_box_layout.addStretch()
        version_box_layout.addWidget(version_label)

        opacity_effect = QGraphicsOpacityEffect()
        opacity_effect.setOpacity(0.3)
        opacity_effect2 = QGraphicsOpacityEffect()
        opacity_effect2.setOpacity(0.7)

        version_label.setGraphicsEffect(opacity_effect)
        support_label.setGraphicsEffect(opacity_effect2)

        tab_layout.addWidget(version_box)

        tab_layout.addSpacing(12)
        tab_layout.addWidget(standard_buttons)

        self.setLayout(tab_layout)
        self.state.state_changed.connect(self.render_ui)
        self.state.state_changed.connect(self.on_state_changed)

        # Connect substates for auto-save
        self.tts_options.state.state_changed.connect(self.on_state_changed)
        self.chat_options.state.state_changed.connect(self.on_state_changed)
        self.image_options.state.state_changed.connect(self.on_state_changed)

        self.render_ui()

    def on_state_changed(self) -> None:
        self.save_timer.start()

    def render_general_tab(self) -> QWidget:
        layout = QVBoxLayout()

        # Header
        header_layout = QVBoxLayout()
        title = QLabel("<h3>✨ Smart Fields</h3>")
        explanation = QLabel(
            "Automatically generate text, voice, and images on any field."
        )
        explanation.setFont(font_small)
        header_layout.addWidget(title)
        header_layout.addWidget(explanation)
        layout.addLayout(header_layout)
        layout.addSpacing(12)

        # Search Bar
        search_layout = QHBoxLayout()
        search_input = ReactiveLineEdit(self.state, "search_text")
        search_input.setPlaceholderText("🔎 Search fields...")
        search_input.on_change.connect(
            lambda text: self.state.update({"search_text": text})
        )
        search_layout.addWidget(search_input)
        layout.addLayout(search_layout)

        # Table
        self.table = self.create_table()
        self.setup_table_context_menu(self.table)
        layout.addWidget(self.table)

        # Buttons
        buttons_layout = QHBoxLayout()

        # Left side: Edit/Remove
        self.edit_button = QPushButton("Edit")
        self.edit_button.setFixedWidth(80)
        self.edit_button.clicked.connect(self.on_edit)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setFixedWidth(80)
        self.remove_button.clicked.connect(self.on_remove)

        buttons_layout.addWidget(self.edit_button)
        buttons_layout.addWidget(self.remove_button)

        buttons_layout.addStretch()

        # Right side: Add buttons
        add_text = QPushButton("💬 New Text Field")
        add_text.clicked.connect(lambda _: self.on_add("chat"))

        add_tts = QPushButton("🔈 New TTS Field")
        add_tts.clicked.connect(lambda _: self.on_add("tts"))

        add_image = QPushButton("🖼️ New Image Field")
        add_image.clicked.connect(lambda _: self.on_add("image"))

        buttons_layout.addWidget(add_image)
        buttons_layout.addWidget(add_tts)
        buttons_layout.addWidget(add_text)

        layout.addLayout(buttons_layout)

        container = QWidget()
        container.setLayout(layout)
        return container

    def setup_table_context_menu(self, table: QTableWidget) -> None:
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self.show_table_context_menu)

    def show_table_context_menu(self, pos: QPoint):
        row = self.table.rowAt(pos.y())
        if row >= 0:
            prompt_item = self.table.item(row, 4)
            prompt_text = get_item_text(prompt_item) if prompt_item else ""

            if prompt_text:
                menu = QMenu(self)
                copy_action = QAction("Copy Prompt", self)

                menu.addAction(copy_action)

                action = menu.exec(self.table.mapToGlobal(pos))
                if action == copy_action:
                    clipboard = QApplication.clipboard()
                    if clipboard:
                        clipboard.setText(prompt_text)

    def render_providers_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)

        form = default_form_layout()

        def add_key_field(label: str, key: str, placeholder: str):
            edit = ReactiveLineEdit(self.state, key)  # type: ignore
            edit.setPlaceholderText(placeholder)
            edit.setMinimumWidth(400)
            edit.on_change.connect(
                lambda text, key=key: self._on_api_key_change(key, text)
            )
            form.addRow(f"<b>{label}:</b>", edit)

        add_key_field("🔑 OpenAI API Key", "openai_api_key", "sk-proj-...")
        add_key_field("🔑 Anthropic API Key", "anthropic_api_key", "sk-ant-...")
        add_key_field("🔑 DeepSeek API Key", "deepseek_api_key", "sk-...")
        add_key_field("🔑 Google API Key (Gemini/TTS)", "google_api_key", "AIzaSy...")
        add_key_field("🔑 ElevenLabs API Key (TTS)", "elevenlabs_api_key", "...")
        add_key_field("🔑 Replicate API Key (Images)", "replicate_api_key", "r8_...")

        group_box = QGroupBox("API Configuration")
        group_box.setLayout(form)
        layout.addWidget(group_box)

        # Custom Providers
        custom_box = QGroupBox("Custom Providers (OpenAI Compatible)")
        custom_layout = QVBoxLayout()
        custom_box.setLayout(custom_layout)

        self.custom_table = QTableWidget(0, 4)
        self.custom_table.setHorizontalHeaderLabels(
            ["Name", "Base URL", "Capabilities", "API Key"]
        )
        header = self.custom_table.horizontalHeader()
        if header:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Populate table
        for provider in self.state.s["custom_providers"]:
            self._add_custom_provider_row(provider)

        # Make table read-only for cells, we use dialog to edit
        self.custom_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.custom_table.itemDoubleClicked.connect(self._on_edit_custom_provider)

        btns = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_custom_provider)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(lambda: self._on_edit_custom_provider(None))

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_custom_provider)

        btns.addWidget(add_btn)
        btns.addWidget(edit_btn)
        btns.addWidget(remove_btn)
        btns.addStretch()

        custom_layout.addWidget(self.custom_table)
        custom_layout.addLayout(btns)

        layout.addWidget(custom_box)

        layout.addStretch()

        return container

    def _add_custom_provider_row(self, provider: CustomProvider) -> None:
        row = self.custom_table.rowCount()
        self.custom_table.insertRow(row)

        name = QTableWidgetItem(provider["name"])
        url = QTableWidgetItem(provider["base_url"])

        caps = provider.get("capabilities", [])
        caps_display = ", ".join(caps) if caps else "chat"
        capabilities = QTableWidgetItem(caps_display)

        # key = QTableWidgetItem(provider["api_key"]) # Hide key or mask it?
        key = QTableWidgetItem("********" if provider["api_key"] else "")

        # Store full provider data in the name item user data
        name.setData(Qt.ItemDataRole.UserRole, provider)

        self.custom_table.setItem(row, 0, name)
        self.custom_table.setItem(row, 1, url)
        self.custom_table.setItem(row, 2, capabilities)
        self.custom_table.setItem(row, 3, key)

    def _on_add_custom_provider(self) -> None:
        dialog = CustomProviderDialog(parent=self)
        if dialog.exec():
            provider = dialog.get_provider()
            self._add_custom_provider_row(provider)
            self._save_custom_providers()

    def _on_edit_custom_provider(self, item: Optional[QTableWidgetItem]) -> None:
        row = self.custom_table.currentRow()
        if row < 0:
            return

        name_item = self.custom_table.item(row, 0)
        provider_data = name_item.data(Qt.ItemDataRole.UserRole)

        dialog = CustomProviderDialog(provider=provider_data, parent=self)
        if dialog.exec():
            new_provider = dialog.get_provider()

            # Update row
            self.custom_table.item(row, 0).setText(new_provider["name"])
            self.custom_table.item(row, 0).setData(
                Qt.ItemDataRole.UserRole, new_provider
            )
            self.custom_table.item(row, 1).setText(new_provider["base_url"])

            caps = new_provider.get("capabilities", [])
            caps_display = ", ".join(caps) if caps else "chat"
            self.custom_table.item(row, 2).setText(caps_display)

            self.custom_table.item(row, 3).setText(
                "********" if new_provider["api_key"] else ""
            )

            self._save_custom_providers()

    def _on_remove_custom_provider(self) -> None:
        row = self.custom_table.currentRow()
        if row >= 0:
            self.custom_table.removeRow(row)
            self._save_custom_providers()

    def _save_custom_providers(self) -> None:
        providers: list[CustomProvider] = []
        for i in range(self.custom_table.rowCount()):
            item = self.custom_table.item(i, 0)
            provider = item.data(Qt.ItemDataRole.UserRole)
            providers.append(provider)

        self.state.update({"custom_providers": providers})

        # Ensure config is updated immediately so tabs can read it
        config.custom_providers = providers

        # Refresh tabs
        if hasattr(self, "chat_options"):
            self.chat_options.refresh_custom_providers()
        if hasattr(self, "tts_options"):
            self.tts_options.refresh_custom_providers()
        if hasattr(self, "image_options"):
            self.image_options.refresh_custom_providers()

    def render_ui(self) -> None:
        self.render_table()
        self.render_buttons()

    def render_table(self) -> None:
        self.table.setRowCount(0)
        search_text = self.state.s.get("search_text", "").lower()

        row = 0
        all_prompts = get_all_prompts(override_prompts_map=self.state.s["prompts_map"])
        for note_type, deck_prompts in all_prompts.items():
            for deck_id, field_prompts in deck_prompts.items():
                for field, prompt in field_prompts.items():
                    # Filter
                    deck_name = deck_id_to_name_map().get(deck_id) or ""

                    if search_text:
                        searchable = f"{note_type} {deck_name} {field} {prompt}".lower()
                        if search_text not in searchable:
                            continue

                    extras = get_extras(
                        note_type=note_type, field=field, deck_id=deck_id
                    )

                    if not extras:
                        continue

                    if not deck_name:
                        continue

                    type = extras["type"]
                    self.table.insertRow(self.table.rowCount())
                    items = [
                        QTableWidgetItem(note_type),
                        QTableWidgetItem(deck_name),
                        QTableWidgetItem(field),
                        QTableWidgetItem(
                            {"chat": "💬", "tts": "🔈", "image": "🖼️"}[type]
                        ),
                        QTableWidgetItem(
                            {
                                "chat": f"{prompt}",
                                "tts": f"{prompt}",
                                "image": f"{prompt}",
                            }[type]
                        ),
                    ]
                    enabled = extras["automatic"]
                    always_overwrite = extras.get("regenerate_when_batching", False)
                    for i, item in enumerate(items):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                        )

                        if always_overwrite:
                            item.setBackground(QColor(255, 193, 7, 50))
                            current_tooltip = item.toolTip()
                            item.setToolTip(
                                f"{current_tooltip}\n(Always overwrites)"
                                if current_tooltip
                                else "(Always overwrites)"
                            )

                        self.table.setItem(row, i, item)

                        if search_text and i in [0, 1, 2, 4]:
                            text = item.text()
                            # Always use QLabel for consistency in search results
                            label = QLabel(highlight_match(text, search_text))
                            style = "background-color: transparent;"
                            if not enabled:
                                style += " color: #a0a0a0;"
                            label.setStyleSheet(style)
                            label.setAlignment(
                                Qt.AlignmentFlag.AlignLeft
                                | Qt.AlignmentFlag.AlignVCenter
                            )
                            label.setAttribute(
                                Qt.WidgetAttribute.WA_TransparentForMouseEvents
                            )
                            label.setIndent(3)
                            label.setToolTip(text)
                            self.table.setCellWidget(row, i, label)
                            item.setData(Qt.ItemDataRole.UserRole, text)
                            item.setText("")

                        if not enabled and not self.table.cellWidget(row, i):
                            item.setForeground(Qt.GlobalColor.lightGray)
                    row += 1

        # Ensure the correct row is always selected
        # shouldn't need the second and condition, but defensive
        selected_row = self.state.s["selected_row"]
        if selected_row is not None and selected_row < self.table.rowCount():
            self.table.selectRow(selected_row)

    def render_plugin_tab(self) -> QWidget:
        plugin_box = QGroupBox("✨Smart Field Generation")
        plugin_form = default_form_layout()
        plugin_box.setLayout(plugin_form)

        # Generate at review
        self.generate_at_review_button = ReactiveCheckBox(
            self.state, "generate_at_review"
        )

        plugin_form.addRow(
            "Generate fields during review:", self.generate_at_review_button
        )
        plugin_form.addRow("", QLabel(""))

        # Regenerate when batching
        self.regenerate_notes_when_batching = ReactiveCheckBox(
            self.state, "regenerate_notes_when_batching"
        )
        plugin_form.addRow(
            "Regenerate all smart fields when batch processing:",
            self.regenerate_notes_when_batching,
        )
        regenerate_info = QLabel(
            "When batch processing a group of notes, whether to regenerate all smart fields from scratch, or only generate empty ones."
        )
        regenerate_info.setFont(font_small)
        plugin_form.addRow(regenerate_info)
        plugin_form.addRow("", QLabel(""))

        self.allow_empty_fields_box = ReactiveCheckBox(self.state, "allow_empty_fields")
        plugin_form.addRow(
            "Generate prompts with some blank fields:", self.allow_empty_fields_box
        )
        empty_fields_info = QLabel(
            "Generate even if the prompt references some blank fields. Prompts referencing *only* blank fields are never generated."
        )
        empty_fields_info.setFont(font_small)
        plugin_form.addRow(empty_fields_info)

        plugin_tab_layout = default_form_layout()
        plugin_tab_layout.addRow(plugin_box)

        self.debug_checkbox = ReactiveCheckBox(self.state, "debug")
        plugin_tab_layout.addRow(QLabel(""))
        plugin_tab_layout.addRow("Debug mode", self.debug_checkbox)

        plugin_settings_tab = QWidget()
        plugin_settings_tab.setLayout(plugin_tab_layout)

        return plugin_settings_tab

    def render_chat_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        layout.setContentsMargins(24, 24, 24, 24)
        self.chat_options = ChatOptions()
        self.chat_options.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        expl = QLabel("Configure default settings for text Smart Fields.")
        subExpl = QLabel("These settings can be further customized for each field.")
        expl.setFont(font_large)
        subExpl.setFont(font_small)
        layout.addWidget(expl)
        layout.addWidget(subExpl)
        layout.addItem(
            QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )
        layout.addWidget(self.chat_options)
        return container

    def render_tts_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        layout.setContentsMargins(24, 24, 24, 24)
        self.tts_options = TTSOptions()
        self.tts_options.setContentsMargins(0, 0, 0, 0)

        expl = QLabel("Configure default voice settings for TTS.")
        subExpl = QLabel("These settings can be overridden on a per-field basis.")
        expl.setFont(font_large)
        subExpl.setFont(font_small)
        layout.addWidget(expl)
        layout.addWidget(subExpl)
        layout.addItem(
            QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )
        layout.addWidget(self.tts_options)
        return container

    def render_images_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        layout.setContentsMargins(24, 24, 24, 24)
        self.image_options = ImageOptions()
        self.image_options.setContentsMargins(0, 0, 0, 0)

        expl = QLabel("Configure default settings for image generation.")
        subExpl = QLabel("These settings can be overridden on a per-field basis.")
        expl.setFont(font_large)
        subExpl.setFont(font_small)
        layout.addWidget(expl)
        layout.addWidget(subExpl)
        layout.addItem(
            QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        )
        layout.addWidget(self.image_options)
        layout.addItem(
            QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        return container

    def create_table(self) -> QTableWidget:
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(
            ["Note Type", "Deck", "Target Field", "Type", "Prompt"]
        )

        # Selection
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)

        # Styling
        table.horizontalHeader().setStretchLastSection(True)  # type: ignore
        table.verticalHeader().setVisible(False)  # type: ignore

        # Wire up slots
        table.currentItemChanged.connect(self.on_row_selected)
        table.itemDoubleClicked.connect(self.on_edit)

        return table

    def on_row_selected(self, current: Optional[QTableWidgetItem]) -> None:
        if current:
            self.state.update({"selected_row": current.row()})

    def on_edit(self, _) -> None:
        row = self.state.s["selected_row"]
        if row is None:
            return

        note_type = get_item_text(self.table.item(row, 0))  # type: ignore
        deck_id = deck_name_to_id_map()[get_item_text(self.table.item(row, 1))]  # type: ignore
        field = get_item_text(self.table.item(row, 2))  # type: ignore
        logger.debug(f"Editing {note_type}, {field}")

        # Get type
        extras = get_extras(note_type=note_type, field=field, deck_id=deck_id)
        if not extras:
            return
        field_type = extras["type"]

        prompts = get_prompts_for_note(
            note_type=note_type,
            to_lower=True,
            deck_id=deck_id,
            fallback_to_global_deck=False,
        )

        all_fields = get_fields(note_type)

        if not prompts or not len(all_fields) or field not in all_fields:
            show_message_box("Note type does not exist or field not in note type!")
            return

        prompt_dialog = PromptDialog(
            self.state.s["prompts_map"],
            self.processor,
            self.on_update_prompts,
            card_type=note_type,
            deck_id=deck_id,
            field=field,
            field_type=field_type,
            prompt=prompts[field.lower()],
            parent=self,
        )

        if prompt_dialog.exec() == QDialog.DialogCode.Accepted:
            self.render_table()

    def render_buttons(self) -> None:
        is_enabled = self.state.s["selected_row"] is not None
        self.remove_button.setEnabled(is_enabled)
        self.edit_button.setEnabled(is_enabled)

    def on_add(self, field_type: SmartFieldType) -> None:
        prompt_dialog = PromptDialog(
            self.state.s["prompts_map"],
            self.processor,
            self.on_update_prompts,
            field_type=field_type,
            deck_id=GLOBAL_DECK_ID,
            parent=self,
        )

        if prompt_dialog.exec() == QDialog.DialogCode.Accepted:
            self.render_table()

    def on_remove(self):
        row = self.state.s["selected_row"]
        if row is None:
            # Should never happen
            return

        note_type = get_item_text(self.table.item(row, 0))  # type: ignore
        deck_id = deck_name_to_id_map()[get_item_text(self.table.item(row, 1))]  # type: ignore
        field = get_item_text(self.table.item(row, 2))  # type: ignore
        new_map = remove_prompt(
            self.state.s["prompts_map"],
            note_type=note_type,
            deck_id=deck_id,
            field=field,
        )

        self.state.update({"prompts_map": new_map, "selected_row": None})

    def on_accept(self) -> None:
        if self.write_config():
            self.accept()

    def on_reject(self) -> None:
        self.reject()

    def write_config(self, silent: bool = False) -> bool:
        logger.debug("Writing config")
        if config.openai_endpoint and not is_valid_url(config.openai_endpoint):
            if not silent:
                show_message_box("Invalid OpenAI Host", "Please provide a valid URL.")
            return False

        if (
            self.tts_options.state.s["tts_provider"] == "elevenLabs"
            and config.tts_provider != "elevenLabs"
        ):
            if not silent:
                did_click_ok = show_message_box(
                    "Are you sure you want to set your default voice provider to a premium model?",
                    show_cancel=True,
                )
                if not did_click_ok:
                    return False
            # If silent (auto-save), we skip the check/dialog to avoid interruption?
            # Or we strictly don't save if check fails?
            # For now let's allow saving in silent mode without dialog to avoid annoyance,
            # assuming user knows what they are doing if they selected it.
            # OR better: only show dialog if it wasn't already ElevenLabs.
            # But here we are checking against `config.tts_provider`.
            pass

        valid_config_attrs = config.__annotations__.keys()

        old_debug = config.debug

        # Automatically inspect all the substates for valid config and write them out
        states: list[StateManager[Any]] = [
            self.state,
            self.tts_options.state,
            self.chat_options.state,
            self.image_options.state,
        ]
        for state in states:
            for k, v in [
                item for item in state.s.items() if item[0] in valid_config_attrs
            ]:
                logger.debug(f"Setting: {k}: {v}")
                config.__setattr__(k, v)

        if not old_debug and self.state.s["debug"] and not silent:
            show_message_box("Debug mode enabled. Please restart Anki.")

        return True

    def _on_api_key_change(self, key: str, value: str) -> None:
        self.state.update({key: value})
        try:
            setattr(config, key, value)
        except Exception as e:
            logger.error(f"Error persisting {key}: {e}")

    def on_update_prompts(self, prompts_map: PromptMap) -> None:
        self.state.update({"prompts_map": prompts_map})
        self.write_config()

    def make_initial_state(self) -> State:
        return {
            "openai_api_key": config.openai_api_key,
            "anthropic_api_key": config.anthropic_api_key,
            "deepseek_api_key": config.deepseek_api_key,
            "google_api_key": config.google_api_key,
            "elevenlabs_api_key": config.elevenlabs_api_key,
            "replicate_api_key": config.replicate_api_key,
            "prompts_map": config.prompts_map,
            "selected_row": None,
            "generate_at_review": config.generate_at_review,
            "regenerate_notes_when_batching": config.regenerate_notes_when_batching,
            "openai_endpoint": config.openai_endpoint,
            "allow_empty_fields": config.allow_empty_fields,
            "debug": config.debug,
            "custom_providers": config.custom_providers or [],
            "search_text": "",
        }

    def on_restore_defaults(self) -> None:
        config.restore_defaults()
        self.state.update(self.make_initial_state())  # type: ignore


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])
