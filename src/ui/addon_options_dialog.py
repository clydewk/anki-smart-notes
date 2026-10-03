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

import json
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, cast
from urllib.parse import urlparse

from anki.decks import DeckId
from aqt import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDesktopServices,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPoint,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QUrl,
    QVBoxLayout,
    QWidget,
    mw,
)
from PyQt6 import sip
from PyQt6.QtCore import Qt

from ..built_in_tools import list_built_in_tools
from ..chat_usage import chat_usage_tracker
from ..config import Config, config
from ..constants import GLOBAL_DECK_ID
from ..decks import deck_id_to_name_map, deck_name_to_id_map
from ..job_context import JobCancelledError
from ..logger import logger, redact_text
from ..mcp_runtime import McpServerProbeResult, mcp_runtime
from ..models import (
    BuiltInToolsConfig,
    CustomProvider,
    McpServerConfig,
    PromptMap,
    SmartFieldType,
    normalize_built_in_tools_config,
)
from ..note_processor import NoteProcessor
from ..prompt_io import (
    ImportPreview,
    PromptImportError,
    create_prompt_export,
    preview_prompt_import,
)
from ..prompts import get_all_prompts, get_extras, get_prompts_for_note, remove_prompt
from ..sentry import run_async_in_background_with_sentry
from ..settings import with_widget_settings
from ..utils import get_fields, get_file_path, get_version
from .chat_options import ChatOptions
from .chatgpt_account import ChatGPTAccountWidget
from .custom_provider_dialog import CustomProviderDialog
from .image_options import ImageOptions
from .mcp_server_dialog import MCP_DOCS_URL, McpServerDialog
from .prompt_dialog import PromptDialog
from .reactive_check_box import ReactiveCheckBox
from .reactive_line_edit import ReactiveLineEdit
from .reactive_spin_box import ReactiveSpinBox
from .state_manager import StateManager
from .tts_options import TTSOptions
from .ui_utils import (
    bound_dialog,
    default_form_layout,
    font_large,
    scroll_page,
    show_message_box,
)

if TYPE_CHECKING:
    from ..tasks import JobHandle


def get_item_text(item: QTableWidgetItem) -> str:
    val = item.data(Qt.ItemDataRole.UserRole)
    if val:
        return str(val)
    return item.text()


class State(TypedDict):
    prompts_map: PromptMap
    selected_row: int | None
    generate_at_review: bool
    regenerate_notes_when_batching: bool
    openai_endpoint: str | None
    allow_empty_fields: bool
    debug: bool
    openai_fast_mode_enabled: bool
    openai_daily_token_budget_enabled: bool
    openai_daily_token_budget: int

    # API Keys
    openai_api_key: str | None
    anthropic_api_key: str | None
    deepseek_api_key: str | None
    google_api_key: str | None
    elevenlabs_api_key: str | None
    fish_api_key: str | None
    replicate_api_key: str | None

    custom_providers: list[CustomProvider]
    mcp_servers: list[McpServerConfig]
    built_in_tools: BuiltInToolsConfig
    search_text: str


class AddonOptionsDialog(QDialog):
    remove_button: QPushButton
    table: QTableWidget
    restore_defaults: QPushButton
    edit_button: QPushButton
    import_button: QPushButton
    export_button: QPushButton
    state: StateManager[State]

    def __init__(self, processor: NoteProcessor):
        super().__init__(mw)
        self.processor = processor
        self.initial_settings = config.snapshot()
        self._rendered_state: State | None = None
        self._mcp_job: JobHandle | None = None
        self.state = StateManager[State](self.make_initial_state())
        self.setup_ui()
        self.finished.connect(self.chatgpt_account.cancel_sign_in)

    def setup_ui(self) -> None:
        self.setWindowTitle("Smart Notes Settings")
        self.scope_label = QLabel()
        self.scope_label.setWordWrap(True)
        self.scope_label.setTextFormat(Qt.TextFormat.PlainText)
        self.tabs = QTabWidget()
        self.tabs.addTab(scroll_page(self.render_general_tab()), "Smart Fields")
        self.tabs.addTab(scroll_page(self.render_providers_tab()), "Providers")
        self.tabs.addTab(scroll_page(self.render_tools_tab()), "Tools")
        self.tabs.addTab(scroll_page(self.render_chat_tab()), "Text")
        self.tts_tab = self.render_tts_tab()
        self.tabs.addTab(scroll_page(self.tts_tab), "Speech")
        self.images_tab = self.render_images_tab()
        self.tabs.addTab(scroll_page(self.images_tab), "Images")
        self.tabs.addTab(scroll_page(self.render_plugin_tab()), "Generation")
        self.tabs.currentChanged.connect(self._update_scope)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Help
        )
        self.restore_defaults = QPushButton("Restore Defaults")
        self.restore_defaults.setToolTip(
            "Reset ordinary options in this draft. Keep prompts, keys, custom providers and MCP servers."
        )
        buttons.addButton(self.restore_defaults, QDialogButtonBox.ButtonRole.ResetRole)
        self.restore_defaults.clicked.connect(self.on_restore_defaults)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.on_reject)
        buttons.helpRequested.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://github.com/clydewk/anki-smart-notes/issues")
            )
        )
        self.save_status = QLabel()
        self.save_status.setWordWrap(True)
        self.save_status.setTextFormat(Qt.TextFormat.PlainText)
        self.save_status.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        layout.addWidget(self.scope_label)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.save_status)
        layout.addWidget(buttons)
        self.state.state_changed.connect(self.render_ui)
        self.render_ui()
        self._update_scope()
        bound_dialog(self)

    def _update_scope(self) -> None:
        profile = mw.pm.name if mw else "current profile"
        local = self.tabs.currentIndex() in {0, 6}
        self.scope_label.setText(
            f"Profile: {profile} · Changes apply when you save."
            if local
            else "Shared across all profiles · Account actions apply immediately; other changes apply when you save."
            if self.tabs.currentIndex() == 1
            else "Shared across all profiles · Changes apply when you save."
        )

    def _page(self, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        heading = QLabel(title)
        heading.setFont(font_large)
        description_label = QLabel(description)
        description_label.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(description_label)
        return page, layout

    def render_general_tab(self) -> QWidget:
        page, layout = self._page(
            "Smart Fields", "Generate text, speech and images from your note fields."
        )
        search = ReactiveLineEdit(self.state, "search_text")
        search.setPlaceholderText("Search note types, decks, fields or prompts")
        search.setAccessibleName("Search Smart Fields")
        search.setClearButtonEnabled(True)
        search.on_change.connect(
            lambda text: self.state.update({"search_text": text, "selected_row": None})
        )
        layout.addWidget(search)
        self.table_count = QLabel()
        layout.addWidget(self.table_count)
        self.table = self.create_table()
        self.setup_table_context_menu(self.table)
        layout.addWidget(self.table, 1)
        self.table_empty = QLabel()
        self.table_empty.setWordWrap(True)
        layout.addWidget(self.table_empty)

        actions = QHBoxLayout()
        add = QPushButton("Add…")
        add.setAccessibleName("Add Smart Field")
        menu = QMenu(add)
        field_types: list[tuple[str, SmartFieldType]] = [
            ("Text", "chat"),
            ("Speech", "tts"),
            ("Image", "image"),
        ]
        for label, kind in field_types:
            action = menu.addAction(label)
            if action is not None:
                action.triggered.connect(partial(self.on_add, kind))
        add.setMenu(menu)
        actions.addWidget(add)
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.on_edit)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.remove_button)
        actions.addStretch()
        self.import_button = QPushButton("Import…")
        self.import_button.clicked.connect(self.on_import_prompts)
        self.export_button = QPushButton("Export…")
        self.export_button.clicked.connect(self.on_export_prompts)
        actions.addWidget(self.import_button)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        return page

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
        page, layout = self._page(
            "Providers",
            "Manage your ChatGPT account, shared API credentials and custom endpoints.",
        )
        self.chatgpt_account = ChatGPTAccountWidget()
        self.chatgpt_account.account_changed.connect(self._on_chatgpt_account_change)
        layout.addWidget(self.chatgpt_account)
        credentials = QGroupBox("Credentials")
        form = default_form_layout()
        credentials.setLayout(form)
        for provider, key in [
            ("OpenAI", "openai_api_key"),
            ("Anthropic", "anthropic_api_key"),
            ("DeepSeek", "deepseek_api_key"),
            ("Google", "google_api_key"),
            ("ElevenLabs", "elevenlabs_api_key"),
            ("Fish Audio", "fish_api_key"),
            ("Replicate", "replicate_api_key"),
        ]:
            edit = ReactiveLineEdit(self.state, key)
            edit.setAccessibleName(f"{provider} API key")
            edit.setPlaceholderText("API key")
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            edit.on_change.connect(
                lambda text, key=key: self._on_api_key_change(key, text)
            )
            status = QLabel("Configured" if edit.text() else "Not set")
            status.setMinimumWidth(status.fontMetrics().horizontalAdvance("Configured"))
            edit.textChanged.connect(
                lambda text, status=status: status.setText(
                    "Configured" if text.strip() else "Not set"
                )
            )
            reveal = QCheckBox("Show")
            reveal.setAccessibleName(f"Show {provider} API key")
            reveal.toggled.connect(
                lambda checked, edit=edit: edit.setEchoMode(
                    QLineEdit.EchoMode.Normal
                    if checked
                    else QLineEdit.EchoMode.Password
                )
            )
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(reveal)
            row.addWidget(status)
            label = QLabel(provider)
            label.setBuddy(edit)
            form.addRow(label, row)
        layout.addWidget(credentials)

        requests = QGroupBox("OpenAI text requests")
        request_form = default_form_layout()
        requests.setLayout(request_form)
        self.openai_fast_mode_checkbox = ReactiveCheckBox(
            self.state, "openai_fast_mode_enabled"
        )
        self.openai_fast_mode_checkbox.setText("Use Fast mode (premium pricing)")
        request_form.addRow(self.openai_fast_mode_checkbox)
        request_form.addRow(
            QLabel(
                "Request faster responses from supported OpenAI models at a higher per-token price."
            )
        )
        layout.addWidget(requests)

        budget = QGroupBox("OpenAI text · estimated token guard")
        budget_form = default_form_layout()
        budget.setLayout(budget_form)
        self.openai_budget_enabled_checkbox = ReactiveCheckBox(
            self.state, "openai_daily_token_budget_enabled"
        )
        self.openai_budget_enabled_checkbox.setText(
            "Enable estimated daily token limit"
        )
        budget_form.addRow(self.openai_budget_enabled_checkbox)
        self.openai_budget_spinbox = ReactiveSpinBox(
            self.state, "openai_daily_token_budget"
        )
        self.openai_budget_spinbox.setRange(1, 1_000_000_000)
        self.openai_budget_spinbox.setSingleStep(10_000)
        self.openai_budget_spinbox.setGroupSeparatorShown(True)
        budget_form.addRow("Daily limit:", self.openai_budget_spinbox)
        self.usage_label = QLabel()
        self.usage_label.setTextFormat(Qt.TextFormat.PlainText)
        budget_form.addRow(self.usage_label)
        budget_form.addRow(
            QLabel(
                "Includes reported OpenAI text usage and conservative estimates for requests with unknown usage. Excludes images, speech and other providers. This is not an exact billing ceiling."
            )
        )
        layout.addWidget(budget)

        custom = QGroupBox("Custom providers")
        custom_layout = QVBoxLayout(custom)
        self.custom_table = QTableWidget(0, 4)
        self.custom_table.setAccessibleName("Custom providers")
        self.custom_table.setHorizontalHeaderLabels(
            ["Name", "Base URL", "Types", "Key"]
        )
        self.custom_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.custom_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.custom_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.custom_table.itemDoubleClicked.connect(self._on_edit_custom_provider)
        header = self.custom_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        vertical = self.custom_table.verticalHeader()
        if vertical is not None:
            vertical.hide()
        for provider in self.state.s["custom_providers"]:
            self._add_custom_provider_row(provider)
        custom_layout.addWidget(self.custom_table)
        self.custom_empty = QLabel(
            "No custom providers. Add one to use a local or compatible API."
        )
        custom_layout.addWidget(self.custom_empty)
        actions = QHBoxLayout()
        add = QPushButton("Add…")
        add.clicked.connect(self._on_add_custom_provider)
        self.custom_edit = QPushButton("Edit…")
        self.custom_edit.clicked.connect(lambda: self._on_edit_custom_provider(None))
        self.custom_remove = QPushButton("Remove")
        self.custom_remove.clicked.connect(self._on_remove_custom_provider)
        for button in [add, self.custom_edit, self.custom_remove]:
            actions.addWidget(button)
        actions.addStretch()
        custom_layout.addLayout(actions)
        self.custom_table.itemSelectionChanged.connect(self._update_custom_buttons)
        self._update_custom_buttons()
        layout.addWidget(custom)
        layout.addStretch()
        return page

    def _on_chatgpt_account_change(self) -> None:
        if self.chat_options.state.s["chat_provider"] == "chatgpt":
            self.chat_options.refresh_custom_providers()

    def _update_custom_buttons(self) -> None:
        selected = bool(self.custom_table.selectedItems())
        self.custom_edit.setEnabled(selected)
        self.custom_remove.setEnabled(selected)
        count = self.custom_table.rowCount()
        self.custom_table.setVisible(count > 0)
        self.custom_empty.setVisible(count == 0)
        header = self.custom_table.horizontalHeader()
        if header is not None:
            height = (
                header.height()
                + sum(self.custom_table.rowHeight(row) for row in range(min(count, 6)))
                + self.custom_table.frameWidth() * 2
            )
            self.custom_table.setFixedHeight(
                max(height, self.custom_table.minimumSizeHint().height())
            )

    def _add_custom_provider_row(self, provider: CustomProvider) -> None:
        row = self.custom_table.rowCount()
        self.custom_table.insertRow(row)
        capabilities = provider.get("capabilities", ["chat"])
        labels = {"chat": "Text", "tts": "Speech", "image": "Image"}
        values = [
            provider["name"],
            provider["base_url"],
            ", ".join(labels.get(value, value) for value in capabilities)
            or "Needs capabilities",
            "Set" if provider.get("api_key") else "Not set",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            if column == 0:
                item.setData(Qt.ItemDataRole.UserRole, provider)
            self.custom_table.setItem(row, column, item)

    def _on_add_custom_provider(self) -> None:
        dialog = CustomProviderDialog(
            parent=self,
            existing_names=[p["name"] for p in self.state.s["custom_providers"]],
        )
        if dialog.exec():
            provider = dialog.get_provider()
            self._add_custom_provider_row(provider)
            self._save_custom_providers()

    def _on_edit_custom_provider(self, item: QTableWidgetItem | None) -> None:
        row = self.custom_table.currentRow()
        if row < 0:
            return

        name_item = self.custom_table.item(row, 0)
        if name_item is None:
            return
        provider_data = name_item.data(Qt.ItemDataRole.UserRole)

        dialog = CustomProviderDialog(
            provider=provider_data,
            parent=self,
            existing_names=[
                p["name"]
                for p in self.state.s["custom_providers"]
                if p["name"] != provider_data["name"]
            ],
        )
        if dialog.exec():
            new_provider = dialog.get_provider()

            self.custom_table.removeRow(row)
            self._add_custom_provider_row(new_provider)

            self._save_custom_providers()

    def _on_remove_custom_provider(self) -> None:
        row = self.custom_table.currentRow()
        if row >= 0:
            self.custom_table.removeRow(row)
            self._save_custom_providers()

    @with_widget_settings
    def _save_custom_providers(self) -> None:
        providers: list[CustomProvider] = []
        for i in range(self.custom_table.rowCount()):
            item = self.custom_table.item(i, 0)
            if item is not None:
                provider = item.data(Qt.ItemDataRole.UserRole)
                providers.append(provider)

        self.state.update({"custom_providers": providers})

        self._update_custom_buttons()
        if hasattr(self, "chat_options"):
            self.chat_options.refresh_custom_providers()
        if hasattr(self, "tts_options"):
            self.tts_options.refresh_custom_providers()
        if hasattr(self, "image_options"):
            self.image_options.refresh_custom_providers()

    def render_tools_tab(self) -> QWidget:
        page, layout = self._page(
            "Tools",
            "Choose tools available to text generation. Allow their use in Text defaults or a Smart Field’s Options.",
        )
        built_in_box = QGroupBox("Anki tools")
        built_in_layout = QVBoxLayout(built_in_box)
        self.built_in_tool_checkboxes: dict[str, QCheckBox] = {}
        for tool in list_built_in_tools():
            checkbox = QCheckBox(tool.title)
            checkbox.setChecked(self.state.s["built_in_tools"].get(tool.id, True))
            checkbox.toggled.connect(
                lambda checked, tool_id=tool.id: self._on_built_in_tool_toggled(
                    tool_id, checked
                )
            )
            built_in_layout.addWidget(checkbox)
            description = QLabel(tool.description)
            description.setWordWrap(True)
            built_in_layout.addWidget(description)
            self.built_in_tool_checkboxes[tool.id] = checkbox
        layout.addWidget(built_in_box)

        servers_box = QGroupBox("External MCP servers")
        servers_layout = QVBoxLayout(servers_box)
        intro = QLabel(
            f"Connect a trusted local command or HTTP service. Models can request its tools when tool use is enabled. <a href='{MCP_DOCS_URL}'>MCP documentation</a>"
        )
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        servers_layout.addWidget(intro)
        self.mcp_selector = QComboBox()
        self.mcp_selector.setAccessibleName("MCP server")
        self.mcp_selector.setPlaceholderText("Select a server…")
        self.mcp_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.mcp_selector.setMinimumContentsLength(18)
        servers_layout.addWidget(self.mcp_selector)
        self.mcp_details = QLabel()
        self.mcp_details.setTextFormat(Qt.TextFormat.PlainText)
        self.mcp_details.setWordWrap(True)
        self.mcp_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.mcp_details.setAccessibleDescription("Selected MCP server connection")
        servers_layout.addWidget(self.mcp_details)
        self.mcp_empty_label = QLabel(
            "No external servers. Add one to make its tools available."
        )
        servers_layout.addWidget(self.mcp_empty_label)
        buttons = QHBoxLayout()
        self.mcp_add_button = QPushButton("Add…")
        self.mcp_add_button.clicked.connect(self._on_add_mcp_server)
        self.mcp_edit_button = QPushButton("Edit…")
        self.mcp_edit_button.clicked.connect(self._on_edit_selected_mcp_server)
        self.mcp_remove_button = QPushButton("Remove")
        self.mcp_remove_button.clicked.connect(self._on_remove_selected_mcp_server)
        self.mcp_toggle_button = QPushButton("Disable")
        self.mcp_toggle_button.clicked.connect(self._on_toggle_selected_mcp_server)
        self.mcp_test_button = QPushButton("Test")
        self.mcp_test_button.setAccessibleName("Test MCP connection")
        self.mcp_test_button.clicked.connect(self._on_test_selected_mcp_server)
        for button in [
            self.mcp_add_button,
            self.mcp_edit_button,
            self.mcp_remove_button,
            self.mcp_toggle_button,
            self.mcp_test_button,
        ]:
            buttons.addWidget(button)
        buttons.addStretch()
        servers_layout.addLayout(buttons)
        status = QHBoxLayout()
        self.mcp_status = QLabel()
        self.mcp_status.setTextFormat(Qt.TextFormat.PlainText)
        self.mcp_status.setWordWrap(True)
        status.addWidget(self.mcp_status, 1)
        self.mcp_cancel_button = QPushButton("Cancel Test")
        self.mcp_cancel_button.clicked.connect(self._cancel_mcp_test)
        status.addWidget(self.mcp_cancel_button)
        servers_layout.addLayout(status)
        self.mcp_selector.currentIndexChanged.connect(self._update_mcp_buttons)
        self._render_mcp_servers()
        self._update_mcp_buttons()
        layout.addWidget(servers_box)
        layout.addStretch()
        return page

    def _render_mcp_servers(self) -> None:
        selected_id = self._selected_mcp_server_id()
        servers = self.state.s["mcp_servers"]
        self.mcp_selector.blockSignals(True)
        self.mcp_selector.clear()
        for server in servers:
            status = "Enabled" if server.get("enabled", True) else "Disabled"
            label = f"{server['name']} ({status})"
            self.mcp_selector.addItem(label, server["id"])
            self.mcp_selector.setItemData(
                self.mcp_selector.count() - 1, label, Qt.ItemDataRole.ToolTipRole
            )
        self.mcp_selector.setCurrentIndex(
            self.mcp_selector.findData(selected_id) if selected_id else -1
        )
        self.mcp_selector.blockSignals(False)
        self.mcp_selector.setVisible(bool(servers))
        self.mcp_details.setVisible(bool(servers))
        self.mcp_empty_label.setVisible(not servers)

    def _selected_mcp_server_id(self) -> str | None:
        server_id = self.mcp_selector.currentData()
        return str(server_id) if server_id else None

    def _selected_mcp_server(self) -> McpServerConfig | None:
        server_id = self._selected_mcp_server_id()
        if server_id is None:
            return None

        return next(
            (
                server
                for server in self.state.s["mcp_servers"]
                if server["id"] == server_id
            ),
            None,
        )

    def _on_add_mcp_server(self) -> None:
        dialog = McpServerDialog(parent=self)
        if dialog.exec():
            server = dialog.get_server()
            servers = [*self.state.s["mcp_servers"], server]
            self.state.update({"mcp_servers": servers})
            self._save_mcp_servers()

    def _on_built_in_tool_toggled(self, tool_id: str, enabled: bool) -> None:
        built_in_tools = dict(self.state.s["built_in_tools"])
        built_in_tools[tool_id] = enabled
        self.state.update(
            {"built_in_tools": cast("BuiltInToolsConfig", built_in_tools)}
        )
        self._save_built_in_tools()

    def _on_edit_selected_mcp_server(self) -> None:
        if self._mcp_job is not None:
            return
        server_id = self._selected_mcp_server_id()
        if server_id is None:
            return
        self._on_edit_mcp_server(server_id)

    def _on_edit_mcp_server(self, server_id: str) -> None:
        server = next(
            (
                existing
                for existing in self.state.s["mcp_servers"]
                if existing["id"] == server_id
            ),
            None,
        )
        if server is None:
            return

        dialog = McpServerDialog(server=server, parent=self)
        if dialog.exec():
            if dialog.was_removed():
                self._remove_mcp_server(server["id"])
                return

            updated_server = dialog.get_server()
            servers = list(self.state.s["mcp_servers"])
            for index, existing_server in enumerate(servers):
                if existing_server["id"] == updated_server["id"]:
                    servers[index] = updated_server
                    break
            self.state.update({"mcp_servers": servers})
            self._save_mcp_servers()

    def _on_remove_selected_mcp_server(self) -> None:
        server_id = self._selected_mcp_server_id()
        if server_id is None:
            return
        self._remove_mcp_server(server_id)

    def _remove_mcp_server(self, server_id: str) -> None:
        servers = [
            existing_server
            for existing_server in self.state.s["mcp_servers"]
            if existing_server["id"] != server_id
        ]
        self.state.update({"mcp_servers": servers})
        self._save_mcp_servers()

    def _on_toggle_selected_mcp_server(self) -> None:
        server = self._selected_mcp_server()
        if server is None:
            return
        self._set_mcp_server_enabled(server["id"], not server.get("enabled", True))

    def _set_mcp_server_enabled(self, server_id: str, enabled: bool) -> None:
        servers = self.state.s["mcp_servers"]
        for server in servers:
            if server["id"] == server_id:
                server["enabled"] = enabled
                break
        self.state.update({"mcp_servers": servers})
        self._save_mcp_servers()

    @with_widget_settings
    def _on_test_selected_mcp_server(self) -> None:
        server = self._selected_mcp_server()
        if server is None or self._mcp_job is not None:
            return
        self.mcp_status.setText(f"Connecting to {server['name']}…")
        self.mcp_test_button.setEnabled(False)

        def on_success(result: McpServerProbeResult) -> None:
            names = ", ".join(tool.name for tool in result.tools[:10])
            detail = f"Connected to {result.server_info.name}. {len(result.tools)} tools available."
            if names:
                detail += f" {names}" + ("…" if len(result.tools) > 10 else "")
            self.mcp_status.setText(redact_text(detail))

        def on_failure(error: Exception) -> None:
            if isinstance(error, JobCancelledError):
                self.mcp_status.setText("Connection test cancelled.")
                return
            self.mcp_status.setText(f"Connection failed: {redact_text(str(error))}")

        def on_finished() -> None:
            if not sip.isdeleted(self):
                self._mcp_job = None
                self._update_mcp_buttons()

        try:
            self._mcp_job = run_async_in_background_with_sentry(
                lambda: mcp_runtime.probe_server(server, isolated=True),
                on_success,
                on_failure,
                owner=self,
                on_finished=on_finished,
                show_error=False,
            )
        except Exception as error:
            on_failure(error)
            on_finished()
        self._update_mcp_buttons()

    def _cancel_mcp_test(self) -> None:
        if self._mcp_job is not None:
            self._mcp_job.cancel()
            self.mcp_status.setText("Connection test cancelled.")
            self.mcp_cancel_button.setEnabled(False)

    def _update_mcp_buttons(self) -> None:
        server = self._selected_mcp_server()
        if server is None:
            self.mcp_details.setText(
                "Select a server to view its connection and actions."
            )
            self.mcp_selector.setToolTip("")
        else:
            transport = (
                "Local command" if server["transport"] == "stdio" else "HTTP endpoint"
            )
            target = (
                server.get("command", "")
                if server["transport"] == "stdio"
                else server.get("url", "")
            )
            status = (
                "Enabled for tool use" if server.get("enabled", True) else "Disabled"
            )
            self.mcp_details.setText(
                redact_text(
                    f"{transport} · {status}\n{target or 'Connection is not configured.'}"
                )
            )
            self.mcp_selector.setToolTip(self.mcp_selector.currentText())
        busy = self._mcp_job is not None
        for button in [
            self.mcp_edit_button,
            self.mcp_remove_button,
            self.mcp_test_button,
            self.mcp_toggle_button,
        ]:
            button.setEnabled(server is not None and not busy)
        self.mcp_add_button.setEnabled(not busy)
        self.mcp_cancel_button.setVisible(busy)
        self.mcp_cancel_button.setEnabled(busy)
        self.mcp_test_button.setText("Testing…" if busy else "Test")
        self.mcp_toggle_button.setText(
            "Disable" if server is None or server.get("enabled", True) else "Enable"
        )

    def _save_mcp_servers(self) -> None:
        self.render_ui()

    def _save_built_in_tools(self) -> None:
        self.render_ui()

    def render_ui(self) -> None:
        current = self.state.s
        previous = self._rendered_state
        self._rendered_state = current
        if previous is None or any(
            current[key] != previous[key] for key in ["prompts_map", "search_text"]
        ):
            self.render_table()
        self.openai_budget_spinbox.setEnabled(
            current["openai_daily_token_budget_enabled"]
        )
        usage = chat_usage_tracker.get_openai_daily_usage()
        self.usage_label.setText(
            f"Today (UTC): {usage.used_total_tokens:,} tokens, including {usage.estimated_total_tokens:,} estimated. Resets at midnight UTC."
        )
        for tool_id, checkbox in self.built_in_tool_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(current["built_in_tools"].get(tool_id, True))
            checkbox.blockSignals(False)
        if previous is not None and current["mcp_servers"] != previous["mcp_servers"]:
            self._render_mcp_servers()
        self._update_mcp_buttons()
        self.render_buttons()

    @with_widget_settings
    def render_table(self) -> None:
        self.table.blockSignals(True)
        search = self.state.s["search_text"].casefold()
        all_prompts = get_all_prompts(override_prompts_map=self.state.s["prompts_map"])
        decks = deck_id_to_name_map()
        row = 0
        matches = 0
        for note_type, deck_prompts in all_prompts.items():
            current_fields = set(get_fields(note_type))
            for deck_id, field_prompts in deck_prompts.items():
                deck_name = decks.get(deck_id) or f"Missing deck ({deck_id})"
                for field, prompt in field_prompts.items():
                    extras = get_extras(
                        note_type=note_type, field=field, deck_id=deck_id
                    )
                    if extras is None:
                        continue
                    missing = field not in current_fields or deck_id not in decks
                    status = (
                        "Needs repair"
                        if missing
                        else "Automatic"
                        if extras["automatic"]
                        else "Manual only"
                    )
                    if extras.get("regenerate_when_batching", False):
                        status += " · Batch overwrite"
                    values = [
                        note_type,
                        deck_name,
                        field,
                        {"chat": "Text", "tts": "Speech", "image": "Image"}[
                            extras["type"]
                        ],
                        prompt,
                        status,
                    ]
                    if row >= self.table.rowCount():
                        self.table.insertRow(row)
                    for column, value in enumerate(values):
                        item = self.table.item(row, column)
                        if item is None:
                            item = QTableWidgetItem()
                            self.table.setItem(row, column, item)
                        item.setText(value)
                        item.setToolTip(
                            value
                            + (
                                "\nEdit to repair the missing note type, deck or field. Generation skips this target."
                                if missing
                                else ""
                            )
                        )
                        if column == 1:
                            item.setData(Qt.ItemDataRole.UserRole + 1, int(deck_id))
                    visible = not search or search in " ".join(values).casefold()
                    self.table.setRowHidden(row, not visible)
                    matches += int(visible)
                    row += 1
        self.table.setRowCount(row)
        selected = self.state.s["selected_row"]
        if (
            selected is not None
            and 0 <= selected < row
            and not self.table.isRowHidden(selected)
        ):
            self.table.selectRow(selected)
        else:
            self.table.clearSelection()
            self.table.setCurrentItem(None)
        self.table.blockSignals(False)
        self.table_count.setText(
            f"{matches} of {row} Smart Fields" if search else f"{row} Smart Fields"
        )
        self.table.setVisible(matches > 0)
        self.table_empty.setVisible(matches == 0)
        self.table_empty.setText(
            "No matching Smart Fields. Clear the search to see all fields."
            if row
            else "No Smart Fields yet. Add a field to choose its prompt and settings."
        )
        self.export_button.setEnabled(row > 0)

    def render_plugin_tab(self) -> QWidget:
        page, layout = self._page(
            "Generation", "Choose when Smart Fields run in this profile."
        )
        box = QGroupBox("Automatic and batch generation")
        form = default_form_layout()
        box.setLayout(form)
        self.generate_at_review_button = ReactiveCheckBox(
            self.state, "generate_at_review"
        )
        self.generate_at_review_button.setText("Generate enabled fields during review")
        form.addRow(self.generate_at_review_button)
        self.regenerate_notes_when_batching = ReactiveCheckBox(
            self.state, "regenerate_notes_when_batching"
        )
        self.regenerate_notes_when_batching.setText(
            "Overwrite existing fields in batches"
        )
        form.addRow(self.regenerate_notes_when_batching)
        form.addRow(
            QLabel(
                "Otherwise, batch generation fills empty fields, except fields with their own overwrite option. Review behavior is unchanged."
            )
        )
        self.allow_empty_fields_box = ReactiveCheckBox(self.state, "allow_empty_fields")
        self.allow_empty_fields_box.setText("Generate with some blank source fields")
        form.addRow(self.allow_empty_fields_box)
        form.addRow(
            QLabel("Prompts whose referenced fields are all blank are always skipped.")
        )
        layout.addWidget(box)
        troubleshooting = QGroupBox("Troubleshooting · shared across profiles")
        details = QVBoxLayout(troubleshooting)
        self.debug_checkbox = ReactiveCheckBox(self.state, "debug")
        self.debug_checkbox.setText("Enable diagnostic logging")
        details.addWidget(self.debug_checkbox)
        log_path = (
            get_file_path("smart-notes.log")
            if mw
            else "smart-notes.log (in the add-on folder)"
        )
        log_help = QLabel(
            f"Use diagnostic logs when reporting a problem. Log file: {log_path}\nSmart Notes {get_version()}"
        )
        log_help.setWordWrap(True)
        log_help.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.addWidget(log_help)
        layout.addWidget(troubleshooting)
        layout.addStretch()
        return page

    def render_chat_tab(self) -> QWidget:
        page, layout = self._page(
            "Text defaults",
            "Used by fields that inherit settings. Individual Smart Fields can customize these options.",
        )
        self.chat_options = ChatOptions()
        layout.addWidget(self.chat_options)
        layout.addStretch()
        return page

    def render_tts_tab(self) -> QWidget:
        page, layout = self._page(
            "Speech defaults",
            "Choose the voices used by fields that inherit settings. Individual Smart Fields can use their own voice pool.",
        )
        self.tts_options = TTSOptions()
        layout.addWidget(self.tts_options)
        layout.addStretch()
        return page

    def render_images_tab(self) -> QWidget:
        page, layout = self._page(
            "Image defaults",
            "Used by fields that inherit settings. Individual Smart Fields can customize these options.",
        )
        self.image_options = ImageOptions()
        layout.addWidget(self.image_options)
        layout.addStretch()
        return page

    def create_table(self) -> QTableWidget:
        table = QTableWidget(0, 6)
        table.setAccessibleName("Smart Fields")
        table.setHorizontalHeaderLabels(
            ["Note type", "Deck", "Target field", "Type", "Prompt", "Behavior"]
        )
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        header = table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            for column in [0, 1, 2, 5]:
                table.setColumnWidth(
                    column, table.fontMetrics().horizontalAdvance("Target field") + 24
                )
            table.setColumnWidth(
                2, table.fontMetrics().horizontalAdvance("SentenceMeaning") + 24
            )
        vertical = table.verticalHeader()
        if vertical is not None:
            vertical.hide()
        table.currentItemChanged.connect(self.on_row_selected)
        table.itemDoubleClicked.connect(self.on_edit)
        return table

    def on_row_selected(self, current: QTableWidgetItem | None) -> None:
        self.state.update({"selected_row": current.row() if current else None})

    def on_export_prompts(self, _) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Smart Notes Prompts",
            "prompts.smart-notes-prompts.json",
            "Smart Notes Prompt Export (*.smart-notes-prompts.json);;JSON (*.json)",
        )
        if not file_path:
            return

        export = create_prompt_export(
            self.state.s["prompts_map"],
            {int(deck_id): name for deck_id, name in deck_id_to_name_map().items()},
        )
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except OSError as e:
            show_message_box("Could not export prompts.", str(e))
            return

        show_message_box("Smart Notes prompts exported.")

    def on_import_prompts(self, _) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Smart Notes Prompts",
            "",
            "Smart Notes Prompt Export (*.smart-notes-prompts.json);;JSON (*.json)",
        )
        if not file_path:
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            show_message_box("Could not import prompts.", str(e))
            return

        try:
            preview = preview_prompt_import(
                payload=payload,
                existing_prompts_map=self.state.s["prompts_map"],
                deck_ids_by_name=deck_name_to_id_map(),
                fields_by_note_type=self.fields_by_note_type(),
            )
        except PromptImportError as e:
            show_message_box("Could not import prompts.", str(e))
            return

        if preview.imported_count == 0:
            show_message_box(
                "No prompts were imported.",
                self.import_preview_details(preview),
            )
            return

        did_confirm = show_message_box(
            "Import Smart Notes prompts?",
            self.import_preview_details(preview),
            custom_ok="Import",
            show_cancel=True,
        )
        if not did_confirm:
            return

        self.state.update({"prompts_map": preview.prompts_map, "selected_row": None})
        show_message_box("Prompts added to the draft. Save to keep them.")

    def fields_by_note_type(self) -> dict[str, list[str]]:
        if not mw or not mw.col:
            return {}

        fields_by_note_type: dict[str, list[str]] = {}
        for model in mw.col.models.all():
            name = model.get("name")
            fields = model.get("flds")
            if not isinstance(name, str) or not isinstance(fields, list):
                continue
            fields_by_note_type[name] = [
                field["name"]
                for field in sorted(fields, key=lambda field: field["ord"])
                if isinstance(field.get("name"), str)
            ]
        return fields_by_note_type

    def import_preview_details(self, preview: ImportPreview) -> str:
        lines = [
            f"Add: {preview.added_count}",
            f"Replace: {preview.replaced_count}",
            f"Skip: {len(preview.skipped)}",
        ]
        if preview.skipped:
            lines.append("")
            lines.append("Skipped prompts:")
            for skipped in preview.skipped[:8]:
                target = f"{skipped.note_type} / {skipped.deck_name} / {skipped.field}"
                lines.append(f"- {target}: {skipped.reason}")
            if len(preview.skipped) > 8:
                lines.append(f"- ...and {len(preview.skipped) - 8} more")
        return "\n".join(lines)

    @with_widget_settings
    def on_edit(self, _) -> None:
        row = self.state.s["selected_row"]
        if (
            row is None
            or not 0 <= row < self.table.rowCount()
            or self.table.isRowHidden(row)
        ):
            return

        note_type = get_item_text(self.table.item(row, 0))  # type: ignore
        deck_item = self.table.item(row, 1)
        if deck_item is None:
            return
        deck_id = DeckId(deck_item.data(Qt.ItemDataRole.UserRole + 1))
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

        if not prompts or field.lower() not in prompts:
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
        row = self.state.s["selected_row"]
        is_enabled = (
            row is not None
            and 0 <= row < self.table.rowCount()
            and not self.table.isRowHidden(row)
        )
        self.remove_button.setEnabled(is_enabled)
        self.edit_button.setEnabled(is_enabled)

    @with_widget_settings
    def on_add(self, field_type: SmartFieldType, _checked: bool = False) -> None:
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
        if (
            row is None
            or not 0 <= row < self.table.rowCount()
            or self.table.isRowHidden(row)
        ):
            return

        note_type = get_item_text(self.table.item(row, 0))  # type: ignore
        deck_item = self.table.item(row, 1)
        if deck_item is None:
            return
        deck_id = DeckId(deck_item.data(Qt.ItemDataRole.UserRole + 1))
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

    def draft_settings(self) -> dict[str, Any]:
        values = deepcopy(self.initial_settings)
        for state in [
            self.state,
            self.tts_options.state,
            self.chat_options.state,
            self.image_options.state,
        ]:
            values.update(
                {
                    key: deepcopy(value)
                    for key, value in state.s.items()
                    if key in Config.__annotations__
                }
            )
        return values

    def write_config(self, silent: bool = False) -> bool:
        values = self.draft_settings()
        if values.get("openai_endpoint") and not is_valid_url(
            values["openai_endpoint"]
        ):
            self.save_status.setText(
                "Invalid OpenAI host. Provide an HTTP(S) URL without credentials or query parameters."
            )
            self.save_status.show()
            return False
        try:
            config.save(values)
        except Exception as error:
            self.save_status.setText(
                f"Settings could not be saved. Your draft is still open. {redact_text(str(error))}"
            )
            self.save_status.show()
            return False
        self.save_status.hide()
        return True

    def _on_api_key_change(self, key: str, value: str) -> None:
        self.state.update({key: value})

    def on_update_prompts(self, prompts_map: PromptMap) -> None:
        self.state.update({"prompts_map": deepcopy(prompts_map)})

    def make_initial_state(self) -> State:
        return {
            "openai_api_key": config.openai_api_key,
            "anthropic_api_key": config.anthropic_api_key,
            "deepseek_api_key": config.deepseek_api_key,
            "google_api_key": config.google_api_key,
            "elevenlabs_api_key": config.elevenlabs_api_key,
            "fish_api_key": config.fish_api_key,
            "replicate_api_key": config.replicate_api_key,
            "prompts_map": config.prompts_map,
            "selected_row": None,
            "generate_at_review": config.generate_at_review,
            "regenerate_notes_when_batching": config.regenerate_notes_when_batching,
            "openai_endpoint": config.openai_endpoint,
            "openai_fast_mode_enabled": bool(config.openai_fast_mode_enabled),
            "openai_daily_token_budget_enabled": bool(
                config.openai_daily_token_budget_enabled
            ),
            "openai_daily_token_budget": int(
                config.openai_daily_token_budget or 1_000_000
            ),
            "allow_empty_fields": config.allow_empty_fields,
            "debug": config.debug,
            "custom_providers": config.custom_providers or [],
            "mcp_servers": config.mcp_servers or [],
            "built_in_tools": normalize_built_in_tools_config(config.built_in_tools),
            "search_text": "",
        }

    def on_restore_defaults(self) -> None:
        defaults = {
            "image_output_format": "webp",
            "image_quality": 85,
            **(config.defaults() or {}),
        }
        preserve = {
            "prompts_map",
            "custom_providers",
            "mcp_servers",
            "provider_settings",
        }
        for state in [
            self.state,
            self.tts_options.state,
            self.chat_options.state,
            self.image_options.state,
        ]:
            state.update(
                {
                    key: deepcopy(defaults[key])
                    for key in state.s
                    if key in defaults
                    and key not in preserve
                    and not key.endswith("_api_key")
                    and key != "auth_token"
                }
            )

        self.save_status.setText(
            "Defaults restored in this draft. Prompts, credentials, custom providers and MCP servers are preserved. Save to apply."
        )
        self.save_status.show()


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        _ = parsed.port
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() for character in url)
        )
    except ValueError:
        return False
