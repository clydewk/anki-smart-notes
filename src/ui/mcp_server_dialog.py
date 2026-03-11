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

from typing import Optional

from aqt import (
    QComboBox,
    QDesktopServices,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    Qt,
    QUrl,
    QVBoxLayout,
    QWidget,
)

from ..logger import logger
from ..mcp_runtime import McpServerProbeResult, mcp_runtime
from ..models import McpKeyValuePair, McpServerConfig, McpServerTransport
from ..sentry import run_async_in_background_with_sentry
from ..utils import make_uuid
from .ui_utils import default_form_layout, font_large, font_small, show_message_box

MCP_DOCS_URL = "https://modelcontextprotocol.io/introduction"


class McpRowsEditor(QGroupBox):
    def __init__(
        self,
        title: str,
        first_placeholder: str,
        add_label: str,
        second_placeholder: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(title, parent)
        self._first_placeholder = first_placeholder
        self._second_placeholder = second_placeholder
        self._rows: list[
            tuple[QWidget, QLineEdit, Optional[QLineEdit], QPushButton]
        ] = []

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.setLayout(layout)

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(10)
        layout.addLayout(self.rows_layout)

        self.add_button = QPushButton(f"+ {add_label}")
        self.add_button.setAutoDefault(False)
        self.add_button.clicked.connect(self.add_empty_row)
        layout.addWidget(self.add_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.add_empty_row()

    def add_empty_row(self) -> None:
        self.add_row("", "" if self._second_placeholder is not None else None)

    def add_row(self, first_value: str, second_value: Optional[str] = None) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row_widget.setLayout(row_layout)

        first_edit = QLineEdit(first_value)
        first_edit.setPlaceholderText(self._first_placeholder)
        row_layout.addWidget(first_edit, 1)

        second_edit: Optional[QLineEdit] = None
        if self._second_placeholder is not None:
            value_edit = QLineEdit(second_value or "")
            value_edit.setPlaceholderText(self._second_placeholder)
            row_layout.addWidget(value_edit, 1)
            second_edit = value_edit

        remove_button = QPushButton("−")
        remove_button.setAutoDefault(False)
        remove_button.setFixedWidth(32)
        remove_button.setToolTip("Remove row")
        remove_button.clicked.connect(lambda: self.remove_row(row_widget))
        row_layout.addWidget(remove_button)

        self.rows_layout.addWidget(row_widget)
        self._rows.append((row_widget, first_edit, second_edit, remove_button))
        self._refresh_remove_buttons()

    def remove_row(self, row_widget: QWidget) -> None:
        for index, (widget, _, _, _) in enumerate(self._rows):
            if widget is not row_widget:
                continue
            self._rows.pop(index)
            self.rows_layout.removeWidget(widget)
            widget.deleteLater()
            break

        if not self._rows:
            self.add_empty_row()
            return

        self._refresh_remove_buttons()

    def set_pairs(self, pairs: list[McpKeyValuePair]) -> None:
        self._clear_rows()
        if not pairs:
            self.add_empty_row()
            return

        for pair in pairs:
            self.add_row(pair.get("key", ""), pair.get("value", ""))

    def pairs(self) -> list[McpKeyValuePair]:
        pairs: list[McpKeyValuePair] = []
        for _, first_edit, second_edit, _ in self._rows:
            key = first_edit.text().strip()
            value = second_edit.text().strip() if second_edit else ""
            if key:
                pairs.append({"key": key, "value": value})
        return pairs

    def set_values(self, values: list[str]) -> None:
        self._clear_rows()
        if not values:
            self.add_empty_row()
            return

        for value in values:
            self.add_row(value, None)

    def values(self) -> list[str]:
        values: list[str] = []
        for _, first_edit, _, _ in self._rows:
            value = first_edit.text().strip()
            if value:
                values.append(value)
        return values

    def _clear_rows(self) -> None:
        while self._rows:
            widget, _, _, _ = self._rows.pop()
            self.rows_layout.removeWidget(widget)
            widget.deleteLater()

    def _refresh_remove_buttons(self) -> None:
        can_remove = len(self._rows) > 1
        for _, _, _, button in self._rows:
            button.setEnabled(can_remove)


class McpServerDialog(QDialog):
    def __init__(
        self,
        server: Optional[McpServerConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.server = server
        self._transport: McpServerTransport = "stdio"
        self._removed = False
        self.setWindowTitle(
            f"Edit MCP Server: {server['name']}" if server else "Add MCP Server"
        )
        self.setMinimumWidth(680)
        self.setMinimumHeight(720)
        self.setup_ui()

    def setup_ui(self) -> None:
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(12)
        self.setLayout(outer_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        content.setLayout(layout)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)

        title = QLabel("Edit MCP Server" if self.server else "Add MCP Server")
        title.setFont(font_large)
        header.addWidget(title)
        header.addStretch()

        docs = QLabel(f"<a href='{MCP_DOCS_URL}'>Documentation</a>")
        docs.setOpenExternalLinks(False)
        docs.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))
        header.addWidget(docs)
        layout.addLayout(header)

        subtitle = QLabel(
            "Configure a local command or HTTP endpoint that exposes MCP tools for chat Smart Fields."
        )
        subtitle.setWordWrap(True)
        subtitle.setFont(font_small)
        layout.addWidget(subtitle)

        warning = QLabel(
            "Only configure MCP servers you trust. Local commands and HTTP endpoints can run tool calls when requested by the model."
        )
        warning.setWordWrap(True)
        warning.setFont(font_small)
        layout.addWidget(warning)

        general_box = QGroupBox("General")
        general_form = self._create_form_layout()
        general_box.setLayout(general_form)

        self.name_edit = self._create_line_edit("MCP server name")
        self.transport_combo = QComboBox()
        self.transport_combo.addItems(["STDIO", "Streamable HTTP"])
        self.transport_combo.currentIndexChanged.connect(
            self._on_transport_index_changed
        )
        general_form.addRow("Name:", self.name_edit)
        general_form.addRow("Transport:", self.transport_combo)
        layout.addWidget(general_box)

        self.command_edit = self._create_line_edit("openai-dev-mcp serve-sqlite")
        self.cwd_edit = self._create_line_edit("~/code")
        self.url_edit = self._create_line_edit("https://developers.openai.com/mcp")

        self.args_editor = McpRowsEditor("Arguments", "Argument", "Add argument")
        self.env_editor = McpRowsEditor(
            "Environment variables",
            "Key",
            "Add environment variable",
            "Value",
        )
        self.env_passthrough_editor = McpRowsEditor(
            "Environment variable passthrough",
            "Variable",
            "Add variable",
        )
        self.headers_editor = McpRowsEditor(
            "Headers",
            "Key",
            "Add header",
            "Value",
        )
        self.header_env_editor = McpRowsEditor(
            "Headers from environment variables",
            "Header",
            "Add variable",
            "Environment variable",
        )

        self.transport_stack = QStackedWidget()
        self.transport_stack.addWidget(self._build_stdio_tab())
        self.transport_stack.addWidget(self._build_http_tab())
        layout.addWidget(self.transport_stack)
        layout.addStretch()

        footer = QHBoxLayout()
        if self.server:
            self.remove_button = QPushButton("Remove")
            self.remove_button.clicked.connect(self._on_remove_clicked)
            footer.addWidget(self.remove_button)

        self.test_button = QPushButton("Test Connection")
        self.test_button.clicked.connect(self._on_test_connection)
        footer.addWidget(self.test_button)
        footer.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        footer.addWidget(buttons)
        outer_layout.addLayout(footer)

        if self.server:
            self._load_server(self.server)
        else:
            self._set_transport("stdio")

    def _build_stdio_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        tab.setLayout(layout)

        hint = QLabel("Launch a local MCP server as a child process.")
        hint.setWordWrap(True)
        hint.setFont(font_small)
        layout.addWidget(hint)

        settings_box = QGroupBox("STDIO Settings")
        settings_form = self._create_form_layout()
        settings_box.setLayout(settings_form)
        settings_form.addRow("Command:", self.command_edit)
        settings_form.addRow("Working directory:", self.cwd_edit)

        layout.addWidget(settings_box)
        layout.addWidget(self.args_editor)
        layout.addWidget(self.env_editor)
        layout.addWidget(self.env_passthrough_editor)
        layout.addStretch()
        return tab

    def _build_http_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        tab.setLayout(layout)

        hint = QLabel("Connect to a streamable HTTP MCP endpoint.")
        hint.setWordWrap(True)
        hint.setFont(font_small)
        layout.addWidget(hint)

        settings_box = QGroupBox("HTTP Settings")
        settings_form = self._create_form_layout()
        settings_box.setLayout(settings_form)
        settings_form.addRow("URL:", self.url_edit)

        layout.addWidget(settings_box)
        layout.addWidget(self.headers_editor)
        layout.addWidget(self.header_env_editor)
        layout.addStretch()
        return tab

    def _create_line_edit(self, placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(True)
        edit.setMinimumWidth(360)
        return edit

    def _create_form_layout(self) -> QFormLayout:
        form = default_form_layout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _on_transport_index_changed(self, index: int) -> None:
        self._transport = "stdio" if index == 0 else "streamable_http"
        self.transport_stack.setCurrentIndex(index)

    def _set_transport(self, transport: McpServerTransport) -> None:
        self._transport = transport
        index = 0 if transport == "stdio" else 1
        if self.transport_combo.currentIndex() != index:
            self.transport_combo.setCurrentIndex(index)
        self.transport_stack.setCurrentIndex(index)

    def _load_server(self, server: McpServerConfig) -> None:
        self.name_edit.setText(server["name"])
        self.command_edit.setText(server.get("command", ""))
        self.cwd_edit.setText(server.get("cwd", ""))
        self.url_edit.setText(server.get("url", ""))
        self.args_editor.set_values(server.get("args", []))
        self.env_editor.set_pairs(server.get("env", []))
        self.env_passthrough_editor.set_values(server.get("env_passthrough", []))
        self.headers_editor.set_pairs(server.get("headers", []))
        self.header_env_editor.set_pairs(server.get("header_env_vars", []))
        self._set_transport(server["transport"])

    def _on_remove_clicked(self) -> None:
        self._removed = True
        self.accept()

    def was_removed(self) -> bool:
        return self._removed

    def _on_test_connection(self) -> None:
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing...")

        def on_success(result: McpServerProbeResult) -> None:
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Connection")
            details = [f"Connected to {result.server_info.name}"]
            if result.server_info.version:
                details.append(f"Version: {result.server_info.version}")
            details.append(f"Tools: {len(result.tools)}")
            if result.tools:
                details.append(
                    "Available: " + ", ".join(tool.name for tool in result.tools[:10])
                )
            show_message_box("\n".join(details), custom_ok="Close")

        def on_failure(error: Exception) -> None:
            logger.error("Failed to test MCP server: %s", error)
            self.test_button.setEnabled(True)
            self.test_button.setText("Test Connection")
            show_message_box(f"Failed to connect to MCP server: {error}")

        run_async_in_background_with_sentry(
            lambda: mcp_runtime.probe_server(self.get_server()),
            on_success,
            on_failure,
            use_collection=False,
        )

    def accept(self) -> None:
        if not self._removed:
            name = self.name_edit.text().strip()
            if not name:
                show_message_box("Please enter a server name.")
                return

            if self._transport == "stdio" and not self.command_edit.text().strip():
                show_message_box("Please enter a command to launch the stdio server.")
                return

            if (
                self._transport == "streamable_http"
                and not self.url_edit.text().strip()
            ):
                show_message_box("Please enter the MCP server URL.")
                return

        super().accept()

    def get_server(self) -> McpServerConfig:
        server_id = self.server["id"] if self.server else f"mcp_{make_uuid()}"
        return {
            "id": server_id,
            "name": self.name_edit.text().strip(),
            "enabled": self.server.get("enabled", True) if self.server else True,
            "transport": self._transport,
            "command": self.command_edit.text().strip(),
            "args": self.args_editor.values(),
            "env": self.env_editor.pairs(),
            "env_passthrough": self.env_passthrough_editor.values(),
            "cwd": self.cwd_edit.text().strip(),
            "url": self.url_edit.text().strip(),
            "headers": self.headers_editor.pairs(),
            "header_env_vars": self.header_env_editor.pairs(),
        }
