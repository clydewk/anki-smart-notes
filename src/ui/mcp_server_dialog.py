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

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from aqt import (
    QComboBox,
    QDesktopServices,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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

from ..job_context import JobCancelledError
from ..logger import redact_text
from ..mcp_runtime import McpServerProbeResult, mcp_runtime
from ..models import McpKeyValuePair, McpServerConfig, McpServerTransport
from ..sentry import run_async_in_background_with_sentry
from ..settings import with_widget_settings
from ..utils import make_uuid
from .ui_utils import (
    bound_dialog,
    default_form_layout,
    font_large,
    font_small,
)

if TYPE_CHECKING:
    from ..tasks import JobHandle


MCP_DOCS_URL = "https://modelcontextprotocol.io/introduction"


class McpRowsEditor(QGroupBox):
    def __init__(
        self,
        title: str,
        first_placeholder: str,
        add_label: str,
        second_placeholder: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(title, parent)
        self._first_placeholder = first_placeholder
        self._second_placeholder = second_placeholder
        self._rows: list[tuple[QWidget, QLineEdit, QLineEdit | None, QPushButton]] = []

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

    def add_row(self, first_value: str, second_value: str | None = None) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)
        row_widget.setLayout(row_layout)

        first_edit = QLineEdit(first_value)
        first_edit.setPlaceholderText(self._first_placeholder)
        form = default_form_layout()
        form.addRow(f"{self._first_placeholder}:", first_edit)
        first_edit.setAccessibleName(f"{self.title()} {self._first_placeholder}")
        row_layout.addLayout(form, 1)

        second_edit: QLineEdit | None = None
        if self._second_placeholder is not None:
            value_edit = QLineEdit(second_value or "")
            value_edit.setPlaceholderText(self._second_placeholder)
            value_row = QHBoxLayout()
            value_row.addWidget(value_edit, 1)
            form.addRow(f"{self._second_placeholder}:", value_row)
            value_edit.setAccessibleName(f"{self.title()} {self._second_placeholder}")
            second_edit = value_edit
            if self._second_placeholder == "Value":
                value_edit.setEchoMode(QLineEdit.EchoMode.Password)
                reveal = QPushButton("Show")
                reveal.setCheckable(True)
                reveal.setAutoDefault(False)
                reveal.toggled.connect(
                    lambda shown, edit=value_edit: edit.setEchoMode(
                        QLineEdit.EchoMode.Normal
                        if shown
                        else QLineEdit.EchoMode.Password
                    )
                )
                reveal.setAccessibleName(f"Show {self.title().lower()} value")
                first_edit.textChanged.connect(
                    lambda key, button=reveal: button.setAccessibleName(
                        f"Show {self.title().lower()} value for {key or 'new row'}"
                    )
                )
                value_row.addWidget(reveal)

        remove_button = QPushButton("Remove")
        remove_button.setAutoDefault(False)
        remove_button.setAccessibleName(f"Remove {self.title().lower()} row")
        first_edit.textChanged.connect(
            lambda key: remove_button.setAccessibleName(
                f"Remove {self.title().lower()} {key or 'row'}"
            )
        )
        remove_button.clicked.connect(lambda: self.remove_row(row_widget))
        row_layout.addWidget(remove_button)

        self.rows_layout.addWidget(row_widget)
        self._rows.append((row_widget, first_edit, second_edit, remove_button))
        first_edit.textChanged.connect(self._refresh_remove_buttons)
        self._refresh_remove_buttons()

    def remove_row(self, row_widget: QWidget) -> None:
        for index, (widget, _, _, _) in enumerate(self._rows):
            if widget is not row_widget:
                continue
            self._rows.pop(index)
            self.rows_layout.removeWidget(widget)
            widget.hide()
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
            widget.hide()
            widget.deleteLater()

    def _refresh_remove_buttons(self) -> None:
        for index, (widget, first, second, button) in enumerate(self._rows, start=1):
            identity = first.text().strip() or f"row {index}"
            button.setEnabled(len(self._rows) > 1)
            button.setAccessibleName(f"Remove {self.title().lower()} {identity}")
            first.setAccessibleName(
                f"{self.title()} {self._first_placeholder}, row {index}"
            )
            if second is not None:
                second.setAccessibleName(
                    f"{self.title()} {self._second_placeholder} for {identity}"
                )
            for reveal in widget.findChildren(QPushButton):
                if reveal.isCheckable():
                    reveal.setAccessibleName(
                        f"Show {self.title().lower()} value for {identity}"
                    )


class McpServerDialog(QDialog):
    def __init__(
        self,
        server: McpServerConfig | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.server = server
        self._transport: McpServerTransport = "stdio"
        self._removed = False
        self._probe_job: JobHandle | None = None
        self._open = True
        self.setWindowTitle(
            f"Edit MCP Server: {server['name']}" if server else "Add MCP Server"
        )
        self.setup_ui()
        bound_dialog(self)

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

        self.command_edit = self._create_line_edit(
            "Executable name or absolute path, e.g. npx"
        )
        self.cwd_edit = self._create_line_edit(
            f"Optional absolute path, e.g. {Path.home()}"
        )
        self.url_edit = self._create_line_edit("https://developers.openai.com/mcp")
        self.command_edit.setAccessibleName("MCP executable")
        self.cwd_edit.setAccessibleName("MCP working directory")
        self.url_edit.setAccessibleName("MCP HTTP endpoint")

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
        self.cancel_test_button = QPushButton("Cancel test")
        self.cancel_test_button.setVisible(False)
        self.cancel_test_button.clicked.connect(self._cancel_probe)
        footer.addWidget(self.cancel_test_button)
        footer.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        done_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        assert done_button is not None
        done_button.setText("Done")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        footer.addWidget(buttons)
        scope = QLabel(
            "Done updates the settings draft. Save in the main settings window to keep it."
        )
        scope.setWordWrap(True)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setAccessibleDescription(
            "MCP connection and validation status"
        )
        outer_layout.addWidget(self.status_label)
        outer_layout.addWidget(scope)
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
        settings_form.addRow("Executable:", self.command_edit)
        command_help = QLabel(
            "Enter the executable alone. Put each argument in a separate row below; no shell command parsing is applied."
        )
        command_help.setWordWrap(True)
        settings_form.addRow(command_help)
        cwd_row = QHBoxLayout()
        cwd_row.addWidget(self.cwd_edit, 1)
        browse = QPushButton("Browse…")
        browse.setAccessibleName("Choose MCP working directory")
        browse.clicked.connect(self._browse_directory)
        cwd_row.addWidget(browse)
        settings_form.addRow("Working directory:", cwd_row)

        layout.addWidget(settings_box)
        layout.addWidget(self.args_editor)
        self.stdio_environment = self._optional_section(
            layout,
            "Environment (optional)",
            "Explicit values are passed only to this server. Passthrough copies only named variables from Anki's environment; the full environment is never inherited.",
            (self.env_editor, self.env_passthrough_editor),
            bool(
                self.server
                and (self.server.get("env") or self.server.get("env_passthrough"))
            ),
        )
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
        self.http_headers = self._optional_section(
            layout,
            "Headers (optional)",
            "Use explicit header values or name an environment variable whose value supplies a header. Secret values remain masked until revealed.",
            (self.headers_editor, self.header_env_editor),
            bool(
                self.server
                and (self.server.get("headers") or self.server.get("header_env_vars"))
            ),
        )
        layout.addStretch()
        return tab

    def _create_line_edit(self, placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(True)
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

    def _optional_section(
        self,
        layout: QVBoxLayout,
        title: str,
        description: str,
        editors: tuple[QWidget, ...],
        expanded: bool,
    ) -> QWidget:
        button = QPushButton(title)
        button.setCheckable(True)
        button.setChecked(expanded)
        button.setAutoDefault(False)
        content = QWidget()
        group_layout = QVBoxLayout(content)
        group_layout.setContentsMargins(0, 0, 0, 0)
        help_text = QLabel(description)
        help_text.setWordWrap(True)
        group_layout.addWidget(help_text)
        for editor in editors:
            group_layout.addWidget(editor)
        content.setVisible(expanded)
        button.toggled.connect(content.setVisible)
        layout.addWidget(button)
        layout.addWidget(content)
        return content

    def _browse_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "MCP working directory", self.cwd_edit.text()
        )
        if directory:
            self.cwd_edit.setText(directory)

    def _validate(self) -> bool:
        error = ""
        target = self.name_edit
        if not self.name_edit.text().strip():
            error = "Enter a server name."
        elif self._transport == "stdio":
            target = self.command_edit
            command = self.command_edit.text().strip()
            if not command:
                error = "Enter an executable. Put arguments in the separate rows below."
            elif (
                self.cwd_edit.text().strip()
                and not Path(self.cwd_edit.text().strip()).is_absolute()
            ):
                error = "Use an absolute working-directory path, or leave it empty. The ~ shorthand is not expanded."
                target = self.cwd_edit
        else:
            target = self.url_edit
            try:
                parsed = urlsplit(self.url_edit.text().strip())
                if (
                    parsed.scheme not in {"https", "http"}
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.fragment
                ):
                    raise ValueError
                _ = parsed.port
            except ValueError:
                error = "Enter an HTTP or HTTPS MCP endpoint without embedded credentials or a fragment."
        if error:
            self.status_label.setText(error)
            target.setFocus()
            return False
        return True

    def _finish_probe(self) -> None:
        self._probe_job = None
        self.test_button.setEnabled(True)
        self.test_button.setText("Test Connection")
        self.cancel_test_button.setVisible(False)
        self.cancel_test_button.setEnabled(True)

    def _cancel_probe(self) -> None:
        if self._probe_job:
            self._probe_job.cancel()
        self.status_label.setText("Cancelling connection test…")
        self.cancel_test_button.setEnabled(False)

    @with_widget_settings
    def _on_test_connection(self) -> None:
        if not self._validate():
            return
        server = self.get_server()
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing…")
        self.cancel_test_button.setVisible(True)
        self.status_label.setText(f"Connecting to {server['name']}…")

        def on_success(result: McpServerProbeResult) -> None:
            self._finish_probe()
            details = [f"Connected to {result.server_info.name}"]
            if result.server_info.version:
                details.append(f"Version: {result.server_info.version}")
            details.append(f"Tools: {len(result.tools)}")
            if result.tools:
                details.append(
                    "Available: " + ", ".join(tool.name for tool in result.tools[:10])
                )
            else:
                details.append(
                    "The server connected successfully but advertises no tools."
                )
            self.status_label.setText("\n".join(details))

        def on_failure(error: Exception) -> None:
            self._finish_probe()
            self.status_label.setText(
                "Connection test cancelled."
                if isinstance(error, JobCancelledError)
                else f"Connection failed: {redact_text(str(error))}"
            )

        try:
            self._probe_job = run_async_in_background_with_sentry(
                lambda: mcp_runtime.probe_server(server, isolated=True),
                on_success,
                on_failure,
                owner_alive=lambda: self._open,
                show_error=False,
            )
        except Exception as error:
            on_failure(error)

    def accept(self) -> None:
        if self._removed or self._validate():
            super().accept()

    def done(self, a0: int) -> None:
        self._open = False
        if self._probe_job:
            self._probe_job.cancel()
        super().done(a0)

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
