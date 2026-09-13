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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, get_args
from urllib.parse import urlsplit

from aqt import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..chat_provider import (
    CustomProviderVerification,
    chat_provider,
    normalize_api_base_url,
)
from ..job_context import JobCancelledError
from ..logger import redact_text
from ..models import ChatProviders, CustomProvider, ImageProviders, TTSProviders
from ..sentry import run_async_in_background_with_sentry
from .ui_utils import bound_dialog, default_form_layout, font_small, scroll_page

if TYPE_CHECKING:
    from ..tasks import JobHandle


def validate_custom_provider(
    name: str, base_url: str, existing_names: Iterable[str] = ()
) -> None:
    reserved = {
        value.casefold()
        for value in (
            *get_args(ChatProviders),
            *get_args(TTSProviders),
            *get_args(ImageProviders),
        )
    }
    if not name.strip():
        raise ValueError("Please enter a provider name.")
    if name.casefold() in reserved:
        raise ValueError("Choose a name different from the built-in providers.")
    if name.casefold() in {value.casefold() for value in existing_names}:
        raise ValueError("A custom provider with this name already exists.")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in base_url)
    ):
        raise ValueError(
            "Enter an HTTP or HTTPS base URL without credentials, query parameters, or a fragment."
        )
    _ = parsed.port  # Validate malformed port numbers.


class CustomProviderDialog(QDialog):
    def __init__(
        self,
        provider: CustomProvider | None = None,
        parent: QWidget | None = None,
        existing_names: Iterable[str] = (),
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Custom Provider" if provider else "Add Custom Provider"
        )
        self.setMinimumWidth(500)
        self.provider = provider
        self.existing_names = tuple(existing_names)
        self._fetch_job: JobHandle | None = None
        self._open = True
        self.setup_ui()
        bound_dialog(self)

    def setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        page = QWidget()
        layout = QVBoxLayout(page)
        outer.addWidget(scroll_page(page))

        form_layout = default_form_layout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. My Local LLM")

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(
            "e.g. http://localhost:11434 or https://api.openai.com"
        )

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("Optional for local models")
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_row = QHBoxLayout()
        key_row.addWidget(self.key_edit)
        reveal_key = QCheckBox("Show")
        reveal_key.setAccessibleName("Show custom provider API key")
        self.key_edit.setAccessibleName("Custom provider API key")
        reveal_key.toggled.connect(
            lambda shown: self.key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(reveal_key)

        self.api_mode_combo = QComboBox()
        self.api_mode_combo.addItems(["Responses", "Chat Completions"])

        self.streaming_mode_combo = QComboBox()
        self.streaming_mode_combo.addItems(["Enabled", "Disabled"])

        if self.provider:
            self.name_edit.setText(self.provider["name"])
            self.url_edit.setText(self.provider["base_url"])
            self.key_edit.setText(self.provider["api_key"])
            self.api_mode_combo.setCurrentText(
                {
                    "responses": "Responses",
                    "chat_completions": "Chat Completions",
                }.get(self.provider.get("chat_api_mode", "responses"), "Responses")
            )
            self.streaming_mode_combo.setCurrentText(
                {
                    "enabled": "Enabled",
                    "disabled": "Disabled",
                }.get(self.provider.get("streaming_mode", "enabled"), "Enabled")
            )

        form_layout.addRow("Name:", self.name_edit)
        form_layout.addRow("Base URL:", self.url_edit)
        form_layout.addRow("API key:", key_row)
        form_layout.addRow("Text API:", self.api_mode_combo)
        form_layout.addRow("Streaming:", self.streaming_mode_combo)

        url_help = QLabel(
            "Use the server's base URL, optionally ending in /v1. Do not include an endpoint such as /responses or /chat/completions."
        )
        url_help.setWordWrap(True)
        form_layout.insertRow(2, url_help)
        api_help = QLabel(
            "Responses uses the OpenAI Responses API. Choose Chat Completions for servers that expose the older chat endpoint. Match the server's documented API."
        )
        api_help.setWordWrap(True)
        form_layout.addRow(api_help)
        layout.addLayout(form_layout)

        caps_group = QGroupBox("Capabilities")
        caps_layout = QHBoxLayout()
        caps_group.setLayout(caps_layout)

        self.chat_check = QCheckBox("Text")
        self.tts_check = QCheckBox("Text-to-Speech")
        self.image_check = QCheckBox("Image Generation")

        caps_layout.addWidget(self.chat_check)
        caps_layout.addWidget(self.tts_check)
        caps_layout.addWidget(self.image_check)

        if self.provider and "capabilities" in self.provider:
            caps = self.provider["capabilities"] or []
            self.chat_check.setChecked("chat" in caps)
            self.tts_check.setChecked("tts" in caps)
            self.image_check.setChecked("image" in caps)
        else:
            self.chat_check.setChecked(True)

        layout.addWidget(caps_group)

        models_group = QGroupBox("Models")
        models_layout = QVBoxLayout()
        models_group.setLayout(models_layout)

        desc = QLabel(
            "Enter one model ID per line. Enable its capability above to use it. Fetch suggests categories; review them before Done."
        )
        desc.setFont(font_small)
        desc.setWordWrap(True)
        models_layout.addWidget(desc)

        self.models_tabs = QTabWidget()

        self.chat_models_edit = QTextEdit()
        self.chat_models_edit.setPlaceholderText("gpt-4o\nclaude-3-opus\n...")
        self.models_tabs.addTab(self.chat_models_edit, "Text Models")

        self.tts_models_edit = QTextEdit()
        self.tts_models_edit.setPlaceholderText("tts-1\n...")
        self.models_tabs.addTab(self.tts_models_edit, "TTS Models")

        self.image_models_edit = QTextEdit()
        self.image_models_edit.setPlaceholderText("dall-e-3\n...")
        self.models_tabs.addTab(self.image_models_edit, "Image Models")

        if self.provider:
            legacy_models = self.provider.get("models", [])
            chat_m = self.provider.get("chat_models")
            tts_m = self.provider.get("tts_models")
            img_m = self.provider.get("image_models")

            if chat_m is None and tts_m is None and img_m is None and legacy_models:
                self.chat_models_edit.setPlainText("\n".join(legacy_models))
            else:
                self.chat_models_edit.setPlainText("\n".join(chat_m or []))
                self.tts_models_edit.setPlainText("\n".join(tts_m or []))
                self.image_models_edit.setPlainText("\n".join(img_m or []))

        models_layout.addWidget(self.models_tabs)

        self.fetch_btn = QPushButton("Fetch from API")
        self.fetch_btn.clicked.connect(self.on_fetch_models)
        self.cancel_fetch_btn = QPushButton("Cancel fetch")
        self.cancel_fetch_btn.setVisible(False)
        self.cancel_fetch_btn.clicked.connect(self._cancel_fetch)
        fetch_row = QHBoxLayout()
        fetch_row.addWidget(self.fetch_btn)
        fetch_row.addWidget(self.cancel_fetch_btn)
        fetch_row.addStretch()
        models_layout.addLayout(fetch_row)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setAccessibleDescription(
            "Provider validation and model discovery status"
        )
        models_layout.addWidget(self.status_label)
        for checkbox in (self.chat_check, self.tts_check, self.image_check):
            checkbox.toggled.connect(self._refresh_capabilities)
        self._refresh_capabilities()

        layout.addWidget(models_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        done_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        assert done_button is not None
        done_button.setText("Done")
        scope = QLabel(
            "Done updates the settings draft. Save in the main settings window to keep it."
        )
        scope.setWordWrap(True)
        outer.addWidget(scope)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _refresh_capabilities(self) -> None:
        for index, checkbox in enumerate(
            (self.chat_check, self.tts_check, self.image_check)
        ):
            self.models_tabs.setTabEnabled(index, checkbox.isChecked())
        self.api_mode_combo.setEnabled(self.chat_check.isChecked())
        self.streaming_mode_combo.setEnabled(self.chat_check.isChecked())
        if not self.models_tabs.isTabEnabled(self.models_tabs.currentIndex()):
            for index in range(self.models_tabs.count()):
                if self.models_tabs.isTabEnabled(index):
                    self.models_tabs.setCurrentIndex(index)
                    break

    def _validate(self) -> bool:
        provider = self.get_provider()
        try:
            validate_custom_provider(
                provider["name"], provider["base_url"], self.existing_names
            )
            if not provider["capabilities"]:
                raise ValueError(
                    "Select at least one capability: Text, Speech, or Images."
                )
        except ValueError as error:
            self.status_label.setText(str(error))
            if not self.name_edit.text().strip() or "name" in str(error).lower():
                self.name_edit.setFocus()
            elif "Select at least" in str(error):
                self.chat_check.setFocus()
            else:
                self.url_edit.setFocus()
            return False
        return True

    def _finish_fetch(self) -> None:
        self._fetch_job = None
        self.fetch_btn.setText("Fetch from API")
        self.fetch_btn.setEnabled(True)
        self.cancel_fetch_btn.setVisible(False)
        self.cancel_fetch_btn.setEnabled(True)

    def _cancel_fetch(self) -> None:
        if self._fetch_job:
            self._fetch_job.cancel()
        self.status_label.setText("Cancelling model discovery…")
        self.cancel_fetch_btn.setEnabled(False)

    def on_fetch_models(self) -> None:
        if not self._validate():
            return
        provider = self.get_provider()
        self.fetch_btn.setText("Fetching…")
        self.fetch_btn.setEnabled(False)
        self.cancel_fetch_btn.setVisible(True)
        self.status_label.setText("Contacting the provider to discover models…")

        def on_success(verification: CustomProviderVerification) -> None:
            self._finish_fetch()
            if not verification.models:
                self.status_label.setText(
                    "Connected, but the provider returned no model IDs. You can enter IDs manually."
                )
                return
            self.api_mode_combo.setCurrentText(
                "Responses"
                if verification.chat_api_mode == "responses"
                else "Chat Completions"
            )
            categories: tuple[list[str], list[str], list[str]] = ([], [], [])
            for model in verification.models:
                lower = model.lower()
                index = (
                    1
                    if any(word in lower for word in ("tts", "audio", "speech"))
                    else 2
                    if any(
                        word in lower
                        for word in ("image", "dall-e", "flux", "diffusion")
                    )
                    else 0
                )
                categories[index].append(model)
            for editor, models in zip(
                (self.chat_models_edit, self.tts_models_edit, self.image_models_edit),
                categories,
                strict=True,
            ):
                existing = {
                    line.strip()
                    for line in editor.toPlainText().splitlines()
                    if line.strip()
                }
                editor.setPlainText("\n".join(sorted(existing.union(models))))
            self.status_label.setText(
                f"Fetched {len(verification.models)} models: {len(categories[0])} text, {len(categories[1])} speech, {len(categories[2])} image. Categories are suggestions; enable the capabilities you intend to use and review their model IDs."
            )

        def on_failure(error: Exception) -> None:
            self._finish_fetch()
            self.status_label.setText(
                "Model discovery cancelled. Existing model lists are unchanged."
                if isinstance(error, JobCancelledError)
                else f"Model discovery failed: {redact_text(str(error))}"
            )

        try:
            self._fetch_job = run_async_in_background_with_sentry(
                lambda: chat_provider.verify_custom_provider(provider),
                on_success,
                on_failure,
                owner_alive=lambda: self._open,
                show_error=False,
            )
        except Exception as error:
            on_failure(error)

    def accept(self) -> None:
        if self._validate():
            super().accept()

    def done(self, a0: int) -> None:
        self._open = False
        if self._fetch_job:
            self._fetch_job.cancel()
        super().done(a0)

    def get_provider(self) -> CustomProvider:
        caps = []
        if self.chat_check.isChecked():
            caps.append("chat")
        if self.tts_check.isChecked():
            caps.append("tts")
        if self.image_check.isChecked():
            caps.append("image")

        def get_clean_list(text_edit: QTextEdit) -> list[str]:
            return [
                line.strip()
                for line in text_edit.toPlainText().splitlines()
                if line.strip()
            ]

        chat_models = get_clean_list(self.chat_models_edit)
        tts_models = get_clean_list(self.tts_models_edit)
        image_models = get_clean_list(self.image_models_edit)
        all_models = sorted(set(chat_models + tts_models + image_models))

        return {
            "name": self.name_edit.text().strip(),
            "base_url": normalize_api_base_url(self.url_edit.text().strip()),
            "api_key": self.key_edit.text().strip(),
            "capabilities": caps,
            "models": all_models,
            "chat_models": chat_models,
            "tts_models": tts_models,
            "image_models": image_models,
            "chat_api_mode": self.current_chat_api_mode(),
            "streaming_mode": self.current_streaming_mode(),
        }

    def current_chat_api_mode(self) -> Literal["responses", "chat_completions"]:
        if self.api_mode_combo.currentText() == "Chat Completions":
            return "chat_completions"
        return "responses"

    def current_streaming_mode(self) -> Literal["enabled", "disabled"]:
        if self.streaming_mode_combo.currentText() == "Disabled":
            return "disabled"
        return "enabled"
