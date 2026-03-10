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

from typing import Literal, Optional

from aqt import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
from ..logger import logger
from ..models import CustomProvider
from ..sentry import run_async_in_background_with_sentry
from .ui_utils import font_small, show_message_box


class CustomProviderDialog(QDialog):
    def __init__(
        self,
        provider: Optional[CustomProvider] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Custom Provider Configuration")
        self.setMinimumWidth(500)
        self.provider = provider
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. My Local LLM")

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(
            "e.g. http://localhost:11434 or https://api.openai.com"
        )

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("Optional for local models")

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

        form_layout.addRow("<b>Name:</b>", self.name_edit)
        form_layout.addRow("<b>Base URL:</b>", self.url_edit)
        form_layout.addRow("<b>API Key:</b>", self.key_edit)
        form_layout.addRow("<b>Chat API:</b>", self.api_mode_combo)
        form_layout.addRow("<b>Streaming:</b>", self.streaming_mode_combo)

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
            "Categorize models by capability. Fetch will populate all (you may need to sort them)."
        )
        desc.setFont(font_small)
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
                self.chat_models_edit.setText("\n".join(legacy_models))
            else:
                self.chat_models_edit.setText("\n".join(chat_m or []))
                self.tts_models_edit.setText("\n".join(tts_m or []))
                self.image_models_edit.setText("\n".join(img_m or []))

        models_layout.addWidget(self.models_tabs)

        self.fetch_btn = QPushButton("Fetch from API")
        self.fetch_btn.clicked.connect(self.on_fetch_models)
        self.fetch_btn.setFixedWidth(120)
        models_layout.addWidget(self.fetch_btn)

        layout.addWidget(models_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def on_fetch_models(self) -> None:
        url = self.url_edit.text().strip()
        key = self.key_edit.text().strip()

        if not url:
            show_message_box("Please enter a Base URL first.")
            return

        self.fetch_btn.setText("Fetching...")
        self.fetch_btn.setEnabled(False)

        def on_success(verification: CustomProviderVerification) -> None:
            self.fetch_btn.setText("Fetch from API")
            self.fetch_btn.setEnabled(True)

            models = verification.models
            if not models:
                show_message_box("No models found.")
                return

            self.api_mode_combo.setCurrentText(
                "Responses"
                if verification.chat_api_mode == "responses"
                else "Chat Completions"
            )

            def merge_models(text_edit: QTextEdit, new_models: list[str]) -> None:
                current_text = text_edit.toPlainText()
                existing = {
                    line.strip() for line in current_text.splitlines() if line.strip()
                }
                all_models = sorted(existing.union(set(new_models)))
                text_edit.setText("\n".join(all_models))

            chat_models: list[str] = []
            tts_models: list[str] = []
            img_models: list[str] = []

            for m in models:
                m_lower = m.lower()
                if "tts" in m_lower or "audio" in m_lower or "speech" in m_lower:
                    tts_models.append(m)
                elif (
                    "image" in m_lower
                    or "dall-e" in m_lower
                    or "flux" in m_lower
                    or "diffusion" in m_lower
                ):
                    img_models.append(m)
                else:
                    chat_models.append(m)

            merge_models(self.chat_models_edit, chat_models)
            merge_models(self.tts_models_edit, tts_models)
            merge_models(self.image_models_edit, img_models)

            msg = f"Fetched {len(models)} models.\n\n"
            msg += f"Chat: {len(chat_models)}\n"
            msg += f"TTS: {len(tts_models)}\n"
            msg += f"Image: {len(img_models)}\n\n"
            msg += "Models were auto-sorted based on keywords. Please review the tabs."

            show_message_box(msg)

        def on_failure(e: Exception) -> None:
            self.fetch_btn.setText("Fetch from API")
            self.fetch_btn.setEnabled(True)
            show_message_box(f"Failed to fetch models: {e}")

        run_async_in_background_with_sentry(
            lambda: self._fetch_models_logic(url, key), on_success, on_failure
        )

    async def _fetch_models_logic(
        self, base_url: str, api_key: str
    ) -> CustomProviderVerification:
        provider: CustomProvider = {
            "name": self.name_edit.text().strip() or "Custom Provider",
            "base_url": normalize_api_base_url(base_url),
            "api_key": api_key,
            "capabilities": ["chat"],
            "models": [],
            "chat_models": [],
            "tts_models": [],
            "image_models": [],
            "chat_api_mode": self.current_chat_api_mode(),
            "streaming_mode": self.current_streaming_mode(),
        }
        logger.debug(
            f"Fetching models for {provider['name']} from {provider['base_url']}"
        )
        return await chat_provider.verify_custom_provider(provider)

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
