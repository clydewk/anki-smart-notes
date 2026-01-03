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

import aiohttp
from aqt import (
    QCheckBox,
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

        # Form Fields
        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. My Local LLM")

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("e.g. http://localhost:11434/v1")

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("Optional for local models")

        if self.provider:
            self.name_edit.setText(self.provider["name"])
            self.url_edit.setText(self.provider["base_url"])
            self.key_edit.setText(self.provider["api_key"])

        form_layout.addRow("<b>Name:</b>", self.name_edit)
        form_layout.addRow("<b>Base URL:</b>", self.url_edit)
        form_layout.addRow("<b>API Key:</b>", self.key_edit)

        layout.addLayout(form_layout)

        # Capabilities
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
            # Default to chat if new
            self.chat_check.setChecked(True)

        layout.addWidget(caps_group)

        # Models Section
        models_group = QGroupBox("Models")
        models_layout = QVBoxLayout()
        models_group.setLayout(models_layout)

        desc = QLabel(
            "Categorize models by capability. Fetch will populate all (you may need to sort them)."
        )
        desc.setFont(font_small)
        models_layout.addWidget(desc)

        # Tabs for model categories
        self.models_tabs = QTabWidget()

        # Chat Models
        self.chat_models_edit = QTextEdit()
        self.chat_models_edit.setPlaceholderText("gpt-4o\nclaude-3-opus\n...")
        self.models_tabs.addTab(self.chat_models_edit, "Text Models")

        # TTS Models
        self.tts_models_edit = QTextEdit()
        self.tts_models_edit.setPlaceholderText("tts-1\n...")
        self.models_tabs.addTab(self.tts_models_edit, "TTS Models")

        # Image Models
        self.image_models_edit = QTextEdit()
        self.image_models_edit.setPlaceholderText("dall-e-3\n...")
        self.models_tabs.addTab(self.image_models_edit, "Image Models")

        # Populate if editing
        if self.provider:
            # Legacy support: if 'models' exists but specific lists don't, dump to chat
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

        # alignment argument in PyQt6 must be a valid Qt.AlignmentFlag, not None.
        models_layout.addWidget(self.fetch_btn)

        layout.addWidget(models_group)

        # Buttons
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

        def on_success(models: list[str]) -> None:
            self.fetch_btn.setText("Fetch from API")
            self.fetch_btn.setEnabled(True)

            if not models:
                show_message_box("No models found.")
                return

            # Helper to merge new models into existing text
            def merge_models(text_edit: QTextEdit, new_models: list[str]):
                current_text = text_edit.toPlainText()
                existing = set(
                    line.strip() for line in current_text.splitlines() if line.strip()
                )
                all_models = sorted(list(existing.union(set(new_models))))
                text_edit.setText("\n".join(all_models))

            # Naive categorization heuristics
            chat_models = []
            tts_models = []
            img_models = []

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
                    # Default to chat for everything else
                    chat_models.append(m)

            # If current tab is specific, prioritize it?
            # Actually just dumping into categorized tabs is better.

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

    async def _fetch_models_logic(self, base_url: str, api_key: str) -> list[str]:
        # Intelligent URL guessing
        # OpenAI standard: GET /v1/models

        urls_to_try = []

        # If user provided a specific /models endpoint, try it first?
        # Or assume base URL needs suffix.

        clean_url = base_url.rstrip("/")

        # Heuristic 1: Replace /chat/completions with /models
        if "/chat/completions" in clean_url:
            urls_to_try.append(clean_url.replace("/chat/completions", "/models"))

        # Heuristic 2: Append /models
        urls_to_try.append(f"{clean_url}/models")

        # Heuristic 3: If ends with /v1, try just appending /models (already covered by 2)
        # If doesn't end with /v1, try appending /v1/models
        if not clean_url.endswith("/v1"):
            urls_to_try.append(f"{clean_url}/v1/models")

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_exception = None

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            for url in urls_to_try:
                try:
                    logger.debug(f"Trying to fetch models from: {url}")
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            # Verify content type
                            content_type = response.headers.get("Content-Type", "")
                            if "application/json" not in content_type:
                                logger.debug(
                                    f"Skipping {url}: content-type is {content_type}"
                                )
                                continue

                            data = await response.json()
                            if "data" in data and isinstance(data["data"], list):
                                return [m["id"] for m in data["data"] if "id" in m]
                            # Handle Ollama list format: { "models": [ { "name": "llama2" } ] }
                            if "models" in data and isinstance(data["models"], list):
                                return [
                                    m["name"] for m in data["models"] if "name" in m
                                ]

                except Exception as e:
                    last_exception = e
                    logger.debug(f"Failed to fetch from {url}: {e}")

        raise last_exception or Exception("Could not find a valid models endpoint.")

    def get_provider(self) -> CustomProvider:
        caps = []
        if self.chat_check.isChecked():
            caps.append("chat")
        if self.tts_check.isChecked():
            caps.append("tts")
        if self.image_check.isChecked():
            caps.append("image")

        # Helper to clean list
        def get_clean_list(text_edit: QTextEdit) -> list[str]:
            return [
                line.strip()
                for line in text_edit.toPlainText().splitlines()
                if line.strip()
            ]

        chat_models = get_clean_list(self.chat_models_edit)
        tts_models = get_clean_list(self.tts_models_edit)
        image_models = get_clean_list(self.image_models_edit)

        # Master list for legacy compatibility
        all_models = sorted(list(set(chat_models + tts_models + image_models)))

        return {
            "name": self.name_edit.text().strip(),
            "base_url": self.url_edit.text().strip(),
            "api_key": self.key_edit.text().strip(),
            "capabilities": caps,
            "models": all_models,
            "chat_models": chat_models,
            "tts_models": tts_models,
            "image_models": image_models,
        }
