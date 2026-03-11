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

from typing import Any, Optional, TypedDict, cast

from aqt import QGroupBox, QLabel, QSpacerItem, QWidget

from ..chat_provider import chat_provider
from ..config import config, key_or_config_val
from ..models import (
    ChatModels,
    ChatProviders,
    OpenAIReasoningEffort,
    OverridableChatOptionsDict,
    ProviderSettings,
    openai_reasoning_efforts_for_model,
    overridable_chat_options,
    provider_model_map,
)
from ..sentry import run_async_in_background_with_sentry
from .reactive_check_box import ReactiveCheckBox
from .reactive_combo_box import ReactiveComboBox
from .reactive_spin_box import ReactiveDoubleSpinBox
from .state_manager import StateManager
from .ui_utils import default_form_layout, font_small


class ChatOptionsState(TypedDict):
    chat_provider: str
    chat_providers: list[str]
    chat_models: list[ChatModels]
    chat_model: ChatModels
    chat_temperature: int
    chat_reasoning_effort: Optional[OpenAIReasoningEffort]
    chat_reasoning_efforts: list[OpenAIReasoningEffort]
    chat_markdown_to_html: bool
    chat_use_mcp: bool
    provider_settings: dict[str, ProviderSettings]


models_map: dict[str, str] = {
    "gpt-5-mini": "GPT-5 Mini (1x cost)",
    "gpt-5-chat-latest": "GPT-5 (No Reasoning, 5x cost)",
    "gpt-5": "GPT-5 (Reasoning, 5x++ cost)",
    "gpt-5-nano": "GPT-5 Nano (0.2x cost)",
    "gpt-4o-mini": "GPT-4o Mini (0.3x cost)",
    "claude-opus-4-1": "Claude Opus 4.1 (40x Cost)",
    "claude-sonnet-4-0": "Claude Sonnet 4.0 (3x Cost)",
    "claude-3-5-haiku-latest": "Claude 3.5 Haiku (2x Cost)",
    "deepseek-v3": "Deepseek v3 (0.7x Cost)",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
}

providers_map = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
}

all_chat_providers: list[ChatProviders] = ["openai", "anthropic", "deepseek", "google"]

reasoning_efforts_map: dict[str, str] = {
    "none": "None (Use Temperature)",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "Extra High",
}


class ChatOptions(QWidget):
    _show_text_processing: bool

    def __init__(
        self,
        chat_options: Optional[OverridableChatOptionsDict] = None,
        show_text_processing: bool = True,
        show_mcp_toggle: bool = True,
    ):
        super().__init__()
        self.state = StateManager[ChatOptionsState](
            self.get_initial_state(chat_options or {})  # type: ignore
        )
        self._show_text_processing = show_text_processing
        self._show_mcp_toggle = show_mcp_toggle
        self._openai_refresh_inflight = False
        self.setup_ui()

    def setup_ui(self) -> None:
        self.chat_provider = ReactiveComboBox(
            self.state, "chat_providers", "chat_provider", providers_map
        )
        self.chat_provider.on_change.connect(self._on_provider_change)

        self.temperature = ReactiveDoubleSpinBox(self.state, "chat_temperature")
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(0.1)
        self.temperature.on_change.connect(self._on_temperature_change)

        is_custom = self.state.s["chat_provider"] not in all_chat_providers
        self.chat_model = ReactiveComboBox(
            self.state, "chat_models", "chat_model", models_map, editable=is_custom
        )
        self.chat_model.setMinimumWidth(350)
        self.chat_model.on_change.connect(self._on_model_change)

        self.reasoning_effort = ReactiveComboBox(
            self.state,
            "chat_reasoning_efforts",
            "chat_reasoning_effort",
            reasoning_efforts_map,
        )
        self.reasoning_effort.on_change.connect(self._on_reasoning_effort_change)

        chat_box = QGroupBox("✨ Language Model")
        chat_form = default_form_layout()
        chat_box.setLayout(chat_form)
        chat_form.addRow("Provider:", self.chat_provider)
        chat_form.addRow("Model:", self.chat_model)

        text_rules = QGroupBox("🔤 Text Processing")
        text_layout = default_form_layout()
        text_rules.setLayout(text_layout)
        text_rules.setHidden(not self._show_text_processing)
        self.convert_box = ReactiveCheckBox(self.state, "chat_markdown_to_html")
        text_layout.addRow(QLabel("Convert Markdown to HTML:"), self.convert_box)
        convert_explainer = QLabel(
            "Language models often use **Markdown** in their responses - convert it to HTML to render within Anki."
        )
        convert_explainer.setFont(font_small)
        text_layout.addRow(convert_explainer)
        advanced = QGroupBox("⚙️ Advanced")
        advanced_layout = default_form_layout()
        advanced.setLayout(advanced_layout)
        advanced_layout.addRow("Temperature:", self.temperature)
        temp_desc = QLabel(
            "Temperature controls the creativity of responses. Values range from 0-2 (ChatGPT default is 1)."
        )
        temp_desc.setFont(font_small)
        advanced_layout.addRow(temp_desc)

        advanced_layout.addRow("Reasoning Effort:", self.reasoning_effort)
        reasoning_desc = QLabel(
            "Controls how hard the model thinks. Only available for some models. Disables temperature when active."
        )
        reasoning_desc.setFont(font_small)
        advanced_layout.addRow(reasoning_desc)

        tools = QGroupBox("🧰 Tools")
        tools_layout = default_form_layout()
        tools.setLayout(tools_layout)
        tools.setHidden(not self._show_mcp_toggle)
        self.use_mcp_box = ReactiveCheckBox(self.state, "chat_use_mcp")
        tools_layout.addRow(QLabel("Use MCP tools by default:"), self.use_mcp_box)
        tools_desc = QLabel(
            "Allow chat Smart Fields to call configured local MCP tools unless a field overrides this setting."
        )
        tools_desc.setFont(font_small)
        tools_layout.addRow(tools_desc)

        chat_layout = default_form_layout()
        chat_layout.addRow(chat_box)
        chat_layout.addItem(QSpacerItem(0, 12))
        chat_layout.addRow(text_rules)
        if self._show_text_processing:
            chat_layout.addItem(QSpacerItem(0, 12))
        chat_layout.addRow(advanced)
        if self._show_mcp_toggle:
            chat_layout.addItem(QSpacerItem(0, 12))
            chat_layout.addRow(tools)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(chat_layout)

        self._on_reasoning_effort_change(self.state.s.get("chat_reasoning_effort"))
        self._on_model_change(self.state.s["chat_model"])
        self._update_editable_state(self.state.s["chat_provider"])
        current_provider = self.state.s["chat_provider"]
        current_model = self.state.s["chat_model"]
        current_temp = self.state.s["chat_temperature"]
        current_effort = self.state.s["chat_reasoning_effort"]

        self._update_provider_settings(
            current_provider,
            model=current_model,
            temperature=current_temp,
            reasoning_effort=current_effort,
        )
        self._refresh_openai_models_if_needed(current_provider)

    def _update_editable_state(self, provider: str) -> None:
        is_custom = provider not in all_chat_providers
        self.chat_model.setEditable(is_custom)

    def _on_provider_change(self, text: str) -> None:
        is_custom = text not in all_chat_providers
        models = provider_model_map.get(text, [])

        if is_custom:
            custom_provider = next(
                (p for p in (config.custom_providers or []) if p["name"] == text), None
            )
            if custom_provider:
                models = cast(
                    "list[ChatModels]",
                    custom_provider.get("chat_models")
                    or custom_provider.get("models")
                    or [],
                )

        last_settings = self.state.s["provider_settings"].get(text)
        new_model = ""
        new_temp = self.state.s["chat_temperature"]
        new_effort = self.state.s["chat_reasoning_effort"]

        if last_settings:
            if is_custom or last_settings["model"] in models:
                new_model = last_settings["model"]
            elif models:
                new_model = models[0]

            new_temp = last_settings["temperature"]
            new_effort = last_settings["reasoning_effort"]
        elif models:
            new_model = models[0]

        efforts = openai_reasoning_efforts_for_model(new_model)

        if new_effort not in efforts:
            new_effort = (
                "none" if "none" in efforts else (efforts[0] if efforts else None)
            )

        self._update_editable_state(text)
        self.state.update(
            {
                "chat_provider": text,
                "chat_models": models,
                "chat_model": new_model,
                "chat_temperature": new_temp,
                "chat_reasoning_effort": new_effort,
                "chat_reasoning_efforts": efforts,
            }
        )

        is_reasoning = new_effort and new_effort != "none"
        self.temperature.setEnabled(not is_reasoning)

        self._update_provider_settings(
            text,
            model=new_model,
            temperature=new_temp,
            reasoning_effort=new_effort,
        )  # type: ignore
        self._refresh_openai_models_if_needed(text)

    def refresh_custom_providers(self) -> None:
        custom_provider_names = [p["name"] for p in (config.custom_providers or [])]
        new_providers = all_chat_providers + custom_provider_names

        self.state.update({"chat_providers": new_providers})

        current = self.state.s["chat_provider"]
        if current not in new_providers:
            self.state.update({"chat_provider": "openai"})
            self._on_provider_change("openai")
        elif current in custom_provider_names:
            self._on_provider_change(current)
        elif current == "openai":
            self._refresh_openai_models_if_needed(current)

    def get_initial_state(
        self, chat_options: OverridableChatOptionsDict
    ) -> ChatOptionsState:
        ret: ChatOptionsState = {
            k: key_or_config_val(chat_options, k)
            for k in overridable_chat_options  # type: ignore
        }
        ret["chat_use_mcp"] = config.chat_use_mcp

        custom_provider_names = [p["name"] for p in (config.custom_providers or [])]
        ret["chat_providers"] = all_chat_providers + custom_provider_names
        ret["provider_settings"] = config.provider_settings or {}

        current_provider = ret["chat_provider"]
        if current_provider in provider_model_map:
            if current_provider == "openai":
                ret["chat_models"] = cast(
                    "list[ChatModels]", chat_provider.get_cached_openai_chat_models()
                )
            else:
                ret["chat_models"] = provider_model_map[current_provider]
        else:
            custom_provider = next(
                (
                    p
                    for p in (config.custom_providers or [])
                    if p["name"] == current_provider
                ),
                None,
            )
            ret["chat_models"] = cast(
                "list[ChatModels]",
                (
                    custom_provider.get("chat_models")
                    or custom_provider.get("models")
                    or []
                )
                if custom_provider
                else [],
            )

        current_model = ret.get("chat_model")
        efforts = (
            openai_reasoning_efforts_for_model(current_model) if current_model else []
        )
        ret["chat_reasoning_efforts"] = efforts

        if not ret.get("chat_reasoning_effort") and efforts:
            ret["chat_reasoning_effort"] = (
                "none" if "none" in efforts else efforts[0] if efforts else None
            )

        return ret

    def _on_model_change(self, model: str) -> None:
        efforts = openai_reasoning_efforts_for_model(model)

        provider = self.state.s["chat_provider"]
        self._update_provider_settings(provider, model=model)

        self.state.update({"chat_reasoning_efforts": efforts})

        current_effort = self.state.s.get("chat_reasoning_effort")
        if current_effort not in efforts:
            new_effort = (
                "none" if "none" in efforts else (efforts[0] if efforts else None)
            )
            self.state.update({"chat_reasoning_effort": new_effort})
            self._update_provider_settings(provider, reasoning_effort=new_effort)

    def _on_reasoning_effort_change(self, effort: Optional[str]) -> None:
        self.state.update({"chat_reasoning_effort": effort})  # type: ignore
        self._update_provider_settings(
            self.state.s["chat_provider"], reasoning_effort=effort
        )  # type: ignore

        is_reasoning = effort and effort != "none"
        self.temperature.setEnabled(not is_reasoning)

    def _update_provider_settings(self, provider: str, **kwargs: Any) -> None:
        settings_map = self.state.s["provider_settings"]

        if provider not in settings_map:
            settings_map[provider] = {
                "model": self.state.s["chat_model"],
                "temperature": self.state.s["chat_temperature"],
                "reasoning_effort": self.state.s["chat_reasoning_effort"],
            }

        for k, v in kwargs.items():
            settings_map[provider][k] = v  # type: ignore

        self.state.update({"provider_settings": settings_map})

    def _on_temperature_change(self, temp: float) -> None:
        self.state.update({"chat_temperature": temp})
        self._update_provider_settings(self.state.s["chat_provider"], temperature=temp)

    def _refresh_openai_models_if_needed(self, provider: str) -> None:
        if (
            provider != "openai"
            or not config.openai_api_key
            or self._openai_refresh_inflight
        ):
            return

        self._openai_refresh_inflight = True

        def on_success(models: list[str]) -> None:
            self._openai_refresh_inflight = False
            if not models or self.state.s["chat_provider"] != "openai":
                return

            current_model = self.state.s["chat_model"]
            next_model = current_model if current_model in models else models[0]
            next_efforts = openai_reasoning_efforts_for_model(next_model)
            next_effort = self.state.s["chat_reasoning_effort"]
            if next_effort not in next_efforts:
                next_effort = (
                    "none"
                    if "none" in next_efforts
                    else (next_efforts[0] if next_efforts else None)
                )

            self.state.update(
                {
                    "chat_models": cast("list[ChatModels]", models),
                    "chat_model": cast("ChatModels", next_model),
                    "chat_reasoning_efforts": next_efforts,
                    "chat_reasoning_effort": next_effort,
                }
            )
            self._update_provider_settings(
                "openai",
                model=next_model,
                reasoning_effort=next_effort,
            )

        def on_failure(_: Exception) -> None:
            self._openai_refresh_inflight = False

        run_async_in_background_with_sentry(
            chat_provider.fetch_openai_chat_models,
            on_success,
            on_failure,
        )
