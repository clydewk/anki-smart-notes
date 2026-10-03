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

from copy import deepcopy
from typing import Any, TypedDict

from aqt import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from aqt.qt import QShowEvent
from PyQt6 import sip

from ..chat_provider import chat_provider
from ..chatgpt_auth import chatgpt_auth
from ..config import config, key_or_config_val
from ..logger import redact_text
from ..models import (
    RETIRED_PROVIDER_MODELS,
    ChatGPTModel,
    ChatModels,
    ChatProviders,
    OpenAIReasoningEffort,
    OverridableChatOptionsDict,
    ProviderSettings,
    openai_chat_models_for_display,
    openai_model_label,
    openai_reasoning_efforts_for_model,
    openai_supports_chat_tools,
    openai_supports_temperature,
    overridable_chat_options,
    provider_model_map,
)
from ..sentry import run_async_in_background_with_sentry
from ..settings import widget_settings, with_widget_settings
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
    chat_temperature: float
    chat_reasoning_effort: OpenAIReasoningEffort | None
    chat_reasoning_efforts: list[OpenAIReasoningEffort]
    chat_markdown_to_html: bool
    chat_use_tools: bool
    show_available_openai_models: bool
    provider_settings: dict[str, ProviderSettings]


models_map: dict[str, str] = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
}

for model in openai_chat_models_for_display([]):
    models_map[model] = openai_model_label(model)

providers_map = {
    "openai": "OpenAI",
    "chatgpt": "ChatGPT",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
}

all_chat_providers: list[ChatProviders] = [
    "openai",
    "chatgpt",
    "anthropic",
    "deepseek",
    "google",
]

reasoning_efforts_map: dict[str, str] = {
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "Extra High",
    "max": "Maximum",
    "ultra": "Ultra",
}


class ChatOptions(QWidget):
    _show_text_processing: bool

    def __init__(
        self,
        chat_options: OverridableChatOptionsDict | None = None,
        show_text_processing: bool = True,
        show_tools_toggle: bool = True,
    ):
        super().__init__()
        self.state = StateManager[ChatOptionsState](
            self.get_initial_state(chat_options or {})  # type: ignore
        )
        self._show_text_processing = show_text_processing
        self._show_tools_toggle = show_tools_toggle
        self._openai_refresh_inflight = False
        self._available_openai_models: list[str] | None = None
        self._chatgpt_refresh_inflight = False
        self._available_chatgpt_models: list[ChatGPTModel] | None = None
        self.setup_ui()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

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
        self.chat_model.setMinimumWidth(0)
        self.chat_model.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.chat_model.on_change.connect(self._on_model_change)
        self._sync_model_labels(self.state.s["chat_models"])

        self.show_available_models = ReactiveCheckBox(
            self.state, "show_available_openai_models"
        )
        self.show_available_models.setText("Show discovered OpenAI models")
        self.show_available_models.on_change.connect(
            self._on_show_available_openai_models_change
        )

        self.reasoning_effort = ReactiveComboBox(
            self.state,
            "chat_reasoning_efforts",
            "chat_reasoning_effort",
            reasoning_efforts_map,
        )
        self.reasoning_effort.on_change.connect(self._on_reasoning_effort_change)

        chat_box = QGroupBox("Language model")
        chat_form = default_form_layout()
        chat_box.setLayout(chat_form)
        chat_form.addRow("Provider:", self.chat_provider)
        chat_form.addRow("Model:", self.chat_model)
        self.model_warning = QLabel()
        self.model_warning.setWordWrap(True)
        chat_form.addRow(self.model_warning)
        self.discovery = QWidget()
        discovery_layout = QVBoxLayout(self.discovery)
        discovery_layout.setContentsMargins(0, 0, 0, 0)
        discovery_layout.addWidget(self.show_available_models)
        status_row = QHBoxLayout()
        self.discovery_status = QLabel()
        self.discovery_status.setWordWrap(True)
        self.discovery_status.setAccessibleDescription("OpenAI model discovery status")
        self.discovery_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.refresh_models = QPushButton("Refresh")
        self.refresh_models.setAccessibleName("Refresh available OpenAI models")
        self.refresh_models.clicked.connect(
            lambda: self._refresh_models_if_needed(self.state.s["chat_provider"])
        )
        status_row.addWidget(self.discovery_status, 1)
        status_row.addWidget(self.refresh_models)
        discovery_layout.addLayout(status_row)
        chat_form.addRow(self.discovery)
        cached = chat_provider.get_cached_openai_chat_models()
        self.discovery_status.setText(
            f"Cached list: {len(cached)} text models. Refresh to check availability."
            if cached
            else "Availability has not been checked for this connection."
        )

        text_rules = QGroupBox("Formatting")
        text_layout = default_form_layout()
        text_rules.setLayout(text_layout)
        text_rules.setHidden(not self._show_text_processing)
        self.convert_box = ReactiveCheckBox(self.state, "chat_markdown_to_html")
        self.convert_box.setText("Render Markdown formatting in Anki fields")
        text_layout.addRow(self.convert_box)
        convert_explainer = QLabel(
            "Render lists, emphasis and code blocks safely. When off, responses are saved as plain text."
        )
        convert_explainer.setWordWrap(True)
        convert_explainer.setFont(font_small)
        text_layout.addRow(convert_explainer)
        advanced = QGroupBox("Response controls")
        advanced_layout = self.response_form = default_form_layout()
        advanced.setLayout(advanced_layout)
        advanced_layout.addRow("Temperature:", self.temperature)
        self.temperature_help = QLabel()
        self.temperature_help.setWordWrap(True)
        self.temperature_help.setFont(font_small)
        advanced_layout.addRow(self.temperature_help)

        advanced_layout.addRow("Reasoning Effort:", self.reasoning_effort)
        self.reasoning_help = QLabel()
        self.reasoning_help.setWordWrap(True)
        self.reasoning_help.setFont(font_small)
        advanced_layout.addRow(self.reasoning_help)

        tools = QGroupBox("Tools")
        tools_layout = default_form_layout()
        tools.setLayout(tools_layout)
        tools.setHidden(not self._show_tools_toggle)
        self.use_tools_box = ReactiveCheckBox(self.state, "chat_use_tools")
        self.use_tools_box.setText("Use enabled tools by default")
        tools_layout.addRow(self.use_tools_box)
        tools_desc = QLabel(
            "Choose available tools on the Tools page. Individual Smart Fields can override this default. Tool calls may share referenced note data with an external server."
        )
        tools_desc.setWordWrap(True)
        tools_desc.setFont(font_small)
        tools_layout.addRow(tools_desc)

        chat_layout = QVBoxLayout(self)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(12)
        chat_layout.addWidget(chat_box)
        chat_layout.addWidget(advanced)
        if self._show_text_processing:
            chat_layout.addWidget(text_rules)
        if self._show_tools_toggle:
            chat_layout.addWidget(tools)
        chat_layout.addStretch()

        self._on_reasoning_effort_change(self.state.s.get("chat_reasoning_effort"))
        self._on_model_change(self.state.s["chat_model"], preserve_effort=True)
        self._update_editable_state(self.state.s["chat_provider"])
        self._update_available_models_visibility(self.state.s["chat_provider"])
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

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
        self._refresh_models_if_needed(self.state.s["chat_provider"])

    def _refresh_models_if_needed(self, provider: str) -> None:
        if provider == "chatgpt":
            self._refresh_chatgpt_models_if_needed()
        else:
            self._refresh_openai_models_if_needed(provider)

    def _update_editable_state(self, provider: str) -> None:
        is_custom = provider == "chatgpt" or provider not in all_chat_providers
        self.chat_model.setEditable(is_custom)
        self.chat_model.setPlaceholderText("Choose or enter a model ID")

    def _update_available_models_visibility(self, provider: str) -> None:
        self.discovery.setVisible(provider in {"openai", "chatgpt"})
        self.show_available_models.setVisible(provider == "openai")
        name = providers_map.get(provider, provider)
        self.discovery_status.setAccessibleDescription(f"{name} model discovery status")
        self.refresh_models.setAccessibleName(f"Refresh available {name} models")

    def _sync_model_labels(
        self, models: list[str], provider: str | None = None
    ) -> None:
        labels = dict(models_map)
        for model in models:
            if model.startswith("gpt-"):
                labels[model] = openai_model_label(model)

        if (provider or self.state.s["chat_provider"]) == "chatgpt":
            labels.update(
                {
                    model.model: model.label
                    for model in chat_provider.get_cached_chatgpt_models()
                }
            )

        self.chat_model.state_to_ui = labels
        self.chat_model.ui_to_state = {v: k for k, v in labels.items()}

    @with_widget_settings
    def _models_for_provider(self, provider: str, current_model: str) -> list[str]:
        if provider == "chatgpt":
            models = [
                model.model for model in chat_provider.get_cached_chatgpt_models()
            ]
            if current_model and current_model not in models:
                models.append(current_model)
            return models
        if provider == "openai":
            return openai_chat_models_for_display(
                chat_provider.get_cached_openai_chat_models(),
                include_available=self.state.s["show_available_openai_models"],
                current_model=current_model,
            )

        if provider in all_chat_providers:
            models = list(provider_model_map[provider])
            if current_model and current_model not in models:
                models.append(current_model)
            return models

        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        models = list(
            (custom_provider.get("chat_models") or custom_provider.get("models") or [])
            if custom_provider
            else []
        )
        if current_model and current_model not in models:
            models.append(current_model)
        return models

    def _on_provider_change(self, text: str) -> None:
        is_custom = text not in all_chat_providers
        last_settings = self.state.s["provider_settings"].get(text)
        current_model = (
            last_settings["model"]
            if last_settings
            else ""
            if text == "chatgpt"
            else self.state.s["chat_model"]
        )
        models = self._models_for_provider(text, current_model=current_model)
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
        elif models and text != "chatgpt":
            new_model = models[0]

        if text == "chatgpt" and not last_settings:
            new_effort = None
        efforts = self._reasoning_efforts(text, new_model)

        if text == "chatgpt":
            if new_effort and new_effort not in efforts:
                efforts.append(new_effort)
        elif new_effort not in efforts:
            new_effort = (
                "none" if "none" in efforts else (efforts[0] if efforts else None)
            )

        self._update_editable_state(text)
        self._update_available_models_visibility(text)
        self._sync_model_labels(models, text)
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

        self._update_capabilities()

        self._update_provider_settings(
            text,
            model=new_model,
            temperature=new_temp,
            reasoning_effort=new_effort,
        )  # type: ignore
        self._refresh_models_if_needed(text)

    def refresh_custom_providers(self) -> None:
        custom_provider_names = [p["name"] for p in (config.custom_providers or [])]
        new_providers = all_chat_providers + custom_provider_names

        self.state.update({"chat_providers": new_providers})

        current = self.state.s["chat_provider"]
        if current not in new_providers:
            self.state.update({"chat_providers": [*new_providers, current]})
            self._update_capabilities()
        elif current in custom_provider_names:
            self._on_provider_change(current)
        elif current in {"openai", "chatgpt"}:
            self._refresh_models_if_needed(current)

    def get_initial_state(
        self, chat_options: OverridableChatOptionsDict
    ) -> ChatOptionsState:
        ret: ChatOptionsState = {
            k: key_or_config_val(chat_options, k)
            for k in overridable_chat_options  # type: ignore
        }
        ret["chat_use_tools"] = config.chat_use_tools
        ret["show_available_openai_models"] = False

        custom_provider_names = [p["name"] for p in (config.custom_providers or [])]
        ret["chat_providers"] = all_chat_providers + custom_provider_names
        ret["provider_settings"] = deepcopy(config.provider_settings or {})

        current_provider = ret["chat_provider"]
        if current_provider in provider_model_map:
            if current_provider == "chatgpt":
                ret["chat_models"] = [
                    model.model for model in chat_provider.get_cached_chatgpt_models()
                ]
            elif current_provider == "openai":
                ret["chat_models"] = openai_chat_models_for_display(
                    chat_provider.get_cached_openai_chat_models(),
                    include_available=False,
                    current_model=ret.get("chat_model"),
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
            ret["chat_models"] = (
                (
                    custom_provider.get("chat_models")
                    or custom_provider.get("models")
                    or []
                )
                if custom_provider
                else []
            )

        if current_provider not in ret["chat_providers"]:
            ret["chat_providers"].append(current_provider)
        current_model = ret.get("chat_model")
        ret["chat_models"] = list(ret["chat_models"])
        if current_model and current_model not in ret["chat_models"]:
            ret["chat_models"].append(current_model)
        efforts = (
            self._reasoning_efforts(current_provider, current_model)
            if current_model
            else []
        )
        ret["chat_reasoning_efforts"] = efforts

        if (
            current_provider != "chatgpt"
            and not ret.get("chat_reasoning_effort")
            and efforts
        ):
            ret["chat_reasoning_effort"] = (
                "none" if "none" in efforts else efforts[0] if efforts else None
            )

        return ret

    def _reasoning_efforts(self, provider: str, model: str) -> list[str]:
        if provider != "chatgpt":
            return openai_reasoning_efforts_for_model(model)
        return next(
            (
                list(item.reasoning_efforts)
                for item in chat_provider.get_cached_chatgpt_models()
                if item.model == model
            ),
            [],
        )

    def _on_model_change(self, model: str, preserve_effort: bool = False) -> None:
        provider = self.state.s["chat_provider"]
        efforts = self._reasoning_efforts(provider, model)
        saved_effort = self.state.s.get("chat_reasoning_effort")
        if preserve_effort and saved_effort is not None and saved_effort not in efforts:
            efforts.append(saved_effort)

        self._update_provider_settings(provider, model=model)

        self.state.update({"chat_reasoning_efforts": efforts})

        current_effort = self.state.s.get("chat_reasoning_effort")
        if current_effort not in efforts and not (
            provider == "chatgpt" and preserve_effort
        ):
            new_effort = (
                "none" if "none" in efforts else (efforts[0] if efforts else None)
            )
            if provider == "chatgpt":
                new_effort = next(
                    (
                        item.default_reasoning_effort
                        for item in chat_provider.get_cached_chatgpt_models()
                        if item.model == model
                    ),
                    None,
                )
            self.state.update({"chat_reasoning_effort": new_effort})
            self._update_provider_settings(provider, reasoning_effort=new_effort)

        self._update_capabilities()

    def _on_reasoning_effort_change(self, effort: str | None) -> None:
        self.state.update({"chat_reasoning_effort": effort})  # type: ignore
        self._update_provider_settings(
            self.state.s["chat_provider"], reasoning_effort=effort
        )  # type: ignore

        self._update_capabilities()

    @with_widget_settings
    def _update_capabilities(self) -> None:
        provider = self.state.s["chat_provider"]
        model = self.state.s["chat_model"]
        effort = self.state.s["chat_reasoning_effort"]
        supports_reasoning = (
            provider in {"openai", "chatgpt"} or provider not in all_chat_providers
        )
        self.chat_model.setToolTip(model)
        self.reasoning_effort.setEnabled(supports_reasoning)
        supports_temperature = provider != "chatgpt" and (
            not supports_reasoning or openai_supports_temperature(model, effort)
        )
        self.response_form.setRowVisible(self.temperature, provider != "chatgpt")
        self.response_form.setRowVisible(self.temperature_help, provider != "chatgpt")
        self.temperature.setEnabled(supports_temperature)
        self.temperature_help.setText(
            "Higher values vary the wording more; lower values make responses more consistent (0–2)."
            if supports_temperature
            else f"{model} does not support temperature."
            if not openai_supports_temperature(model, None)
            else "Temperature is unavailable while reasoning is enabled. Choose None for reasoning to use temperature."
        )
        self.reasoning_help.setText(
            "Higher effort can take longer and use more tokens. Available levels depend on the model."
            if supports_reasoning
            else f"Reasoning effort is not sent to {providers_map.get(provider, provider)} by this plugin."
        )
        custom = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        supports_tools = (
            not custom
            or custom.get("chat_api_mode") != "chat_completions"
            or openai_supports_chat_tools(model)
        )
        if hasattr(self, "use_tools_box"):
            self.use_tools_box.setEnabled(supports_tools)
        unavailable = (provider, model) in RETIRED_PROVIDER_MODELS or (
            provider not in all_chat_providers and custom is None
        )
        absent_from_connection = (
            provider == "openai"
            and self._available_openai_models is not None
            and model not in self._available_openai_models
        )
        if provider == "chatgpt":
            absent_from_connection = (
                self._available_chatgpt_models is not None
                and model not in {item.model for item in self._available_chatgpt_models}
            )
        invalid_effort = (
            provider == "chatgpt"
            and effort is not None
            and effort not in self._reasoning_efforts(provider, model)
        )
        self.model_warning.setText(
            "Saved provider or model is unavailable. Choose a supported selection before generating."
            if unavailable
            else "The saved model was not returned by this connection. It is preserved; verify access or choose a listed model."
            if absent_from_connection
            else "The saved reasoning effort was not returned for this model. Choose an available level before generating."
            if invalid_effort
            else "This model requires Responses mode for tool calling."
            if not supports_tools
            else ""
        )
        self.model_warning.setVisible(
            unavailable
            or absent_from_connection
            or invalid_effort
            or not supports_tools
        )

    def _on_show_available_openai_models_change(self, _: bool) -> None:
        if self.state.s["chat_provider"] != "openai":
            return

        current_model = self.state.s["chat_model"]
        models = self._models_for_provider("openai", current_model=current_model)
        self._sync_model_labels(models)

        self.state.update({"chat_models": models})

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

    @with_widget_settings
    def _refresh_chatgpt_models_if_needed(self) -> None:
        if not self.isVisible() or self._chatgpt_refresh_inflight:
            return
        if chatgpt_auth.status() is None:
            self.discovery_status.setText(
                "Sign in with ChatGPT on the Providers page, then refresh model availability."
            )
            self.refresh_models.setEnabled(False)
            return

        account_generation = chatgpt_auth.generation
        self._chatgpt_refresh_inflight = True
        self._available_chatgpt_models = None
        self.refresh_models.setEnabled(False)
        self.discovery_status.setText("Checking ChatGPT model availability…")

        def on_success(models: list[ChatGPTModel]) -> None:
            if self.state.s["chat_provider"] != "chatgpt":
                return
            if account_generation != chatgpt_auth.generation:
                self.discovery_status.setText(
                    "Account changed. Refresh to check availability."
                )
                return
            self._available_chatgpt_models = models
            self.discovery_status.setText(
                f"Fetched {len(models)} ChatGPT models. Choose or enter a model ID above."
                if models
                else "No models were returned. Your selection is preserved; you can enter a model ID."
            )
            display_models = self._models_for_provider(
                "chatgpt", self.state.s["chat_model"]
            )
            self._sync_model_labels(display_models)
            self.state.update({"chat_models": display_models})
            self._on_model_change(self.state.s["chat_model"], preserve_effort=True)

        def on_failure(error: Exception) -> None:
            if self.state.s["chat_provider"] != "chatgpt":
                return
            self.refresh_models.setText("Retry")
            self.discovery_status.setText(
                f"Could not check availability: {redact_text(str(error))}. Your selection is preserved; you can enter a model ID."
            )

        def finish() -> None:
            self._chatgpt_refresh_inflight = False
            if not sip.isdeleted(self):
                self.refresh_models.setEnabled(True)

        try:
            run_async_in_background_with_sentry(
                chat_provider.fetch_chatgpt_models,
                on_success,
                on_failure,
                show_error=False,
                on_finished=finish,
            )
        except Exception:
            finish()
            raise

    @with_widget_settings
    def _refresh_openai_models_if_needed(self, provider: str) -> None:
        if (
            not self.isVisible()
            or provider != "openai"
            or self._openai_refresh_inflight
        ):
            return
        if not config.openai_api_key:
            self.discovery_status.setText(
                "Add an OpenAI API key on the Providers page to check model availability."
            )
            self.refresh_models.setEnabled(False)
            return

        connection = (config.openai_endpoint, config.openai_api_key)
        self._available_openai_models = None
        self._openai_refresh_inflight = True
        self.refresh_models.setEnabled(False)
        self.refresh_models.setText("Refresh")
        cached = chat_provider.get_cached_openai_chat_models()
        self.discovery_status.setText(
            f"Refreshing model availability… Showing {len(cached)} cached text models."
            if cached
            else "Checking model availability…"
        )

        def on_success(models: list[str]) -> None:
            if self.state.s["chat_provider"] != "openai":
                return
            self.refresh_models.setEnabled(True)
            current_settings = widget_settings(self)
            if connection != (
                current_settings.get("openai_endpoint"),
                current_settings.get("openai_api_key"),
            ):
                self.discovery_status.setText(
                    "Connection changed while loading. Refresh to check the current credentials."
                )
                return
            self._available_openai_models = models
            self.discovery_status.setText(
                f"Fetched {len(models)} text models. Availability checked for this connection."
                if models
                else "This connection returned no supported text models. Saved selections are preserved."
            )
            display_models = openai_chat_models_for_display(
                models,
                include_available=self.state.s["show_available_openai_models"],
                current_model=self.state.s["chat_model"],
            )
            self._sync_model_labels(display_models)
            self.state.update({"chat_models": display_models})
            self._update_capabilities()

        def on_failure(error: Exception) -> None:
            if self.state.s["chat_provider"] != "openai":
                return
            self.refresh_models.setEnabled(True)
            self.refresh_models.setText("Retry")
            fallback = (
                f" Keeping {len(cached)} cached models."
                if cached
                else " Saved selections are preserved."
            )
            self.discovery_status.setText(
                f"Could not check availability: {redact_text(str(error))}.{fallback}"
            )

        def finish() -> None:
            self._openai_refresh_inflight = False

        try:
            run_async_in_background_with_sentry(
                chat_provider.fetch_openai_chat_models,
                on_success,
                on_failure,
                show_error=False,
                on_finished=finish,
            )
        except Exception:
            finish()
            self.refresh_models.setEnabled(True)
            raise
