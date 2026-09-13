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

from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, Literal, TypedDict, cast

from anki.decks import DeckId
from anki.notes import Note
from aqt import (
    QAction,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    Qt,
    QTabWidget,
    QTextEdit,
    QTextOption,
    QTimer,
    QVBoxLayout,
    QWidget,
    mw,
)
from aqt.qt import QCursor
from PyQt6 import sip

from ..chat_usage import (
    build_prompt_usage_key,
    build_prompt_usage_signature,
    chat_usage_tracker,
    format_token_count,
)
from ..config import config, key_or_config_val
from ..constants import API_KEY_MISSING_MESSAGE, GLOBAL_DECK_ID
from ..dag import prompt_has_error
from ..decks import deck_id_to_name_map, get_all_deck_ids
from ..field_processor import PromptUsageContext
from ..logger import logger, redact_text
from ..media_utils import image_preview_data
from ..models import (
    DEFAULT_EXTRAS,
    OverridableChatOptionsDict,
    OverridableImageOptionsDict,
    OverrideableTTSOptionsDict,
    PromptMap,
    SmartFieldType,
    TTSModels,
    TTSProviders,
    TTSVoiceTarget,
    openai_supports_temperature,
    overridable_chat_options,
    overridable_image_options,
    overridable_tts_options,
)
from ..note_processor import NoteProcessor, snapshot_note
from ..notes import get_note_types, get_random_note, get_valid_fields_for_prompt
from ..prompts import (
    add_or_update_prompts,
    get_extras,
    get_prompt_fields,
    get_prompts_for_note,
    interpolate_prompt_with_values,
    remove_prompt,
)
from ..sentry import run_async_in_background_with_sentry
from ..settings import resolve_field_settings, with_widget_settings
from ..tts_routing import select_tts_target
from ..tts_utils import play_audio
from ..utils import get_fields
from .chat_options import ChatOptions, all_chat_providers
from .image_displayer import ImageDisplayer
from .image_options import ImageOptions, image_output_summary
from .reactive_check_box import ReactiveCheckBox
from .reactive_combo_box import ReactiveComboBox
from .reactive_edit_text import ReactiveEditText
from .reactive_line_edit import ReactiveLineEdit
from .state_manager import StateManager
from .tts_options import TTSOptions
from .ui_utils import (
    bound_dialog,
    default_form_layout,
    font_bold,
    scroll_page,
    show_message_box,
)

PROVIDER_API_KEY_ATTRS: dict[str, str] = {
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "deepseek": "deepseek_api_key",
    "google": "google_api_key",
    "elevenLabs": "elevenlabs_api_key",
    "fish": "fish_api_key",
    "replicate": "replicate_api_key",
}


explanation = """Write a prompt to help the chat model generate your Smart Field.

Your prompt may reference other fields via {{double curly braces}}. Valid fields are listed below for convenience.

Use the Preview tab to inspect a sample and test the result.
"""

tts_explanation = """Write text to be spoken, or include the field to speak.

You can use {{double curly braces}} to reference fields.
Example: "{{Front}}" or "Hello {{Front}}"
"""


class State(TypedDict):
    prompt: str
    note_types: list[str]
    selected_note_type: str
    note_fields: list[str]
    tts_source_fields: list[str]
    selected_tts_source_field: str
    selected_note_field: str
    is_loading_prompt: bool
    generate_automatically: bool

    use_custom_model: bool
    type: SmartFieldType

    selected_deck: DeckId
    decks: list[DeckId]
    regenerate_when_batching: bool
    tts_style: str
    tts_language: str
    chat_use_tools: bool


class PartialState(TypedDict):
    prompt: str
    note_fields: list[str]
    selected_note_field: str
    tts_source_fields: list[str]
    selected_tts_source_field: str
    selected_note_type: str
    selected_deck: DeckId
    tts_style: str
    tts_language: str


class PromptDialog(QDialog):
    prompt_text_box: QTextEdit
    tts_style_box: QTextEdit
    test_button: QPushButton
    valid_fields: QLabel
    note_combo_box: QComboBox
    state: StateManager[State]
    prompts_map: PromptMap
    chat_options: ChatOptions
    image_options: ImageOptions
    tts_options: TTSOptions
    mode: Literal["new", "edit"]
    field_type: SmartFieldType
    token_usage_label: QLabel

    def __init__(
        self,
        prompts_map: PromptMap,
        processor: NoteProcessor,
        on_accept_callback: Callable[[PromptMap], None],
        field_type: SmartFieldType,
        deck_id: DeckId,
        card_type: str | None = None,
        field: str | None = None,
        prompt: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.processor = processor
        self.on_accept_callback = on_accept_callback
        self.prompts_map = prompts_map
        self.mode = "edit" if card_type else "new"
        self.field_type = field_type
        self.original_identity = (
            (card_type, deck_id, field) if card_type and field else None
        )
        self.sample_note: Note | None = None
        known_types = get_note_types()
        self.repairing = bool(
            card_type
            and (
                card_type not in known_types
                or field not in get_fields(card_type)
                or deck_id not in get_all_deck_ids()
            )
        )
        note_types = self._get_note_types(deck_id=deck_id)
        if card_type in known_types and card_type not in note_types:
            note_types.insert(0, card_type)

        # Ensure there are valid fields to select
        if not len(note_types):
            show_message_box(
                "No valid note types left. Edit or delete some fields to continue!"
            )
            QTimer.singleShot(0, self.close)
            return

        selected_note_type = card_type if card_type in note_types else note_types[0]
        selected_deck = deck_id if deck_id in get_all_deck_ids() else GLOBAL_DECK_ID
        default_note_state = self._state_for_new_card_type(
            selected_note_type, field_type, deck_id=selected_deck
        )
        selected_target_field = (
            field
            if field in get_fields(selected_note_type)
            else default_note_state["selected_note_field"]
        )

        extras = (
            get_extras(
                note_type=card_type or selected_note_type,
                field=field or selected_target_field,
                prompts=self.prompts_map,
                deck_id=deck_id,
            )
            or DEFAULT_EXTRAS
        )

        # Only if it's a new card, we need to get the fields for the selected card type
        target_fields = (
            default_note_state["note_fields"]
            if self.mode == "new" or self.repairing
            else get_fields(selected_note_type)
        )

        # If it's an edit, we need to get the source field from the prompt
        selected_tts_source_field = (
            (self._attempt_to_parse_source_field(prompt) or "")
            if prompt
            else default_note_state["selected_tts_source_field"]
        )

        initial_state: State = {
            "prompt": prompt or default_note_state["prompt"],
            # Note types
            "selected_note_type": selected_note_type,
            "note_types": note_types,
            # tts
            "tts_source_fields": default_note_state["tts_source_fields"],
            "selected_tts_source_field": selected_tts_source_field,
            "tts_style": extras.get("tts_style") or "",
            "tts_language": extras.get("tts_language") or "",
            # target fields
            "note_fields": target_fields,
            "selected_note_field": selected_target_field,
            # other
            "is_loading_prompt": False,
            "decks": get_all_deck_ids(),
            "selected_deck": selected_deck,
            "type": field_type,
            "generate_automatically": extras["automatic"],
            "use_custom_model": extras["use_custom_model"],
            "regenerate_when_batching": extras.get("regenerate_when_batching", False),
            "chat_use_tools": key_or_config_val(extras, "chat_use_tools")
            if field_type == "chat"
            else False,
        }
        self.state = StateManager[State](initial_state)

        self.enabled_box = ReactiveCheckBox(
            self.state,
            "generate_automatically",
            text="Generate this field automatically",
        )
        self.standard_buttons = self.create_buttons()

        tabs = QTabWidget()
        tabs.addTab(scroll_page(self.render_main_tab()), "Field")
        tabs.addTab(scroll_page(self.render_options_tab()), "Options")
        tabs.addTab(scroll_page(self.render_preview_tab()), "Preview")

        container = QVBoxLayout()
        container.addWidget(tabs)

        scope = QLabel(
            "Done updates the settings draft. Save in the main settings window to keep it."
        )
        scope.setWordWrap(True)
        container.addWidget(scope)
        container.addWidget(self.standard_buttons)
        self.setLayout(container)
        self.state.state_changed.connect(self.render_ui)
        self.setup_ui()
        bound_dialog(self)

    def setup_ui(self) -> None:
        self.render_ui()

    def render_ui(self) -> None:
        self.render_buttons()
        self.render_automatic_button()
        if hasattr(self, "token_usage_label"):
            self.render_token_usage()
        if hasattr(self, "effective_summary"):
            self.render_effective_settings()
        if hasattr(self, "sample_fields"):
            self.render_sample()

    def render_main_tab(self) -> QWidget:
        layout = QVBoxLayout()

        field_type = self.state.s["type"]
        media_name = {"chat": "Text", "tts": "Speech", "image": "Image"}[field_type]
        self.setWindowTitle(
            f"{'Edit' if self.mode == 'edit' else 'New'} {media_name} Smart Field"
        )
        if self.repairing:
            repair = QLabel(
                "This Smart Field refers to a deleted target. Choose a current note type, deck, and field below. Your prompt and options are preserved."
            )
            repair.setWordWrap(True)
            layout.addWidget(repair)

        # 1. Settings Area
        settings_group = QGroupBox("Settings")
        form_layout = default_form_layout()

        self.note_combo_box = ReactiveComboBox(
            self.state, "note_types", "selected_note_type"
        )
        form_layout.addRow("Note Type:", self.note_combo_box)

        self.deck_combo_box = ReactiveComboBox(
            self.state,
            "decks",
            "selected_deck",
            render_map={str(k): v for k, v in deck_id_to_name_map().items()},
            int_keys=True,
        )
        self.deck_combo_box.setToolTip(
            "Optionally apply this field only to a specific deck (useful for sharing note types between decks)."
        )
        form_layout.addRow("Deck:", self.deck_combo_box)

        if self.state.s["type"] == "tts":
            self.tts_source_combo_box = ReactiveComboBox(
                self.state, "tts_source_fields", "selected_tts_source_field"
            )
            self.tts_source_combo_box.on_change.connect(self.on_source_changed)
            self.tts_source_combo_box.setToolTip("The field that will be spoken.")
            form_layout.addRow("Source Field:", self.tts_source_combo_box)

            self.tts_language_box = ReactiveLineEdit(self.state, "tts_language")
            self.tts_language_box.setPlaceholderText("e.g. ja, en-US, Japanese")
            self.tts_language_box.setToolTip(
                "Optional language tag or name used to route this field to matching voices. Blank uses any enabled voice."
            )
            self.tts_language_box.on_change.connect(
                lambda text: self.state.update({"tts_language": text})
            )
            form_layout.addRow("Language (optional):", self.tts_language_box)

        self.field_combo_box = ReactiveComboBox(
            self.state, "note_fields", "selected_note_field"
        )
        self.field_combo_box.setToolTip(
            {
                "chat": "The field that stores generated text.",
                "tts": "The field that stores and plays generated audio.",
                "image": "The field that displays the generated image.",
            }[field_type]
        )
        form_layout.addRow("Target Field:", self.field_combo_box)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        # 2. Prompting Area
        prompt_group = QGroupBox("Speech content" if field_type == "tts" else "Prompt")
        prompt_layout = QVBoxLayout()

        # Style Instructions for Gemini TTS
        if self.state.s["type"] == "tts":
            style_label = QLabel("Voice style (optional)")
            style_label.setFont(font_bold)
            self.tts_style_box = ReactiveEditText(self.state, "tts_style")
            self.tts_style_box.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.tts_style_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
            self.tts_style_box.setWordWrapMode(
                QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
            )
            self.tts_style_box.setPlaceholderText(
                'e.g. "Read aloud in a warm and friendly tone: "'
            )
            self.tts_style_box.setMinimumHeight(60)
            self.tts_style_box.setMaximumHeight(80)
            self.tts_style_box.on_change.connect(
                lambda text: self.state.update({"tts_style": text})
            )

            prompt_layout.addWidget(style_label)
            prompt_layout.addWidget(self.tts_style_box)
            style_help = QLabel(
                "Style applies to Google Gemini voices and OpenAI gpt-4o-mini-tts. Other models speak the content without these instructions."
            )
            style_help.setWordWrap(True)
            prompt_layout.addWidget(style_help)

        # Prompt Label + Insert Button
        prompt_label_layout = QHBoxLayout()
        prompt_label = QLabel(
            {
                "chat": "Instructions",
                "tts": "Text to speak",
                "image": "Image description",
            }[field_type]
        )
        prompt_label.setFont(font_bold)
        prompt_label_layout.addWidget(prompt_label)
        prompt_label_layout.addStretch()

        insert_btn = QPushButton("Insert field…")
        insert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        insert_btn.clicked.connect(self.show_insert_field_menu)
        prompt_label_layout.addWidget(insert_btn)

        prompt_layout.addLayout(prompt_label_layout)

        self.prompt_text_box = ReactiveEditText(self.state, "prompt")
        self.prompt_text_box.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.prompt_text_box.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.prompt_text_box.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )

        current_explanation = (
            tts_explanation
            if field_type == "tts"
            else "Describe the image to generate. Reference note fields using {{Front}} or another field name."
            if field_type == "image"
            else explanation
        )
        self.prompt_text_box.setPlaceholderText(current_explanation)
        self.prompt_text_box.setMinimumHeight(120)

        prompt_layout.addWidget(self.prompt_text_box)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # 3. Footer
        self.test_button = QPushButton("Generate preview")
        self.test_button.setCursor(Qt.CursorShape.PointingHandCursor)

        footer_layout = QHBoxLayout()

        # Enabled Box with tooltip explanation
        self.enabled_box.setToolTip(
            "When off, you can still generate this field manually from the editor. Automatic generation also follows this profile’s Generation settings."
        )
        footer_layout.addWidget(self.enabled_box)
        footer_layout.addStretch()

        layout.addLayout(footer_layout)

        self.note_combo_box.on_change.connect(self._on_new_card_type_selected)
        self.field_combo_box.on_change.connect(self.on_target_field_changed)
        self.deck_combo_box.on_change.connect(self.on_deck_selected)
        self.prompt_text_box.on_change.connect(
            lambda text: self.state.update({"prompt": text})
        )

        self.test_button.clicked.connect(self.on_test)

        # On small screens, make it a proportion of screen height. Otherwise set a fixed height

        self.setMinimumHeight(400)
        container = QWidget()
        container.setLayout(layout)

        # Control visibility depending on mode
        if self.mode == "edit" and not self.repairing:
            self.note_combo_box.setEnabled(False)
            self.field_combo_box.setEnabled(False)
            self.deck_combo_box.setEnabled(False)
            if hasattr(self, "tts_source_combo_box"):
                self.tts_source_combo_box.setEnabled(False)
        return container

    def show_insert_field_menu(self):
        menu = QMenu(self)
        fields = get_valid_fields_for_prompt(
            selected_note_type=self.state.s["selected_note_type"],
            selected_note_field=self.state.s["selected_note_field"],
            deck_id=self.state.s["selected_deck"],
            prompts_map=self.prompts_map,
        )

        if not fields:
            no_fields = QAction("No fields available", self)
            no_fields.setEnabled(False)
            menu.addAction(no_fields)
        else:
            for field in fields:
                action = QAction(field, self)
                action.triggered.connect(lambda _, f=field: self.insert_field_token(f))
                menu.addAction(action)

        menu.exec(QCursor.pos())

    def insert_field_token(self, field: str):
        self.prompt_text_box.insertPlainText(f"{{{{{field}}}}}")
        self.prompt_text_box.setFocus()

    def render_options_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.effective_summary = QLabel()
        self.effective_summary.setWordWrap(True)
        self.effective_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.effective_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.effective_summary)
        self.custom_model = ReactiveCheckBox(
            self.state,
            "use_custom_model",
            text="Customize model and output settings for this field",
        )
        layout.addWidget(self.custom_model)
        self.model_options = self.render_custom_model()
        self.model_options.setVisible(self.state.s["use_custom_model"])
        layout.addWidget(self.model_options)
        self.regenerate_batch_checkbox = ReactiveCheckBox(
            self.state,
            "regenerate_when_batching",
            text="Overwrite this field during batch generation",
        )
        layout.addWidget(self.regenerate_batch_checkbox)
        batch_desc = QLabel(
            "This applies only to batches. Manual generation can replace a selected field; review generation follows the profile’s settings."
        )
        batch_desc.setWordWrap(True)
        layout.addWidget(batch_desc)
        if self.state.s["type"] == "chat":
            self.chat_use_tools_checkbox = ReactiveCheckBox(
                self.state,
                "chat_use_tools",
                text="Allow this field to use enabled tools",
            )
            layout.addWidget(self.chat_use_tools_checkbox)
            tools_desc = QLabel(
                "Tool permission is stored separately from model overrides. Make tools available on the main Tools page."
            )
            tools_desc.setWordWrap(True)
            layout.addWidget(tools_desc)
            token_group = QGroupBox("Estimated token usage")
            token_layout = QVBoxLayout(token_group)
            self.token_usage_label = QLabel()
            self.token_usage_label.setWordWrap(True)
            token_layout.addWidget(self.token_usage_label)
            layout.addWidget(token_group)
        layout.addStretch()
        self.state.state_changed.connect(self.on_state_update)
        self.render_effective_settings()
        return container

    @with_widget_settings
    def render_effective_settings(self) -> None:
        kind = self.state.s["type"]
        options = self.model_options.state.s
        settings = resolve_field_settings(
            config.snapshot(), options if self.state.s["use_custom_model"] else {}
        )
        origin = (
            "Custom settings"
            if self.state.s["use_custom_model"]
            else "Using the defaults from the main settings draft"
        )
        if kind == "tts":
            voices = [
                voice
                for voice in settings["tts_voice_pool"]
                if voice.get("enabled", True)
            ]
            details = "; ".join(
                f"{voice['provider']} / {voice['model']} / {voice['voice']}"
                for voice in voices[:3]
            )
            if len(voices) > 3:
                details += f"; and {len(voices) - 3} more"
            summary = f"{len(voices)} enabled voices: {details or 'Configure a voice on the Speech page.'}"
        else:
            summary = f"{settings[f'{kind}_provider']} / {settings[f'{kind}_model']}"
            if kind == "chat":
                provider = settings["chat_provider"]
                effort = settings.get("chat_reasoning_effort")
                supports_reasoning = (
                    provider == "openai" or provider not in all_chat_providers
                )
                supports_temperature = (
                    not supports_reasoning
                    or openai_supports_temperature(settings["chat_model"], effort)
                )
                details = [
                    f"Temperature: {settings.get('chat_temperature', 1)}"
                    if supports_temperature
                    else "Temperature: not used"
                ]
                if supports_reasoning:
                    details.append(f"Reasoning: {effort or 'default'}")
                summary += "\n" + " · ".join(details)
                summary += f"\nMarkdown formatting: {'on' if settings['chat_markdown_to_html'] else 'off'}"
            else:
                summary += "\n" + image_output_summary(
                    settings["image_provider"],
                    settings["image_model"],
                    settings["image_aspect_ratio"],
                    settings["image_resolution"],
                    settings.get("image_output_format") or "webp",
                )
        self.effective_summary.setText(f"{origin}\n{summary}")

    def render_preview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        help_text = QLabel(
            "Load a sample to inspect the referenced note data and effective provider before generating. A preview sends this data to that provider and may incur charges. It does not change the note."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.load_sample_button = QPushButton("Load sample note")
        self.load_sample_button.clicked.connect(self.load_sample)
        layout.addWidget(self.load_sample_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.sample_description = QLabel("No sample loaded.")
        self.sample_description.setWordWrap(True)
        self.sample_description.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.sample_description)
        self.sample_fields = QPlainTextEdit()
        self.sample_fields.setReadOnly(True)
        self.sample_fields.setAccessibleName("Referenced sample note data")
        self.sample_fields.setPlaceholderText(
            "Referenced note fields will appear here before anything is sent."
        )
        layout.addWidget(self.sample_fields, 1)
        layout.addWidget(self.test_button, alignment=Qt.AlignmentFlag.AlignRight)
        return page

    def load_sample(self) -> None:
        self.sample_note = get_random_note(
            self.state.s["selected_note_type"], deck_id=self.state.s["selected_deck"]
        )
        self.render_sample()
        self.render_buttons()

    @with_widget_settings
    def render_sample(self) -> None:
        if self.sample_note is None:
            self.sample_description.setText(
                "No sample loaded. Load one to enable Generate preview."
            )
            self.sample_fields.clear()
            return
        kind = self.state.s["type"]
        settings = resolve_field_settings(
            config.snapshot(),
            self.model_options.state.s if self.state.s["use_custom_model"] else {},
        )
        try:
            if kind == "tts":
                voice = self.get_effective_tts_target(
                    f"{self.sample_note.id}:{self.state.s['selected_note_field'].lower()}"
                )
                destination = (
                    f"{voice['provider']} / {voice['model']} / {voice['voice']}"
                )
            else:
                destination = (
                    f"{settings[f'{kind}_provider']} / {settings[f'{kind}_model']}"
                )
        except ValueError as error:
            destination = str(error)
        self.sample_description.setText(
            f"Sample: {self.state.s['selected_note_type']} · note {self.sample_note.id}\nProvider: {destination}"
        )
        values = {name.lower(): value for name, value in self.sample_note.items()}
        fields = get_prompt_fields(self.state.s["prompt"])
        self.sample_fields.setPlainText(
            "\n\n".join(
                f"{name}:\n{values.get(name, '[Missing field]')}" for name in fields
            )
            or "This prompt references no note fields."
        )

    def create_buttons(self) -> QWidget:
        standard_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )

        done_button = standard_buttons.button(QDialogButtonBox.StandardButton.Save)
        assert done_button is not None
        done_button.setText("Done")
        standard_buttons.accepted.connect(self.on_accept)
        standard_buttons.rejected.connect(self.reject)
        return standard_buttons

    def render_custom_model(self) -> ChatOptions | TTSOptions | ImageOptions:
        extras = get_extras(
            note_type=self.original_identity[0]
            if self.original_identity
            else self.state.s["selected_note_type"],
            field=self.original_identity[2]
            if self.original_identity
            else self.state.s["selected_note_field"],
            deck_id=self.original_identity[1]
            if self.original_identity
            else self.state.s["selected_deck"],
            prompts=self.prompts_map,
            fallback_to_global_deck=False,
        )

        use_custom_model = extras and extras["use_custom_model"]

        if self.state.s["type"] == "tts":
            self.tts_options = TTSOptions(
                {
                    "tts_voice_pool": extras.get("tts_voice_pool"),
                    "tts_strip_html": extras.get("tts_strip_html"),
                }
                if extras and use_custom_model
                else None
            )
            options = self.tts_options
        elif self.state.s["type"] == "chat":
            self.chat_options = ChatOptions(
                cast(
                    "OverridableChatOptionsDict",
                    {key: extras.get(key) for key in overridable_chat_options},
                )
                if extras and use_custom_model
                else None,
                show_tools_toggle=False,
            )
            options = self.chat_options
        else:
            self.image_options = ImageOptions(
                cast(
                    "OverridableImageOptionsDict",
                    {key: extras.get(key) for key in overridable_image_options},
                )
                if extras and use_custom_model
                else None
            )
            options = self.image_options
        options.state.state_changed.connect(self.on_state_update)
        return options

    @with_widget_settings
    def _sync_regenerate_flag(self) -> None:
        extras = get_extras(
            note_type=self.state.s["selected_note_type"],
            field=self.state.s["selected_note_field"],
            deck_id=self.state.s["selected_deck"],
            prompts=self.prompts_map,
            fallback_to_global_deck=True,
        )
        updates: dict[str, Any] = {
            "regenerate_when_batching": extras.get("regenerate_when_batching", False)
            if extras
            else False,
            "chat_use_tools": key_or_config_val(extras, "chat_use_tools")
            if self.state.s["type"] == "chat"
            else False,
        }
        if self.state.s["type"] == "tts":
            updates["tts_style"] = (extras.get("tts_style") or "") if extras else ""
            updates["tts_language"] = (
                (extras.get("tts_language") or "") if extras else ""
            )
        self.state.update(updates)

    def on_state_update(self):
        self.model_options.setVisible(self.state.s["use_custom_model"])
        self.render_effective_settings()
        if hasattr(self, "sample_fields"):
            self.render_sample()
        if hasattr(self, "token_usage_label"):
            self.render_token_usage()

    def _get_note_types(self, deck_id: DeckId) -> list[str]:
        """Returns note types for which there are valid target fields remaining"""
        note_types = get_note_types()
        # Need to find a note type where there are valid field
        return [
            note_type
            for note_type in note_types
            if self._valid_fields_remain(note_type, deck_id=deck_id)
        ]

    def _valid_fields_remain(self, note_type: str, deck_id: DeckId) -> bool:
        target_fields = self._get_valid_target_fields(note_type, deck_id=deck_id)
        return len(target_fields) > 0

    def _state_for_new_card_type(
        self, note_type: str, type: SmartFieldType, deck_id: DeckId
    ) -> PartialState:
        target_fields = self._get_valid_target_fields(note_type, deck_id=deck_id)
        target_field = target_fields[0] if len(target_fields) else "None"

        source_fields = get_valid_fields_for_prompt(
            note_type, deck_id=deck_id, prompts_map=self.prompts_map
        )
        source_field = self._get_initial_source_field(note_type, deck_id=deck_id)
        prompt = self.get_tts_prompt(source_field) if type == "tts" else ""

        return {
            "selected_note_type": note_type,
            "prompt": prompt,
            "selected_deck": deck_id,
            "selected_note_field": target_field,
            "note_fields": target_fields,
            "tts_source_fields": source_fields,
            "selected_tts_source_field": source_field,
            "tts_style": "",
            "tts_language": "",
        }

    def _on_new_card_type_selected(self, note_type: str) -> None:
        new_state = self._state_for_new_card_type(
            note_type=note_type,
            type=self.state.s["type"],
            deck_id=self.state.s["selected_deck"],
        )
        if self.repairing:
            new_state["prompt"] = self.state.s["prompt"]
            new_state["tts_style"] = self.state.s["tts_style"]
            new_state["tts_language"] = self.state.s["tts_language"]
        self.sample_note = None
        self.state.update(cast("dict[str, Any]", new_state))
        if not self.repairing:
            self._sync_regenerate_flag()

    def on_deck_selected(self, deck: str) -> None:
        # Dumb hack bc of leaky abstraction reactive combo box + int keys
        deck_id = DeckId(int(deck))
        note_type = self.state.s["selected_note_type"]
        new_state = self._state_for_new_card_type(
            note_type=note_type, type=self.state.s["type"], deck_id=deck_id
        )

        if self.repairing:
            new_state["prompt"] = self.state.s["prompt"]
            new_state["tts_style"] = self.state.s["tts_style"]
            new_state["tts_language"] = self.state.s["tts_language"]
        self.sample_note = None
        self.state.update(cast("dict[str, Any]", new_state))
        if not self.repairing:
            self._sync_regenerate_flag()

    def on_source_changed(self, source: str) -> None:
        self.state.update({"prompt": self.get_tts_prompt(source)})

    def get_tts_prompt(self, source: str) -> str:
        return f"{{{{{source}}}}}"

    def on_target_field_changed(self, field: str | None) -> None:
        if not field:
            return

        self.state.update({"selected_note_field": field})
        if not self.repairing:
            self._sync_regenerate_flag()

    def render_buttons(self) -> None:
        is_enabled = (
            bool(self.state.s["prompt"]) and not self.state.s["is_loading_prompt"]
        )

        self.test_button.setEnabled(is_enabled and self.sample_note is not None)
        if hasattr(self, "load_sample_button"):
            self.load_sample_button.setEnabled(not self.state.s["is_loading_prompt"])
        self.standard_buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(  # type: ignore
            is_enabled
        )

        if self.state.s["is_loading_prompt"]:
            self.test_button.setText("Generating preview…")
        else:
            self.test_button.setText("Generate preview")

    @with_widget_settings
    def get_current_chat_settings(self) -> tuple[str, str, str | None, bool]:
        settings = resolve_field_settings(
            config.snapshot(),
            self.chat_options.state.s if self.state.s["use_custom_model"] else {},
        )
        provider = settings["chat_provider"]
        model = settings["chat_model"]
        reasoning_effort = settings.get("chat_reasoning_effort")
        use_tools = self.state.s["chat_use_tools"]
        return str(provider), str(model), reasoning_effort, use_tools

    def get_current_prompt_signature(self) -> str:
        provider, model, reasoning_effort, use_tools = self.get_current_chat_settings()
        return build_prompt_usage_signature(
            prompt=self.state.s["prompt"],
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            use_tools=use_tools,
        )

    @with_widget_settings
    def get_saved_prompt_signature(self) -> str | None:
        if self.mode != "edit" or self.state.s["type"] != "chat":
            return None

        note_type = self.state.s["selected_note_type"]
        deck_id = self.state.s["selected_deck"]
        field = self.state.s["selected_note_field"]
        prompts_for_note = get_prompts_for_note(
            note_type=note_type,
            deck_id=deck_id,
            override_prompts_map=self.prompts_map,
            fallback_to_global_deck=False,
        )
        saved_prompt = prompts_for_note.get(field) if prompts_for_note else None
        if not saved_prompt:
            return None

        saved_extras = (
            get_extras(
                note_type=note_type,
                field=field,
                deck_id=deck_id,
                prompts=self.prompts_map,
                fallback_to_global_deck=False,
            )
            or DEFAULT_EXTRAS
        )
        saved_settings = resolve_field_settings(config.snapshot(), saved_extras)
        saved_provider = saved_settings["chat_provider"]
        saved_model = saved_settings["chat_model"]
        saved_reasoning_effort = saved_settings.get("chat_reasoning_effort")
        saved_use_tools = saved_settings["chat_use_tools"]

        return build_prompt_usage_signature(
            prompt=saved_prompt,
            provider=str(saved_provider),
            model=str(saved_model),
            reasoning_effort=saved_reasoning_effort,
            use_tools=bool(saved_use_tools),
        )

    def build_saved_prompt_usage_context(self) -> PromptUsageContext | None:
        saved_signature = self.get_saved_prompt_signature()
        if (
            not saved_signature
            or self.get_current_prompt_signature() != saved_signature
        ):
            return None

        return PromptUsageContext(
            prompt_key=build_prompt_usage_key(
                note_type=self.state.s["selected_note_type"],
                deck_id=int(self.state.s["selected_deck"]),
                field_lower=self.state.s["selected_note_field"].lower(),
            ),
            signature=saved_signature,
        )

    @with_widget_settings
    def render_token_usage(self) -> None:
        if self.state.s["type"] != "chat":
            return

        current_signature = self.get_current_prompt_signature()
        selected_deck_id = int(self.state.s["selected_deck"])
        snapshot = chat_usage_tracker.get_prompt_usage(
            note_type=self.state.s["selected_note_type"],
            deck_id=selected_deck_id,
            field_lower=self.state.s["selected_note_field"].lower(),
            signature=current_signature,
        )
        if snapshot is None and selected_deck_id == GLOBAL_DECK_ID:
            snapshot = chat_usage_tracker.get_prompt_usage_across_decks(
                note_type=self.state.s["selected_note_type"],
                field_lower=self.state.s["selected_note_field"].lower(),
                signature=current_signature,
            )
        saved_signature = self.get_saved_prompt_signature()

        lines: list[str] = []
        if snapshot is not None:
            lines.extend(
                [
                    f"Average: {format_token_count(round(snapshot.avg_total_tokens))} total tokens/run",
                    f"Input: {format_token_count(round(snapshot.avg_input_tokens))} avg",
                    f"Output: {format_token_count(round(snapshot.avg_output_tokens))} avg",
                    f"Runs: {snapshot.run_count}",
                ]
            )
        elif saved_signature and saved_signature != current_signature:
            lines.append("Stats will reset when this prompt is saved.")
        else:
            lines.append(
                "No token usage yet. Stats appear after a successful chat response that reports usage."
            )

        provider, _, _, _ = self.get_current_chat_settings()
        if provider == "openai" and config.openai_daily_token_budget_enabled:
            daily_usage = chat_usage_tracker.get_openai_daily_usage()
            lines.append(
                f"Today: {format_token_count(daily_usage.used_total_tokens)} / "
                f"{format_token_count(config.openai_daily_token_budget)} estimated guard"
            )
            if daily_usage.estimated_total_tokens:
                lines.append(
                    f"Includes {format_token_count(daily_usage.estimated_total_tokens)} estimated tokens."
                )
            lines.append("Resets: 00:00 UTC")

        self.token_usage_label.setText("\n".join(lines))

    @with_widget_settings
    def on_test(self) -> None:
        prompt = self.state.s["prompt"]

        if not mw or not self.state.s["prompt"]:
            return

        sample_note = self.sample_note
        if sample_note is None:
            return
        selected_deck = self.state.s["selected_deck"]
        snapshot = snapshot_note(
            sample_note,
            selected_deck,
            deck_id_to_name_map().get(selected_deck),
        )
        values = snapshot.lower_values()
        new_prompts_map = self._create_new_prompts_map()

        error = prompt_has_error(
            prompt,
            note=sample_note,
            target_field=self.state.s["selected_note_field"],
            prompts_map=new_prompts_map,
            deck_id=self.state.s["selected_deck"],
        )

        if error:
            show_message_box(f"Invalid prompt: {error}")
            return

        use_custom_model = self.state.s["use_custom_model"]
        effective_settings = resolve_field_settings(
            config.snapshot(), self.model_options.state.s if use_custom_model else {}
        )
        chat_provider = effective_settings["chat_provider"]
        chat_model = effective_settings["chat_model"]

        tts_target: TTSVoiceTarget | None = None
        tts_provider: TTSProviders | None = None
        tts_model: TTSModels | None = None
        tts_voice = ""
        if self.state.s["type"] == "tts":
            try:
                tts_target = self.get_effective_tts_target(
                    f"{snapshot.note_id}:{self.state.s['selected_note_field'].lower()}"
                )
            except ValueError as error:
                show_message_box(str(error))
                return
            tts_provider = cast("TTSProviders", tts_target["provider"])
            tts_model = cast("TTSModels", tts_target["model"])
            tts_voice = tts_target["voice"]

        if self.state.s["type"] == "chat":
            if not self._ensure_api_key(chat_provider):
                return
        elif self.state.s["type"] == "tts":
            if tts_provider is None or not self._ensure_api_key(tts_provider):
                return
        else:
            image_provider = (
                self.image_options.state.s["image_provider"]
                if self.state.s["use_custom_model"]
                else config.image_provider
            )
            if not self._ensure_api_key(image_provider):
                return

        field_state = deepcopy(self.state.s)
        chat_settings = image_settings = tts_settings = effective_settings
        daily_budget = config.openai_daily_token_budget
        budget_enabled = config.openai_daily_token_budget_enabled
        allow_empty_fields = config.allow_empty_fields
        saved_usage_context = self.build_saved_prompt_usage_context()
        self.state["is_loading_prompt"] = True

        stringified_vals = "\n".join(
            f"{field}: {values.get(field, '')}" for field in get_prompt_fields(prompt)
        )

        usage_scope_id = (
            chat_usage_tracker.open_scope() if field_state["type"] == "chat" else None
        )

        def finish() -> None:
            chat_usage_tracker.close_scope(usage_scope_id)
            if not sip.isdeleted(self):
                self.state["is_loading_prompt"] = False

        def on_success(arg: bytes | None) -> None:
            if field_state["type"] == "tts":
                msg = f"Ran with fields: \n{stringified_vals}.\n Voice: {tts_provider} - {tts_voice} ({tts_model})\n\n"
                if arg is not None:
                    play_audio(arg)
                else:
                    msg += "No audio response received"
                show_message_box(msg, custom_ok="Close")
            elif arg is not None:
                test_window = ImageTestDialog(
                    arg,
                    interpolate_prompt_with_values(prompt, values, allow_empty_fields)
                    or "",
                    parent=self,
                )
                test_window.exec()
            else:
                show_message_box("No image response received", custom_ok="Close")

        def on_failure(error: Exception) -> None:
            show_message_box(f"Failed to get response: {redact_text(str(error))}")

        def launch(
            operation: Callable[[], Awaitable[Any]],
            on_result: Callable[[Any], None],
        ) -> None:
            try:
                run_async_in_background_with_sentry(
                    operation,
                    on_result,
                    on_failure,
                    on_finished=finish,
                    show_error=False,
                )
            except Exception as error:
                finish()
                on_failure(error)

        if field_state["type"] == "chat":

            def on_chat_success(result: str | None) -> None:
                response_text = result if result is not None else "No response received"
                message_parts = [
                    f"Ran with fields: \n{stringified_vals}.",
                    f"Model: {chat_model}",
                    "",
                    f"Response: {response_text}",
                ]

                usage_summary = chat_usage_tracker.close_scope(usage_scope_id)
                if usage_summary.request_count:
                    message_parts.extend(
                        [
                            "",
                            f"Tokens: {format_token_count(usage_summary.total_tokens)} total "
                            f"({format_token_count(usage_summary.input_tokens)} input, "
                            f"{format_token_count(usage_summary.output_tokens)} output)",
                        ]
                    )

                if chat_provider == "openai" and budget_enabled:
                    daily_usage = chat_usage_tracker.get_openai_daily_usage()
                    message_parts.append(
                        f"OpenAI today: {format_token_count(daily_usage.used_total_tokens)} / "
                        f"{format_token_count(daily_budget)} estimated guard"
                    )
                    if daily_usage.estimated_total_tokens:
                        message_parts.append(
                            f"Includes {format_token_count(daily_usage.estimated_total_tokens)} estimated tokens."
                        )

                TextTestDialog("\n".join(message_parts), parent=self).exec()

            def chat_fn():
                return self.processor.field_processor.get_chat_response(
                    note_id=snapshot.note_id,
                    note_type=snapshot.note_type,
                    deck_name=snapshot.deck_name,
                    field_order=snapshot.field_order,
                    values=values,
                    prompt=prompt,
                    provider=chat_provider,
                    model=chat_model,
                    field_lower=field_state["selected_note_field"].lower(),
                    deck_id=field_state["selected_deck"],
                    temperature=key_or_config_val(chat_settings, "chat_temperature"),
                    reasoning_effort=chat_settings.get("chat_reasoning_effort"),
                    should_convert_to_html=False,  # Don't show HTML here bc it's confusing
                    use_tools=field_state["chat_use_tools"],
                    show_error_box=False,
                    prompt_usage_context=saved_usage_context,
                    usage_scope_id=usage_scope_id,
                )

            launch(chat_fn, on_chat_success)
        elif field_state["type"] == "tts":

            def tts_fn():
                assert tts_provider is not None and tts_model is not None
                prompt_to_use = prompt
                style = field_state.get("tts_style")
                if style and "gemini" in tts_model and tts_provider == "google":
                    prompt_to_use = f"{style} {prompt}"
                instructions = (
                    style
                    if style
                    and tts_provider == "openai"
                    and tts_model == "gpt-4o-mini-tts"
                    else None
                )

                return self.processor.field_processor.get_tts_response(
                    note_id=snapshot.note_id,
                    values=values,
                    input_text=prompt_to_use,
                    provider=tts_provider,
                    model=tts_model,
                    voice=tts_voice,
                    strip_html=tts_settings["tts_strip_html"],
                    instructions=instructions,
                )

            launch(tts_fn, on_success)
        else:

            async def img_fn() -> bytes | None:
                provider = image_settings["image_provider"]
                model = key_or_config_val(image_settings, "image_model")
                image = await self.processor.field_processor.get_image_response(
                    note_id=snapshot.note_id,
                    values=values,
                    input_text=prompt,
                    model=model,
                    provider=provider,
                    aspect_ratio=key_or_config_val(
                        image_settings, "image_aspect_ratio"
                    ),
                    resolution=key_or_config_val(image_settings, "image_resolution"),
                    output_format=key_or_config_val(
                        image_settings, "image_output_format"
                    ),
                    quality=key_or_config_val(image_settings, "image_quality"),
                    generation_quality=key_or_config_val(
                        image_settings, "image_generation_quality"
                    ),
                )

                return await image_preview_data(image) if image else None

            launch(img_fn, on_success)

    @with_widget_settings
    def get_effective_tts_target(self, selection_key: str) -> TTSVoiceTarget:
        pool = (
            self.tts_options.state.s["tts_voice_pool"]
            if self.state.s["use_custom_model"]
            else config.tts_voice_pool
        )
        return select_tts_target(
            pool,
            language=self.state.s["tts_language"],
            selection_key=selection_key,
        )

    def _ensure_api_key(self, provider: str) -> bool:
        # Check custom providers first
        if config.custom_providers:
            for p in config.custom_providers:
                if p["name"] == provider:
                    return True

        key_attr = PROVIDER_API_KEY_ATTRS.get(provider)
        if not key_attr:
            return True
        if getattr(config, key_attr, None):
            return True
        show_message_box(API_KEY_MISSING_MESSAGE.format(provider))
        return False

    def render_automatic_button(self) -> None:
        self.enabled_box.setChecked(self.state.s["generate_automatically"])

    def render_valid_fields(self) -> None:
        pass  # Replaced by Insert button

    def _get_valid_target_fields(
        self,
        selected_note_type: str,
        deck_id: DeckId,
        selected_note_field: str | None = None,
    ) -> list[str]:
        """Gets all fields excluding selected and existing prompts"""
        all_valid_fields = get_valid_fields_for_prompt(
            selected_note_type=selected_note_type,
            selected_note_field=selected_note_field,
            deck_id=deck_id,
            prompts_map=self.prompts_map,
        )
        existing_prompts = set(
            (
                get_prompts_for_note(
                    note_type=selected_note_type,
                    override_prompts_map=self.prompts_map,
                    deck_id=deck_id,
                    fallback_to_global_deck=False,
                )
                or {}
            ).keys()
        )

        return [field for field in all_valid_fields if field not in existing_prompts]

    def _get_initial_source_field(self, note_type: str, deck_id: DeckId) -> str:
        """Get the first valid source field for a note type by finding the first field that isn't the default target field"""
        fields = get_fields(note_type)
        # Strange case of cards with a single field
        if (len(fields)) == 1:
            logger.debug(f"Note type {note_type} has no valid fields")
            return fields[0]

        valid_target_fields = get_valid_fields_for_prompt(
            note_type, deck_id=deck_id, prompts_map=self.prompts_map
        )
        default_target_field = (
            valid_target_fields[0] if len(valid_target_fields) > 0 else None
        )
        return next(
            (f for f in fields if f != default_target_field),
            "No valid source fields remaining",
        )

    def _attempt_to_parse_source_field(self, prompt: str) -> str | None:
        fields = get_prompt_fields(prompt, lower=False)

        if len(fields) != 1:
            return None

        return fields[0]

    @with_widget_settings
    def on_accept(self):
        if not mw or not mw.col:
            return

        prompt = self.state.s["prompt"]
        selected_card_type = self.state.s["selected_note_type"]
        selected_field = self.state.s["selected_note_field"]

        if not prompt:
            return
        if (
            self.mode == "new" or self.repairing
        ) and selected_field not in self._get_valid_target_fields(
            selected_card_type, self.state.s["selected_deck"]
        ):
            show_message_box(
                "Choose an unused target field. This field already has a Smart Field, or no longer exists."
            )
            self.field_combo_box.setFocus()
            return

        if self.state.s["type"] == "tts":
            try:
                self.get_effective_tts_target("validation")
            except ValueError as error:
                show_message_box(str(error))
                return

        new_prompts_map = self._create_new_prompts_map()
        logger.debug("Created new prompts map")

        # Make an ephemeral note
        note_type = next(
            (e for e in mw.col.models.all() if e["name"] == selected_card_type), None
        )

        if not note_type:
            logger.error("Unexpectedly find note type in prompt_dialog")
            return

        sample_note = Note(mw.col, note_type)

        err = prompt_has_error(
            prompt,
            note=sample_note,
            target_field=selected_field,
            prompts_map=new_prompts_map,
            deck_id=self.state.s["selected_deck"],
        )

        if err:
            show_message_box(f"Invalid prompt: {err}")
            return

        # Ensure only openai for legacy
        self.on_accept_callback(new_prompts_map)
        self.accept()

    def _create_new_prompts_map(self) -> PromptMap:
        s = self.state.s

        tts_options = cast(
            "OverrideableTTSOptionsDict",
            {k: self.tts_options.state.s[k] for k in overridable_tts_options}
            if s["type"] == "tts"
            else {},
        )

        prompts_map = self.prompts_map
        if self.original_identity is not None:
            original_type, original_deck, original_field = self.original_identity
            if (original_type, original_deck, original_field) != (
                s["selected_note_type"],
                s["selected_deck"],
                s["selected_note_field"],
            ):
                prompts_map = remove_prompt(
                    prompts_map, original_type, original_deck, original_field
                )
        return add_or_update_prompts(
            prompts_map=prompts_map,
            note_type=s["selected_note_type"],
            deck_id=s["selected_deck"],
            field=s["selected_note_field"],
            prompt=s["prompt"],
            is_automatic=s["generate_automatically"],
            is_custom_model=s["use_custom_model"],
            type=s["type"],
            tts_options=tts_options,
            tts_style=s.get("tts_style"),
            tts_language=s.get("tts_language"),
            chat_options={
                k: self.chat_options.state.s[k] for k in overridable_chat_options
            }
            if s["type"] == "chat"
            else {},
            image_options={
                k: self.image_options.state.s[k] for k in overridable_image_options
            }
            if s["type"] == "image"
            else {},
            chat_use_tools=s["chat_use_tools"] if s["type"] == "chat" else None,
            regenerate_when_batching=self.state.s["regenerate_when_batching"],
        )


class TextTestDialog(QDialog):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Field preview")
        layout = QVBoxLayout(self)
        result = QPlainTextEdit(text)
        result.setReadOnly(True)
        result.setAccessibleName("Preview response and referenced fields")
        layout.addWidget(result, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        bound_dialog(self)


class ImageTestDialog(QDialog):
    def __init__(
        self, image: bytes, prompt: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Previewer")
        layout = QVBoxLayout()
        self.setLayout(layout)
        explainer = QPlainTextEdit(f"Prompt:\n{prompt}")
        explainer.setReadOnly(True)
        explainer.setAccessibleName("Image preview prompt")
        explainer.setMaximumHeight(explainer.fontMetrics().lineSpacing() * 5)
        layout.addWidget(explainer)
        displayer = ImageDisplayer(image=image)
        layout.addWidget(displayer, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        bound_dialog(self)
