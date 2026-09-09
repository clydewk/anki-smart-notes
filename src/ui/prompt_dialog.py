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

from collections.abc import Callable
from typing import Any, Literal, Optional, TypedDict, Union, cast

from anki.decks import DeckId
from anki.notes import Note
from aqt import (
    QAction,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
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
from ..logger import logger
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
    interpolate_prompt,
)
from ..sentry import run_async_in_background_with_sentry
from ..tts_routing import select_tts_target
from ..tts_utils import play_audio
from ..utils import get_fields, none_defaulting, to_lowercase_dict
from .chat_options import ChatOptions
from .image_displayer import ImageDisplayer
from .image_options import ImageOptions
from .reactive_check_box import ReactiveCheckBox
from .reactive_combo_box import ReactiveComboBox
from .reactive_edit_text import ReactiveEditText
from .reactive_line_edit import ReactiveLineEdit
from .state_manager import StateManager
from .tts_options import TTSOptions
from .ui_utils import default_form_layout, font_bold, font_small, show_message_box

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

Test out your prompt with the test button before saving it!
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
        card_type: Optional[str] = None,
        field: Optional[str] = None,
        prompt: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self.processor = processor
        self.on_accept_callback = on_accept_callback
        self.prompts_map = prompts_map
        self.mode = "edit" if card_type else "new"
        self.field_type = field_type
        note_types = self._get_note_types(deck_id=deck_id)
        selected_note_type = card_type or note_types[0]

        # Ensure there are valid fields to select
        if not len(note_types):
            show_message_box(
                "No valid note types left. Edit or delete some fields to continue!"
            )
            QTimer.singleShot(0, self.close)
            return

        default_note_state = self._state_for_new_card_type(
            selected_note_type, field_type, deck_id=deck_id
        )
        selected_target_field = field or default_note_state["selected_note_field"]

        extras = (
            get_extras(
                note_type=selected_note_type,
                field=selected_target_field,
                prompts=self.prompts_map,
                deck_id=deck_id,
            )
            or DEFAULT_EXTRAS
        )

        # Only if it's a new card, we need to get the fields for the selected card type
        target_fields = (
            default_note_state["note_fields"]
            if self.mode == "new"
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
            "selected_deck": deck_id,
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
            text="Enabled",
        )
        self.standard_buttons = self.create_buttons()

        tabs = QTabWidget()
        tabs.addTab(self.render_main_tab(), "General")
        tabs.addTab(self.render_options_tab(), "Options")

        container = QVBoxLayout()
        container.addWidget(tabs)

        container.addWidget(self.standard_buttons)
        self.setLayout(container)
        self.setup_ui()

    def setup_ui(self) -> None:
        self.render_ui()

    def render_ui(self) -> None:
        self.render_buttons()
        self.render_automatic_button()
        if hasattr(self, "token_usage_label"):
            self.render_token_usage()

    def render_main_tab(self) -> QWidget:
        layout = QVBoxLayout()

        field_type = self.state.s["type"]
        text = {
            "title": {
                "chat": "💬 New Text Field",
                "tts": "🔈️ New Text to Speech Field",
                "image": " 🖼️ New Image Field",
            },
            "explanation": {
                "chat": "The note that will have the Smart Field",
                "tts": "The note type that will have the TTS field",
                "image": "The note type that will have the image field",
            },
            "destination": {
                "chat": "Target Field",
                "tts": "Target Field",
                "image": "Target Field",
            },
            "destination_explanation": {
                "chat": "The AI generated Smart Field.",
                "tts": "The field that will store and play the audio file.",
                "image": "The field that will display the image.",
            },
        }

        self.setWindowTitle(text["title"][field_type])

        # 1. Settings Area
        settings_group = QGroupBox("Settings")
        form_layout = QFormLayout()

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
        self.field_combo_box.setToolTip(text["destination_explanation"][field_type])
        form_layout.addRow("Target Field:", self.field_combo_box)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        # 2. Prompting Area
        prompt_group = QGroupBox("Prompt Generation")
        prompt_layout = QVBoxLayout()

        # Style Instructions for Gemini TTS
        if self.state.s["type"] == "tts":
            style_label = QLabel("Style Instructions (OpenAI/Gemini):")
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
            prompt_layout.addSpacing(12)

        # Prompt Label + Insert Button
        prompt_label_layout = QHBoxLayout()
        prompt_label = QLabel("Prompt / Text to Speak")
        prompt_label.setFont(font_bold)
        prompt_label_layout.addWidget(prompt_label)
        prompt_label_layout.addStretch()

        insert_btn = QPushButton("Insert Field ➕")
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

        current_explanation = tts_explanation if field_type == "tts" else explanation
        self.prompt_text_box.setPlaceholderText(current_explanation)
        self.prompt_text_box.setMinimumHeight(120)

        prompt_layout.addWidget(self.prompt_text_box)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

        # 3. Footer
        self.test_button = QPushButton("✨ Test Smart Field ✨")
        self.test_button.setCursor(Qt.CursorShape.PointingHandCursor)

        footer_layout = QHBoxLayout()

        # Enabled Box with tooltip explanation
        self.enabled_box.setToolTip(
            "Enable or disable this field. Disabled fields can be generated via right clicking a field in the editor."
        )
        footer_layout.addWidget(self.enabled_box)
        footer_layout.addStretch()
        footer_layout.addWidget(self.test_button)

        layout.addLayout(footer_layout)

        self.state.state_changed.connect(self.render_ui)
        self.note_combo_box.on_change.connect(self._on_new_card_type_selected)
        self.field_combo_box.on_change.connect(self.on_target_field_changed)
        self.deck_combo_box.on_change.connect(self.on_deck_selected)
        self.prompt_text_box.on_change.connect(
            lambda text: self.state.update({"prompt": text})
        )

        self.test_button.clicked.connect(self.on_test)

        # On small screens, make it a proportion of screen height. Otherwise set a fixed height
        FIXED_HEIGHT = 800
        screen = mw and mw.screen()
        screen_height = FIXED_HEIGHT if not screen else screen.geometry().height()

        min_height = min(FIXED_HEIGHT, int(screen_height * 0.8))
        self.setMinimumHeight(min_height)
        container = QWidget()
        container.setLayout(layout)

        # Control visibility depending on mode
        if self.mode == "edit":
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
        models_layout = default_form_layout()
        self.model_options = self.render_custom_model()
        self.model_options.setEnabled(self.state.s["use_custom_model"])
        self.custom_model = ReactiveCheckBox(self.state, "use_custom_model")
        self.state.state_changed.connect(self.on_state_update)
        override_box = QWidget()
        override_layout = QHBoxLayout()
        override_layout.setContentsMargins(0, 0, 0, 0)
        override_box.setLayout(override_layout)
        override_layout.addWidget(QLabel("Override Default Settings"))
        override_layout.addWidget(self.custom_model)
        models_layout.addWidget(override_box)
        # Regenerate when batching
        self.regenerate_batch_checkbox = ReactiveCheckBox(
            self.state, "regenerate_when_batching"
        )
        batch_box = QWidget()
        batch_layout = QHBoxLayout()
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_box.setLayout(batch_layout)
        batch_layout.addWidget(QLabel("Regenerate when batch processing:"))
        batch_layout.addWidget(self.regenerate_batch_checkbox)
        models_layout.addWidget(batch_box)

        batch_desc = QLabel(
            "If checked, this field always overwrites its value during batch generation."
        )
        batch_desc.setFont(font_small)
        models_layout.addRow(batch_desc)

        if self.state.s["type"] == "chat":
            self.chat_use_tools_checkbox = ReactiveCheckBox(
                self.state, "chat_use_tools"
            )
            tools_box = QWidget()
            tools_layout = QHBoxLayout()
            tools_layout.setContentsMargins(0, 0, 0, 0)
            tools_box.setLayout(tools_layout)
            tools_layout.addWidget(QLabel("Use tools for this field:"))
            tools_layout.addWidget(self.chat_use_tools_checkbox)
            models_layout.addWidget(tools_box)

            tools_desc = QLabel(
                "This setting is stored per field and does not depend on model overrides."
            )
            tools_desc.setFont(font_small)
            models_layout.addRow(tools_desc)

            token_group = QGroupBox("Token Usage")
            token_layout = QVBoxLayout()
            self.token_usage_label = QLabel()
            self.token_usage_label.setWordWrap(True)
            self.token_usage_label.setFont(font_small)
            token_layout.addWidget(self.token_usage_label)
            token_group.setLayout(token_layout)
            models_layout.addRow(token_group)

        models_layout.addWidget(self.model_options)
        model_box = QGroupBox("⚙️ Model Settings")
        model_box.setEnabled(True)

        model_box.setLayout(models_layout)
        model_box.setContentsMargins(0, 24, 0, 24)

        container_layout = default_form_layout()
        container_layout.addRow(QLabel(""), None)
        container_layout.addRow(model_box)
        container = QWidget()
        container.setLayout(container_layout)
        return container

    def create_buttons(self) -> QWidget:
        standard_buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )

        standard_buttons.accepted.connect(self.on_accept)
        standard_buttons.rejected.connect(self.reject)
        return standard_buttons

    def render_custom_model(self) -> QWidget:
        # TODO: could use a refactor
        # Setup the dummy options; only one will be used
        self.tts_options = TTSOptions()
        self.chat_options = ChatOptions(show_tools_toggle=False)
        self.image_options = ImageOptions()

        extras = get_extras(
            note_type=self.state.s["selected_note_type"],
            field=self.state.s["selected_note_field"],
            deck_id=self.state.s["selected_deck"],
            prompts=self.prompts_map,
            fallback_to_global_deck=False,
        )

        use_custom_model = extras and extras["use_custom_model"]

        if self.state.s["type"] == "tts":
            if extras and use_custom_model:
                self.tts_options = TTSOptions(
                    {
                        "tts_voice_pool": extras.get("tts_voice_pool"),
                        "tts_strip_html": extras.get("tts_strip_html"),
                    }
                )
            self.tts_options.state.state_changed.connect(self.on_state_update)
            return self.tts_options

        elif self.state.s["type"] == "chat":
            if extras and use_custom_model:
                self.chat_options = ChatOptions(
                    cast(
                        "OverridableChatOptionsDict",
                        {
                            "chat_provider": extras.get("chat_provider"),
                            "chat_model": extras.get("chat_model"),
                            "chat_temperature": extras.get("chat_temperature"),
                            "chat_reasoning_effort": extras.get(
                                "chat_reasoning_effort"
                            ),
                            "chat_markdown_to_html": extras.get(
                                "chat_markdown_to_html"
                            ),
                        },
                    ),
                    show_tools_toggle=False,
                )
            self.chat_options.state.state_changed.connect(self.on_state_update)
            return self.chat_options

        elif self.state.s["type"] == "image":
            if extras and use_custom_model:
                self.image_options = ImageOptions(
                    cast(
                        "OverridableImageOptionsDict",
                        {
                            "image_model": extras.get("image_model"),
                            "image_provider": extras.get("image_provider"),
                            "image_aspect_ratio": extras.get("image_aspect_ratio"),
                            "image_resolution": extras.get("image_resolution"),
                            "image_generation_quality": extras.get(
                                "image_generation_quality"
                            ),
                            "image_output_format": extras.get("image_output_format"),
                            "image_quality": extras.get("image_quality"),
                        },
                    )
                )
            self.image_options.state.state_changed.connect(self.on_state_update)
            return self.image_options

        # Should never get here
        return QWidget()

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
        self.model_options.setEnabled(self.state.s["use_custom_model"])
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
            note_type=note_type, type=self.state.s["type"], deck_id=GLOBAL_DECK_ID
        )
        self.state.update(cast("dict[str, Any]", new_state))
        # Force re-layout every time
        self.adjustSize()
        self._sync_regenerate_flag()

    def on_deck_selected(self, deck: str) -> None:
        # Dumb hack bc of leaky abstraction reactive combo box + int keys
        deck_id = DeckId(int(deck))
        note_type = self.state.s["selected_note_type"]
        new_state = self._state_for_new_card_type(
            note_type=note_type, type=self.state.s["type"], deck_id=deck_id
        )

        self.state.update(cast("dict[str, Any]", new_state))
        self._sync_regenerate_flag()

    def on_source_changed(self, source: str) -> None:
        self.state.update({"prompt": self.get_tts_prompt(source)})

    def get_tts_prompt(self, source: str) -> str:
        return f"{{{{{source}}}}}"

    def on_target_field_changed(self, field: Optional[str]) -> None:
        if not field:
            return

        self.state.update({"selected_note_field": field})
        self._sync_regenerate_flag()

    def render_buttons(self) -> None:
        is_enabled = (
            bool(self.state.s["prompt"]) and not self.state.s["is_loading_prompt"]
        )

        self.test_button.setEnabled(is_enabled)
        self.standard_buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(  # type: ignore
            is_enabled
        )

        if self.state.s["is_loading_prompt"]:
            self.test_button.setText("Loading...")
        else:
            self.test_button.setText("Test With Random Note✨")

    def get_current_chat_settings(self) -> tuple[str, str, Optional[str], bool]:
        use_custom_model = self.state.s["use_custom_model"]
        provider = key_or_config_val(
            self.chat_options.state.s if use_custom_model else None,
            "chat_provider",
        )
        model = key_or_config_val(
            self.chat_options.state.s if use_custom_model else None,
            "chat_model",
        )
        reasoning_effort = key_or_config_val(
            self.chat_options.state.s if use_custom_model else None,
            "chat_reasoning_effort",
        )
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

    def get_saved_prompt_signature(self) -> Optional[str]:
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
        saved_provider = key_or_config_val(saved_extras, "chat_provider")
        saved_model = key_or_config_val(saved_extras, "chat_model")
        saved_reasoning_effort = key_or_config_val(
            saved_extras, "chat_reasoning_effort"
        )
        saved_use_tools = key_or_config_val(saved_extras, "chat_use_tools")

        return build_prompt_usage_signature(
            prompt=saved_prompt,
            provider=str(saved_provider),
            model=str(saved_model),
            reasoning_effort=saved_reasoning_effort,
            use_tools=bool(saved_use_tools),
        )

    def build_saved_prompt_usage_context(self) -> Optional[PromptUsageContext]:
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
                f"{format_token_count(config.openai_daily_token_budget)}"
            )
            lines.append("Resets: 00:00 UTC")

        self.token_usage_label.setText("\n".join(lines))

    def on_test(self) -> None:
        prompt = self.state.s["prompt"]

        if not mw or not self.state.s["prompt"]:
            return

        selected_note_type = self.state.s["selected_note_type"]
        sample_note = get_random_note(
            selected_note_type, deck_id=self.state.s["selected_deck"]
        )

        if not sample_note:
            show_message_box("Smart Notes: need at least one note of this note type!")
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
        chat_provider = (
            self.chat_options.state.s["chat_provider"]
            if use_custom_model
            else config.chat_provider
        ) or config.chat_provider
        chat_model = (
            self.chat_options.state.s["chat_model"]
            if use_custom_model
            else config.chat_model
        ) or config.chat_model

        tts_target: Optional[TTSVoiceTarget] = None
        tts_provider: Optional[TTSProviders] = None
        tts_model: Optional[TTSModels] = None
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

        self.state["is_loading_prompt"] = True

        def on_success(arg: Union[bytes, None]):
            prompt = self.state.s["prompt"]
            if not prompt:
                return

            prompt_fields = get_prompt_fields(prompt)

            # clumsy stuff to make it work with lowercase fields...
            fields = to_lowercase_dict(sample_note)  # type: ignore
            field_map = {
                prompt_field: fields[prompt_field] for prompt_field in prompt_fields
            }

            stringified_vals = "\n".join([f"{k}: {v}" for k, v in field_map.items()])
            self.state["is_loading_prompt"] = False
            field_type = self.state.s["type"]
            if field_type == "tts":
                msg = f"Ran with fields: \n{stringified_vals}.\n Voice: {tts_provider} - {tts_voice} ({tts_model})\n\n"
                if arg is not None and isinstance(arg, bytes):
                    play_audio(arg)
                else:
                    msg += "No audio response received"
                show_message_box(msg, custom_ok="Close")
            else:
                if arg is not None and isinstance(arg, bytes):
                    test_window = ImageTestDialog(
                        arg, interpolate_prompt(prompt, sample_note) or ""
                    )
                    test_window.exec()
                else:
                    show_message_box("No image response received", custom_ok="Close")

        def on_failure(e: Exception) -> None:
            show_message_box(f"Failed to get response: {e}")
            self.state["is_loading_prompt"] = False

        if self.state.s["type"] == "chat":
            usage_scope_id = chat_usage_tracker.open_scope()

            def on_chat_success(result: Optional[str]) -> None:
                prompt = self.state.s["prompt"]
                if not prompt:
                    chat_usage_tracker.close_scope(usage_scope_id)
                    return

                prompt_fields = get_prompt_fields(prompt)
                fields = to_lowercase_dict(sample_note)  # type: ignore
                field_map = {
                    prompt_field: fields[prompt_field] for prompt_field in prompt_fields
                }
                stringified_vals = "\n".join(
                    [f"{k}: {v}" for k, v in field_map.items()]
                )
                self.state["is_loading_prompt"] = False

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

                if (
                    chat_provider == "openai"
                    and config.openai_daily_token_budget_enabled
                ):
                    daily_usage = chat_usage_tracker.get_openai_daily_usage()
                    message_parts.append(
                        f"OpenAI today: {format_token_count(daily_usage.used_total_tokens)} / "
                        f"{format_token_count(config.openai_daily_token_budget)}"
                    )

                show_message_box("\n".join(message_parts), custom_ok="Close")

            def on_chat_failure(error: Exception) -> None:
                chat_usage_tracker.close_scope(usage_scope_id)
                on_failure(error)

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
                    field_lower=self.state.s["selected_note_field"].lower(),
                    deck_id=self.state.s["selected_deck"],
                    temperature=key_or_config_val(
                        self.chat_options.state.s, "chat_temperature"
                    ),
                    should_convert_to_html=False,  # Don't show HTML here bc it's confusing
                    use_tools=self.state.s["chat_use_tools"],
                    show_error_box=False,
                    prompt_usage_context=self.build_saved_prompt_usage_context(),
                    usage_scope_id=usage_scope_id,
                )

            run_async_in_background_with_sentry(
                chat_fn, on_chat_success, on_chat_failure
            )
        elif self.state.s["type"] == "tts":

            def tts_fn():
                assert tts_provider is not None and tts_model is not None
                prompt_to_use = prompt
                style = self.state.s.get("tts_style")
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
                    strip_html=none_defaulting(
                        self.tts_options.state.s, "tts_strip_html", True
                    ),
                    instructions=instructions,
                )

            run_async_in_background_with_sentry(tts_fn, on_success, on_failure)
        else:

            def img_fn():
                image_settings = (
                    self.image_options.state.s
                    if self.state.s["use_custom_model"]
                    else None
                )
                provider = (
                    self.image_options.state.s["image_provider"]
                    if self.state.s["use_custom_model"]
                    else config.image_provider
                )
                model = key_or_config_val(image_settings, "image_model")
                return self.processor.field_processor.get_image_response(
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

            run_async_in_background_with_sentry(img_fn, on_success, on_failure)

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
        selected_note_field: Optional[str] = None,
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

    def _attempt_to_parse_source_field(self, prompt: str) -> Optional[str]:
        fields = get_prompt_fields(prompt, lower=False)

        if len(fields) != 1:
            return None

        return fields[0]

    def on_accept(self):
        if not mw or not mw.col:
            return

        prompt = self.state.s["prompt"]
        selected_card_type = self.state.s["selected_note_type"]
        selected_field = self.state.s["selected_note_field"]

        if not prompt:
            return

        if self.state.s["type"] == "tts":
            try:
                self.get_effective_tts_target("validation")
            except ValueError as error:
                show_message_box(str(error))
                return

        new_prompts_map = self._create_new_prompts_map()
        logger.debug("Created new prompts map")
        logger.debug(new_prompts_map)

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
            {k: self.tts_options.state.s[k] for k in overridable_tts_options},
        )

        return add_or_update_prompts(
            prompts_map=self.prompts_map,
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
            },
            image_options={
                k: self.image_options.state.s[k] for k in overridable_image_options
            },
            chat_use_tools=s["chat_use_tools"] if s["type"] == "chat" else None,
            regenerate_when_batching=self.state.s["regenerate_when_batching"],
        )


class ImageTestDialog(QDialog):
    def __init__(self, image: bytes, prompt: str):
        super().__init__()
        self.setWindowTitle("Image Previewer")
        layout = QVBoxLayout()
        self.setLayout(layout)
        explainer = QLabel(f"Ran with prompt: {prompt}")
        explainer.setWordWrap(True)
        explainer.setMaximumWidth(480)
        layout.addWidget(explainer)
        displayer = ImageDisplayer(image=image)
        layout.addWidget(displayer)
