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

# Note to future self / contributors:
# these classes are fairly brittle & tightly coupled, and deviate a good bit from
# existing patterns in the codebase (making heavy use of inheritance, using StateManager in an adhoc-way, etc)

from collections.abc import Callable
from copy import deepcopy
from typing import TypedDict

from anki.decks import DeckId
from anki.errors import NotFoundError
from anki.notes import Note
from aqt import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpacerItem,
    Qt,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    mw,
)
from PyQt6 import sip

from .. import tasks
from ..decks import deck_id_to_name_map
from ..field_processor import SpeechResult, field_processor
from ..job_context import JobContext
from ..logger import logger, redact_text
from ..markdown import convert_markdown_to_html, convert_text_to_html
from ..media_utils import get_media_path, image_preview_data
from ..models import DEFAULT_EXTRAS, OverrideableTTSOptionsDict
from ..note_processor import NoteSnapshot, PendingMedia, ProcessedNote, snapshot_note
from ..notes import get_note_type, get_valid_fields_for_prompt
from ..prompts import get_extras, get_prompts_for_note
from ..sentry import run_async_in_background_with_sentry
from ..settings import with_widget_settings
from ..tts_routing import tts_audio_extension
from ..tts_utils import play_audio
from .chat_options import ChatOptions
from .image_displayer import ImageDisplayer
from .image_options import ImageOptions
from .reactive_combo_box import ReactiveComboBox
from .state_manager import StateManager
from .tts_options import TTSOptions
from .ui_utils import bound_dialog, font_small, scroll_page, show_message_box


class CustomPrompt(QDialog):
    _initial_prompt: str | None
    _generate_button: QPushButton
    _save_button: QPushButton
    _loading: bool = False
    _prompt_window: QTextEdit
    _note: Note
    _deck_id: DeckId
    _on_success: Callable[[ProcessedNote], None]
    _field_upper: str

    def __init__(
        self,
        note: Note,
        deck_id: DeckId,
        field_upper: str,
        on_success: Callable[[ProcessedNote], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        all_prompts = get_prompts_for_note(
            note_type=get_note_type(note),
            deck_id=deck_id,
            to_lower=False,
            fallback_to_global_deck=True,
        )
        self._initial_prompt = (
            all_prompts.get(field_upper, None) if all_prompts else None
        )
        self._origin = (
            JobContext(
                tasks.worker.collection, tasks.worker.generation, tasks.worker.active
            )
            if tasks.worker
            else None
        )
        self._original = snapshot_note(note, deck_id)
        self._note = note
        self._deck_id = deck_id
        self._field_upper = field_upper
        self._on_success = on_success

        self._setup_ui()

    def _snapshot(self) -> NoteSnapshot:
        return snapshot_note(
            self._note,
            self._deck_id,
            deck_id_to_name_map().get(self._deck_id),
        )

    def on_generate(self) -> None:
        raise Exception("Not Implemented")

    def render_custom_model(self) -> QWidget:
        raise Exception("Not implemented")

    def render_to_text(self) -> str | None:
        raise Exception("Not Implemented")

    def pending_media(self) -> list[PendingMedia]:
        return []

    def render_response_box(self) -> QWidget:
        raise Exception("Not Implemented")

    def has_output(self) -> bool:
        raise Exception("Not Implemented")

    def update_ui_states(self) -> None:
        raise Exception("Not implemented")

    def setup_ui(self) -> None:
        pass

    def _setup_ui(self) -> None:
        container = QVBoxLayout()

        box_layout = QHBoxLayout()
        box = QGroupBox()
        box.setLayout(box_layout)
        self.setWindowTitle("⚙️ Prompt Palette")
        self.setMinimumWidth(600)

        # Left
        left_container = QWidget()
        self.left_layout = QVBoxLayout()
        left_container.setLayout(self.left_layout)
        self._prompt_window = QTextEdit()
        self._prompt_window.setPlainText(self._initial_prompt or "")
        self._prompt_window.textChanged.connect(self._on_prompt_changed)
        self._generate_button = QPushButton("Generate")
        self._generate_button.clicked.connect(self._on_generate)

        valid_fields = QLabel(
            "Valid fields: "
            + ", ".join(
                [
                    "{{" + field + "}} "
                    for field in get_valid_fields_for_prompt(
                        get_note_type(self._note), self._deck_id, self._field_upper
                    )
                ]
            )
        )
        valid_fields.setWordWrap(True)
        valid_fields.setFont(font_small)
        valid_fields.setMaximumWidth(600)

        self.left_layout.addWidget(QLabel("💬 Prompt"))
        self.left_layout.addWidget(self._prompt_window)
        self.left_layout.addWidget(valid_fields)
        self.left_layout.addWidget(self._generate_button)

        # Right
        self.right_container = QWidget()
        right_layout = QVBoxLayout()
        self.right_container.setLayout(right_layout)

        self._response_edit = QTextEdit()
        self._response_edit.setReadOnly(True)

        right_layout.addWidget(QLabel("✨ Response"))
        right_layout.addWidget(self.render_response_box())

        box_layout.addWidget(left_container)
        box_layout.addWidget(self.right_container)

        # Custom model

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.addWidget(box)
        page_layout.addItem(QSpacerItem(0, 16))
        page_layout.addWidget(self._render_custom_model())
        container.addWidget(scroll_page(page))

        self._save_button = QPushButton("Save Result")
        self._save_button.setEnabled(False)

        standard_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)

        standard_buttons.addButton(
            self._save_button, QDialogButtonBox.ButtonRole.AcceptRole
        )

        standard_buttons.accepted.connect(self._on_save_result)
        standard_buttons.rejected.connect(self.close)
        container.addWidget(standard_buttons)

        self.setLayout(container)
        self.setup_ui()
        bound_dialog(self)

    def _render_custom_model(self) -> QWidget:
        container = QGroupBox("⚙️ Custom Settings")
        layout = QVBoxLayout()
        container.setLayout(layout)
        layout.addWidget(self.render_custom_model())

        return container

    def _on_save_result(self) -> None:
        if self._origin is None:
            return
        try:
            self._origin.check(mw.col if mw else None)
        except Exception as error:
            show_message_box(str(error))
            return
        try:
            current = (
                mw.col.get_note(self._note.id)
                if self._note.id and mw and mw.col
                else self._note
            )
        except NotFoundError:
            show_message_box("The note was deleted. This result cannot be saved.")
            return
        if (
            current.mid != self._original.note_type_id
            or tuple(current.keys()) != self._original.field_order
            or any(
                current[key] != value for key, value in self._original.fields.items()
            )
        ):
            show_message_box(
                "The note changed while this palette was open. Generate again from the updated note."
            )
            return
        res = self.render_to_text()
        if res is not None:
            self._on_success(
                ProcessedNote(
                    note_id=self._original.note_id,
                    original_values=self._original.fields,
                    updates={self._field_upper: res},
                    media=self.pending_media(),
                    field_failures=[],
                    note_type_id=self._original.note_type_id,
                    field_order=self._original.field_order,
                )
            )
        self.accept()

    def _on_prompt_changed(self) -> None:
        self._update_ui_states()

    def reject(self):
        if not self.has_output() or show_message_box(
            "Close and lose results?", show_cancel=True
        ):
            QDialog.reject(self)

    @with_widget_settings
    def _on_generate(self) -> None:
        self._loading = True
        self._update_ui_states()
        try:
            self.on_generate()
        except Exception as error:
            self._finish_generation()
            show_message_box(redact_text(str(error)))

    def _finish_generation(self) -> None:
        self._loading = False
        if not sip.isdeleted(self):
            self._update_ui_states()

    def _update_ui_states(self) -> None:
        self._save_button.setEnabled(not self._loading and self.has_output())
        self._generate_button.setEnabled(
            not self._loading and bool(self._prompt_window.toPlainText())
        )
        self.update_ui_states()


class CustomTextPrompt(CustomPrompt):
    _response_edit: QTextEdit
    _chat_options: ChatOptions

    def render_response_box(self) -> QWidget:
        self._response_edit = QTextEdit()
        self._response_edit.setReadOnly(True)
        return self._response_edit

    def on_generate(self) -> None:
        self._response_edit.clear()
        prompt = self._prompt_window.toPlainText()

        def on_success(text: str):
            self._loading = False
            self._response_edit.setPlainText(text or "")
            self._update_ui_states()

        def on_error(error: Exception):
            msg = f"Error generating text: {redact_text(str(error))}"
            logger.error(msg)
            show_message_box(msg)
            self._loading = False
            self._update_ui_states()

        snapshot = self._snapshot()

        chat_settings = deepcopy(self._chat_options.state.s)

        async def generate_text():
            return await field_processor.get_chat_response(
                note_id=snapshot.note_id,
                note_type=snapshot.note_type,
                deck_name=snapshot.deck_name,
                field_order=snapshot.field_order,
                values=snapshot.lower_values(),
                deck_id=snapshot.deck_id,
                prompt=prompt,
                field_lower=self._field_upper.lower(),
                temperature=chat_settings["chat_temperature"],
                model=chat_settings["chat_model"],
                provider=chat_settings["chat_provider"],
                should_convert_to_html=False,
                reasoning_effort=chat_settings.get("chat_reasoning_effort"),
                use_tools=chat_settings.get("chat_use_tools", False),
            )

        run_async_in_background_with_sentry(
            generate_text, on_success, on_error, on_finished=self._finish_generation
        )

    def render_custom_model(self) -> QWidget:
        self._chat_options = ChatOptions(show_text_processing=False)
        return self._chat_options

    def has_output(self) -> bool:
        return bool(self._response_edit.toPlainText())

    def render_to_text(self) -> str | None:
        if self._chat_options.state.s["chat_markdown_to_html"]:
            return convert_markdown_to_html(self._response_edit.toPlainText())
        else:
            return convert_text_to_html(self._response_edit.toPlainText())

    def update_ui_states(self) -> None:
        self._response_edit.setReadOnly(self._loading)


class CustomImagePrompt(CustomPrompt):
    response_image: ImageDisplayer
    raw_image: bytes | None = None
    preview_image: bytes | None = None
    image_options: ImageOptions

    def render_response_box(self) -> QWidget:
        self.response_image = ImageDisplayer(parent=self)
        return self.response_image

    def on_generate(self) -> None:
        self.raw_image = None
        self.preview_image = None
        self._update_ui_states()
        prompt = self._prompt_window.toPlainText()

        def on_success(images: tuple[bytes, bytes] | None) -> None:
            self._loading = False
            if images is not None:
                self.raw_image, self.preview_image = images
            self._update_ui_states()

        def on_error(error: Exception):
            logger.error(f"Error generating image: {error}")
            show_message_box("Error generating image: " + redact_text(str(error)))
            self._loading = False
            self._update_ui_states()

        snapshot = self._snapshot()

        image_settings = deepcopy(self.image_options.state.s)
        self.image_format = image_settings["image_output_format"]

        async def generate_image() -> tuple[bytes, bytes] | None:
            image = await field_processor.get_image_response(
                note_id=snapshot.note_id,
                values=snapshot.lower_values(),
                input_text=prompt,
                model=image_settings["image_model"],
                provider=image_settings["image_provider"],
                aspect_ratio=image_settings["image_aspect_ratio"],
                resolution=image_settings["image_resolution"],
                output_format=image_settings["image_output_format"],
                quality=image_settings["image_quality"],
                generation_quality=image_settings["image_generation_quality"],
            )

            return (image, await image_preview_data(image)) if image else None

        run_async_in_background_with_sentry(
            generate_image, on_success, on_error, on_finished=self._finish_generation
        )

    def render_custom_model(self) -> QWidget:
        self.image_options = ImageOptions()
        return self.image_options

    def has_output(self) -> bool:
        return bool(self.raw_image)

    def render_to_text(self) -> str | None:
        if not self.raw_image:
            return None
        file_name = get_media_path(self._note, self._field_upper, self.image_format)
        return f'<img src="{file_name}"/>'

    def pending_media(self) -> list[PendingMedia]:
        return (
            [
                PendingMedia(
                    get_media_path(self._note, self._field_upper, self.image_format),
                    self.raw_image,
                )
            ]
            if self.raw_image
            else []
        )

    def update_ui_states(self) -> None:
        self.response_image.setVisible(bool(self.preview_image))
        if self.preview_image:
            self.response_image.set_image(self.preview_image)


class TTSPromptState(TypedDict):
    source_field: str
    source_fields: list[str]


class CustomTTSPrompt(CustomPrompt):
    audio: bytes | None = None
    audio_extension = "mp3"

    def __init__(
        self,
        note: Note,
        deck_id: DeckId,
        field_upper: str,
        on_success: Callable[[ProcessedNote], None],
        parent: QWidget | None = None,
    ) -> None:
        self._note = note
        self._deck_id = deck_id
        self._field_upper = field_upper

        source_fields = get_valid_fields_for_prompt(
            get_note_type(self._note), self._deck_id, self._field_upper
        )
        self.state = StateManager[TTSPromptState](
            {
                "source_fields": source_fields,
                "source_field": source_fields[0] if source_fields else "",
            }
        )

        super().__init__(
            note=note,
            deck_id=deck_id,
            field_upper=field_upper,
            on_success=on_success,
            parent=parent,
        )

    def render_response_box(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.speech_details = QLabel(
            "Generate speech to inspect its voice and delivery."
        )
        self.speech_details.setWordWrap(True)
        self.speech_details.setTextFormat(Qt.TextFormat.PlainText)
        self.speech_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.speech_details)
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)
        return page

    def on_generate(self) -> None:
        self.audio = None
        self.speech_details.setText("Generating speech…")
        prompt = self._prompt_window.toPlainText()
        tts_settings = deepcopy(self.tts_options.state.s)
        snapshot = self._snapshot()
        extras = deepcopy(
            get_extras(
                note_type=snapshot.note_type,
                field=self._field_upper,
                deck_id=self._deck_id,
            )
            or DEFAULT_EXTRAS
        )
        extras.update(
            type="tts",
            use_custom_model=True,
            tts_voice_pool=tts_settings["tts_voice_pool"],
            tts_strip_html=tts_settings["tts_strip_html"],
        )

        async def generate_speech() -> SpeechResult | None:
            return await field_processor.generate_speech(
                note_id=snapshot.note_id,
                field=self._field_upper,
                values=snapshot.lower_values(),
                input_text=prompt,
                extras=extras,
            )

        def on_success(speech: SpeechResult | None) -> None:
            self._loading = False
            self.audio = speech.data if speech else None
            if speech:
                self.audio_extension = tts_audio_extension(speech.target)
                self.speech_details.setText(
                    f"Voice: {speech.target['voice']}\n"
                    f"Model: {speech.target['model']}\n"
                    f"Direction: {speech.style or 'Natural delivery'}"
                )
            else:
                self.speech_details.setText("No audio response received.")
            self._update_ui_states()
            if self.audio:
                play_audio(self.audio)

        def on_failure(error: Exception):
            self.speech_details.setText("Speech generation failed.")
            self._loading = False
            self._update_ui_states()

        run_async_in_background_with_sentry(
            generate_speech,
            on_success=on_success,
            on_failure=on_failure,
            on_finished=self._finish_generation,
        )

    def render_custom_model(self) -> QWidget:
        extras = get_extras(
            note_type=get_note_type(self._note),
            field=self._field_upper,
            deck_id=self._deck_id,
        )
        overrides: OverrideableTTSOptionsDict | None = (
            {
                "tts_voice_pool": extras.get("tts_voice_pool"),
                "tts_strip_html": extras.get("tts_strip_html"),
            }
            if extras and extras["use_custom_model"]
            else None
        )
        self.tts_options = TTSOptions(overrides, extras_visible=False)
        return self.tts_options

    def has_output(self) -> bool:
        return bool(self.audio)

    def render_to_text(self) -> str | None:
        if not self.audio:
            return None
        file_name = get_media_path(self._note, self._field_upper, self.audio_extension)
        return f"[sound:{file_name}]"

    def pending_media(self) -> list[PendingMedia]:
        return (
            [
                PendingMedia(
                    get_media_path(self._note, self._field_upper, self.audio_extension),
                    self.audio,
                )
            ]
            if self.audio
            else []
        )

    def update_ui_states(self) -> None:
        self.play_button.setEnabled(not self._loading and self.has_output())

    def setup_ui(self) -> None:
        self.source_combo = ReactiveComboBox(
            self.state, "source_fields", "source_field"
        )

        def update_source_combo(pick: str):
            self._prompt_window.setPlainText(f"{{{{{pick}}}}}" if pick else "")

        update_source_combo(self.state.s["source_field"])

        self.source_combo.on_change.connect(update_source_combo)
        self.left_layout.addItem(QSpacerItem(0, 16))
        self.left_layout.addWidget(QLabel("Source Field"))
        self.left_layout.addWidget(self.source_combo)
        self.right_container.hide()
