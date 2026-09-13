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

import json
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from aqt import (
    QAbstractItemView,
    QAbstractListModel,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QItemSelection,
    QItemSelectionModel,
    QLabel,
    QListView,
    QModelIndex,
    QPersistentModelIndex,
    QPushButton,
    QSizePolicy,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from PyQt6 import sip

from ..config import config, key_or_config_val
from ..job_context import JobCancelledError
from ..logger import redact_text
from ..models import (
    OverrideableTTSOptionsDict,
    TTSModels,
    TTSProviders,
    TTSVoiceTarget,
    make_tts_voice_target,
    normalize_tts_voice_pool,
)
from ..sentry import run_async_in_background_with_sentry
from ..settings import with_widget_settings
from ..tts_provider import TTSProvider
from ..tts_utils import play_audio
from ..utils import load_file, none_defaulting
from .reactive_check_box import ReactiveCheckBox
from .reactive_combo_box import ReactiveComboBox
from .reactive_line_edit import ReactiveLineEdit
from .state_manager import StateManager
from .ui_utils import default_form_layout, font_small

if TYPE_CHECKING:
    from ..tasks import JobHandle

ALL: Literal["All"] = "All"

AllTTSProviders = Literal["All"] | TTSProviders

Gender = Literal["All", "Male", "Female"]
default_texts: dict[str, str] = {
    ALL: "I'm sorry Dave, I'm afraid I can't do that.",
}

price_tier_copy = {
    "free": "Free",
    "low": "Low Cost",
    "standard": "Standard Cost",
    "high": "High Cost",
    "ultra-high": "Ultra High Cost",
}

PriceTiers = Literal["free", "low", "standard", "high", "ultra-high"]

TTS_PROVIDER_LABELS = {
    "google": "Google",
    "azure": "Azure (unavailable)",
    "openai": "OpenAI",
    "elevenLabs": "ElevenLabs",
    "fish": "Fish Audio",
}


class TTSMeta(TypedDict):
    tts_provider: TTSProviders
    voice: str
    model: str
    friendly_voice: str
    gender: Literal["Male", "Female", "All"]
    language: str
    price_tier: PriceTiers


class TTSState(TypedDict):
    # Combo box fields

    providers: list[AllTTSProviders]
    selected_provider: AllTTSProviders
    genders: list[Gender]
    selected_gender: Gender
    languages: list[str]
    selected_language: str
    models: list[str]
    selected_filter_model: str

    # Current catalog selection. Persisted TTS configuration is the pool below.
    catalog_provider: TTSProviders
    catalog_voice: str
    catalog_model: TTSModels

    tts_voice_pool: list[TTSVoiceTarget]
    tts_strip_html: bool

    test_text: str
    test_enabled: bool
    search_text: str


# OpenAI TTS voice/gender mappings
OPENAI_VOICE_GENDERS: dict[str, Literal["Male", "Female"]] = {
    "alloy": "Female",
    "ash": "Male",
    "ballad": "Male",
    "coral": "Female",
    "echo": "Male",
    "fable": "Female",
    "nova": "Female",
    "onyx": "Male",
    "sage": "Female",
    "shimmer": "Female",
    "verse": "Male",
    "marin": "Female",
    "cedar": "Male",
}

# Voices available for each model
OPENAI_TTS1_VOICES = [
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]
OPENAI_GPT4O_VOICES = OPENAI_TTS1_VOICES + ["ballad", "verse", "marin", "cedar"]


def generate_openai_voices() -> list[TTSMeta]:
    """Generate OpenAI TTS voice entries for all models."""
    voices: list[TTSMeta] = []

    # gpt-4o-mini-tts voices (13 voices, recommended model)
    for voice in OPENAI_GPT4O_VOICES:
        voices.append(
            {
                "tts_provider": "openai",
                "voice": voice,
                "model": "gpt-4o-mini-tts",
                "friendly_voice": f"{voice.capitalize()} (gpt-4o-mini-tts)",
                "gender": OPENAI_VOICE_GENDERS[voice],
                "language": ALL,
                "price_tier": "standard",
            }
        )

    # tts-1-hd voices (9 voices, higher quality)
    for voice in OPENAI_TTS1_VOICES:
        voices.append(
            {
                "tts_provider": "openai",
                "voice": voice,
                "model": "tts-1-hd",
                "friendly_voice": f"{voice.capitalize()} (tts-1-hd)",
                "gender": OPENAI_VOICE_GENDERS[voice],
                "language": ALL,
                "price_tier": "standard",
            }
        )

    # tts-1 voices (9 voices, lower latency)
    for voice in OPENAI_TTS1_VOICES:
        voices.append(
            {
                "tts_provider": "openai",
                "voice": voice,
                "model": "tts-1",
                "friendly_voice": f"{voice.capitalize()} (tts-1)",
                "gender": OPENAI_VOICE_GENDERS[voice],
                "language": ALL,
                "price_tier": "low",
            }
        )

    return voices


openai_voices: list[TTSMeta] = generate_openai_voices()


class GoogleVoice(TypedDict):
    gender: Literal["Male", "Female"]
    languageCode: str
    language: str
    name: str
    type: Literal["Standard", "Wavenet", "Neural", "Chirp"]


def get_google_voices() -> list[TTSMeta]:
    s = load_file("google_voices.json", test_override="[]")
    google_voices: list[GoogleVoice] = json.loads(s)
    voices: list[TTSMeta] = []
    tiers = {
        "Standard": "low",
        "Wavenet": "standard",
        "Neural": "standard",
        "Chirp": "standard",
    }
    for voice in google_voices:
        voices.append(
            {
                "tts_provider": "google",
                "language": voice["language"],
                "gender": voice["gender"],
                "voice": voice["name"],
                "model": voice["type"].lower(),
                "friendly_voice": f"{voice['name'].removeprefix(voice['languageCode'] + '-').replace('-', ' ')} ({voice['languageCode']})",
                "price_tier": tiers[voice["type"]],  # type: ignore
            }
        )
    return voices


def get_eleven_voices() -> list[TTSMeta]:
    s = load_file("eleven_voices.json", test_override="[]")
    eleven_voices = json.loads(s)
    voices: list[TTSMeta] = []

    models: list[dict[str, str]] = [
        {"model": "eleven_v3", "price_tier": "ultra-high"},
        {"model": "eleven_multilingual_v2", "price_tier": "ultra-high"},
        {"model": "eleven_flash_v2_5", "price_tier": "high"},
    ]

    friendly_models = {
        "eleven_v3": "V3",
        "eleven_multilingual_v2": "Multilingual V2",
        "eleven_flash_v2_5": "Flash V2.5",
    }

    for voice in eleven_voices:
        for model in models:
            ttsMeta: TTSMeta = {
                "tts_provider": "elevenLabs",
                "language": voice["language"],
                "voice": voice["voice_id"],
                "model": model["model"],
                "friendly_voice": f"{voice['name'].capitalize()} ({friendly_models[model['model']]})",
                "gender": voice["gender"],
                "price_tier": cast("PriceTiers", model["price_tier"]),
            }

            voices.append(ttsMeta)

    return voices


class AzureVoice(TypedDict):
    name: str
    displayName: str
    locale: str
    language: str
    gender: Literal["Male", "Female"]
    voiceType: Literal["Neural", "NeuralHD"]
    styleList: list[str]
    sampleRateHertz: str


def get_azure_voices() -> list[TTSMeta]:
    s = load_file("azure_voices.json", test_override="[]")
    azure_voices: list[AzureVoice] = json.loads(s)
    voices: list[TTSMeta] = []
    tiers: dict[str, Literal["low", "standard", "high", "ultra-high"]] = {
        "Neural": "standard",
        "NeuralHD": "standard",
    }
    for voice in azure_voices:
        voices.append(
            {
                "tts_provider": "azure",
                "language": voice["language"],
                "gender": voice["gender"],
                "voice": voice["name"],
                "model": voice["voiceType"].lower(),
                "friendly_voice": f"{voice['displayName'].title()} ({voice['voiceType']})",
                "price_tier": tiers[voice["voiceType"]],
            }
        )
    return voices


def get_fish_voices() -> list[TTSMeta]:
    models: list[tuple[str, str, PriceTiers]] = [
        ("s2.1-pro-free", "S2.1 Pro Free", "free"),
        ("s2.1-pro", "S2.1 Pro", "standard"),
        ("s2-pro", "S2 Pro", "standard"),
        ("s1", "S1", "standard"),
    ]
    return [
        {
            "tts_provider": "fish",
            "voice": "",
            "model": model,
            "friendly_voice": f"Custom voice ({label})",
            "gender": ALL,
            "language": ALL,
            "price_tier": price_tier,
        }
        for model, label, price_tier in models
    ]


def get_gemini_voices() -> list[TTSMeta]:
    gemini_voice_names = [
        "Zephyr",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    ]
    voices: list[TTSMeta] = []

    models: list[tuple[str, str, PriceTiers]] = [
        ("gemini-3.1-flash-tts-preview", "Gemini 3.1 Flash", "low"),
        ("gemini-2.5-flash-preview-tts", "Gemini 2.5 Flash", "low"),
        ("gemini-2.5-pro-preview-tts", "Gemini 2.5 Pro", "standard"),
    ]

    for name in gemini_voice_names:
        for model, friendly_model, tier in models:
            voices.append(
                {
                    "tts_provider": "google",
                    "voice": name,
                    "model": model,
                    "friendly_voice": f"{name} ({friendly_model})",
                    "gender": "All",
                    "language": ALL,
                    "price_tier": tier,
                }
            )
    return voices


# Combine all voices
base_voices = (
    get_google_voices()
    + openai_voices
    + get_eleven_voices()
    + get_gemini_voices()
    + get_fish_voices()
)
voices = base_voices.copy()

languages: list[str] = [ALL] + sorted({voice["language"] for voice in voices} - {ALL})
base_providers: list[AllTTSProviders] = [
    ALL,
    "google",
    "openai",
    "elevenLabs",
    "fish",
]
providers: list[AllTTSProviders] = base_providers.copy()


def format_voice(voice: TTSMeta) -> str:
    language = "Multilingual" if voice["language"] == ALL else voice["language"]
    provider = TTS_PROVIDER_LABELS.get(voice["tts_provider"], voice["tts_provider"])
    gender = f" · {voice['gender']}" if voice["gender"] != ALL else ""
    return (
        f"{voice['friendly_voice']}\n"
        f"{provider} · {language}{gender} · {price_tier_copy[voice['price_tier']]}"
    )


voice_search_cache: dict[tuple[str, str, str], list[str]] = {
    (v["tts_provider"], v["voice"], v["model"]): format_voice(v).lower().split()
    for v in voices
}


class CustomListModel(QAbstractListModel):
    def __init__(self, data: list[TTSMeta]):
        super().__init__()
        self._data = data

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid() or not 0 <= index.row() < len(self._data):
            return None
        voice = self._data[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return format_voice(voice)
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{format_voice(voice)}\nModel: {voice['model']}\nVoice ID: {voice['voice']}"
        return None

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return 0 if parent.isValid() else len(self._data)

    def update_data(self, new_data: list[TTSMeta]) -> None:
        self.beginResetModel()
        self._data = new_data
        self.endResetModel()

    def get_data(self) -> list[TTSMeta]:
        return self._data


class SelectedVoiceLabel(QLabel):
    def __init__(self, meta: TTSMeta | None):
        super().__init__("Select a voice to add or preview.")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setWordWrap(True)
        self.setAccessibleDescription("Selected voice")
        if meta:
            self.update_text(meta)

    def update_text(self, meta: TTSMeta, *, filtered: bool = False) -> None:
        prefix = "Selected (outside current filters)" if filtered else "Selected"
        self.setText(f"{prefix}: {format_voice(meta)}")
        self.setToolTip(f"Model: {meta['model']}\nVoice ID: {meta['voice']}")


class TTSOptions(QWidget):
    extras_visible: bool

    def __init__(
        self,
        tts_options: OverrideableTTSOptionsDict | None = None,
        extras_visible: bool = True,
    ):
        super().__init__()
        self.extras_visible = extras_visible
        self._preview_job: JobHandle | None = None
        self._rendered_pool: list[TTSVoiceTarget] | None = None

        # Merge custom models into available models
        self._inject_custom_models()

        self.state = StateManager[TTSState](self.get_initial_state(tts_options))
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.update_search)
        self.pending_search_text = ""
        self.setup_ui()

    def refresh_custom_providers(self) -> None:
        global voices, providers, voice_search_cache

        # Reset to base
        voices[:] = base_voices[:]
        providers[:] = base_providers[:]

        # Rebuild cache for base voices
        voice_search_cache.clear()
        for v in voices:
            voice_search_cache[(v["tts_provider"], v["voice"], v["model"])] = (
                format_voice(v).lower().split()
            )

        # Inject custom models
        self._inject_custom_models()

        # Update state
        available_models = [ALL] + sorted({v["model"] for v in voices})

        self.state.update({"providers": providers[:], "models": available_models})

        # Refresh the list view
        self.update_ui()

    def _inject_custom_models(self) -> None:
        if not config.custom_providers:
            return

        for provider in config.custom_providers:
            caps = provider.get("capabilities", [])
            # If capabilities list is empty or explicitly contains "tts"
            if not caps or "tts" in caps:
                p_name = provider["name"]

                # Check for tts_models first, then fallback to models
                models_to_add = []
                if provider.get("tts_models"):
                    models_to_add = provider["tts_models"]  # type: ignore
                elif provider.get("models"):
                    models_to_add = provider["models"]  # type: ignore

                if not models_to_add:
                    continue

                # Add provider to providers list if not present
                if p_name not in providers:
                    providers.append(p_name)  # type: ignore

                # Add voices/models
                for m in models_to_add:
                    # Check if already added to avoid duplicates
                    exists = any(
                        v["tts_provider"] == p_name and v["model"] == m for v in voices
                    )
                    if exists:
                        continue

                    voices.append(
                        {
                            "tts_provider": p_name,  # type: ignore
                            "voice": m,  # Use model name as voice name for simplicity in list
                            "model": m,
                            "friendly_voice": f"{m} ({p_name})",
                            "gender": ALL,
                            "language": ALL,
                            "price_tier": "standard",
                        }
                    )

                    # Update search cache
                    voice_search_cache[(p_name, m, m)] = (
                        format_voice(voices[-1]).lower().split()
                    )

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.render_voice_pool())

        self.voice_box = QGroupBox("Add a voice")
        catalog_layout = QVBoxLayout(self.voice_box)
        catalog_layout.addWidget(self.render_filters())
        catalog_layout.addWidget(self.render_voices_list())
        self.selected_voice_label = SelectedVoiceLabel(None)
        catalog_layout.addWidget(self.selected_voice_label)
        catalog_layout.addWidget(self.render_fish_voice())
        catalog_layout.addWidget(self.render_test_voice())
        layout.addWidget(self.voice_box)
        layout.addWidget(self.render_processing_rules())
        layout.addStretch()

        self.processing_box.setVisible(self.extras_visible)
        self.state.state_changed.connect(self.update_ui)
        self.update_ui()

    def render_voice_pool(self) -> QWidget:
        box = QGroupBox("Voice pool")
        layout = QVBoxLayout(box)
        description = QLabel(
            "Each note keeps a stable choice among enabled voices. "
            "Set a language to restrict a voice; leave it blank for any language. "
            "Double-click a Language cell to edit it."
        )
        description.setWordWrap(True)
        description.setFont(font_small)
        layout.addWidget(description)

        self.pool_table = QTableWidget(0, 3)
        self.pool_table.setAccessibleName("Enabled voices and language routing")
        self.pool_table.setHorizontalHeaderLabels(
            ["Active", "Voice", "Language (edit)"]
        )
        self.pool_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.pool_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        header = self.pool_table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        vertical_header = self.pool_table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        self.pool_table.itemChanged.connect(self.pool_item_changed)
        self.pool_table.itemSelectionChanged.connect(self._update_pool_selection)
        layout.addWidget(self.pool_table)

        controls = QHBoxLayout()
        self.pool_status = QLabel()
        self.pool_status.setWordWrap(True)
        controls.addWidget(self.pool_status, 1)
        self.remove_voice_button = QPushButton("Remove")
        self.remove_voice_button.setAccessibleName("Remove selected voice from pool")
        self.remove_voice_button.clicked.connect(self.remove_selected_voice)
        controls.addWidget(self.remove_voice_button)
        layout.addLayout(controls)
        return box

    def _update_pool_selection(self) -> None:
        self.remove_voice_button.setEnabled(bool(self.pool_table.selectedItems()))

    def render_fish_voice(self) -> QWidget:
        self.fish_voice_box = QGroupBox("Fish Audio voice")
        layout = default_form_layout()
        self.fish_voice_box.setLayout(layout)
        voice_id = ReactiveLineEdit(self.state, "catalog_voice")
        voice_id.setPlaceholderText("Voice model ID from your Fish Audio library")
        voice_id.setAccessibleName("Fish Audio voice model ID")
        voice_id.on_change.connect(
            lambda value: self.state.update({"catalog_voice": value})
        )
        layout.addRow("Voice model ID:", voice_id)
        return self.fish_voice_box

    def render_filters(self) -> QWidget:
        filters = QWidget()
        layout = QGridLayout(filters)
        layout.setContentsMargins(0, 0, 0, 0)
        language = ReactiveComboBox(self.state, "languages", "selected_language")
        gender = ReactiveComboBox(self.state, "genders", "selected_gender")
        provider = ReactiveComboBox(
            self.state, "providers", "selected_provider", render_map=TTS_PROVIDER_LABELS
        )
        provider.on_change.connect(self._on_provider_filter_change)
        model = ReactiveComboBox(self.state, "models", "selected_filter_model")
        for position, (name, control) in enumerate(
            [
                ("Provider", provider),
                ("Model", model),
                ("Language", language),
                ("Gender", gender),
            ]
        ):
            row, column = divmod(position, 2)
            label = QLabel(f"{name}:")
            label.setBuddy(control)
            control.setAccessibleName(f"Filter voices by {name.lower()}")
            control.setMinimumContentsLength(6)
            control.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            control.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            control.currentTextChanged.connect(control.setToolTip)
            control.setToolTip(control.currentText())
            layout.addWidget(label, row, column * 2)
            layout.addWidget(control, row, column * 2 + 1)
            layout.setColumnStretch(column * 2 + 1, 1)
        return filters

    def _on_provider_filter_change(self, provider: str) -> None:
        # Filter models based on provider
        available_models = {
            v["model"]
            for v in voices
            if provider == ALL or v["tts_provider"] == provider
        }
        new_models = [ALL] + sorted(available_models)

        self.state.update(
            {
                "selected_provider": provider,
                "models": new_models,
                "selected_filter_model": ALL,
            }
        )

    def render_voices_list(self) -> QWidget:
        catalog = QWidget()
        layout = QVBoxLayout(catalog)
        layout.setContentsMargins(0, 0, 0, 0)
        search_layout = QHBoxLayout()
        self.search_input = ReactiveLineEdit(self.state, "search_text")
        self.search_input.setPlaceholderText("Search voice names, languages or IDs")
        self.search_input.setAccessibleName("Search voices")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.on_change.connect(self.debounced_search)
        search_layout.addWidget(self.search_input, 1)
        self.clear_filters_button = QPushButton("Clear filters")
        self.clear_filters_button.clicked.connect(self.clear_filters)
        search_layout.addWidget(self.clear_filters_button)
        layout.addLayout(search_layout)

        self.voices_list = QListView()
        self.voices_list.setAccessibleName("Available voices")
        self.voices_list.setUniformItemSizes(True)
        self.voices_list.setSpacing(3)
        self.voices_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.voices_list.setMinimumHeight(self.fontMetrics().lineSpacing() * 7)
        self.voices_list.setMaximumHeight(self.fontMetrics().lineSpacing() * 12)
        self.voices_models = CustomListModel(self.get_visible_voice_filters())
        self.voices_list.setModel(self.voices_models)
        selection_model = self.voices_list.selectionModel()
        if selection_model:
            selection_model.selectionChanged.connect(self.voice_did_change)
        layout.addWidget(self.voices_list)
        self.catalog_status = QLabel()
        self.catalog_status.setWordWrap(True)
        layout.addWidget(self.catalog_status)
        return catalog

    def clear_filters(self) -> None:
        self.search_timer.stop()
        self.pending_search_text = ""
        self.state.update(
            {
                "selected_provider": ALL,
                "selected_filter_model": ALL,
                "selected_language": ALL,
                "selected_gender": ALL,
                "search_text": "",
                "models": [ALL] + sorted({voice["model"] for voice in voices}),
            }
        )
        self.search_input.setFocus()

    def voice_did_change(self, selected: QItemSelection) -> None:
        indexes = selected.indexes()
        if indexes and indexes[0].isValid():
            selected_index = indexes[0]
            visible = self.voices_models.get_data()
            if not 0 <= selected_index.row() < len(visible):
                return
            selected_voice = visible[selected_index.row()]
            current_provider = self.state.s["catalog_provider"]
            voice = selected_voice["voice"]
            if selected_voice["tts_provider"] == "fish":
                voice = (
                    self.state.s["catalog_voice"] if current_provider == "fish" else ""
                )

            self.state.update(
                {
                    "catalog_provider": selected_voice["tts_provider"],
                    "catalog_voice": voice,
                    "catalog_model": selected_voice["model"],
                }
            )

    def update_ui(self) -> None:
        self.update_list_ui()
        self.render_pool_rows()
        self.fish_voice_box.setVisible(self.state.s["catalog_provider"] == "fish")
        has_selection = bool(
            self._selected_voice() and self.state.s["catalog_voice"].strip()
        )
        self.add_voice_button.setEnabled(has_selection)
        can_preview = has_selection and bool(self.state.s["test_text"].strip())
        self.test_button.setEnabled(self.state.s["test_enabled"] and can_preview)

    def _selected_voice(self) -> TTSMeta | None:
        state = self.state.s
        return next(
            (
                voice
                for voice in voices
                if voice["tts_provider"] == state["catalog_provider"]
                and voice["model"] == state["catalog_model"]
                and (
                    state["catalog_provider"] == "fish"
                    or voice["voice"] == state["catalog_voice"]
                )
            ),
            None,
        )

    def update_list_ui(self) -> None:
        visible = self.get_visible_voice_filters()
        if visible != self.voices_models.get_data():
            self.voices_models.update_data(visible)
        self.catalog_status.setText(
            f"{len(visible):,} voices"
            if visible
            else "No matching voices. Clear filters to see the full catalog."
        )
        self.voices_list.setVisible(bool(visible))
        state = self.state.s
        self.clear_filters_button.setEnabled(
            bool(state["search_text"])
            or any(
                state[key] != ALL
                for key in (
                    "selected_provider",
                    "selected_filter_model",
                    "selected_language",
                    "selected_gender",
                )
            )
        )
        selected = self._selected_voice()
        if selected is None:
            self.selected_voice_label.setText(
                f"Saved voice unavailable: {state['catalog_provider']} / "
                f"{state['catalog_model']} / {state['catalog_voice']}"
                if state["catalog_voice"]
                else "Select a voice to add or preview."
            )
            self.selected_voice_label.setToolTip("")
            selection_model = self.voices_list.selectionModel()
            if selection_model is not None:
                selection_model.clearSelection()
            return
        self.selected_voice_label.update_text(
            selected, filtered=selected not in visible
        )
        if selected["tts_provider"] == "fish":
            self.selected_voice_label.setToolTip(f"Voice ID: {state['catalog_voice']}")
        selection_model = self.voices_list.selectionModel()
        if selection_model is not None and selected in visible:
            selection_model.blockSignals(True)
            index = self.voices_models.index(visible.index(selected), 0)
            if selection_model.currentIndex() != index:
                selection_model.setCurrentIndex(
                    index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                )
                self.voices_list.scrollTo(index)
            selection_model.blockSignals(False)

    def render_pool_rows(self) -> None:
        pool = self.state.s["tts_voice_pool"]
        if pool == self._rendered_pool:
            return
        self._rendered_pool = pool
        selected_row = self.pool_table.currentRow()
        self.pool_table.blockSignals(True)
        if self.pool_table.rowCount() != len(pool):
            self.pool_table.clearSelection()
            self.pool_table.setCurrentItem(None)
            self.pool_table.setRowCount(len(pool))
        for row, target in enumerate(pool):
            meta = next(
                (
                    voice
                    for voice in voices
                    if voice["tts_provider"] == target["provider"]
                    and voice["model"] == target["model"]
                    and (
                        target["provider"] == "fish"
                        or voice["voice"] == target["voice"]
                    )
                ),
                None,
            )
            provider = TTS_PROVIDER_LABELS.get(target["provider"], target["provider"])
            name = meta["friendly_voice"] if meta else target["voice"]
            if target["provider"] == "fish":
                voice_id = target["voice"]
                short_id = (
                    f"{voice_id[:8]}…{voice_id[-4:]}"
                    if len(voice_id) > 12
                    else voice_id
                )
                name = f"{short_id} · {name}"
            identity = f"{name} · {provider}"
            details = (
                f"{identity}\nModel: {target['model']}\nVoice ID: {target['voice']}"
            )
            active = self.pool_table.item(row, 0)
            if active is None:
                active = QTableWidgetItem()
                self.pool_table.setItem(row, 0, active)
            active.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            active.setCheckState(
                Qt.CheckState.Checked if target["enabled"] else Qt.CheckState.Unchecked
            )
            active.setData(Qt.ItemDataRole.AccessibleTextRole, f"Enable {identity}")
            active.setToolTip(f"Enable {identity}")
            item = self.pool_table.item(row, 1)
            if item is None:
                item = QTableWidgetItem()
                self.pool_table.setItem(row, 1, item)
            item.setText(identity)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(details)
            language = self.pool_table.item(row, 2)
            if language is None:
                language = QTableWidgetItem()
                self.pool_table.setItem(row, 2, language)
            language.setText(target.get("language") or "")
            language.setToolTip(
                "Double-click or press F2 to edit. Optional language: ja, en-US, Japanese. Blank means any language."
            )
            language.setData(
                Qt.ItemDataRole.AccessibleTextRole,
                f"Language for {identity}: {target.get('language') or 'any language'}",
            )
        if pool and selected_row >= 0:
            self.pool_table.selectRow(min(selected_row, len(pool) - 1))
        self.pool_table.blockSignals(False)
        self.pool_table.resizeRowsToContents()
        header = self.pool_table.horizontalHeader()
        assert header is not None
        self.pool_table.setFixedHeight(
            header.sizeHint().height()
            + sum(self.pool_table.rowHeight(row) for row in range(min(len(pool), 4)))
            + self.pool_table.frameWidth() * 2
        )
        self.pool_table.setVisible(bool(pool))
        enabled = sum(target["enabled"] for target in pool)
        self.pool_status.setText(
            f"{enabled} of {len(pool)} voices enabled"
            if pool
            else "No voices yet. Add one from the catalog below."
        )
        self._update_pool_selection()

    def pool_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        pool = self.state.s["tts_voice_pool"]
        if row < 0 or row >= len(pool):
            return
        target = pool[row].copy()
        if item.column() == 0:
            target["enabled"] = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 2:
            target["language"] = item.text().strip() or None
        else:
            return
        pool[row] = target
        self.state.update({"tts_voice_pool": pool})

    def add_selected_voice(self) -> None:
        provider = self.state.s["catalog_provider"]
        model = self.state.s["catalog_model"]
        voice = self.state.s["catalog_voice"].strip()
        if not voice or self._selected_voice() is None:
            return

        meta = self._selected_voice()
        language: str | None = None
        if meta and meta["language"] != ALL:
            if provider in ("google", "azure") and len(voice.split("-")) >= 2:
                language = "-".join(voice.split("-")[:2])
            else:
                language = meta["language"]

        target = make_tts_voice_target(provider, model, voice, language=language)
        if not target:
            return
        pool = self.state.s["tts_voice_pool"]
        if target in pool:
            self.pool_status.setText("That voice is already in the pool.")
            return
        pool.append(target)
        self.state.update({"tts_voice_pool": pool})
        self.pool_table.selectRow(len(pool) - 1)

    def remove_selected_voice(self) -> None:
        row = self.pool_table.currentRow()
        pool = self.state.s["tts_voice_pool"]
        if row < 0 or row >= len(pool):
            return
        pool.pop(row)
        self.state.update({"tts_voice_pool": pool})

    def render_test_voice(self) -> QWidget:
        self.test_voice_box = QWidget()
        layout = QVBoxLayout(self.test_voice_box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.preview_text = ReactiveLineEdit(self.state, "test_text")
        self.preview_text.setAccessibleName("Voice preview text")
        self.preview_text.setPlaceholderText("Text to speak in the preview")
        self.preview_text.on_change.connect(
            lambda text: self.state.update({"test_text": text})
        )
        layout.addWidget(self.preview_text)
        controls = QHBoxLayout()
        self.add_voice_button = QPushButton("Add to pool")
        self.add_voice_button.clicked.connect(self.add_selected_voice)
        controls.addWidget(self.add_voice_button)
        controls.addStretch()
        self.test_button = QPushButton("Preview voice")
        self.test_button.clicked.connect(self.test_and_play)
        controls.addWidget(self.test_button)
        self.cancel_preview_button = QPushButton("Cancel")
        self.cancel_preview_button.setAccessibleName("Cancel voice preview")
        self.cancel_preview_button.clicked.connect(self.cancel_preview)
        self.cancel_preview_button.hide()
        controls.addWidget(self.cancel_preview_button)
        layout.addLayout(controls)
        self.preview_status = QLabel(
            "Preview sends this text to the selected provider and may incur charges."
        )
        self.preview_status.setWordWrap(True)
        self.preview_status.setTextFormat(Qt.TextFormat.PlainText)
        self.preview_status.setAccessibleDescription("Voice preview status")
        layout.addWidget(self.preview_status)
        return self.test_voice_box

    def render_processing_rules(self) -> QWidget:
        self.processing_box = QGroupBox("Text preparation")
        layout = QVBoxLayout(self.processing_box)
        strip_html_box = ReactiveCheckBox(self.state, "tts_strip_html")
        strip_html_box.setText("Remove HTML formatting before speaking")
        layout.addWidget(strip_html_box)
        return self.processing_box

    def cancel_preview(self) -> None:
        if self._preview_job is not None:
            self.preview_status.setText("Cancelling preview…")
            self.cancel_preview_button.setEnabled(False)
            self._preview_job.cancel()

    @with_widget_settings
    def test_and_play(self) -> None:
        state = self.state.s
        selected = self._selected_voice()
        if (
            not state["test_enabled"]
            or selected is None
            or not state["catalog_voice"].strip()
            or not state["test_text"].strip()
        ):
            return
        provider = state["catalog_provider"]
        voice = state["catalog_voice"]
        model = state["catalog_model"]
        name = selected["friendly_voice"]
        provider_label = TTS_PROVIDER_LABELS.get(provider, provider)
        completed = False

        def on_success(audio: bytes) -> None:
            nonlocal completed
            completed = True
            play_audio(audio)
            self.preview_status.setText(f"Playing {name} · {provider_label}")

        def on_failure(err: Exception) -> None:
            nonlocal completed
            completed = True
            self.preview_status.setText(
                "Preview cancelled."
                if isinstance(err, JobCancelledError)
                else f"Preview failed: {redact_text(str(err))}"
            )

        def on_finished() -> None:
            self._preview_job = None
            if sip.isdeleted(self):
                return
            self.state.update({"test_enabled": True})
            self.cancel_preview_button.hide()
            if not completed:
                self.preview_status.setText("Preview cancelled.")

        async def fetch_audio() -> bytes:
            return await TTSProvider().async_get_tts_response(
                input=state["test_text"],
                model=model,
                provider=provider,
                voice=voice,
                strip_html=none_defaulting(state, "tts_strip_html", True),
            )

        self.state.update({"test_enabled": False})
        self.preview_status.setText(
            f"Generating preview with {name} · {provider_label}…"
        )
        self.cancel_preview_button.setEnabled(True)
        self.cancel_preview_button.show()
        try:
            self._preview_job = run_async_in_background_with_sentry(
                fetch_audio,
                on_success=on_success,
                on_failure=on_failure,
                on_finished=on_finished,
                show_error=False,
            )
        except Exception as error:
            on_failure(error)
            on_finished()

    def debounced_search(self, text: str) -> None:
        self.pending_search_text = text
        self.search_timer.stop()
        self.search_timer.start(100)

    def update_search(self) -> None:
        self.state.update({"search_text": self.pending_search_text})

    def get_visible_voice_filters(self) -> list[TTSMeta]:
        filtered = []
        state = self.state.s
        search_terms = state["search_text"].lower().strip().split()

        for voice in voices:
            matches_provider = (
                state["selected_provider"] == ALL
                or voice["tts_provider"] == state["selected_provider"]
            )
            if not matches_provider:
                continue

            matches_gender = (
                state["selected_gender"] == ALL
                or voice["gender"] == state["selected_gender"]
            )
            if not matches_gender:
                continue

            matches_language = (
                state["selected_language"] == ALL
                or voice["language"] == ALL
                or voice["language"] == state["selected_language"]
            )

            if not matches_language:
                continue

            matches_model = (
                state["selected_filter_model"] == ALL
                or voice["model"] == state["selected_filter_model"]
            )

            if not matches_model:
                continue

            # Search works by splitting the user's input into terms and the formatted
            # voice display text into words. Each search term must match (via substring)
            # at least one word in the voice. All search terms must match for inclusion.
            # The voice_search_cache contains pre-split words from format_voice() output.
            voice_key = (voice["tts_provider"], voice["voice"], voice["model"])
            voice_words = voice_search_cache[voice_key] + [
                voice["voice"].lower(),
                voice["model"].lower(),
            ]
            matches_search = not search_terms or all(
                any(term in word for word in voice_words) for term in search_terms
            )
            if not matches_search:
                continue
            filtered.append(voice)
        return filtered

    def get_initial_state(
        self, tts_options: OverrideableTTSOptionsDict | None
    ) -> TTSState:
        pool = normalize_tts_voice_pool(
            key_or_config_val(tts_options, "tts_voice_pool")
        )
        selected_target = next(
            (target for target in pool if target["enabled"]), pool[0] if pool else None
        )
        return {
            "providers": providers,
            "selected_provider": ALL,
            "genders": [ALL, "Male", "Female"],
            "selected_gender": ALL,
            "languages": languages,
            "selected_language": ALL,
            "models": [ALL] + sorted({v["model"] for v in voices}),
            "selected_filter_model": ALL,
            "catalog_provider": cast(
                "TTSProviders",
                selected_target["provider"] if selected_target else config.tts_provider,
            ),
            "catalog_voice": selected_target["voice"]
            if selected_target
            else config.tts_voice,
            "catalog_model": cast(
                "TTSModels",
                selected_target["model"] if selected_target else config.tts_model,
            ),
            "tts_voice_pool": pool,
            "tts_strip_html": key_or_config_val(tts_options, "tts_strip_html"),
            "test_text": default_texts[ALL],
            "test_enabled": True,
            "search_text": "",
        }
