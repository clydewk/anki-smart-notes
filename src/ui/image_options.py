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

from aqt import QGroupBox, QVBoxLayout, QWidget

from ..config import config, key_or_config_val
from ..models import (
    ImageAspectRatio,
    ImageModels,
    ImageProviders,
    ImageResolution,
    OverridableImageOptionsDict,
)
from .reactive_combo_box import ReactiveComboBox
from .state_manager import StateManager
from .ui_utils import default_form_layout


class State(TypedDict):
    image_provider: ImageProviders
    image_providers: list[ImageProviders]
    image_model: ImageModels
    image_models: list[ImageModels]
    image_aspect_ratio: ImageAspectRatio
    image_aspect_ratios: list[ImageAspectRatio]
    image_resolution: ImageResolution
    image_resolutions: list[ImageResolution]


provider_labels = {
    "replicate": "Replicate",
    "google": "Google",
}

model_labels = {
    "flux-schnell": "Flux Schnell (1x Image Cost)",
    "flux-dev": "Flux Dev (8x Image Cost)",
    "gemini-3-pro-image-preview": "Google Gemini 3 Pro (Image Preview)",
}

provider_models: dict[ImageProviders, list[ImageModels]] = {
    "replicate": ["flux-schnell", "flux-dev"],
    "google": ["gemini-3-pro-image-preview"],
}

aspect_ratios: list[ImageAspectRatio] = ["1:1", "16:9", "4:3", "3:4", "9:16"]
resolutions: list[ImageResolution] = ["1024x1024", "2048x2048"]


class ImageOptions(QWidget):
    def __init__(
        self, image_options: Optional[OverridableImageOptionsDict] = None
    ) -> None:
        super().__init__()

        # Helper to get initial value
        def get_val(key: str, default: Any = None) -> Any:
            return key_or_config_val(image_options or {}, key)

        provider = get_val("image_provider", config.image_provider)
        # Fallback if provider is invalid or not in map (shouldn't happen with correct config)
        if provider not in provider_models:
            provider = "replicate"

        models = provider_models[provider]
        model = get_val("image_model", config.image_model)
        if model not in models:
            model = models[0]

        aspect_ratio = get_val("image_aspect_ratio", config.image_aspect_ratio) or "1:1"
        resolution = get_val("image_resolution", config.image_resolution) or "1024x1024"

        self.state = StateManager[State](
            {
                "image_provider": provider,
                "image_providers": list(provider_models.keys()),
                "image_model": model,
                "image_models": models,
                "image_aspect_ratio": aspect_ratio,
                "image_aspect_ratios": aspect_ratios,
                "image_resolution": resolution,
                "image_resolutions": resolutions,
            }
        )

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.provider_picker = ReactiveComboBox(
            self.state,
            "image_providers",
            "image_provider",
            provider_labels,
        )
        self.provider_picker.on_change.connect(self._on_provider_change)

        self.model_picker = ReactiveComboBox(
            self.state,
            "image_models",
            "image_model",
            model_labels,
        )
        self.model_picker.setMaximumWidth(300)

        self.ratio_picker = ReactiveComboBox(
            self.state,
            "image_aspect_ratios",
            "image_aspect_ratio",
        )

        self.resolution_picker = ReactiveComboBox(
            self.state,
            "image_resolutions",
            "image_resolution",
        )

        box = QGroupBox("🖼️ Image Model Settings")
        layout = QVBoxLayout()
        layout.addWidget(box)
        box_layout = default_form_layout()
        box.setLayout(box_layout)

        box_layout.addRow("Provider:", self.provider_picker)
        box_layout.addRow("Model:", self.model_picker)
        box_layout.addRow("Aspect Ratio:", self.ratio_picker)
        box_layout.addRow("Resolution:", self.resolution_picker)

        self.setLayout(layout)

    def _on_provider_change(self, provider: str) -> None:
        provider = cast("ImageProviders", provider)
        models = provider_models[provider]
        self.state.update({"image_models": models})

        # Select first model if current is not in new list
        current_model = self.state.s["image_model"]
        if current_model not in models:
            self.state.update({"image_model": models[0]})
