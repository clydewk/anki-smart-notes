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

from aqt import QGroupBox, QLabel, QVBoxLayout, QWidget

from ..config import config, key_or_config_val
from ..models import (
    ImageAspectRatio,
    ImageModels,
    ImageOutputFormat,
    ImageProviders,
    ImageResolution,
    OverridableImageOptionsDict,
)
from .reactive_combo_box import ReactiveComboBox
from .reactive_spin_box import ReactiveSpinBox
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
    image_output_format: ImageOutputFormat
    image_output_formats: list[ImageOutputFormat]
    image_quality: int


provider_labels = {
    "replicate": "Replicate",
    "google": "Google",
    "openai": "OpenAI",
}

model_labels = {
    "flux-schnell": "Flux Schnell (1x Image Cost)",
    "flux-dev": "Flux Dev (8x Image Cost)",
    "gemini-3-pro-image-preview": "Google Gemini 3 Pro (Image Preview)",
    "gpt-image-1.5": "GPT Image 1.5 (SOTA)",
    "gpt-image-1": "GPT Image 1",
    "gpt-image-1-mini": "GPT Image 1 Mini (Cost Effective)",
    "dall-e-3": "DALL·E 3 (Deprecated)",
}

provider_models: dict[ImageProviders, list[ImageModels]] = {
    "replicate": ["flux-schnell", "flux-dev"],
    "google": ["gemini-3-pro-image-preview"],
    "openai": ["gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini", "dall-e-3"],
}

aspect_ratios: list[ImageAspectRatio] = ["1:1", "16:9", "4:3", "3:4", "9:16"]
resolutions: list[ImageResolution] = ["1024x1024", "2048x2048", "4096x4096"]
output_formats: list[ImageOutputFormat] = ["webp", "png", "jpeg", "avif"]

format_labels = {
    "webp": "WebP (Default)",
    "png": "PNG (Lossless)",
    "jpeg": "JPEG",
    "avif": "AVIF",
}


class ImageOptions(QWidget):
    def __init__(
        self, image_options: Optional[OverridableImageOptionsDict] = None
    ) -> None:
        super().__init__()

        # Merge custom providers
        self._inject_custom_models()

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
            # Handle case where model might be a custom one
            if provider in provider_models and model in provider_models[provider]:
                pass  # all good
            elif models:
                model = models[0]
            else:
                model = ""  # Should not happen unless empty provider

        aspect_ratio = get_val("image_aspect_ratio", config.image_aspect_ratio) or "1:1"
        resolution = get_val("image_resolution", config.image_resolution) or "1024x1024"
        format = get_val("image_output_format", config.image_output_format) or "webp"
        quality = get_val("image_quality", config.image_quality)
        if quality is None:
            quality = 85

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
                "image_output_format": format,
                "image_output_formats": output_formats,
                "image_quality": int(quality),
            }
        )

        self._setup_ui()
        self._on_format_change(format)

    def refresh_custom_providers(self) -> None:
        # Reset to base state
        self._reset_to_base_models()
        # Inject new custom models
        self._inject_custom_models()

        # Update state
        new_providers = list(provider_models.keys())
        self.state.update({"image_providers": new_providers})

        # Handle current provider validity
        current_provider = self.state.s["image_provider"]
        if current_provider not in new_providers:
            # Fallback
            fallback = "replicate"
            self.state.update({"image_provider": fallback})
            self._on_provider_change(fallback)
        else:
            # Refresh models for current provider
            self._on_provider_change(current_provider)

    def _reset_to_base_models(self) -> None:
        # Restore provider_models to initial state (remove non-standard keys)
        # and clean up model_labels

        standard_providers = ["replicate", "google", "openai"]

        # Remove custom providers from provider_models
        keys_to_remove = [
            k for k in provider_models if k not in standard_providers
        ]
        for k in keys_to_remove:
            del provider_models[k]
            if k in provider_labels:
                del provider_labels[k]

        # We don't easily know which models in model_labels are custom without tracking them
        # but we can just rebuild the custom labels in _inject_custom_models
        # For now, let's just leave potentially unused labels in model_labels as they don't harm
        # unless name collision, but _inject will overwrite.

    def _inject_custom_models(self) -> None:
        if not config.custom_providers:
            return

        for provider in config.custom_providers:
            caps = provider.get("capabilities", [])
            # If capabilities list is empty or explicitly contains "image"
            if not caps or "image" in caps:
                p_name = provider["name"]

                # Check for image_models first, then fallback to models
                models_to_add = []
                if provider.get("image_models"):
                    models_to_add = provider["image_models"]  # type: ignore
                elif provider.get("models"):
                    models_to_add = provider["models"]  # type: ignore

                if not models_to_add:
                    continue

                # Add/Update provider models map
                provider_models[p_name] = models_to_add  # type: ignore

                # Add to labels map if not present
                if p_name not in provider_labels:
                    provider_labels[p_name] = p_name

                for m in models_to_add:
                    if m not in model_labels:
                        model_labels[m] = f"{m} ({p_name})"

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

        self.format_picker = ReactiveComboBox(
            self.state,
            "image_output_formats",
            "image_output_format",
            format_labels,
        )
        self.format_picker.on_change.connect(self._on_format_change)

        self.quality_spinner = ReactiveSpinBox(
            self.state,
            "image_quality",
        )
        self.quality_spinner.setRange(1, 100)
        self.quality_spinner.setSuffix("%")

        self.quality_label = QLabel("Quality:")

        box = QGroupBox("🖼️ Image Model Settings")
        layout = QVBoxLayout()
        layout.addWidget(box)
        box_layout = default_form_layout()
        box.setLayout(box_layout)

        box_layout.addRow("Provider:", self.provider_picker)
        box_layout.addRow("Model:", self.model_picker)
        box_layout.addRow("Aspect Ratio:", self.ratio_picker)
        box_layout.addRow("Resolution:", self.resolution_picker)
        box_layout.addRow("Output Format:", self.format_picker)
        box_layout.addRow(self.quality_label, self.quality_spinner)

        self.setLayout(layout)

    def _on_provider_change(self, provider: str) -> None:
        provider = cast("ImageProviders", provider)
        models = provider_models[provider]
        self.state.update({"image_models": models})

        # Select first model if current is not in new list
        current_model = self.state.s["image_model"]
        if current_model not in models:
            self.state.update({"image_model": models[0]})

    def _on_format_change(self, format: str) -> None:
        # Show quality only for lossy formats
        # PNG is usually lossless (compression level 0-9), we can reuse quality spinner
        # but standard PIL usage maps quality to JPEG/WEBP/AVIF.
        # For this UI, let's hide quality for PNG to keep it simple as "Lossless"
        is_lossy = format in ["jpeg", "webp", "avif"]
        self.quality_label.setVisible(is_lossy)
        self.quality_spinner.setVisible(is_lossy)
