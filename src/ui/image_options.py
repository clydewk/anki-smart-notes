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

from typing import Any, TypedDict, cast

from aqt import QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..config import config, key_or_config_val
from ..models import (
    RETIRED_PROVIDER_MODELS,
    ImageAspectRatio,
    ImageGenerationQuality,
    ImageModels,
    ImageOutputFormat,
    ImageProviders,
    ImageResolution,
    OverridableImageOptionsDict,
    image_generation_qualities,
    openai_image_size,
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
    image_generation_quality: ImageGenerationQuality
    image_generation_qualities: list[ImageGenerationQuality]


provider_labels = {
    "replicate": "Replicate",
    "google": "Google",
    "openai": "OpenAI",
}

model_labels = {
    "gpt-image-2.5-sunburst": "GPT Image 2.5 Sunburst",
    "gpt-image-2.5-flare": "GPT Image 2.5 Flare",
    "flux-schnell": "Flux Schnell",
    "flux-dev": "Flux Dev",
    "gemini-3-pro-image": "Google Gemini 3 Pro Image",
    "gemini-3-pro-image-preview": "Google Gemini 3 Pro Image (Unavailable)",
    "gpt-image-2": "GPT Image 2",
    "gpt-image-1.5": "GPT Image 1.5",
    "gpt-image-1": "GPT Image 1",
    "gpt-image-1-mini": "GPT Image 1 Mini",
    "dall-e-3": "DALL·E 3 (Deprecated)",
}

provider_models: dict[ImageProviders, list[ImageModels]] = {
    "replicate": ["flux-schnell", "flux-dev"],
    "google": ["gemini-3-pro-image"],
    "openai": [
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
        "gpt-image-2",
        "gpt-image-1.5",
        "gpt-image-1",
        "gpt-image-1-mini",
        "dall-e-3",
    ],
}

aspect_ratios: list[ImageAspectRatio] = ["1:1", "16:9", "4:3", "3:4", "9:16"]
resolutions: list[ImageResolution] = ["1024x1024", "2048x2048", "4096x4096"]
output_formats: list[ImageOutputFormat] = ["webp", "png", "jpeg", "avif"]

format_labels = {
    "webp": "WebP",
    "png": "PNG (Lossless)",
    "jpeg": "JPEG",
    "avif": "AVIF",
}


def image_output_summary(
    provider: str,
    model: str,
    aspect_ratio: ImageAspectRatio,
    resolution: ImageResolution,
    output_format: ImageOutputFormat,
) -> str:
    if provider == "google":
        preset = {"1024x1024": "1K", "2048x2048": "2K", "4096x4096": "4K"}.get(
            resolution, resolution
        )
        dimensions = f"{aspect_ratio}, {preset} requested; pixel dimensions are selected by the provider"
    elif provider == "replicate":
        dimensions = f"{aspect_ratio}; pixel dimensions are selected by the provider"
    else:
        dimensions = (
            openai_image_size(model, aspect_ratio, resolution).replace("x", " × ")
            + " pixels requested"
        )
    return f"Output: {dimensions}. Saved as {output_format.upper()}."


class ImageOptions(QWidget):
    def __init__(
        self, image_options: OverridableImageOptionsDict | None = None
    ) -> None:
        super().__init__()

        self._provider_models: dict[str, list[str]] = {}
        self._provider_labels = dict(provider_labels)
        self._model_labels = dict(model_labels)
        self._reset_to_base_models()
        self._inject_custom_models()

        # Helper to get initial value
        def get_val(key: str, default: Any = None) -> Any:
            return key_or_config_val(image_options or {}, key)

        provider = get_val("image_provider", config.image_provider)
        available_providers = list(self._provider_models)
        if provider not in available_providers:
            available_providers.append(provider)
        models = list(self._provider_models.get(provider, []))
        model = get_val("image_model", config.image_model)
        if model and model not in models:
            models.append(model)

        aspect_ratio = get_val("image_aspect_ratio", config.image_aspect_ratio) or "1:1"
        resolution = get_val("image_resolution", config.image_resolution) or "1024x1024"
        format = get_val("image_output_format", config.image_output_format) or "webp"
        quality = get_val("image_quality", config.image_quality)
        if quality is None:
            quality = 85

        self.state = StateManager[State](
            {
                "image_provider": provider,
                "image_providers": cast("list[ImageProviders]", available_providers),
                "image_model": model,
                "image_models": cast("list[ImageModels]", models),
                "image_aspect_ratio": aspect_ratio,
                "image_aspect_ratios": aspect_ratios,
                "image_resolution": resolution,
                "image_resolutions": resolutions,
                "image_output_format": format,
                "image_output_formats": output_formats,
                "image_quality": int(quality),
                "image_generation_quality": key_or_config_val(
                    image_options or {}, "image_generation_quality"
                )
                or "auto",
                "image_generation_qualities": image_generation_qualities(model),
            }
        )

        self._setup_ui()
        self._on_format_change(format)
        self._on_model_change(model)

    def refresh_custom_providers(self) -> None:
        self._reset_to_base_models()
        self._inject_custom_models()
        current_provider = self.state.s["image_provider"]
        providers = list(self._provider_models)
        if current_provider not in providers:
            providers.append(current_provider)
        self.state.update({"image_providers": providers})
        self._on_provider_change(current_provider, preserve_current=True)

    def _reset_to_base_models(self) -> None:
        self._provider_models = {
            provider: list(models) for provider, models in provider_models.items()
        }
        self._provider_labels = dict(provider_labels)
        self._model_labels = dict(model_labels)

    def _inject_custom_models(self) -> None:
        for provider in config.custom_providers or []:
            capabilities = provider.get("capabilities", [])
            if capabilities and "image" not in capabilities:
                continue
            name = provider["name"]
            self._provider_models[name] = list(
                provider.get("image_models") or provider.get("models") or []
            )
            self._provider_labels[name] = name

    def _setup_ui(self) -> None:
        self.provider_picker = ReactiveComboBox(
            self.state,
            "image_providers",
            "image_provider",
            self._provider_labels,
        )
        self.provider_picker.on_change.connect(self._on_provider_change)

        self.model_picker = ReactiveComboBox(
            self.state,
            "image_models",
            "image_model",
            self._model_labels,
        )
        self.model_picker.setMinimumWidth(0)
        self.model_picker.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.model_picker.on_change.connect(self._on_model_change)
        self.generation_quality_picker = ReactiveComboBox(
            self.state,
            "image_generation_qualities",
            "image_generation_quality",
            {
                "auto": "Auto",
                "low": "Low",
                "medium": "Medium",
                "high": "High",
                "xhigh": "Extra High",
                "max": "Maximum",
            },
        )
        self.generation_quality_picker.on_change.connect(
            lambda _: self._on_model_change(self.state.s["image_model"])
        )
        self.generation_quality_label = QLabel("Rendering quality:")

        self.ratio_picker = ReactiveComboBox(
            self.state,
            "image_aspect_ratios",
            "image_aspect_ratio",
        )

        self.ratio_picker.on_change.connect(lambda _: self._update_output_summary())

        self.resolution_picker = ReactiveComboBox(
            self.state,
            "image_resolutions",
            "image_resolution",
            {
                "1024x1024": "1K preset",
                "2048x2048": "2K preset",
                "4096x4096": "4K preset",
            },
        )
        self.resolution_picker.on_change.connect(lambda _: self._update_size_controls())

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
        self.quality_spinner.setRange(0, 100)
        self.quality_spinner.setSuffix("%")

        self.quality_label = QLabel("File compression quality:")

        self.generation_quality_help = QLabel(
            "Higher rendering quality can take longer and cost more. This changes the image generated by the model."
        )
        self.generation_quality_help.setWordWrap(True)
        self.compression_help = QLabel(
            "Higher values retain more detail and produce larger files. This controls encoding, not model rendering quality."
        )
        self.compression_help.setWordWrap(True)
        self.output_summary = QLabel()
        self.output_summary.setWordWrap(True)
        self.output_summary.setAccessibleDescription("Effective image output")
        self.resolution_label = QLabel("Resolution preset:")
        self.ratio_label = QLabel("Image shape:")
        box = QGroupBox("Image model")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(box)
        box_layout = default_form_layout()
        box.setLayout(box_layout)

        box_layout.addRow("Provider:", self.provider_picker)
        box_layout.addRow("Model:", self.model_picker)
        self.model_warning = QLabel()
        self.model_warning.setWordWrap(True)
        box_layout.addRow(self.model_warning)
        box_layout.addRow(self.generation_quality_label, self.generation_quality_picker)
        box_layout.addRow(self.generation_quality_help)
        box_layout.addRow(self.ratio_label, self.ratio_picker)
        box_layout.addRow(self.resolution_label, self.resolution_picker)
        box_layout.addRow(self.output_summary)
        media_box = QGroupBox("Saved media")
        media_form = default_form_layout()
        media_box.setLayout(media_form)
        media_form.addRow("File format:", self.format_picker)
        media_form.addRow(self.quality_label, self.quality_spinner)
        media_form.addRow(self.compression_help)
        media_help = QLabel(
            "The generated image is converted locally when needed to save it in this format. PNG is lossless; other formats use the selected compression quality."
        )
        media_help.setWordWrap(True)
        media_form.addRow(media_help)
        layout.addWidget(media_box)
        layout.addStretch()

        self.setLayout(layout)

    def _on_provider_change(
        self, provider: str, preserve_current: bool = False
    ) -> None:
        models = list(self._provider_models.get(provider, []))
        current_model = self.state.s["image_model"]
        if preserve_current and current_model and current_model not in models:
            models.append(current_model)
        selected = (
            current_model if current_model in models else (models[0] if models else "")
        )
        self.state.update({"image_models": models, "image_model": selected})
        self._on_model_change(selected)

    def _on_model_change(self, model: str) -> None:
        provider = self.state.s["image_provider"]
        unavailable = (
            provider not in self._provider_models
            or (provider, model) in RETIRED_PROVIDER_MODELS
        )
        self.model_warning.setText(
            "Saved provider or model is unavailable. Choose a supported selection before generating."
            if unavailable
            else ""
        )
        self.model_warning.setVisible(unavailable)
        self.model_picker.setToolTip(model)
        qualities = image_generation_qualities(model)
        quality = self.state.s["image_generation_quality"]
        if model.startswith("gpt-image-") and quality not in qualities:
            qualities.append(quality)
            self.model_warning.setText(
                self.model_warning.text()
                + (" " if unavailable else "")
                + f"Saved rendering quality ‘{quality}’ is not supported by this model. Choose a supported quality."
            )
            self.model_warning.show()
        self.state.update({"image_generation_qualities": qualities})
        has_quality = len(qualities) > 1
        self.generation_quality_picker.setVisible(has_quality)
        self.generation_quality_label.setVisible(has_quality)
        self.generation_quality_help.setVisible(has_quality)
        self._update_size_controls()

    def _update_size_controls(self) -> None:
        provider = self.state.s["image_provider"]
        model = self.state.s["image_model"]
        resolution = self.state.s["image_resolution"]
        uses_openai_sizes = provider not in {"google", "replicate"}
        flexible_size = model == "gpt-image-2" or model.startswith("gpt-image-2.5-")
        uses_resolution = provider == "google" or (uses_openai_sizes and flexible_size)
        self.resolution_picker.setVisible(uses_resolution)
        self.resolution_label.setVisible(uses_resolution)
        ratios: list[ImageAspectRatio] = list(aspect_ratios)
        current = self.state.s["image_aspect_ratio"]
        labels: dict[str, str] = {}
        if uses_openai_sizes:
            if not flexible_size:
                ratios = ["1:1", "16:9", "9:16"]
                if current not in ratios:
                    ratios.append(current)
            for ratio in ratios:
                size = openai_image_size(model, ratio, resolution).replace("x", " × ")
                shape = {"1:1": "square", "16:9": "landscape", "9:16": "portrait"}.get(
                    ratio, f"saved {ratio}"
                )
                labels[ratio] = (
                    f"{size} · {ratio} preset" if flexible_size else f"{size} · {shape}"
                )
        self.ratio_picker.state_to_ui = labels
        self.ratio_picker.ui_to_state = {
            label: ratio for ratio, label in labels.items()
        }
        self.state.update({"image_aspect_ratios": ratios})
        self.ratio_picker.update_from_state(dict(self.state.s))
        self._update_output_summary()

    def _update_output_summary(self) -> None:
        self.output_summary.setText(
            image_output_summary(
                self.state.s["image_provider"],
                self.state.s["image_model"],
                self.state.s["image_aspect_ratio"],
                self.state.s["image_resolution"],
                self.state.s["image_output_format"],
            )
        )

    def _on_format_change(self, format: str) -> None:
        is_lossy = format in {"jpeg", "webp", "avif"}
        self.quality_label.setVisible(is_lossy)
        self.quality_spinner.setVisible(is_lossy)
        self.compression_help.setVisible(is_lossy)
        self._update_output_summary()
