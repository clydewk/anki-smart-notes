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

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, NotRequired, TypedDict, cast
from urllib.parse import urlparse

# Providers

TTSProviders = Literal["openai", "elevenLabs", "google", "azure", "fish"]
ChatProviders = Literal["openai", "chatgpt", "anthropic", "deepseek", "google"]

# Reasoning Efforts
# "none" is a UI concept meaning "don't use reasoning, use temperature instead"
OpenAIReasoningEffort = str
OPENAI_DEFAULT_REASONING_EFFORTS: tuple[OpenAIReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
)
OPENAI_REASONING_EFFORTS_WITH_NONE_AND_XHIGH: tuple[OpenAIReasoningEffort, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)
OPENAI_REASONING_EFFORTS_WITH_MAX: tuple[OpenAIReasoningEffort, ...] = (
    *OPENAI_REASONING_EFFORTS_WITH_NONE_AND_XHIGH,
    "max",
)


# Chat Models

OpenAIModels = str
DeepseekModels = str
AnthropicModels = str
GoogleChatModels = str
ChatModels = str


@dataclass(frozen=True)
class ChatGPTModel:
    model: str
    label: str
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str | None


@dataclass(frozen=True)
class OpenAIChatModelSpec:
    model: str
    label: str
    reasoning_efforts: tuple[OpenAIReasoningEffort, ...]
    temperature_without_reasoning: bool = True
    chat_tools: bool = True
    chat_tools_without_reasoning_only: bool = False


OPENAI_CHAT_MODEL_CATALOG: tuple[OpenAIChatModelSpec, ...] = (
    OpenAIChatModelSpec(
        "gpt-6-astra",
        "GPT-6 Astra",
        ("low", "medium", "high", "xhigh", "max"),
        temperature_without_reasoning=False,
        chat_tools=False,
    ),
    OpenAIChatModelSpec(
        "gpt-6.1-sol",
        "GPT-6.1 Sol",
        ("low", "medium", "high", "xhigh", "max"),
        temperature_without_reasoning=False,
        chat_tools=False,
    ),
    OpenAIChatModelSpec(
        "gpt-6-sol",
        "GPT-6 Sol",
        OPENAI_REASONING_EFFORTS_WITH_MAX,
        chat_tools_without_reasoning_only=True,
    ),
    OpenAIChatModelSpec(
        "gpt-6-luna",
        "GPT-6 Luna",
        OPENAI_REASONING_EFFORTS_WITH_MAX,
        chat_tools_without_reasoning_only=True,
    ),
    OpenAIChatModelSpec(
        "gpt-5.6-sol", "GPT-5.6 Sol", OPENAI_REASONING_EFFORTS_WITH_MAX
    ),
    OpenAIChatModelSpec(
        "gpt-5.6-terra", "GPT-5.6 Terra", OPENAI_REASONING_EFFORTS_WITH_MAX
    ),
    OpenAIChatModelSpec(
        "gpt-5.6-luna", "GPT-5.6 Luna", OPENAI_REASONING_EFFORTS_WITH_MAX
    ),
    OpenAIChatModelSpec(
        "gpt-5.5",
        "GPT-5.5",
        OPENAI_REASONING_EFFORTS_WITH_NONE_AND_XHIGH,
    ),
)

# Order that the models are displayed in the curated OpenAI UI
openai_chat_models: list[ChatModels] = [
    spec.model for spec in OPENAI_CHAT_MODEL_CATALOG
]
OPENAI_CHAT_MODEL_LABELS: dict[str, str] = {
    spec.model: spec.label for spec in OPENAI_CHAT_MODEL_CATALOG
}

anthropic_chat_models: list[ChatModels] = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

deepseek_chat_models: list[ChatModels] = ["deepseek-v4-flash", "deepseek-v4-pro"]

google_chat_models: list[ChatModels] = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

provider_model_map: dict[ChatProviders, list[ChatModels]] = {
    "openai": openai_chat_models,
    "chatgpt": [],
    "anthropic": anthropic_chat_models,
    "deepseek": deepseek_chat_models,
    "google": google_chat_models,
}


RETIRED_PROVIDER_MODELS: set[tuple[str, str]] = {
    ("anthropic", "claude-opus-4-1"),
    ("anthropic", "claude-opus-4-1-20250805"),
    ("anthropic", "claude-sonnet-4-0"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("deepseek", "deepseek-v3"),
    ("deepseek", "deepseek-chat"),
    ("deepseek", "deepseek-reasoner"),
    ("google", "gemini-3-pro-preview"),
    ("google", "gemini-3-pro-image-preview"),
}


OPENAI_MODEL_SHUTDOWN_DATES: dict[str, date] = {
    model: shutdown
    for shutdown, model_ids in (
        (date(2025, 10, 27), ("o1-mini",)),
        (date(2026, 2, 17), ("chatgpt-4o-latest",)),
        (date(2026, 5, 12), ("dall-e-2", "dall-e-3")),
        (
            date(2026, 7, 23),
            (
                "gpt-5-chat-latest",
                "gpt-5-codex",
                "gpt-5.1-chat-latest",
                "gpt-5.1-codex",
                "gpt-5.1-codex-max",
                "gpt-5.1-codex-mini",
                "gpt-5.2-codex",
                "o3-deep-research",
                "o3-deep-research-2025-06-26",
                "o4-mini-deep-research",
                "o4-mini-deep-research-2025-06-26",
            ),
        ),
        (date(2026, 8, 10), ("gpt-5.2-chat-latest", "gpt-5.3-chat-latest")),
        (
            date(2026, 10, 23),
            (
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-completions",
                "gpt-4-0613",
                "gpt-4",
                "gpt-4-0613-completions",
                "gpt-4-completions",
                "gpt-4-1106-preview",
                "gpt-4-turbo",
                "gpt-4-turbo-2024-04-09",
                "gpt-4-turbo-completions",
                "gpt-4.1-nano",
                "gpt-4.1-nano-2025-04-14",
                "gpt-4o-2024-05-13",
                "gpt-image-1",
                "o1-2024-12-17",
                "o1",
                "o1-pro-2025-03-19",
                "o1-pro",
                "o3-mini-2025-01-31",
                "o3-mini",
                "o4-mini-2025-04-16",
                "o4-mini",
            ),
        ),
        (
            date(2026, 12, 1),
            ("gpt-image-1.5", "gpt-image-1-mini", "chatgpt-image-latest"),
        ),
        (
            date(2026, 12, 11),
            (
                "gpt-5-2025-08-07",
                "gpt-5-mini-2025-08-07",
                "gpt-5-nano-2025-08-07",
                "gpt-5-pro-2025-10-06",
                "o3-2025-04-16",
                "o3-pro-2025-06-10",
            ),
        ),
        (
            date(2027, 1, 6),
            (
                "tts-1",
                "tts-1-hd",
                "gpt-4o-mini-tts-2025-03-20",
                "gpt-4o-mini-tts-2025-12-15",
            ),
        ),
        (date(2027, 4, 1), ("gpt-5.1", "gpt-5.4-nano", "gpt-5.3-codex")),
    )
    for model in model_ids
}


def provider_model_unavailable(provider: str, model: str) -> bool:
    shutdown = OPENAI_MODEL_SHUTDOWN_DATES.get(model) if provider == "openai" else None
    return (provider, model) in RETIRED_PROVIDER_MODELS or (
        shutdown is not None and date.today() >= shutdown
    )


def provider_model_warning(provider: str, model: str) -> str | None:
    if provider_model_unavailable(provider, model):
        return f"{provider} model {model} is unavailable. Choose a supported model in settings."
    shutdown = OPENAI_MODEL_SHUTDOWN_DATES.get(model) if provider == "openai" else None
    if shutdown is not None:
        return f"{model} retires on {shutdown.isoformat()}. Choose a supported model before then."
    if provider == "openai" and model == "gpt-4o-mini-tts":
        return (
            "This alias currently uses a speech snapshot that retires on 2027-01-06. "
            "Consider gpt-realtime-2.1-mini; OpenAI has not announced an alias shutdown date."
        )
    return None


def validate_provider_model(provider: str, model: str) -> None:
    if provider_model_unavailable(provider, model):
        raise ValueError(provider_model_warning(provider, model))


OpenAIServiceTier = Literal["auto", "default", "fast", "ultrafast"]
OPENAI_SERVICE_TIERS: dict[OpenAIServiceTier, str] = {
    "auto": "Project default",
    "default": "Standard",
    "fast": "Fast (premium pricing)",
    "ultrafast": "Ultrafast (premium pricing)",
}
OPENAI_FAST_MODELS = frozenset(
    [spec.model for spec in OPENAI_CHAT_MODEL_CATALOG]
    + [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "o3",
        "o4-mini",
        "gpt-5.3-codex",
    ]
)


def validate_openai_service_tier(model: str, tier: str, endpoint: str | None) -> None:
    if tier not in OPENAI_SERVICE_TIERS:
        raise ValueError(
            "Choose a supported OpenAI service tier in Providers settings."
        )
    if tier in {"auto", "default"}:
        return
    hostname = urlparse(endpoint or "https://api.openai.com").hostname
    if tier == "ultrafast":
        regions = {"api.openai.com", "us.api.openai.com"}
        if model == "gpt-6.1-sol":
            regions.add("eu.api.openai.com")
        if model not in {"gpt-6-astra", "gpt-6.1-sol"} or hostname not in regions:
            raise ValueError(
                "Ultrafast requires GPT-6 Astra (global/US) or GPT-6.1 Sol "
                "(global/US/EU) on the OpenAI Responses API. Change the model, host, or service tier."
            )
    elif model not in OPENAI_FAST_MODELS:
        raise ValueError(
            f"Fast mode is not supported for {model}. Change the model or service tier."
        )
    elif model == "gpt-6-astra" and hostname == "eu.api.openai.com":
        raise ValueError(
            "GPT-6 Astra Fast mode is unavailable with EU processing. Change the model or service tier."
        )


legacy_openai_chat_models: list[str] = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "o3-mini",
    "o1-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o4-mini",
]

OPENAI_REASONING_EFFORTS_BY_MODEL: dict[str, tuple[OpenAIReasoningEffort, ...]] = {
    **{spec.model: spec.reasoning_efforts for spec in OPENAI_CHAT_MODEL_CATALOG},
    "gpt-5.6": OPENAI_REASONING_EFFORTS_WITH_MAX,
}


def openai_reasoning_efforts_for_model(model: str) -> list[OpenAIReasoningEffort]:
    key = model.lower()
    return list(
        OPENAI_REASONING_EFFORTS_BY_MODEL.get(key, OPENAI_DEFAULT_REASONING_EFFORTS)
    )


def openai_supports_temperature(model: str, reasoning_effort: str | None) -> bool:
    spec = next(
        (spec for spec in OPENAI_CHAT_MODEL_CATALOG if spec.model == model.lower()),
        None,
    )
    return reasoning_effort in {None, "none"} and (
        spec is None or spec.temperature_without_reasoning
    )


def openai_supports_chat_tools(model: str, reasoning_effort: str | None = None) -> bool:
    spec = next(
        (spec for spec in OPENAI_CHAT_MODEL_CATALOG if spec.model == model.lower()),
        None,
    )
    return spec is None or (
        spec.chat_tools
        and (not spec.chat_tools_without_reasoning_only or reasoning_effort == "none")
    )


def openai_model_label(model: str) -> str:
    if model in OPENAI_CHAT_MODEL_LABELS:
        return OPENAI_CHAT_MODEL_LABELS[model]

    chat_latest_match = re.fullmatch(r"gpt-(\d+(?:\.\d+)?)-chat-latest", model)
    if chat_latest_match:
        return f"GPT-{chat_latest_match.group(1)} Chat Latest"

    model_match = re.fullmatch(
        r"gpt-(\d+(?:\.\d+)?)(?:-(mini|nano|pro|sol|terra|luna))?", model
    )
    if not model_match:
        return model

    label = f"GPT-{model_match.group(1)}"
    tier = model_match.group(2)
    if tier:
        label = f"{label} {tier.title()}"
    return label


def is_dated_openai_model_snapshot(model: str) -> bool:
    return re.search(r"-\d{4}-\d{2}-\d{2}$", model) is not None


def is_available_openai_chat_model(model: str) -> bool:
    normalized = model.lower()
    if provider_model_unavailable("openai", normalized):
        return False
    if normalized in openai_chat_models:
        return True
    if is_dated_openai_model_snapshot(normalized):
        return False
    model_match = re.fullmatch(
        r"gpt-(5(?:\.\d+)?)(?:-(mini|nano|pro|chat-latest|sol|terra|luna))?",
        normalized,
    )
    if not model_match:
        return False

    tier = model_match.group(2)
    if model_match.group(1) == "5.6":
        return tier in {None, "sol", "terra", "luna"}
    return tier not in {"sol", "terra", "luna"}


def openai_chat_model_sort_key(model: str) -> tuple[int, int, int, int, str]:
    if model in openai_chat_models:
        return (0, openai_chat_models.index(model), 0, 0, model)

    model_match = re.fullmatch(
        r"gpt-(5)(?:\.(\d+))?(?:-(mini|nano|chat-latest|pro|sol|terra|luna))?",
        model,
    )
    if not model_match:
        return (2, 0, 0, 0, model)

    minor = int(model_match.group(2) or 0)
    tier_order = {
        "": 0,
        "sol": 1,
        "terra": 2,
        "luna": 3,
        "mini": 4,
        "nano": 5,
        "chat-latest": 6,
        "pro": 7,
    }
    tier = model_match.group(3) or ""
    return (1, -minor, tier_order.get(tier, 9), 0, model)


def filter_openai_text_models(models: list[str]) -> list[str]:
    filtered = [
        model.lower()
        for model in models
        if is_available_openai_chat_model(model.lower())
    ]
    return sorted(dict.fromkeys(filtered), key=openai_chat_model_sort_key)


def openai_chat_models_for_display(
    available_models: list[str],
    *,
    include_available: bool = False,
    current_model: str | None = None,
) -> list[ChatModels]:
    models = list(openai_chat_models)

    if include_available:
        for model in filter_openai_text_models(available_models):
            if model not in models:
                models.append(model)

    if current_model and current_model not in models:
        models.append(current_model)

    return models


# TTS Models

OpenAITTSModels = Literal[
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-tts-2025-03-20",
    "gpt-4o-mini-tts-2025-12-15",
    "gpt-realtime-2.1-mini",
]
OPENAI_LEGACY_TTS_VOICES = (
    "alloy",
    "ash",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
)
OPENAI_SPEECH_VOICES = (*OPENAI_LEGACY_TTS_VOICES, "ballad", "verse", "marin", "cedar")
OPENAI_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)


def openai_tts_voices(model: str) -> tuple[str, ...]:
    if model == "gpt-realtime-2.1-mini":
        return OPENAI_REALTIME_VOICES
    if model in {"tts-1", "tts-1-hd"}:
        return OPENAI_LEGACY_TTS_VOICES
    return OPENAI_SPEECH_VOICES


ElevenTTSModels = Literal["eleven_multilingual_v2"]
FishTTSModels = Literal["s2.1-pro-free", "s2.1-pro", "s2-pro", "s1"]
GoogleModels = Literal[
    "standard",
    "wavenet",
    "neural",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]
GEMINI_METADATA_TTS_MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")
TTSDeliveryMode = Literal["natural", "custom", "automatic"]
AzureModels = Literal["standard", "neural"]
TTSModels = (
    OpenAITTSModels | ElevenTTSModels | FishTTSModels | GoogleModels | AzureModels
)

# TTS Voices

# Legacy voices for tts-1/tts-1-hd: alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
# All voices for gpt-4o-mini-tts: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar
OpenAIVoices = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]

ElevenVoices = Literal["male-1", "male-2", "female-1", "female-2"]


class TTSVoiceTarget(TypedDict):
    provider: str
    model: str
    voice: str
    language: str | None
    enabled: bool


SmartFieldType = Literal["chat", "tts", "image"]

# Image Models

ReplicateImageModels = Literal["flux-dev", "flux-schnell"]
GoogleImageModels = Literal["gemini-3-pro-image", "gemini-3-pro-image-preview"]
OpenAIImageModels = Literal[
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
    "dall-e-3",
]
ImageModels = ReplicateImageModels | GoogleImageModels | OpenAIImageModels

ImageProviders = Literal["replicate", "google", "openai"]
ImageAspectRatio = Literal["1:1", "16:9", "4:3", "3:4", "9:16"]
ImageResolution = Literal["1024x1024", "2048x2048", "4096x4096"]  # Simplified
ImageOutputFormat = Literal["webp", "png", "jpeg", "avif"]
ImageGenerationQuality = Literal["auto", "low", "medium", "high", "xhigh", "max"]


def image_generation_qualities(model: str) -> list[ImageGenerationQuality]:
    if model.startswith("gpt-image-2.5-"):
        return ["auto", "low", "medium", "high", "xhigh", "max"]
    if model.startswith("gpt-image-"):
        return ["auto", "low", "medium", "high"]
    return ["auto"]


def openai_image_size(
    model: str,
    aspect_ratio: ImageAspectRatio | None,
    resolution: ImageResolution | None,
) -> str:
    if model == "gpt-image-2" or model.startswith("gpt-image-2.5-"):
        return openai_image_2_size(aspect_ratio, resolution)

    if "gpt-image" in model:
        if aspect_ratio == "16:9" or aspect_ratio == "4:3":
            return "1536x1024"
        if aspect_ratio == "9:16" or aspect_ratio == "3:4":
            return "1024x1536"
        return "1024x1024"

    if aspect_ratio == "16:9":
        return "1792x1024"
    if aspect_ratio == "9:16":
        return "1024x1792"
    return "1024x1024"


def openai_image_2_size(
    aspect_ratio: ImageAspectRatio | None,
    resolution: ImageResolution | None,
) -> str:
    if resolution == "4096x4096":
        if aspect_ratio == "16:9":
            return "3840x2160"
        if aspect_ratio == "4:3":
            return "3264x2448"
        if aspect_ratio == "9:16":
            return "2160x3840"
        if aspect_ratio == "3:4":
            return "2448x3264"
        return "2880x2880"

    if resolution == "2048x2048":
        if aspect_ratio == "16:9":
            return "2048x1152"
        if aspect_ratio == "4:3":
            return "2048x1536"
        if aspect_ratio == "9:16":
            return "1152x2048"
        if aspect_ratio == "3:4":
            return "1536x2048"
        return "2048x2048"

    if aspect_ratio == "16:9":
        return "1536x864"
    if aspect_ratio == "4:3":
        return "1024x768"
    if aspect_ratio == "9:16":
        return "864x1536"
    if aspect_ratio == "3:4":
        return "768x1024"
    return "1024x1024"


class FieldExtras(TypedDict):
    automatic: bool
    type: SmartFieldType
    use_custom_model: bool

    # Chat
    chat_model: ChatModels | None
    chat_provider: str | None
    chat_temperature: float | None
    chat_reasoning_effort: OpenAIReasoningEffort | None
    chat_markdown_to_html: bool | None
    chat_use_tools: bool | None

    # TTS
    tts_voice_pool: list[TTSVoiceTarget] | None
    tts_language: str | None
    tts_strip_html: bool | None
    tts_style: str | None
    tts_delivery: NotRequired[TTSDeliveryMode | None]
    # Legacy single-voice settings retained for import/migration compatibility.
    tts_provider: TTSProviders | None
    tts_model: TTSModels | None
    tts_voice: str | None

    # Images
    image_provider: ImageProviders | None
    image_model: ImageModels | None
    image_aspect_ratio: ImageAspectRatio | None
    image_resolution: ImageResolution | None
    image_output_format: ImageOutputFormat | None
    image_quality: int | None  # 0-100, -1 for default/lossless if format supports it
    image_generation_quality: ImageGenerationQuality | None
    regenerate_when_batching: bool


# Any non-mandatory fields should default to none, and will be displayed from global config instead
DEFAULT_EXTRAS: FieldExtras = {
    "automatic": True,
    "type": "chat",
    "use_custom_model": False,
    # Overridable Chat Options
    "chat_markdown_to_html": None,
    "chat_model": None,
    "chat_provider": None,
    "chat_temperature": None,
    "chat_reasoning_effort": None,
    "chat_use_tools": None,
    # TTS Options
    "tts_voice_pool": None,
    "tts_language": None,
    "tts_strip_html": None,
    "tts_style": None,
    "tts_delivery": None,
    # Legacy single-voice settings.
    "tts_model": None,
    "tts_provider": None,
    "tts_voice": None,
    # Overridable Image Options
    "image_provider": None,
    "image_model": None,
    "image_aspect_ratio": None,
    "image_resolution": None,
    "image_output_format": None,
    "image_quality": None,
    "image_generation_quality": None,
    "regenerate_when_batching": False,
}


def tts_delivery_mode(extras: FieldExtras) -> TTSDeliveryMode:
    mode = extras.get("tts_delivery")
    if mode is None:
        return "custom" if extras.get("tts_style") else "natural"
    if mode not in ("natural", "custom", "automatic"):
        raise ValueError(
            "Unknown speech delivery mode. Choose Natural, Custom, or Automatic."
        )
    return mode


class NoteTypeMap(TypedDict):
    fields: dict[str, str]
    extras: dict[str, FieldExtras]


class PromptMap(TypedDict):
    note_types: dict[str, dict[str, NoteTypeMap]]


# Overridable Options

OverridableChatOptions = Literal[
    "chat_provider",
    "chat_model",
    "chat_temperature",
    "chat_reasoning_effort",
    "chat_markdown_to_html",
]

overridable_chat_options: list[OverridableChatOptions] = [
    "chat_provider",
    "chat_model",
    "chat_temperature",
    "chat_reasoning_effort",
    "chat_markdown_to_html",
]


class OverridableChatOptionsDict(TypedDict):
    chat_provider: str | None
    chat_model: ChatModels | None
    chat_temperature: float | None
    chat_reasoning_effort: OpenAIReasoningEffort | None
    chat_markdown_to_html: bool | None


McpServerTransport = Literal["stdio", "streamable_http"]


class McpKeyValuePair(TypedDict):
    key: str
    value: str


class McpServerConfig(TypedDict):
    id: str
    name: str
    enabled: bool
    transport: McpServerTransport
    command: str
    args: list[str]
    env: list[McpKeyValuePair]
    env_passthrough: list[str]
    cwd: str
    url: str
    headers: list[McpKeyValuePair]
    header_env_vars: list[McpKeyValuePair]


BuiltInToolId = Literal["anki_search_notes", "anki_get_deck_overview"]


class BuiltInToolsConfig(TypedDict):
    anki_search_notes: bool
    anki_get_deck_overview: bool


DEFAULT_BUILT_IN_TOOLS: BuiltInToolsConfig = {
    "anki_search_notes": True,
    "anki_get_deck_overview": True,
}


def normalize_built_in_tools_config(
    built_in_tools: dict[str, Any] | BuiltInToolsConfig | None,
) -> BuiltInToolsConfig:
    normalized = dict(DEFAULT_BUILT_IN_TOOLS)
    if not built_in_tools:
        return cast("BuiltInToolsConfig", normalized)

    for key in DEFAULT_BUILT_IN_TOOLS:
        value = built_in_tools.get(key)
        if isinstance(value, bool):
            normalized[key] = value

    return cast("BuiltInToolsConfig", normalized)


def make_tts_voice_target(
    provider: object,
    model: object,
    voice: object,
    *,
    language: str | None = None,
    enabled: bool = True,
) -> TTSVoiceTarget | None:
    if not all(
        isinstance(value, str) and value.strip() for value in (provider, model, voice)
    ):
        return None
    return {
        "provider": cast("str", provider).strip(),
        "model": cast("str", model).strip(),
        "voice": cast("str", voice).strip(),
        "language": language.strip()
        if isinstance(language, str) and language.strip()
        else None,
        "enabled": enabled,
    }


def normalize_tts_voice_pool(value: object) -> list[TTSVoiceTarget]:
    if not isinstance(value, list):
        return []

    normalized: list[TTSVoiceTarget] = []
    for raw_target in value:
        if not isinstance(raw_target, dict):
            continue
        target = make_tts_voice_target(
            raw_target.get("provider"),
            raw_target.get("model"),
            raw_target.get("voice"),
            language=raw_target.get("language")
            if isinstance(raw_target.get("language"), str)
            else None,
            enabled=raw_target.get("enabled", True)
            if isinstance(raw_target.get("enabled", True), bool)
            else True,
        )
        if target:
            normalized.append(target)
    return normalized


def normalize_field_extras(
    extras: dict[str, Any] | FieldExtras | None,
) -> FieldExtras:
    normalized = cast("FieldExtras", dict(DEFAULT_EXTRAS))
    if not extras:
        return normalized

    raw_extras = dict(extras)
    legacy_use_tools = raw_extras.get("chat_use_mcp")
    if "chat_use_tools" not in raw_extras and isinstance(legacy_use_tools, bool):
        raw_extras["chat_use_tools"] = legacy_use_tools

    for key in DEFAULT_EXTRAS:
        if key in raw_extras:
            normalized[key] = raw_extras[key]

    if raw_extras.get("tts_voice_pool") is not None:
        normalized["tts_voice_pool"] = normalize_tts_voice_pool(
            raw_extras["tts_voice_pool"]
        )
    elif (
        raw_extras.get("use_custom_model")
        and raw_extras.get("tts_provider")
        and raw_extras.get("tts_model")
        and raw_extras.get("tts_voice")
    ):
        legacy_target = make_tts_voice_target(
            raw_extras["tts_provider"],
            raw_extras["tts_model"],
            raw_extras["tts_voice"],
        )
        normalized["tts_voice_pool"] = [legacy_target] if legacy_target else None

    return normalized


OverridableTTSOptions = Literal["tts_voice_pool", "tts_strip_html"]

overridable_tts_options: list[OverridableTTSOptions] = [
    "tts_voice_pool",
    "tts_strip_html",
]


class CustomProvider(TypedDict):
    name: str
    base_url: str
    api_key: str
    models: list[str]
    capabilities: list[str]  # "chat", "tts", "image"
    chat_api_mode: Literal["auto", "responses", "chat_completions"]
    streaming_mode: Literal["auto", "enabled", "disabled"]

    # Granular capability lists
    chat_models: list[str] | None
    tts_models: list[str] | None
    image_models: list[str] | None


class ProviderSettings(TypedDict):
    model: str
    temperature: float
    reasoning_effort: OpenAIReasoningEffort | None


class OverrideableTTSOptionsDict(TypedDict):
    tts_voice_pool: list[TTSVoiceTarget] | None
    tts_strip_html: bool | None


OverridableImageOptions = Literal[
    "image_provider",
    "image_model",
    "image_aspect_ratio",
    "image_resolution",
    "image_output_format",
    "image_quality",
    "image_generation_quality",
]

overridable_image_options: list[OverridableImageOptions] = [
    "image_model",
    "image_provider",
    "image_aspect_ratio",
    "image_resolution",
    "image_output_format",
    "image_quality",
    "image_generation_quality",
]


class OverridableImageOptionsDict(TypedDict):
    image_model: ImageModels | None
    image_provider: ImageProviders | None
    image_aspect_ratio: ImageAspectRatio | None
    image_resolution: ImageResolution | None
    image_output_format: ImageOutputFormat | None
    image_quality: int | None
    image_generation_quality: ImageGenerationQuality | None
