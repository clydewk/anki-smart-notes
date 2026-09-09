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
from typing import Any, Literal, Optional, TypedDict, Union, cast

# Providers

TTSProviders = Literal["openai", "elevenLabs", "google", "azure", "fish"]
ChatProviders = Literal["openai", "anthropic", "deepseek", "google"]

# Reasoning Efforts
# "none" is a UI concept meaning "don't use reasoning, use temperature instead"
OpenAIReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
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
class OpenAIChatModelSpec:
    model: str
    label: str
    reasoning_efforts: tuple[OpenAIReasoningEffort, ...]


OPENAI_CHAT_MODEL_CATALOG: tuple[OpenAIChatModelSpec, ...] = (
    OpenAIChatModelSpec(
        "gpt-6-astra", "GPT-6 Astra", ("low", "medium", "high", "xhigh", "max")
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
    "claude-opus-4-1",
    "claude-sonnet-4-0",
    "claude-3-5-haiku-latest",
]

deepseek_chat_models: list[ChatModels] = ["deepseek-v3"]

google_chat_models: list[ChatModels] = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

provider_model_map: dict[ChatProviders, list[ChatModels]] = {
    "openai": openai_chat_models,
    "anthropic": anthropic_chat_models,
    "deepseek": deepseek_chat_models,
    "google": google_chat_models,
}


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
    current_model: Optional[str] = None,
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

OpenAITTSModels = Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
ElevenTTSModels = Literal["eleven_multilingual_v2"]
FishTTSModels = Literal["s2.1-pro-free", "s2.1-pro", "s2-pro", "s1"]
GoogleModels = Literal[
    "standard",
    "wavenet",
    "neural",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]
AzureModels = Literal["standard", "neural"]
TTSModels = Union[
    OpenAITTSModels, ElevenTTSModels, FishTTSModels, GoogleModels, AzureModels
]

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
    language: Optional[str]
    enabled: bool


SmartFieldType = Literal["chat", "tts", "image"]

# Image Models

ReplicateImageModels = Literal["flux-dev", "flux-schnell"]
GoogleImageModels = Literal["gemini-3-pro-image-preview"]
OpenAIImageModels = Literal[
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
    "dall-e-3",
]
ImageModels = Union[ReplicateImageModels, GoogleImageModels, OpenAIImageModels]

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


class FieldExtras(TypedDict):
    automatic: bool
    type: SmartFieldType
    use_custom_model: bool

    # Chat
    chat_model: Optional[ChatModels]
    chat_provider: Optional[ChatProviders]
    chat_temperature: Optional[int]
    chat_reasoning_effort: Optional[OpenAIReasoningEffort]
    chat_markdown_to_html: Optional[bool]
    chat_use_tools: Optional[bool]

    # TTS
    tts_voice_pool: Optional[list[TTSVoiceTarget]]
    tts_language: Optional[str]
    tts_strip_html: Optional[bool]
    tts_style: Optional[str]
    # Legacy single-voice settings retained for import/migration compatibility.
    tts_provider: Optional[TTSProviders]
    tts_model: Optional[TTSModels]
    tts_voice: Optional[str]

    # Images
    image_provider: Optional[ImageProviders]
    image_model: Optional[ImageModels]
    image_aspect_ratio: Optional[ImageAspectRatio]
    image_resolution: Optional[ImageResolution]
    image_output_format: Optional[ImageOutputFormat]
    image_quality: Optional[int]  # 0-100, -1 for default/lossless if format supports it
    image_generation_quality: Optional[ImageGenerationQuality]
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


class NoteTypeMap(TypedDict):
    fields: dict[str, str]
    extras: dict[str, FieldExtras]


class PromptMap(TypedDict):
    note_types: dict[str, dict[str, NoteTypeMap]]


# Overridable Options

OverridableChatOptions = Union[
    Literal["chat_provider"],
    Literal["chat_model"],
    Literal["chat_temperature"],
    Literal["chat_reasoning_effort"],
    Literal["chat_markdown_to_html"],
]

overridable_chat_options: list[OverridableChatOptions] = [
    "chat_provider",
    "chat_model",
    "chat_temperature",
    "chat_reasoning_effort",
    "chat_markdown_to_html",
]


class OverridableChatOptionsDict(TypedDict):
    chat_provider: Optional[ChatProviders]
    chat_model: Optional[ChatModels]
    chat_temperature: Optional[int]
    chat_reasoning_effort: Optional[OpenAIReasoningEffort]
    chat_markdown_to_html: Optional[bool]


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
    built_in_tools: Optional[Union[dict[str, Any], BuiltInToolsConfig]],
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
    language: Optional[str] = None,
    enabled: bool = True,
) -> Optional[TTSVoiceTarget]:
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
    extras: Optional[Union[dict[str, Any], FieldExtras]],
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


OverridableTTSOptions = Union[
    Literal["tts_voice_pool"],
    Literal["tts_strip_html"],
]

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
    chat_models: Optional[list[str]]
    tts_models: Optional[list[str]]
    image_models: Optional[list[str]]


class ProviderSettings(TypedDict):
    model: str
    temperature: float
    reasoning_effort: Optional[OpenAIReasoningEffort]


class OverrideableTTSOptionsDict(TypedDict):
    tts_voice_pool: Optional[list[TTSVoiceTarget]]
    tts_strip_html: Optional[bool]


OverridableImageOptions = Union[
    Literal["image_provider"],
    Literal["image_model"],
    Literal["image_aspect_ratio"],
    Literal["image_resolution"],
    Literal["image_output_format"],
    Literal["image_quality"],
    Literal["image_generation_quality"],
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
    image_model: Optional[ImageModels]
    image_provider: Optional[ImageProviders]
    image_aspect_ratio: Optional[ImageAspectRatio]
    image_resolution: Optional[ImageResolution]
    image_output_format: Optional[ImageOutputFormat]
    image_quality: Optional[int]
    image_generation_quality: Optional[ImageGenerationQuality]
