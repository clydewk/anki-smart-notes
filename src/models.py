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

from typing import Literal, Optional, TypedDict, Union

# Providers

TTSProviders = Literal["openai", "elevenLabs", "google", "azure"]
ChatProviders = Literal["openai", "anthropic", "deepseek", "google"]

# Chat Models

OpenAIModels = Literal[
    "gpt-5.2",
    "gpt-5.2-chat-latest",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-4o-mini",
]
DeepseekModels = Literal["deepseek-v3"]
AnthropicModels = Literal[
    "claude-3-5-haiku-latest", "claude-sonnet-4-0", "claude-opus-4-1"
]
GoogleChatModels = Literal["gemini-3-pro-preview", "gemini-3-flash-preview"]

ChatModels = Union[OpenAIModels, AnthropicModels, DeepseekModels, GoogleChatModels]

# Order that the models are displayed in the UI
openai_chat_models: list[ChatModels] = [
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5.2-chat-latest",
    "gpt-5",
    "gpt-5.2",
]

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
    "gpt-5.2-chat-latest",
    "gpt-5.2",
    "gpt-5",
    "gpt-5-nano",
    "gpt-5-mini",
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

# Reasoning Efforts
# "none" is a UI concept meaning "don't use reasoning, use temperature instead"
# gpt-5: added "minimal"
# gpt-5.1: dropped "minimal", added "none"
# gpt-5.2: added "xhigh"
OpenAIReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
OPENAI_DEFAULT_REASONING_EFFORTS: tuple[OpenAIReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
)
OPENAI_REASONING_EFFORTS_BY_MODEL: dict[str, tuple[OpenAIReasoningEffort, ...]] = {
    "gpt-5": ("minimal", "low", "medium", "high"),
    "gpt-5.1": ("none", "low", "medium", "high"),
    "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),
}


def openai_reasoning_efforts_for_model(model: str) -> list[OpenAIReasoningEffort]:
    key = model.lower()
    return list(
        OPENAI_REASONING_EFFORTS_BY_MODEL.get(key, OPENAI_DEFAULT_REASONING_EFFORTS)
    )


# TTS Models

OpenAITTSModels = Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
ElevenTTSModels = Literal["eleven_multilingual_v2"]
GoogleModels = Literal[
    "standard",
    "wavenet",
    "neural",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]
AzureModels = Literal["standard", "neural"]
TTSModels = Union[OpenAITTSModels, ElevenTTSModels, GoogleModels, AzureModels]

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

SmartFieldType = Literal["chat", "tts", "image"]

# Image Models

ReplicateImageModels = Literal["flux-dev", "flux-schnell"]
GoogleImageModels = Literal["gemini-3-pro-image-preview"]
OpenAIImageModels = Literal[
    "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini", "dall-e-3"
]
ImageModels = Union[ReplicateImageModels, GoogleImageModels, OpenAIImageModels]

ImageProviders = Literal["replicate", "google", "openai"]
ImageAspectRatio = Literal["1:1", "16:9", "4:3", "3:4", "9:16"]
ImageResolution = Literal["1024x1024", "2048x2048", "4096x4096"]  # Simplified
ImageOutputFormat = Literal["webp", "png", "jpeg", "avif"]


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

    # TTS
    tts_provider: Optional[TTSProviders]
    tts_model: Optional[TTSModels]
    tts_voice: Optional[str]
    tts_strip_html: Optional[bool]
    tts_style: Optional[str]

    # Images
    image_provider: Optional[ImageProviders]
    image_model: Optional[ImageModels]
    image_aspect_ratio: Optional[ImageAspectRatio]
    image_resolution: Optional[ImageResolution]
    image_output_format: Optional[ImageOutputFormat]
    image_quality: Optional[int]  # 0-100, -1 for default/lossless if format supports it
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
    # Overridable TTS Options
    "tts_model": None,
    "tts_provider": None,
    "tts_voice": None,
    "tts_strip_html": None,
    "tts_style": None,
    # Overridable Image Options
    "image_provider": None,
    "image_model": None,
    "image_aspect_ratio": None,
    "image_resolution": None,
    "image_output_format": None,
    "image_quality": None,
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


OverridableTTSOptions = Union[
    Literal["tts_model"],
    Literal["tts_provider"],
    Literal["tts_voice"],
    Literal["tts_strip_html"],
    Literal["tts_style"],
]

overridable_tts_options: list[OverridableTTSOptions] = [
    "tts_model",
    "tts_provider",
    "tts_voice",
    "tts_strip_html",
    "tts_style",
]


class CustomProvider(TypedDict):
    name: str
    base_url: str
    api_key: str
    models: list[str]
    capabilities: list[str]  # "chat", "tts", "image"

    # Granular capability lists
    chat_models: Optional[list[str]]
    tts_models: Optional[list[str]]
    image_models: Optional[list[str]]


class ProviderSettings(TypedDict):
    model: str
    temperature: float
    reasoning_effort: Optional[OpenAIReasoningEffort]


class OverrideableTTSOptionsDict(TypedDict):
    tts_model: Optional[TTSModels]
    tts_provider: Optional[TTSProviders]
    tts_voice: Optional[str]
    tts_strip_html: Optional[bool]
    tts_style: Optional[str]


OverridableImageOptions = Union[
    Literal["image_provider"],
    Literal["image_model"],
    Literal["image_aspect_ratio"],
    Literal["image_resolution"],
    Literal["image_output_format"],
    Literal["image_quality"],
]

overridable_image_options: list[OverridableImageOptions] = [
    "image_model",
    "image_provider",
    "image_aspect_ratio",
    "image_resolution",
    "image_output_format",
    "image_quality",
]


class OverridableImageOptionsDict(TypedDict):
    image_model: Optional[ImageModels]
    image_provider: Optional[ImageProviders]
    image_aspect_ratio: Optional[ImageAspectRatio]
    image_resolution: Optional[ImageResolution]
    image_output_format: Optional[ImageOutputFormat]
    image_quality: Optional[int]
