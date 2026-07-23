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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Union

from anki.decks import DeckId

from .built_in_tools import BuiltInToolContext
from .chat_provider import (
    ChatProvider,
    TextToolDefinition,
    chat_provider,
    prompt_cache_key_for_request,
)
from .chat_usage import (
    OpenAITokenBudgetExceededError,
    build_prompt_usage_key,
    build_prompt_usage_signature,
    chat_usage_tracker,
)
from .config import config, key_or_config_val
from .constants import API_KEY_MISSING_MESSAGE, GLOBAL_DECK_ID
from .image_provider import ImageProvider, image_provider
from .logger import logger
from .markdown import convert_markdown_to_html
from .media_utils import build_media_filename, convert_image_data
from .models import (
    DEFAULT_EXTRAS,
    ChatModels,
    ChatProviders,
    ElevenVoices,
    ImageAspectRatio,
    ImageModels,
    ImageOutputFormat,
    ImageProviders,
    ImageResolution,
    OpenAIReasoningEffort,
    OpenAIVoices,
    SmartFieldType,
    TTSModels,
    TTSProviders,
    normalize_built_in_tools_config,
)
from .nodes import FieldNode
from .note_proccessor import PendingMedia, ResolvedField
from .prompts import get_extras, interpolate_prompt_with_values
from .tool_registry import ToolRegistry
from .tts_provider import TTSProvider, tts_provider
from .ui.ui_utils import show_message_box
from .utils import run_on_main


@dataclass(frozen=True)
class PromptUsageContext:
    prompt_key: str
    signature: str


class FieldProcessor:
    def __init__(
        self,
        chat_provider: ChatProvider,
        tts_provider: TTSProvider,
        image_provider: ImageProvider,
    ):
        self.chat_provider = chat_provider
        self.tts_provider = tts_provider
        self.image_provider = image_provider

    async def resolve(
        self,
        node: FieldNode,
        *,
        note_id: int,
        note_type: str,
        deck_name: str | None,
        field_order: Sequence[str],
        values: Mapping[str, str],
        show_error_box: bool = False,
        usage_scope_id: Optional[str] = None,
    ) -> ResolvedField:
        input_text = node.input
        field_type: SmartFieldType = node.field_type
        extras = (
            get_extras(
                note_type=note_type,
                field=node.field,
                deck_id=node.deck_id,
                fallback_to_global_deck=True,
            )
            or DEFAULT_EXTRAS
        )

        if field_type == "tts":
            should_strip_html: bool = key_or_config_val(extras, "tts_strip_html")
            tts_provider: TTSProviders = key_or_config_val(extras, "tts_provider")
            tts_model: TTSModels = key_or_config_val(extras, "tts_model")
            voice: Union[OpenAIVoices, ElevenVoices] = key_or_config_val(
                extras, "tts_voice"
            )
            style: Optional[str] = key_or_config_val(extras, "tts_style")

            if style and tts_provider == "google" and "gemini" in tts_model:
                input_text = f"{style} {input_text}"

            instructions = (
                style
                if style and tts_provider == "openai" and tts_model == "gpt-4o-mini-tts"
                else None
            )
            data = await self.get_tts_response(
                note_id=note_id,
                values=values,
                input_text=input_text,
                model=tts_model,
                voice=voice,
                provider=tts_provider,
                strip_html=should_strip_html,
                show_error_box=show_error_box,
                instructions=instructions,
            )
            if not data:
                return ResolvedField(None)

            extension = (
                "wav" if tts_provider == "google" and "gemini" in tts_model else "mp3"
            )
            filename = build_media_filename(
                note_type,
                note_id,
                node.field,
                extension,
            )
            return ResolvedField(
                f"[sound:{filename}]",
                PendingMedia(filename, data),
            )

        if field_type == "chat":
            chat_model: ChatModels = key_or_config_val(extras, "chat_model")
            chat_provider: ChatProviders = key_or_config_val(extras, "chat_provider")
            temperature: float = key_or_config_val(extras, "chat_temperature")
            reasoning_effort: Optional[OpenAIReasoningEffort] = key_or_config_val(
                extras, "chat_reasoning_effort"
            )
            should_convert: bool = key_or_config_val(extras, "chat_markdown_to_html")
            use_tools: bool = key_or_config_val(extras, "chat_use_tools")
            prompt_usage_context = PromptUsageContext(
                prompt_key=build_prompt_usage_key(
                    note_type=note_type,
                    deck_id=int(node.deck_id),
                    field_lower=node.field,
                ),
                signature=build_prompt_usage_signature(
                    prompt=input_text,
                    provider=str(chat_provider),
                    model=str(chat_model),
                    reasoning_effort=reasoning_effort,
                    use_tools=use_tools,
                ),
            )
            value = await self.get_chat_response(
                note_id=note_id,
                note_type=note_type,
                deck_name=deck_name,
                field_order=field_order,
                values=values,
                deck_id=node.deck_id,
                prompt=input_text,
                model=chat_model,
                provider=chat_provider,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                field_lower=node.field,
                should_convert_to_html=should_convert,
                use_tools=use_tools,
                show_error_box=show_error_box,
                prompt_usage_context=prompt_usage_context,
                usage_scope_id=usage_scope_id,
            )
            return ResolvedField(value)

        if field_type == "image":
            image_model: ImageModels = key_or_config_val(extras, "image_model")
            image_provider: ImageProviders = key_or_config_val(extras, "image_provider")
            aspect_ratio: Optional[ImageAspectRatio] = key_or_config_val(
                extras, "image_aspect_ratio"
            )
            resolution: Optional[ImageResolution] = key_or_config_val(
                extras, "image_resolution"
            )
            output_format: ImageOutputFormat = (
                key_or_config_val(extras, "image_output_format") or "webp"
            )
            quality: int = key_or_config_val(extras, "image_quality") or -1
            data = await self.get_image_response(
                note_id=note_id,
                values=values,
                input_text=input_text,
                model=image_model,
                provider=image_provider,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                quality=quality,
                show_error_box=show_error_box,
            )
            if not data:
                return ResolvedField(None)

            extension = output_format if output_format != "jpeg" else "jpg"
            filename = build_media_filename(
                note_type,
                note_id,
                node.field,
                extension,
            )
            return ResolvedField(
                f'<img src="{filename}"/>',
                PendingMedia(filename, data),
            )

        raise Exception(f"Unexpected field type {field_type}")

    async def get_chat_response(
        self,
        *,
        note_id: int,
        note_type: str,
        deck_name: str | None,
        field_order: Sequence[str],
        values: Mapping[str, str],
        deck_id: DeckId,
        prompt: str,
        model: ChatModels,
        provider: ChatProviders,
        field_lower: str,
        temperature: float,
        should_convert_to_html: bool,
        reasoning_effort: Optional[OpenAIReasoningEffort] = None,
        use_tools: bool = False,
        show_error_box: bool = True,
        prompt_usage_context: Optional[PromptUsageContext] = None,
        usage_scope_id: Optional[str] = None,
    ) -> Optional[str]:
        interpolated_prompt = interpolate_prompt_with_values(
            prompt,
            values,
            config.allow_empty_fields,
        )
        if not interpolated_prompt or not self._check_api_key(provider, show_error_box):
            return None

        tools: Optional[list[TextToolDefinition]] = None
        tool_executor = None
        if use_tools:
            registry = ToolRegistry(
                context=BuiltInToolContext(
                    note_id=note_id,
                    deck_id=deck_id,
                    deck_name=None if deck_id == GLOBAL_DECK_ID else deck_name,
                    note_type=note_type,
                    note_type_fields=tuple(field_order),
                    field_name=field_lower,
                ),
                built_in_tools=normalize_built_in_tools_config(config.built_in_tools),
                mcp_servers=config.mcp_servers or [],
            )
            tools, warnings = await registry.build_tool_registry()
            for warning in warnings:
                logger.warning("Tools warning: %s", warning)
            if not tools:
                raise Exception(
                    "Tools are enabled for this field, but no built-in tools or external MCP tools are available."
                )
            tool_executor = registry.execute_tool_call

        cache_seed = f"{provider}:{model}:{note_type}:{deck_id}:{field_lower}:{prompt}"
        prompt_chars = len(interpolated_prompt)
        try:
            provider_result = await self.chat_provider.async_get_chat_response_result(
                interpolated_prompt,
                model=model,
                provider=provider,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                prompt_cache_key=prompt_cache_key_for_request(str(model), cache_seed),
                note_id=note_id,
                tools=tools,
                tool_executor=tool_executor,
                prompt_usage_key=prompt_usage_context.prompt_key
                if prompt_usage_context
                else None,
                prompt_usage_signature=prompt_usage_context.signature
                if prompt_usage_context
                else None,
            )
        except OpenAITokenBudgetExceededError as error:
            if show_error_box:
                run_on_main(lambda msg=str(error): show_message_box(msg))
                return None
            raise

        response_text = provider_result.text
        if response_text and should_convert_to_html:
            response_text = convert_markdown_to_html(response_text)

        chat_usage_tracker.finalize_request(
            provider=str(provider),
            model=str(model),
            reasoning_effort=reasoning_effort,
            use_tools=use_tools,
            prompt_chars=prompt_chars,
            raw_usage=provider_result.usage,
            response_chars=len(response_text),
            scope_id=usage_scope_id,
            prompt_key=prompt_usage_context.prompt_key
            if prompt_usage_context
            else None,
            prompt_signature=(
                prompt_usage_context.signature if prompt_usage_context else None
            ),
            record_openai_daily_usage=False,
        )
        return response_text

    async def get_tts_response(
        self,
        *,
        note_id: int,
        values: Mapping[str, str],
        input_text: str,
        model: TTSModels,
        provider: TTSProviders,
        voice: str,
        strip_html: bool,
        show_error_box: bool = True,
        instructions: Optional[str] = None,
    ) -> Optional[bytes]:
        interpolated_prompt = interpolate_prompt_with_values(
            input_text,
            values,
            config.allow_empty_fields,
        )
        if not interpolated_prompt or not self._check_api_key(provider, show_error_box):
            return None

        return await self.tts_provider.async_get_tts_response(
            input=interpolated_prompt,
            model=model,
            provider=provider,
            voice=voice,
            note_id=note_id,
            strip_html=strip_html,
            instructions=instructions,
        )

    async def get_image_response(
        self,
        *,
        note_id: int,
        values: Mapping[str, str],
        input_text: str,
        model: ImageModels,
        provider: ImageProviders,
        aspect_ratio: Optional[ImageAspectRatio] = None,
        resolution: Optional[ImageResolution] = None,
        output_format: Optional[str] = None,
        quality: Optional[int] = None,
        show_error_box: bool = True,
    ) -> Optional[bytes]:
        interpolated_prompt = interpolate_prompt_with_values(
            input_text,
            values,
            config.allow_empty_fields,
        )
        if not interpolated_prompt or not self._check_api_key(provider, show_error_box):
            return None

        raw_bytes = await self.image_provider.async_get_image_response(
            prompt=interpolated_prompt,
            model=model,
            provider=provider,
            note_id=note_id,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
        )

        if not raw_bytes:
            return None

        # Ensure format and quality
        try:
            return convert_image_data(
                raw_bytes,
                format=output_format or "webp",
                quality=quality if quality is not None else -1,
            )
        except Exception as e:
            logger.error(f"Failed to convert image: {e}")
            if show_error_box:
                error_msg = f"Failed to convert image: {e}"
                run_on_main(lambda: show_message_box(error_msg))
            return None

    def _check_api_key(self, provider: str, show_error_box: bool) -> bool:
        # First check if this matches a custom provider
        # Custom providers manage their own keys
        if config.custom_providers:
            for p in config.custom_providers:
                if p["name"] == provider:
                    return True

        has_key = True
        if provider == "openai":
            has_key = bool(config.openai_api_key)
        elif provider == "anthropic":
            has_key = bool(config.anthropic_api_key)
        elif provider == "deepseek":
            has_key = bool(config.deepseek_api_key)
        elif provider == "google":
            has_key = bool(config.google_api_key)
        elif provider == "elevenLabs":
            has_key = bool(config.elevenlabs_api_key)
        elif provider == "replicate":
            has_key = bool(config.replicate_api_key)

        if not has_key:
            logger.error(f"Missing API key for {provider}")
            if show_error_box:
                run_on_main(
                    lambda: show_message_box(API_KEY_MISSING_MESSAGE.format(provider))
                )
            return False

        return True


field_processor = FieldProcessor(
    chat_provider=chat_provider,
    tts_provider=tts_provider,
    image_provider=image_provider,
)
