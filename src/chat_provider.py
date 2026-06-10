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

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from .chat_usage import OpenAITurnPermit, chat_usage_tracker
from .config import config
from .constants import DEFAULT_TEMPERATURE
from .logger import logger
from .models import (
    ChatModels,
    ChatProviders,
    CustomProvider,
    OpenAIReasoningEffort,
    filter_openai_text_models,
    openai_reasoning_efforts_for_model,
)
from .provider_runtime import (
    ProviderHTTPError,
    ProviderTimeoutError,
    RequestTimeouts,
    ResponseFormatError,
    StreamingNotSupportedError,
    format_provider_http_error_for_log,
    provider_runtime,
)

OPENAI_BASE_URL = "https://api.openai.com"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
GOOGLE_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TextEventType = Literal[
    "response_started",
    "text_delta",
    "usage_reported",
    "response_completed",
]
ResolvedApiMode = Literal["responses", "chat_completions"]
ConversationRole = Literal["user", "assistant", "tool"]

MAX_TOOL_TURNS = 6
REASONING_TRACE_MAX_CHARS = 4000


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class TextGenerationRequest:
    prompt: str
    model: str
    provider: str
    temperature: float
    reasoning_effort: OpenAIReasoningEffort | None
    prompt_cache_key: str | None = None
    tools: list[TextToolDefinition] | None = None
    prompt_usage_key: str | None = None
    prompt_usage_signature: str | None = None


@dataclass(frozen=True)
class TextToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class TextToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TextConversationMessage:
    role: ConversationRole
    content: str = ""
    tool_calls: tuple[TextToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class TextGenerationEvent:
    type: TextEventType
    text: str = ""
    response_id: str | None = None
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    response_id: str | None
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class ProviderTurnResult:
    text: str
    tool_calls: tuple[TextToolCall, ...]
    usage: dict[str, Any] | None
    response_id: str | None = None


@dataclass
class ResponsesFunctionCallBuffer:
    output_index: int
    item_id: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments_text: str = ""

    def update_from_item(self, item: Any) -> None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return

        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            self.item_id = item_id

        call_id = item.get("call_id") or item.get("id")
        if isinstance(call_id, str) and call_id:
            self.call_id = call_id

        name = item.get("name")
        if isinstance(name, str) and name:
            self.name = name

        arguments = item.get("arguments")
        if isinstance(arguments, str) and arguments and not self.arguments_text:
            self.arguments_text = arguments

    def append_arguments(self, delta: Any) -> None:
        if isinstance(delta, str) and delta:
            self.arguments_text += delta

    def update_from_done_event(self, event: dict[str, Any]) -> None:
        self.update_from_item(event.get("item"))

        call_id = event.get("call_id")
        if isinstance(call_id, str) and call_id:
            self.call_id = call_id

        name = event.get("name")
        if isinstance(name, str) and name:
            self.name = name

        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            self.item_id = item_id

        arguments = event.get("arguments")
        if isinstance(arguments, str):
            self.arguments_text = arguments

    def is_ready(self) -> bool:
        return bool(self.name and self.call_id)

    def to_tool_call(self) -> TextToolCall:
        if not self.name:
            raise ResponseFormatError("Responses stream omitted the tool name.")
        if not self.call_id:
            raise ResponseFormatError("Responses stream omitted the tool call ID.")

        return TextToolCall(
            id=self.call_id,
            name=self.name,
            arguments=parse_tool_arguments(self.arguments_text),
        )


@dataclass(frozen=True)
class CustomProviderVerification:
    models: list[str]
    chat_api_mode: ResolvedApiMode


def normalize_api_base_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    suffixes = (
        "/v1/responses",
        "/responses",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/models",
        "/models",
        "/v1/audio/speech",
        "/v1/images/generations",
    )

    for suffix in suffixes:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
            break

    return clean.rstrip("/")


def build_v1_base_url(base_url: str) -> str:
    normalized = normalize_api_base_url(base_url)
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def build_v1_endpoint(base_url: str, path: str) -> str:
    return f"{build_v1_base_url(base_url)}{path}"


def text_initial_window(provider: str) -> int:
    lower_provider = provider.lower()
    if lower_provider in {"anthropic", "deepseek", "google"}:
        return 2
    return 4


def text_transport_key(provider: str, model: str) -> str:
    return f"text:{provider}:{model}"


def payload_length_metrics(payload: str | list[dict[str, Any]]) -> tuple[int, int]:
    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
        return len(payload), len(encoded)

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = serialized.encode("utf-8")
    return len(serialized), len(encoded)


def is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(("o", "gpt-5"))


def responses_timeouts(
    reasoning_effort: OpenAIReasoningEffort | None,
) -> RequestTimeouts:
    first_event_timeout = responses_liveness_timeout_sec(reasoning_effort)
    return RequestTimeouts(
        connect_timeout_sec=10.0,
        sock_read_timeout_sec=None,
        first_event_timeout_sec=first_event_timeout,
        stream_idle_timeout_sec=first_event_timeout,
    )


def responses_liveness_timeout_sec(
    reasoning_effort: OpenAIReasoningEffort | None,
) -> float:
    return 90.0 if reasoning_effort in {"high", "xhigh"} else 45.0


def responses_json_timeouts(
    reasoning_effort: OpenAIReasoningEffort | None,
) -> RequestTimeouts:
    return RequestTimeouts(
        connect_timeout_sec=10.0,
        sock_read_timeout_sec=responses_liveness_timeout_sec(reasoning_effort),
    )


def json_timeouts(sock_read_timeout_sec: float = 30.0) -> RequestTimeouts:
    return RequestTimeouts(
        connect_timeout_sec=10.0, sock_read_timeout_sec=sock_read_timeout_sec
    )


def looks_like_reasoning_schema_error(body: str) -> bool:
    text = body.lower()
    return "reasoning" in text and (
        "unknown" in text or "invalid" in text or "unsupported" in text
    )


def looks_like_unsupported_api(error: ProviderHTTPError) -> bool:
    if error.status in {404, 405, 415}:
        return True

    text = error.body.lower()

    if error.status == 500 and (
        "not implemented" in text or "convert_request_failed" in text
    ):
        return True

    if error.status not in {400, 422}:
        return False

    return (
        "responses" in text
        or "chat/completions" in text
        or "unknown field" in text
        or "unexpected field" in text
        or "unsupported" in text
        or "schema" in text
    )


def looks_like_streaming_unsupported_http_error(error: ProviderHTTPError) -> bool:
    text = error.body.lower()
    mentions_streaming = any(
        marker in text for marker in ("stream", "streaming", "event-stream", "sse")
    )
    if not mentions_streaming:
        return False

    return error.status in {400, 404, 405, 415, 422, 500} and (
        "unsupported" in text
        or "not supported" in text
        or "not implemented" in text
        or "disabled" in text
        or "invalid" in text
        or "unknown field" in text
        or "unexpected field" in text
    )


def custom_provider_api_unsupported_message(
    provider_name: str,
    *,
    api_mode: ResolvedApiMode,
    model: str,
    uses_tools: bool,
) -> str:
    if api_mode == "responses":
        if uses_tools:
            return (
                f"Custom provider {provider_name} does not support the Responses API "
                f"for model {model}, so tool calling cannot be used. Re-open the "
                "provider settings and switch Chat API to Chat Completions, or choose "
                "a provider/model that supports Responses."
            )
        return (
            f"Custom provider {provider_name} does not support the Responses API for "
            f"model {model}. Re-open the provider settings and switch Chat API to "
            "Chat Completions, or choose a provider/model that supports Responses."
        )

    if uses_tools:
        return (
            f"Custom provider {provider_name} does not support Chat Completions tool "
            f"calling for model {model}. Disable tools for this field or "
            "choose a provider/model that supports tool calls."
        )

    return (
        f"Custom provider {provider_name} does not support the Chat Completions API "
        f"for model {model}. Re-open the provider settings and choose Responses, or "
        "choose a provider/model that supports Chat Completions."
    )


def prompt_cache_key_for_request(model: str, prompt: str) -> str:
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return f"smart-notes:{model}:{prompt_hash}"


def choose_reasoning_effort(
    model: str, reasoning_effort: OpenAIReasoningEffort | None
) -> OpenAIReasoningEffort | None:
    if not is_reasoning_model(model):
        return None

    efforts = openai_reasoning_efforts_for_model(model)
    if reasoning_effort in efforts:
        return reasoning_effort
    if "none" in efforts:
        return "none"
    return efforts[0] if efforts else None


def merge_usage_dicts(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left

    merged = dict(left)
    for key, value in right.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            nested = merge_usage_dicts(existing, value)
            if nested is not None:
                merged[key] = nested
        elif isinstance(existing, (int, float)) and isinstance(value, (int, float)):
            merged[key] = existing + value
        else:
            merged[key] = value
    return merged


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return cast("dict[str, Any]", raw_arguments)

    if isinstance(raw_arguments, str) and raw_arguments:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ResponseFormatError(
                "Tool call arguments were not valid JSON."
            ) from exc
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)

    return {}


class ChatProvider:
    def __init__(self) -> None:
        self._openai_models_cache: list[str] = []

    async def async_get_chat_response(
        self,
        prompt: str,
        model: ChatModels,
        provider: ChatProviders,
        note_id: int,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: OpenAIReasoningEffort | None = None,
        prompt_cache_key: str | None = None,
        retry_count: int = 0,
        tools: list[TextToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        prompt_usage_key: str | None = None,
        prompt_usage_signature: str | None = None,
    ) -> str:
        result = await self.async_get_chat_response_result(
            prompt=prompt,
            model=model,
            provider=provider,
            note_id=note_id,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=prompt_cache_key,
            retry_count=retry_count,
            tools=tools,
            tool_executor=tool_executor,
            prompt_usage_key=prompt_usage_key,
            prompt_usage_signature=prompt_usage_signature,
        )
        return result.text

    async def async_get_chat_response_result(
        self,
        prompt: str,
        model: ChatModels,
        provider: ChatProviders,
        note_id: int,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: OpenAIReasoningEffort | None = None,
        prompt_cache_key: str | None = None,
        retry_count: int = 0,
        tools: list[TextToolDefinition] | None = None,
        tool_executor: ToolExecutor | None = None,
        prompt_usage_key: str | None = None,
        prompt_usage_signature: str | None = None,
    ) -> TextGenerationResult:
        del note_id, retry_count
        request = TextGenerationRequest(
            prompt=prompt,
            model=str(model),
            provider=str(provider),
            temperature=temperature,
            reasoning_effort=choose_reasoning_effort(str(model), reasoning_effort),
            prompt_cache_key=(
                prompt_cache_key if provider == "openai" and not tools else None
            ),
            tools=tools,
            prompt_usage_key=prompt_usage_key,
            prompt_usage_signature=prompt_usage_signature,
        )
        return await self.generate_text(request, tool_executor=tool_executor)

    async def generate_text(
        self,
        request: TextGenerationRequest,
        tool_executor: ToolExecutor | None = None,
    ) -> TextGenerationResult:
        if request.tools:
            if tool_executor is None:
                raise Exception("Tool-enabled requests require a tool executor.")
            return await self._generate_text_with_tools(request, tool_executor)

        text_parts: list[str] = []
        response_id: str | None = None
        usage: dict[str, Any] | None = None

        async for event in self.async_iter_text_events(request):
            if event.type == "text_delta":
                text_parts.append(event.text)
            elif event.type == "usage_reported":
                usage = event.usage
            elif event.type == "response_started":
                response_id = event.response_id
            elif event.type == "response_completed":
                response_id = event.response_id or response_id
                usage = event.usage or usage

        return TextGenerationResult(
            text="".join(text_parts),
            response_id=response_id,
            usage=usage,
        )

    async def async_iter_text_events(
        self, request: TextGenerationRequest
    ) -> AsyncIterator[TextGenerationEvent]:
        custom_provider = next(
            (
                p
                for p in (config.custom_providers or [])
                if p["name"] == request.provider
            ),
            None,
        )

        if custom_provider is not None:
            async for event in self._iterate_custom_provider_events(
                request, custom_provider
            ):
                yield event
            return

        if request.provider == "openai":
            async for event in self._iterate_official_openai_events(request):
                yield event
            return

        if request.provider == "anthropic":
            async for event in self._iterate_anthropic_events(request):
                yield event
            return

        if request.provider == "deepseek":
            async for event in self._iterate_deepseek_events(request):
                yield event
            return

        if request.provider == "google":
            async for event in self._iterate_google_events(request):
                yield event
            return

        raise ValueError(f"Unknown provider: {request.provider}")

    async def verify_custom_provider(
        self, provider_config: CustomProvider
    ) -> CustomProviderVerification:
        models = await self._fetch_custom_provider_models(provider_config)
        configured_mode = provider_config.get("chat_api_mode", "auto")
        chat_api_mode: ResolvedApiMode
        if configured_mode == "chat_completions":
            chat_api_mode = "chat_completions"
        elif configured_mode == "responses":
            chat_api_mode = "responses"
        else:
            probe_model = self._pick_probe_model(models)
            chat_api_mode = await self._probe_custom_provider_api_mode(
                provider_config, probe_model
            )

        return CustomProviderVerification(models=models, chat_api_mode=chat_api_mode)

    async def fetch_openai_chat_models(self) -> list[str]:
        api_key = config.openai_api_key
        if not api_key:
            return list(self._openai_models_cache)

        base_url = config.openai_endpoint or OPENAI_BASE_URL
        try:
            payload = await provider_runtime.request_json(
                key="models:openai",
                initial_window=1,
                method="GET",
                url=build_v1_endpoint(base_url, "/models"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                provider="openai",
                model="models",
                timeouts=json_timeouts(sock_read_timeout_sec=10.0),
                max_retries=1,
            )
            models = filter_openai_text_models(self._extract_model_ids(payload))
        except Exception as exc:
            logger.warning(f"Failed to auto-detect OpenAI models: {exc}")
            return list(self._openai_models_cache)

        if models:
            self._openai_models_cache = models
        return list(self._openai_models_cache)

    def get_cached_openai_chat_models(self) -> list[str]:
        return list(self._openai_models_cache)

    async def acquire_openai_turn_permit(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        input_payload: str | list[dict[str, Any]],
        prompt_usage_key: str | None,
        prompt_usage_signature: str | None,
    ) -> OpenAITurnPermit | None:
        if provider_name != "openai":
            return None

        transport_key = text_transport_key(provider_name, request.model)
        prompt_chars, prompt_bytes = payload_length_metrics(input_payload)
        transport_window = provider_runtime.get_concurrency_window(
            key=transport_key,
            initial_window=text_initial_window(provider_name),
        )

        return await chat_usage_tracker.acquire_openai_turn_permit(
            provider=provider_name,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            use_tools=bool(request.tools),
            prompt_chars=prompt_chars,
            prompt_bytes=prompt_bytes,
            transport_key=transport_key,
            transport_window=transport_window,
            budget_enabled=bool(config.openai_daily_token_budget_enabled),
            budget_limit=int(config.openai_daily_token_budget or 1_000_000),
            prompt_key=prompt_usage_key,
            prompt_signature=prompt_usage_signature,
        )

    async def _generate_text_with_tools(
        self, request: TextGenerationRequest, tool_executor: ToolExecutor
    ) -> TextGenerationResult:
        if self._uses_responses_tool_loop(request):
            return await self._generate_text_with_tools_responses(
                request, tool_executor
            )
        return await self._generate_text_with_tools_messages(request, tool_executor)

    async def _generate_text_with_tools_responses(
        self, request: TextGenerationRequest, tool_executor: ToolExecutor
    ) -> TextGenerationResult:
        custom_provider = next(
            (
                provider
                for provider in (config.custom_providers or [])
                if provider["name"] == request.provider
            ),
            None,
        )

        if custom_provider is not None:
            provider_name = custom_provider["name"]
            base_url = normalize_api_base_url(custom_provider["base_url"])
            api_key = custom_provider["api_key"]
        else:
            provider_name = "openai"
            base_url = config.openai_endpoint or OPENAI_BASE_URL
            api_key = config.openai_api_key or ""
            if not api_key:
                raise Exception(
                    "OpenAI API key not found. Please set it in the settings."
                )

        stream_enabled = (
            custom_provider is None
            or self._custom_provider_streaming_enabled(custom_provider)
        )
        response_id: str | None = None
        usage: dict[str, Any] | None = None
        next_input: str | list[dict[str, Any]] = request.prompt

        for turn in range(MAX_TOOL_TURNS):
            turn_result, stream_enabled = await self._responses_tool_turn(
                request=request,
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                custom_provider=custom_provider,
                stream_enabled=stream_enabled,
                input_payload=next_input,
                previous_response_id=response_id,
                include_prompt_cache_key=turn == 0 and provider_name == "openai",
            )

            response_id = turn_result.response_id or response_id
            usage = merge_usage_dicts(usage, turn_result.usage)

            if not turn_result.tool_calls:
                return TextGenerationResult(
                    text=turn_result.text,
                    response_id=response_id,
                    usage=usage,
                )

            next_input = await self._execute_tool_calls_for_responses(
                turn_result.tool_calls, tool_executor
            )

            if turn == MAX_TOOL_TURNS - 1:
                final_result, stream_enabled = await self._responses_tool_turn(
                    request=request,
                    provider_name=provider_name,
                    base_url=base_url,
                    api_key=api_key,
                    custom_provider=custom_provider,
                    stream_enabled=stream_enabled,
                    input_payload=next_input,
                    previous_response_id=response_id,
                    include_prompt_cache_key=False,
                    allow_tools=False,
                )
                response_id = final_result.response_id or response_id
                usage = merge_usage_dicts(usage, final_result.usage)
                if final_result.tool_calls:
                    raise Exception("Model exceeded the maximum number of tool rounds.")
                return TextGenerationResult(
                    text=final_result.text,
                    response_id=response_id,
                    usage=usage,
                )

        raise Exception("Model exceeded the maximum number of tool rounds.")

    async def _responses_tool_turn(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        base_url: str,
        api_key: str,
        custom_provider: CustomProvider | None,
        stream_enabled: bool,
        input_payload: str | list[dict[str, Any]],
        previous_response_id: str | None,
        include_prompt_cache_key: bool,
        allow_tools: bool = True,
    ) -> tuple[ProviderTurnResult, bool]:
        if stream_enabled:
            try:
                return (
                    await self._responses_stream_turn(
                        request=request,
                        provider_name=provider_name,
                        base_url=base_url,
                        api_key=api_key,
                        input_payload=input_payload,
                        previous_response_id=previous_response_id,
                        include_prompt_cache_key=include_prompt_cache_key,
                        allow_tools=allow_tools,
                    ),
                    True,
                )
            except StreamingNotSupportedError as exc:
                if custom_provider is None:
                    raise
                detail = " ".join(str(exc).split())[:200]
                logger.info(
                    "Custom provider %s does not support streamed Responses tool calls; retrying without streaming. detail=%s",
                    provider_name,
                    detail,
                )
                stream_enabled = False
            except ResponseFormatError as exc:
                if custom_provider is None:
                    raise
                detail = " ".join(str(exc).split())[:200]
                logger.info(
                    "Custom provider %s emitted an incompatible Responses tool stream; retrying without streaming. detail=%s",
                    provider_name,
                    detail,
                )
                stream_enabled = False
            except ProviderHTTPError as exc:
                if (
                    custom_provider is not None
                    and looks_like_streaming_unsupported_http_error(exc)
                ):
                    logger.info(
                        "Custom provider %s rejected streamed Responses tool calls; retrying without streaming. %s",
                        provider_name,
                        format_provider_http_error_for_log(exc),
                    )
                    stream_enabled = False
                elif custom_provider is not None and looks_like_unsupported_api(exc):
                    raise Exception(
                        custom_provider_api_unsupported_message(
                            provider_name,
                            api_mode="responses",
                            model=request.model,
                            uses_tools=True,
                        )
                    ) from exc
                else:
                    raise

        try:
            return (
                await self._responses_turn(
                    request=request,
                    provider_name=provider_name,
                    base_url=base_url,
                    api_key=api_key,
                    input_payload=input_payload,
                    previous_response_id=previous_response_id,
                    include_prompt_cache_key=include_prompt_cache_key,
                    allow_tools=allow_tools,
                ),
                False,
            )
        except ProviderHTTPError as exc:
            if custom_provider is not None and looks_like_unsupported_api(exc):
                raise Exception(
                    custom_provider_api_unsupported_message(
                        provider_name,
                        api_mode="responses",
                        model=request.model,
                        uses_tools=True,
                    )
                ) from exc
            raise

    async def _generate_text_with_tools_messages(
        self, request: TextGenerationRequest, tool_executor: ToolExecutor
    ) -> TextGenerationResult:
        messages = [TextConversationMessage(role="user", content=request.prompt)]
        response_id: str | None = None
        usage: dict[str, Any] | None = None
        seen_calls: set[tuple[str, str]] = set()

        for _ in range(MAX_TOOL_TURNS):
            turn_result = await self._message_turn(request, messages)
            response_id = turn_result.response_id or response_id
            usage = merge_usage_dicts(usage, turn_result.usage)
            messages.append(
                TextConversationMessage(
                    role="assistant",
                    content=turn_result.text,
                    tool_calls=turn_result.tool_calls,
                )
            )

            if not turn_result.tool_calls:
                return TextGenerationResult(
                    text=turn_result.text,
                    response_id=response_id,
                    usage=usage,
                )

            tool_messages = await self._execute_tool_calls_for_messages(
                turn_result.tool_calls, tool_executor
            )
            for tool_message in tool_messages:
                call_signature = (
                    tool_message.tool_call_id or "",
                    tool_message.content,
                )
                if call_signature in seen_calls:
                    raise Exception("Model repeated the same tool result indefinitely.")
                seen_calls.add(call_signature)
                messages.append(tool_message)

        raise Exception("Model exceeded the maximum number of tool rounds.")

    async def _run_tool_call(
        self,
        tool_call: TextToolCall,
        tool_executor: ToolExecutor,
    ) -> tuple[str, bool]:
        try:
            tool_output = await tool_executor(tool_call.name, tool_call.arguments)
        except Exception as exc:
            return f"Tool error: {exc}", True

        return tool_output, tool_output.startswith("Tool error:")

    async def _execute_tool_calls_for_responses(
        self, tool_calls: tuple[TextToolCall, ...], tool_executor: ToolExecutor
    ) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for tool_call in tool_calls:
            tool_output, _ = await self._run_tool_call(tool_call, tool_executor)
            results.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.id,
                    "output": tool_output,
                }
            )
        return results

    async def _execute_tool_calls_for_messages(
        self, tool_calls: tuple[TextToolCall, ...], tool_executor: ToolExecutor
    ) -> list[TextConversationMessage]:
        results: list[TextConversationMessage] = []
        for tool_call in tool_calls:
            tool_output, is_error = await self._run_tool_call(tool_call, tool_executor)
            results.append(
                TextConversationMessage(
                    role="tool",
                    content=tool_output,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    is_error=is_error,
                )
            )
        return results

    def _uses_responses_tool_loop(self, request: TextGenerationRequest) -> bool:
        custom_provider = next(
            (
                provider
                for provider in (config.custom_providers or [])
                if provider["name"] == request.provider
            ),
            None,
        )
        if custom_provider is None:
            return request.provider == "openai"
        return self._resolve_custom_provider_api_mode(custom_provider) == "responses"

    async def _responses_turn(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        base_url: str,
        api_key: str,
        input_payload: str | list[dict[str, Any]],
        previous_response_id: str | None,
        include_prompt_cache_key: bool,
        allow_tools: bool = True,
    ) -> ProviderTurnResult:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        prompt_usage_key = (
            request.prompt_usage_key if previous_response_id is None else None
        )
        prompt_usage_signature = (
            request.prompt_usage_signature if previous_response_id is None else None
        )
        permit = await self.acquire_openai_turn_permit(
            request=request,
            provider_name=provider_name,
            input_payload=input_payload,
            prompt_usage_key=prompt_usage_key,
            prompt_usage_signature=prompt_usage_signature,
        )

        try:
            response = await provider_runtime.request_json(
                key=text_transport_key(provider_name, request.model),
                initial_window=text_initial_window(provider_name),
                method="POST",
                url=build_v1_endpoint(base_url, "/responses"),
                headers=headers,
                json_payload=self._responses_payload(
                    request=request,
                    input_payload=input_payload,
                    previous_response_id=previous_response_id,
                    include_prompt_cache_key=include_prompt_cache_key,
                    allow_tools=allow_tools,
                ),
                provider=provider_name,
                model=request.model,
                timeouts=responses_json_timeouts(request.reasoning_effort),
            )
        except Exception:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise

        self._log_reasoning_trace(
            provider=provider_name,
            model=request.model,
            source="responses",
            payload=response,
        )

        try:
            turn_result = self._parse_openai_responses_turn(response)
        except Exception:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise

        await chat_usage_tracker.finalize_openai_turn(
            provider=provider_name,
            raw_usage=turn_result.usage,
            permit=permit,
        )
        return turn_result

    async def _responses_stream_turn(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        base_url: str,
        api_key: str,
        input_payload: str | list[dict[str, Any]],
        previous_response_id: str | None,
        include_prompt_cache_key: bool,
        allow_tools: bool = True,
    ) -> ProviderTurnResult:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        prompt_usage_key = (
            request.prompt_usage_key if previous_response_id is None else None
        )
        prompt_usage_signature = (
            request.prompt_usage_signature if previous_response_id is None else None
        )
        url = build_v1_endpoint(base_url, "/responses")
        payload = self._responses_payload(
            request=request,
            input_payload=input_payload,
            previous_response_id=previous_response_id,
            include_prompt_cache_key=include_prompt_cache_key,
            allow_tools=allow_tools,
        )
        payload["stream"] = True
        permit = await self.acquire_openai_turn_permit(
            request=request,
            provider_name=provider_name,
            input_payload=input_payload,
            prompt_usage_key=prompt_usage_key,
            prompt_usage_signature=prompt_usage_signature,
        )

        response_id: str | None = None
        usage: dict[str, Any] | None = None
        text_parts: list[str] = []
        response: dict[str, Any] | None = None
        pending_calls: dict[int, ResponsesFunctionCallBuffer] = {}
        completed_calls: dict[int, TextToolCall] = {}
        completed = False

        def function_call_buffer(output_index: int) -> ResponsesFunctionCallBuffer:
            existing = pending_calls.get(output_index)
            if existing is not None:
                return existing

            created = ResponsesFunctionCallBuffer(output_index=output_index)
            pending_calls[output_index] = created
            return created

        try:
            async for event in provider_runtime.stream_sse_json(
                key=text_transport_key(provider_name, request.model),
                initial_window=text_initial_window(provider_name),
                method="POST",
                url=url,
                headers=headers,
                json_payload=payload,
                provider=provider_name,
                model=request.model,
                timeouts=responses_timeouts(request.reasoning_effort),
            ):
                self._log_reasoning_trace(
                    provider=provider_name,
                    model=request.model,
                    source="responses_stream",
                    payload=event,
                )
                event_type = str(event.get("type", ""))
                if event_type == "done":
                    continue

                response_data = event.get("response")
                if isinstance(response_data, dict) and isinstance(
                    response_data.get("id"), str
                ):
                    response_id = response_data["id"]
                else:
                    stream_response_id = event.get("response_id")
                    if isinstance(stream_response_id, str) and stream_response_id:
                        response_id = stream_response_id

                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                    continue

                if event_type == "response.output_item.added":
                    output_index = event.get("output_index")
                    if isinstance(output_index, int):
                        item = event.get("item")
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                        ):
                            function_call_buffer(output_index).update_from_item(item)
                    continue

                if event_type == "response.function_call_arguments.delta":
                    output_index = event.get("output_index")
                    if isinstance(output_index, int):
                        call_buffer = function_call_buffer(output_index)
                        item_id = event.get("item_id")
                        if isinstance(item_id, str) and item_id:
                            call_buffer.item_id = item_id
                        call_buffer.append_arguments(event.get("delta"))
                    continue

                if event_type == "response.function_call_arguments.done":
                    output_index = event.get("output_index")
                    if not isinstance(output_index, int):
                        raise ResponseFormatError(
                            "Responses stream omitted the tool call output index."
                        )
                    call_buffer = function_call_buffer(output_index)
                    call_buffer.update_from_done_event(event)
                    if call_buffer.is_ready():
                        completed_calls[output_index] = call_buffer.to_tool_call()
                    continue

                if event_type == "response.completed":
                    response_payload = event.get("response")
                    if not isinstance(response_payload, dict):
                        raise ResponseFormatError(
                            "Responses stream completed without a response payload."
                        )
                    completed = True
                    response = cast("dict[str, Any]", response_payload)
                    response_id = cast("str | None", response.get("id", response_id))
                    usage = self._extract_usage(response)
                    if not text_parts:
                        final_text = self._extract_openai_responses_text(response)
                        if final_text:
                            text_parts.append(final_text)
                    continue

                if event_type in {"response.failed", "error", "response.incomplete"}:
                    raise Exception(self._describe_event_error(event))
        except Exception:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise

        if not completed:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise ResponseFormatError(
                "Responses stream ended before the response completed."
            )

        try:
            tool_calls = tuple(
                tool_call for _, tool_call in sorted(completed_calls.items())
            )

            if response is not None:
                parsed_tool_calls = self._extract_openai_responses_tool_calls(response)
                if parsed_tool_calls:
                    tool_calls = parsed_tool_calls

            if not tool_calls and pending_calls:
                ready_pending_calls = [
                    pending_calls[index]
                    for index in sorted(pending_calls)
                    if pending_calls[index].is_ready()
                ]
                if ready_pending_calls:
                    tool_calls = tuple(
                        pending_call.to_tool_call()
                        for pending_call in ready_pending_calls
                    )

            if not tool_calls and pending_calls:
                incomplete_pending_calls = [
                    pending_calls[index]
                    for index in sorted(pending_calls)
                    if pending_calls[index].arguments_text
                ]
                if incomplete_pending_calls:
                    first_incomplete = incomplete_pending_calls[0]
                    if not first_incomplete.name:
                        raise ResponseFormatError(
                            "Responses stream omitted the tool name."
                        )
                    if not first_incomplete.call_id:
                        raise ResponseFormatError(
                            "Responses stream omitted the tool call ID."
                        )

            result = ProviderTurnResult(
                text="".join(text_parts),
                tool_calls=tool_calls,
                usage=usage,
                response_id=response_id,
            )
        except Exception:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise

        await chat_usage_tracker.finalize_openai_turn(
            provider=provider_name,
            raw_usage=result.usage,
            permit=permit,
        )
        return result

    async def _message_turn(
        self, request: TextGenerationRequest, messages: list[TextConversationMessage]
    ) -> ProviderTurnResult:
        custom_provider = next(
            (
                provider
                for provider in (config.custom_providers or [])
                if provider["name"] == request.provider
            ),
            None,
        )

        if custom_provider is not None:
            api_mode = self._resolve_custom_provider_api_mode(custom_provider)
            if api_mode != "chat_completions":
                raise Exception(
                    custom_provider_api_unsupported_message(
                        request.provider,
                        api_mode=api_mode,
                        model=request.model,
                        uses_tools=True,
                    )
                )
            try:
                return await self._openai_compatible_chat_turn(
                    request=request,
                    provider_name=custom_provider["name"],
                    base_url=normalize_api_base_url(custom_provider["base_url"]),
                    api_key=custom_provider["api_key"],
                    messages=messages,
                )
            except ProviderHTTPError as exc:
                if looks_like_unsupported_api(exc):
                    raise Exception(
                        custom_provider_api_unsupported_message(
                            request.provider,
                            api_mode="chat_completions",
                            model=request.model,
                            uses_tools=True,
                        )
                    ) from exc
                raise

        if request.provider == "deepseek":
            return await self._deepseek_turn(request, messages)
        if request.provider == "anthropic":
            return await self._anthropic_turn(request, messages)
        if request.provider == "google":
            return await self._google_turn(request, messages)

        raise ValueError(f"Unknown provider: {request.provider}")

    async def _iterate_official_openai_events(
        self, request: TextGenerationRequest
    ) -> AsyncIterator[TextGenerationEvent]:
        api_key = config.openai_api_key
        if not api_key:
            raise Exception("OpenAI API key not found. Please set it in the settings.")

        base_url = config.openai_endpoint or OPENAI_BASE_URL
        async for event in self._iterate_openai_compatible_responses_events(
            request,
            provider_name="openai",
            base_url=base_url,
            api_key=api_key,
            use_stream=True,
            include_prompt_cache_key=True,
        ):
            yield event

    async def _iterate_openai_compatible_responses_events(
        self,
        request: TextGenerationRequest,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        use_stream: bool,
        include_prompt_cache_key: bool = False,
    ) -> AsyncIterator[TextGenerationEvent]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = build_v1_endpoint(base_url, "/responses")
        payload = self._responses_payload(
            request=request,
            input_payload=request.prompt,
            previous_response_id=None,
            include_prompt_cache_key=include_prompt_cache_key,
        )
        if use_stream:
            payload["stream"] = True

        permit = await self.acquire_openai_turn_permit(
            request=request,
            provider_name=provider_name,
            input_payload=request.prompt,
            prompt_usage_key=request.prompt_usage_key,
            prompt_usage_signature=request.prompt_usage_signature,
        )

        if not use_stream:
            try:
                response = await provider_runtime.request_json(
                    key=text_transport_key(provider_name, request.model),
                    initial_window=text_initial_window(provider_name),
                    method="POST",
                    url=url,
                    headers=headers,
                    json_payload=payload,
                    provider=provider_name,
                    model=request.model,
                    timeouts=responses_json_timeouts(request.reasoning_effort),
                )
            except Exception:
                await chat_usage_tracker.release_openai_turn_permit(permit)
                raise
            if not isinstance(response, dict):
                await chat_usage_tracker.release_openai_turn_permit(permit)
                raise ResponseFormatError("Responses API returned an invalid payload.")

            self._log_reasoning_trace(
                provider=provider_name,
                model=request.model,
                source="responses",
                payload=response,
            )

            try:
                response_id = self._response_id(response)
                usage = self._extract_usage(response)
                final_text = self._extract_openai_responses_text(response)
            except Exception:
                await chat_usage_tracker.release_openai_turn_permit(permit)
                raise

            await chat_usage_tracker.finalize_openai_turn(
                provider=provider_name,
                raw_usage=usage,
                permit=permit,
            )

            yield TextGenerationEvent(type="response_started", response_id=response_id)
            if final_text:
                yield TextGenerationEvent(
                    type="text_delta",
                    text=final_text,
                    response_id=response_id,
                )
            if usage:
                yield TextGenerationEvent(
                    type="usage_reported",
                    usage=usage,
                    response_id=response_id,
                )
            yield TextGenerationEvent(
                type="response_completed",
                response_id=response_id,
                usage=usage,
            )
            return

        started = False
        accumulated_text = ""
        last_response_id: str | None = None
        last_usage: dict[str, Any] | None = None

        try:
            async for event in provider_runtime.stream_sse_json(
                key=text_transport_key(provider_name, request.model),
                initial_window=text_initial_window(provider_name),
                method="POST",
                url=url,
                headers=headers,
                json_payload=payload,
                provider=provider_name,
                model=request.model,
                timeouts=responses_timeouts(request.reasoning_effort),
            ):
                self._log_reasoning_trace(
                    provider=provider_name,
                    model=request.model,
                    source="responses_stream",
                    payload=event,
                )
                event_type = str(event.get("type", ""))

                if event_type == "done":
                    continue

                if not started:
                    response_data = event.get("response")
                    response_id = (
                        response_data.get("id")
                        if isinstance(response_data, dict)
                        else event.get("response_id")
                    )
                    last_response_id = cast("str | None", response_id)
                    started = True
                    logger.debug(
                        "Provider %s stream started via responses at %s",
                        provider_name,
                        url,
                    )
                    yield TextGenerationEvent(
                        type="response_started",
                        response_id=last_response_id,
                    )

                if event_type == "response.output_text.delta":
                    delta = str(event.get("delta", ""))
                    if delta:
                        accumulated_text += delta
                        yield TextGenerationEvent(type="text_delta", text=delta)
                    continue

                if event_type == "response.completed":
                    response_data = event.get("response", {})
                    if isinstance(response_data, dict):
                        last_response_id = cast(
                            "str | None", response_data.get("id", last_response_id)
                        )
                        usage = self._extract_usage(response_data)
                        if usage:
                            last_usage = usage
                            yield TextGenerationEvent(
                                type="usage_reported",
                                usage=usage,
                                response_id=last_response_id,
                            )

                        if not accumulated_text:
                            final_text = self._extract_openai_responses_text(
                                response_data
                            )
                            if final_text:
                                accumulated_text = final_text
                                yield TextGenerationEvent(
                                    type="text_delta",
                                    text=final_text,
                                    response_id=last_response_id,
                                )

                    yield TextGenerationEvent(
                        type="response_completed",
                        response_id=last_response_id,
                        usage=last_usage,
                    )
                    continue

                if event_type in {"response.failed", "error", "response.incomplete"}:
                    raise Exception(self._describe_event_error(event))
        except Exception:
            await chat_usage_tracker.release_openai_turn_permit(permit)
            raise

        await chat_usage_tracker.finalize_openai_turn(
            provider=provider_name,
            raw_usage=last_usage,
            permit=permit,
        )

    async def _iterate_custom_provider_events(
        self,
        request: TextGenerationRequest,
        provider_config: CustomProvider,
    ) -> AsyncIterator[TextGenerationEvent]:
        api_mode = self._resolve_custom_provider_api_mode(provider_config)
        use_stream = self._custom_provider_streaming_enabled(provider_config)
        provider_name = provider_config["name"]
        base_url = normalize_api_base_url(provider_config["base_url"])
        api_key = provider_config["api_key"]

        logger.debug(
            "Custom provider %s using %s at %s (streaming=%s, model=%s)",
            provider_name,
            api_mode,
            base_url,
            use_stream,
            request.model,
        )

        stream_candidates = [use_stream, False] if use_stream else [False]
        for stream_enabled in stream_candidates:
            should_retry_without_stream = False
            reasoning_candidates = self._reasoning_candidates(request.reasoning_effort)
            for reasoning_effort in reasoning_candidates:
                adjusted_request = TextGenerationRequest(
                    prompt=request.prompt,
                    model=request.model,
                    provider=provider_name,
                    temperature=request.temperature,
                    reasoning_effort=reasoning_effort,
                    prompt_cache_key=None,
                )
                try:
                    if api_mode == "responses":
                        async for (
                            event
                        ) in self._iterate_openai_compatible_responses_events(
                            adjusted_request,
                            provider_name=provider_name,
                            base_url=base_url,
                            api_key=api_key,
                            use_stream=stream_enabled,
                        ):
                            yield event
                    else:
                        async for event in self._iterate_openai_compatible_chat_events(
                            request=adjusted_request,
                            provider_name=provider_name,
                            base_url=base_url,
                            api_key=api_key,
                            use_stream=stream_enabled,
                        ):
                            yield event
                    return
                except StreamingNotSupportedError as exc:
                    if stream_enabled:
                        detail = " ".join(str(exc).split())[:200]
                        logger.info(
                            "Custom provider %s does not support streaming on %s; retrying without streaming. detail=%s",
                            provider_name,
                            api_mode,
                            detail,
                        )
                        should_retry_without_stream = True
                        break
                    raise Exception(
                        f"Custom provider {provider_name} does not support streaming on {api_mode}. Disable streaming in provider settings."
                    ) from exc
                except ProviderTimeoutError as exc:
                    logger.warning(
                        "Custom provider %s timed out via %s at %s: phase=%s last_event=%s",
                        provider_name,
                        api_mode,
                        exc.url,
                        exc.phase,
                        exc.last_event_type,
                    )
                    raise
                except ProviderHTTPError as exc:
                    if stream_enabled and looks_like_streaming_unsupported_http_error(
                        exc
                    ):
                        logger.info(
                            "Custom provider %s rejected streaming on %s; retrying without streaming. %s",
                            provider_name,
                            api_mode,
                            format_provider_http_error_for_log(exc),
                        )
                        should_retry_without_stream = True
                        break

                    if (
                        reasoning_effort is not None
                        and looks_like_reasoning_schema_error(exc.body)
                    ):
                        logger.info(
                            "Custom provider %s rejected reasoning_effort on %s; retrying without it. %s",
                            provider_name,
                            api_mode,
                            format_provider_http_error_for_log(exc),
                        )
                        continue

                    if provider_config.get(
                        "chat_api_mode", "responses"
                    ) == "auto" and looks_like_unsupported_api(exc):
                        raise Exception(
                            f"Custom provider {provider_name} still uses legacy Auto mode. Re-open the provider settings and choose Responses or Chat Completions explicitly."
                        ) from exc

                    if looks_like_unsupported_api(exc):
                        raise Exception(
                            custom_provider_api_unsupported_message(
                                provider_name,
                                api_mode=api_mode,
                                model=request.model,
                                uses_tools=False,
                            )
                        ) from exc

                    raise

            if should_retry_without_stream:
                continue

    async def _iterate_openai_compatible_chat_events(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        base_url: str,
        api_key: str,
        use_stream: bool,
    ) -> AsyncIterator[TextGenerationEvent]:
        url = build_v1_endpoint(base_url, "/chat/completions")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }

        if request.reasoning_effort and request.reasoning_effort != "none":
            payload["reasoning_effort"] = request.reasoning_effort
        else:
            payload["temperature"] = request.temperature

        if use_stream:
            payload["stream"] = True
            logger.debug(
                "Provider %s stream started via chat_completions at %s",
                provider_name,
                url,
            )
            async for event in provider_runtime.stream_sse_json(
                key=f"text:{provider_name}:{request.model}",
                initial_window=text_initial_window(provider_name),
                method="POST",
                url=url,
                headers=headers,
                json_payload=payload,
                provider=provider_name,
                model=request.model,
                timeouts=responses_timeouts(request.reasoning_effort),
            ):
                self._log_reasoning_trace(
                    provider=provider_name,
                    model=request.model,
                    source="chat_stream",
                    payload=event,
                )
                event_type = str(event.get("type", ""))
                if event_type == "done":
                    continue

                if "choices" not in event:
                    continue

                choices = event.get("choices", [])
                if not isinstance(choices, list) or not choices:
                    continue

                choice = choices[0]
                if not isinstance(choice, dict):
                    continue

                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    continue

                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield TextGenerationEvent(type="text_delta", text=content)
            yield TextGenerationEvent(type="response_completed")
            return

        response = await provider_runtime.request_json(
            key=f"text:{provider_name}:{request.model}",
            initial_window=text_initial_window(provider_name),
            method="POST",
            url=url,
            headers=headers,
            json_payload=payload,
            provider=provider_name,
            model=request.model,
            timeouts=json_timeouts(),
        )
        self._log_reasoning_trace(
            provider=provider_name,
            model=request.model,
            source="chat",
            payload=response,
        )
        text = self._extract_openai_chat_text(response)
        usage = self._extract_usage(response)
        yield TextGenerationEvent(type="response_started")
        yield TextGenerationEvent(type="text_delta", text=text)
        if usage:
            yield TextGenerationEvent(type="usage_reported", usage=usage)
        yield TextGenerationEvent(type="response_completed", usage=usage)

    async def _iterate_anthropic_events(
        self, request: TextGenerationRequest
    ) -> AsyncIterator[TextGenerationEvent]:
        api_key = config.anthropic_api_key
        if not api_key:
            raise Exception(
                "Anthropic API key not found. Please set it in the settings."
            )

        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": 4096,
            "temperature": request.temperature,
        }
        response = await provider_runtime.request_json(
            key=f"text:anthropic:{request.model}",
            initial_window=text_initial_window("anthropic"),
            method="POST",
            url=ANTHROPIC_ENDPOINT,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json_payload=payload,
            provider="anthropic",
            model=request.model,
            timeouts=json_timeouts(),
        )
        text = self._extract_anthropic_text(response)
        usage = self._extract_usage(response)
        yield TextGenerationEvent(type="response_started")
        yield TextGenerationEvent(type="text_delta", text=text)
        if usage:
            yield TextGenerationEvent(type="usage_reported", usage=usage)
        yield TextGenerationEvent(type="response_completed", usage=usage)

    async def _iterate_deepseek_events(
        self, request: TextGenerationRequest
    ) -> AsyncIterator[TextGenerationEvent]:
        api_key = config.deepseek_api_key
        if not api_key:
            raise Exception(
                "DeepSeek API key not found. Please set it in the settings."
            )

        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        response = await provider_runtime.request_json(
            key=f"text:deepseek:{request.model}",
            initial_window=text_initial_window("deepseek"),
            method="POST",
            url=DEEPSEEK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json_payload=payload,
            provider="deepseek",
            model=request.model,
            timeouts=json_timeouts(),
        )
        text = self._extract_openai_chat_text(response)
        usage = self._extract_usage(response)
        yield TextGenerationEvent(type="response_started")
        yield TextGenerationEvent(type="text_delta", text=text)
        if usage:
            yield TextGenerationEvent(type="usage_reported", usage=usage)
        yield TextGenerationEvent(type="response_completed", usage=usage)

    async def _iterate_google_events(
        self, request: TextGenerationRequest
    ) -> AsyncIterator[TextGenerationEvent]:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found. Please set it in the settings.")

        response = await provider_runtime.request_json(
            key=f"text:google:{request.model}",
            initial_window=text_initial_window("google"),
            method="POST",
            url=f"{GOOGLE_ENDPOINT_BASE}/{request.model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json_payload={
                "contents": [{"parts": [{"text": request.prompt}]}],
                "generationConfig": {"temperature": request.temperature},
            },
            provider="google",
            model=request.model,
            timeouts=json_timeouts(),
        )
        error = response.get("error")
        if error:
            raise Exception(f"google API error: {error}")

        text = self._extract_google_text(response)
        usage = self._extract_usage(response)
        yield TextGenerationEvent(type="response_started")
        yield TextGenerationEvent(type="text_delta", text=text)
        if usage:
            yield TextGenerationEvent(type="usage_reported", usage=usage)
        yield TextGenerationEvent(type="response_completed", usage=usage)

    async def _openai_compatible_chat_turn(
        self,
        *,
        request: TextGenerationRequest,
        provider_name: str,
        base_url: str,
        api_key: str,
        messages: list[TextConversationMessage],
    ) -> ProviderTurnResult:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._chat_completion_messages_payload(messages),
            "parallel_tool_calls": False,
        }
        if request.tools:
            payload["tools"] = self._chat_tools_payload(request.tools)
            payload["tool_choice"] = "auto"

        if request.reasoning_effort and request.reasoning_effort != "none":
            payload["reasoning_effort"] = request.reasoning_effort
        else:
            payload["temperature"] = request.temperature

        response = await provider_runtime.request_json(
            key=f"text:{provider_name}:{request.model}",
            initial_window=text_initial_window(provider_name),
            method="POST",
            url=build_v1_endpoint(base_url, "/chat/completions"),
            headers=headers,
            json_payload=payload,
            provider=provider_name,
            model=request.model,
            timeouts=json_timeouts(),
        )
        self._log_reasoning_trace(
            provider=provider_name,
            model=request.model,
            source="chat",
            payload=response,
        )
        return self._parse_openai_chat_turn(response)

    async def _deepseek_turn(
        self, request: TextGenerationRequest, messages: list[TextConversationMessage]
    ) -> ProviderTurnResult:
        api_key = config.deepseek_api_key
        if not api_key:
            raise Exception(
                "DeepSeek API key not found. Please set it in the settings."
            )

        return await self._openai_compatible_chat_turn(
            request=request,
            provider_name="deepseek",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            messages=messages,
        )

    async def _anthropic_turn(
        self, request: TextGenerationRequest, messages: list[TextConversationMessage]
    ) -> ProviderTurnResult:
        api_key = config.anthropic_api_key
        if not api_key:
            raise Exception(
                "Anthropic API key not found. Please set it in the settings."
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._anthropic_messages_payload(messages),
            "max_tokens": 4096,
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = self._anthropic_tools_payload(request.tools)

        response = await provider_runtime.request_json(
            key=f"text:anthropic:{request.model}",
            initial_window=text_initial_window("anthropic"),
            method="POST",
            url=ANTHROPIC_ENDPOINT,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json_payload=payload,
            provider="anthropic",
            model=request.model,
            timeouts=json_timeouts(),
        )
        return self._parse_anthropic_turn(response)

    async def _google_turn(
        self, request: TextGenerationRequest, messages: list[TextConversationMessage]
    ) -> ProviderTurnResult:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found. Please set it in the settings.")

        payload: dict[str, Any] = {
            "contents": self._google_contents_payload(messages),
            "generationConfig": {"temperature": request.temperature},
        }
        if request.tools:
            payload["tools"] = [
                {"functionDeclarations": self._google_tools_payload(request.tools)}
            ]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        response = await provider_runtime.request_json(
            key=f"text:google:{request.model}",
            initial_window=text_initial_window("google"),
            method="POST",
            url=f"{GOOGLE_ENDPOINT_BASE}/{request.model}:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json_payload=payload,
            provider="google",
            model=request.model,
            timeouts=json_timeouts(),
        )
        error = response.get("error")
        if error:
            raise Exception(f"google API error: {error}")
        return self._parse_google_turn(response)

    def _chat_completion_messages_payload(
        self, messages: list[TextConversationMessage]
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
                continue

            item: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(
                                tool_call.arguments,
                                ensure_ascii=True,
                                sort_keys=True,
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            payload.append(item)
        return payload

    def _chat_tools_payload(
        self, tools: list[TextToolDefinition]
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _parse_openai_chat_turn(self, response: Any) -> ProviderTurnResult:
        if not isinstance(response, dict):
            raise ResponseFormatError(
                "OpenAI-compatible response was not a JSON object."
            )

        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise ResponseFormatError("OpenAI-compatible response had no choices.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ResponseFormatError("OpenAI-compatible choice was invalid.")

        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise ResponseFormatError("OpenAI-compatible response had no message.")

        tool_calls: list[TextToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                function = raw_tool_call.get("function", {})
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                call_id = raw_tool_call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = name
                tool_calls.append(
                    TextToolCall(
                        id=call_id,
                        name=name,
                        arguments=parse_tool_arguments(function.get("arguments")),
                    )
                )

        return ProviderTurnResult(
            text=self._extract_openai_chat_text(response),
            tool_calls=tuple(tool_calls),
            usage=self._extract_usage(response),
            response_id=self._response_id(response),
        )

    def _anthropic_messages_payload(
        self, messages: list[TextConversationMessage]
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                payload.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or "",
                                "content": message.content,
                                "is_error": message.is_error,
                            }
                        ],
                    }
                )
                continue

            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for tool_call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.arguments,
                    }
                )
            payload.append({"role": message.role, "content": content})
        return payload

    def _anthropic_tools_payload(
        self, tools: list[TextToolDefinition]
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _parse_anthropic_turn(self, response: Any) -> ProviderTurnResult:
        if not isinstance(response, dict):
            raise ResponseFormatError("Anthropic response was not a JSON object.")

        content = response.get("content", [])
        if not isinstance(content, list):
            raise ResponseFormatError("Anthropic response content was invalid.")

        text_parts: list[str] = []
        tool_calls: list[TextToolCall] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
                continue
            if item_type != "tool_use":
                continue

            tool_name = item.get("name")
            tool_id = item.get("id")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if not isinstance(tool_id, str) or not tool_id:
                tool_id = tool_name
            tool_calls.append(
                TextToolCall(
                    id=tool_id,
                    name=tool_name,
                    arguments=parse_tool_arguments(item.get("input")),
                )
            )

        return ProviderTurnResult(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            usage=self._extract_usage(response),
            response_id=self._response_id(response),
        )

    def _google_contents_payload(
        self, messages: list[TextConversationMessage]
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                payload.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.tool_name or "",
                                    "response": {
                                        "content": message.content,
                                        "is_error": message.is_error,
                                    },
                                }
                            }
                        ],
                    }
                )
                continue

            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for tool_call in message.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "name": tool_call.name,
                            "args": tool_call.arguments,
                        }
                    }
                )
            payload.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": parts,
                }
            )
        return payload

    def _google_tools_payload(
        self, tools: list[TextToolDefinition]
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]

    def _parse_google_turn(self, response: Any) -> ProviderTurnResult:
        if not isinstance(response, dict):
            raise ResponseFormatError("Google response was not a JSON object.")

        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ResponseFormatError("Google response had no candidates.")

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ResponseFormatError("Google candidate payload was invalid.")

        content = candidate.get("content", {})
        if not isinstance(content, dict):
            raise ResponseFormatError("Google candidate content was invalid.")

        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise ResponseFormatError("Google candidate parts were invalid.")

        text_parts: list[str] = []
        tool_calls: list[TextToolCall] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                text_parts.append(str(part["text"]))
                continue

            function_call = part.get("functionCall")
            if not isinstance(function_call, dict):
                continue

            tool_name = function_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            tool_calls.append(
                TextToolCall(
                    id=str(function_call.get("id") or tool_name),
                    name=tool_name,
                    arguments=parse_tool_arguments(function_call.get("args")),
                )
            )

        response_id = response.get("responseId")
        return ProviderTurnResult(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            usage=self._extract_usage(response),
            response_id=response_id if isinstance(response_id, str) else None,
        )

    async def _fetch_custom_provider_models(
        self, provider_config: CustomProvider
    ) -> list[str]:
        url = build_v1_endpoint(provider_config["base_url"], "/models")
        headers = {"Content-Type": "application/json"}
        if provider_config["api_key"]:
            headers["Authorization"] = f"Bearer {provider_config['api_key']}"
        payload = await provider_runtime.request_json(
            key=f"models:{provider_config['name']}",
            initial_window=1,
            method="GET",
            url=url,
            headers=headers,
            provider=provider_config["name"],
            model="models",
            timeouts=json_timeouts(sock_read_timeout_sec=10.0),
            max_retries=1,
        )
        return self._extract_model_ids(payload)

    async def _probe_custom_provider_api_mode(
        self, provider_config: CustomProvider, model: str | None
    ) -> ResolvedApiMode:
        if not model:
            return "responses"

        probe_request = TextGenerationRequest(
            prompt="Reply with OK.",
            model=model,
            provider=provider_config["name"],
            temperature=0.0,
            reasoning_effort=None,
        )

        for api_mode in ("responses", "chat_completions"):
            try:
                if api_mode == "responses":
                    async for _ in self._iterate_openai_compatible_responses_events(
                        probe_request,
                        provider_name=provider_config["name"],
                        base_url=provider_config["base_url"],
                        api_key=provider_config["api_key"],
                        use_stream=False,
                    ):
                        pass
                else:
                    async for _ in self._iterate_openai_compatible_chat_events(
                        request=probe_request,
                        provider_name=provider_config["name"],
                        base_url=provider_config["base_url"],
                        api_key=provider_config["api_key"],
                        use_stream=False,
                    ):
                        pass
                logger.debug(
                    "Resolved custom provider %s to %s during model verification.",
                    provider_config["name"],
                    api_mode,
                )
                return api_mode
            except (ResponseFormatError, StreamingNotSupportedError) as exc:
                logger.debug(
                    "Custom provider %s rejected %s during probe: %s",
                    provider_config["name"],
                    api_mode,
                    exc,
                )
                continue
            except ProviderHTTPError as exc:
                if looks_like_unsupported_api(exc):
                    logger.debug(
                        "Custom provider %s rejected %s during probe: %s",
                        provider_config["name"],
                        api_mode,
                        format_provider_http_error_for_log(exc),
                    )
                    continue
                raise

        raise Exception(
            f"Could not determine whether {provider_config['name']} uses Responses or Chat Completions. Choose the Chat API explicitly in provider settings."
        )

    def _responses_payload(
        self,
        request: TextGenerationRequest,
        *,
        input_payload: str | list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        include_prompt_cache_key: bool,
        allow_tools: bool = True,
    ) -> dict[str, Any]:
        payload_input = input_payload if input_payload is not None else request.prompt
        payload: dict[str, Any] = {
            "model": request.model,
            "input": payload_input
            if isinstance(payload_input, list)
            else [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": payload_input}],
                }
            ],
            "store": bool(request.tools or previous_response_id),
            "truncation": "disabled",
            "tool_choice": "none",
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if request.reasoning_effort and request.reasoning_effort != "none":
            payload["reasoning"] = {"effort": request.reasoning_effort}
        else:
            payload["temperature"] = request.temperature

        if request.tools and allow_tools:
            payload["tools"] = self._responses_tools_payload(request.tools)
            payload["tool_choice"] = "auto"

        if include_prompt_cache_key and request.prompt_cache_key:
            payload["prompt_cache_key"] = request.prompt_cache_key

        return payload

    def _responses_tools_payload(
        self, tools: list[TextToolDefinition]
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]

    def _extract_openai_responses_tool_calls(
        self, response: Any
    ) -> tuple[TextToolCall, ...]:
        if not isinstance(response, dict):
            raise ResponseFormatError("Responses API returned an invalid payload.")

        output = response.get("output", [])
        if not isinstance(output, list):
            output = []

        tool_calls: list[TextToolCall] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue

            tool_name = item.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            call_id = item.get("call_id") or item.get("id") or tool_name
            tool_calls.append(
                TextToolCall(
                    id=str(call_id),
                    name=tool_name,
                    arguments=parse_tool_arguments(item.get("arguments")),
                )
            )

        return tuple(tool_calls)

    def _parse_openai_responses_turn(self, response: Any) -> ProviderTurnResult:
        if not isinstance(response, dict):
            raise ResponseFormatError("Responses API returned an invalid payload.")

        return ProviderTurnResult(
            text=self._extract_openai_responses_text(response),
            tool_calls=self._extract_openai_responses_tool_calls(response),
            usage=self._extract_usage(response),
            response_id=self._response_id(response),
        )

    def _log_reasoning_trace(
        self,
        *,
        provider: str,
        model: str,
        source: str,
        payload: Any,
    ) -> None:
        if not bool(getattr(config, "debug", False)):
            return

        for trace in self._extract_reasoning_traces(payload):
            clean_trace = " ".join(trace.split())
            if not clean_trace:
                continue
            if len(clean_trace) > REASONING_TRACE_MAX_CHARS:
                clean_trace = (
                    clean_trace[:REASONING_TRACE_MAX_CHARS].rstrip() + "... [truncated]"
                )
            logger.debug(
                "Reasoning trace provider=%s model=%s source=%s: %s",
                provider,
                model,
                source,
                clean_trace,
            )

    def _extract_reasoning_traces(self, payload: Any) -> list[str]:
        traces: list[str] = []

        def collect(value: Any, *, in_reasoning: bool = False) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item, in_reasoning=in_reasoning)
                return

            if not isinstance(value, dict):
                return

            raw_type = value.get("type")
            item_type = raw_type if isinstance(raw_type, str) else ""
            next_in_reasoning = in_reasoning or any(
                marker in item_type.lower()
                for marker in ("reasoning", "thinking", "thought")
            )

            for key, child in value.items():
                lower_key = str(key).lower()
                if any(
                    blocked in lower_key
                    for blocked in ("encrypted", "signature", "cipher")
                ):
                    continue

                child_in_reasoning = next_in_reasoning or any(
                    marker in lower_key
                    for marker in ("reasoning", "thinking", "thought")
                )

                if isinstance(child, str):
                    if child_in_reasoning and lower_key in {
                        "text",
                        "content",
                        "summary",
                        "reasoning",
                        "reasoning_content",
                        "thinking",
                        "thought",
                        "delta",
                    }:
                        traces.append(child)
                    continue

                collect(child, in_reasoning=child_in_reasoning)

        collect(payload)
        return traces

    def _extract_model_ids(self, payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            raise ResponseFormatError("Model endpoint returned an unexpected payload.")

        data = payload.get("data")
        if isinstance(data, list):
            return sorted(
                [
                    str(model["id"])
                    for model in data
                    if isinstance(model, dict) and "id" in model
                ]
            )

        models = payload.get("models")
        if isinstance(models, list):
            return sorted(
                [
                    str(model["name"])
                    for model in models
                    if isinstance(model, dict) and "name" in model
                ]
            )

        raise ResponseFormatError("Could not find any models in the payload.")

    def _extract_openai_responses_text(self, response: Any) -> str:
        if not isinstance(response, dict):
            raise ResponseFormatError("Responses API returned an invalid payload.")

        direct_text = response.get("output_text")
        if isinstance(direct_text, str):
            return direct_text

        output = response.get("output", [])
        if not isinstance(output, list):
            return ""

        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"} and isinstance(
                    part.get("text"), str
                ):
                    text_parts.append(str(part["text"]))
        return "".join(text_parts)

    def _extract_openai_chat_text(self, response: Any) -> str:
        if not isinstance(response, dict):
            raise ResponseFormatError(
                "OpenAI-compatible response was not a JSON object."
            )

        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise ResponseFormatError("OpenAI-compatible response had no choices.")

        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise ResponseFormatError("OpenAI-compatible response had no message.")

        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
                and part.get("type") in {"text", "output_text"}
            )
        raise ResponseFormatError("OpenAI-compatible message content was invalid.")

    def _extract_anthropic_text(self, response: Any) -> str:
        if not isinstance(response, dict):
            raise ResponseFormatError("Anthropic response was not a JSON object.")

        content = response.get("content", [])
        if not isinstance(content, list) or not content:
            raise ResponseFormatError("Anthropic response had no content.")

        item = content[0]
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ResponseFormatError("Anthropic response content was invalid.")
        return str(item["text"])

    def _extract_google_text(self, response: Any) -> str:
        if not isinstance(response, dict):
            raise ResponseFormatError("Google response was not a JSON object.")

        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ResponseFormatError("Google response had no candidates.")

        content = candidates[0].get("content", {})
        if not isinstance(content, dict):
            raise ResponseFormatError("Google candidate content was invalid.")

        parts = content.get("parts", [])
        if not isinstance(parts, list):
            raise ResponseFormatError("Google candidate parts were invalid.")

        return "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict)
        )

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        if not isinstance(response, dict):
            return None

        usage = response.get("usage")
        if isinstance(usage, dict):
            return cast("dict[str, Any]", usage)

        usage = response.get("usageMetadata")
        if isinstance(usage, dict):
            return cast("dict[str, Any]", usage)

        return None

    def _resolve_custom_provider_api_mode(
        self, provider_config: CustomProvider
    ) -> ResolvedApiMode:
        configured_mode = provider_config.get("chat_api_mode", "responses")
        if configured_mode == "chat_completions":
            return "chat_completions"
        if configured_mode == "responses":
            return "responses"

        logger.warning(
            "Custom provider %s still uses legacy Auto mode; defaulting to Responses until the config is re-saved.",
            provider_config["name"],
        )
        return "responses"

    def _custom_provider_streaming_enabled(
        self, provider_config: CustomProvider
    ) -> bool:
        return provider_config.get("streaming_mode", "enabled") != "disabled"

    def _reasoning_candidates(
        self, reasoning_effort: OpenAIReasoningEffort | None
    ) -> list[OpenAIReasoningEffort | None]:
        if reasoning_effort in {None, "none"}:
            return [None]
        return [reasoning_effort, None]

    def _pick_probe_model(self, models: list[str]) -> str | None:
        blocked_fragments = (
            "tts",
            "audio",
            "speech",
            "image",
            "dall-e",
            "flux",
            "diffusion",
            "embedding",
            "moderation",
            "whisper",
        )
        for model in models:
            lowered = model.lower()
            if not any(fragment in lowered for fragment in blocked_fragments):
                return model
        return models[0] if models else None

    def _describe_event_error(self, event: Any) -> str:
        if isinstance(event, dict):
            error = event.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = event.get("message")
            if isinstance(message, str) and message:
                return message
            return json.dumps(event, ensure_ascii=True)

        return str(event)

    def _response_id(self, response: dict[str, Any] | None) -> str | None:
        if response is None:
            return None
        response_id = response.get("id")
        return str(response_id) if isinstance(response_id, str) else None


chat_provider = ChatProvider()
