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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .config import config
from .constants import DEFAULT_TEMPERATURE
from .logger import logger
from .models import (
    ChatModels,
    ChatProviders,
    CustomProvider,
    OpenAIReasoningEffort,
    openai_chat_models,
    openai_reasoning_efforts_for_model,
    provider_model_map,
)
from .provider_runtime import (
    ProviderHTTPError,
    ProviderTimeoutError,
    RequestTimeouts,
    ResponseFormatError,
    StreamingNotSupportedError,
    provider_runtime,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

OPENAI_BASE_URL = "https://api.openai.com"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
GOOGLE_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

TextEventType = Literal[
    "response_started",
    "text_delta",
    "tool_call",
    "usage_reported",
    "response_completed",
]
ResolvedApiMode = Literal["responses", "chat_completions"]


@dataclass(frozen=True)
class TextGenerationRequest:
    prompt: str
    model: str
    provider: str
    temperature: float
    reasoning_effort: OpenAIReasoningEffort | None
    prompt_cache_key: str | None = None


@dataclass(frozen=True)
class TextGenerationEvent:
    type: TextEventType
    text: str = ""
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    response_id: str | None
    usage: dict[str, Any] | None


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


def is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith(("o", "gpt-5"))


def responses_timeouts(
    reasoning_effort: OpenAIReasoningEffort | None,
) -> RequestTimeouts:
    first_event_timeout = 90.0 if reasoning_effort in {"high", "xhigh"} else 45.0
    return RequestTimeouts(
        connect_timeout_sec=10.0,
        sock_read_timeout_sec=None,
        first_event_timeout_sec=first_event_timeout,
        stream_idle_timeout_sec=first_event_timeout,
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

    if error.status not in {400, 422}:
        return False

    text = error.body.lower()
    return (
        "responses" in text
        or "chat/completions" in text
        or "unknown field" in text
        or "unexpected field" in text
        or "unsupported" in text
        or "schema" in text
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


def update_openai_chat_models(models: list[str]) -> None:
    openai_chat_models.clear()
    openai_chat_models.extend(cast("list[ChatModels]", models))
    provider_model_map["openai"] = openai_chat_models


def filter_openai_text_models(models: list[str]) -> list[str]:
    blocked_fragments = (
        "audio",
        "tts",
        "transcribe",
        "embedding",
        "moderation",
        "realtime",
        "image",
        "dall-e",
        "whisper",
        "search-preview",
    )

    filtered = [
        model
        for model in models
        if model.startswith(("gpt-", "o1", "o3", "o4"))
        and not any(fragment in model for fragment in blocked_fragments)
    ]

    def sort_key(model: str) -> tuple[int, str]:
        if model == "gpt-5-nano":
            return (0, model)
        if model == "gpt-4o-mini":
            return (1, model)
        if model == "gpt-5-mini":
            return (2, model)
        if model == "gpt-5.3-chat-latest":
            return (3, model)
        if model == "gpt-5-chat-latest":
            return (4, model)
        if model == "gpt-5":
            return (5, model)
        return (6, model)

    return sorted(dict.fromkeys(filtered), key=sort_key)


class ChatProvider:
    def __init__(self) -> None:
        self._openai_models_cache: list[str] = list(openai_chat_models)

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
    ) -> str:
        del note_id, retry_count
        request = TextGenerationRequest(
            prompt=prompt,
            model=str(model),
            provider=str(provider),
            temperature=temperature,
            reasoning_effort=choose_reasoning_effort(str(model), reasoning_effort),
            prompt_cache_key=(prompt_cache_key if provider == "openai" else None),
        )
        result = await self.generate_text(request)
        return result.text

    async def generate_text(
        self, request: TextGenerationRequest
    ) -> TextGenerationResult:
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
            update_openai_chat_models(models)
        return list(self._openai_models_cache)

    def get_cached_openai_chat_models(self) -> list[str]:
        return list(self._openai_models_cache)

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
            request,
            include_prompt_cache_key=include_prompt_cache_key,
        )
        if use_stream:
            payload["stream"] = True

        if not use_stream:
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
            if not isinstance(response, dict):
                raise ResponseFormatError("Responses API returned an invalid payload.")

            response_id = self._response_id(response)
            usage = self._extract_usage(response)
            final_text = self._extract_openai_responses_text(response)

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
                        final_text = self._extract_openai_responses_text(response_data)
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

            if event_type == "response.output_item.done":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    arguments = item.get("arguments")
                    parsed_arguments: dict[str, Any] | None = None
                    if isinstance(arguments, dict):
                        parsed_arguments = arguments
                    yield TextGenerationEvent(
                        type="tool_call",
                        tool_name=cast("str | None", item.get("name")),
                        tool_arguments=parsed_arguments,
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
                    async for event in self._iterate_openai_compatible_responses_events(
                        adjusted_request,
                        provider_name=provider_name,
                        base_url=base_url,
                        api_key=api_key,
                        use_stream=use_stream,
                    ):
                        yield event
                else:
                    async for event in self._iterate_openai_compatible_chat_events(
                        request=adjusted_request,
                        provider_name=provider_name,
                        base_url=base_url,
                        api_key=api_key,
                        use_stream=use_stream,
                    ):
                        yield event
                return
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
                if reasoning_effort is not None and looks_like_reasoning_schema_error(
                    exc.body
                ):
                    logger.info(
                        "Custom provider %s rejected reasoning_effort on %s; retrying without it.",
                        provider_name,
                        api_mode,
                    )
                    continue

                if provider_config.get(
                    "chat_api_mode", "responses"
                ) == "auto" and looks_like_unsupported_api(exc):
                    raise Exception(
                        f"Custom provider {provider_name} still uses legacy Auto mode. Re-open the provider settings and choose Responses or Chat Completions explicitly."
                    ) from exc

                raise

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
                        exc.body[:200],
                    )
                    continue
                raise

        raise Exception(
            f"Could not determine whether {provider_config['name']} uses Responses or Chat Completions. Choose the Chat API explicitly in provider settings."
        )

    def _responses_payload(
        self, request: TextGenerationRequest, *, include_prompt_cache_key: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": request.prompt}],
                }
            ],
            "store": False,
            "truncation": "disabled",
            "tool_choice": "none",
        }

        if request.reasoning_effort and request.reasoning_effort != "none":
            payload["reasoning"] = {"effort": request.reasoning_effort}
        else:
            payload["temperature"] = request.temperature

        if include_prompt_cache_key and request.prompt_cache_key:
            payload["prompt_cache_key"] = request.prompt_cache_key

        return payload

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
