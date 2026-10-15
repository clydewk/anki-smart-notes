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

import asyncio
import json
import re
from contextlib import suppress
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx

from .chatgpt_auth import account_epoch, chatgpt_auth
from .logger import redact_text
from .models import ChatGPTModel
from .provider_runtime import ResponseFormatError, provider_runtime, request_accounting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator


RESPONSES_URL = "https://api.openai.com/v1/responses"
MODELS_URL = "https://api.openai.com/v1/models"
USAGE_LIMIT_CODE = "subscription_sharing_usage_limit_exceeded"
USAGE_LIMIT_MESSAGE = (
    "ChatGPT plan requests are paused. Review ChatGPT settings → Usage, then choose "
    "Resume requests in Providers. An app-specific limit may apply."
)
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
CONTROL_TIMEOUT = 60.0


class ChatGPTTransportError(RuntimeError):
    """A transport failure containing only bounded, redacted diagnostic fields."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        param: str | None = None,
        request_id: str | None = None,
        body_shape: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.param = param
        self.request_id = request_id
        self.body_shape = body_shape
        details = [f"HTTP {status_code}"] if status_code is not None else []
        details.extend(
            f"{name}: {value}"
            for name, value in (
                ("code", code),
                ("param", param),
                ("request ID", request_id),
                ("body", body_shape),
            )
            if value is not None
        )
        super().__init__(message + (f" ({'; '.join(details)})" if details else ""))


def diagnostic_field(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,200}", value) is None
    ):
        return None
    return redact_text(value)


def response_error(
    payload: dict[str, Any], status: int | None, request_id: str | None
) -> ChatGPTTransportError:
    error = payload.get("error")
    error = error if isinstance(error, dict) else {}
    code = diagnostic_field(error.get("code"))
    messages = {
        USAGE_LIMIT_CODE: USAGE_LIMIT_MESSAGE,
        "subscription_sharing_user_not_eligible": "ChatGPT plan usage is unavailable for this account, workspace, or policy.",
        "subscription_sharing_usage_unavailable": "ChatGPT usage availability could not be checked. Try again later.",
        "subscription_sharing_unsupported_capability": "ChatGPT plan usage does not support a requested option. Review the indicated setting.",
        "subscription_sharing_route_not_supported": "ChatGPT plan usage rejected this API route. Check the integration.",
        "subscription_sharing_invalid_user": "ChatGPT could not validate the selected account. Check its connection and permissions.",
        "chatpass_v2_scope_not_authorized": "ChatGPT plan permission was not accepted. Check the connection and client configuration.",
        "chatpass_v2_invalid_authorization_context": "ChatGPT plan permission was not accepted. Check the connection and client configuration.",
        "subscription_sharing_user_unavailable": "ChatGPT account information is temporarily unavailable. Try again later.",
    }
    fallback = {
        401: "ChatGPT did not accept the selected account or its plan permission. Check the connection.",
        403: "A ChatGPT policy or permission restriction prevented this request. Check the connection and serving region.",
        429: "ChatGPT temporarily limited this request. Try again later.",
        503: "ChatGPT plan routing is temporarily unavailable. Try again later.",
    }.get(status or 0, "ChatGPT did not complete the request. No text was saved.")
    return ChatGPTTransportError(
        messages.get(code or "", fallback),
        status_code=status,
        code=code,
        param=diagnostic_field(error.get("param")),
        request_id=diagnostic_field(request_id),
        body_shape="error object"
        if error
        else "detail"
        if "detail" in payload
        else "other",
    )


def parse_event(lines: list[bytes]) -> dict[str, Any] | None:
    data = b"\n".join(
        line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:")
    )
    if not data:
        return None
    if data == b"[DONE]":
        return {"type": "done"}
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ResponseFormatError("ChatGPT returned an invalid stream event.") from None
    if not isinstance(payload, dict):
        raise ResponseFormatError("ChatGPT returned an invalid stream event.")
    return payload


async def response_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    pending = bytearray()
    lines: list[bytes] = []
    event_bytes = 0
    async for chunk in response.aiter_bytes():
        pending.extend(chunk)
        while (end := pending.find(b"\n")) >= 0:
            event_bytes += end + 1
            if event_bytes > MAX_MESSAGE_BYTES:
                raise ResponseFormatError(
                    "ChatGPT stream event exceeded the size limit."
                )
            line = bytes(pending[:end]).removesuffix(b"\r")
            del pending[: end + 1]
            if line:
                lines.append(line)
                continue
            payload = parse_event(lines)
            lines.clear()
            event_bytes = 0
            if payload is not None:
                yield payload
        if event_bytes + len(pending) > MAX_MESSAGE_BYTES:
            raise ResponseFormatError("ChatGPT stream event exceeded the size limit.")
    if pending or lines:
        raise ResponseFormatError("ChatGPT ended with an incomplete stream event.")


async def bounded_json(response: httpx.Response) -> dict[str, Any]:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > MAX_MESSAGE_BYTES:
            raise ResponseFormatError("ChatGPT response exceeded the size limit.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise ResponseFormatError("ChatGPT returned an invalid response.") from None
    if not isinstance(payload, dict):
        raise ResponseFormatError("ChatGPT returned an invalid response.")
    return payload


def parse_models(payload: dict[str, Any]) -> list[ChatGPTModel]:
    entries = payload.get("models")
    if not isinstance(entries, list):
        raise ResponseFormatError("ChatGPT returned an invalid model catalog.")
    models: dict[str, ChatGPTModel] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ResponseFormatError("ChatGPT returned an invalid model catalog.")
        if entry.get("visibility") != "list":
            continue
        model, label = entry.get("slug"), entry.get("display_name")
        levels = entry.get("supported_reasoning_levels", [])
        default = entry.get("default_reasoning_level")
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(label, str)
            or not isinstance(levels, list)
            or (default is not None and not isinstance(default, str))
        ):
            raise ResponseFormatError("ChatGPT returned an invalid model catalog.")
        efforts: list[str] = []
        for level in levels:
            if not isinstance(level, dict) or not isinstance(level.get("effort"), str):
                raise ResponseFormatError("ChatGPT returned invalid reasoning options.")
            efforts.append(level["effort"])
        models[model] = ChatGPTModel(
            model, label, tuple(dict.fromkeys(efforts)), default
        )
    return list(models.values())


class ChatGPTTransport:
    def __init__(self) -> None:
        self._models: tuple[ChatGPTModel, ...] = ()
        self._models_generation: int | None = None
        self._usage_lock = Lock()
        self._usage_limit_generation: int | None = None
        self._usage_revision = 0

    @property
    def usage_limit_reached(self) -> bool:
        with self._usage_lock:
            return self._usage_limit_generation == chatgpt_auth.generation

    def resume_after_usage_review(self) -> None:
        with self._usage_lock:
            self._usage_limit_generation = None
            self._usage_revision += 1

    def _check_available(self, generation: int, revision: int | None = None) -> int:
        with self._usage_lock:
            if generation != chatgpt_auth.generation:
                raise ChatGPTTransportError(
                    "The ChatGPT account changed. Run generation again."
                )
            if self._usage_limit_generation == generation:
                raise ChatGPTTransportError(USAGE_LIMIT_MESSAGE, code=USAGE_LIMIT_CODE)
            if revision is not None and revision != self._usage_revision:
                raise ChatGPTTransportError(
                    "ChatGPT usage settings changed. Run generation again."
                )
            return self._usage_revision

    def _raise_response_error(
        self,
        payload: dict[str, Any],
        generation: int,
        status: int | None,
        request_id: str | None,
        revision: int,
    ) -> None:
        error = response_error(payload, status, request_id)
        if error.code == USAGE_LIMIT_CODE:
            with self._usage_lock:
                if (
                    generation == chatgpt_auth.generation
                    and revision == self._usage_revision
                ):
                    self._usage_limit_generation = generation
                    self._usage_revision += 1
        raise error

    def get_cached_models(self) -> list[ChatGPTModel]:
        if self._models_generation != chatgpt_auth.generation:
            return []
        return list(self._models)

    async def fetch_models(self) -> list[ChatGPTModel]:
        generation = chatgpt_auth.generation
        payload = await self._get_json(MODELS_URL)
        models = parse_models(payload)
        if generation != chatgpt_auth.generation:
            raise ChatGPTTransportError(
                "The ChatGPT account changed during model discovery."
            )
        self._models = tuple(models)
        self._models_generation = generation
        return models

    async def _get_json(self, url: str) -> dict[str, Any]:
        generation = chatgpt_auth.generation
        with self._usage_lock:
            revision = self._usage_revision
        try:
            async with asyncio.timeout(CONTROL_TIMEOUT), chatgpt_auth.session():
                headers = await chatgpt_auth.headers()
                client = await provider_runtime.get_http_client()
                async with client.stream(
                    "GET",
                    url,
                    headers=headers,
                    follow_redirects=False,
                    timeout=CONTROL_TIMEOUT,
                ) as response:
                    payload: dict[str, Any] = {}
                    if response.status_code != 200:
                        with suppress(ResponseFormatError):
                            payload = await bounded_json(response)
                        self._raise_response_error(
                            payload,
                            generation,
                            response.status_code,
                            response.headers.get("x-request-id"),
                            revision,
                        )
                    return await bounded_json(response)
        except (TimeoutError, httpx.TransportError):
            raise ChatGPTTransportError(
                "The ChatGPT request could not complete. Try again."
            ) from None

    async def stream_response(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        try:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise ResponseFormatError(
                "ChatGPT request contains invalid data."
            ) from None
        if len(body) > MAX_MESSAGE_BYTES:
            raise ResponseFormatError("ChatGPT request exceeded the size limit.")
        generation = account_epoch.get()
        generation = chatgpt_auth.generation if generation is None else generation
        revision = self._check_available(generation)
        controller = provider_runtime.controller(f"chatgpt:{generation}", 4)
        await controller.acquire()
        try:
            self._check_available(generation, revision)
            client = await provider_runtime.get_http_client()
            self._check_available(generation, revision)
            accounting = request_accounting.get()
            if accounting is not None:
                accounting.start_request()
            async with client.stream(
                "POST",
                RESPONSES_URL,
                content=body,
                headers={
                    **headers,
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                follow_redirects=False,
                timeout=httpx.Timeout(60.0, connect=10.0),
            ) as response:
                request_id = response.headers.get("x-request-id")
                if response.status_code != 200:
                    error_payload: dict[str, Any] = {}
                    with suppress(ResponseFormatError):
                        error_payload = await bounded_json(response)
                    if accounting is not None:
                        accounting.observe(error_payload)
                    self._raise_response_error(
                        error_payload,
                        generation,
                        response.status_code,
                        request_id,
                        revision,
                    )
                if (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                    != "text/event-stream"
                ):
                    if accounting is not None:
                        with suppress(ResponseFormatError):
                            accounting.observe(await bounded_json(response))
                    raise ResponseFormatError(
                        "ChatGPT did not return a streaming response."
                    )
                async for event in response_events(response):
                    if accounting is not None:
                        accounting.observe(event)
                    if event.get("type") in {
                        "error",
                        "response.failed",
                        "response.incomplete",
                    }:
                        failure = event.get("response", event)
                        if event.get("type") == "error" and "code" in event:
                            failure = {"error": event}
                        if isinstance(failure, dict) and isinstance(
                            failure.get("error"), dict
                        ):
                            self._raise_response_error(
                                failure,
                                generation,
                                response.status_code,
                                request_id,
                                revision,
                            )
                    if generation != chatgpt_auth.generation:
                        raise ChatGPTTransportError(
                            "The ChatGPT account changed. Run generation again."
                        )
                    yield event
        except (TimeoutError, httpx.TransportError):
            raise ChatGPTTransportError(
                "The ChatGPT connection ended before completion."
            ) from None
        finally:
            await controller.release()


chatgpt_transport = ChatGPTTransport()
