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
import math
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import httpx

from .chatgpt_auth import chatgpt_auth
from .models import ChatGPTModel
from .provider_runtime import ResponseFormatError, provider_runtime, request_accounting

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator


RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
LIMITS_URL = "https://chatgpt.com/backend-api/wham/usage"
CATALOG_CLIENT_VERSION = "0.149.0"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
CONTROL_TIMEOUT = 60.0


class ChatGPTTransportError(RuntimeError):
    """A transport failure safe to display without provider diagnostics."""


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


def require_success(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    if response.status_code in {401, 403}:
        message = "ChatGPT authorization failed. Sign in again in Providers."
    elif response.status_code == 429:
        message = "ChatGPT usage limit reached. Check your subscription limits."
    else:
        message = f"ChatGPT request failed (HTTP {response.status_code})."
    raise ChatGPTTransportError(message)


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


def format_limits(payload: dict[str, Any]) -> str:
    limits = payload.get("rate_limit")
    parts: list[str] = []
    if isinstance(limits, dict):
        for key in ("primary_window", "secondary_window"):
            window = limits.get(key)
            if not isinstance(window, dict):
                continue
            used = window.get("used_percent")
            seconds = window.get("limit_window_seconds")
            if (
                isinstance(used, bool)
                or not isinstance(used, (int, float))
                or not math.isfinite(used)
                or isinstance(seconds, bool)
                or not isinstance(seconds, int)
                or seconds <= 0
            ):
                continue
            if seconds % 86400 == 0:
                label = f"{seconds // 86400}-day"
            elif seconds % 3600 == 0:
                label = f"{seconds // 3600}-hour"
            else:
                label = f"{seconds / 60:g}-minute"
            parts.append(f"{label} window: {max(0, min(100, 100 - used)):g}% remaining")
    return "; ".join(parts) if parts else "Subscription limits are unavailable."


class ChatGPTTransport:
    def __init__(self) -> None:
        self._models: tuple[ChatGPTModel, ...] = ()
        self._models_generation: int | None = None

    def get_cached_models(self) -> list[ChatGPTModel]:
        if self._models_generation != chatgpt_auth.generation:
            return []
        return list(self._models)

    async def fetch_models(self) -> list[ChatGPTModel]:
        generation = chatgpt_auth.generation
        payload = await self._get_json(
            MODELS_URL, params={"client_version": CATALOG_CLIENT_VERSION}
        )
        models = parse_models(payload)
        if generation != chatgpt_auth.generation:
            raise ChatGPTTransportError(
                "The ChatGPT account changed during model discovery."
            )
        self._models = tuple(models)
        self._models_generation = generation
        return models

    async def fetch_limits(self) -> str:
        return format_limits(await self._get_json(LIMITS_URL))

    async def _get_json(
        self, url: str, *, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(CONTROL_TIMEOUT), chatgpt_auth.session():
                headers = await chatgpt_auth.headers()
                client = await provider_runtime.get_http_client()
                async with client.stream(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                    follow_redirects=False,
                    timeout=CONTROL_TIMEOUT,
                ) as response:
                    require_success(response)
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
        controller = provider_runtime.controller(
            f"chatgpt:{payload.get('model', '')}", 4
        )
        await controller.acquire()
        try:
            client = await provider_runtime.get_http_client()
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
                if response.status_code != 200 and accounting is not None:
                    with suppress(ResponseFormatError):
                        accounting.observe(await bounded_json(response))
                require_success(response)
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
                    yield event
        except (TimeoutError, httpx.TransportError):
            raise ChatGPTTransportError(
                "The ChatGPT connection ended before completion."
            ) from None
        finally:
            await controller.release()


chatgpt_transport = ChatGPTTransport()
