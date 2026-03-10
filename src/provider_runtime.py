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
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx

from .constants import MAX_RETRIES, MAX_RETRY_WAIT_SECONDS, RETRY_BASE_SECONDS
from .logger import logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping


class ProviderUnavailableError(Exception):
    """Raised when a provider has entered a temporary cooldown state."""


class StreamingNotSupportedError(Exception):
    """Raised when a provider does not support SSE streaming."""


class ResponseFormatError(Exception):
    """Raised when a provider returns an unexpected payload shape."""


class ProviderHTTPError(Exception):
    """Raised when a provider returns a non-success HTTP response."""

    def __init__(
        self,
        status: int,
        message: str,
        body: str,
        url: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body
        self.url = url
        self.headers = dict(headers or {})


TimeoutPhase = Literal["connect", "read_idle", "first_event", "stream_idle"]


class ProviderTimeoutError(TimeoutError):
    """Raised when a provider request exceeds a liveness timeout."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        url: str,
        phase: TimeoutPhase,
        timeout_sec: float | None,
        attempt: int,
        last_event_type: str | None = None,
    ) -> None:
        detail = f"{provider}:{model} {phase.replace('_', ' ')} timeout"
        if timeout_sec is not None:
            detail += f" after {timeout_sec:.0f}s"
        if last_event_type is not None:
            detail += f" (last event: {last_event_type})"
        detail += f" [{url}]"
        super().__init__(detail)
        self.provider = provider
        self.model = model
        self.url = url
        self.phase = phase
        self.timeout_sec = timeout_sec
        self.attempt = attempt
        self.last_event_type = last_event_type


@dataclass(frozen=True)
class RequestTimeouts:
    connect_timeout_sec: float = 10.0
    sock_read_timeout_sec: float | None = 30.0
    first_event_timeout_sec: float | None = None
    stream_idle_timeout_sec: float | None = None


@dataclass
class TransportMetrics:
    concurrency_window: int
    inflight: int = 0
    retries: int = 0
    throttles: int = 0
    timeouts: int = 0
    cooldown_until: float | None = None
    ramp_blocked_until: float | None = None


class TrafficController:
    def __init__(self, initial_window: int) -> None:
        self._window = max(1, initial_window)
        self._inflight = 0
        self._success_streak = 0
        self._condition = asyncio.Condition()
        self._cooldown_until: float | None = None
        self._ramp_blocked_until: float | None = None
        self._metrics = TransportMetrics(concurrency_window=self._window)

    async def acquire(self) -> None:
        while True:
            sleep_for = 0.0
            async with self._condition:
                now = time.time()
                if self._cooldown_until is not None and now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                elif self._inflight < self._window:
                    self._inflight += 1
                    self._metrics.inflight = self._inflight
                    return
                else:
                    await self._condition.wait()
                    continue

            await asyncio.sleep(min(sleep_for, 1.0))

    async def release(self) -> None:
        async with self._condition:
            self._inflight = max(0, self._inflight - 1)
            self._metrics.inflight = self._inflight
            self._condition.notify_all()

    def note_retry(self) -> None:
        self._metrics.retries += 1

    def note_success(self) -> None:
        now = time.time()
        self._success_streak += 1

        if self._ramp_blocked_until is not None and now >= self._ramp_blocked_until:
            self._ramp_blocked_until = None
            self._metrics.ramp_blocked_until = None

        if self._ramp_blocked_until is None and self._success_streak >= 20:
            self._window += 1
            self._success_streak = 0
            self._metrics.concurrency_window = self._window

    def note_throttle(
        self,
        retry_after_sec: float | None,
        reset_after_sec: float | None,
        shared_capacity: bool,
    ) -> None:
        now = time.time()
        self._metrics.throttles += 1
        self._success_streak = 0
        self._window = max(1, self._window // 2)
        self._metrics.concurrency_window = self._window

        cooldown_sec = retry_after_sec or reset_after_sec or RETRY_BASE_SECONDS
        next_cooldown = now + cooldown_sec
        if self._cooldown_until is None or next_cooldown > self._cooldown_until:
            self._cooldown_until = next_cooldown
            self._metrics.cooldown_until = next_cooldown

        if shared_capacity and retry_after_sec is None and reset_after_sec is None:
            self._ramp_blocked_until = now + 900
            self._metrics.ramp_blocked_until = self._ramp_blocked_until

    def note_timeout(self) -> None:
        self._metrics.timeouts += 1
        self._success_streak = 0
        self._window = max(1, self._window - 1)
        self._metrics.concurrency_window = self._window

    def snapshot(self) -> dict[str, float]:
        metrics = {
            "window": float(self._metrics.concurrency_window),
            "inflight": float(self._metrics.inflight),
            "retries": float(self._metrics.retries),
            "throttles": float(self._metrics.throttles),
            "timeouts": float(self._metrics.timeouts),
        }

        if self._metrics.cooldown_until is not None:
            metrics["cooldown_remaining"] = max(
                0.0, self._metrics.cooldown_until - time.time()
            )
        if self._metrics.ramp_blocked_until is not None:
            metrics["ramp_blocked_remaining"] = max(
                0.0, self._metrics.ramp_blocked_until - time.time()
            )
        return metrics


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None

    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        return None


def parse_reset_header(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        seconds = float(value.rstrip("s").rstrip("ms"))
        if value.endswith("ms"):
            seconds /= 1000.0
        return seconds
    except ValueError:
        pass

    try:
        reset_at = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None

    return max(0.0, reset_at - time.time())


def parse_rate_limit_reset(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None

    request_reset = parse_reset_header(headers.get("x-ratelimit-reset-requests"))
    token_reset = parse_reset_header(headers.get("x-ratelimit-reset-tokens"))

    resets = [value for value in (request_reset, token_reset) if value is not None]
    if not resets:
        return None
    return max(resets)


def is_shared_capacity_error(provider: str, body_text: str) -> bool:
    if provider != "openai":
        return False

    text = body_text.lower()
    return (
        "significantly increased" in text
        or "overloading the model" in text
        or "maintain service stability" in text
    )


def is_daily_quota_error(body_text: str) -> bool:
    text = body_text.lower()
    return (
        "quota" in text
        or "daily limit" in text
        or "insufficient_quota" in text
        or "resource_exhausted" in text
        or "monthly usage limit" in text
    )


def build_httpx_timeout(
    timeouts: RequestTimeouts, *, streaming: bool = False
) -> httpx.Timeout:
    read_timeout = None if streaming else timeouts.sock_read_timeout_sec
    return httpx.Timeout(
        timeout=None,
        connect=timeouts.connect_timeout_sec,
        read=read_timeout,
        write=timeouts.connect_timeout_sec,
        pool=timeouts.connect_timeout_sec,
    )


QUEUE_LOG_THRESHOLD_SEC = 0.5
SLOW_REQUEST_LOG_THRESHOLD_SEC = 5.0


class ProviderRuntime:
    def __init__(self) -> None:
        self._controllers: dict[str, TrafficController] = {}
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

    async def get_http_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(follow_redirects=True, timeout=None)
            self._clients[loop] = client
        return client

    async def close_current_session(self) -> None:
        loop = asyncio.get_running_loop()
        client = self._clients.pop(loop, None)
        if client is not None and not client.is_closed:
            await client.aclose()

    def controller(self, key: str, initial_window: int) -> TrafficController:
        if key not in self._controllers:
            self._controllers[key] = TrafficController(initial_window)
        return self._controllers[key]

    async def request_json(
        self,
        *,
        key: str,
        initial_window: int,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: Any | None = None,
        params: dict[str, str] | None = None,
        provider: str,
        model: str,
        timeouts: RequestTimeouts,
        max_retries: int = MAX_RETRIES,
    ) -> Any:
        response_bytes = await self.request_bytes(
            key=key,
            initial_window=initial_window,
            method=method,
            url=url,
            headers=headers,
            json_payload=json_payload,
            data=data,
            params=params,
            provider=provider,
            model=model,
            timeouts=timeouts,
            max_retries=max_retries,
        )

        try:
            return json.loads(response_bytes)
        except json.JSONDecodeError as exc:
            raise ResponseFormatError(
                f"{provider}:{model} returned non-JSON content from {url}"
            ) from exc

    async def request_bytes(
        self,
        *,
        key: str,
        initial_window: int,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        data: Any | None = None,
        params: dict[str, str] | None = None,
        provider: str,
        model: str,
        timeouts: RequestTimeouts,
        max_retries: int = MAX_RETRIES,
    ) -> bytes:
        controller = self.controller(key, initial_window)
        client = await self.get_http_client()

        for attempt in range(max_retries + 1):
            queued_at = time.perf_counter()
            await controller.acquire()
            queue_time = time.perf_counter() - queued_at
            request_started_at = time.perf_counter()
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_payload,
                    content=data,
                    params=params,
                    timeout=build_httpx_timeout(timeouts),
                )

                if response.status_code >= 400:
                    raise self._http_error_from_response(response)

                self._log_request_timing(
                    provider=provider,
                    model=model,
                    url=url,
                    queue_time=queue_time,
                    request_time=time.perf_counter() - request_started_at,
                )
                controller.note_success()
                return response.content
            except ProviderHTTPError as exc:
                retry = await self._handle_http_error(
                    controller=controller,
                    provider=provider,
                    model=model,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=exc,
                )
                if retry:
                    continue
                raise
            except httpx.TimeoutException as exc:
                timeout_error = self._timeout_from_httpx_error(
                    provider=provider,
                    model=model,
                    url=url,
                    attempt=attempt + 1,
                    timeouts=timeouts,
                    error=exc,
                )
                retry = await self._handle_timeout_error(
                    controller=controller,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=timeout_error,
                    allow_retry=True,
                )
                if retry:
                    continue
                raise timeout_error from exc
            except httpx.TransportError as exc:
                controller.note_timeout()
                if attempt < max_retries:
                    controller.note_retry()
                    await asyncio.sleep(self._retry_delay(attempt, None, None))
                    continue
                raise exc
            finally:
                await controller.release()

        raise RuntimeError("unreachable")

    async def stream_sse_json(
        self,
        *,
        key: str,
        initial_window: int,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        provider: str,
        model: str,
        timeouts: RequestTimeouts,
        max_retries: int = MAX_RETRIES,
    ) -> AsyncIterator[dict[str, Any]]:
        controller = self.controller(key, initial_window)
        client = await self.get_http_client()

        for attempt in range(max_retries + 1):
            emitted_event = False
            last_event_type: str | None = None
            first_event_time: float | None = None
            event_count = 0
            queued_at = time.perf_counter()
            await controller.acquire()
            queue_time = time.perf_counter() - queued_at
            request_started_at = time.perf_counter()
            try:
                async with client.stream(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_payload,
                    params=params,
                    timeout=build_httpx_timeout(timeouts, streaming=True),
                ) as response:
                    if response.status_code in {404, 405, 415, 422}:
                        body_text = await self._read_stream_response_text(response)
                        raise StreamingNotSupportedError(body_text[:500])

                    if response.status_code >= 400:
                        raise await self._stream_http_error_from_response(response)

                    content_type = response.headers.get("Content-Type", "")
                    if "text/event-stream" not in content_type:
                        body_text = await self._read_stream_response_text(response)
                        raise StreamingNotSupportedError(body_text[:500])

                    buffer: list[str] = []
                    first_timeout = timeouts.first_event_timeout_sec or 45.0
                    idle_timeout = timeouts.stream_idle_timeout_sec or 20.0
                    line_iter = response.aiter_lines()

                    while True:
                        timeout_sec = (
                            first_timeout if not emitted_event else idle_timeout
                        )
                        try:
                            line = await asyncio.wait_for(
                                line_iter.__anext__(),
                                timeout=timeout_sec,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            raise ProviderTimeoutError(
                                provider=provider,
                                model=model,
                                url=url,
                                phase="first_event"
                                if not emitted_event
                                else "stream_idle",
                                timeout_sec=timeout_sec,
                                attempt=attempt + 1,
                                last_event_type=last_event_type,
                            ) from exc

                        if line == "":
                            payload = self._parse_sse_payload(buffer)
                            buffer.clear()
                            if payload is None:
                                continue
                            emitted_event = True
                            event_count += 1
                            if first_event_time is None:
                                first_event_time = (
                                    time.perf_counter() - request_started_at
                                )
                            last_event_type = str(payload.get("type", "unknown"))
                            yield payload
                            continue

                        buffer.append(line)

                    payload = self._parse_sse_payload(buffer)
                    if payload is not None:
                        emitted_event = True
                        event_count += 1
                        if first_event_time is None:
                            first_event_time = time.perf_counter() - request_started_at
                        last_event_type = str(payload.get("type", "unknown"))
                        yield payload

                    self._log_stream_timing(
                        provider=provider,
                        model=model,
                        url=url,
                        queue_time=queue_time,
                        request_time=time.perf_counter() - request_started_at,
                        first_event_time=first_event_time,
                        event_count=event_count,
                    )
                    controller.note_success()
                    return
            except StreamingNotSupportedError:
                raise
            except ProviderHTTPError as exc:
                retry = await self._handle_http_error(
                    controller=controller,
                    provider=provider,
                    model=model,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=exc,
                )
                if retry:
                    continue
                raise
            except ProviderTimeoutError as exc:
                retry = await self._handle_timeout_error(
                    controller=controller,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=exc,
                    allow_retry=not emitted_event,
                )
                if retry:
                    continue
                raise
            except httpx.TimeoutException as exc:
                timeout_error = self._timeout_from_httpx_error(
                    provider=provider,
                    model=model,
                    url=url,
                    attempt=attempt + 1,
                    timeouts=timeouts,
                    error=exc,
                    streaming=True,
                    last_event_type=last_event_type,
                )
                retry = await self._handle_timeout_error(
                    controller=controller,
                    attempt=attempt,
                    max_retries=max_retries,
                    error=timeout_error,
                    allow_retry=not emitted_event,
                )
                if retry:
                    continue
                raise timeout_error from exc
            except httpx.TransportError as exc:
                controller.note_timeout()
                if not emitted_event and attempt < max_retries:
                    controller.note_retry()
                    await asyncio.sleep(self._retry_delay(attempt, None, None))
                    continue
                raise exc
            finally:
                await controller.release()

        raise RuntimeError("unreachable")

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        return {
            key: controller.snapshot() for key, controller in self._controllers.items()
        }

    def _log_request_timing(
        self,
        *,
        provider: str,
        model: str,
        url: str,
        queue_time: float,
        request_time: float,
    ) -> None:
        if (
            queue_time < QUEUE_LOG_THRESHOLD_SEC
            and request_time < SLOW_REQUEST_LOG_THRESHOLD_SEC
        ):
            return

        logger.debug(
            "Request %s:%s completed in %.1fs (queued %.1fs) [%s]",
            provider,
            model,
            request_time,
            queue_time,
            url,
        )

    def _log_stream_timing(
        self,
        *,
        provider: str,
        model: str,
        url: str,
        queue_time: float,
        request_time: float,
        first_event_time: float | None,
        event_count: int,
    ) -> None:
        if (
            queue_time < QUEUE_LOG_THRESHOLD_SEC
            and request_time < SLOW_REQUEST_LOG_THRESHOLD_SEC
        ):
            return

        first_event_display = (
            f"{first_event_time:.1f}s" if first_event_time is not None else "n/a"
        )
        logger.debug(
            "Stream %s:%s completed in %.1fs (queued %.1fs, first event %s, %d events) [%s]",
            provider,
            model,
            request_time,
            queue_time,
            first_event_display,
            event_count,
            url,
        )

    async def _handle_http_error(
        self,
        *,
        controller: TrafficController,
        provider: str,
        model: str,
        attempt: int,
        max_retries: int,
        error: ProviderHTTPError,
    ) -> bool:
        if error.status not in {408, 429, 500, 502, 503, 504}:
            raise error

        retry_after = parse_retry_after(error.headers)
        reset_after = parse_rate_limit_reset(error.headers)

        if error.status == 429 and is_daily_quota_error(error.body):
            controller.note_throttle(
                retry_after_sec=retry_after,
                reset_after_sec=reset_after,
                shared_capacity=False,
            )
            raise ProviderUnavailableError(
                f"{provider}:{model} quota exhausted: {error.body[:200]}"
            ) from error

        controller.note_throttle(
            retry_after_sec=retry_after,
            reset_after_sec=reset_after,
            shared_capacity=is_shared_capacity_error(provider, error.body),
        )

        if attempt < max_retries:
            controller.note_retry()
            await asyncio.sleep(self._retry_delay(attempt, retry_after, reset_after))
            return True

        raise error

    async def _handle_timeout_error(
        self,
        *,
        controller: TrafficController,
        attempt: int,
        max_retries: int,
        error: ProviderTimeoutError,
        allow_retry: bool,
    ) -> bool:
        controller.note_timeout()
        logger.warning(str(error))
        if allow_retry and attempt < max_retries:
            controller.note_retry()
            await asyncio.sleep(self._retry_delay(attempt, None, None))
            return True
        return False

    @staticmethod
    def _http_error_from_response(response: httpx.Response) -> ProviderHTTPError:
        body_text = response.text
        return ProviderHTTPError(
            response.status_code,
            response.reason_phrase or body_text[:200],
            body_text,
            str(response.url),
            dict(response.headers),
        )

    async def _stream_http_error_from_response(
        self, response: httpx.Response
    ) -> ProviderHTTPError:
        body_text = await self._read_stream_response_text(response)
        return ProviderHTTPError(
            response.status_code,
            response.reason_phrase or body_text[:200],
            body_text,
            str(response.url),
            dict(response.headers),
        )

    @staticmethod
    async def _read_stream_response_text(response: httpx.Response) -> str:
        body = await response.aread()
        return body.decode("utf-8", errors="replace")

    @staticmethod
    def _timeout_from_httpx_error(
        *,
        provider: str,
        model: str,
        url: str,
        attempt: int,
        timeouts: RequestTimeouts,
        error: httpx.TimeoutException,
        streaming: bool = False,
        last_event_type: str | None = None,
    ) -> ProviderTimeoutError:
        if isinstance(error, httpx.ConnectTimeout):
            return ProviderTimeoutError(
                provider=provider,
                model=model,
                url=url,
                phase="connect",
                timeout_sec=timeouts.connect_timeout_sec,
                attempt=attempt,
                last_event_type=last_event_type,
            )

        return ProviderTimeoutError(
            provider=provider,
            model=model,
            url=url,
            phase="stream_idle" if streaming else "read_idle",
            timeout_sec=(
                timeouts.stream_idle_timeout_sec
                if streaming
                else timeouts.sock_read_timeout_sec
            ),
            attempt=attempt,
            last_event_type=last_event_type,
        )

    @staticmethod
    def _retry_delay(
        attempt: int,
        retry_after_sec: float | None,
        reset_after_sec: float | None,
    ) -> float:
        if retry_after_sec is not None:
            return retry_after_sec
        if reset_after_sec is not None:
            return min(reset_after_sec, MAX_RETRY_WAIT_SECONDS)
        return min((2**attempt) * RETRY_BASE_SECONDS, MAX_RETRY_WAIT_SECONDS)

    @staticmethod
    def _parse_sse_payload(lines: list[str]) -> dict[str, Any] | None:
        if not lines:
            return None

        data_parts = []
        for line in lines:
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_parts.append(line[5:].strip())

        if not data_parts:
            return None

        payload = "\n".join(data_parts)
        if payload == "[DONE]":
            return {"type": "done"}

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResponseFormatError(f"Invalid SSE payload: {payload[:500]}") from exc

        if not isinstance(data, dict):
            raise ResponseFormatError(f"Unexpected SSE payload: {payload[:500]}")
        return data


provider_runtime = ProviderRuntime()
