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

import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, TypedDict, cast
from uuid import uuid4

from .logger import logger
from .utils import get_file_path

DEFAULT_OPENAI_OUTPUT_ESTIMATE = 128
DEFAULT_OPENAI_TOOLS_OUTPUT_ESTIMATE = 192


class StoredUsageStats(TypedDict):
    run_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_prompt_chars: int
    last_used_at: str | None


class StoredPromptUsageStats(StoredUsageStats):
    signature: str


class StoredOpenAIDailyUsage(TypedDict):
    day_utc: str
    used_input_tokens: int
    used_output_tokens: int
    used_total_tokens: int
    request_count: int


class StoredChatUsageState(TypedDict):
    prompt_stats: dict[str, StoredPromptUsageStats]
    runtime_profile_stats: dict[str, StoredUsageStats]
    openai_daily_usage: StoredOpenAIDailyUsage


@dataclass(frozen=True)
class NormalizedChatUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class PromptUsageStatsSnapshot:
    run_count: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float


@dataclass(frozen=True)
class OpenAIDailyUsageSnapshot:
    day_utc: str
    used_input_tokens: int
    used_output_tokens: int
    used_total_tokens: int
    request_count: int


@dataclass(frozen=True)
class ProviderUsageSummary:
    provider: str
    request_count: int
    total_tokens: int


@dataclass(frozen=True)
class ChatRunUsageSummary:
    request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    providers: tuple[ProviderUsageSummary, ...] = ()


@dataclass(frozen=True)
class OpenAIUsageReservation:
    reservation_id: str
    estimated_usage: NormalizedChatUsage


@dataclass
class MutableUsageScope:
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    providers: dict[str, ProviderUsageSummary] | None = None

    def add(self, provider: str, usage: NormalizedChatUsage) -> None:
        if self.providers is None:
            self.providers = {}

        self.request_count += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens

        existing = self.providers.get(provider)
        if existing is None:
            self.providers[provider] = ProviderUsageSummary(
                provider=provider,
                request_count=1,
                total_tokens=usage.total_tokens,
            )
            return

        self.providers[provider] = ProviderUsageSummary(
            provider=provider,
            request_count=existing.request_count + 1,
            total_tokens=existing.total_tokens + usage.total_tokens,
        )

    def freeze(self) -> ChatRunUsageSummary:
        provider_summaries = tuple(
            sorted(
                self.providers.values() if self.providers else [],
                key=lambda summary: summary.provider.lower(),
            )
        )
        return ChatRunUsageSummary(
            request_count=self.request_count,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            providers=provider_summaries,
        )


EMPTY_CHAT_RUN_USAGE_SUMMARY = ChatRunUsageSummary(
    request_count=0,
    input_tokens=0,
    output_tokens=0,
    total_tokens=0,
    providers=(),
)


class OpenAITokenBudgetExceededError(Exception):
    pass


def format_token_count(count: int) -> str:
    return f"{count:,}"


def build_prompt_usage_key(note_type: str, deck_id: int, field_lower: str) -> str:
    return f"{note_type}\t{deck_id}\t{field_lower.lower()}"


def build_runtime_profile_key(
    provider: str,
    model: str,
    reasoning_effort: str | None,
    use_tools: bool,
) -> str:
    return json.dumps(
        {
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "use_tools": use_tools,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_prompt_usage_signature(
    *,
    prompt: str,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    use_tools: bool,
) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "use_tools": use_tools,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def normalize_chat_usage(
    raw_usage: dict[str, Any] | None,
) -> NormalizedChatUsage | None:
    if not isinstance(raw_usage, dict):
        return None

    input_tokens = first_int(
        raw_usage.get("input_tokens"),
        raw_usage.get("prompt_tokens"),
        raw_usage.get("promptTokenCount"),
    )
    output_tokens = first_int(
        raw_usage.get("output_tokens"),
        raw_usage.get("completion_tokens"),
        raw_usage.get("candidatesTokenCount"),
    )
    total_tokens = first_int(
        raw_usage.get("total_tokens"),
        raw_usage.get("totalTokenCount"),
    )
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    normalized_input = input_tokens or 0
    normalized_output = output_tokens or 0
    normalized_total = (
        total_tokens
        if total_tokens is not None
        else normalized_input + normalized_output
    )

    return NormalizedChatUsage(
        input_tokens=normalized_input,
        output_tokens=normalized_output,
        total_tokens=normalized_total,
    )


def first_int(*values: Any) -> int | None:
    for value in values:
        normalized = coerce_int(value)
        if normalized is not None:
            return normalized
    return None


def nested_int(raw: dict[str, Any], *keys: str) -> int | None:
    current: Any = raw
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return coerce_int(current)


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class ChatUsageTracker:
    def __init__(
        self,
        *,
        state_path: str | None = None,
        persist_state: bool | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        if persist_state is None:
            persist_state = state_path is not None or not bool(os.getenv("IS_TEST"))

        self._persist_state = persist_state
        self._explicit_state_path = state_path
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._loaded = False
        self._state: StoredChatUsageState = self.default_state()
        self._reservations: dict[str, OpenAIUsageReservation] = {}
        self._scopes: dict[str, MutableUsageScope] = {}

    def open_scope(self) -> str:
        scope_id = uuid4().hex
        with self._lock:
            self._scopes[scope_id] = MutableUsageScope()
        return scope_id

    def close_scope(self, scope_id: str | None) -> ChatRunUsageSummary:
        if not scope_id:
            return EMPTY_CHAT_RUN_USAGE_SUMMARY

        with self._lock:
            scope = self._scopes.pop(scope_id, None)

        if scope is None:
            return EMPTY_CHAT_RUN_USAGE_SUMMARY
        return scope.freeze()

    def get_prompt_usage(
        self,
        *,
        note_type: str,
        deck_id: int,
        field_lower: str,
        signature: str,
    ) -> PromptUsageStatsSnapshot | None:
        with self._lock:
            self.ensure_loaded_locked()
            prompt_key = build_prompt_usage_key(note_type, deck_id, field_lower)
            prompt_stats = self._state["prompt_stats"].get(prompt_key)

            if prompt_stats is None or prompt_stats["signature"] != signature:
                return None

            return self.snapshot_for_stats(prompt_stats)

    def get_prompt_usage_across_decks(
        self,
        *,
        note_type: str,
        field_lower: str,
        signature: str,
    ) -> PromptUsageStatsSnapshot | None:
        with self._lock:
            self.ensure_loaded_locked()
            prefix = f"{note_type}\t"
            suffix = f"\t{field_lower.lower()}"
            aggregate: StoredUsageStats | None = None

            for prompt_key, prompt_stats in self._state["prompt_stats"].items():
                if not prompt_key.startswith(prefix) or not prompt_key.endswith(suffix):
                    continue
                if prompt_stats["signature"] != signature:
                    continue
                if aggregate is None:
                    aggregate = self.new_usage_stats()
                aggregate["run_count"] += prompt_stats["run_count"]
                aggregate["total_input_tokens"] += prompt_stats["total_input_tokens"]
                aggregate["total_output_tokens"] += prompt_stats["total_output_tokens"]
                aggregate["total_tokens"] += prompt_stats["total_tokens"]
                aggregate["total_prompt_chars"] += prompt_stats["total_prompt_chars"]

            return self.snapshot_for_stats(aggregate) if aggregate else None

    def get_openai_daily_usage(self) -> OpenAIDailyUsageSnapshot:
        with self._lock:
            self.ensure_loaded_locked()
            changed = self.ensure_openai_daily_usage_current_locked()
            if changed:
                self.save_locked()

            usage = self._state["openai_daily_usage"]
            return OpenAIDailyUsageSnapshot(
                day_utc=usage["day_utc"],
                used_input_tokens=usage["used_input_tokens"],
                used_output_tokens=usage["used_output_tokens"],
                used_total_tokens=usage["used_total_tokens"],
                request_count=usage["request_count"],
            )

    def reserve_openai_budget(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None,
        use_tools: bool,
        prompt_chars: int,
        budget_enabled: bool,
        budget_limit: int,
        prompt_key: str | None = None,
        prompt_signature: str | None = None,
    ) -> OpenAIUsageReservation | None:
        if provider.lower() != "openai" or not budget_enabled:
            return None

        with self._lock:
            self.ensure_loaded_locked()
            changed = self.ensure_openai_daily_usage_current_locked()
            estimate = self.estimate_usage_locked(
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                use_tools=use_tools,
                prompt_chars=prompt_chars,
                prompt_key=prompt_key,
                prompt_signature=prompt_signature,
            )

            ledger = self._state["openai_daily_usage"]
            reserved_total = sum(
                reservation.estimated_usage.total_tokens
                for reservation in self._reservations.values()
            )
            projected_total = (
                ledger["used_total_tokens"] + reserved_total + estimate.total_tokens
            )

            if projected_total > budget_limit:
                if changed:
                    self.save_locked()
                remaining = max(
                    0, budget_limit - ledger["used_total_tokens"] - reserved_total
                )
                raise OpenAITokenBudgetExceededError(
                    self.format_openai_budget_error(
                        used_total=ledger["used_total_tokens"],
                        reserved_total=reserved_total,
                        budget_limit=budget_limit,
                        estimate=estimate,
                        remaining=remaining,
                    )
                )

            reservation = OpenAIUsageReservation(
                reservation_id=uuid4().hex,
                estimated_usage=estimate,
            )
            self._reservations[reservation.reservation_id] = reservation
            if changed:
                self.save_locked()
            return reservation

    def release_reservation(self, reservation: OpenAIUsageReservation | None) -> None:
        if reservation is None:
            return

        with self._lock:
            self._reservations.pop(reservation.reservation_id, None)

    def finalize_request(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None,
        use_tools: bool,
        prompt_chars: int,
        raw_usage: dict[str, Any] | None,
        reservation: OpenAIUsageReservation | None = None,
        scope_id: str | None = None,
        prompt_key: str | None = None,
        prompt_signature: str | None = None,
    ) -> NormalizedChatUsage | None:
        normalized_usage = normalize_chat_usage(raw_usage)
        if normalized_usage is None and provider.lower() == "openai" and reservation:
            normalized_usage = reservation.estimated_usage

        with self._lock:
            self.ensure_loaded_locked()
            changed = self.ensure_openai_daily_usage_current_locked()

            if reservation is not None:
                self._reservations.pop(reservation.reservation_id, None)

            if normalized_usage is None:
                if changed:
                    self.save_locked()
                return None

            runtime_key = build_runtime_profile_key(
                provider=provider,
                model=model,
                reasoning_effort=reasoning_effort,
                use_tools=use_tools,
            )

            runtime_stats = self._state["runtime_profile_stats"].get(runtime_key)
            if runtime_stats is None:
                runtime_stats = self.new_usage_stats()
                self._state["runtime_profile_stats"][runtime_key] = runtime_stats

            self.add_usage_to_stats(
                stats=runtime_stats,
                usage=normalized_usage,
                prompt_chars=prompt_chars,
            )
            changed = True

            if prompt_key and prompt_signature:
                prompt_stats = self._state["prompt_stats"].get(prompt_key)
                if (
                    prompt_stats is None
                    or prompt_stats["signature"] != prompt_signature
                ):
                    prompt_stats = self.new_prompt_usage_stats(prompt_signature)
                    self._state["prompt_stats"][prompt_key] = prompt_stats

                self.add_usage_to_stats(
                    stats=prompt_stats,
                    usage=normalized_usage,
                    prompt_chars=prompt_chars,
                )

            if provider.lower() == "openai":
                ledger = self._state["openai_daily_usage"]
                ledger["used_input_tokens"] += normalized_usage.input_tokens
                ledger["used_output_tokens"] += normalized_usage.output_tokens
                ledger["used_total_tokens"] += normalized_usage.total_tokens
                ledger["request_count"] += 1

            if scope_id:
                scope = self._scopes.get(scope_id)
                if scope is None:
                    scope = MutableUsageScope()
                    self._scopes[scope_id] = scope
                scope.add(provider, normalized_usage)

            if changed:
                self.save_locked()

        return normalized_usage

    def default_state(self) -> StoredChatUsageState:
        return {
            "prompt_stats": {},
            "runtime_profile_stats": {},
            "openai_daily_usage": self.new_openai_daily_usage(self.current_day_utc()),
        }

    def new_usage_stats(self) -> StoredUsageStats:
        return {
            "run_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_prompt_chars": 0,
            "last_used_at": None,
        }

    def new_prompt_usage_stats(self, signature: str) -> StoredPromptUsageStats:
        stats = cast("StoredPromptUsageStats", self.new_usage_stats())
        stats["signature"] = signature
        return stats

    def new_openai_daily_usage(self, day_utc: str) -> StoredOpenAIDailyUsage:
        return {
            "day_utc": day_utc,
            "used_input_tokens": 0,
            "used_output_tokens": 0,
            "used_total_tokens": 0,
            "request_count": 0,
        }

    def add_usage_to_stats(
        self,
        *,
        stats: StoredUsageStats,
        usage: NormalizedChatUsage,
        prompt_chars: int,
    ) -> None:
        stats["run_count"] += 1
        stats["total_input_tokens"] += usage.input_tokens
        stats["total_output_tokens"] += usage.output_tokens
        stats["total_tokens"] += usage.total_tokens
        stats["total_prompt_chars"] += max(prompt_chars, 0)
        stats["last_used_at"] = self.utc_now().isoformat()

    def estimate_usage_locked(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None,
        use_tools: bool,
        prompt_chars: int,
        prompt_key: str | None = None,
        prompt_signature: str | None = None,
    ) -> NormalizedChatUsage:
        baseline_input = max(1, math.ceil(max(prompt_chars, 1) / 4))
        estimated_input = baseline_input

        prompt_stats: StoredPromptUsageStats | None = None
        if prompt_key and prompt_signature:
            candidate = self._state["prompt_stats"].get(prompt_key)
            if candidate is not None and candidate["signature"] == prompt_signature:
                prompt_stats = candidate

        runtime_key = build_runtime_profile_key(
            provider=provider,
            model=model,
            reasoning_effort=reasoning_effort,
            use_tools=use_tools,
        )
        runtime_stats = self._state["runtime_profile_stats"].get(runtime_key)

        for stats in (prompt_stats, runtime_stats):
            if not stats or stats["run_count"] <= 0 or stats["total_prompt_chars"] <= 0:
                continue
            ratio = stats["total_input_tokens"] / stats["total_prompt_chars"]
            estimated_input = max(
                estimated_input,
                math.ceil(max(prompt_chars, 1) * ratio),
            )

        estimated_output = (
            DEFAULT_OPENAI_TOOLS_OUTPUT_ESTIMATE
            if use_tools
            else DEFAULT_OPENAI_OUTPUT_ESTIMATE
        )
        estimated_output = max(estimated_output, estimated_input)
        for stats in (prompt_stats, runtime_stats):
            if not stats or stats["run_count"] <= 0:
                continue
            estimated_output = round(stats["total_output_tokens"] / stats["run_count"])
            break

        estimated_output = max(0, estimated_output)

        return NormalizedChatUsage(
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            total_tokens=estimated_input + estimated_output,
        )

    def snapshot_for_stats(
        self, stats: StoredUsageStats | StoredPromptUsageStats
    ) -> PromptUsageStatsSnapshot | None:
        run_count = stats["run_count"]
        if run_count <= 0:
            return None

        return PromptUsageStatsSnapshot(
            run_count=run_count,
            avg_input_tokens=stats["total_input_tokens"] / run_count,
            avg_output_tokens=stats["total_output_tokens"] / run_count,
            avg_total_tokens=stats["total_tokens"] / run_count,
        )

    def ensure_loaded_locked(self) -> None:
        if self._loaded:
            return

        if not self._persist_state:
            self._state = self.default_state()
            self._loaded = True
            return

        state_path = self.state_path()
        try:
            with open(state_path, encoding="utf-8") as handle:
                raw_state = json.load(handle)
        except FileNotFoundError:
            self._state = self.default_state()
        except Exception as error:
            logger.error("Failed to load chat usage state: %s", error)
            self._state = self.default_state()
        else:
            self._state = self.coerce_state(raw_state)

        self._loaded = True

    def state_path(self) -> str:
        if self._explicit_state_path:
            return self._explicit_state_path

        try:
            return get_file_path("chat_usage.json")
        except Exception:
            return os.path.join(os.getcwd(), "chat_usage.json")

    def save_locked(self) -> None:
        if not self._persist_state:
            return

        state_path = self.state_path()
        directory = os.path.dirname(state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f"{state_path}.tmp"

        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2, sort_keys=True)
            os.replace(temp_path, state_path)
        except Exception as error:
            logger.error("Failed to save chat usage state: %s", error)
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def coerce_state(self, raw_state: Any) -> StoredChatUsageState:
        coerced = self.default_state()
        if not isinstance(raw_state, dict):
            return coerced

        prompt_stats = raw_state.get("prompt_stats")
        if isinstance(prompt_stats, dict):
            for key, value in prompt_stats.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                signature = value.get("signature")
                if not isinstance(signature, str) or not signature:
                    continue
                coerced["prompt_stats"][key] = self.coerce_prompt_usage_stats(
                    value, signature
                )

        runtime_profile_stats = raw_state.get("runtime_profile_stats")
        if isinstance(runtime_profile_stats, dict):
            for key, value in runtime_profile_stats.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                coerced["runtime_profile_stats"][key] = self.coerce_usage_stats(value)

        openai_daily_usage = raw_state.get("openai_daily_usage")
        if isinstance(openai_daily_usage, dict):
            coerced["openai_daily_usage"] = self.coerce_openai_daily_usage(
                openai_daily_usage
            )

        return coerced

    def coerce_prompt_usage_stats(
        self, raw_stats: dict[str, Any], signature: str
    ) -> StoredPromptUsageStats:
        stats = cast("StoredPromptUsageStats", self.coerce_usage_stats(raw_stats))
        stats["signature"] = signature
        return stats

    def coerce_usage_stats(self, raw_stats: dict[str, Any]) -> StoredUsageStats:
        stats = self.new_usage_stats()
        for key in (
            "run_count",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_prompt_chars",
        ):
            coerced = coerce_int(raw_stats.get(key))
            if coerced is not None and coerced >= 0:
                stats[key] = coerced

        last_used_at = raw_stats.get("last_used_at")
        if isinstance(last_used_at, str) or last_used_at is None:
            stats["last_used_at"] = last_used_at

        return stats

    def coerce_openai_daily_usage(
        self, raw_usage: dict[str, Any]
    ) -> StoredOpenAIDailyUsage:
        day_utc = raw_usage.get("day_utc")
        usage = self.new_openai_daily_usage(
            day_utc if isinstance(day_utc, str) and day_utc else self.current_day_utc()
        )
        for key in (
            "used_input_tokens",
            "used_output_tokens",
            "used_total_tokens",
            "request_count",
        ):
            coerced = coerce_int(raw_usage.get(key))
            if coerced is not None and coerced >= 0:
                usage[key] = coerced
        return usage

    def ensure_openai_daily_usage_current_locked(self) -> bool:
        current_day = self.current_day_utc()
        usage = self._state["openai_daily_usage"]
        if usage["day_utc"] == current_day:
            return False

        self._state["openai_daily_usage"] = self.new_openai_daily_usage(current_day)
        return True

    def current_day_utc(self) -> str:
        return self.utc_now().date().isoformat()

    def utc_now(self) -> datetime:
        current = self._now_fn()
        if current.tzinfo is None:
            return current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    def format_openai_budget_error(
        self,
        *,
        used_total: int,
        reserved_total: int,
        budget_limit: int,
        estimate: NormalizedChatUsage,
        remaining: int,
    ) -> str:
        reserved_text = (
            f" (+ {format_token_count(reserved_total)} reserved in-flight)"
            if reserved_total
            else ""
        )
        return (
            "OpenAI daily token budget reached.\n\n"
            f"Today: {format_token_count(used_total)} / {format_token_count(budget_limit)}{reserved_text}\n"
            f"Estimated next request: {format_token_count(estimate.total_tokens)} total "
            f"({format_token_count(estimate.input_tokens)} input, "
            f"{format_token_count(estimate.output_tokens)} output)\n"
            f"Remaining before this request: {format_token_count(remaining)}\n"
            "Resets at 00:00 UTC."
        )


chat_usage_tracker = ChatUsageTracker()
