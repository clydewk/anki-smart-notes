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
from datetime import datetime, timedelta, timezone

import pytest

from src.chat_usage import (
    ChatUsageTracker,
    OpenAITokenBudgetExceededError,
    build_prompt_usage_key,
    build_prompt_usage_signature,
    normalize_chat_usage,
)


class Clock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        return self.current


@pytest.mark.parametrize(
    ("raw_usage", "expected"),
    [
        (
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
            (10, 5, 15, 2, 3),
        ),
        (
            {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
            (7, 9, 16, 0, 0),
        ),
        (
            {
                "promptTokenCount": 12,
                "candidatesTokenCount": 8,
                "totalTokenCount": 20,
                "cachedContentTokenCount": 4,
                "thoughtsTokenCount": 2,
            },
            (12, 8, 20, 4, 2),
        ),
        ({"input_tokens": 6, "output_tokens": 4}, (6, 4, 10, 0, 0)),
    ],
)
def test_normalize_chat_usage(raw_usage, expected) -> None:
    usage = normalize_chat_usage(raw_usage)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == expected[:3]


def test_prompt_usage_lookup_resets_when_signature_changes(tmp_path) -> None:
    clock = Clock(datetime(2026, 3, 22, tzinfo=timezone.utc))
    tracker = ChatUsageTracker(
        state_path=str(tmp_path / "chat_usage.json"),
        now_fn=clock.now,
    )
    prompt_key = build_prompt_usage_key("Basic", 1, "Front")
    first_signature = build_prompt_usage_signature(
        prompt="Question: {{Front}}",
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
    )
    second_signature = build_prompt_usage_signature(
        prompt="Question: {{Back}}",
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
    )

    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=20,
        raw_usage={"input_tokens": 10, "output_tokens": 5},
        prompt_key=prompt_key,
        prompt_signature=first_signature,
    )

    snapshot = tracker.get_prompt_usage(
        note_type="Basic",
        deck_id=1,
        field_lower="Front",
        signature=first_signature,
    )
    assert snapshot is not None
    assert snapshot.run_count == 1

    changed_snapshot = tracker.get_prompt_usage(
        note_type="Basic",
        deck_id=1,
        field_lower="Front",
        signature=second_signature,
    )
    assert changed_snapshot is None


def test_global_prompt_usage_lookup_aggregates_matching_decks(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    signature = build_prompt_usage_signature(
        prompt="Question: {{Front}}",
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
    )

    for deck_id, input_tokens, output_tokens in ((1, 10, 5), (2, 14, 9)):
        tracker.finalize_request(
            provider="openai",
            model="gpt-5.4",
            reasoning_effort=None,
            use_tools=False,
            prompt_chars=20,
            raw_usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            prompt_key=build_prompt_usage_key("Basic", deck_id, "Front"),
            prompt_signature=signature,
        )

    exact_snapshot = tracker.get_prompt_usage(
        note_type="Basic",
        deck_id=0,
        field_lower="Front",
        signature=signature,
    )
    assert exact_snapshot is None

    aggregate_snapshot = tracker.get_prompt_usage_across_decks(
        note_type="Basic",
        field_lower="Front",
        signature=signature,
    )
    assert aggregate_snapshot is not None
    assert aggregate_snapshot.run_count == 2
    assert aggregate_snapshot.avg_input_tokens == 12
    assert aggregate_snapshot.avg_output_tokens == 7
    assert aggregate_snapshot.avg_total_tokens == 19


def test_openai_daily_usage_resets_on_utc_rollover(tmp_path) -> None:
    clock = Clock(datetime(2026, 3, 22, 23, 59, tzinfo=timezone.utc))
    tracker = ChatUsageTracker(
        state_path=str(tmp_path / "chat_usage.json"),
        now_fn=clock.now,
    )

    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=12,
        raw_usage={"input_tokens": 4, "output_tokens": 3},
    )

    first_day = tracker.get_openai_daily_usage()
    assert first_day.used_total_tokens == 7
    assert first_day.day_utc == "2026-03-22"

    clock.current = clock.current + timedelta(minutes=2)
    second_day = tracker.get_openai_daily_usage()
    assert second_day.day_utc == "2026-03-23"
    assert second_day.used_total_tokens == 0
    assert second_day.request_count == 0


@pytest.mark.asyncio
async def test_openai_budget_waits_for_active_turns_before_rejecting(
    tmp_path,
) -> None:
    clock = Clock(datetime(2026, 3, 22, tzinfo=timezone.utc))
    tracker = ChatUsageTracker(
        state_path=str(tmp_path / "chat_usage.json"),
        now_fn=clock.now,
    )

    first_permit = await tracker.acquire_openai_turn_permit(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=40,
        prompt_bytes=40,
        transport_key="text:openai:gpt-4o-mini",
        transport_window=4,
        budget_enabled=True,
        budget_limit=4_200,
    )
    assert first_permit is not None

    second_attempt = asyncio.create_task(
        tracker.acquire_openai_turn_permit(
            provider="openai",
            model="gpt-4o-mini",
            reasoning_effort=None,
            use_tools=False,
            prompt_chars=40,
            prompt_bytes=40,
            transport_key="text:openai:gpt-4o-mini",
            transport_window=4,
            budget_enabled=True,
            budget_limit=4_200,
        )
    )
    await asyncio.sleep(0)
    assert not second_attempt.done()

    await tracker.finalize_openai_turn(
        provider="openai",
        raw_usage={"input_tokens": 100, "output_tokens": 4_090},
        permit=first_permit,
    )

    with pytest.raises(OpenAITokenBudgetExceededError):
        await asyncio.wait_for(second_attempt, timeout=1.0)

    daily_usage = tracker.get_openai_daily_usage()
    assert daily_usage.used_total_tokens == 4_190
    assert daily_usage.request_count == 1


@pytest.mark.asyncio
async def test_openai_budget_respects_transport_window(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))

    first_permit = await tracker.acquire_openai_turn_permit(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=10,
        prompt_bytes=10,
        transport_key="text:openai:gpt-4o-mini",
        transport_window=1,
        budget_enabled=True,
        budget_limit=20_000,
    )
    assert first_permit is not None

    second_attempt = asyncio.create_task(
        tracker.acquire_openai_turn_permit(
            provider="openai",
            model="gpt-4o-mini",
            reasoning_effort=None,
            use_tools=False,
            prompt_chars=10,
            prompt_bytes=10,
            transport_key="text:openai:gpt-4o-mini",
            transport_window=1,
            budget_enabled=True,
            budget_limit=20_000,
        )
    )
    await asyncio.sleep(0)
    assert not second_attempt.done()

    await tracker.release_openai_turn_permit(first_permit)
    second_permit = await asyncio.wait_for(second_attempt, timeout=1.0)
    assert second_permit is not None
    await tracker.release_openai_turn_permit(second_permit)


@pytest.mark.asyncio
async def test_openai_budget_uses_max_observed_output_tokens(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    prompt_key = build_prompt_usage_key("Basic", 1, "Front")
    prompt_signature = build_prompt_usage_signature(
        prompt="Question: {{Front}}",
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
    )

    tracker.finalize_request(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=20,
        raw_usage={"input_tokens": 30, "output_tokens": 7_000},
        prompt_key=prompt_key,
        prompt_signature=prompt_signature,
        record_openai_daily_usage=False,
    )

    permit = await tracker.acquire_openai_turn_permit(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=40,
        prompt_bytes=40,
        transport_key="text:openai:gpt-4o-mini",
        transport_window=4,
        budget_enabled=True,
        budget_limit=20_000,
        prompt_key=prompt_key,
        prompt_signature=prompt_signature,
    )
    assert permit is not None
    assert permit.estimated_usage.output_tokens == 7_000
    await tracker.release_openai_turn_permit(permit)


@pytest.mark.asyncio
async def test_openai_budget_uses_utf8_bytes_for_cjk_prompts(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    prompt = "漢" * 12

    permit = await tracker.acquire_openai_turn_permit(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=len(prompt),
        prompt_bytes=len(prompt.encode("utf-8")),
        transport_key="text:openai:gpt-4o-mini",
        transport_window=4,
        budget_enabled=True,
        budget_limit=20_000,
    )
    assert permit is not None
    assert permit.estimated_usage.input_tokens >= 12
    await tracker.release_openai_turn_permit(permit)


def test_finalize_request_can_skip_openai_daily_ledger(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))

    tracker.finalize_request(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=10,
        raw_usage={"input_tokens": 3, "output_tokens": 2},
        record_openai_daily_usage=False,
    )

    daily_usage = tracker.get_openai_daily_usage()
    assert daily_usage.used_total_tokens == 0
    assert daily_usage.request_count == 0


def test_finalize_request_updates_scope_and_provider_breakdown(tmp_path) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    scope_id = tracker.open_scope()

    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=10,
        raw_usage={"input_tokens": 3, "output_tokens": 2},
        scope_id=scope_id,
    )
    tracker.finalize_request(
        provider="anthropic",
        model="claude-sonnet-4-0",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=10,
        raw_usage={"input_tokens": 4, "output_tokens": 6},
        scope_id=scope_id,
    )

    summary = tracker.close_scope(scope_id)
    assert summary.request_count == 2
    assert summary.input_tokens == 7
    assert summary.output_tokens == 8
    assert summary.total_tokens == 15
    assert [
        (provider.provider, provider.total_tokens) for provider in summary.providers
    ] == [
        ("anthropic", 10),
        ("openai", 5),
    ]


def test_planning_estimate_prefers_prompt_history_over_runtime_average(
    tmp_path,
) -> None:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    prompt_key = build_prompt_usage_key("Basic", 1, "Front")
    prompt_signature = build_prompt_usage_signature(
        prompt="Question: {{Front}}",
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
    )
    other_prompt_key = build_prompt_usage_key("Basic", 1, "Back")
    other_prompt_signature = build_prompt_usage_signature(
        prompt="Essay: {{Back}}",
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
    )

    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=200,
        raw_usage={"input_tokens": 50, "output_tokens": 600},
        prompt_key=prompt_key,
        prompt_signature=prompt_signature,
        record_openai_daily_usage=False,
    )
    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=50,
        raw_usage={"input_tokens": 2_000, "output_tokens": 12_000},
        prompt_key=other_prompt_key,
        prompt_signature=other_prompt_signature,
        record_openai_daily_usage=False,
    )

    usage = tracker.estimate_request_usage(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=200,
        prompt_bytes=200,
        prompt_key=prompt_key,
        prompt_signature=prompt_signature,
        mode="planning",
    )

    assert usage.input_tokens == 67
    assert usage.output_tokens == 1_024
