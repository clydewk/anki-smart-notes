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

from typing import Any, Optional, cast

import pytest

from src import models, rate_limiter
from src.chat_provider import ChatProvider
from src.models import ChatModels, OpenAIReasoningEffort


def test_gpt_5_4_present_in_openai_models() -> None:
    assert "gpt-5.4" in models.openai_chat_models
    assert "gpt-5.4" in models.provider_model_map["openai"]


def test_gpt_5_3_chat_latest_present() -> None:
    assert "gpt-5.3-chat-latest" in models.openai_chat_models
    assert "gpt-5.3-chat-latest" in models.provider_model_map["openai"]


def test_gpt_5_4_reasoning_efforts() -> None:
    efforts_54 = models.openai_reasoning_efforts_for_model("gpt-5.4")
    efforts_52 = models.openai_reasoning_efforts_for_model("gpt-5.2")

    assert efforts_54 == efforts_52
    assert "xhigh" in efforts_54

    efforts_chat = models.openai_reasoning_efforts_for_model("gpt-5.3-chat-latest")
    assert efforts_chat == list(models.OPENAI_DEFAULT_REASONING_EFFORTS)


def test_provider_rate_limit_includes_54() -> None:
    assert "openai:gpt-5.4" in rate_limiter.DEFAULT_RATE_LIMITS
    assert (
        rate_limiter.DEFAULT_RATE_LIMITS["openai:gpt-5.4"].rpm
        == rate_limiter.DEFAULT_RATE_LIMITS["openai:gpt-5.2"].rpm
    )


@pytest.mark.asyncio
async def test_openai_payload_with_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = ChatProvider()
    captured: dict[str, Any] = {}

    async def fake_execute(
        url: str,
        headers: dict[str, Any],
        json_payload: dict[str, Any],
        timeout_sec: int,
        retry_count: int,
        provider: str,
        prompt: str,
        model: ChatModels,
        temperature: float,
        reasoning_effort: Optional[OpenAIReasoningEffort],
    ) -> str:
        del (
            url,
            headers,
            timeout_sec,
            retry_count,
            provider,
            prompt,
            model,
            temperature,
            reasoning_effort,
        )
        captured["payload"] = json_payload
        return "fake-response"

    monkeypatch.setattr(cp, "_execute_request", fake_execute)

    await cast(Any, cp)._get_openai_response(
        "hi",
        "gpt-5.4",
        temperature=0.5,
        reasoning_effort="high",
        retry_count=0,
    )

    assert captured["payload"]["model"] == "gpt-5.4"
    assert captured["payload"]["reasoning_effort"] == "high"
