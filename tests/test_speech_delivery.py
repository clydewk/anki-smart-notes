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
from copy import deepcopy
from dataclasses import asdict, dataclass
from html import unescape
from threading import Event
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chat_provider import (
    ChatProvider,
    TextGenerationEvent,
    TextGenerationRequest,
    TextGenerationResult,
)
from src.config import config_context
from src.field_processor import FieldProcessor, SpeechResult, parse_speech_style
from src.job_context import JobCancelledError, JobContext, current_job
from src.models import (
    DEFAULT_EXTRAS,
    FieldExtras,
    normalize_field_extras,
    tts_delivery_mode,
)
from src.provider_runtime import ResponseFormatError
from src.tts_provider import TTSProvider
from tests.mocks import MockConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@dataclass
class SpeechHarness:
    processor: FieldProcessor
    planner: AsyncMock
    synthesize: AsyncMock


@pytest.fixture
def speech_processor(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpeechHarness]:
    settings = asdict(
        MockConfig(prompts_map={"note_types": {}}, google_api_key="synthetic")
    )
    settings["tts_voice_pool"] = [
        {
            "provider": "google",
            "model": "gemini-3.8-flash-tts",
            "voice": "Kore",
            "language": None,
            "enabled": True,
        }
    ]
    token = config_context.set(settings)
    chat = MagicMock()
    chat.async_get_chat_response_result = AsyncMock(
        return_value=TextGenerationResult(
            '{"style":"Warm and measured."}',
            None,
            {"input_tokens": 10, "output_tokens": 8},
        )
    )
    tts = MagicMock()
    tts.async_get_tts_response = AsyncMock(return_value=b"audio")
    monkeypatch.setattr("src.field_processor.chat_usage_tracker", MagicMock())
    monkeypatch.setattr("src.tts_provider.anki_strip_html", unescape)
    try:
        yield SpeechHarness(
            FieldProcessor(chat, tts, MagicMock()),
            chat.async_get_chat_response_result,
            tts.async_get_tts_response,
        )
    finally:
        config_context.reset(token)


async def generate(processor: SpeechHarness, extras: FieldExtras) -> SpeechResult:
    result = await processor.processor.generate_speech(
        note_id=17,
        field="Audio",
        values={"text": "待って <short pause> いいよ。", "emotion": "reassuring"},
        input_text="{{Text}}",
        extras=extras,
        show_error_box=False,
        usage_scope_id="speech-test",
    )
    assert result is not None
    return result


def test_legacy_delivery_survives_normalization_and_explicit_natural_wins() -> None:
    extras = normalize_field_extras({"type": "tts", "tts_style": "Warm"})
    assert tts_delivery_mode(extras) == "custom"
    extras["tts_delivery"] = "natural"
    assert tts_delivery_mode(normalize_field_extras(extras)) == "natural"
    assert tts_delivery_mode(DEFAULT_EXTRAS) == "natural"


@pytest.mark.parametrize("part", ["text", "direction"])
async def test_openai_speech_limits_apply_after_field_substitution(
    speech_processor: SpeechHarness, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    speech_processor.processor.tts_provider = TTSProvider()
    network = AsyncMock()
    monkeypatch.setattr("src.tts_provider.provider_runtime.request_bytes", network)
    extras = deepcopy(DEFAULT_EXTRAS)
    extras.update(
        use_custom_model=True,
        tts_voice_pool=[
            {
                "provider": "openai",
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
                "enabled": True,
                "language": None,
            }
        ],
        tts_delivery="custom",
        tts_style="{{Direction}}",
    )
    values = {"text": "Hello", "direction": "Warm"}
    values[part] = "x" * 4097
    with pytest.raises(ValueError, match="4,096"):
        await speech_processor.processor.generate_speech(
            note_id=17,
            field="Audio",
            values=values,
            input_text="{{Text}}",
            extras=extras,
            show_error_box=False,
        )
    network.assert_not_awaited()


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        '{"style":null}',
        '{"style":12}',
        '{"style":"x","text":"rewritten"}',
        '```json\n{"style":"warm"}\n```',
        '{"style":"' + "a" * 401 + '"}',
        '{"style":"a\\nb"}',
        "[" * 2000,
        "a" * 4097,
    ],
)
def test_invalid_director_output_is_rejected(response: str) -> None:
    with pytest.raises(ValueError):
        parse_speech_style(response)


@pytest.mark.parametrize("style", ["", " Gentle, measured delivery. "])
def test_director_accepts_empty_or_short_style(style: str) -> None:
    assert parse_speech_style(json.dumps({"style": style})) == style.strip()


@pytest.mark.asyncio
async def test_automatic_uses_default_model_without_tools_or_rewriting(
    speech_processor: SpeechHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = MagicMock()
    monkeypatch.setattr("src.field_processor.chat_usage_tracker", ledger)
    extras = deepcopy(DEFAULT_EXTRAS)
    extras.update(
        tts_delivery="automatic",
        chat_model="ignored-field-model",
        chat_use_tools=True,
        tts_style="ignored {{Missing}}",
    )
    result = await generate(speech_processor, extras)
    assert result.transcript == "待って <short pause> いいよ。"
    assert result.style == "Warm and measured."
    planner = speech_processor.planner.call_args
    assert planner.kwargs["model"] == "gpt-4o-mini"
    assert planner.kwargs["provider"] == "openai"
    assert planner.kwargs["tools"] is None and planner.kwargs["tool_executor"] is None
    assert planner.kwargs["output_token_limit"] == 1024
    assert planner.kwargs["response_character_limit"] == 4096
    assert (
        json.loads(planner.args[0].split("Transcript (JSON string):\n")[1])
        == result.transcript
    )
    synthesis = speech_processor.synthesize.call_args.kwargs
    assert synthesis["input"] == result.transcript
    assert synthesis["instructions"] == result.style
    assert synthesis["voice"] == "Kore"
    assert synthesis["strip_html"] is False
    ledger.finalize_request.assert_called_once()
    assert ledger.finalize_request.call_args.kwargs["scope_id"] == "speech-test"


@pytest.mark.asyncio
async def test_custom_resolves_directions_and_natural_ignores_them_without_planner(
    speech_processor: SpeechHarness,
) -> None:
    extras = deepcopy(DEFAULT_EXTRAS)
    extras["tts_style"] = "Read with {{Emotion}} delivery."
    custom = await generate(speech_processor, extras)
    assert custom.style == "Read with reassuring delivery."
    assert custom.transcript == "待って <short pause> いいよ。"
    extras["tts_delivery"] = "natural"
    extras["tts_style"] = "{{Missing}}"
    natural = await generate(speech_processor, extras)
    assert natural.style == ""
    assert natural.target == custom.target
    speech_processor.planner.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_planner_output_records_usage_but_never_synthesizes(
    speech_processor: SpeechHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = MagicMock()
    monkeypatch.setattr("src.field_processor.chat_usage_tracker", ledger)
    speech_processor.planner.return_value = TextGenerationResult(
        "invalid", None, {"input_tokens": 4, "output_tokens": 2}
    )
    extras = deepcopy(DEFAULT_EXTRAS)
    extras["tts_delivery"] = "automatic"
    with pytest.raises(ValueError, match="JSON"):
        await generate(speech_processor, extras)
    ledger.finalize_request.assert_called_once()
    speech_processor.synthesize.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [asyncio.CancelledError(), TimeoutError(), ResponseFormatError("incomplete")],
)
async def test_planner_failure_never_synthesizes(
    speech_processor: SpeechHarness, error: BaseException
) -> None:
    extras = deepcopy(DEFAULT_EXTRAS)
    extras["tts_delivery"] = "automatic"
    speech_processor.planner.side_effect = error
    with pytest.raises(type(error)):
        await generate(speech_processor, extras)
    speech_processor.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_invalidated_after_planning_prevents_synthesis(
    speech_processor: SpeechHarness,
) -> None:
    active = Event()
    active.set()
    job = JobContext(object(), 1, active)
    token = current_job.set(job)

    async def plan(*args: Any, **kwargs: Any) -> TextGenerationResult:
        active.clear()
        return TextGenerationResult('{"style":""}', None, None)

    speech_processor.planner.side_effect = plan
    extras = deepcopy(DEFAULT_EXTRAS)
    extras["tts_delivery"] = "automatic"
    try:
        with pytest.raises(JobCancelledError):
            await generate(speech_processor, extras)
    finally:
        current_job.reset(token)
    speech_processor.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_automatic_rejects_unsupported_voice_before_planning(
    speech_processor: SpeechHarness,
) -> None:
    extras = deepcopy(DEFAULT_EXTRAS)
    extras.update(
        tts_delivery="automatic",
        use_custom_model=True,
        tts_voice_pool=[
            {
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy",
                "language": None,
                "enabled": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="Gemini 3.8"):
        await generate(speech_processor, extras)
    speech_processor.planner.assert_not_awaited()
    speech_processor.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_bound_closes_stream_and_subsequent_request_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = []

    async def events(
        request: TextGenerationRequest,
    ) -> AsyncIterator[TextGenerationEvent]:
        try:
            yield TextGenerationEvent("text_delta", text="too large")
            yield TextGenerationEvent("response_completed")
        finally:
            closed.append(True)

    provider = ChatProvider()
    monkeypatch.setattr(provider, "async_iter_text_events", events)
    request = TextGenerationRequest(
        "test",
        "gemini-2.5-flash",
        "google",
        0,
        None,
        output_token_limit=512,
        response_character_limit=4,
    )
    with pytest.raises(ResponseFormatError, match="text limit"):
        await provider.generate_text(request)
    assert closed == [True]
    assert request.max_output_tokens == 512
    assert (
        await provider.generate_text(
            TextGenerationRequest("test", "gemini-2.5-flash", "google", 0, None)
        )
    ).text == "too large"


@pytest.mark.asyncio
async def test_empty_transcript_skips_requests_and_empty_style_is_actionable(
    speech_processor: SpeechHarness,
) -> None:
    extras = deepcopy(DEFAULT_EXTRAS)
    assert (
        await speech_processor.processor.generate_speech(
            note_id=17,
            field="Audio",
            values={"text": ""},
            input_text="{{Text}}",
            extras=extras,
        )
        is None
    )
    extras["tts_style"] = "Read with {{Missing}} delivery"
    with pytest.raises(ValueError, match="empty or unavailable field"):
        await generate(speech_processor, extras)
    speech_processor.planner.assert_not_awaited()
    speech_processor.synthesize.assert_not_awaited()


@pytest.mark.asyncio
async def test_planner_deadline_cancels_then_natural_delivery_still_works(
    speech_processor: SpeechHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_timeout = asyncio.timeout

    def short_timeout(delay: float) -> asyncio.Timeout:
        assert delay == 60
        return original_timeout(0.001)

    monkeypatch.setattr("src.field_processor.asyncio.timeout", short_timeout)
    closed = []

    async def plan(*args: Any, **kwargs: Any) -> TextGenerationResult:
        try:
            await asyncio.Event().wait()
            raise AssertionError("Unreachable")
        finally:
            closed.append(True)

    speech_processor.planner.side_effect = plan
    extras = deepcopy(DEFAULT_EXTRAS)
    extras["tts_delivery"] = "automatic"
    with pytest.raises(TimeoutError, match="Automatic delivery timed out"):
        await generate(speech_processor, extras)
    assert closed == [True]
    speech_processor.synthesize.assert_not_awaited()
    extras["tts_delivery"] = "natural"
    assert (await generate(speech_processor, extras)).data == b"audio"
