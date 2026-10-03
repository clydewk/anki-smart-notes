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

from typing import TYPE_CHECKING, Any, cast

import pytest

import src.prompts
from src.models import DEFAULT_EXTRAS, TTSDeliveryMode
from src.prompts import (
    add_or_update_prompts,
    get_field_dependencies,
    get_prompt_fields,
    interpolate_prompt,
)
from tests.mocks import MockConfig, MockNote

if TYPE_CHECKING:
    from anki.decks import DeckId
    from anki.notes import Note

    from src.models import (
        OverridableChatOptions,
        OverridableImageOptions,
        OverrideableTTSOptionsDict,
        PromptMap,
    )


def setup_prompts_config(
    monkeypatch: pytest.MonkeyPatch, allow_empty_fields: bool = False
) -> None:

    monkeypatch.setattr(
        src.prompts,
        "config",
        MockConfig(
            prompts_map={"note_types": {}},
            allow_empty_fields=allow_empty_fields,
        ),
    )


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("{{Front}}", ["front"]),
        ("{{Front}} and {{Back}}", ["front", "back"]),
        ("{{c1::answer}}", []),
        ("{{c2::word::hint}}", []),
        ("{{Front}} with {{c1::answer}}", ["front"]),
        ("{{Front}} {{c1::cloze1}} {{Back}} {{c2::cloze2}}", ["front", "back"]),
        ("no fields here", []),
        ("{{Field With Spaces}}", ["field with spaces"]),
    ],
)
def test_get_prompt_fields(prompt: str, expected: list[str]) -> None:

    assert get_prompt_fields(prompt) == expected


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("{{Front}}", ["Front"]),
        ("{{c1::answer}}", []),
        ("{{Front}} {{c1::answer}}", ["Front"]),
    ],
)
def test_get_prompt_fields_no_lower(prompt: str, expected: list[str]) -> None:

    assert get_prompt_fields(prompt, lower=False) == expected


@pytest.mark.parametrize(
    "mode, expected",
    [
        (None, ["front", "tone"]),
        ("custom", ["front", "tone"]),
        ("natural", ["front"]),
        ("automatic", ["front"]),
    ],
)
def test_speech_direction_dependencies_follow_delivery_mode(
    mode: TTSDeliveryMode | None, expected: list[str]
) -> None:
    extras = DEFAULT_EXTRAS.copy()
    extras.update(type="tts", tts_style="{{Tone}} for {{Front}}", tts_delivery=mode)
    assert get_field_dependencies("Read {{Front}}", extras) == expected
    extras["type"] = "chat"
    assert get_field_dependencies("Read {{Front}}", extras) == ["front"]


@pytest.mark.parametrize(
    "prompt, note_data, allow_empty_fields, expected",
    [
        (
            "Define {{Front}}",
            {"Front": "hello"},
            False,
            "Define hello",
        ),
        (
            "{{Front}} with {{c1::cloze}}",
            {"Front": "hello"},
            False,
            "hello with {{c1::cloze}}",
        ),
        (
            "Use {{Word}} in a sentence: {{c1::answer}} and {{c2::another::hint}}",
            {"Word": "apple"},
            False,
            "Use apple in a sentence: {{c1::answer}} and {{c2::another::hint}}",
        ),
        (
            "{{c1::only cloze}}",
            {},
            False,
            "{{c1::only cloze}}",
        ),
        (
            "{{Front}} and {{Back}}",
            {"Front": "hello", "Back": "world"},
            False,
            "hello and world",
        ),
        (
            "{{Front}} and {{Back}}",
            {"Front": "hello", "Back": ""},
            False,
            None,
        ),
        (
            "{{Front}} and {{Back}}",
            {"Front": "hello", "Back": ""},
            True,
            "hello and ",
        ),
    ],
)
def test_interpolate_prompt(
    prompt: str,
    note_data: dict[str, str],
    allow_empty_fields: bool,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    setup_prompts_config(monkeypatch, allow_empty_fields=allow_empty_fields)

    note = MockNote(note_type="Basic", data=note_data)
    assert interpolate_prompt(prompt, cast("Note", note)) == expected


def test_add_or_update_prompts_persists_chat_use_tools() -> None:

    prompts_map = cast("PromptMap", {"note_types": {}})

    updated = add_or_update_prompts(
        prompts_map=prompts_map,
        note_type="Basic",
        deck_id=cast("DeckId", 1),
        field="Back",
        prompt="Use {{Front}}",
        is_automatic=True,
        is_custom_model=False,
        type="chat",
        tts_options=cast(
            "OverrideableTTSOptionsDict",
            {
                "tts_voice_pool": None,
                "tts_strip_html": None,
            },
        ),
        tts_style=None,
        tts_language=None,
        chat_options=cast("dict[OverridableChatOptions, Any]", {}),
        image_options=cast("dict[OverridableImageOptions, Any]", {}),
        chat_use_tools=True,
        regenerate_when_batching=False,
    )

    extras = updated["note_types"]["Basic"]["1"]["extras"]["Back"]
    assert extras["chat_use_tools"] is True


def test_tts_language_is_independent_of_voice_pool_override() -> None:

    pool = [
        {
            "provider": "fish",
            "model": "s2.1-pro-free",
            "voice": "voice-id",
            "language": "ja",
            "enabled": True,
        }
    ]
    updated = add_or_update_prompts(
        prompts_map=cast("PromptMap", {"note_types": {}}),
        note_type="Basic",
        deck_id=cast("DeckId", 1),
        field="Audio",
        prompt="{{Front}}",
        is_automatic=True,
        is_custom_model=False,
        type="tts",
        tts_options=cast(
            "OverrideableTTSOptionsDict",
            {"tts_voice_pool": pool, "tts_strip_html": True},
        ),
        tts_style="Warm",
        tts_language="Japanese",
        chat_options=cast("dict[OverridableChatOptions, Any]", {}),
        image_options=cast("dict[OverridableImageOptions, Any]", {}),
        chat_use_tools=None,
        regenerate_when_batching=False,
    )

    extras = updated["note_types"]["Basic"]["1"]["extras"]["Audio"]
    assert extras["tts_language"] == "Japanese"
    assert extras["tts_style"] == "Warm"
    assert extras.get("tts_delivery") == "custom"
    assert extras["tts_voice_pool"] is None
    assert extras["tts_provider"] is None
    assert extras["tts_model"] is None
    assert extras["tts_voice"] is None

    overridden = add_or_update_prompts(
        prompts_map=cast("PromptMap", {"note_types": {}}),
        note_type="Basic",
        deck_id=cast("DeckId", 1),
        field="Audio",
        prompt="{{Front}}",
        is_automatic=True,
        is_custom_model=True,
        type="tts",
        tts_options=cast(
            "OverrideableTTSOptionsDict",
            {"tts_voice_pool": pool, "tts_strip_html": True},
        ),
        tts_style=None,
        tts_language="Japanese",
        chat_options=cast("dict[OverridableChatOptions, Any]", {}),
        image_options=cast("dict[OverridableImageOptions, Any]", {}),
        chat_use_tools=None,
        regenerate_when_batching=False,
        tts_delivery="automatic",
    )
    override_extras = overridden["note_types"]["Basic"]["1"]["extras"]["Audio"]
    assert override_extras["tts_voice_pool"] == pool
    assert override_extras["tts_strip_html"] is True
    assert override_extras.get("tts_delivery") == "automatic"
