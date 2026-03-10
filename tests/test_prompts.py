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

from typing import Optional

import pytest

from tests.mocks import MockConfig, MockNote


def setup_prompts_config(
    monkeypatch: pytest.MonkeyPatch, allow_empty_fields: bool = False
) -> None:
    import src.prompts

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
    from src.prompts import get_prompt_fields

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
    from src.prompts import get_prompt_fields

    assert get_prompt_fields(prompt, lower=False) == expected


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
    expected: Optional[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.prompts import interpolate_prompt

    setup_prompts_config(monkeypatch, allow_empty_fields=allow_empty_fields)

    note = MockNote(note_type="Basic", data=note_data)
    assert interpolate_prompt(prompt, note) == expected
