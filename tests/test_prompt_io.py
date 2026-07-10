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

from typing import Any, cast

import pytest

from src.constants import GLOBAL_DECK_ID, GLOBAL_DECK_NAME
from src.models import PromptMap
from src.prompt_io import (
    PROMPT_EXPORT_SCHEMA,
    PROMPT_EXPORT_SCHEMA_VERSION,
    PromptImportError,
    create_prompt_export,
    preview_prompt_import,
)


def prompt_map(value: dict[str, Any]) -> PromptMap:
    return cast("PromptMap", value)


def test_create_prompt_export_includes_schema_and_deterministic_decks() -> None:
    prompts_map = prompt_map(
        {
            "note_types": {
                "Basic": {
                    "-1": {
                        "fields": {"Back": "Define {{Front}}"},
                        "extras": {"Back": {"type": "chat"}},
                    }
                }
            }
        }
    )

    exported = create_prompt_export(
        prompts_map,
        {
            42: "Zulu",
            GLOBAL_DECK_ID: GLOBAL_DECK_NAME,
            7: "Alpha",
        },
    )

    assert exported["schema"] == PROMPT_EXPORT_SCHEMA
    assert exported["schema_version"] == PROMPT_EXPORT_SCHEMA_VERSION
    assert list(exported["decks"].items()) == [
        ("-1", "All Decks"),
        ("7", "Alpha"),
        ("42", "Zulu"),
    ]
    extras = exported["prompts_map"]["note_types"]["Basic"]["-1"]["extras"]["Back"]
    assert extras["type"] == "chat"
    assert extras["automatic"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema": "wrong", "schema_version": 1, "prompts_map": {"note_types": {}}},
        {
            "schema": PROMPT_EXPORT_SCHEMA,
            "schema_version": 999,
            "prompts_map": {"note_types": {}},
        },
    ],
)
def test_preview_prompt_import_rejects_invalid_schema(payload: dict[str, Any]) -> None:
    with pytest.raises(PromptImportError):
        preview_prompt_import(
            payload=payload,
            existing_prompts_map=prompt_map({"note_types": {}}),
            deck_ids_by_name={},
            fields_by_note_type={},
        )


def test_preview_prompt_import_remaps_decks_and_global_deck() -> None:
    payload = {
        "schema": PROMPT_EXPORT_SCHEMA,
        "schema_version": PROMPT_EXPORT_SCHEMA_VERSION,
        "decks": {"-1": "All Decks", "123": "Spanish"},
        "prompts_map": {
            "note_types": {
                "Basic": {
                    "-1": {
                        "fields": {"Back": "Global {{Front}}"},
                        "extras": {"Back": {"type": "chat"}},
                    },
                    "123": {
                        "fields": {"Back": "Spanish {{Front}}"},
                        "extras": {"Back": {"type": "tts"}},
                    },
                }
            }
        },
    }

    preview = preview_prompt_import(
        payload=payload,
        existing_prompts_map=prompt_map({"note_types": {}}),
        deck_ids_by_name={"Spanish": 456},
        fields_by_note_type={"Basic": ["Front", "Back"]},
    )

    prompts = preview.prompts_map["note_types"]["Basic"]
    assert preview.added_count == 2
    assert preview.replaced_count == 0
    assert prompts["-1"]["fields"]["Back"] == "Global {{Front}}"
    assert prompts["456"]["fields"]["Back"] == "Spanish {{Front}}"
    assert prompts["456"]["extras"]["Back"]["type"] == "tts"


def test_preview_prompt_import_merges_replaces_and_skips() -> None:
    existing = prompt_map(
        {
            "note_types": {
                "Basic": {
                    "456": {
                        "fields": {
                            "Back": "Old",
                            "Extra": "Keep",
                        },
                        "extras": {
                            "Back": {"type": "chat"},
                            "Extra": {"type": "image"},
                        },
                    }
                }
            }
        }
    )
    payload = {
        "schema": PROMPT_EXPORT_SCHEMA,
        "schema_version": PROMPT_EXPORT_SCHEMA_VERSION,
        "decks": {"123": "Spanish", "999": "Missing"},
        "prompts_map": {
            "note_types": {
                "Basic": {
                    "123": {
                        "fields": {
                            "Back": "New",
                            "Front": "Added",
                            "Not Local": "Skip",
                        },
                        "extras": {
                            "Back": {"type": "tts"},
                            "Front": {"type": "chat"},
                            "Not Local": {"type": "image"},
                        },
                    },
                    "999": {
                        "fields": {"Back": "Skip deck"},
                        "extras": {"Back": {"type": "chat"}},
                    },
                },
                "Missing Note": {
                    "123": {
                        "fields": {"Back": "Skip note"},
                        "extras": {"Back": {"type": "chat"}},
                    }
                },
            }
        },
    }

    preview = preview_prompt_import(
        payload=payload,
        existing_prompts_map=existing,
        deck_ids_by_name={"Spanish": 456},
        fields_by_note_type={"Basic": ["Front", "Back", "Extra"]},
    )

    merged = preview.prompts_map["note_types"]["Basic"]["456"]
    assert preview.added_count == 1
    assert preview.replaced_count == 1
    assert len(preview.skipped) == 3
    assert merged["fields"] == {
        "Back": "New",
        "Extra": "Keep",
        "Front": "Added",
    }
    assert merged["extras"]["Back"]["type"] == "tts"
    assert merged["extras"]["Extra"]["type"] == "image"
    assert {skipped.reason for skipped in preview.skipped} == {
        "field not found",
        "deck not found",
        "note type not found",
    }
