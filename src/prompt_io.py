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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict, cast

from .constants import GLOBAL_DECK_ID, GLOBAL_DECK_NAME
from .models import FieldExtras, PromptMap, normalize_field_extras

PROMPT_EXPORT_SCHEMA = "smart-notes-prompts"
PROMPT_EXPORT_SCHEMA_VERSION = 1


class PromptExport(TypedDict):
    schema: str
    schema_version: int
    exported_at: str
    prompts_map: PromptMap
    decks: dict[str, str]


@dataclass(frozen=True)
class SkippedPrompt:
    note_type: str
    deck_name: str
    field: str
    reason: str


@dataclass(frozen=True)
class ImportPreview:
    prompts_map: PromptMap
    added_count: int
    replaced_count: int
    skipped: list[SkippedPrompt]

    @property
    def imported_count(self) -> int:
        return self.added_count + self.replaced_count


class PromptImportError(Exception):
    pass


def create_prompt_export(
    prompts_map: PromptMap,
    deck_names_by_id: Mapping[int, str],
) -> PromptExport:
    decks = {
        str(int(deck_id)): name
        for deck_id, name in sorted(
            deck_names_by_id.items(), key=lambda item: int(item[0])
        )
    }
    if str(int(GLOBAL_DECK_ID)) not in decks:
        decks[str(int(GLOBAL_DECK_ID))] = GLOBAL_DECK_NAME

    return {
        "schema": PROMPT_EXPORT_SCHEMA,
        "schema_version": PROMPT_EXPORT_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "prompts_map": normalize_prompt_map(prompts_map),
        "decks": decks,
    }


def preview_prompt_import(
    payload: Mapping[str, Any],
    existing_prompts_map: PromptMap,
    deck_ids_by_name: Mapping[str, int],
    fields_by_note_type: Mapping[str, list[str]],
) -> ImportPreview:
    imported_prompts_map = prompt_map_from_payload(payload)
    exported_decks = decks_from_payload(payload)
    merged = deepcopy(existing_prompts_map)
    skipped: list[SkippedPrompt] = []
    added_count = 0
    replaced_count = 0

    for note_type, decks_map in imported_prompts_map["note_types"].items():
        local_fields = fields_by_note_type.get(note_type)
        if local_fields is None:
            skipped.extend(
                skipped_prompt_entries(
                    note_type=note_type,
                    deck_name="",
                    decks_map=decks_map,
                    reason="note type not found",
                )
            )
            continue

        local_fields_by_lower = {field.lower(): field for field in local_fields}

        for exported_deck_id, note_type_map in decks_map.items():
            local_deck_id = local_deck_id_for_exported_deck(
                exported_deck_id, exported_decks, deck_ids_by_name
            )
            deck_name = exported_decks.get(exported_deck_id, "")
            if local_deck_id is None:
                skipped.extend(
                    skipped_prompt_entries(
                        note_type=note_type,
                        deck_name=deck_name,
                        decks_map={exported_deck_id: note_type_map},
                        reason="deck not found",
                    )
                )
                continue

            local_deck_key = str(int(local_deck_id))
            for exported_field, prompt in note_type_map.get("fields", {}).items():
                local_field = local_fields_by_lower.get(exported_field.lower())
                if local_field is None:
                    skipped.append(
                        SkippedPrompt(
                            note_type=note_type,
                            deck_name=deck_name,
                            field=exported_field,
                            reason="field not found",
                        )
                    )
                    continue

                if note_type not in merged["note_types"]:
                    merged["note_types"][note_type] = {}
                if local_deck_key not in merged["note_types"][note_type]:
                    merged["note_types"][note_type][local_deck_key] = {
                        "fields": {},
                        "extras": {},
                    }

                local_note_type_map = merged["note_types"][note_type][local_deck_key]
                if local_field in local_note_type_map["fields"]:
                    replaced_count += 1
                else:
                    added_count += 1

                extras = note_type_map.get("extras", {}).get(exported_field)
                local_note_type_map["fields"][local_field] = prompt
                local_note_type_map["extras"][local_field] = normalize_field_extras(
                    cast("Optional[dict[str, Any] | FieldExtras]", extras)
                )

    return ImportPreview(
        prompts_map=merged,
        added_count=added_count,
        replaced_count=replaced_count,
        skipped=skipped,
    )


def normalize_prompt_map(prompts_map: PromptMap) -> PromptMap:
    normalized = deepcopy(prompts_map)
    for decks_map in normalized["note_types"].values():
        for note_type_map in decks_map.values():
            for field in note_type_map.get("fields", {}):
                raw_extras = note_type_map.get("extras", {}).get(field)
                note_type_map.setdefault("extras", {})[field] = normalize_field_extras(
                    cast("Optional[dict[str, Any] | FieldExtras]", raw_extras)
                )
    return normalized


def prompt_map_from_payload(payload: Mapping[str, Any]) -> PromptMap:
    if payload.get("schema") != PROMPT_EXPORT_SCHEMA:
        raise PromptImportError("Invalid Smart Notes prompt export file.")

    if payload.get("schema_version") != PROMPT_EXPORT_SCHEMA_VERSION:
        raise PromptImportError("Unsupported Smart Notes prompt export version.")

    raw_prompts_map = payload.get("prompts_map")
    if not isinstance(raw_prompts_map, dict):
        raise PromptImportError("Prompt export is missing prompts_map.")

    raw_note_types = raw_prompts_map.get("note_types")
    if not isinstance(raw_note_types, dict):
        raise PromptImportError("Prompt export has an invalid prompts_map.")

    prompts_map: PromptMap = {"note_types": {}}
    for note_type, decks_map in raw_note_types.items():
        if not isinstance(note_type, str) or not isinstance(decks_map, dict):
            raise PromptImportError("Prompt export has an invalid note type map.")
        prompts_map["note_types"][note_type] = {}
        for deck_id, note_type_map in decks_map.items():
            if not isinstance(deck_id, str) or not isinstance(note_type_map, dict):
                raise PromptImportError("Prompt export has an invalid deck map.")
            raw_fields = note_type_map.get("fields")
            raw_extras = note_type_map.get("extras", {})
            if not isinstance(raw_fields, dict) or not isinstance(raw_extras, dict):
                raise PromptImportError("Prompt export has invalid field data.")
            fields: dict[str, str] = {}
            extras: dict[str, FieldExtras] = {}
            for field, prompt in raw_fields.items():
                if not isinstance(field, str) or not isinstance(prompt, str):
                    raise PromptImportError("Prompt export has invalid prompt data.")
                fields[field] = prompt
                raw_field_extras = raw_extras.get(field)
                if raw_field_extras is not None and not isinstance(
                    raw_field_extras, dict
                ):
                    raise PromptImportError("Prompt export has invalid extras data.")
                extras[field] = normalize_field_extras(
                    cast("Optional[dict[str, Any]]", raw_field_extras)
                )
            prompts_map["note_types"][note_type][deck_id] = {
                "fields": fields,
                "extras": extras,
            }
    return prompts_map


def decks_from_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    raw_decks = payload.get("decks", {})
    if not isinstance(raw_decks, dict):
        raise PromptImportError("Prompt export has invalid deck metadata.")

    decks: dict[str, str] = {}
    for deck_id, deck_name in raw_decks.items():
        if isinstance(deck_id, str) and isinstance(deck_name, str):
            decks[deck_id] = deck_name
    decks.setdefault(str(int(GLOBAL_DECK_ID)), GLOBAL_DECK_NAME)
    return decks


def local_deck_id_for_exported_deck(
    exported_deck_id: str,
    exported_decks: Mapping[str, str],
    deck_ids_by_name: Mapping[str, int],
) -> Optional[int]:
    if exported_deck_id == str(int(GLOBAL_DECK_ID)):
        return GLOBAL_DECK_ID

    deck_name = exported_decks.get(exported_deck_id)
    if not deck_name:
        return None
    return deck_ids_by_name.get(deck_name)


def skipped_prompt_entries(
    note_type: str,
    deck_name: str,
    decks_map: Mapping[str, Any],
    reason: str,
) -> list[SkippedPrompt]:
    skipped: list[SkippedPrompt] = []
    for note_type_map in decks_map.values():
        if not isinstance(note_type_map, dict):
            continue
        fields = note_type_map.get("fields", {})
        if not isinstance(fields, dict):
            continue
        for field in fields:
            if isinstance(field, str):
                skipped.append(
                    SkippedPrompt(
                        note_type=note_type,
                        deck_name=deck_name,
                        field=field,
                        reason=reason,
                    )
                )
    return skipped
