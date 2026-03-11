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

import html
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .chat_provider import TextToolDefinition
from .constants import GLOBAL_DECK_ID
from .decks import deck_id_to_name_map, deck_name_to_id_map

if TYPE_CHECKING:
    from .models import BuiltInToolId, BuiltInToolsConfig

SearchScope = Literal["current_deck", "current_note_type", "all_decks"]
DeckOverviewScope = Literal["current_deck", "all_decks"]
FieldMatchMode = Literal["exact", "contains", "regex", "exact_token"]
FieldFilterMode = Literal["and", "or"]
BuiltInToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

SEARCH_LIMIT_DEFAULT = 5
SEARCH_LIMIT_MAX = 10
FIELD_PREVIEW_LIMIT = 120
FIELD_PREVIEW_COUNT = 3
DECK_LIST_LIMIT = 20
NOTE_TYPE_SAMPLE_LIMIT = 50
NOTE_TYPE_RESULT_LIMIT = 5
KNOWN_FIELD_LIST_LIMIT = 8


@dataclass(frozen=True)
class SearchFieldFilter:
    field: str
    value: str
    match_mode: FieldMatchMode


@dataclass(frozen=True)
class BuiltInToolContext:
    note_id: int
    deck_id: int
    note_type: str
    field_name: str
    collection: Any


@dataclass(frozen=True)
class BuiltInToolMetadata:
    id: BuiltInToolId
    title: str
    description: str
    input_schema: dict[str, Any]


BUILT_IN_TOOL_METADATA: tuple[BuiltInToolMetadata, ...] = (
    BuiltInToolMetadata(
        id="anki_search_notes",
        title="Search Notes",
        description=(
            "Search notes with field-aware filters and return compact, "
            "model-friendly summaries."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text_query": {"type": "string"},
                "query": {
                    "type": "string",
                    "description": "Deprecated raw Anki search fragment.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["current_deck", "current_note_type", "all_decks"],
                },
                "note_type": {"type": "string"},
                "field_filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "value": {"type": "string"},
                            "match_mode": {
                                "type": "string",
                                "enum": ["exact", "contains", "regex", "exact_token"],
                            },
                        },
                        "required": ["field", "value"],
                    },
                },
                "field_filters_mode": {
                    "type": "string",
                    "enum": ["and", "or"],
                },
                "fields_to_return": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "exclude_current_note": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
    ),
    BuiltInToolMetadata(
        id="anki_get_deck_overview",
        title="Get Deck Overview",
        description=(
            "Inspect the current deck or list available decks with lightweight "
            "card and note metadata."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["current_deck", "all_decks"],
                },
                "deck_name": {"type": "string"},
            },
        },
    ),
)


def list_built_in_tools() -> list[BuiltInToolMetadata]:
    return list(BUILT_IN_TOOL_METADATA)


def escape_search_term(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_deck_search_query(deck_name: str) -> str:
    escaped = escape_search_term(deck_name)
    return f'(deck:"*::{escaped}" or deck:"{escaped}")'


def compact_text(value: Any, *, limit: int = FIELD_PREVIEW_LIMIT) -> str:
    text = plain_text(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def plain_text(value: Any) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def normalize_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize_text(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value, re.UNICODE)


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(value)
    return deduped


def note_field_items(note: Any) -> list[tuple[str, Any]]:
    return list(note.items())


def note_field_map(note: Any) -> dict[str, tuple[str, Any]]:
    return {
        field_name.lower(): (field_name, value)
        for field_name, value in note_field_items(note)
    }


def summarize_note(
    note: Any,
    *,
    requested_fields: list[str] | None = None,
    preferred_fields: list[str] | None = None,
) -> dict[str, Any]:
    note_type = note.note_type()
    note_type_name = note_type["name"] if note_type else "Unknown"
    field_map = note_field_map(note)

    fields: dict[str, str] = {}
    if requested_fields:
        for requested_field in requested_fields:
            actual = field_map.get(requested_field.lower())
            if actual is None:
                continue
            actual_name, value = actual
            fields[actual_name] = compact_text(value)
    else:
        ordered_fields = dedupe_strings(
            (preferred_fields or [])
            + [field_name for field_name, _ in note_field_items(note)]
        )
        for field_name in ordered_fields:
            actual = field_map.get(field_name.lower())
            if actual is None:
                continue
            actual_name, value = actual
            preview = compact_text(value)
            if not preview:
                continue
            fields[actual_name] = preview
            if len(fields) >= FIELD_PREVIEW_COUNT:
                break

    if not fields:
        for field_name, value in note_field_items(note):
            fields[field_name] = compact_text(value)
            break

    return {
        "note_id": note.id,
        "note_type": note_type_name,
        "fields": fields,
    }


def json_output(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class BuiltInToolProvider:
    def __init__(
        self,
        context: BuiltInToolContext,
        enabled_tools: BuiltInToolsConfig,
    ) -> None:
        self._context = context
        self._enabled_tools = enabled_tools
        self._tool_map: dict[str, BuiltInToolHandler] = {}

    def build_tool_registry(self) -> list[TextToolDefinition]:
        self._tool_map = {}
        tools: list[TextToolDefinition] = []

        handlers: dict[str, BuiltInToolHandler] = {
            "anki_search_notes": self._handle_anki_search_notes,
            "anki_get_deck_overview": self._handle_anki_get_deck_overview,
        }

        for metadata in BUILT_IN_TOOL_METADATA:
            if not self._enabled_tools.get(metadata.id, False):
                continue
            self._tool_map[metadata.id] = handlers[metadata.id]
            if metadata.id == "anki_search_notes":
                tools.append(self._anki_search_notes_definition(metadata))
            else:
                tools.append(
                    TextToolDefinition(
                        name=metadata.id,
                        description=metadata.description,
                        input_schema=metadata.input_schema,
                    )
                )

        return tools

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._tool_map

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        handler = self._tool_map.get(tool_name)
        if handler is None:
            raise Exception(f"Unknown built-in tool: {tool_name}")
        return await handler(arguments)

    def _collection(self) -> Any:
        if self._context.collection is None:
            raise Exception("Anki collection is not available.")
        return self._context.collection

    def _context_deck_name(self) -> str | None:
        if self._context.deck_id == GLOBAL_DECK_ID:
            return None
        return deck_id_to_name_map().get(self._context.deck_id)

    def _current_note_type_fields(self) -> list[str]:
        collection = self._collection()
        models = getattr(collection, "models", None)
        by_name = getattr(models, "by_name", None)
        if not callable(by_name):
            return []

        model = by_name(self._context.note_type)
        if not isinstance(model, dict):
            return []

        raw_fields = model.get("flds")
        if not isinstance(raw_fields, list):
            return []

        ordered_fields = sorted(
            (field for field in raw_fields if isinstance(field, dict)),
            key=lambda field: cast("int", field.get("ord", 0)),
        )
        return [
            cast("str", field["name"])
            for field in ordered_fields
            if isinstance(field.get("name"), str)
        ]

    def _anki_search_notes_definition(
        self, metadata: BuiltInToolMetadata
    ) -> TextToolDefinition:
        current_fields = self._current_note_type_fields()
        fields_hint = ", ".join(current_fields[:KNOWN_FIELD_LIST_LIMIT])
        if len(current_fields) > KNOWN_FIELD_LIST_LIMIT:
            fields_hint += ", ..."

        description = (
            "Search notes with precise filters. Prefer field_filters and "
            "fields_to_return over raw query syntax. Use this to find notes with "
            "the same reading, headword, grammar pattern, or existing field style "
            "examples."
        )
        if fields_hint:
            description += f" Current note type fields: {fields_hint}."
        description += (
            " Default field filter matching is contains. Set field_filters_mode to "
            "'or' when searching for any of several readings. For structured "
            "reading or pronunciation fields like pinyin, kana, or IPA, prefer "
            "exact_token over contains to avoid substring false positives."
        )

        return TextToolDefinition(
            name=metadata.id,
            description=description,
            input_schema=metadata.input_schema,
        )

    def _resolve_search_scope(self, arguments: dict[str, Any]) -> SearchScope:
        scope = arguments.get("scope")
        if scope in {"current_deck", "current_note_type", "all_decks"}:
            return cast("SearchScope", scope)
        return "current_deck"

    def _resolve_deck_overview_scope(
        self, arguments: dict[str, Any]
    ) -> DeckOverviewScope:
        scope = arguments.get("scope")
        if scope in {"current_deck", "all_decks"}:
            return cast("DeckOverviewScope", scope)
        return "current_deck"

    def _bounded_limit(self, raw_value: Any) -> int:
        if not isinstance(raw_value, int):
            return SEARCH_LIMIT_DEFAULT
        return max(1, min(raw_value, SEARCH_LIMIT_MAX))

    def _resolve_field_filters_mode(self, arguments: dict[str, Any]) -> FieldFilterMode:
        mode = arguments.get("field_filters_mode")
        if mode in {"and", "or"}:
            return cast("FieldFilterMode", mode)
        return "and"

    def _resolve_fields_to_return(self, arguments: dict[str, Any]) -> list[str]:
        raw_fields = arguments.get("fields_to_return")
        if not isinstance(raw_fields, list):
            return []
        return [
            field.strip()
            for field in raw_fields
            if isinstance(field, str) and field.strip()
        ]

    def _resolve_exclude_current_note(self, arguments: dict[str, Any]) -> bool:
        raw_value = arguments.get("exclude_current_note")
        if isinstance(raw_value, bool):
            return raw_value
        return True

    def _resolve_note_type_filter(self, arguments: dict[str, Any]) -> str | None:
        note_type = arguments.get("note_type")
        if isinstance(note_type, str) and note_type.strip():
            return note_type.strip()
        return None

    def _resolve_text_query(self, arguments: dict[str, Any]) -> str:
        text_query = arguments.get("text_query")
        if isinstance(text_query, str) and text_query.strip():
            return text_query.strip()

        legacy_query = arguments.get("query")
        if isinstance(legacy_query, str) and legacy_query.strip():
            return legacy_query.strip()

        return ""

    def _resolve_field_filters(
        self, arguments: dict[str, Any]
    ) -> list[SearchFieldFilter]:
        raw_filters = arguments.get("field_filters")
        if not isinstance(raw_filters, list):
            return []

        filters: list[SearchFieldFilter] = []
        for raw_filter in raw_filters:
            if not isinstance(raw_filter, dict):
                continue

            field = raw_filter.get("field")
            value = raw_filter.get("value")
            if not isinstance(field, str) or not field.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue

            raw_mode = raw_filter.get("match_mode")
            match_mode: FieldMatchMode
            if raw_mode in {"exact", "contains", "regex", "exact_token"}:
                match_mode = cast("FieldMatchMode", raw_mode)
            else:
                match_mode = "contains"

            filters.append(
                SearchFieldFilter(
                    field=field.strip(),
                    value=value.strip(),
                    match_mode=match_mode,
                )
            )

        return filters

    def _note_type_query(self, note_type_name: str) -> str:
        escaped = escape_search_term(note_type_name)
        return f'note:"{escaped}"'

    def _field_query_token(
        self, field_name: str, value: str, mode: FieldMatchMode
    ) -> str:
        search_value = value if mode == "regex" else escape_search_term(value)
        if mode in {"contains", "exact_token"}:
            search_value = f"*{search_value}*"
        elif mode == "regex":
            search_value = f"re:{value}"

        token = f"{field_name}:{search_value}"
        if " " in token:
            return f'"{escape_search_term(token)}"'
        return token

    def _field_filters_query(
        self,
        filters: list[SearchFieldFilter],
        filters_mode: FieldFilterMode,
    ) -> str:
        if not filters:
            return ""

        joiner = " or " if filters_mode == "or" else " "
        terms = [
            self._field_query_token(
                field_filter.field, field_filter.value, field_filter.match_mode
            )
            for field_filter in filters
        ]
        if len(terms) == 1:
            return terms[0]
        return f"({joiner.join(terms)})"

    def _note_matches_field_filter(
        self, note: Any, field_filter: SearchFieldFilter
    ) -> tuple[bool, str | None]:
        field_map = note_field_map(note)
        actual = field_map.get(field_filter.field.lower())
        if actual is None:
            return False, None

        _actual_name, raw_value = actual
        value = plain_text(raw_value)
        needle = plain_text(field_filter.value)

        if field_filter.match_mode == "regex":
            try:
                return re.search(
                    field_filter.value, value, re.IGNORECASE
                ) is not None, value
            except re.error as exc:
                raise Exception(
                    f"Invalid regex for field {field_filter.field}: {exc}"
                ) from exc

        haystack = value.casefold()
        target = needle.casefold()
        if field_filter.match_mode == "exact":
            return haystack == target, value
        if field_filter.match_mode == "exact_token":
            haystack_tokens = [normalize_token(token) for token in tokenize_text(value)]
            target_tokens = [normalize_token(token) for token in tokenize_text(needle)]
            if not target_tokens:
                return False, value
            return any(token in haystack_tokens for token in target_tokens), value
        return target in haystack, value

    def _note_matches_filters(
        self,
        note: Any,
        filters: list[SearchFieldFilter],
        filters_mode: FieldFilterMode,
    ) -> tuple[bool, list[dict[str, str]]]:
        if not filters:
            return True, []

        matches: list[dict[str, str]] = []
        did_match_all = True

        for field_filter in filters:
            matched, actual_value = self._note_matches_field_filter(note, field_filter)
            if matched:
                matches.append(
                    {
                        "field": field_filter.field,
                        "requested_value": field_filter.value,
                        "match_mode": field_filter.match_mode,
                        "actual_value": compact_text(actual_value or ""),
                    }
                )
            else:
                did_match_all = False
                if filters_mode == "and":
                    return False, []

        if filters_mode == "or":
            return bool(matches), matches
        return did_match_all, matches

    def _result_fields(
        self,
        note: Any,
        *,
        requested_fields: list[str],
        matched_fields: list[str],
    ) -> dict[str, str]:
        preferred_fields = dedupe_strings(
            matched_fields + self._current_note_type_fields()
        )
        return summarize_note(
            note,
            requested_fields=requested_fields or None,
            preferred_fields=preferred_fields or None,
        )["fields"]

    async def _handle_anki_search_notes(self, arguments: dict[str, Any]) -> str:
        scope = self._resolve_search_scope(arguments)
        text_query = self._resolve_text_query(arguments)
        note_type_filter = self._resolve_note_type_filter(arguments)
        field_filters = self._resolve_field_filters(arguments)
        fields_to_return = self._resolve_fields_to_return(arguments)
        field_filters_mode = self._resolve_field_filters_mode(arguments)
        exclude_current_note = self._resolve_exclude_current_note(arguments)
        limit = self._bounded_limit(arguments.get("limit"))
        effective_scope = scope

        if (
            not text_query
            and not field_filters
            and note_type_filter is None
            and scope != "current_note_type"
        ):
            raise Exception(
                "anki_search_notes requires text_query, query, field_filters, or note_type."
            )

        query_parts: list[str] = []
        if text_query:
            query_parts.append(text_query)
        if scope == "current_deck":
            deck_name = self._context_deck_name()
            if deck_name:
                query_parts.append(build_deck_search_query(deck_name))
            else:
                effective_scope = "all_decks"
        elif scope == "current_note_type" and note_type_filter is None:
            query_parts.append(self._note_type_query(self._context.note_type))

        if note_type_filter is not None:
            query_parts.append(self._note_type_query(note_type_filter))

        field_filter_query = self._field_filters_query(
            field_filters,
            field_filters_mode,
        )
        if field_filter_query:
            query_parts.append(field_filter_query)

        search_query = " ".join(part for part in query_parts if part).strip()
        collection = self._collection()
        note_ids = collection.find_notes(search_query) if search_query else []

        results: list[dict[str, Any]] = []
        total_matches = 0
        for note_id in note_ids:
            if exclude_current_note and note_id == self._context.note_id:
                continue

            note = collection.get_note(note_id)
            did_match, matched_on = self._note_matches_filters(
                note,
                field_filters,
                field_filters_mode,
            )
            if not did_match:
                continue

            total_matches += 1
            if len(results) >= limit:
                continue

            matched_fields = [match["field"] for match in matched_on]
            fields = self._result_fields(
                note,
                requested_fields=fields_to_return,
                matched_fields=matched_fields,
            )
            note_type = note.note_type()
            note_type_name = note_type["name"] if note_type else "Unknown"
            results.append(
                {
                    "note_id": note.id,
                    "note_type": note_type_name,
                    "matched_on": matched_on,
                    "fields": fields,
                }
            )

        return json_output(
            {
                "tool": "anki_search_notes",
                "text_query": text_query,
                "scope": effective_scope,
                "note_type": note_type_filter,
                "field_filters_mode": field_filters_mode,
                "field_filters": [
                    {
                        "field": field_filter.field,
                        "value": field_filter.value,
                        "match_mode": field_filter.match_mode,
                    }
                    for field_filter in field_filters
                ],
                "fields_to_return": fields_to_return,
                "exclude_current_note": exclude_current_note,
                "limit": limit,
                "match_count": total_matches,
                "current_note_type": self._context.note_type,
                "current_note_type_fields": self._current_note_type_fields(),
                "results": results,
            }
        )

    async def _handle_anki_get_deck_overview(self, arguments: dict[str, Any]) -> str:
        scope = self._resolve_deck_overview_scope(arguments)
        deck_name = arguments.get("deck_name")
        if deck_name is not None and (not isinstance(deck_name, str) or not deck_name):
            raise Exception("deck_name must be a non-empty string when provided.")

        if deck_name:
            deck_map = deck_name_to_id_map()
            deck_id = deck_map.get(deck_name)
            if deck_id is None:
                raise Exception(f"Unknown deck: {deck_name}")
            return self._deck_summary(deck_id, deck_name)

        if scope == "all_decks" or self._context.deck_id == GLOBAL_DECK_ID:
            decks = [
                {"deck_id": int(deck_id), "deck_name": name}
                for deck_id, name in deck_id_to_name_map().items()
                if deck_id != GLOBAL_DECK_ID
            ]
            decks.sort(key=lambda deck: deck["deck_name"])
            return json_output(
                {
                    "tool": "anki_get_deck_overview",
                    "scope": "all_decks",
                    "deck_count": len(decks),
                    "decks": decks[:DECK_LIST_LIMIT],
                }
            )

        deck_name = self._context_deck_name()
        if deck_name is None:
            raise Exception("Current deck is not available.")

        return self._deck_summary(self._context.deck_id, deck_name)

    def _deck_summary(self, deck_id: int, deck_name: str) -> str:
        collection = self._collection()
        deck_query = build_deck_search_query(deck_name)
        card_ids = collection.find_cards(deck_query)
        note_ids = collection.find_notes(deck_query)

        note_type_counts: dict[str, int] = {}
        for note_id in note_ids[:NOTE_TYPE_SAMPLE_LIMIT]:
            note = collection.get_note(note_id)
            note_type = note.note_type()
            note_type_name = note_type["name"] if note_type else "Unknown"
            note_type_counts[note_type_name] = (
                note_type_counts.get(note_type_name, 0) + 1
            )

        top_note_types = [
            {"name": name, "count": count}
            for name, count in sorted(
                note_type_counts.items(), key=lambda item: (-item[1], item[0])
            )[:NOTE_TYPE_RESULT_LIMIT]
        ]

        return json_output(
            {
                "tool": "anki_get_deck_overview",
                "scope": "current_deck",
                "deck": {"deck_id": int(deck_id), "deck_name": deck_name},
                "card_count": len(card_ids),
                "note_count": len(note_ids),
                "sample_note_types": top_note_types,
            }
        )
