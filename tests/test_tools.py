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

import json
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.built_in_tools import BuiltInToolContext, BuiltInToolProvider
from src.chat_provider import TextGenerationResult
from src.config import Config
from src.field_processor import FieldProcessor
from src.mcp_runtime import (
    McpServerInfo,
    McpServerProbeResult,
    McpToolCallResult,
    McpToolSpec,
)
from src.nodes import FieldNode
from src.tool_registry import ToolRegistry
from tests.mocks import MockConfig


class FakeNote:
    def __init__(self, note_id: int, note_type: str, data: dict[str, str]) -> None:
        self.id = note_id
        self._note_type = note_type
        self._data = data

    def note_type(self) -> dict[str, str]:
        return {"name": self._note_type}

    @property
    def note_type_name(self) -> str:
        return self._note_type

    def field_names(self) -> list[str]:
        return list(self._data.keys())

    def items(self) -> Any:
        return self._data.items()


class FakeModels:
    def __init__(self, fields_by_note_type: dict[str, list[str]]) -> None:
        self._fields_by_note_type = fields_by_note_type

    def by_name(self, name: str) -> dict[str, Any]:
        fields = self._fields_by_note_type.get(name, [])
        return {
            "flds": [
                {"name": field_name, "ord": index}
                for index, field_name in enumerate(fields)
            ]
        }


class FakeCollection:
    def __init__(
        self,
        *,
        notes: list[FakeNote],
        note_queries: list[tuple[str, list[int]]],
        card_queries: Optional[list[tuple[str, list[int]]]] = None,
    ) -> None:
        self._notes = {note.id: note for note in notes}
        self._note_queries = note_queries
        self._card_queries = card_queries or []
        fields_by_note_type: dict[str, list[str]] = {}
        for note in notes:
            existing_fields = fields_by_note_type.setdefault(note.note_type_name, [])
            for field_name in note.field_names():
                if field_name not in existing_fields:
                    existing_fields.append(field_name)
        self.models = FakeModels(fields_by_note_type)
        self.recorded_note_queries: list[str] = []
        self.recorded_card_queries: list[str] = []

    def find_notes(self, query: str) -> list[int]:
        self.recorded_note_queries.append(query)
        for needle, result in self._note_queries:
            if needle in query:
                return list(result)
        return []

    def find_cards(self, query: str) -> list[int]:
        self.recorded_card_queries.append(query)
        for needle, result in self._card_queries:
            if needle in query:
                return list(result)
        return []

    def get_note(self, note_id: int) -> FakeNote:
        return self._notes[note_id]


def make_server_config(server_id: str = "server-1") -> dict[str, Any]:
    return {
        "id": server_id,
        "name": "Server 1",
        "enabled": True,
        "transport": "stdio",
        "command": "cmd",
        "args": [],
        "env": [],
        "env_passthrough": [],
        "cwd": "",
        "url": "",
        "headers": [],
        "header_env_vars": [],
    }


def test_config_cleanup_migrates_legacy_tools_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_data: dict[str, Any] = {
        "chat_temperature": 1,
        "did_cleanup_config_defaults": True,
        "chat_use_mcp": True,
        "chat_use_tools": None,
        "built_in_tools": {"anki_search_notes": False},
        "prompts_map": {
            "note_types": {
                "Basic": {
                    "1": {
                        "fields": {"Back": "Prompt"},
                        "extras": {"Back": {"chat_use_mcp": True}},
                    }
                }
            }
        },
    }

    class FakeAddonManager:
        def getConfig(self, name: str) -> dict[str, Any]:
            del name
            return config_data

        def writeConfig(self, name: str, value: dict[str, Any]) -> None:
            del name
            new_value = dict(value)
            config_data.clear()
            config_data.update(new_value)

    monkeypatch.setattr(
        "src.config.mw",
        SimpleNamespace(addonManager=FakeAddonManager()),
    )

    cfg = Config()
    cfg.perform_extras_cleanup()

    assert config_data["chat_use_tools"] is True
    assert config_data["built_in_tools"] == {
        "anki_search_notes": False,
        "anki_get_deck_overview": True,
    }
    extras = config_data["prompts_map"]["note_types"]["Basic"]["1"]["extras"]["Back"]
    assert extras["chat_use_tools"] is True


def use_collection(
    monkeypatch: pytest.MonkeyPatch,
    collection: FakeCollection,
) -> None:
    async def query(operation: Any) -> Any:
        return operation(collection)

    monkeypatch.setattr("src.built_in_tools.query_collection", query)


@pytest.mark.asyncio
async def test_built_in_search_notes_defaults_to_current_deck_and_excludes_current_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(1, "Basic", {"Front": "current", "Back": "ignored"}),
        FakeNote(
            2,
            "Basic",
            {
                "Front": "a" * 160,
                "Back": "second note",
                "Extra": "example",
            },
        ),
        FakeNote(3, "Basic", {"Front": "third", "Back": "note"}),
    ]
    collection = FakeCollection(
        notes=notes,
        note_queries=[('deck:"*::Japanese"', [1, 2, 3])],
    )

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Japanese",
            note_type="Basic",
            note_type_fields=("Front", "Back", "Extra"),
            field_name="Back",
        ),
        {"anki_search_notes": True, "anki_get_deck_overview": False},
    )

    tools = provider.build_tool_registry()
    assert [tool.name for tool in tools] == ["anki_search_notes"]

    payload = json.loads(
        await provider.execute_tool_call("anki_search_notes", {"query": "reading:かな"})
    )

    assert collection.recorded_note_queries[0].endswith(
        '(deck:"*::Japanese" or deck:"Japanese")'
    )
    assert payload["scope"] == "current_deck"
    assert payload["match_count"] == 2
    assert [result["note_id"] for result in payload["results"]] == [2, 3]
    assert payload["results"][0]["fields"]["Front"].endswith("...")


@pytest.mark.asyncio
async def test_built_in_search_notes_honors_all_decks_scope_and_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(2, "Basic", {"Front": "two"}),
        FakeNote(3, "Basic", {"Front": "three"}),
        FakeNote(4, "Basic", {"Front": "four"}),
    ]
    collection = FakeCollection(notes=notes, note_queries=[("meaning", [2, 3, 4])])

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=99,
            deck_id=10,
            deck_name="Japanese",
            note_type="Basic",
            note_type_fields=("Front", "Back"),
            field_name="Back",
        ),
        {"anki_search_notes": True, "anki_get_deck_overview": False},
    )
    provider.build_tool_registry()

    payload = json.loads(
        await provider.execute_tool_call(
            "anki_search_notes",
            {"query": "meaning", "scope": "all_decks", "limit": 2},
        )
    )

    assert collection.recorded_note_queries[0] == "meaning"
    assert payload["scope"] == "all_decks"
    assert payload["limit"] == 2
    assert len(payload["results"]) == 2
    assert payload["match_count"] == 3


@pytest.mark.asyncio
async def test_built_in_search_notes_supports_field_filters_and_requested_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(
            2,
            "Chinese",
            {
                "Simplified": "<b>的</b>",
                "Pinyin": "&nbsp;[de/dí/dì]",
                "Meaning": "of; possessive particle",
                "Notes": "common structural particle",
            },
        ),
        FakeNote(
            3,
            "Chinese",
            {
                "Simplified": "上",
                "Pinyin": "[shàng / shǎng]",
                "Meaning": "up",
            },
        ),
    ]
    collection = FakeCollection(
        notes=notes,
        note_queries=[('deck:"*::Chinese"', [2, 3])],
    )

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Chinese",
            note_type="Chinese",
            note_type_fields=("Simplified", "Pinyin", "Meaning", "Notes"),
            field_name="Notes",
        ),
        {"anki_search_notes": True, "anki_get_deck_overview": False},
    )
    tools = provider.build_tool_registry()
    assert "field_filters" in tools[0].input_schema["properties"]
    assert (
        "exact_token"
        in tools[0].input_schema["properties"]["field_filters"]["items"]["properties"][
            "match_mode"
        ]["enum"]
    )

    payload = json.loads(
        await provider.execute_tool_call(
            "anki_search_notes",
            {
                "scope": "current_deck",
                "field_filters": [
                    {
                        "field": "Pinyin",
                        "value": "dí",
                        "match_mode": "contains",
                    }
                ],
                "fields_to_return": ["Simplified", "Pinyin", "Meaning"],
            },
        )
    )

    assert payload["current_note_type_fields"] == [
        "Simplified",
        "Pinyin",
        "Meaning",
        "Notes",
    ]
    assert payload["match_count"] == 1
    assert payload["results"][0]["fields"] == {
        "Meaning": "of; possessive particle",
        "Pinyin": "[de/dí/dì]",
        "Simplified": "的",
    }
    assert payload["results"][0]["matched_on"] == [
        {
            "actual_value": "[de/dí/dì]",
            "field": "Pinyin",
            "match_mode": "contains",
            "requested_value": "dí",
        }
    ]


@pytest.mark.asyncio
async def test_built_in_search_notes_supports_exact_token_without_substring_false_positives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(2, "Chinese", {"Simplified": "的", "Pinyin": "[de/dí/dì]"}),
        FakeNote(3, "Chinese", {"Simplified": "丁", "Pinyin": "[ding]"}),
        FakeNote(4, "Chinese", {"Simplified": "低", "Pinyin": "[didi]"}),
    ]
    collection = FakeCollection(
        notes=notes,
        note_queries=[('deck:"*::Chinese"', [2, 3, 4])],
    )

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Chinese",
            note_type="Chinese",
            note_type_fields=("Simplified", "Pinyin"),
            field_name="Notes",
        ),
        {"anki_search_notes": True, "anki_get_deck_overview": False},
    )
    provider.build_tool_registry()

    payload = json.loads(
        await provider.execute_tool_call(
            "anki_search_notes",
            {
                "scope": "current_deck",
                "field_filters": [
                    {
                        "field": "Pinyin",
                        "value": "di",
                        "match_mode": "exact_token",
                    }
                ],
            },
        )
    )

    assert payload["match_count"] == 1
    assert [result["note_id"] for result in payload["results"]] == [2]
    assert payload["results"][0]["matched_on"] == [
        {
            "actual_value": "[de/dí/dì]",
            "field": "Pinyin",
            "match_mode": "exact_token",
            "requested_value": "di",
        }
    ]


@pytest.mark.asyncio
async def test_built_in_search_notes_supports_or_field_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(2, "Chinese", {"Simplified": "的", "Pinyin": "[de/dí/dì]"}),
        FakeNote(3, "Chinese", {"Simplified": "上", "Pinyin": "[shàng / shǎng]"}),
        FakeNote(4, "Chinese", {"Simplified": "们", "Pinyin": "[men]"}),
    ]
    collection = FakeCollection(
        notes=notes,
        note_queries=[('deck:"*::Chinese"', [2, 3, 4])],
    )

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Chinese",
            note_type="Chinese",
            note_type_fields=("Simplified", "Pinyin"),
            field_name="Notes",
        ),
        {"anki_search_notes": True, "anki_get_deck_overview": False},
    )
    provider.build_tool_registry()

    payload = json.loads(
        await provider.execute_tool_call(
            "anki_search_notes",
            {
                "scope": "current_deck",
                "field_filters_mode": "or",
                "field_filters": [
                    {"field": "Pinyin", "value": "dí"},
                    {"field": "Pinyin", "value": "shàng"},
                ],
            },
        )
    )

    assert payload["match_count"] == 2
    assert [result["note_id"] for result in payload["results"]] == [2, 3]


@pytest.mark.asyncio
async def test_built_in_deck_overview_returns_current_deck_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [
        FakeNote(2, "Basic", {"Front": "two"}),
        FakeNote(3, "Basic", {"Front": "three"}),
        FakeNote(4, "Cloze", {"Text": "four"}),
    ]
    collection = FakeCollection(
        notes=notes,
        note_queries=[('deck:"*::Japanese"', [2, 3, 4])],
        card_queries=[('deck:"*::Japanese"', [11, 12, 13, 14])],
    )

    use_collection(monkeypatch, collection)
    provider = BuiltInToolProvider(
        BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Japanese",
            note_type="Basic",
            note_type_fields=("Front", "Back"),
            field_name="Back",
        ),
        {"anki_search_notes": False, "anki_get_deck_overview": True},
    )
    provider.build_tool_registry()

    payload = json.loads(await provider.execute_tool_call("anki_get_deck_overview", {}))

    assert collection.recorded_card_queries[0].endswith(
        '(deck:"*::Japanese" or deck:"Japanese")'
    )
    assert payload["deck"]["deck_name"] == "Japanese"
    assert payload["card_count"] == 4
    assert payload["note_count"] == 3
    assert payload["sample_note_types"][0]["name"] == "Basic"


@pytest.mark.asyncio
async def test_tool_registry_merges_built_in_and_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = [FakeNote(2, "Basic", {"Front": "hello"})]
    collection = FakeCollection(notes=notes, note_queries=[("hello", [2])])

    async def fake_probe_server(server: dict[str, Any]) -> McpServerProbeResult:
        del server
        return McpServerProbeResult(
            server_info=McpServerInfo(name="fake", version="1.0"),
            tools=[
                McpToolSpec(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                )
            ],
        )

    async def fake_call_tool(
        server: dict[str, Any], tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        del server
        return McpToolCallResult(
            text=f"{tool_name}:{arguments['value']}",
            structured_content=None,
            is_error=False,
        )

    monkeypatch.setattr("src.mcp_manager.mcp_runtime.probe_server", fake_probe_server)
    monkeypatch.setattr("src.mcp_manager.mcp_runtime.call_tool", fake_call_tool)

    use_collection(monkeypatch, collection)
    registry = ToolRegistry(
        context=BuiltInToolContext(
            note_id=1,
            deck_id=10,
            deck_name="Japanese",
            note_type="Basic",
            note_type_fields=("Front", "Back"),
            field_name="Back",
        ),
        built_in_tools={
            "anki_search_notes": True,
            "anki_get_deck_overview": False,
        },
        mcp_servers=cast("list[Any]", [make_server_config()]),
    )

    tools, warnings = await registry.build_tool_registry()

    assert warnings == []
    assert [tool.name for tool in tools] == ["anki_search_notes", "mcp_server_1_echo"]
    assert (
        await registry.execute_tool_call("mcp_server_1_echo", {"value": "hello"})
        == "echo:hello"
    )

    payload = json.loads(
        await registry.execute_tool_call("anki_search_notes", {"query": "hello"})
    )
    assert payload["tool"] == "anki_search_notes"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_type", "extras", "method_name", "data", "expected_value"),
    [
        (
            "tts",
            {
                "tts_strip_html": True,
                "tts_provider": "openai",
                "tts_model": "tts-1",
                "tts_voice": "alloy",
                "tts_style": None,
            },
            "get_tts_response",
            b"audio",
            "[sound:Basic-back-42.mp3]",
        ),
        (
            "image",
            {
                "image_model": "gpt-image-1",
                "image_provider": "openai",
                "image_aspect_ratio": None,
                "image_resolution": None,
                "image_output_format": "webp",
                "image_quality": 80,
            },
            "get_image_response",
            b"image",
            '<img src="Basic-back-42.webp"/>',
        ),
    ],
)
async def test_field_processor_defers_media_writes(
    monkeypatch: pytest.MonkeyPatch,
    field_type: str,
    extras: dict[str, Any],
    method_name: str,
    data: bytes,
    expected_value: str,
) -> None:
    monkeypatch.setattr(
        "src.field_processor.get_extras", lambda *args, **kwargs: extras
    )
    processor = FieldProcessor(MagicMock(), MagicMock(), MagicMock())
    generate = AsyncMock(return_value=data)
    monkeypatch.setattr(processor, method_name, generate)
    node = FieldNode(
        field="back",
        field_upper="Back",
        existing_value="",
        out_nodes=[],
        in_nodes=[],
        manual=False,
        overwrite=False,
        deck_id=10,
        input="{{Front}}",
        field_type=cast("Any", field_type),
    )

    result = await processor.resolve(
        node,
        note_id=42,
        note_type="Basic",
        deck_name="Japanese",
        field_order=("Front", "Back"),
        values={"front": "hello", "back": ""},
    )

    assert result.value == expected_value
    assert result.media is not None
    assert result.media.filename in expected_value
    assert result.media.data == data
    generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_field_processor_does_not_register_tools_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChatProvider:
        async def async_get_chat_response_result(
            self, prompt: str, **kwargs: Any
        ) -> TextGenerationResult:
            captured["prompt"] = prompt
            captured.update(kwargs)
            return TextGenerationResult(text="ok", response_id=None, usage=None)

    monkeypatch.setattr(
        "src.field_processor.config",
        MockConfig(
            prompts_map={"note_types": {}},
            built_in_tools={
                "anki_search_notes": True,
                "anki_get_deck_overview": True,
            },
            mcp_servers=[make_server_config()],
        ),
    )

    processor = FieldProcessor(
        cast("Any", FakeChatProvider()), MagicMock(), MagicMock()
    )
    result = await processor.get_chat_response(
        note_id=1,
        note_type="Basic",
        deck_name="Japanese",
        field_order=("Front", "Back"),
        values={"front": "hello", "back": ""},
        deck_id=10,
        prompt="Use {{Front}}",
        model="gpt-4o-mini",
        provider="openai",
        field_lower="back",
        temperature=0.5,
        should_convert_to_html=False,
        use_tools=False,
    )

    assert result == "ok"
    assert captured["tools"] is None
    assert captured["tool_executor"] is None
