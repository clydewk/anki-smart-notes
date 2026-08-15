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

import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

from src.chat_usage import (
    ChatUsageTracker,
    build_prompt_usage_key,
    build_prompt_usage_signature,
)
from tests.mocks import (
    MockConfig,
    MockNote,
    p,
)

NOTE_TYPE_NAME = "note_type_1"


class FakeFieldProcessor:
    async def resolve(
        self,
        node: Any,
        *,
        values: dict[str, str],
        show_error_box: bool = False,
        usage_scope_id: str | None = None,
        **context: Any,
    ) -> Any:
        from src.note_processor import ResolvedField
        from src.prompts import config, interpolate_prompt_with_values

        del show_error_box, usage_scope_id, context

        interpolated = interpolate_prompt_with_values(
            node.input,
            values,
            config.allow_empty_fields,
        )
        return ResolvedField(p(interpolated) if interpolated else None)


def make_snapshot(note: Any, deck_id: int = 1) -> Any:
    from src.note_processor import snapshot_note

    return snapshot_note(note, deck_id)


def setup_data(monkeypatch, note, prompts_map, options, allow_empty_fields):
    import src.config
    import src.prompts

    fake_field_processor_module = types.ModuleType("src.field_processor")
    fake_field_processor_module.FieldProcessor = object
    monkeypatch.setitem(sys.modules, "src.field_processor", fake_field_processor_module)

    fake_sentry_module = types.ModuleType("src.sentry")
    fake_sentry_module.run_async_in_background_with_sentry = (
        lambda *args, **kwargs: None
    )
    monkeypatch.setitem(sys.modules, "src.sentry", fake_sentry_module)

    fake_ui_utils_module = types.ModuleType("src.ui.ui_utils")
    fake_ui_utils_module.show_message_box = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "src.ui.ui_utils", fake_ui_utils_module)

    from src.note_processor import NoteProcessor

    extras = {
        field: {
            "automatic": not options.get(field, {}).get("manual", False),
            "type": "chat",
            "use_custom_model": False,
            "chat_model": "gpt-4o-mini",
            "chat_provider": "openai",
            "chat_temperature": 0,
            "chat_reasoning_effort": None,
            "chat_markdown_to_html": False,
        }
        for field in prompts_map
    }

    prompts_map = {
        "note_types": {NOTE_TYPE_NAME: {"1": {"fields": prompts_map, "extras": extras}}}
    }

    c = MockConfig(prompts_map=prompts_map, allow_empty_fields=allow_empty_fields)
    f = FakeFieldProcessor()
    p = NoteProcessor(field_processor=f, config=c)

    monkeypatch.setattr(src.config, "config", c)
    monkeypatch.setattr(src.prompts, "config", c)
    monkeypatch.setattr(
        src.prompts,
        "get_prompts_for_note",
        lambda note_type, deck_id, override_prompts_map=None: prompts_map["note_types"][
            note_type
        ]["1"]["fields"],
    )
    monkeypatch.setattr(
        src.prompts,
        "get_extras",
        lambda note_type,
        field,
        deck_id,
        prompts=None,
        fallback_to_global_deck=True: extras.get(field, {"automatic": True}),
    )

    return p


def setup_tracker(monkeypatch, tmp_path: Path) -> ChatUsageTracker:
    tracker = ChatUsageTracker(state_path=str(tmp_path / "chat_usage.json"))
    monkeypatch.setattr("src.note_processor.chat_usage_tracker", tracker)
    return tracker


"""
test_processor_1 Parameters:
    name: str - Test case name for identification
    note: dict[str, str] - Note field data, e.g. {"f1": "value", "f2": ""}
    prompts_map: dict[str, str] - Field prompts, e.g. {"f2": "{{f1}}"}
    expected: dict[str, str] - Expected field values after processing
    options: dict[str, Any] - Test options:
        - "overwrite": bool - Whether to overwrite existing field values
        - "target_field": str - Specific field to process (if any)
        - "allow_empty": bool - Whether to allow processing with empty reference fields
        - "{field_name}": dict - Field-specific options:
            - "manual": bool - Whether field is marked as manual

Example: ("basic", {"f1": "1", "f2": ""}, {"f2": "{{f1}}"}, {"f2": "p_1"}, {})
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, note, prompts_map, expected, options",
    [
        # ------ Basic -------
        # A super basic, single field example
        ("basic", {"f1": "1", "f2": ""}, {"f2": "{{f1}}"}, {"f2": p("1")}, {}),
        # Two fields, in parallel
        (
            "parallel",
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            {"f3": p("1"), "f4": p("2")},
            {},
        ),
        # ---- Overwrite ----
        # no overwrite
        ("no overwrite", {"f1": "1", "f2": "old"}, {"f2": "{{f1}}"}, {"f2": "old"}, {}),
        # yes overwrite
        (
            "yes overwrite",
            {"f1": "1", "f2": "old"},
            {"f2": "{{f1}}"},
            {"f2": p("1")},
            {"overwrite": True},
        ),
        # Chained overwrite does overwrite
        (
            "chained overwrite does overwrite",
            {"f1": "1", "f2": "old", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": p("1"), "f3": p(p("1"))},
            {"overwrite": True},
        ),
        # Chained no overwrite does not overwrite
        (
            "chained overwrite does not overwrite",
            {"f1": "1", "f2": "old", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": "old", "f3": p("old")},
            {"overwrite": False},
        ),
        # ---- Allow Empty -----
        # Not allowed, references 2 fields, 0 empty
        (
            "allow empty 1",
            {"f1": "1", "f2": "2", "f3": ""},
            {"f3": "{{f1}} {{f2}}"},
            {"f3": p("1 2")},
            {},
        ),
        # Not allowed, references 2 fields, 1 empty
        (
            "allow empty 2",
            {"f1": "1", "f2": "", "f3": ""},
            {"f3": "{{f1}} {{f2}}"},
            {"f3": ""},
            {},
        ),
        # Allowed, references 2 fields, 1 empty
        (
            "allow empty 3",
            {"f1": "1", "f2": "", "f3": ""},
            {"f3": "{{f1}} {{f2}}"},
            {"f3": p("1 ")},
            {
                "allow_empty": True,
            },
        ),
        # Allowed, references 2 field, both empty
        (
            "allow empty 4",
            {"f1": "", "f2": "", "f3": ""},
            {"f3": "{{f1}} {{f2}}"},
            {"f3": ""},
            {
                "allow_empty": True,
            },
        ),
        # Allowed, references 1 field, empty
        (
            "allow empty 5",
            {"f1": "", "f2": ""},
            {"f2": "{{f1}}"},
            {"f2": ""},
            {
                "allow_empty": True,
            },
        ),
        # Chained, 1 empty, not allow empty
        # f1 -> f2 -> f3
        # f4 -> f5 ---^
        (
            "chained, not allowing empty",
            {"f1": "1", "f2": "", "f3": "", "f4": "", "f5": ""},
            {"f2": "{{f1}}", "f3": "{{f2}} {{f5}}", "f5": "{{f4}}"},
            {"f2": p("1"), "f3": "", "f4": "", "f5": ""},
            {
                "allow_empty": False,
            },
        ),
        # Chained, allow empty
        # f1 -> f2 -> f3
        # f4 -> f5 ---^
        (
            "chained, allowing empty",
            {"f1": "1", "f2": "", "f3": "", "f4": "", "f5": ""},
            {"f2": "{{f1}}", "f3": "{{f2}} {{f5}}", "f5": "{{f4}}"},
            {"f2": p("1"), "f3": p(p("1") + " "), "f4": "", "f5": ""},
            {
                "allow_empty": True,
            },
        ),
        # ----- Target Field ------
        # Target field specified, only that field is updated
        (
            "target only updates target",
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            {"f3": p("1"), "f4": "", "f1": "1"},
            {
                "target_field": "f3",
            },
        ),
        # with target should NOT overwrite prior fields
        (
            "chained target overwrite doesn't overwrite prior fields",
            {"f1": "1", "f2": "old", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": "old", "f3": p("old")},
            {"target_field": "f3"},
        ),
        # Target field specified, always regenerates
        (
            "target always regenerates even if filled",
            {"f1": "1", "f2": "2", "f3": "OLD", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            {"f3": p("1"), "f4": ""},
            {
                "target_field": "f3",
            },
        ),
        # ----- Manual fields ---------
        # Manual field not generated
        (
            "manual not generated",
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            {"f3": "", "f4": p("2")},
            {
                "f3": {"manual": True},
            },
        ),
        # Manual field + target is generated
        (
            "manual + target is generated",
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            {"f3": p("1"), "f4": ""},
            {
                "target_field": "f3",
                "f3": {"manual": True},
            },
        ),
        # ------- Chained Prompts ------
        # Simple case
        (
            "chained simple",
            {"f1": "1", "f2": "", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": p("1"), "f3": p(p("1"))},
            {},
        ),
        # Complex chain
        (
            "chained complex",
            {"f1": "1", "f2": "", "f3": "", "f4": "", "f5": ""},
            {
                "f2": "{{f1}}",
                "f3": "{{f2}}",
                "f4": "{{f2}}",
                "f5": "{{f3}} {{f2}} {{f4}}",
            },
            {
                "f2": p("1"),
                "f3": p(p("1")),
                "f4": p(p("1")),
                "f5": p(p(p("1")) + " " + p("1") + " " + p(p("1"))),
            },
            {},
        ),
        # Chain, shouldn't regenerate fields that already exist
        (
            "chain preserves already filled fields",
            {"f1": "1", "f2": "old", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": "old", "f3": p("old")},
            {},
        ),
        # ------ Target Fields ------
        # Generate the target field, it should only update that field + things before
        #             T
        # f1 -> f2 -> f3 -> f4
        #    -> f5 ---^
        #    -> f6
        (
            "chained target updates",
            {"f1": "1", "f2": "", "f3": "", "f4": "", "f5": "", "f6": ""},
            {
                "f2": "{{f1}}",
                "f3": "{{f2}} {{f5}}",
                "f4": "{{f3}}",
                "f5": "{{f1}}",
                "f6": "{{f1}}",
            },
            {
                "f2": p("1"),
                "f5": p("1"),
                "f3": p(p("1") + " " + p("1")),
                "f4": "",
                "f6": "",
            },
            {
                "target_field": "f3",
            },
        ),
        # ------ Chained manual ------
        # Behavior:
        # - A) If no target field is specified, the manual field is not generated
        #   and should stop the chain
        # - B) If a target field is specified, the manual field is generated if it's
        #   anywhere BEFORE the target field
        #
        # A) case where manual field stops the chain
        #       .     X     X
        # f1 -> f2 -> f3 -> f2
        #             M
        (
            "chained manual stops chain",
            {"f1": "1", "f2": "", "f3": "old", "f4": "old"},
            {"f2": "{{f1}}", "f3": "{{f2}}", "f4": "{{f3}}"},
            {"f2": p("1"), "f3": "old", "f4": "old"},
            {
                "f3": {"manual": True},
            },
        ),
        (
            "manual dependency with value keeps chain alive",
            {"f1": "1", "f2": "existing", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": "existing", "f3": p("existing")},
            {
                "f2": {"manual": True},
            },
        ),
        (
            "manual dependency without value still stops chain",
            {"f1": "1", "f2": "", "f3": ""},
            {"f2": "{{f1}}", "f3": "{{f2}}"},
            {"f2": "", "f3": ""},
            {
                "f2": {"manual": True},
            },
        ),
        # B)case
        # Self is target, should generate self + any manual BEFORE self
        # f1 -> f2 -> f3 -> f4 -> f5 -> f6
        #       M     MT          M
        (
            "chained manual before target is generated",
            {"f1": "1", "f2": "", "f3": "old", "f4": "old", "f5": "old", "f6": "old"},
            {"f2": "{{f1}}", "f3": "{{f2}}", "f4": "{{f3}}"},
            {
                "f2": p("1"),
                "f3": p(p("1")),
                "f4": "old",
                "f5": "old",
                "f6": "old",
            },
            {
                "f2": {"manual": True},
                "f3": {"manual": True},
                "f5": {"manual": True},
                "target_field": "f3",
            },
        ),
        # LEFT OFF:
        # Overwrite! I think this is the last real one?
        # TODO: next chains + overwrite
        # TODO: error handling?
    ],
)
async def test_processor_1(name, note, prompts_map, expected, options, monkeypatch):
    overwrite_fields = bool(options.get("overwrite"))
    target_field = options.get("target_field")
    allow_empty_fields = bool(options.get("allow_empty"))

    n = MockNote(note_type=NOTE_TYPE_NAME, data=note)
    p = setup_data(  # type: ignore
        monkeypatch=monkeypatch,
        note=n,
        prompts_map=prompts_map,
        options=options,
        allow_empty_fields=allow_empty_fields,
    )

    result = await p._process_note(
        make_snapshot(n),
        overwrite_fields=overwrite_fields,
        target_field=target_field,
    )

    for field, expected_value in expected.items():
        actual = result.updates.get(field, n[field])
        assert actual == expected_value, (
            f"{name}: Field {field} is {actual}, expected {expected_value}"
        )


"""
test_cycle Parameters:
    note: dict[str, str] - Note field data
    prompts_map: dict[str, str] - Field prompts that may contain cycles
    expected: bool - Whether a cycle should be detected

Example: ({"f1": "1", "f2": ""}, {"f2": "{{f1}} {{f4}}", "f4": "{{f2}}"}, True)
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note, prompts_map, expected",
    [
        # No cycle
        (
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f3": "{{f1}}", "f4": "{{f2}}"},
            False,
        ),
        # Cycle
        # f1 -> f2 -> f3 -> f4
        # .     ^-----------|
        (
            {"f1": "1", "f2": "2", "f3": "", "f4": ""},
            {"f2": "{{f1}} {{f4}}", "f3": "{{f2}}", "f4": "{{f3}}"},
            True,
        ),
        # Diamond shaped DAG - no cycle
        # f1 -> f2 -> f4
        # f1 -> f3 -> f4
        (
            {"f1": "1", "f2": "", "f3": "", "f4": ""},
            {"f2": "{{f1}}", "f3": "{{f1}}", "f4": "{{f2}} {{f3}}"},
            False,
        ),
    ],
)
async def test_cycle(note, prompts_map, expected, monkeypatch):
    import src.dag
    import src.prompts

    n = MockNote(note_type=NOTE_TYPE_NAME, data=note)

    # Set up the same mocks that setup_data does for prompts
    extras = {
        field: {"automatic": True, "type": "chat", "use_custom_model": False}
        for field in prompts_map
    }
    prompts_data = {
        "note_types": {NOTE_TYPE_NAME: {"1": {"fields": prompts_map, "extras": extras}}}
    }
    c = MockConfig(prompts_map=prompts_data, allow_empty_fields=False)

    monkeypatch.setattr(src.prompts, "config", c)
    monkeypatch.setattr(
        src.prompts,
        "get_prompts_for_note",
        lambda note_type, deck_id, override_prompts_map=None: prompts_data[
            "note_types"
        ][note_type]["1"]["fields"],
    )

    snapshot = make_snapshot(n)
    dag = src.dag.generate_fields_dag(
        note_type=snapshot.note_type,
        field_order=snapshot.field_order,
        values=snapshot.lower_values(),
        deck_id=snapshot.deck_id,
        overwrite_fields=True,
    )
    cycle = src.dag.has_cycle(dag)
    assert cycle == expected


"""
test_returns_if_updated Parameters:
    note: dict[str, str] - Note field data
    prompts_map: dict[str, str] - Field prompts
    expected: bool - Whether the note should be marked as updated

Example: ({"f1": "1", "f2": ""}, {"f2": "{{f1}}"}, True)
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note, prompts_map, expected",
    [
        (
            {"f1": "1", "f2": ""},
            {"f2": "{{f1}}"},
            True,
        ),
        (
            {"f1": "1", "f2": "1"},
            {"f2": "{{f1}}"},
            False,
        ),
    ],
)
async def test_returns_if_updated(note, prompts_map, expected, monkeypatch):
    n = MockNote(note_type=NOTE_TYPE_NAME, data=note)
    p = setup_data(  # type: ignore
        monkeypatch=monkeypatch,
        note=n,
        prompts_map=prompts_map,
        options={},
        allow_empty_fields=False,
    )

    result = await p._process_note(  # type: ignore
        make_snapshot(n),
        overwrite_fields=False,
        target_field=None,
    )
    assert result.did_update == expected


@pytest.mark.asyncio
async def test_process_note_reports_field_failures(monkeypatch):
    import src.prompts

    note = MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "1", "f2": "", "f3": ""})
    processor = setup_data(  # type: ignore
        monkeypatch=monkeypatch,
        note=note,
        prompts_map={"f2": "{{f1}}", "f3": "{{f2}}"},
        options={},
        allow_empty_fields=False,
    )

    async def fail_resolve(
        node: Any,
        *,
        values: dict[str, str],
        show_error_box: bool = False,
        usage_scope_id: str | None = None,
        **context: Any,
    ) -> Any:
        from src.note_processor import ResolvedField

        del show_error_box, usage_scope_id, context
        if node.field == "f2":
            raise TimeoutError()

        interpolated = src.prompts.interpolate_prompt_with_values(
            node.input,
            values,
            src.prompts.config.allow_empty_fields,
        )
        return ResolvedField(p(interpolated) if interpolated else None)

    processor.field_processor.resolve = fail_resolve

    result = await processor._process_note(  # type: ignore
        make_snapshot(note),
        overwrite_fields=False,
        target_field=None,
    )

    assert result.did_update is False
    assert result.updated_fields == []
    assert len(result.field_failures) == 1
    assert result.field_failures[0].field == "f2"
    assert result.field_failures[0].error == "TimeoutError"
    assert result.field_failures[0].aborted_dependents == ("f3",)


def test_openai_batch_preflight_recommends_fast_note_count(
    monkeypatch,
    tmp_path,
) -> None:
    import src.config

    note = MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "seed", "f2": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"f2": "Prompt {{f1}}"},
        {},
        allow_empty_fields=False,
    )
    tracker = setup_tracker(monkeypatch, tmp_path)

    src.config.config.openai_daily_token_budget_enabled = True
    src.config.config.openai_daily_token_budget = 5_000
    tracker.finalize_request(
        provider="openai",
        model="gpt-5.4",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=10,
        raw_usage={"input_tokens": 2_000, "output_tokens": 0},
    )

    snapshots = [
        make_snapshot(MockNote(note_type=NOTE_TYPE_NAME, data={"f1": value, "f2": ""}))
        for value in ("one", "two", "three")
    ]
    preflight = processor.estimate_openai_batch_preflight_for_snapshots(snapshots)

    assert preflight is not None
    assert preflight.note_count == 3
    assert preflight.request_count == 3
    assert preflight.recommended_note_count == 1
    assert preflight.estimated_total_tokens > preflight.fast_budget_tokens


def test_openai_batch_preflight_counts_chained_openai_fields(
    monkeypatch,
    tmp_path,
) -> None:
    import src.config

    note = MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "source", "f2": "", "f3": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"f2": "Summarize {{f1}}", "f3": "Expand {{f2}}"},
        {},
        allow_empty_fields=False,
    )
    setup_tracker(monkeypatch, tmp_path)

    src.config.config.openai_daily_token_budget_enabled = True
    src.config.config.openai_daily_token_budget = 10_000

    estimated_requests = processor.estimate_openai_requests_for_snapshot(
        make_snapshot(note)
    )
    assert len(estimated_requests) == 2


def test_estimated_generated_field_value_uses_visible_response_chars(
    monkeypatch,
    tmp_path,
) -> None:
    from src.dag import generate_fields_dag

    note = MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "source", "f2": "", "f3": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"f2": "Summarize {{f1}}", "f3": "Expand {{f2}}"},
        {},
        allow_empty_fields=False,
    )
    tracker = setup_tracker(monkeypatch, tmp_path)
    snapshot = make_snapshot(note)
    dag = generate_fields_dag(
        note_type=snapshot.note_type,
        field_order=snapshot.field_order,
        values=snapshot.lower_values(),
        overwrite_fields=False,
        deck_id=snapshot.deck_id,
    )
    node = dag["f2"]
    prompt = "Summarize {{f1}}"

    tracker.finalize_request(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=16,
        raw_usage={"input_tokens": 200, "output_tokens": 4_000},
        response_chars=96,
        prompt_key=build_prompt_usage_key(NOTE_TYPE_NAME, 1, "f2"),
        prompt_signature=build_prompt_usage_signature(
            prompt=prompt,
            provider="openai",
            model="gpt-4o-mini",
            reasoning_effort=None,
            use_tools=False,
        ),
        record_openai_daily_usage=False,
    )

    assert (
        len(
            processor.estimated_generated_field_value(
                note_type=NOTE_TYPE_NAME,
                node=node,
                prompt=prompt,
            )
        )
        == 96
    )


def test_snapshot_note_copies_fields() -> None:
    note = MockNote(NOTE_TYPE_NAME, {"Front": "original", "Back": ""})

    snapshot = make_snapshot(note, deck_id=7)
    note["Front"] = "changed"

    assert snapshot.note_id == note.id
    assert snapshot.deck_id == 7
    assert snapshot.note_type == NOTE_TYPE_NAME
    assert snapshot.field_order == ("Front", "Back")
    assert snapshot.fields == {"Front": "original", "Back": ""}


class CommitNote:
    def __init__(self, note_id: int, fields: dict[str, str]) -> None:
        self.id = note_id
        self.fields = fields

    def __getitem__(self, field: str) -> str:
        return self.fields[field]

    def __setitem__(self, field: str, value: str) -> None:
        self.fields[field] = value


class CommitMedia:
    def __init__(self, written_names: Optional[dict[str, str]] = None) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.written_names = written_names or {}

    def write_data(self, filename: str, data: bytes) -> str:
        self.writes.append((filename, data))
        return self.written_names.get(filename, filename)


class CommitCollection:
    def __init__(
        self,
        note: CommitNote,
        written_names: Optional[dict[str, str]] = None,
    ) -> None:
        self.note = note
        self.media = CommitMedia(written_names)
        self.updated_notes: list[CommitNote] = []
        self.changes = object()

    def get_note(self, note_id: int) -> CommitNote:
        assert note_id == self.note.id
        return self.note

    def update_notes(self, notes: list[CommitNote]) -> object:
        self.updated_notes = notes
        return self.changes


def test_commit_processed_notes_applies_unchanged_result() -> None:
    from src.note_processor import PendingMedia, ProcessedNote, commit_processed_notes

    note = CommitNote(1, {"Front": "source", "Back": "", "Extra": "edited"})
    collection = CommitCollection(note)
    result = ProcessedNote(
        note_id=1,
        original_values={"Front": "source", "Back": ""},
        updates={"Back": "generated"},
        media=[PendingMedia("generated.mp3", b"audio")],
        field_failures=[],
    )

    outcome = commit_processed_notes(collection, [result])

    assert outcome.committed == [1]
    assert outcome.conflicted == []
    assert outcome.changes is collection.changes
    assert note.fields == {
        "Front": "source",
        "Back": "generated",
        "Extra": "edited",
    }
    assert collection.media.writes == [("generated.mp3", b"audio")]
    assert collection.updated_notes == [note]


def test_commit_processed_notes_uses_written_media_filenames() -> None:
    from src.note_processor import PendingMedia, ProcessedNote, commit_processed_notes

    note = CommitNote(1, {"Audio": "", "Image": ""})
    collection = CommitCollection(
        note,
        {
            "Animecards-audio.wav": "animecards-audio.wav",
            "Animecards-image.webp": "Animecards-image-1.webp",
        },
    )
    result = ProcessedNote(
        note_id=1,
        original_values={"Audio": "", "Image": ""},
        updates={
            "Audio": "[sound:Animecards-audio.wav]",
            "Image": '<img src="Animecards-image.webp"/>',
        },
        media=[
            PendingMedia("Animecards-audio.wav", b"audio"),
            PendingMedia("Animecards-image.webp", b"image"),
        ],
        field_failures=[],
    )

    commit_processed_notes(collection, [result])

    assert note.fields == {
        "Audio": "[sound:animecards-audio.wav]",
        "Image": '<img src="Animecards-image-1.webp"/>',
    }


def test_commit_processed_notes_skips_conflicted_result() -> None:
    from src.note_processor import PendingMedia, ProcessedNote, commit_processed_notes

    note = CommitNote(1, {"Front": "user edit", "Back": ""})
    collection = CommitCollection(note)
    result = ProcessedNote(
        note_id=1,
        original_values={"Front": "source", "Back": ""},
        updates={"Back": "generated"},
        media=[PendingMedia("generated.mp3", b"audio")],
        field_failures=[],
    )

    outcome = commit_processed_notes(collection, [result])

    assert outcome.committed == []
    assert outcome.conflicted == [1]
    assert note.fields == {"Front": "user edit", "Back": ""}
    assert collection.media.writes == []
    assert collection.updated_notes == []


class BrokenCommitCollection(CommitCollection):
    def get_note(self, note_id: int) -> CommitNote:
        raise RuntimeError("database failed")


def test_commit_processed_notes_propagates_collection_failure() -> None:
    from src.note_processor import ProcessedNote, commit_processed_notes

    collection = BrokenCommitCollection(CommitNote(1, {}))
    result = ProcessedNote(1, {}, {"Back": "generated"}, [], [])

    with pytest.raises(RuntimeError, match="database failed"):
        commit_processed_notes(collection, [result])


@pytest.mark.asyncio
async def test_process_note_returns_deferred_media_without_mutating_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.note_processor import PendingMedia, ResolvedField

    note = MockNote(NOTE_TYPE_NAME, {"Front": "source", "Back": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"Back": "{{Front}}"},
        {},
        allow_empty_fields=False,
    )
    media = PendingMedia("note-back.mp3", b"audio")

    async def resolve(node: Any, **kwargs: Any) -> Any:
        assert "note" not in kwargs
        assert kwargs["note_id"] == note.id
        assert kwargs["note_type"] == NOTE_TYPE_NAME
        assert kwargs["values"] == {"front": "source", "back": ""}
        return ResolvedField("[sound:note-back.mp3]", media)

    processor.field_processor.resolve = resolve
    result = await processor._process_note(  # pyright: ignore[reportPrivateUsage]
        make_snapshot(note)
    )

    assert note["Back"] == ""
    assert result.updates == {"Back": "[sound:note-back.mp3]"}
    assert result.media == [media]
    assert result.original_values == {"Front": "source", "Back": ""}


@pytest.mark.asyncio
async def test_batch_cancellation_stops_scheduling_new_notes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio
    from types import SimpleNamespace

    import src.note_processor as note_processor
    from src.note_processor import MAX_ACTIVE_NOTE_TASKS, NoteSnapshot

    note = MockNote(NOTE_TYPE_NAME, {"Front": "source", "Back": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"Back": "{{Front}}"},
        {},
        allow_empty_fields=False,
    )
    processor.config.debug = False
    setup_tracker(monkeypatch, tmp_path)

    snapshots = [
        NoteSnapshot(
            note_id=note_id,
            deck_id=1,
            deck_name=None,
            note_type=NOTE_TYPE_NAME,
            field_order=("Front", "Back"),
            fields={"Front": "source", "Back": ""},
        )
        for note_id in range(MAX_ACTIVE_NOTE_TASKS * 2)
    ]
    captured: dict[str, Any] = {}

    class FakeProgressDialog:
        instance: "FakeProgressDialog | None" = None

        def __init__(self, label: str, max_val: int, on_cancel: Any) -> None:
            del label, max_val
            self.on_cancel = on_cancel
            self.closed = False
            FakeProgressDialog.instance = self

        def show(self) -> None:
            pass

        def set_label(self, _: str) -> None:
            pass

        def disable_cancel(self) -> None:
            pass

        def set_maximum(self, _: int) -> None:
            pass

        def set_value(self, _: int) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    async def query(_: Any) -> list[NoteSnapshot]:
        return snapshots

    def capture_background(
        operation: Any,
        on_success: Any,
        on_failure: Any,
    ) -> None:
        captured.update(
            operation=operation,
            on_success=on_success,
            on_failure=on_failure,
        )

    started = 0

    async def slow_process(*_: Any, **__: Any) -> Any:
        nonlocal started
        started += 1
        if started == 1:
            assert FakeProgressDialog.instance is not None
            FakeProgressDialog.instance.on_cancel()
        await asyncio.Event().wait()

    monkeypatch.setattr(note_processor, "mw", SimpleNamespace(col=object()))
    monkeypatch.setattr(note_processor, "ProgressDialog", FakeProgressDialog)
    monkeypatch.setattr(note_processor, "query_collection", query)
    monkeypatch.setattr(note_processor, "run_on_main", lambda operation: operation())
    monkeypatch.setattr(note_processor, "bump_usage_counter", lambda: None)
    monkeypatch.setattr(
        note_processor,
        "run_async_in_background_with_sentry",
        capture_background,
    )
    monkeypatch.setattr(
        note_processor.provider_runtime, "get_metrics_summary", lambda: {}
    )
    monkeypatch.setattr(processor, "_process_note", slow_process)

    processor.process_cards_with_progress([1], on_success=None)
    stats = await captured["operation"]()

    assert stats.was_cancelled
    assert 1 <= started <= MAX_ACTIVE_NOTE_TASKS
    assert started < len(snapshots)
    assert not stats.updated
    assert not stats.partial
    assert not stats.failed
    assert not stats.blocked
    assert not stats.unchanged
    assert not stats.conflicted

    captured["on_success"](stats)
    assert not processor.req_in_progress
    assert FakeProgressDialog.instance is not None
    assert FakeProgressDialog.instance.closed


def test_process_card_failure_releases_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import src.note_processor as note_processor

    note = MockNote(NOTE_TYPE_NAME, {"Front": "source", "Back": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"Back": "{{Front}}"},
        {},
        allow_empty_fields=False,
    )
    captured: dict[str, Any] = {}
    failure_seen: list[Exception] = []

    class FakeCard:
        did = 1

        def note(self) -> MockNote:
            return note

    def capture_background(
        operation: Any,
        on_success: Any,
        on_failure: Any,
    ) -> None:
        captured.update(
            operation=operation,
            on_success=on_success,
            on_failure=on_failure,
        )

    decks = SimpleNamespace(name=lambda _: None)
    monkeypatch.setattr(
        note_processor, "mw", SimpleNamespace(col=SimpleNamespace(decks=decks))
    )
    monkeypatch.setattr(
        note_processor,
        "run_async_in_background_with_sentry",
        capture_background,
    )
    monkeypatch.setattr(processor, "_handle_failure", lambda _: None)

    card: Any = FakeCard()
    processor.process_card(
        card,
        show_progress=False,
        on_failure=failure_seen.append,
    )
    assert processor.req_in_progress

    error = RuntimeError("failed")
    captured["on_failure"](error)

    assert failure_seen == [error]
    assert not processor.req_in_progress


def test_estimated_generated_field_value_defaults_without_response_chars(
    monkeypatch,
    tmp_path,
) -> None:
    from src.dag import generate_fields_dag

    note = MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "source", "f2": "", "f3": ""})
    processor = setup_data(
        monkeypatch,
        note,
        {"f2": "Summarize {{f1}}", "f3": "Expand {{f2}}"},
        {},
        allow_empty_fields=False,
    )
    tracker = setup_tracker(monkeypatch, tmp_path)
    snapshot = make_snapshot(note)
    dag = generate_fields_dag(
        note_type=snapshot.note_type,
        field_order=snapshot.field_order,
        values=snapshot.lower_values(),
        overwrite_fields=False,
        deck_id=snapshot.deck_id,
    )
    node = dag["f2"]
    prompt = "Summarize {{f1}}"

    tracker.finalize_request(
        provider="openai",
        model="gpt-4o-mini",
        reasoning_effort=None,
        use_tools=False,
        prompt_chars=16,
        raw_usage={"input_tokens": 200, "output_tokens": 4_000},
        prompt_key=build_prompt_usage_key(NOTE_TYPE_NAME, 1, "f2"),
        prompt_signature=build_prompt_usage_signature(
            prompt=prompt,
            provider="openai",
            model="gpt-4o-mini",
            reasoning_effort=None,
            use_tools=False,
        ),
        record_openai_daily_usage=False,
    )

    assert (
        len(
            processor.estimated_generated_field_value(
                note_type=NOTE_TYPE_NAME,
                node=node,
                prompt=prompt,
            )
        )
        == 256
    )
