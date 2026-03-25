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

import pytest

from src.chat_usage import ChatUsageTracker
from tests.mocks import (
    MockConfig,
    MockNote,
    p,
)

NOTE_TYPE_NAME = "note_type_1"


class FakeFieldProcessor:
    async def resolve(self, node, note, show_error_box=False, usage_scope_id=None):
        from src.prompts import interpolate_prompt

        del show_error_box, usage_scope_id

        interpolated = interpolate_prompt(node.input, note)
        if not interpolated:
            return None

        return p(interpolated)


def setup_data(monkeypatch, note, prompts_map, options, allow_empty_fields):
    import src.config
    import src.dag
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

    from src.note_proccessor import NoteProcessor

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

    monkeypatch.setattr(
        src.dag,
        "get_fields",
        lambda _: note.fields(),
    )

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
    monkeypatch.setattr("src.note_proccessor.chat_usage_tracker", tracker)
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

    await p._process_note(
        n, deck_id=1, overwrite_fields=overwrite_fields, target_field=target_field
    )

    for k, v in expected.items():
        assert n[k] == v, f"{name}: Field {k} is {n[k]}, expected {v}"


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

    # Mock get_fields like in setup_data
    monkeypatch.setattr(
        src.dag,
        "get_fields",
        lambda _: n.fields(),
    )

    dag = src.dag.generate_fields_dag(n, deck_id=1, overwrite_fields=True)
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
        n, deck_id=1, overwrite_fields=False, target_field=None
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

    async def fail_resolve(node, note, show_error_box=False, usage_scope_id=None):
        del show_error_box, usage_scope_id
        if node.field == "f2":
            raise TimeoutError()

        interpolated = src.prompts.interpolate_prompt(node.input, note)
        return p(interpolated) if interpolated else None

    processor.field_processor.resolve = fail_resolve

    result = await processor._process_note(  # type: ignore
        note, deck_id=1, overwrite_fields=False, target_field=None
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

    notes_with_decks = [
        (MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "one", "f2": ""}), 1),
        (MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "two", "f2": ""}), 1),
        (MockNote(note_type=NOTE_TYPE_NAME, data={"f1": "three", "f2": ""}), 1),
    ]
    preflight = processor.estimate_openai_batch_preflight_for_notes(notes_with_decks)

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

    estimated_requests = processor.estimate_openai_requests_for_note(note, deck_id=1)
    assert len(estimated_requests) == 2
