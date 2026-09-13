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

import os
import subprocess
import sys
from pathlib import Path


def test_real_collection_commit_undo_conflicts_and_unsaved_notes(
    tmp_path: Path,
) -> None:
    script = r"""
import re
import sys
from pathlib import Path
from anki.collection import Collection
from src.note_processor import PendingMedia, ProcessedNote, apply_note_result, commit_processed_notes, snapshot_note

collection = Collection(str(Path(sys.argv[1]) / "collection.anki2"))
try:
    model = collection.models.by_name("Basic")
    note = collection.new_note(model)
    note["Front"] = "source"
    note["Back"] = "original"
    note.tags = ["keep-tag"]
    collection.add_note(note, 1)
    card = note.cards()[0]
    card.queue = 2
    card.type = 2
    card.due = 42
    card.ivl = 30
    card.factor = 2500
    card.reps = 4
    card.lapses = 1
    collection.update_card(card)
    schedule = (card.queue, card.type, card.due, card.ivl, card.factor, card.reps, card.lapses)
    media = Path(collection.media.dir())
    collection.media.write_data("generated.mp3", b"original audio")

    def result_for(note, text, filename=None):
        snapshot = snapshot_note(note, 1)
        return ProcessedNote(
            note_id=note.id, original_values=snapshot.fields, updates={"Back": text},
            media=[PendingMedia(filename, b"generated audio")] if filename else [],
            field_failures=[], note_type_id=snapshot.note_type_id, field_order=snapshot.field_order,
        )

    pending = result_for(note, "[sound:generated.mp3]", "generated.mp3")
    result = commit_processed_notes(collection, [pending])
    assert result.committed == [note.id] and not result.conflicted
    saved = collection.get_note(note.id)
    actual_filename = re.fullmatch(r"\[sound:([^]]+)\]", saved["Back"]).group(1)
    assert actual_filename != "generated.mp3"
    assert (media / actual_filename).read_bytes() == b"generated audio"
    assert (media / "generated.mp3").read_bytes() == b"original audio"
    assert saved["Front"] == "source" and saved.tags == ["keep-tag"]
    current_card = collection.get_card(card.id)
    assert (current_card.queue, current_card.type, current_card.due, current_card.ivl, current_card.factor, current_card.reps, current_card.lapses) == schedule
    assert collection.undo_status().undo
    collection.undo()
    assert collection.get_note(note.id)["Back"] == "original"
    assert collection.get_note(note.id).tags == ["keep-tag"]
    assert (media / "generated.mp3").read_bytes() == b"original audio"

    deleted = collection.new_note(model)
    deleted["Front"] = "deleted"
    collection.add_note(deleted, 1)
    missing_result = result_for(deleted, "[sound:missing.mp3]", "missing.mp3")
    collection.remove_notes([deleted.id])
    valid_result = result_for(collection.get_note(note.id), "saved valid note")
    result = commit_processed_notes(collection, [missing_result, valid_result])
    assert result.conflicted == [deleted.id] and result.committed == [note.id]
    assert collection.get_note(note.id)["Back"] == "saved valid note"
    assert not (media / "missing.mp3").exists()

    note = collection.get_note(note.id)
    stale = result_for(note, "[sound:stale.mp3]", "stale.mp3")
    note["Front"] = "concurrent sync-like edit"
    collection.update_note(note)
    result = commit_processed_notes(collection, [stale])
    assert result.conflicted == [note.id] and not result.committed
    assert collection.get_note(note.id)["Front"] == "concurrent sync-like edit"
    assert collection.get_note(note.id)["Back"] == "saved valid note"
    assert not (media / "stale.mp3").exists()

    stale_schema = result_for(collection.get_note(note.id), "[sound:schema.mp3]", "schema.mp3")
    field = collection.models.new_field("Added field")
    collection.models.add_field(model, field)
    collection.models.save(model)
    result = commit_processed_notes(collection, [stale_schema])
    assert result.conflicted == [note.id] and not result.committed
    assert not (media / "schema.mp3").exists()

    unsaved = collection.new_note(model)
    unsaved["Front"] = "unsaved source"
    pending = result_for(unsaved, "[sound:unsaved.mp3]", "unsaved.mp3")
    count_before = collection.note_count()
    result = commit_processed_notes(collection, [pending])
    assert result.conflicted == [0] and not result.committed
    assert not (media / "unsaved.mp3").exists()
    apply_note_result(collection, unsaved, pending)
    assert unsaved.id == 0 and unsaved["Back"] == "[sound:unsaved.mp3]"
    assert collection.note_count() == count_before
finally:
    collection.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_profiles_with_matching_ids_reject_late_jobs_and_media(
    tmp_path: Path,
) -> None:
    script = """
import asyncio
import queue
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from anki.collection import Collection
import src.collection_ops as collection_ops
import src.tasks as tasks
from src.note_processor import PendingMedia, ProcessedNote, commit_processed_notes, snapshot_note

root = Path(sys.argv[1])
for name in ("A", "B"):
    (root / name).mkdir()
a_path = root / "A" / "collection.anki2"
b_path = root / "B" / "collection.anki2"
collection_a = Collection(str(a_path))
note = collection_a.new_note(collection_a.models.by_name("Basic"))
note["Front"] = "same source"
note["Back"] = "original"
collection_a.add_note(note, 1)
note_id = note.id
collection_a.close()
shutil.copy2(a_path, b_path)
collection_a = Collection(str(a_path))
collection_b = Collection(str(b_path))
callbacks = queue.Queue()
window = SimpleNamespace(col=collection_a, taskman=SimpleNamespace(run_on_main=callbacks.put))
tasks.mw = window
collection_ops.mw = window
tasks.config = SimpleNamespace(snapshot=lambda: {})
outputs = []
try:
    assert collection_a.get_note(note_id).id == collection_b.get_note(note_id).id
    snapshot = snapshot_note(collection_a.get_note(note_id), 1)
    result = ProcessedNote(
        note_id=note_id, original_values=snapshot.fields,
        updates={"Back": "[sound:late.mp3]"}, media=[PendingMedia("late.mp3", b"late audio")],
        field_failures=[], note_type_id=snapshot.note_type_id, field_order=snapshot.field_order,
    )
    async def generate():
        return result
    def save_result(result):
        outputs.append(commit_processed_notes(window.col, [result]))
    tasks.open_profile_worker()
    handle = tasks.run_async_in_background(generate, save_result)
    assert handle.future.result(timeout=2) is result
    late_callback = callbacks.get(timeout=2)
    tasks.close_profile_worker()
    window.col = collection_b
    tasks.open_profile_worker()
    late_callback()
    assert not outputs
    assert collection_b.get_note(note_id)["Back"] == "original"
    assert not (Path(collection_b.media.dir()) / "late.mp3").exists()

    async def commit_later():
        return await collection_ops.mutate_collection(lambda collection: commit_processed_notes(collection, [result]))
    handle = tasks.run_async_in_background(commit_later, outputs.append)
    queued_commit = callbacks.get(timeout=2)
    tasks.close_profile_worker()
    window.col = collection_a
    tasks.open_profile_worker()
    queued_commit()
    assert handle.future.cancelled() and not outputs
    for collection in (collection_a, collection_b):
        assert collection.get_note(note_id)["Back"] == "original"
        assert not (Path(collection.media.dir()) / "late.mp3").exists()
finally:
    tasks.close_profile_worker()
    collection_b.close()
    collection_a.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
