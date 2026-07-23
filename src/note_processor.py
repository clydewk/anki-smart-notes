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
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aqt import mw
from aqt.qt import QDialog, QLabel, QProgressBar, QPushButton, Qt, QVBoxLayout

from .chat_provider import text_initial_window, text_transport_key
from .chat_usage import (
    EMPTY_CHAT_RUN_USAGE_SUMMARY,
    ChatRunUsageSummary,
    build_prompt_usage_key,
    build_prompt_usage_signature,
    chat_usage_tracker,
)
from .collection_ops import mutate_collection, query_collection
from .config import Config, bump_usage_counter, key_or_config_val
from .dag import generate_fields_dag
from .logger import logger
from .models import (
    DEFAULT_EXTRAS,
    ChatModels,
    ChatProviders,
    OpenAIReasoningEffort,
)
from .notes import get_note_type
from .prompts import (
    get_extras,
    get_prompt_fields,
    get_prompts_for_note,
    interpolate_prompt_with_values,
)
from .provider_runtime import (
    ProviderHTTPError,
    format_provider_http_error_for_log,
    provider_runtime,
)
from .sentry import run_async_in_background_with_sentry
from .ui.ui_utils import show_message_box
from .utils import run_on_main

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from anki.cards import Card, CardId
    from anki.decks import DeckId
    from anki.notes import Note, NoteId

    from .field_processor import FieldProcessor
    from .nodes import FieldNode


@dataclass(frozen=True)
class PendingMedia:
    filename: str
    data: bytes


@dataclass(frozen=True)
class ResolvedField:
    value: str | None
    media: PendingMedia | None = None


@dataclass(frozen=True)
class NoteSnapshot:
    note_id: NoteId
    deck_id: DeckId
    deck_name: str | None
    note_type: str
    field_order: tuple[str, ...]
    fields: dict[str, str]

    def lower_values(self) -> dict[str, str]:
        return {field.lower(): value for field, value in self.fields.items()}


@dataclass(frozen=True)
class FieldFailureDetail:
    field: str
    error: str
    aborted_dependents: tuple[str, ...] = ()

    def summary(self) -> str:
        if not self.aborted_dependents:
            return f"{self.field}: {self.error}"

        blocked = ", ".join(self.aborted_dependents)
        return f"{self.field}: {self.error}. Blocked downstream: {blocked}"


@dataclass(frozen=True)
class ProcessedNote:
    note_id: NoteId
    original_values: dict[str, str]
    updates: dict[str, str]
    media: list[PendingMedia]
    field_failures: list[FieldFailureDetail]

    @property
    def did_update(self) -> bool:
        return bool(self.updates or self.media)

    @property
    def updated_fields(self) -> list[str]:
        return [field.lower() for field in self.updates]


@dataclass(frozen=True)
class CommitResult:
    committed: list[NoteId]
    conflicted: list[NoteId]
    changes: Any


@dataclass
class BatchStatistics:
    updated: list[NoteId]
    partial: list[NoteId]
    failed: dict[NoteId, str]
    blocked: dict[NoteId, list[FieldFailureDetail]]
    unchanged: list[NoteId]
    conflicted: list[NoteId]
    updated_fields: set[str]
    start_time: float
    end_time: float
    commit_count: int
    transport_metrics: dict[str, dict[str, float]]
    logs: list[str]
    chat_usage_summary: ChatRunUsageSummary = EMPTY_CHAT_RUN_USAGE_SUMMARY
    was_cancelled: bool = False


@dataclass(frozen=True)
class OpenAIBatchPreflight:
    note_count: int
    request_count: int
    estimated_total_tokens: int
    remaining_tokens: int
    fast_budget_tokens: int
    recommended_note_count: int


@dataclass(frozen=True)
class EstimatedOpenAIRequest:
    transport_key: str
    estimated_tokens: int


COMMIT_CHUNK_SIZE = 50


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            self.handleError(record)


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    error_type = type(error).__name__
    return f"{error_type}: {message}" if message else error_type


def snapshot_note(
    note: Note,
    deck_id: DeckId,
    deck_name: str | None = None,
) -> NoteSnapshot:
    note_type = get_note_type(note)
    fields = {field: str(value or "") for field, value in note.items()}
    return NoteSnapshot(
        note_id=note.id,
        deck_id=deck_id,
        deck_name=deck_name,
        note_type=note_type,
        field_order=tuple(fields),
        fields=fields,
    )


def snapshot_cards(collection: Any, card_ids: Sequence[CardId]) -> list[NoteSnapshot]:
    cards = [collection.get_card(card_id) for card_id in card_ids]
    unique_cards = {card.nid: card for card in cards}.values()
    return [
        snapshot_note(
            collection.get_note(card.nid),
            card.did,
            collection.decks.name(card.did),
        )
        for card in unique_cards
    ]


def commit_processed_notes(
    collection: Any,
    results: Sequence[ProcessedNote],
) -> CommitResult:
    """Apply unchanged results in one short collection operation."""
    notes: list[Note] = []
    committed: list[NoteId] = []
    conflicted: list[NoteId] = []

    for result in results:
        note = collection.get_note(result.note_id)
        try:
            unchanged = all(
                str(note[field] or "") == original
                for field, original in result.original_values.items()
            )
        except KeyError:
            unchanged = False

        if not unchanged:
            conflicted.append(result.note_id)
            continue

        for pending_media in result.media:
            collection.media.write_data(pending_media.filename, pending_media.data)
        for field, value in result.updates.items():
            note[field] = value

        notes.append(note)
        committed.append(result.note_id)

    changes = collection.update_notes(notes)
    return CommitResult(
        committed=committed,
        conflicted=conflicted,
        changes=changes,
    )


class ProgressDialog(QDialog):
    def __init__(self, label: str, max_val: int, on_cancel: Callable[[], None]):
        super().__init__(mw)
        self.setWindowTitle("Smart Notes")
        # NonModal prevents blocking other windows
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(400)
        # Prevent focus stealing - show without activating
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        layout = QVBoxLayout()

        self.label = QLabel(label)
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setRange(0, max_val)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(on_cancel)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def set_value(self, val: int) -> None:
        self.bar.setValue(val)

    def set_maximum(self, maximum: int) -> None:
        self.bar.setMaximum(maximum)

    def set_label(self, text: str) -> None:
        self.label.setText(text)

    def disable_cancel(self) -> None:
        self.cancel_button.setEnabled(False)


class NoteProcessor:
    def __init__(self, field_processor: FieldProcessor, config: Config):
        self.field_processor = field_processor
        self.config = config
        self.req_in_progress = False

    def get_openai_batch_preflight(
        self,
        card_ids: Sequence[CardId],
        overwrite_fields: bool = False,
    ) -> OpenAIBatchPreflight | None:
        if not mw or not mw.col or not self.config.openai_daily_token_budget_enabled:
            return None

        snapshots = snapshot_cards(mw.col, card_ids)
        return self.estimate_openai_batch_preflight_for_snapshots(
            snapshots,
            overwrite_fields=overwrite_fields,
        )

    def estimate_openai_batch_preflight_for_snapshots(
        self,
        snapshots: Sequence[NoteSnapshot],
        overwrite_fields: bool = False,
    ) -> OpenAIBatchPreflight | None:
        if not self.config.openai_daily_token_budget_enabled:
            return None

        budget_limit = int(self.config.openai_daily_token_budget or 1_000_000)
        daily_usage = chat_usage_tracker.get_openai_daily_usage()
        remaining_tokens = max(0, budget_limit - daily_usage.used_total_tokens)

        note_token_totals: list[int] = []
        request_count = 0
        estimated_total_tokens = 0
        max_request_tokens_by_transport: dict[str, int] = {}

        for snapshot in snapshots:
            estimated_requests = self.estimate_openai_requests_for_snapshot(
                snapshot,
                overwrite_fields=overwrite_fields,
            )
            note_total_tokens = sum(
                request.estimated_tokens for request in estimated_requests
            )
            note_token_totals.append(note_total_tokens)
            request_count += len(estimated_requests)
            estimated_total_tokens += note_total_tokens

            for request in estimated_requests:
                max_request_tokens_by_transport[request.transport_key] = max(
                    max_request_tokens_by_transport.get(request.transport_key, 0),
                    request.estimated_tokens,
                )

        if request_count <= 0:
            return None

        slowdown_buffer_tokens = sum(
            text_initial_window("openai") * max_tokens
            for max_tokens in max_request_tokens_by_transport.values()
        )
        fast_budget_tokens = max(0, remaining_tokens - slowdown_buffer_tokens)

        running_total = 0
        recommended_note_count = 0
        for note_total_tokens in note_token_totals:
            if running_total + note_total_tokens > fast_budget_tokens:
                break
            running_total += note_total_tokens
            recommended_note_count += 1

        if estimated_total_tokens <= fast_budget_tokens:
            return None

        return OpenAIBatchPreflight(
            note_count=len(note_token_totals),
            request_count=request_count,
            estimated_total_tokens=estimated_total_tokens,
            remaining_tokens=remaining_tokens,
            fast_budget_tokens=fast_budget_tokens,
            recommended_note_count=recommended_note_count,
        )

    def estimate_openai_requests_for_snapshot(
        self,
        snapshot: NoteSnapshot,
        *,
        overwrite_fields: bool = False,
    ) -> list[EstimatedOpenAIRequest]:
        values = snapshot.lower_values()
        dag = generate_fields_dag(
            note_type=snapshot.note_type,
            field_order=snapshot.field_order,
            values=values,
            overwrite_fields=overwrite_fields,
            deck_id=snapshot.deck_id,
        )
        if not dag:
            return []

        pending_inputs = {
            field: {in_node.field for in_node in node.in_nodes}
            for field, node in dag.items()
        }
        ready_fields = [
            field for field, in_fields in pending_inputs.items() if not in_fields
        ]
        simulated_values = values.copy()
        estimated_requests: list[EstimatedOpenAIRequest] = []

        while ready_fields:
            field = ready_fields.pop(0)
            node = dag[field]

            if not self.should_skip_node_for_estimate(node, simulated_values):
                simulated_prompt = interpolate_prompt_with_values(
                    node.input,
                    simulated_values,
                    self.config.allow_empty_fields,
                )
                if simulated_prompt is not None:
                    estimated_request = self.estimate_openai_request_for_node(
                        note_type=snapshot.note_type,
                        node=node,
                        prompt=node.input,
                        interpolated_prompt=simulated_prompt,
                    )
                    if estimated_request is not None:
                        estimated_requests.append(estimated_request)

                    simulated_values[node.field] = self.estimated_generated_field_value(
                        note_type=snapshot.note_type,
                        node=node,
                        prompt=node.input,
                    )

            for out_node in node.out_nodes:
                pending_inputs[out_node.field].discard(node.field)
                if not pending_inputs[out_node.field]:
                    ready_fields.append(out_node.field)

        return estimated_requests

    def should_skip_node_for_estimate(
        self,
        node: FieldNode,
        simulated_values: dict[str, str],
    ) -> bool:
        if node.manual and not (node.is_target or node.generate_despite_manual):
            return True

        current_value = simulated_values.get(node.field, "")
        return bool(current_value and not (node.is_target or node.overwrite))

    def estimate_openai_request_for_node(
        self,
        *,
        note_type: str,
        node: FieldNode,
        prompt: str,
        interpolated_prompt: str,
    ) -> EstimatedOpenAIRequest | None:
        if node.field_type != "chat":
            return None

        extras = (
            get_extras(
                note_type=note_type,
                field=node.field,
                deck_id=node.deck_id,
                fallback_to_global_deck=True,
            )
            or DEFAULT_EXTRAS
        )
        chat_provider: ChatProviders = key_or_config_val(extras, "chat_provider")
        if chat_provider != "openai":
            return None

        chat_model: ChatModels = key_or_config_val(extras, "chat_model")
        chat_reasoning_effort: OpenAIReasoningEffort | None = key_or_config_val(
            extras, "chat_reasoning_effort"
        )
        use_tools: bool = key_or_config_val(extras, "chat_use_tools")
        prompt_key = build_prompt_usage_key(
            note_type=note_type,
            deck_id=int(node.deck_id),
            field_lower=node.field,
        )
        prompt_signature = build_prompt_usage_signature(
            prompt=prompt,
            provider=str(chat_provider),
            model=str(chat_model),
            reasoning_effort=chat_reasoning_effort,
            use_tools=use_tools,
        )
        usage = chat_usage_tracker.estimate_request_usage(
            provider=str(chat_provider),
            model=str(chat_model),
            reasoning_effort=chat_reasoning_effort,
            use_tools=use_tools,
            prompt_chars=len(interpolated_prompt),
            prompt_bytes=len(interpolated_prompt.encode("utf-8")),
            prompt_key=prompt_key,
            prompt_signature=prompt_signature,
            mode="planning",
        )
        return EstimatedOpenAIRequest(
            transport_key=text_transport_key("openai", str(chat_model)),
            estimated_tokens=usage.total_tokens,
        )

    def estimated_generated_field_value(
        self,
        *,
        note_type: str,
        node: FieldNode,
        prompt: str,
    ) -> str:
        if node.field_type == "image":
            return '<img src="generated"/>'
        if node.field_type == "tts":
            return "[sound:generated]"
        if node.field_type != "chat":
            return "generated"

        extras = (
            get_extras(
                note_type=note_type,
                field=node.field,
                deck_id=node.deck_id,
                fallback_to_global_deck=True,
            )
            or DEFAULT_EXTRAS
        )
        chat_provider: ChatProviders = key_or_config_val(extras, "chat_provider")
        chat_model: ChatModels = key_or_config_val(extras, "chat_model")
        chat_reasoning_effort: OpenAIReasoningEffort | None = key_or_config_val(
            extras, "chat_reasoning_effort"
        )
        use_tools: bool = key_or_config_val(extras, "chat_use_tools")
        prompt_signature = build_prompt_usage_signature(
            prompt=prompt,
            provider=str(chat_provider),
            model=str(chat_model),
            reasoning_effort=chat_reasoning_effort,
            use_tools=use_tools,
        )
        snapshot = chat_usage_tracker.get_prompt_usage(
            note_type=note_type,
            deck_id=int(node.deck_id),
            field_lower=node.field,
            signature=prompt_signature,
        )

        estimated_chars = 256
        if snapshot is not None and snapshot.avg_response_chars is not None:
            estimated_chars = round(snapshot.avg_response_chars)
        else:
            runtime_snapshot = chat_usage_tracker.get_runtime_profile_usage(
                provider=str(chat_provider),
                model=str(chat_model),
                reasoning_effort=chat_reasoning_effort,
                use_tools=use_tools,
            )
            if (
                runtime_snapshot is not None
                and runtime_snapshot.avg_response_chars is not None
            ):
                estimated_chars = round(runtime_snapshot.avg_response_chars)
        estimated_chars = max(32, min(estimated_chars, 2048))
        return "x" * estimated_chars

    def process_cards_with_progress(
        self,
        card_ids: Sequence[CardId],
        on_success: Callable[[BatchStatistics], None] | None,
        overwrite_fields: bool = False,
    ) -> None:
        """Process detached note snapshots with a non-modal progress dialog."""
        if not mw or not mw.col or not self._assert_preconditions():
            return

        bump_usage_counter()
        requested_card_ids = tuple(card_ids)
        total_notes = len(requested_card_ids)
        cancel_event = threading.Event()

        def on_cancel() -> None:
            cancel_event.set()
            logger.info("Cancellation requested")
            progress.set_label("Cancelling... please wait for active tasks to finish.")
            progress.disable_cancel()

        progress = ProgressDialog(
            f"✨ Generating... (0/{total_notes})",
            total_notes,
            on_cancel,
        )
        progress.show()
        usage_scope_id = chat_usage_tracker.open_scope()

        log_handler = ListHandler() if self.config.debug else None
        if log_handler:
            log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(message)s")
            )
            logger.addHandler(log_handler)

        def finish() -> None:
            if log_handler:
                logger.removeHandler(log_handler)
            progress.close()
            self._release_request()

        def wrapped_on_success(stats: BatchStatistics) -> None:
            finish()
            if on_success:
                on_success(stats)

        def on_failure(error: Exception) -> None:
            chat_usage_tracker.close_scope(usage_scope_id)
            finish()
            show_message_box(f"Error: {error}")

        def update_progress(processed_count: int, finished: bool = False) -> None:
            progress.set_value(processed_count)
            if finished:
                logger.info(
                    "Batch processing cancelled"
                    if cancel_event.is_set()
                    else "Finished processing all notes"
                )
            elif not cancel_event.is_set():
                progress.set_label(
                    f"✨ Generating... ({processed_count}/{total_notes})"
                )

        async def op() -> BatchStatistics:
            nonlocal total_notes
            start_time = time.time()
            snapshots = await query_collection(
                lambda collection: snapshot_cards(collection, requested_card_ids)
            )
            total_notes = len(snapshots)
            run_on_main(lambda: progress.set_maximum(total_notes))

            updated: list[NoteId] = []
            partial: list[NoteId] = []
            failed: dict[NoteId, str] = {}
            blocked: dict[NoteId, list[FieldFailureDetail]] = {}
            unchanged: list[NoteId] = []
            conflicted: list[NoteId] = []
            updated_fields: set[str] = set()
            update_buffer: list[ProcessedNote] = []
            processed_count = 0
            commit_count = 0

            async def commit_chunk(results: list[ProcessedNote]) -> None:
                nonlocal commit_count
                if not results:
                    return

                outcome = await mutate_collection(
                    lambda collection: commit_processed_notes(collection, results)
                )
                if outcome.committed:
                    commit_count += 1
                committed = set(outcome.committed)
                conflicted.extend(outcome.conflicted)

                for result in results:
                    if result.note_id not in committed:
                        continue
                    updated_fields.update(result.updated_fields)
                    if result.field_failures:
                        partial.append(result.note_id)
                    else:
                        updated.append(result.note_id)

            async def worker(
                snapshot: NoteSnapshot,
            ) -> tuple[NoteId, ProcessedNote | Exception]:
                if cancel_event.is_set():
                    return (
                        snapshot.note_id,
                        ProcessedNote(
                            note_id=snapshot.note_id,
                            original_values={},
                            updates={},
                            media=[],
                            field_failures=[],
                        ),
                    )

                try:
                    return (
                        snapshot.note_id,
                        await self._process_note(
                            snapshot,
                            overwrite_fields=overwrite_fields,
                            usage_scope_id=usage_scope_id,
                        ),
                    )
                except Exception as error:
                    return snapshot.note_id, error

            active_tasks = {
                asyncio.create_task(worker(snapshot)) for snapshot in snapshots
            }

            while active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks,
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for completed_task in done:
                    note_id, status = completed_task.result()
                    processed_count += 1

                    if isinstance(status, Exception):
                        failed[note_id] = describe_exception(status)
                        logger.error(
                            "Error processing note %s: %s",
                            note_id,
                            failed[note_id],
                        )
                    elif status.did_update:
                        update_buffer.append(status)
                    elif status.field_failures:
                        blocked[note_id] = status.field_failures
                    else:
                        unchanged.append(note_id)

                    if len(update_buffer) >= COMMIT_CHUNK_SIZE:
                        chunk = update_buffer[:COMMIT_CHUNK_SIZE]
                        del update_buffer[:COMMIT_CHUNK_SIZE]
                        await commit_chunk(chunk)

                    run_on_main(lambda count=processed_count: update_progress(count))

                if cancel_event.is_set():
                    for active_task in active_tasks:
                        active_task.cancel()
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*active_tasks, return_exceptions=True),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Some tasks did not cancel cleanly within timeout"
                        )
                    break

            await commit_chunk(update_buffer)
            run_on_main(lambda count=processed_count: update_progress(count, True))

            return BatchStatistics(
                updated=updated,
                partial=partial,
                failed=failed,
                blocked=blocked,
                unchanged=unchanged,
                conflicted=conflicted,
                updated_fields=updated_fields,
                start_time=start_time,
                end_time=time.time(),
                commit_count=commit_count,
                transport_metrics=provider_runtime.get_metrics_summary(),
                logs=list(log_handler.logs) if log_handler else [],
                chat_usage_summary=chat_usage_tracker.close_scope(usage_scope_id),
                was_cancelled=cancel_event.is_set(),
            )

        try:
            run_async_in_background_with_sentry(op, wrapped_on_success, on_failure)
        except Exception:
            chat_usage_tracker.close_scope(usage_scope_id)
            finish()
            raise

    def process_card(
        self,
        card: Card,
        show_progress: bool,
        overwrite_fields: bool = False,
        on_success: Callable[[bool], None] = lambda _: None,
        on_failure: Callable[[Exception], None] | None = None,
        target_field: str | None = None,
        on_field_update: Callable[[], None] | None = None,
    ) -> None:
        """Process one detached snapshot and commit it through the collection worker."""
        if not self._assert_preconditions():
            return

        deck_name = mw.col.decks.name(card.did) if mw and mw.col else None
        snapshot = snapshot_note(card.note(), card.did, deck_name)
        if show_progress and mw:
            mw.progress.start(label="✨ Generating...", immediate=True)

        def wrapped_on_success(
            result: tuple[ProcessedNote, CommitResult | None],
        ) -> None:
            processed, committed = result
            did_update = bool(committed and committed.committed)
            if committed and committed.conflicted:
                logger.info("Skipped note %s because it changed", processed.note_id)
            if on_field_update and did_update:
                on_field_update()
            if show_progress and mw:
                mw.progress.finish()

            self._release_request()
            on_success(did_update)

        def wrapped_failure(error: Exception) -> None:
            if show_progress and mw:
                mw.progress.finish()
            self._handle_failure(error)
            self._release_request()
            if on_failure:
                on_failure(error)

        async def op() -> tuple[ProcessedNote, CommitResult | None]:
            result = await self._process_note(
                snapshot,
                overwrite_fields=overwrite_fields,
                target_field=target_field,
                usage_scope_id=None,
            )
            committed = None
            if result.did_update:
                committed = await mutate_collection(
                    lambda collection: commit_processed_notes(collection, [result])
                )
            return result, committed

        run_async_in_background_with_sentry(op, wrapped_on_success, wrapped_failure)

    async def _process_note(
        self,
        snapshot: NoteSnapshot,
        overwrite_fields: bool = False,
        target_field: str | None = None,
        usage_scope_id: str | None = None,
    ) -> ProcessedNote:
        """Generate field and media updates from a detached note snapshot."""
        values = snapshot.lower_values()
        prompts_for_note = get_prompts_for_note(
            snapshot.note_type,
            snapshot.deck_id,
        )
        if not prompts_for_note:
            logger.debug("no prompts found for note type")
            return ProcessedNote(
                note_id=snapshot.note_id,
                original_values={},
                updates={},
                media=[],
                field_failures=[],
            )

        dag = generate_fields_dag(
            note_type=snapshot.note_type,
            field_order=snapshot.field_order,
            values=values,
            target_field=target_field,
            overwrite_fields=overwrite_fields,
            deck_id=snapshot.deck_id,
        )
        canonical_fields = {field.lower(): field for field in snapshot.field_order}
        relevant_fields = set(dag)
        for node in dag.values():
            relevant_fields.update(get_prompt_fields(node.input))

        original_values = {
            canonical_fields[field]: values.get(field, "")
            for field in relevant_fields
            if field in canonical_fields
        }
        updates: dict[str, str] = {}
        media: list[PendingMedia] = []
        field_failures: list[FieldFailureDetail] = []

        while dag:
            next_batch = [node for node in dag.values() if not node.in_nodes]
            logger.debug("Processing next nodes: %s", [n.field for n in next_batch])
            batch_tasks = {
                node.field: self._process_node(
                    node,
                    snapshot,
                    values,
                    show_error_message_box=node.is_target,
                    usage_scope_id=usage_scope_id,
                )
                for node in next_batch
            }
            responses = await asyncio.gather(
                *batch_tasks.values(),
                return_exceptions=True,
            )

            for field, response in zip(batch_tasks, responses):
                node = dag[field]
                if isinstance(response, BaseException):
                    failure_detail = FieldFailureDetail(
                        field=field,
                        error=describe_exception(response),
                        aborted_dependents=tuple(
                            sorted(out_node.field for out_node in node.out_nodes)
                        ),
                    )
                    field_failures.append(failure_detail)
                    logger.warning(
                        "Field '%s' failed: %s. Continuing with other fields.",
                        field,
                        failure_detail.summary(),
                    )
                    node.abort = True
                    for out_node in node.out_nodes:
                        out_node.in_nodes.remove(node)
                        out_node.abort = True
                    dag.pop(field)
                    continue

                if response.value is not None:
                    values[node.field] = response.value
                if node.did_update and response.value is not None:
                    updates[node.field_upper] = response.value
                    if response.media:
                        media.append(response.media)

                for out_node in node.out_nodes:
                    out_node.in_nodes.remove(node)
                dag.pop(field)

        return ProcessedNote(
            note_id=snapshot.note_id,
            original_values=original_values,
            updates=updates,
            media=media,
            field_failures=field_failures,
        )

    def _handle_failure(self, e: Exception) -> None:
        logger.debug("Handling failure")

        # Simplified error handling for BYOK
        if isinstance(e, ProviderHTTPError):
            status = e.status
            logger.error(
                "Processing failed with provider HTTP error: %s",
                format_provider_http_error_for_log(e),
            )

            error_map = {
                401: "Smart Notes Error: 401 Unauthorized. Please check your API Key in settings.",
                404: "Smart Notes Error: 404 Not Found. Please check your model configuration.",
                429: "Smart Notes Error: 429 Rate Limit Exceeded. You are sending too many requests too quickly.",
                402: "Smart Notes Error: 402 Payment Required. Please check your provider's billing status.",
            }

            msg = error_map.get(
                status, f"Smart Notes Error: HTTP {status} - {e.message}"
            )
            show_message_box(msg)
        else:
            logger.error(f"Got non-HTTP error: {e}")
            show_message_box(f"Smart Notes Error: {e}")

    def _assert_preconditions(self) -> bool:
        no_existing_req = self.assert_no_req_in_process()
        return no_existing_req

    def assert_no_req_in_process(self) -> bool:
        if self.req_in_progress:
            logger.info("A request is already in progress.")
            return False

        self.req_in_progress = True
        return True

    def _release_request(self) -> None:
        self.req_in_progress = False

    async def _process_node(
        self,
        node: FieldNode,
        snapshot: NoteSnapshot,
        values: dict[str, str],
        show_error_message_box: bool,
        usage_scope_id: str | None = None,
    ) -> ResolvedField:
        started_at = time.perf_counter()
        status = "completed"

        try:
            if node.abort:
                return ResolvedField(None)

            value = values.get(node.field, "")
            if node.manual and not (node.is_target or node.generate_despite_manual):
                if value:
                    return ResolvedField(value)
                node.abort = True
                logger.debug("Skipping field %s", node.field)
                return ResolvedField(None)

            if value and not (node.is_target or node.overwrite):
                return ResolvedField(value)

            result = await self.field_processor.resolve(
                node,
                note_id=snapshot.note_id,
                note_type=snapshot.note_type,
                deck_name=snapshot.deck_name,
                field_order=snapshot.field_order,
                values=values,
                show_error_box=show_error_message_box,
                usage_scope_id=usage_scope_id,
            )
            if result.value:
                node.did_update = True
            return result
        except Exception:
            status = "failed"
            raise
        finally:
            duration = time.perf_counter() - started_at
            if duration >= 1.0 or status == "failed":
                logger.debug(
                    "Field %s (%s) %s in %.1fs",
                    node.field,
                    node.field_type,
                    status,
                    duration,
                )
