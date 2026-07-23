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

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional, Union

from anki.cards import Card, CardId
from anki.decks import DeckId
from anki.notes import Note, NoteId
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
from .config import Config, bump_usage_counter, key_or_config_val
from .dag import generate_fields_dag
from .field_processor import FieldProcessor
from .logger import logger
from .models import (
    DEFAULT_EXTRAS,
    ChatModels,
    ChatProviders,
    OpenAIReasoningEffort,
)
from .nodes import FieldNode
from .notes import get_note_type
from .prompts import get_extras, get_prompts_for_note, interpolate_prompt_with_values
from .provider_runtime import (
    ProviderHTTPError,
    format_provider_http_error_for_log,
    provider_runtime,
)
from .sentry import run_async_in_background_with_sentry
from .ui.ui_utils import show_message_box
from .utils import run_on_main


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
class NoteProcessingResult:
    did_update: bool
    updated_fields: list[str]
    field_failures: list[FieldFailureDetail]


@dataclass
class BatchStatistics:
    processed: list[Note]
    partial: list[Note]
    failed: list[Note]
    blocked: list[Note]
    no_updates: list[Note]
    updated_fields: set[str]
    error_details: dict[int, str]
    field_error_details: dict[int, list[FieldFailureDetail]]
    start_time: float
    end_time: float
    db_writes: int
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
    ) -> Optional[OpenAIBatchPreflight]:
        if not mw or not mw.col or not self.config.openai_daily_token_budget_enabled:
            return None

        cards = [mw.col.get_card(card_id) for card_id in card_ids]
        cards = list({card.nid: card for card in cards}.values())
        notes_with_decks = [(mw.col.get_note(card.nid), card.did) for card in cards]
        return self.estimate_openai_batch_preflight_for_notes(
            notes_with_decks,
            overwrite_fields=overwrite_fields,
        )

    def estimate_openai_batch_preflight_for_notes(
        self,
        notes_with_decks: Sequence[tuple[Note, DeckId]],
        overwrite_fields: bool = False,
    ) -> Optional[OpenAIBatchPreflight]:
        if not self.config.openai_daily_token_budget_enabled:
            return None

        budget_limit = int(self.config.openai_daily_token_budget or 1_000_000)
        daily_usage = chat_usage_tracker.get_openai_daily_usage()
        remaining_tokens = max(0, budget_limit - daily_usage.used_total_tokens)

        note_token_totals: list[int] = []
        request_count = 0
        estimated_total_tokens = 0
        max_request_tokens_by_transport: dict[str, int] = {}

        for note, deck_id in notes_with_decks:
            estimated_requests = self.estimate_openai_requests_for_note(
                note,
                deck_id=deck_id,
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

    def estimate_openai_requests_for_note(
        self,
        note: Note,
        *,
        deck_id: DeckId,
        overwrite_fields: bool = False,
    ) -> list[EstimatedOpenAIRequest]:
        note_type = get_note_type(note)
        dag = generate_fields_dag(
            note,
            overwrite_fields=overwrite_fields,
            deck_id=deck_id,
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
        simulated_values = {
            field.lower(): str(value or "") for field, value in note.items()
        }
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
                        note_type=note_type,
                        node=node,
                        prompt=node.input,
                        interpolated_prompt=simulated_prompt,
                    )
                    if estimated_request is not None:
                        estimated_requests.append(estimated_request)

                    simulated_values[node.field] = self.estimated_generated_field_value(
                        note_type=note_type,
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
    ) -> Optional[EstimatedOpenAIRequest]:
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
        chat_reasoning_effort: Optional[OpenAIReasoningEffort] = key_or_config_val(
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
        chat_reasoning_effort: Optional[OpenAIReasoningEffort] = key_or_config_val(
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
        on_success: Optional[Callable[[BatchStatistics], None]],
        overwrite_fields: bool = False,
    ) -> None:
        """Process notes in the background with a non-modal progress dialog."""

        if not mw or not mw.col:
            return

        bump_usage_counter()
        cards = [mw.col.get_card(card_in) for card_in in card_ids]

        # If a card appears multiple times in the same deck, process it just a single time
        cards = list({card.nid: card for card in cards}.values())

        note_ids = [card.nid for card in cards]
        did_map = {card.nid: card.did for card in cards}

        if not self._assert_preconditions():
            return

        logger.debug("Processing notes...")
        logger.debug("Scheduling %d note workers", len(note_ids))

        cancellation_state = {"cancelled": False}

        def on_cancel() -> None:
            cancellation_state["cancelled"] = True
            logger.info("Cancellation requested")
            if progress:
                progress.set_label(
                    "Cancelling... please wait for active tasks to finish."
                )
                progress.disable_cancel()

        progress = ProgressDialog(
            f"✨Generating... (0/{len(note_ids)})", len(note_ids), on_cancel
        )
        progress.show()
        usage_scope_id = chat_usage_tracker.open_scope()

        # Capture logs
        log_handler = ListHandler()
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        )
        logger.addHandler(log_handler)

        def wrapped_on_success(res: BatchStatistics) -> None:
            logger.removeHandler(log_handler)

            if progress:
                progress.close()
            self._release_request()
            if on_success:
                on_success(res)

        def on_failure(e: Exception) -> None:
            logger.removeHandler(log_handler)
            chat_usage_tracker.close_scope(usage_scope_id)

            if progress:
                progress.close()
            self._release_request()
            show_message_box(f"Error: {e}")

        def on_update(
            updated: list[Note], processed_count: int, finished: bool
        ) -> None:
            if not mw or not mw.col:
                return

            if updated:
                mw.col.update_notes(updated)

            if not finished:
                if progress:
                    progress.set_value(processed_count)
                    if not cancellation_state["cancelled"]:
                        progress.set_label(
                            f"✨ Generating... ({processed_count}/{len(note_ids)})"
                        )
            else:
                logger.info("Finished processing all notes")
                if progress:
                    progress.set_value(len(note_ids))

        async def op():
            start_time = time.time()

            total_processed = []
            total_partial = []
            total_failed = []
            total_blocked = []
            total_no_updates = []
            all_updated_fields: set[str] = set()
            error_details: dict[int, str] = {}
            field_error_details: dict[int, list[FieldFailureDetail]] = {}

            update_buffer: list[Note] = []
            processed_count = 0
            db_writes = 0

            async def worker(
                nid: NoteId,
            ) -> tuple[Optional[Note], Union[NoteProcessingResult, Exception]]:
                if cancellation_state["cancelled"]:
                    return (
                        None,
                        NoteProcessingResult(
                            did_update=False,
                            updated_fields=[],
                            field_failures=[],
                        ),
                    )

                try:
                    # Note: Accessing mw.col in background thread.
                    # This avoids main-thread blocking but is technically unsafe in Anki.
                    # However, since we only read here and write on main thread, it is generally stable.
                    note = mw.col.get_note(nid)

                    # Filter
                    note_type = get_note_type(note)
                    prompts = get_prompts_for_note(note_type, did_map[nid])
                    if not prompts:
                        return (
                            note,
                            NoteProcessingResult(
                                did_update=False,
                                updated_fields=[],
                                field_failures=[],
                            ),
                        )

                    result = await self._process_note(
                        note,
                        deck_id=did_map[nid],
                        overwrite_fields=overwrite_fields,
                        usage_scope_id=usage_scope_id,
                    )
                    return (note, result)

                except Exception as e:
                    # Try to retrieve note just for reporting purposes
                    try:
                        n = mw.col.get_note(nid)
                        return (n, e)
                    except Exception:
                        return (None, e)

            active_tasks = [asyncio.create_task(worker(nid)) for nid in note_ids]

            for completed_task in asyncio.as_completed(active_tasks):
                # If cancelled, force-cancel all active tasks instead of waiting
                if cancellation_state["cancelled"]:
                    if active_tasks:
                        logger.info(f"Cancelling {len(active_tasks)} active tasks...")
                        for active_task in active_tasks:
                            active_task.cancel()
                        # Wait for cancelled tasks to finish (with timeout)
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*active_tasks, return_exceptions=True),
                                timeout=5.0,  # Give tasks 5 seconds to clean up
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Some tasks did not cancel cleanly within timeout"
                            )
                    break
                processed_count += 1
                try:
                    note_obj, status = await completed_task
                except asyncio.CancelledError:
                    logger.debug("Task was cancelled")
                    continue
                except Exception as e:
                    note_obj, status = (None, e)

                if isinstance(status, Exception):
                    if note_obj:
                        total_failed.append(note_obj)
                        error_details[note_obj.id] = describe_exception(status)
                        logger.error(
                            f"Error processing note {note_obj.id}: {describe_exception(status)}"
                        )
                    else:
                        logger.error(
                            f"Error processing note: {describe_exception(status)}"
                        )

                else:
                    if note_obj is None:
                        continue

                    if status.field_failures:
                        field_error_details[note_obj.id] = status.field_failures

                    if status.did_update:
                        update_buffer.append(note_obj)
                        all_updated_fields.update(status.updated_fields)
                        if status.field_failures:
                            total_partial.append(note_obj)
                        else:
                            total_processed.append(note_obj)
                    elif status.field_failures:
                        total_blocked.append(note_obj)
                    else:
                        total_no_updates.append(note_obj)

                batch_to_update = []
                if len(update_buffer) >= 100:
                    batch_to_update = update_buffer[:]
                    update_buffer.clear()

                if batch_to_update:
                    db_writes += 1
                run_on_main(
                    lambda u=batch_to_update, p=processed_count: on_update(u, p, False)
                )

            # Final flush
            if update_buffer:
                db_writes += 1
                run_on_main(
                    lambda u=update_buffer, p=processed_count: on_update(u, p, True)
                )
            else:
                run_on_main(lambda u=[], p=processed_count: on_update(u, p, True))

            end_time = time.time()
            transport_metrics = provider_runtime.get_metrics_summary()

            # Retrieve logs from the handler
            logs = list(log_handler.logs)

            return BatchStatistics(
                processed=total_processed,
                partial=total_partial,
                failed=total_failed,
                blocked=total_blocked,
                no_updates=total_no_updates,
                updated_fields=all_updated_fields,
                error_details=error_details,
                field_error_details=field_error_details,
                start_time=start_time,
                end_time=end_time,
                db_writes=db_writes,
                transport_metrics=transport_metrics,
                logs=logs,
                chat_usage_summary=chat_usage_tracker.close_scope(usage_scope_id),
                was_cancelled=cancellation_state["cancelled"],
            )

        try:
            run_async_in_background_with_sentry(
                op, wrapped_on_success, on_failure, with_progress=False
            )
        except Exception as e:
            logger.removeHandler(log_handler)
            chat_usage_tracker.close_scope(usage_scope_id)
            if progress:
                progress.close()
            self._release_request()
            raise e

    def process_card(
        self,
        card: Card,
        show_progress: bool,
        overwrite_fields: bool = False,
        on_success: Callable[[bool], None] = lambda _: None,
        on_failure: Optional[Callable[[Exception], None]] = None,
        target_field: Optional[str] = None,
        on_field_update: Optional[Callable[[], None]] = None,
    ):
        """Process a single note, filling in fields with prompts from the user"""
        if not self._assert_preconditions():
            return

        note = card.note()

        def wrapped_on_success(updated: bool) -> None:
            # Save the note if it was updated
            if updated and mw and mw.col:
                mw.col.update_note(note)

            self._release_request()
            on_success(updated)

        def wrapped_failure(e: Exception) -> None:
            self._handle_failure(e)
            self._release_request()
            if on_failure:
                on_failure(e)

        # NOTE: for some reason i can't run bump_usage_counter in this hook without causing a
        # an PyQT crash, so I'm running it in the on_success callback instead
        run_async_in_background_with_sentry(
            lambda: self._process_note(
                note,
                overwrite_fields=overwrite_fields,
                deck_id=card.did,
                target_field=target_field,
                on_field_update=on_field_update,
                show_progress=show_progress,
                usage_scope_id=None,
            ),
            lambda result: wrapped_on_success(result.did_update),
            wrapped_failure,
        )

    # Note: one quirk is that if overwrite_fields = True AND there's a target field,
    # it will regenerate any fields up until the target field. A bit weird but
    # this combination of values doesn't really make sense anyways so it's probably fine.
    # Would be better modeled with some mode switch or something.
    async def _process_note(
        self,
        note: Note,
        deck_id: DeckId,
        overwrite_fields: bool = False,
        target_field: Optional[str] = None,
        on_field_update: Optional[Callable[[], None]] = None,
        show_progress: bool = False,
        usage_scope_id: Optional[str] = None,
    ) -> NoteProcessingResult:
        """Process a single note and return updated fields plus any field-level failures."""

        note_type = get_note_type(note)
        prompts_for_note = get_prompts_for_note(note_type, deck_id)

        if not prompts_for_note:
            logger.debug("no prompts found for note type")
            return NoteProcessingResult(
                did_update=False,
                updated_fields=[],
                field_failures=[],
            )

        # Topsort + parallel process the DAG
        dag = generate_fields_dag(
            note,
            target_field=target_field,
            overwrite_fields=overwrite_fields,
            deck_id=deck_id,
        )

        did_update = False
        updated_fields: list[str] = []
        field_failures: list[FieldFailureDetail] = []

        will_show_progress = show_progress and len(dag)
        if will_show_progress:
            run_on_main(
                lambda: mw.progress.start(  # type: ignore
                    label="✨ Generating...",
                    min=0,
                    max=len(dag),
                    immediate=True,
                )
            )

        try:
            while len(dag):
                next_batch: list[FieldNode] = [
                    node for node in dag.values() if not node.in_nodes
                ]
                logger.debug(f"Processing next nodes: {[n.field for n in next_batch]}")
                batch_tasks = {
                    node.field: self._process_node(
                        # Only show the error box for the target field
                        node,
                        note,
                        show_error_message_box=node.is_target,
                        usage_scope_id=usage_scope_id,
                    )
                    for node in next_batch
                }

                responses = await asyncio.gather(
                    *batch_tasks.values(), return_exceptions=True
                )

                for field, response in zip(batch_tasks.keys(), responses):
                    node = dag[field]

                    # Handle field-level exceptions gracefully
                    if isinstance(response, Exception):
                        failure_detail = FieldFailureDetail(
                            field=field,
                            error=describe_exception(response),
                            aborted_dependents=tuple(
                                sorted(out_node.field for out_node in node.out_nodes)
                            ),
                        )
                        field_failures.append(failure_detail)
                        logger.warning(
                            f"Field '{field}' failed: {failure_detail.summary()}. Continuing with other fields."
                        )
                        # Mark node as aborted so downstream fields are skipped
                        node.abort = True
                        # Remove from DAG and continue processing other fields
                        for out_node in node.out_nodes:
                            out_node.in_nodes.remove(node)
                            # Mark downstream nodes as aborted since their dependency failed
                            out_node.abort = True
                        dag.pop(field)
                        continue

                    if response is not None:
                        current_val = note[node.field_upper]
                        if response != current_val:
                            logger.debug(
                                f"Updating field {field} with response: {response}"
                            )
                            note[node.field_upper] = response
                        else:
                            # Use trace instead of debug to reduce noise
                            # logger.debug(f"Field {field} unchanged")
                            pass

                    for out_node in node.out_nodes:
                        out_node.in_nodes.remove(node)

                    # New notes have ID 0 and don't exist in the DB yet, so can't be updated!
                    if note.id and node.did_update:
                        did_update = True
                        updated_fields.append(node.field)
                        # Note: we do NOT update DB here anymore. Caller must handle persistence.
                        # This improves performance and safety in batch operations.

                    dag.pop(field)
                    if on_field_update:
                        run_on_main(on_field_update)

        finally:
            if will_show_progress:
                run_on_main(lambda: mw.progress.finish())  # type: ignore

        return NoteProcessingResult(
            did_update=did_update,
            updated_fields=updated_fields,
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
        note: Note,
        show_error_message_box: bool,
        usage_scope_id: Optional[str] = None,
    ) -> Optional[str]:
        started_at = time.perf_counter()
        status = "completed"

        try:
            if node.abort:
                return None

            value = note[node.field_upper]

            if node.manual and not (node.is_target or node.generate_despite_manual):
                if value:
                    return value
                node.abort = True
                logger.debug(f"Skipping field {node.field}")
                return None

            if value and not (node.is_target or node.overwrite):
                return value

            new_value = await self.field_processor.resolve(
                node,
                note,
                show_error_box=show_error_message_box,
                usage_scope_id=usage_scope_id,
            )
            if new_value:
                node.did_update = True

            return new_value
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
