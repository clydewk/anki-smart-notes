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
from typing import Optional

from anki.cards import Card, CardId
from anki.decks import DeckId
from anki.notes import Note, NoteId
from aqt import mw
from aqt.qt import QDialog, QLabel, QProgressBar, QPushButton, Qt, QVBoxLayout

from .chat_usage import (
    EMPTY_CHAT_RUN_USAGE_SUMMARY,
    ChatRunUsageSummary,
    chat_usage_tracker,
)
from .config import Config, bump_usage_counter
from .constants import STANDARD_BATCH_LIMIT
from .dag import generate_fields_dag
from .field_processor import FieldProcessor
from .logger import logger
from .nodes import FieldNode
from .notes import get_note_type
from .prompts import get_prompts_for_note
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

    def process_cards_with_progress(
        self,
        card_ids: Sequence[CardId],
        on_success: Optional[Callable[[BatchStatistics], None]],
        overwrite_fields: bool = False,
    ) -> None:
        """Processes notes in the background with a progress bar, batching into a single undo op"""

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

        initial_limit = STANDARD_BATCH_LIMIT
        logger.debug(f"Global concurrency limit: {initial_limit}")

        # Only show fancy progress meter for large batches
        cancellation_state = {"cancelled": False}

        # Disable autosave to avoid "Backing up..." popups during batch processing
        autosave_was_active = False
        if hasattr(mw, "autosaveTimer") and mw.autosaveTimer.isActive():
            mw.autosaveTimer.stop()
            autosave_was_active = True

        # Also try to disable automatic backups if the method exists (Anki 2.1.50+)
        if hasattr(mw.col, "set_autosave_enabled"):
            # Some versions allow disabling at collection level
            mw.col.set_autosave_enabled(False)

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
            stats = res

            logger.removeHandler(log_handler)

            if autosave_was_active and hasattr(mw, "autosaveTimer"):
                mw.autosaveTimer.start()
            if hasattr(mw.col, "set_autosave_enabled"):
                mw.col.set_autosave_enabled(True)

            if progress:
                progress.close()
            if not mw or not mw.col:
                return
            # Note: DB updates are mainly handled during processing to allow incremental progress saving
            # But we might have stragglers or need to finalize things.
            self._reqlinquish_req_in_process()
            if on_success:
                on_success(stats)

        def on_failure(e: Exception) -> None:
            logger.removeHandler(log_handler)
            chat_usage_tracker.close_scope(usage_scope_id)

            if autosave_was_active and hasattr(mw, "autosaveTimer"):
                mw.autosaveTimer.start()
            if hasattr(mw.col, "set_autosave_enabled"):
                mw.col.set_autosave_enabled(True)

            if progress:
                progress.close()
            self._reqlinquish_req_in_process()
            show_message_box(f"Error: {e}")

        def on_update(
            updated: list[Note], processed_count: int, finished: bool
        ) -> None:
            if not mw or not mw.col:
                return

            if updated:
                # Temporarily block Anki's progress manager from showing dialogs
                # to prevent focus-stealing "Processing..." popup during DB writes.
                # mw.progress uses a timer that shows a dialog after ~600ms of main
                # thread blocking, so we suppress it during our update.
                progress_blocked = False
                if hasattr(mw, "progress") and hasattr(mw.progress, "_win"):
                    # If a progress window already exists, don't interfere
                    pass
                elif hasattr(mw, "progress"):
                    # Block the progress manager's timer
                    try:
                        if hasattr(mw.progress, "_timer"):
                            mw.progress._timer.stop()
                            progress_blocked = True
                    except Exception:
                        pass

                try:
                    mw.col.update_notes(updated)
                finally:
                    # Restore progress timer if we blocked it
                    if progress_blocked:
                        try:
                            if hasattr(mw.progress, "_timer"):
                                mw.progress._timer.start()
                        except Exception:
                            pass

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

            concurrency_limit = STANDARD_BATCH_LIMIT

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

            to_process_ids = note_ids[:]
            active_tasks: set[asyncio.Task] = set()

            async def worker(
                nid: NoteId,
            ) -> tuple[Optional[Note], NoteProcessingResult | Exception]:
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

            while to_process_ids or active_tasks:
                # If cancelled, force-cancel all active tasks instead of waiting
                if cancellation_state["cancelled"]:
                    if active_tasks:
                        logger.info(f"Cancelling {len(active_tasks)} active tasks...")
                        for task in active_tasks:
                            task.cancel()
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
                        active_tasks.clear()
                    break

                # Fill the pool
                while to_process_ids and len(active_tasks) < concurrency_limit:
                    if cancellation_state["cancelled"]:
                        break
                    nid = to_process_ids.pop(0)
                    task = asyncio.create_task(worker(nid))
                    active_tasks.add(task)

                if not active_tasks:
                    break

                # Wait for at least one task to finish (with timeout to check cancellation)
                try:
                    done, pending = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=1.0,  # Check cancellation state every second
                    )
                except asyncio.TimeoutError:
                    # No task completed, but check cancellation state
                    continue
                active_tasks = pending

                for task in done:
                    processed_count += 1
                    try:
                        note_obj, status = await task
                    except asyncio.CancelledError:
                        # Task was cancelled - don't count as error
                        logger.debug("Task was cancelled")
                        continue
                    except Exception as e:
                        # Should be caught inside worker, but just in case
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

                    # Flush buffer periodically (DB write)
                    batch_to_update = []
                    # Use larger buffer (100 notes) to reduce DB operations and avoid
                    # triggering Anki's internal "Processing..." dialog which appears
                    # when main thread is blocked for more than ~600ms
                    if len(update_buffer) >= 100:
                        batch_to_update = update_buffer[:]
                        update_buffer.clear()

                    # Update UI/DB
                    # Progress bar updates are cheap (just Qt widget updates)
                    # DB writes are expensive but batched (every 100 notes) and protected
                    # by progress manager suppression, so it's safe to update on every note
                    if batch_to_update:
                        db_writes += 1
                    run_on_main(
                        lambda u=batch_to_update, p=processed_count: on_update(
                            u, p, False
                        )
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
            if autosave_was_active and hasattr(mw, "autosaveTimer"):
                mw.autosaveTimer.start()
            if hasattr(mw.col, "set_autosave_enabled"):
                mw.col.set_autosave_enabled(True)
            if progress:
                progress.close()
            self._reqlinquish_req_in_process()
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

            self._reqlinquish_req_in_process()
            on_success(updated)

        def wrapped_failure(e: Exception) -> None:
            self._handle_failure(e)
            self._reqlinquish_req_in_process()
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
        usage_scope_id: str | None = None,
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

    def _reqlinquish_req_in_process(self) -> None:
        self.req_in_progress = False

    async def _process_node(
        self,
        node: FieldNode,
        note: Note,
        show_error_message_box: bool,
        usage_scope_id: str | None = None,
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
