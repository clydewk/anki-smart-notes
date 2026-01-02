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
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import aiohttp
from anki.cards import Card, CardId
from anki.decks import DeckId
from anki.notes import Note, NoteId
from aqt import mw
from aqt.qt import QDialog, QLabel, QProgressBar, QPushButton, Qt, QVBoxLayout

from .config import Config, bump_usage_counter
from .constants import STANDARD_BATCH_LIMIT
from .dag import generate_fields_dag
from .field_processor import FieldProcessor
from .logger import logger
from .nodes import FieldNode
from .notes import get_note_type
from .prompts import get_prompts_for_note
from .rate_limiter import RateLimitManager
from .sentry import run_async_in_background_with_sentry
from .ui.ui_utils import show_message_box
from .utils import run_on_main

@dataclass
class BatchStatistics:
    processed: list[Note]
    failed: list[Note]
    skipped: list[Note]
    updated_fields: set[str]
    error_details: dict[int, str]
    start_time: float
    end_time: float
    db_writes: int
    rate_limits: dict[str, float]
    logs: list[str]


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


class ProgressDialog(QDialog):
    def __init__(self, label: str, max_val: int, on_cancel: Callable[[], None]):
        super().__init__(mw)
        self.setWindowTitle("Smart Notes")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(400)

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

        # Relax global concurrency limit to allow specialized limiters (per-provider) to control flow.
        # We set it to STANDARD_BATCH_LIMIT to prevent OOM/file descriptor exhaustion, but the actual rate limiting
        # happens inside the providers.
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
            "✨Generating... (0/{})".format(len(note_ids)), len(note_ids), on_cancel
        )
        progress.show()

        # Capture logs
        log_handler = ListHandler()
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
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
            
            # Worker Pool with relaxed global limit
            # We set a high fixed concurrency limit (safeguard) and rely on 
            # provider-specific RPM limiters for actual flow control.
            
            concurrency_limit = STANDARD_BATCH_LIMIT
            
            total_updated = []
            total_failed = []
            total_skipped = []
            all_updated_fields: set[str] = set()
            error_details: dict[int, str] = {}
            
            update_buffer: list[Note] = []
            processed_count = 0
            db_writes = 0
            
            to_process_ids = note_ids[:]
            active_tasks: set[asyncio.Task] = set()

            async def worker(nid: NoteId) -> tuple[Optional[Note], bool | Exception, list[str]]:
                if cancellation_state["cancelled"]:
                    return (None, False, [])
                
                try:
                    # Note: Accessing mw.col in background thread.
                    # This avoids main-thread blocking but is technically unsafe in Anki.
                    # However, since we only read here and write on main thread, it is generally stable.
                    note = mw.col.get_note(nid)
                    
                    # Filter
                    note_type = get_note_type(note)
                    prompts = get_prompts_for_note(note_type, did_map[nid])
                    if not prompts:
                        return (note, False, []) # Treated as skipped
                    
                    # Process
                    # This will internally hit provider-specific rate limiters and block if needed
                    did_update, updated_fields = await self._process_note(
                        note, 
                        deck_id=did_map[nid],
                        overwrite_fields=overwrite_fields
                    )
                    return (note, did_update, updated_fields)
                    
                except Exception as e:
                     # Try to retrieve note just for reporting purposes
                    try:
                        n = mw.col.get_note(nid)
                        return (n, e, [])
                    except:
                        return (None, e, [])

            while to_process_ids or active_tasks:
                if cancellation_state["cancelled"]:
                    # Wait for active tasks to drain then break
                    if not active_tasks:
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
                
                # Wait for at least one task to finish
                done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                active_tasks = pending
                
                for task in done:
                    processed_count += 1
                    try:
                        note_obj, status, u_fields = await task
                    except Exception as e:
                        # Should be caught inside worker, but just in case
                        note_obj, status, u_fields = (None, e, [])

                    if isinstance(status, Exception):
                        if note_obj:
                            total_failed.append(note_obj)
                            error_details[note_obj.id] = str(status)
                            logger.error(f"Error processing note {note_obj.id}: {status}")
                        else:
                            logger.error(f"Error processing note: {status}")
                            
                    else:
                        # Success
                        if status is True:
                            # Note updated
                            total_updated.append(note_obj)
                            update_buffer.append(note_obj)
                            all_updated_fields.update(u_fields)
                        elif status is False:
                            total_skipped.append(note_obj)
                            
                    # Flush buffer periodically (DB write)
                    batch_to_update = []
                    # Increase buffer size to reduce Anki backup/undo creation spam
                    if len(update_buffer) >= 50:
                        batch_to_update = update_buffer[:]
                        update_buffer.clear()
                    
                    # Update UI/DB
                    # We always want to call this if we have DB updates
                    # If no DB updates, we only want to call it occasionally to save UI repaints
                    if batch_to_update or processed_count % 5 == 0:
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
                run_on_main(
                    lambda u=[], p=processed_count: on_update(u, p, True)
                )
            
            end_time = time.time()
            rate_limits = {k: v._rpm for k, v in RateLimitManager.get_instance().limiters.items()}
            
            # Retrieve logs from the handler
            logs = list(log_handler.logs)

            return BatchStatistics(
                processed=total_updated,
                failed=total_failed,
                skipped=total_skipped,
                updated_fields=all_updated_fields,
                error_details=error_details,
                start_time=start_time,
                end_time=end_time,
                db_writes=db_writes,
                rate_limits=rate_limits,
                logs=logs
            )

        try:
            run_async_in_background_with_sentry(
                op, wrapped_on_success, on_failure, with_progress=False
            )
        except Exception as e:
            logger.removeHandler(log_handler)
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
            ),
            wrapped_on_success,
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
    ) -> tuple[bool, list[str]]:
        """Process a single note, returns (updated_status, updated_fields). Optionally can target specific fields. Caller responsible for handling any exceptions."""

        note_type = get_note_type(note)
        prompts_for_note = get_prompts_for_note(note_type, deck_id)

        if not prompts_for_note:
            logger.debug("no prompts found for note type")
            return (False, [])

        # Topsort + parallel process the DAG
        dag = generate_fields_dag(
            note,
            target_field=target_field,
            overwrite_fields=overwrite_fields,
            deck_id=deck_id,
        )
        
        did_update = False
        updated_fields: list[str] = []

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
                    )
                    for node in next_batch
                }

                responses = await asyncio.gather(*batch_tasks.values())

                for field, response in zip(batch_tasks.keys(), responses):
                    node = dag[field]
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

                    if node.abort:
                        for out_node in node.out_nodes:
                            out_node.abort = True

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

        return (did_update, updated_fields)

    def _handle_failure(self, e: Exception) -> None:
        logger.debug("Handling failure")
        
        # Simplified error handling for BYOK
        if isinstance(e, aiohttp.ClientResponseError):
            status = e.status
            logger.debug(f"Got status: {status}")
            
            error_map = {
                401: "Smart Notes Error: 401 Unauthorized. Please check your API Key in settings.",
                404: "Smart Notes Error: 404 Not Found. Please check your model configuration.",
                429: "Smart Notes Error: 429 Rate Limit Exceeded. You are sending too many requests too quickly.",
                402: "Smart Notes Error: 402 Payment Required. Please check your provider's billing status.",
            }
            
            msg = error_map.get(status, f"Smart Notes Error: HTTP {status} - {e.message}")
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
        self, node: FieldNode, note: Note, show_error_message_box: bool
    ) -> Optional[str]:
        if node.abort:
            # logger.debug(f"Skipping field {node.field}")
            return None

        # logger.debug(f"Processing field {node.field}")

        value = note[node.field_upper]

        # If not target and manual, skip
        if node.manual and not (node.is_target or node.generate_despite_manual):
            node.abort = True
            logger.debug(f"Skipping field {node.field}")
            return None

        # Skip it if there's a value and we don't want to overwrite
        if value and not (node.is_target or node.overwrite):
            return value

        new_value = await self.field_processor.resolve(
            node, note, show_error_message_box
        )
        if new_value:
            node.did_update = True

        return new_value
