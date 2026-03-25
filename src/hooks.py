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

"""
Setup the hooks for the Anki plugin
"""


from collections.abc import Sequence
from typing import Any, Callable, Optional

from anki.cards import Card
from aqt import QAction, QMenu, browser, editor, gui_hooks, mw
from aqt.addcards import AddCards
from aqt.browser.sidebar.item import SidebarItemType

from .chat_usage import chat_usage_tracker, format_token_count
from .config import bump_usage_counter, config
from .decks import deck_id_to_name_map
from .logger import logger, setup_logger
from .migrations import migrate_models
from .note_proccessor import BatchStatistics, NoteProcessor
from .notes import get_field_from_index, is_ai_field, is_card_fully_processed
from .sentry import with_sentry
from .tasks import run_async_in_background
from .ui.addon_options_dialog import AddonOptionsDialog
from .ui.changelog import perform_update_check
from .ui.field_menu import FieldMenu
from .ui.sparkle import Sparkle
from .ui.ui_utils import show_message_box


def with_processor(fn: Any):
    # Too annoying to type this thing
    """Decorator to pass the processor to the function."""

    def wrapper(processor: NoteProcessor):
        @with_sentry
        def inner(*args: Any, **kwargs: Any):
            return fn(processor, *args, **kwargs)

        return inner

    return wrapper


@with_processor  # type: ignore
def on_options(processor: NoteProcessor):
    dialog = AddonOptionsDialog(processor)
    dialog.exec()


@with_processor  # type: ignore
def add_editor_top_button(
    processor: NoteProcessor, buttons: list[str], e: editor.Editor
):
    @with_sentry
    def fn(editor: editor.Editor):
        if not mw:
            return

        card = editor.card
        note = editor.note

        if note is None:
            logger.error("Unexpectedly found no note")
            return

        # New notes don't have cards yet, fetch into the deck_chooser to get the deckId
        if card is None:
            deck_id: Optional[int] = None
            parent = editor.parentWindow
            # Parent should always be AddCards if there's no card
            if isinstance(parent, AddCards):
                deck_id = parent.deck_chooser.selected_deck_id
                logger.debug(f"Setting deck_id to {deck_id}")
            card = note.ephemeral_card()
            if deck_id:
                card.did = deck_id

        # Imperatively set the button styling and disabled state 🤦‍♂️
        # y u do dis, anki

        def set_button_disabled() -> None:
            if not e or not e.web:
                return
            e.web.eval(
                """
                    (() => {
                        const button = document.querySelector("#generate_smart_fields")
                        button.disabled = true
                        button.style.opacity = 0.25
                    })()
                """
            )

        def set_button_enabled() -> None:
            if not e or not e.web:
                return

            e.web.eval(
                """
                    (() => {
                        const button = document.querySelector("#generate_smart_fields")
                        button.disabled = false
                        button.style.opacity = 1.0
                    })()
                """
            )

        set_button_disabled()

        def reload_note() -> None:
            # Don't reload notes if they don't exist yet
            if note.id:
                note.load()
            editor.loadNote()

            parent = editor.parentWindow
            if isinstance(parent, browser.Browser) and getattr(  # type: ignore
                parent, "_previewer", None
            ):  # type: ignore
                parent._previewer.render_card()  # type: ignore

        def on_success(did_change: bool):
            set_button_enabled()

            if not did_change:
                return

            reload_note()

        def on_field() -> None:
            reload_note()

        is_fully_processed = is_card_fully_processed(card)
        processor.process_card(
            card,
            overwrite_fields=is_fully_processed,
            on_success=on_success,
            on_failure=lambda _: set_button_enabled(),
            on_field_update=on_field,
            show_progress=False,
        )

    button = e.addButton(
        cmd="Generate Smart Fields",
        label="✨",
        func=fn,
        icon=None,
        tip="Ctrl+Shift+G: Generate Smart Fields",
        id="generate_smart_fields",
        keys="Ctrl+Shift+G",
    )

    buttons.append(button)


def make_on_batch_success(
    browser: browser.Browser,  # type: ignore
) -> Callable[[BatchStatistics], None]:
    def wrapped_on_batch_success(stats: BatchStatistics):
        processed = stats.processed
        partial = stats.partial
        errors = stats.failed
        blocked = stats.blocked
        no_updates = stats.no_updates
        updated_fields = stats.updated_fields
        error_details = stats.error_details
        field_error_details = stats.field_error_details
        chat_usage_summary = stats.chat_usage_summary

        browser.on_all_or_selected_rows_changed()

        def pluralize(word: str, count: int) -> str:
            return f"{count} {word}{'s' if count != 1 else ''}"

        debug_info = ""
        was_cancelled = stats.was_cancelled
        updated_count = len(processed) + len(partial)
        completed_count = (
            len(processed) + len(partial) + len(errors) + len(blocked) + len(no_updates)
        )

        if config.debug:
            duration = stats.end_time - stats.start_time
            notes_per_sec = completed_count / duration if duration > 0 else 0

            debug_parts = []
            debug_parts.append("--- Batch Processing Report ---")
            if was_cancelled:
                debug_parts.append("Status: Cancelled")
            debug_parts.append(f"Time Taken: {duration:.2f}s")
            debug_parts.append(f"Processing Speed: {notes_per_sec:.2f} notes/sec")
            debug_parts.append(f"Database Writes: {stats.db_writes}")
            debug_parts.append(f"Updated: {updated_count}")
            debug_parts.append(f"Processed Cleanly: {len(processed)}")
            debug_parts.append(f"Processed With Field Failures: {len(partial)}")
            debug_parts.append(f"Failed: {len(errors)}")
            debug_parts.append(f"Blocked By Field Failures: {len(blocked)}")
            debug_parts.append(f"No Updates: {len(no_updates)}")
            if chat_usage_summary.request_count:
                debug_parts.append(
                    "Chat tokens: "
                    f"{format_token_count(chat_usage_summary.total_tokens)} across "
                    f"{chat_usage_summary.request_count} requests"
                )

            if updated_fields:
                field_list = ", ".join(sorted(updated_fields))
                debug_parts.append(f"Fields updated: {field_list}")

            if stats.transport_metrics:
                debug_parts.append("\n--- Transport Metrics ---")
                for provider, metrics in stats.transport_metrics.items():
                    window = metrics.get("window", 0)
                    inflight = metrics.get("inflight", 0)
                    retries = metrics.get("retries", 0)
                    throttles = metrics.get("throttles", 0)
                    timeouts = metrics.get("timeouts", 0)
                    debug_parts.append(
                        f"{provider}: window={window:.0f}, inflight={inflight:.0f}, "
                        f"retries={retries:.0f}, throttles={throttles:.0f}, "
                        f"timeouts={timeouts:.0f}"
                    )

            if errors:
                debug_parts.append("\n--- Failures ---")
                for note in errors:
                    msg = error_details.get(note.id, "Unknown error")
                    debug_parts.append(f"Note ID {note.id} failed: {msg}")

            if field_error_details:
                debug_parts.append("\n--- Field Failures ---")
                for note_id in sorted(field_error_details):
                    debug_parts.append(f"Note ID {note_id}:")
                    for detail in field_error_details[note_id]:
                        debug_parts.append(f"  - {detail.summary()}")

            if stats.logs:
                debug_parts.append("\n--- Execution Logs ---")
                debug_parts.extend(stats.logs)

            debug_info = "\n".join(debug_parts)

        if not updated_count and not len(no_updates) and (len(errors) or len(blocked)):
            show_message_box(
                "No notes were updated. Check the field failure details in Debug Info.",
                copy_button_text="Copy Debug Info" if debug_info else None,
                copy_button_content=debug_info if debug_info else None,
            )
        else:
            parts = []
            if len(processed):
                parts.append(f"Updated {pluralize('note', len(processed))}")

            if len(partial):
                parts.append(
                    f"{pluralize('note', len(partial))} updated with field failures"
                )

            if len(errors):
                parts.append(f"{pluralize('note', len(errors))} failed")

            if len(blocked):
                parts.append(
                    f"{pluralize('note', len(blocked))} blocked by field failures"
                )

            if len(no_updates):
                parts.append(f"{pluralize('note', len(no_updates))} had no updates")

            if was_cancelled:
                parts.append("Batch processing cancelled")

            if chat_usage_summary.request_count:
                token_line = (
                    f"Chat tokens: {format_token_count(chat_usage_summary.total_tokens)} "
                    f"across {chat_usage_summary.request_count} request"
                    f"{'s' if chat_usage_summary.request_count != 1 else ''}"
                )
                if len(chat_usage_summary.providers) > 1:
                    provider_bits = ", ".join(
                        f"{summary.provider}: {format_token_count(summary.total_tokens)}"
                        for summary in chat_usage_summary.providers
                    )
                    token_line += f" ({provider_bits})"

                if config.openai_daily_token_budget_enabled:
                    daily_usage = chat_usage_tracker.get_openai_daily_usage()
                    token_line += (
                        f". OpenAI today: {format_token_count(daily_usage.used_total_tokens)}"
                        f" / {format_token_count(config.openai_daily_token_budget)}"
                    )

                parts.append(token_line)

            show_message_box(
                (". ".join(parts) + ".") if parts else "No notes were processed.",
                copy_button_text="Copy Debug Info" if debug_info else None,
                copy_button_content=debug_info if debug_info else None,
            )

    return wrapped_on_batch_success


def confirm_openai_batch_preflight(
    processor: NoteProcessor,
    card_ids: Sequence[int],
    overwrite_fields: bool,
) -> bool:
    preflight = processor.get_openai_batch_preflight(
        card_ids,
        overwrite_fields=overwrite_fields,
    )
    if preflight is None:
        return True

    if preflight.recommended_note_count <= 0:
        recommendation = "0 notes. You're already close to today's OpenAI budget."
    else:
        recommendation = (
            f"{preflight.recommended_note_count} note"
            f"{'s' if preflight.recommended_note_count != 1 else ''}."
        )

    details = "\n".join(
        [
            f"Selected notes: {preflight.note_count}",
            f"Estimated OpenAI requests: {preflight.request_count}",
            f"Estimated batch tokens: {format_token_count(preflight.estimated_total_tokens)}",
            f"Remaining today: {format_token_count(preflight.remaining_tokens)}",
            f"Fast-run budget: {format_token_count(preflight.fast_budget_tokens)}",
            f"Recommended fast batch size: {recommendation}",
            "Near the limit, Smart Notes drains in-flight OpenAI requests before starting more, which slows the batch down.",
        ]
    )
    return show_message_box(
        "OpenAI batch may slow down near today's token limit.",
        details=details,
        custom_ok="Continue",
        show_cancel=True,
    )


@with_processor  # type: ignore
def on_browser_context(processor: NoteProcessor, browser: browser.Browser, menu: QMenu):  # type: ignore
    item = QAction("✨ Generate Smart Fields", menu)
    menu.addSeparator()
    menu.addAction(item)

    cards = browser.selected_cards()

    def wrapped():
        if not confirm_openai_batch_preflight(
            processor,
            cards,
            overwrite_fields=config.regenerate_notes_when_batching,
        ):
            return
        processor.process_cards_with_progress(
            cards,
            on_success=make_on_batch_success(browser),
            overwrite_fields=config.regenerate_notes_when_batching,
        )

    item.triggered.connect(wrapped)


def on_start_actions() -> None:
    perform_update_check()

    # Cache decks for autocomplete
    async def cache_leaf_decks_map():
        deck_id_to_name_map()

    run_async_in_background(cache_leaf_decks_map)


@with_processor  # type: ignore
def on_main_window(processor: NoteProcessor):
    if not mw:
        return

    # Setup logger as first thing
    setup_logger()
    # Then setup config, which depends on logger
    config.setup_config()
    migrate_models()

    # Add options to Anki Menu
    options_action = QAction("Smart Notes", mw)
    # Triggered passes a bool, so we need to use a lambda to pass the processor
    options_action.triggered.connect(lambda _: on_options(processor)())
    mw.form.menuTools.addAction(options_action)
    mw.addonManager.setConfigAction(__name__, on_options(processor))
    on_start_actions()


@with_processor  # type: ignore
def on_editor_context(
    processor: NoteProcessor, editor_web_view: editor.EditorWebView, menu: QMenu
):
    editor = editor_web_view.editor
    card = editor.card
    note = editor.note

    # Add flow cards don't exist. Make ephemeral.
    if card is None and note is not None:
        deck_id = None
        parent = editor.parentWindow

        # When adding a new card the parent window is AddCards – grab the
        # selected deck so that prompts are fetched for the correct deck.
        if isinstance(parent, AddCards):
            deck_id = parent.deck_chooser.selected_deck_id
            logger.debug(
                f"on_editor_context: generated ephemeral card with deck_id {deck_id}"
            )

        card = note.ephemeral_card()
        if deck_id:
            card.did = deck_id

    # If we still do not have a card (or a note) there is nothing we can do.
    if card is None:
        return

    current_field_num = editor.currentField
    if current_field_num is None:
        return

    is_smart_field = bool(is_ai_field(current_field_num, card))

    # The FieldMenu UI component will add its own separators/actions as needed.

    field = get_field_from_index(card.note(), current_field_num)
    if not field:
        return

    field_menu = FieldMenu(
        editor_instance=editor,
        menu=menu,
        processor=processor,
        card=card,
        field_upper=field,
        is_smart_field=is_smart_field,
    )

    # Keep a reference to avoid premature garbage collection
    menu._smartnotes_field_menu = field_menu  # type: ignore


@with_processor  # type: ignore
def on_review(processor: NoteProcessor, card: Card):
    logger.debug("Reviewing...")

    if not config.generate_at_review:
        return

    note = card.note()

    def on_success(did_change: bool):
        if not did_change:
            return

        if not mw or not mw.col:
            logger.error("Error: mw not found")
            return

        logger.debug("Did update card on review...")

        mw.col.update_note(note)
        card.load()
        Sparkle()

        bump_usage_counter()

    processor.process_card(
        card, overwrite_fields=False, on_success=on_success, show_progress=False
    )


@with_processor  # type: ignore
def add_deck_option(
    processor: NoteProcessor,
    tree_view: browser.sidebar.SidebarTreeView,  # type: ignore
    menu: QMenu,
    sidebar_item: browser.SidebarItem,  # type: ignore
    _,
) -> None:
    if not mw or not mw.col:
        return
    cards: Sequence[int] = []

    if sidebar_item.item_type == SidebarItemType.NOTETYPE:
        cards = mw.col.find_cards(f'"note:{sidebar_item.name}"')
    elif sidebar_item.item_type in [SidebarItemType.DECK, SidebarItemType.DECK_CURRENT]:
        query = f'"deck:{sidebar_item.full_name}"'
        cards = mw.col.find_cards(query)
    else:
        return

    item = QAction("✨ Generate Smart Fields", menu)
    menu.addSeparator()
    menu.addAction(item)

    def wrapped():
        if not confirm_openai_batch_preflight(
            processor,
            cards,
            overwrite_fields=config.regenerate_notes_when_batching,
        ):
            return
        processor.process_cards_with_progress(
            cards,
            on_success=make_on_batch_success(tree_view.browser),
            overwrite_fields=config.regenerate_notes_when_batching,
        )

    item.triggered.connect(wrapped)


@with_sentry
def cleanup() -> None:
    logger.debug("Shutting down loggers")
    logger.handlers.clear()


@with_sentry
def setup_hooks(processor: NoteProcessor):
    gui_hooks.browser_will_show_context_menu.append(on_browser_context(processor))
    gui_hooks.browser_sidebar_will_show_context_menu.append(add_deck_option(processor))
    gui_hooks.editor_did_init_buttons.append(add_editor_top_button(processor))
    gui_hooks.editor_will_show_context_menu.append(on_editor_context(processor))
    gui_hooks.reviewer_did_show_question.append(on_review(processor))
    gui_hooks.main_window_did_init.append(on_main_window(processor))
    gui_hooks.profile_will_close.append(cleanup)
