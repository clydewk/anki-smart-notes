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

from collections.abc import Coroutine
from functools import partial
from typing import TYPE_CHECKING

from aqt.qt import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShowEvent,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    pyqtSignal,
)
from aqt.utils import openLink
from PyQt6 import sip

from ..chatgpt_auth import ChatGPTAccount, ChatGPTAuthError, chatgpt_auth
from ..chatgpt_transport import chatgpt_transport
from ..logger import redact_text
from ..sentry import run_async_in_background_with_sentry
from ..utils import run_on_main
from .ui_utils import show_message_box

if TYPE_CHECKING:
    from ..tasks import JobHandle

CHATGPT_USAGE_URL = "https://chatgpt.com/settings/usage"


class ChatGPTAccountWidget(QGroupBox):
    account_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("ChatGPT account")
        self._job: JobHandle | None = None
        self._signing_in = False
        self._signing_out = False
        layout = QVBoxLayout(self)
        self.account_picker = QComboBox()
        self.account_picker.setAccessibleName("Active ChatGPT account")
        self.account_picker.setMinimumContentsLength(16)
        self.account_picker.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.account_picker.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.account_picker.currentIndexChanged.connect(self._select_account)
        layout.addWidget(self.account_picker)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setAccessibleName("ChatGPT account status")
        layout.addWidget(self.status_label)
        explanation = QLabel(
            "Use your ChatGPT plan for text generation. Account actions apply immediately "
            "across all profiles, including when you cancel settings. Sign-in opens your browser."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.limits_label = QLabel()
        self.limits_label.setWordWrap(True)
        self.limits_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.limits_label)
        self.sign_in_button = QPushButton("Continue with ChatGPT")
        self.cancel_button = QPushButton("Cancel sign-in")
        self.add_account_button = QPushButton("Add account…")
        self.sign_out_button = QPushButton("Sign out")
        self.manage_usage_button = QPushButton("Manage usage")
        self.resume_button = QPushButton("Resume requests after reviewing usage")
        for buttons in (
            (self.sign_in_button, self.cancel_button, self.add_account_button),
            (self.manage_usage_button, self.sign_out_button),
            (self.resume_button,),
        ):
            actions = QHBoxLayout()
            for button in buttons:
                actions.addWidget(button)
            actions.addStretch()
            layout.addLayout(actions)
        self.sign_in_button.clicked.connect(lambda: self._sign_in())
        self.add_account_button.clicked.connect(lambda: self._sign_in(new_account=True))
        self.cancel_button.clicked.connect(self.cancel_sign_in)
        self.sign_out_button.clicked.connect(self._sign_out)
        self.manage_usage_button.clicked.connect(lambda: openLink(CHATGPT_USAGE_URL))
        self.resume_button.clicked.connect(self._resume_requests)
        self.refresh_status()

    def showEvent(self, a0: QShowEvent | None) -> None:
        self.refresh_status()
        super().showEvent(a0)

    def refresh_status(self) -> None:
        try:
            account = chatgpt_auth.status()
            accounts = chatgpt_auth.list_accounts()
            needs_reauthorization = chatgpt_auth.needs_reauthorization
        except ChatGPTAuthError as error:
            self.status_label.setText(redact_text(str(error)))
            self.limits_label.setText("Your other provider settings remain available.")
            for control in (
                self.account_picker,
                self.sign_in_button,
                self.add_account_button,
                self.cancel_button,
                self.sign_out_button,
                self.manage_usage_button,
                self.resume_button,
            ):
                control.hide()
            return
        self.account_picker.blockSignals(True)
        self.account_picker.clear()
        for index, saved in enumerate(accounts, 1):
            label = f"Account {index} · {saved.email or 'Not signed in'}"
            if saved.plan:
                label += f" · {saved.plan}"
            self.account_picker.addItem(label, saved.account_id)
            if account and saved.account_id == account.account_id:
                self.account_picker.setCurrentIndex(index - 1)
        self.account_picker.blockSignals(False)
        self.account_picker.setVisible(bool(accounts))
        busy = self._signing_in or self._signing_out
        connected = bool(account and account.connected)
        sharing = bool(connected and account and account.sharing)
        limited = sharing and chatgpt_transport.usage_limit_reached
        self.account_picker.setEnabled(not busy)
        self.status_label.setText(
            "Waiting for browser sign-in…"
            if self._signing_in
            else "Signing out…"
            if self._signing_out
            else f"Signed in: {account.email or 'ChatGPT account'}"
            + (f" · {account.plan}" if account.plan else "")
            if connected and account
            else "Sign in again to migrate your ChatGPT connection."
            if needs_reauthorization
            else "Not signed in."
        )
        self.limits_label.setText(
            "Usage limit reached. Review your plan or app limit in ChatGPT settings. "
            "New ChatGPT requests are paused."
            if limited
            else "Using ChatGPT plan. The OpenAI API estimated token guard does not apply."
            if sharing
            else "ChatGPT plan use is not enabled. Enable it to generate text."
            if connected
            else "ChatGPT plan usage is separate from the OpenAI API estimated token guard."
        )
        self.sign_in_button.setText(
            "Enable ChatGPT plan"
            if connected and not sharing
            else "Continue with ChatGPT"
        )
        self.sign_in_button.setVisible(not busy and not sharing)
        self.cancel_button.setVisible(self._signing_in)
        self.cancel_button.setEnabled(True)
        self.add_account_button.setVisible(bool(accounts) and not busy)
        self.sign_out_button.setVisible(connected and not busy)
        self.manage_usage_button.setVisible(connected and not busy)
        self.resume_button.setVisible(limited and not busy)
        self.manage_usage_button.setAutoDefault(limited)

    def _select_account(self, index: int) -> None:
        account_id = self.account_picker.itemData(index)
        if not isinstance(account_id, str):
            return
        try:
            chatgpt_auth.select_account(account_id)
        except Exception as error:
            self.refresh_status()
            self.status_label.setText(redact_text(str(error)))
            return
        self.refresh_status()
        self.account_changed.emit()

    def _sign_in(self, *, new_account: bool = False) -> None:
        if self._signing_in or self._signing_out:
            return
        self._signing_in = True
        self.refresh_status()

        def open_browser(url: str) -> None:
            run_on_main(partial(openLink, url))

        async def sign_in() -> ChatGPTAccount:
            return await chatgpt_auth.sign_in(open_browser, new_account=new_account)

        def success(_: ChatGPTAccount) -> None:
            self._signing_in = False
            self.refresh_status()
            self.account_changed.emit()
            if chatgpt_auth.welcome_needed:
                show_message_box(
                    "You’re using your ChatGPT plan",
                    "Eligible text generation in Smart Notes uses your ChatGPT plan. "
                    "Review usage and set this app’s limits in ChatGPT Settings → Usage.",
                    custom_ok="Got it",
                )
                try:
                    chatgpt_auth.mark_welcome_seen()
                except ChatGPTAuthError as error:
                    self.status_label.setText(redact_text(str(error)))

        def failure(error: Exception) -> None:
            self._signing_in = False
            self.refresh_status()
            self.status_label.setText(f"Could not sign in: {redact_text(str(error))}")

        def finish() -> None:
            self._job = None
            interrupted = self._signing_in
            self._signing_in = False
            if not sip.isdeleted(self) and interrupted:
                self.refresh_status()

        try:
            self._job = run_async_in_background_with_sentry(
                sign_in,
                success,
                failure,
                owner=self,
                on_finished=finish,
                show_error=False,
            )
        except Exception as error:
            failure(error)
            finish()

    def cancel_sign_in(self) -> None:
        if not self._signing_in:
            return
        chatgpt_auth.cancel_sign_in()
        if self._job is not None:
            self._job.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling sign-in…")

    def _sign_out(self) -> None:
        if self._signing_in or self._signing_out:
            return
        try:
            revocation = chatgpt_auth.sign_out()
        except Exception as error:
            self.refresh_status()
            self.status_label.setText(redact_text(str(error)))
            self.account_changed.emit()
            return
        self._signing_out = True
        self.refresh_status()
        self.account_changed.emit()
        started = False

        async def revoke() -> None:
            nonlocal started
            started = True
            await revocation

        def failure(_: Exception) -> None:
            self.status_label.setText(
                "Signed out locally. Remote revocation was not confirmed; "
                "disconnect Smart Notes in ChatGPT Settings → Usage."
            )

        def finish() -> None:
            if not started and isinstance(revocation, Coroutine):
                revocation.close()
            self._signing_out = False
            if not sip.isdeleted(self):
                message = self.status_label.text()
                self.refresh_status()
                if message != "Signing out…":
                    self.status_label.setText(message)

        try:
            run_async_in_background_with_sentry(
                revoke,
                lambda _: None,
                failure,
                owner=self,
                on_finished=finish,
                show_error=False,
            )
        except Exception as error:
            failure(error)
            finish()

    def _resume_requests(self) -> None:
        chatgpt_transport.resume_after_usage_review()
        self.refresh_status()
        self.account_changed.emit()
