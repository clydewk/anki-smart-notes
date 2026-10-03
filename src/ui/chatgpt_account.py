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

from functools import partial
from typing import TYPE_CHECKING

from aqt.qt import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    Qt,
    QVBoxLayout,
    pyqtSignal,
)
from aqt.utils import openLink
from PyQt6 import sip

from ..chatgpt_auth import ChatGPTAccount, chatgpt_auth
from ..chatgpt_transport import chatgpt_transport
from ..logger import redact_text
from ..sentry import run_async_in_background_with_sentry
from ..utils import run_on_main

if TYPE_CHECKING:
    from ..tasks import JobHandle


class ChatGPTAccountWidget(QGroupBox):
    account_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("ChatGPT account")
        self._job: JobHandle | None = None
        self._signing_in = False
        self._limits_inflight = False
        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setAccessibleName("ChatGPT account status")
        layout.addWidget(self.status_label)
        explanation = QLabel(
            "Account actions apply immediately across all profiles, including when you cancel settings. "
            "Sign-in opens your browser. This connection is separate from the Codex app and the OpenAI API."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.limits_label = QLabel(
            "Uses your ChatGPT subscription. The OpenAI API estimated token guard does not apply."
        )
        self.limits_label.setWordWrap(True)
        self.limits_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.limits_label)
        self.sign_in_button = QPushButton("Sign in with ChatGPT")
        self.cancel_button = QPushButton("Cancel sign-in")
        self.sign_out_button = QPushButton("Sign out")
        self.limits_button = QPushButton("Check limits")
        actions = QHBoxLayout()
        for button in (
            self.sign_in_button,
            self.cancel_button,
            self.sign_out_button,
            self.limits_button,
        ):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        self.sign_in_button.clicked.connect(self._sign_in)
        self.cancel_button.clicked.connect(self.cancel_sign_in)
        self.sign_out_button.clicked.connect(self._sign_out)
        self.limits_button.clicked.connect(self._check_limits)
        self.refresh_status()

    def refresh_status(self) -> None:
        account = chatgpt_auth.status()
        self.status_label.setText(
            "Waiting for browser sign-in…"
            if self._signing_in
            else f"Signed in: {account.email or account.account_id}"
            + (f" · {account.plan}" if account.plan else "")
            if account
            else "Not signed in."
        )
        self.sign_in_button.setVisible(not self._signing_in and account is None)
        self.cancel_button.setVisible(self._signing_in)
        self.cancel_button.setEnabled(True)
        self.sign_out_button.setVisible(account is not None and not self._signing_in)
        self.limits_button.setVisible(account is not None and not self._signing_in)
        self.limits_button.setEnabled(not self._limits_inflight)

    def _sign_in(self) -> None:
        if self._signing_in:
            return
        self._signing_in = True
        self.refresh_status()

        def open_browser(url: str) -> None:
            run_on_main(partial(openLink, url))

        async def sign_in() -> ChatGPTAccount:
            return await chatgpt_auth.sign_in(open_browser)

        def success(_: ChatGPTAccount) -> None:
            self._signing_in = False
            self.refresh_status()
            self.account_changed.emit()

        def failure(error: Exception) -> None:
            self._signing_in = False
            self.refresh_status()
            self.status_label.setText(f"Could not sign in: {redact_text(str(error))}")

        def finish() -> None:
            self._job = None
            interrupted = self._signing_in
            self._signing_in = False
            if not sip.isdeleted(self):
                if interrupted:
                    self.refresh_status()
                self.cancel_button.hide()
                account = chatgpt_auth.status()
                self.sign_in_button.setVisible(account is None)
                self.sign_out_button.setVisible(account is not None)

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
        error_message: str | None = None
        try:
            chatgpt_auth.sign_out()
        except Exception as error:
            error_message = redact_text(str(error))
        self.refresh_status()
        if error_message:
            self.status_label.setText(error_message)
        self.limits_label.setText(
            "Uses your ChatGPT subscription. The OpenAI API estimated token guard does not apply."
        )
        self.account_changed.emit()

    def _check_limits(self) -> None:
        if self._limits_inflight:
            return
        self._limits_inflight = True
        self.limits_button.setEnabled(False)
        self.limits_label.setText("Checking subscription limits…")
        generation = chatgpt_auth.generation

        def success(limits: str) -> None:
            if generation == chatgpt_auth.generation:
                self.limits_label.setText(
                    f"{limits}. The OpenAI API estimated token guard does not apply."
                )

        def failure(error: Exception) -> None:
            if generation == chatgpt_auth.generation:
                self.limits_label.setText(
                    f"Could not check subscription limits: {redact_text(str(error))}"
                )

        def finish() -> None:
            self._limits_inflight = False
            if not sip.isdeleted(self):
                self.limits_button.setEnabled(True)

        try:
            run_async_in_background_with_sentry(
                chatgpt_transport.fetch_limits,
                success,
                failure,
                owner=self,
                on_finished=finish,
                show_error=False,
            )
        except Exception as error:
            failure(error)
            finish()
