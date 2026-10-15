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

from tests.test_settings_layout import SETTINGS_SETUP

CHATGPT_SETUP = (
    SETTINGS_SETUP
    + """
import asyncio
from dataclasses import replace
import src.ui.chatgpt_account as account_module
import src.ui.chat_options as chat_module
from src.chatgpt_auth import ChatGPTAccount
from src.models import ChatGPTModel

class FakeAuth:
    generation = 0
    account = None
    cancellations = 0
    needs_reauthorization = True
    welcome_needed = False
    welcomed = False
    revoke_error = None
    def __init__(self):
        self.accounts = []
        self.attempts = []
    def status(self):
        return self.account
    def list_accounts(self):
        return self.accounts.copy()
    def select_account(self, account_id):
        self.account = next(item for item in self.accounts if item.account_id == account_id)
        self.generation += 1
    async def sign_in(self, open_browser, *, new_account=False):
        self.attempts.append(new_account)
        open_browser("https://auth.example.test/sign-in")
        account_id = f"client-{len(self.accounts) + 1}" if new_account or self.account is None else self.account.account_id
        self.account = ChatGPTAccount("person@example.test", account_id, "pro")
        self.accounts = [item if item.account_id != account_id else self.account for item in self.accounts]
        if self.account not in self.accounts:
            self.accounts.append(self.account)
        self.needs_reauthorization = False
        self.welcome_needed = not self.welcomed
        self.generation += 1
        return self.account
    def mark_welcome_seen(self):
        self.welcomed = True
        self.welcome_needed = False
    def cancel_sign_in(self):
        self.cancellations += 1
    def sign_out(self):
        self.account = replace(self.account, connected=False, sharing=False)
        self.accounts = [self.account if item.account_id == self.account.account_id else item for item in self.accounts]
        self.generation += 1
        async def revoke():
            if self.revoke_error:
                raise RuntimeError(self.revoke_error)
        self.last_revocation = revoke()
        return self.last_revocation

class FakeJob:
    cancelled = False
    def cancel(self):
        self.cancelled = True

class FakeTransport:
    usage_limit_reached = False
    def resume_after_usage_review(self):
        self.usage_limit_reached = False

auth = FakeAuth()
transport = FakeTransport()
account_module.chatgpt_auth = auth
account_module.chatgpt_transport = transport
chat_module.chatgpt_auth = auth
requests = []
def launch(operation, success, failure=None, **kwargs):
    job = FakeJob()
    requests.append((operation, success, failure, kwargs, job))
    return job
account_module.run_async_in_background_with_sentry = launch
chat_module.run_async_in_background_with_sentry = launch
account_module.run_on_main = lambda work: work()
browser_urls = []
account_module.openLink = browser_urls.append
welcomes = []
account_module.show_message_box = lambda *args, **kwargs: welcomes.append((args, kwargs))
models = [ChatGPTModel("subscription-model", "Subscription model", ("low", "ultra"), "low")]
chat_module.chat_provider.get_cached_chatgpt_models = lambda: models

def finish_request(request):
    operation, success, failure, callbacks, job = request
    try:
        result = asyncio.run(operation())
    except Exception as error:
        failure(error)
    else:
        success(result)
    callbacks["on_finished"]()
"""
)


def test_chatgpt_account_actions_are_immediate_and_model_edits_are_drafts(
    tmp_path: Path,
) -> None:
    script = (
        CHATGPT_SETUP
        + """
before = deepcopy(settings)
first = options_module.AddonOptionsDialog(MagicMock())
first.resize(800, 650)
first.show()
first.tabs.setCurrentIndex(1)
settle()
account = first.chatgpt_account
assert "immediately" in first.scope_label.text()
assert "OpenAI API" in account.limits_label.text()
assert "migrate" in account.status_label.text()
assert account.sign_in_button.text() == "Continue with ChatGPT"
account.sign_in_button.click()
assert account.cancel_button.isVisible()
assert not account.sign_in_button.isVisible()
request = requests[-1]
assert request[3]["owner"] is account and request[3]["show_error"] is False
finish_request(request)
assert browser_urls == ["https://auth.example.test/sign-in"]
assert "person@example.test" in account.status_label.text()
assert "Using ChatGPT plan" in account.limits_label.text()
assert len(welcomes) == 1 and welcomes[0][1]["custom_ok"] == "Got it"
account.manage_usage_button.click()
assert browser_urls[-1] == "https://chatgpt.com/settings/usage"
first.chat_options.chat_provider.setCurrentText("ChatGPT")
assert "Using ChatGPT plan" in first.chat_options.chatgpt_usage.text()
assert first.chat_options.state.s["chat_model"] == "", "A new provider must not silently select a model"
first.chat_options.chat_model.setCurrentText("Subscription model")
assert first.chat_options.state.s["chat_model"] == "subscription-model"
assert first.chat_options.state.s["chat_reasoning_effort"] == "low"
assert first.chat_options.temperature.isHidden()
assert first.chat_options.response_form.labelForField(first.chat_options.temperature).isHidden()
assert first.chat_options.temperature_help.isHidden()
first.reject()
assert settings == before
assert auth.status().connected, "Cancel settings must not undo immediate sign-in"

second = options_module.AddonOptionsDialog(MagicMock())
second.show()
second.tabs.setCurrentIndex(1)
settle()
account = second.chatgpt_account
assert account.sign_out_button.isVisible()
account_changes = []
account.account_changed.connect(lambda: account_changes.append(auth.generation))
auth.revoke_error = "Disconnected locally, but remote revocation was not confirmed. Disconnect the app in ChatGPT Settings."
account.sign_out_button.click()
assert not auth.status().connected
assert account_changes == [auth.generation]
finish_request(requests[-1])
assert account.sign_in_button.isVisible()
assert account.sign_out_button.isHidden()
assert account.manage_usage_button.isHidden()
assert "revocation was not confirmed" in account.status_label.text()
account.sign_in_button.click()
request = requests[-1]
account.cancel_button.click()
assert request[-1].cancelled and auth.cancellations == 1
assert not account.sign_in_button.isVisible(), "Wait for cancellation cleanup before another sign-in"
request[3]["on_finished"]()
assert account.sign_in_button.isVisible()
assert account.cancel_button.isHidden()
account.sign_in_button.click()
request = requests[-1]
request[2](RuntimeError("https://example.test?api_key=private-token"))
request[3]["on_finished"]()
assert "private-token" not in account.status_label.text()
assert account.sign_in_button.isVisible()
account.sign_in_button.click()
request = requests[-1]
second.reject()
assert request[-1].cancelled and auth.cancellations == 2
request[3]["on_finished"]()
assert settings == before
saved_values = []
options_module.Config.save = lambda self, values: saved_values.append(values)
third = options_module.AddonOptionsDialog(MagicMock())
third.chat_options.chat_provider.setCurrentText("ChatGPT")
third.chat_options.chat_model.setCurrentText("Subscription model")
third.on_accept()
assert saved_values[-1]["chat_provider"] == "chatgpt"
assert saved_values[-1]["chat_model"] == "subscription-model"
assert not {"access_token", "refresh_token", "chatgpt_account"}.intersection(saved_values[-1])
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_chatgpt_account_picker_consent_and_usage_recovery_fit_small_windows(
    tmp_path: Path,
) -> None:
    script = (
        CHATGPT_SETUP
        + """
auth.needs_reauthorization = False
auth.account = ChatGPTAccount("person@example.test", "client-1", None, sharing=False)
auth.accounts = [auth.account]
dialog = options_module.AddonOptionsDialog(MagicMock())
dialog.show()
dialog.tabs.setCurrentIndex(1)
settle()
account = dialog.chatgpt_account
assert account.sign_in_button.text() == "Enable ChatGPT plan"
assert "not enabled" in account.limits_label.text()
dialog.chat_options.chat_provider.setCurrentText("ChatGPT")
dialog.tabs.setCurrentIndex(3)
settle()
assert not requests, "Identity-only sign-in must not start model discovery"
assert "Enable your ChatGPT plan" in dialog.chat_options.chatgpt_usage.text()
dialog.tabs.setCurrentIndex(1)
account.sign_in_button.click()
finish_request(requests[-1])
assert len(welcomes) == 1
account.add_account_button.click()
finish_request(requests[-1])
assert auth.attempts == [False, True]
assert len(welcomes) == 1, "A later sign-in must not repeat the welcome"
assert account.account_picker.count() == 2
assert account.account_picker.itemText(0) != account.account_picker.itemText(1)
assert account.account_picker.itemData(0) == "client-1"
assert account.account_picker.itemData(1) == "client-2"
account.account_picker.setCurrentIndex(0)
assert auth.account.account_id == "client-1"
assert account.account_picker.currentData() == "client-1"
transport.usage_limit_reached = True
dialog.tabs.setCurrentIndex(3)
dialog.tabs.setCurrentIndex(1)
settle()
assert "Usage limit reached" in account.limits_label.text()
assert "plan or app limit" in account.limits_label.text()
assert account.resume_button.isVisible()
account.manage_usage_button.click()
assert browser_urls[-1] == "https://chatgpt.com/settings/usage"
assert transport.usage_limit_reached, "Opening settings must not silently resume generation"
for width, height in [(800, 650), (640, 470)]:
    dialog.resize(width, height)
    settle()
    scroll = dialog.tabs.currentWidget()
    assert scroll.horizontalScrollBar().maximum() == 0
    for button in (account.manage_usage_button, account.sign_out_button, account.add_account_button, account.resume_button):
        assert button.width() >= button.sizeHint().width(), (button.text(), button.width(), button.sizeHint().width())
account.resume_button.click()
assert not transport.usage_limit_reached
assert account.resume_button.isHidden()
dialog.reject()
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_chatgpt_discovery_preserves_saved_selections_and_manual_model_ids(
    tmp_path: Path,
) -> None:
    script = (
        SETTINGS_SETUP
        + """
import src.ui.chat_options as chat_module
from src.models import ChatGPTModel
from src.chatgpt_auth import ChatGPTAccount
settings.update(chat_provider="chatgpt", chat_model="unavailable-saved-model", chat_reasoning_effort="saved-effort")
requests = []
def launch(operation, success, failure=None, **kwargs):
    requests.append((success, failure, kwargs))
chat_module.run_async_in_background_with_sentry = launch
chat_module.chatgpt_auth = SimpleNamespace(status=lambda: ChatGPTAccount("person@example.test", "client-1", None), generation=0)
models = [ChatGPTModel("subscription-model", "Subscription model", ("low", "ultra"), "low")]
chat_module.chat_provider.get_cached_chatgpt_models = lambda: models
chat = chat_module.ChatOptions()
host = QDialog()
from PyQt6.QtWidgets import QVBoxLayout
from src.ui.ui_utils import scroll_page
layout = QVBoxLayout(host)
scroll = scroll_page(chat)
layout.addWidget(scroll)
host.resize(800, 650)
host.show()
settle()
assert requests
requests[-1][0](models)
requests[-1][2]["on_finished"]()
assert chat.state.s["chat_model"] == "unavailable-saved-model"
assert chat.state.s["chat_reasoning_effort"] == "saved-effort"
assert "not returned" in chat.model_warning.text()
assert chat.chat_model.isEditable()
assert chat.show_available_models.isHidden()
assert scroll.horizontalScrollBar().maximum() == 0
chat.chat_model.setCurrentText("Subscription model")
assert chat.state.s["chat_model"] == "subscription-model"
assert chat.state.s["chat_reasoning_effort"] == "low"
chat.reasoning_effort.setCurrentText("Ultra")
assert chat.state.s["chat_reasoning_effort"] == "ultra"
assert chat.model_warning.isHidden()
chat.chat_model.setCurrentText("manual-model-id")
assert chat.state.s["chat_model"] == "manual-model-id"
chat.refresh_models.click()
requests[-1][1](TimeoutError("https://example.test?api_key=private-token"))
requests[-1][2]["on_finished"]()
assert "private-token" not in chat.discovery_status.text()
assert chat.state.s["chat_model"] == "manual-model-id"
assert chat.refresh_models.isEnabled()
assert chat.refresh_models.text() == "Retry"
chat.chat_provider.setCurrentText("OpenAI")
assert not chat.temperature.isHidden()
chat.chat_provider.setCurrentText("ChatGPT")
assert chat.state.s["chat_model"] == "manual-model-id"
host.reject()
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unreadable_chatgpt_store_does_not_block_other_settings(tmp_path: Path) -> None:
    script = (
        SETTINGS_SETUP
        + """
import src.ui.chat_options as chat_module
from src.chatgpt_auth import ChatGPTAuthError
settings.update(chat_provider="chatgpt")
def unavailable():
    raise ChatGPTAuthError("Could not read the ChatGPT connection. Check application-data folder permissions.")
account_module.chatgpt_auth.status = unavailable
chat_module.chatgpt_auth = account_module.chatgpt_auth
dialog = options_module.AddonOptionsDialog(MagicMock())
dialog.show()
dialog.tabs.setCurrentIndex(1)
settle()
account = dialog.chatgpt_account
assert "folder permissions" in account.status_label.text()
assert "other provider settings" in account.limits_label.text()
assert account.sign_in_button.isHidden()
assert account.account_picker.isHidden()
dialog.tabs.setCurrentIndex(3)
settle()
assert "folder permissions" in dialog.chat_options.discovery_status.text()
assert not dialog.chat_options.refresh_models.isEnabled()
dialog.chat_options.chat_provider.setCurrentText("OpenAI")
assert dialog.chat_options.chatgpt_usage.isHidden()
dialog.reject()
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_signout_submission_failure_and_early_cancellation_clear_pending_work(
    tmp_path: Path,
) -> None:
    script = (
        CHATGPT_SETUP
        + """
asyncio.run(auth.sign_in(lambda url: None))
dialog = options_module.AddonOptionsDialog(MagicMock())
dialog.show()
dialog.tabs.setCurrentIndex(1)
settle()
def reject_submission(*args, **kwargs):
    raise RuntimeError("The profile is closing.")
account_module.run_async_in_background_with_sentry = reject_submission
dialog.chatgpt_account.sign_out_button.click()
assert not auth.account.connected
assert auth.last_revocation.cr_frame is None
assert "Remote revocation was not confirmed" in dialog.chatgpt_account.status_label.text()
assert dialog.chatgpt_account.sign_in_button.isVisible()
account_module.run_async_in_background_with_sentry = launch
asyncio.run(auth.sign_in(lambda url: None))
dialog.chatgpt_account.refresh_status()
dialog.chatgpt_account.sign_out_button.click()
assert not auth.account.connected
assert auth.last_revocation.cr_frame is not None
dialog.reject()
requests[-1][3]["on_finished"]()
assert auth.last_revocation.cr_frame is None
collection.close()
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
