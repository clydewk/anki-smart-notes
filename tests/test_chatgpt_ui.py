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


def test_chatgpt_account_actions_are_immediate_and_model_edits_are_drafts(
    tmp_path: Path,
) -> None:
    script = (
        SETTINGS_SETUP
        + """
import asyncio
import src.ui.chatgpt_account as account_module
import src.ui.chat_options as chat_module
from src.chatgpt_auth import ChatGPTAccount
from src.models import ChatGPTModel

class FakeAuth:
    generation = 0
    account = None
    cancellations = 0
    def status(self):
        return self.account
    async def sign_in(self, open_browser):
        open_browser("https://auth.example.test/sign-in")
        self.account = ChatGPTAccount("person@example.test", "account-1", "pro")
        self.generation += 1
        return self.account
    def cancel_sign_in(self):
        self.cancellations += 1
    def sign_out(self):
        self.account = None
        self.generation += 1
        raise RuntimeError("Disconnected, but saved credentials could not be removed. Check folder permissions.")

class FakeJob:
    cancelled = False
    def cancel(self):
        self.cancelled = True

auth = FakeAuth()
account_module.chatgpt_auth = auth
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
models = [ChatGPTModel("subscription-model", "Subscription model", ("low", "ultra"), "low")]
chat_module.chat_provider.get_cached_chatgpt_models = lambda: models
before = deepcopy(settings)

first = options_module.AddonOptionsDialog(MagicMock())
first.resize(800, 650)
first.show()
first.tabs.setCurrentIndex(1)
settle()
account = first.chatgpt_account
assert "immediately" in first.scope_label.text()
assert "OpenAI API" in account.limits_label.text()
account.sign_in_button.click()
assert account.cancel_button.isVisible()
assert not account.sign_in_button.isVisible()
operation, success, failure, callbacks, job = requests[-1]
assert callbacks["owner"] is account and callbacks["show_error"] is False
success(asyncio.run(operation()))
callbacks["on_finished"]()
assert browser_urls == ["https://auth.example.test/sign-in"]
assert "person@example.test" in account.status_label.text()
account.limits_button.click()
limits_request = requests[-1]
limits_request[1]("5-hour window: 60% remaining")
limits_request[3]["on_finished"]()
assert "60% remaining" in account.limits_label.text()
assert account.limits_button.isEnabled()
first.chat_options.chat_provider.setCurrentText("ChatGPT")
assert first.chat_options.state.s["chat_model"] == "", "A new provider must not silently select a model"
first.chat_options.chat_model.setCurrentText("Subscription model")
assert first.chat_options.state.s["chat_model"] == "subscription-model"
assert first.chat_options.state.s["chat_reasoning_effort"] == "low"
assert first.chat_options.temperature.isHidden()
assert first.chat_options.response_form.labelForField(first.chat_options.temperature).isHidden()
assert first.chat_options.temperature_help.isHidden()
first.reject()
assert settings == before
assert auth.status() is not None, "Cancel settings must not undo immediate sign-in"

second = options_module.AddonOptionsDialog(MagicMock())
second.show()
second.tabs.setCurrentIndex(1)
settle()
assert second.chatgpt_account.sign_out_button.isVisible()
account_changes = []
second.chatgpt_account.account_changed.connect(lambda: account_changes.append(auth.generation))
second.chatgpt_account.sign_out_button.click()
assert auth.status() is None
assert second.chatgpt_account.sign_in_button.isVisible()
assert second.chatgpt_account.sign_out_button.isHidden()
assert second.chatgpt_account.limits_button.isHidden()
assert "folder permissions" in second.chatgpt_account.status_label.text()
assert account_changes == [auth.generation]
second.chatgpt_account.sign_in_button.click()
request = requests[-1]
second.chatgpt_account.cancel_button.click()
assert request[-1].cancelled and auth.cancellations == 1
assert not second.chatgpt_account.sign_in_button.isVisible(), "Wait for cancellation cleanup before another sign-in"
request[3]["on_finished"]()
assert second.chatgpt_account.sign_in_button.isVisible()
assert second.chatgpt_account.cancel_button.isHidden()
second.chatgpt_account.sign_in_button.click()
request = requests[-1]
request[2](RuntimeError("https://example.test?api_key=private-token"))
request[3]["on_finished"]()
assert "private-token" not in second.chatgpt_account.status_label.text()
assert second.chatgpt_account.sign_in_button.isVisible()
second.chatgpt_account.sign_in_button.click()
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


def test_chatgpt_discovery_preserves_saved_selections_and_manual_model_ids(
    tmp_path: Path,
) -> None:
    script = (
        SETTINGS_SETUP
        + """
import src.ui.chat_options as chat_module
from src.models import ChatGPTModel
settings.update(chat_provider="chatgpt", chat_model="unavailable-saved-model", chat_reasoning_effort="saved-effort")
requests = []
def launch(operation, success, failure=None, **kwargs):
    requests.append((success, failure, kwargs))
chat_module.run_async_in_background_with_sentry = launch
chat_module.chatgpt_auth = SimpleNamespace(status=lambda: object(), generation=0)
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
