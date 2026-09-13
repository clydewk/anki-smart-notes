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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.chat_provider import ChatProvider


@pytest.mark.asyncio
async def test_discovery_failure_preserves_cache_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = SimpleNamespace(openai_api_key="test-key", openai_endpoint=None)
    request = AsyncMock(return_value={"data": [{"id": "gpt-5.6-sol"}]})
    monkeypatch.setattr("src.chat_provider.config", configuration)
    monkeypatch.setattr("src.chat_provider.provider_runtime.request_json", request)
    provider = ChatProvider()
    assert await provider.fetch_openai_chat_models() == ["gpt-5.6-sol"]
    request.side_effect = TimeoutError("connection timeout")
    with pytest.raises(TimeoutError, match="connection timeout"):
        await provider.fetch_openai_chat_models()
    assert provider.get_cached_openai_chat_models() == ["gpt-5.6-sol"]
    configuration.openai_api_key = "different-key"
    assert provider.get_cached_openai_chat_models() == []


def test_native_model_options_explain_discovery_and_effective_image_output() -> None:
    script = """
import json
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QSizePolicy
from PyQt6.QtGui import QFont
from src.config import config_context
from src.models import openai_image_size
from src.ui.image_options import ImageOptions
from src.ui.ui_utils import scroll_page
import src.ui.chat_options as chat_module

app = QApplication([])
settings = json.loads(Path("config.json").read_text())
settings.update(openai_api_key="", chat_provider="openai", chat_model="gpt-5.6-sol")
config_context.set(settings)
requests = []
def launch(operation, success, failure=None, **kwargs):
    requests.append((success, failure, kwargs))
chat_module.run_async_in_background_with_sentry = launch
chat_module.chat_provider.get_cached_openai_chat_models = lambda: ["gpt-5.6-sol"]
chat = chat_module.ChatOptions()
assert not requests
host = QDialog()
layout = QVBoxLayout(host)
scroll = scroll_page(chat)
layout.addWidget(scroll)
host.resize(640, 470)
host.show()
app.processEvents()
assert not requests
assert "API key" in chat.discovery_status.text()
assert chat.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
assert chat.chat_model.width() > 250
for point_size in (13, 18):
    enlarged = QFont(app.font())
    enlarged.setPointSize(point_size)
    host.setFont(enlarged)
    app.processEvents()
    assert scroll.horizontalScrollBar().maximum() == 0
    for label in (chat.temperature_help, chat.reasoning_help, chat.discovery_status):
        assert label.height() >= label.heightForWidth(label.width())
assert chat.show_available_models.text()
assert "**" not in chat.temperature_help.text()
settings["openai_api_key"] = "configured-key"
chat._refresh_openai_models_if_needed("openai")
assert len(requests) == 1
assert "Refreshing" in chat.discovery_status.text()
assert not chat.refresh_models.isEnabled()
assert requests[-1][2]["show_error"] is False
saved_model = chat.state.s["chat_model"]
saved_effort = chat.state.s["chat_reasoning_effort"]
requests[-1][1](TimeoutError("https://example.test?api_key=private-value"))
requests[-1][2]["on_finished"]()
assert "private-value" not in chat.discovery_status.text()
assert "cached" in chat.discovery_status.text()
assert chat.refresh_models.text() == "Retry"
assert chat.refresh_models.isEnabled()
chat.refresh_models.click()
assert len(requests) == 2
requests[-1][0]([])
requests[-1][2]["on_finished"]()
assert "no supported text models" in chat.discovery_status.text()
assert chat.state.s["chat_model"] == saved_model
assert chat.state.s["chat_reasoning_effort"] == saved_effort
assert "not returned" in chat.model_warning.text()
chat.refresh_models.click()
requests[-1][0]([saved_model])
requests[-1][2]["on_finished"]()
assert "Fetched 1 text models" in chat.discovery_status.text()
assert chat.model_warning.isHidden()
chat.chat_provider.setCurrentText("Anthropic")
assert chat.discovery.isHidden()
assert not chat.reasoning_effort.isEnabled()
assert "not sent to Anthropic" in chat.reasoning_help.text()
host.close()

images = ImageOptions({
    "image_provider": "openai", "image_model": "gpt-image-1.5",
    "image_aspect_ratio": "16:9", "image_resolution": "4096x4096",
    "image_output_format": "avif", "image_generation_quality": "high",
})
assert images.state.s["image_resolution"] == "4096x4096"
assert images.resolution_picker.isHidden()
assert "1536 × 1024" in images.ratio_picker.currentText()
assert "1536 × 1024" in images.output_summary.text()
assert "AVIF" in images.output_summary.text()
assert images.model_picker.maximumWidth() > 300
images.state.update({"image_aspect_ratio": "4:3"})
images._update_size_controls()
assert images.state.s["image_aspect_ratio"] == "4:3"
assert "saved 4:3" in images.ratio_picker.currentText()
images.state.update({"image_model": "gpt-image-2", "image_aspect_ratio": "16:9"})
images._on_model_change("gpt-image-2")
assert not images.resolution_picker.isHidden()
assert "3840 × 2160" in images.output_summary.text()
assert openai_image_size("gpt-image-2", "16:9", "4096x4096") == "3840x2160"
images.resolution_picker.setCurrentText("2K preset")
assert "2048 × 1152" in images.ratio_picker.currentText()
assert "2048 × 1152" in images.output_summary.text()
images.resolution_picker.setCurrentText("4K preset")
assert "3840 × 2160" in images.ratio_picker.currentText()
images.state.update({"image_model": "gpt-image-1.5", "image_generation_quality": "xhigh"})
images._on_model_change("gpt-image-1.5")
assert images.state.s["image_generation_quality"] == "xhigh"
assert "not supported" in images.model_warning.text()
images.generation_quality_picker.setCurrentText("High")
assert images.model_warning.isHidden()
images.state.update({"image_provider": "google", "image_model": "gemini-3-pro-image"})
images._on_model_change("gemini-3-pro-image")
assert "4K requested" in images.output_summary.text()
assert "selected by the provider" in images.output_summary.text()
images.state.update({"image_provider": "replicate", "image_model": "flux-schnell"})
images._on_model_change("flux-schnell")
assert images.resolution_picker.isHidden()
assert "selected by the provider" in images.output_summary.text()
images.format_picker.setCurrentText("PNG (Lossless)")
assert images.quality_spinner.isHidden()
assert "Saved as PNG" in images.output_summary.text()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
