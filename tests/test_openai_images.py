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
from datetime import date
from pathlib import Path
from typing import Any, get_args
from unittest.mock import AsyncMock

import pytest

from src.image_provider import ImageProvider
from src.models import ImageAspectRatio, ImageModels, ImageResolution, openai_image_size
from tests.test_settings_layout import SETTINGS_SETUP


@pytest.mark.parametrize(
    "model", ["gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]
)
@pytest.mark.parametrize("ratio", get_args(ImageAspectRatio))
@pytest.mark.parametrize("resolution", get_args(ImageResolution))
def test_flexible_image_presets_match_their_ratio_and_provider_limits(
    model: str, ratio: ImageAspectRatio, resolution: ImageResolution
) -> None:
    width, height = map(int, openai_image_size(model, ratio, resolution).split("x"))
    ratio_width, ratio_height = map(int, ratio.split(":"))
    assert width * ratio_height == height * ratio_width
    assert width % 16 == height % 16 == 0
    assert max(width, height) <= 3840
    assert 655_360 <= width * height <= 8_294_400


@pytest.mark.parametrize("model", ["gpt-image-1", "gpt-image-1.5", "gpt-image-1-mini"])
def test_legacy_image_presets_keep_supported_dimensions(model: str) -> None:
    for resolution in get_args(ImageResolution):
        assert openai_image_size(model, "16:9", resolution) == "1536x1024"
        assert openai_image_size(model, "4:3", resolution) == "1536x1024"
        assert openai_image_size(model, "3:4", resolution) == "1024x1536"
        assert openai_image_size(model, "9:16", resolution) == "1024x1536"
        assert openai_image_size(model, "1:1", resolution) == "1024x1024"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "gpt-image-1",
        "gpt-image-1.5",
        "gpt-image-1-mini",
        "gpt-image-2",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst",
    ],
)
async def test_all_gpt_images_allow_slow_generation(
    monkeypatch: pytest.MonkeyPatch, mock_config: Any, model: ImageModels
) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2026, 10, 13)

    mock_config.openai_endpoint = None
    monkeypatch.setattr("src.models.date", FixedDate)
    monkeypatch.setattr("src.image_provider.config", mock_config)
    request = AsyncMock(return_value=b'{"data": [{"b64_json": "aW1hZ2U="}]}')
    monkeypatch.setattr("src.image_provider.provider_runtime.request_bytes", request)
    result = await ImageProvider().async_get_image_response(
        prompt="A cat",
        provider="openai",
        model=model,
        note_id=1,
    )
    assert result == b"image"
    assert request.call_args.kwargs["timeouts"].sock_read_timeout_sec == 180
    assert request.call_args.kwargs["retry_ambiguous"] is False


@pytest.mark.asyncio
async def test_openai_image_generation_does_not_replay_ambiguous_requests(
    monkeypatch: pytest.MonkeyPatch, mock_config: Any
) -> None:
    mock_config.openai_endpoint = None
    request = AsyncMock(return_value=b'{"data": [{"b64_json": "aW1hZ2U="}]}')
    monkeypatch.setattr("src.image_provider.config", mock_config)
    monkeypatch.setattr("src.image_provider.provider_runtime.request_bytes", request)
    result = await ImageProvider().async_get_image_response(
        "A cat",
        "gpt-image-2.5-flare",
        "openai",
        1,
        aspect_ratio="16:9",
        generation_quality="max",
        output_format="webp",
    )
    assert result == b"image"
    assert request.call_args.kwargs["retry_ambiguous"] is False
    payload = request.call_args.kwargs["json_payload"]
    assert payload["size"] == "1536x864"
    assert payload["quality"] == "max"
    assert payload["output_format"] == "webp"
    assert "response_format" not in payload


@pytest.mark.asyncio
async def test_retired_image_selection_fails_before_a_request(
    monkeypatch: pytest.MonkeyPatch, mock_config: Any
) -> None:
    request = AsyncMock()
    monkeypatch.setattr("src.image_provider.config", mock_config)
    monkeypatch.setattr("src.image_provider.provider_runtime.request_bytes", request)
    with pytest.raises(ValueError, match="unavailable"):
        await ImageProvider().async_get_image_response("A cat", "dall-e-3", "openai", 1)
    request.assert_not_called()


def test_image_options_preserve_retired_ids_and_label_real_sizes(
    tmp_path: Path,
) -> None:
    script = (
        SETTINGS_SETUP
        + """
from datetime import date
import src.models as models
from src.ui.image_options import ImageOptions

class Today(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 13)
models.date = Today
settings.update(image_provider="openai", image_model="dall-e-3")
options = ImageOptions()
assert options.state.s["image_model"] == "dall-e-3"
assert options.model_picker.currentText() == "DALL·E 3 (Retired)"
assert "unavailable" in options.model_warning.text()
assert "dall-e-3" not in options._provider_models["openai"]
settings.update(image_model="gpt-image-2.5-flare", image_aspect_ratio="16:9", image_resolution="1024x1024")
options = ImageOptions()
assert "1536 × 864 · 16:9" == options.ratio_picker.currentText()
assert options.resolution_picker.currentText() == "Small"
options.state.update({"image_resolution": "4096x4096"})
options._update_size_controls()
assert "3840 × 2160 · 16:9" == options.ratio_picker.currentText()
assert options.resolution_picker.currentText() == "Large (experimental)"
settings.update(image_model="gpt-image-1", image_aspect_ratio="4:3")
options = ImageOptions()
assert "2026-10-23" in options.model_warning.text()
assert "1536 × 1024 · landscape (saved 4:3)" == options.ratio_picker.currentText()
assert len(options.ratio_picker.ui_to_state) == len(options.state.s["image_aspect_ratios"])
assert options.state.s["image_aspect_ratio"] == "4:3"
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
