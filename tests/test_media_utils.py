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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.field_processor import FieldProcessor
from src.media_utils import extract_sound_file_name, run_media_conversion


def test_extract_sound_file_name() -> None:
    assert (
        extract_sound_file_name("[sound:文法+-example 1_audio-1481165522264.wav]")
        == "文法+-example 1_audio-1481165522264.wav"
    )


def test_extract_sound_file_name_returns_none_without_sound_tag() -> None:
    assert extract_sound_file_name("") is None
    assert extract_sound_file_name("not audio") is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_media_subprocess_is_reaped_on_timeout_or_cancel(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    started = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    create_subprocess = asyncio.create_subprocess_exec

    async def create(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_subprocess(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr("src.media_utils.asyncio.create_subprocess_exec", create)
    monkeypatch.setattr("src.media_utils.MEDIA_TIMEOUT_SECONDS", 60 if cancel else 0.02)
    task = asyncio.create_task(
        run_media_conversion([sys.executable, "-c", "import time; time.sleep(60)"])
    )
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert processes[0].returncode is not None


def test_native_avif_conversion_and_preview_with_installed_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("Optional FFmpeg encoder is not installed")
    script = """
import asyncio
import aqt
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QImage
from src.media_utils import convert_image_data, image_preview_data

image = QImage(128, 128, QImage.Format.Format_RGB32)
image.fill(0xff55bb88)
data = QByteArray()
buffer = QBuffer(data)
buffer.open(QIODevice.OpenModeFlag.WriteOnly)
assert image.save(buffer, "PNG")
png = data.data()

async def verify():
    encoded = await convert_image_data(png, "avif", 85)
    assert b"avif" in encoded[:64]
    preview = await image_preview_data(encoded)
    decoded = QImage.fromData(preview)
    assert (decoded.width(), decoded.height()) == (128, 128)
    if QImage.fromData(encoded).isNull():
        assert preview.startswith(b"\\x89PNG")
    assert await image_preview_data(png) == png
    assert (await convert_image_data(encoded, "png")).startswith(b"\\x89PNG")

asyncio.run(verify())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


async def test_image_conversion_failure_reaches_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_provider = MagicMock()
    image_provider.async_get_image_response = AsyncMock(return_value=b"image bytes")
    processor = FieldProcessor(MagicMock(), MagicMock(), image_provider)
    monkeypatch.setattr(processor, "_check_api_key", lambda *_: True)
    monkeypatch.setattr(
        "src.field_processor.config", MagicMock(allow_empty_fields=False)
    )
    monkeypatch.setattr(
        "src.field_processor.convert_image_data",
        AsyncMock(side_effect=ValueError("Image encoder is unavailable")),
    )
    with pytest.raises(ValueError, match="Image encoder is unavailable"):
        await processor.get_image_response(
            note_id=1,
            values={},
            input_text="A green square",
            model="gpt-image-1.5",
            provider="openai",
            output_format="avif",
            show_error_box=False,
        )
