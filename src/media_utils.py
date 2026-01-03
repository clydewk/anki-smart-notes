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

import functools
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from anki.notes import Note
from aqt import mw
from aqt.qt import QBuffer, QByteArray, QImage, QImageWriter, QIODevice

from .notes import get_note_type

HARDCODED_PATHS = (
    "/usr/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/bin",
    os.path.join(os.getenv("HOME", "/home/user"), ".local", "bin"),
)


@functools.cache
def find_executable(name: str) -> Optional[str]:
    """
    Find executable in system PATH or hardcoded fallback paths.
    Matches AJT Media Converter logic for finding ffmpeg/cwebp on macOS.
    """
    if path := shutil.which(name):
        return path

    for path_to_dir in HARDCODED_PATHS:
        if os.path.isfile(path_to_exe := os.path.join(path_to_dir, name)):
            return path_to_exe

    return None


def get_media_path(note: Note, field: str, format: str) -> str:
    return f"{get_note_type(note)}-{field}-{note.id}.{format}"


def convert_image_data(data: bytes, format: str, quality: int = -1) -> bytes:
    """
    Convert image data to specified format and quality using Qt, with fallback to external tools (ffmpeg/cwebp).

    Args:
        data: Input image bytes
        format: Target format (e.g. "webp", "png", "jpeg", "avif")
        quality: Compression quality 0-100, or -1 for default

    Returns:
        Converted image bytes
    """
    image = QImage()
    if not image.loadFromData(data):
        raise ValueError(
            "Failed to load image data. Input data might be corrupted or in an unsupported format."
        )

    fmt = format.upper()
    if fmt == "JPEG":
        fmt = "JPG"

    byte_array = QByteArray()
    buffer = QBuffer(byte_array)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)

    # Try Qt conversion first
    if image.save(buffer, fmt, quality):
        return byte_array.data()

    # Qt failed, try external tools fallback
    # Specifically for AVIF which is often missing in Qt builds
    if fmt == "AVIF":
        try:
            return _convert_to_avif_ffmpeg(data, quality)
        except Exception:
            # Fall through to raise the support error
            pass

    # Try cwebp for WebP if Qt failed (unlikely but possible)
    if fmt == "WEBP":
        try:
            return _convert_to_webp_cwebp(data, quality)
        except Exception:
            pass

    # If we get here, everything failed.
    supported = [bytes(f).decode() for f in QImageWriter.supportedImageFormats()]
    msg = f"Failed to convert image to {fmt}. It might not be supported by your Anki version. Supported formats: {', '.join(supported)}"

    if fmt == "AVIF":
        msg += ". \n\nTo enable AVIF support, please install 'ffmpeg' and make sure it is available in your system PATH (restart Anki after installing)."
    elif fmt == "WEBP":
        msg += ". \n\nTo enable WebP support, please install 'cwebp' (libwebp) and make sure it is available in your system PATH."

    raise ValueError(msg)


def _convert_to_avif_ffmpeg(data: bytes, quality: int) -> bytes:
    ffmpeg_path = find_executable("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError("ffmpeg not found")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_file:
        in_file.write(data)
        in_path = in_file.name

    out_path = in_path + ".avif"

    try:
        # Map 0-100 quality to CRF (approximate)
        # ffmpeg libaom-av1: crf 0-63 (0 is lossless)
        # 100 -> 0, 0 -> 63
        if quality == -1:
            crf = 23  # Default
        else:
            # Simple mapping: 100 quality is CRF 0. 0 quality is CRF 63.
            # Invert: q = 100 - (crf * 100 / 63) => crf = (100 - q) * 63 / 100
            crf = int((100 - quality) * 63 / 100)
            crf = max(0, min(63, crf))

        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            in_path,
            "-c:v",
            "libaom-av1",
            "-crf",
            str(crf),
            "-b:v",
            "0",
            out_path,
        ]

        # Run ffmpeg
        subprocess.run(cmd, check=True, capture_output=True)

        if not os.path.exists(out_path):
            raise Exception("ffmpeg failed to produce output file")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


def _convert_to_webp_cwebp(data: bytes, quality: int) -> bytes:
    cwebp_path = find_executable("cwebp")
    if not cwebp_path:
        raise FileNotFoundError("cwebp not found")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_file:
        in_file.write(data)
        in_path = in_file.name

    out_path = in_path + ".webp"

    try:
        q = quality if quality != -1 else 75

        cmd = [cwebp_path, "-q", str(q), in_path, "-o", out_path]

        subprocess.run(cmd, check=True, capture_output=True)

        if not os.path.exists(out_path):
            raise Exception("cwebp failed to produce output file")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


def write_media(file_name: str, file: bytes) -> Optional[str]:
    if not mw or not mw.col:
        return None
    media = mw.col.media
    if not media:
        return None
    return media.write_data(file_name, file)


def trash_files(file_names: list[str]) -> None:
    if not mw or not mw.col:
        return
    media = mw.col.media
    if not media:
        return
    media.trash_files(file_names)
