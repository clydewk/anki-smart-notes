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

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from anki.sound import SoundOrVideoTag
from anki.utils import tmpfile
from aqt import gui_hooks, sound

if TYPE_CHECKING:
    from anki.sound import AVTag
    from aqt.sound import Player

preview_path: Path | None = None
preview_playing = False


def clear_audio_preview(*, stop: bool = True) -> None:
    global preview_path, preview_playing
    path, preview_path = preview_path, None
    preview_playing = False
    if path is None:
        return
    if stop:
        sound.av_player.stop_and_clear_queue_if_caller(path)
    # Anki's temporary directory also cleans up files still held open on Windows.
    with suppress(OSError):
        path.unlink(missing_ok=True)


def audio_preview_did_begin(player: "Player", tag: "AVTag") -> None:
    global preview_playing
    if preview_path is None:
        return
    if isinstance(tag, SoundOrVideoTag) and tag.filename == str(preview_path):
        preview_playing = True
    elif preview_playing:
        clear_audio_preview(stop=False)


def audio_preview_did_end(player: "Player | None") -> None:
    if preview_playing:
        clear_audio_preview(stop=False)


def play_audio(audio: bytes) -> None:
    global preview_path
    clear_audio_preview()
    suffix = ".wav" if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE" else ".mp3"
    preview_path = Path(tmpfile(prefix="smart-notes-preview-", suffix=suffix))
    try:
        preview_path.write_bytes(audio)
        sound.av_player.play_file_with_caller(str(preview_path), preview_path)
    except Exception:
        clear_audio_preview()
        raise


gui_hooks.av_player_did_begin_playing.append(audio_preview_did_begin)
gui_hooks.av_player_did_end_playing.append(audio_preview_did_end)
gui_hooks.profile_will_close.append(clear_audio_preview)
