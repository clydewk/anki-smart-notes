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

from src.media_utils import extract_sound_file_name


def test_extract_sound_file_name() -> None:
    assert (
        extract_sound_file_name("[sound:文法+-example 1_audio-1481165522264.wav]")
        == "文法+-example 1_audio-1481165522264.wav"
    )


def test_extract_sound_file_name_returns_none_without_sound_tag() -> None:
    assert extract_sound_file_name("") is None
    assert extract_sound_file_name("not audio") is None
