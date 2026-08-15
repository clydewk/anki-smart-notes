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

from typing import Optional

import pytest

from src.models import TTSVoiceTarget
from src.tts_routing import (
    eligible_tts_targets,
    select_tts_target,
    tts_audio_extension,
)


def target(
    voice: str, *, language: Optional[str] = None, enabled: bool = True
) -> TTSVoiceTarget:
    return {
        "provider": "fish",
        "model": "s2.1-pro-free",
        "voice": voice,
        "language": language,
        "enabled": enabled,
    }


def test_language_routing_accepts_aliases_and_unrestricted_voices() -> None:
    japanese = target("ja", language="ja-JP")
    unrestricted = target("any")
    english = target("en", language="en-US")
    disabled = target("disabled", language="ja", enabled=False)

    assert eligible_tts_targets(
        [japanese, unrestricted, english, disabled], "Japanese"
    ) == [japanese, unrestricted]
    assert eligible_tts_targets(
        [target("eleven", language="English (United States)")], "English"
    ) == [target("eleven", language="English (United States)")]


def test_selection_is_stable_and_varies_across_notes() -> None:
    pool = [target("a"), target("b"), target("c")]

    selected = select_tts_target(pool, language=None, selection_key="42:audio")
    assert select_tts_target(pool, language=None, selection_key="42:audio") == selected

    voices = {
        select_tts_target(pool, language=None, selection_key=f"{note}:audio")["voice"]
        for note in range(20)
    }
    assert len(voices) > 1


def test_selection_requires_an_eligible_voice() -> None:
    with pytest.raises(ValueError, match="Japanese"):
        select_tts_target(
            [target("english", language="en")],
            language="Japanese",
            selection_key="1:audio",
        )


def test_audio_extension_depends_on_selected_target() -> None:
    gemini: TTSVoiceTarget = {
        "provider": "google",
        "model": "gemini-3.1-flash-tts-preview",
        "voice": "Kore",
        "language": None,
        "enabled": True,
    }

    assert tts_audio_extension(gemini) == "wav"
    assert tts_audio_extension(target("fish")) == "mp3"
