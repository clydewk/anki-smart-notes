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

import hashlib
from typing import Optional

from .models import TTSVoiceTarget

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "chinese": "zh",
    "chinese (mandarin)": "zh",
    "english": "en",
    "english (united states)": "en-us",
    "french": "fr",
    "french (france)": "fr-fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "spanish (spain)": "es-es",
    "swedish": "sv",
    "vietnamese": "vi",
    "vietnamese (north)": "vi",
}


def normalize_tts_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        return None
    return LANGUAGE_ALIASES.get(normalized, normalized)


def tts_language_matches(
    target_language: Optional[str], requested: Optional[str]
) -> bool:
    target = normalize_tts_language(target_language)
    requested_language = normalize_tts_language(requested)
    if target is None or requested_language is None:
        return True
    return (
        target == requested_language
        or target.split("-", 1)[0] == requested_language.split("-", 1)[0]
    )


def eligible_tts_targets(
    pool: list[TTSVoiceTarget], language: Optional[str] = None
) -> list[TTSVoiceTarget]:
    return [
        target
        for target in pool
        if target["enabled"]
        and target["provider"]
        and target["model"]
        and target["voice"]
        and tts_language_matches(target.get("language"), language)
    ]


def select_tts_target(
    pool: list[TTSVoiceTarget], *, language: Optional[str], selection_key: str
) -> TTSVoiceTarget:
    eligible = eligible_tts_targets(pool, language)
    if not eligible:
        language_suffix = f" for language '{language}'" if language else ""
        raise ValueError(f"No enabled TTS voices are available{language_suffix}.")

    def score(target: TTSVoiceTarget) -> bytes:
        identity = "\0".join(
            [
                selection_key,
                target["provider"],
                target["model"],
                target["voice"],
                target.get("language") or "",
            ]
        )
        return hashlib.blake2b(identity.encode("utf-8"), digest_size=16).digest()

    return max(eligible, key=score)


def tts_audio_extension(target: TTSVoiceTarget) -> str:
    return (
        "wav"
        if target["provider"] == "google" and "gemini" in target["model"]
        else "mp3"
    )
