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

from copy import deepcopy

import pytest

from src.migrations import migrate_models
from tests.mocks import MockConfig


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openai", "gpt-4o"),
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.5-pro"),
        ("Local Gateway", "gpt-4o"),
        ("anthropic", "claude-custom-model"),
        ("elevenLabs", "eleven_turbo_v2_5"),
        ("fish", "eleven_turbo_v2_5"),
        ("azure", "unavailable-model"),
    ],
)
def test_upgrade_preserves_global_and_field_model_and_voice_ids(
    mock_config: MockConfig, provider: str, model: str
) -> None:
    target = {
        "provider": provider,
        "model": model,
        "voice": "saved-voice-id",
        "enabled": True,
        "language": "ja",
    }
    mock_config.chat_provider = provider
    mock_config.chat_model = model
    mock_config.tts_provider = provider
    mock_config.tts_model = model
    mock_config.tts_voice = "saved-voice-id"
    mock_config.tts_voice_pool = [target]
    mock_config.prompts_map["note_types"]["Basic"]["All"]["extras"]["Front"].update(
        {
            "chat_provider": provider,
            "chat_model": model,
            "tts_provider": provider,
            "tts_model": model,
            "tts_voice": "saved-voice-id",
            "tts_voice_pool": [target],
        }
    )
    before = deepcopy(vars(mock_config))

    migrate_models()
    migrate_models()

    assert vars(mock_config) == before
