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
import sys
from unittest.mock import MagicMock

import pytest
from tests.mocks import MockConfig

# Mock anki modules before they are imported
sys.modules["aqt"] = MagicMock()
sys.modules["aqt.addons"] = MagicMock()
sys.modules["anki"] = MagicMock()
sys.modules["anki.decks"] = MagicMock()

os.environ["IS_TEST"] = "True"


@pytest.fixture
def mock_config(monkeypatch):
    config = MockConfig(
        prompts_map={
            "note_types": {
                "Basic": {
                    "All": {
                        "fields": {"Front": "test", "Back": "test"},
                        "extras": {
                            "Front": {
                                "type": "chat",
                                "chat_model": None,
                                "tts_model": None,
                                "tts_voice": None,
                            },
                            "Back": {
                                "type": "tts",
                                "chat_model": None,
                                "tts_model": None,
                                "tts_voice": None,
                            },
                        },
                    }
                }
            }
        }
    )
    monkeypatch.setattr("src.migrations.config", config)
    return config


@pytest.fixture
def mock_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr("src.migrations.logger", logger)
    return logger
