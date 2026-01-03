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

from typing import Any

from attr import dataclass


@dataclass
class MockConfig:
    prompts_map: Any
    allow_empty_fields: bool = False
    chat_provider: str = "openai"
    chat_model: str = "gpt-4o-mini"
    chat_temperature: int = 0
    tts_provider: str = "openai"
    tts_voice: str = "alloy"
    tts_model: str = "tts-1"
    openai_api_key: str = ""
    auth_token: str = ""
    uuid: str = "test-uuid-12345"
    debug: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


@dataclass
class MockNote:
    _note_type: str
    _data: dict[str, Any]

    id = 1

    def note_type(self):
        return {"name": self._note_type}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def items(self):
        return self._data.items()

    def fields(self):
        return self._data.keys()


def p(str) -> str:
    return f"p_{str}"


class MockOpenAIClient:
    async def async_get_chat_response(self, prompt: str):
        return p(prompt)


class MockChatClient:
    async def async_get_chat_response(
        self,
        prompt: str,
        model: str,
        provider: str,
        note_id: int,
        temperature: int = 0,
        retry_count: int = 0,
    ) -> str:
        return p(prompt)


class MockAppState:
    """Mock app state that simulates an unlocked app with unlimited capacity"""

    state = {
        "subscription": "PAID_PLAN_ACTIVE",  # Unlocked state
        "plan": {
            "planId": "test_plan",
            "planName": "Test Plan",
            "notesUsed": 0,
            "notesLimit": 1000,
            "daysLeft": 30,
            "textCreditsUsed": 0,
            "textCreditsCapacity": 1000,
            "voiceCreditsUsed": 0,
            "voiceCreditsCapacity": 1000,
            "imageCreditsUsed": 0,
            "imageCreditsCapacity": 1000,
        },
    }
