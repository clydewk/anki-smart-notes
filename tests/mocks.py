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
    chat_reasoning_effort: Any = None
    chat_markdown_to_html: bool = False
    chat_use_mcp: bool = False
    tts_provider: str = "openai"
    tts_voice: str = "alloy"
    tts_model: str = "tts-1"
    tts_strip_html: bool = True
    image_provider: str = "openai"
    image_model: str = "gpt-image-1.5"
    image_aspect_ratio: Any = None
    image_resolution: Any = None
    image_output_format: Any = None
    image_quality: Any = None
    openai_api_key: str = "test-openai-key"
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    google_api_key: str = ""
    elevenlabs_api_key: str = ""
    replicate_api_key: str = ""
    custom_providers: Any = None
    mcp_servers: Any = None
    provider_settings: Any = None
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
