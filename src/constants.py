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

from typing import cast

from anki.decks import DeckId

from .models import ChatModels, ChatProviders

RETRY_BASE_SECONDS = 5
MAX_RETRIES = 10
CHAT_CLIENT_TIMEOUT_SEC = 300
REASONING_CLIENT_TIMEOUT_SEC = 1200
TTS_PROVIDER_TIMEOUT_SEC = 30
IMAGE_PROVIDER_TIMEOUT_SEC = 90
GOOGLE_IMAGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_IMAGE_DEFAULT_SIZE = "1024x1024"

STANDARD_BATCH_LIMIT = 500

DEFAULT_CHAT_MODEL: ChatModels = "gpt-4o-mini"
DEFAULT_CHAT_PROVIDER: ChatProviders = "openai"

DEFAULT_TEMPERATURE = 1

API_KEY_MISSING_MESSAGE = (
    "Smart Notes: API Key missing for provider {}. Please configure it in settings."
)

GLOBAL_DECK_ID: DeckId = cast("DeckId", -1)
GLOBAL_DECK_NAME = "All Decks"
