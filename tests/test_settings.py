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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.config import config, config_context
from src.job_context import ui_job_owner
from src.settings import resolve_field_settings, with_widget_settings

if TYPE_CHECKING:
    from aqt.qt import QWidget


def test_effective_settings_keep_zero_false_inherit_none_and_detach_connections() -> (
    None
):
    defaults: dict[str, Any] = {
        "chat_provider": "Local",
        "chat_model": "local-model",
        "chat_temperature": 1,
        "chat_markdown_to_html": True,
        "image_quality": 80,
        "custom_providers": [
            {
                "name": "Local",
                "base_url": "http://localhost:11434",
                "api_key": "saved-key",
            }
        ],
        "mcp_servers": [{"env": [{"key": "API_KEY", "value": "tool-key"}]}],
        "tts_voice_pool": [{"provider": "openai", "model": "tts-1", "voice": "alloy"}],
    }
    overrides = {
        "chat_model": None,
        "chat_temperature": 0,
        "chat_markdown_to_html": False,
        "image_quality": 0,
    }
    production = resolve_field_settings(defaults, overrides)
    preview = resolve_field_settings(deepcopy(defaults), deepcopy(overrides))
    assert production == preview
    assert production["chat_model"] == "local-model"
    assert production["chat_temperature"] == 0
    assert production["chat_markdown_to_html"] is False
    assert production["image_quality"] == 0
    defaults["custom_providers"][0]["api_key"] = "edited-key"
    defaults["mcp_servers"][0]["env"][0]["value"] = "edited-tool-key"
    defaults["tts_voice_pool"][0]["voice"] = "echo"
    assert production["custom_providers"][0]["api_key"] == "saved-key"
    assert production["mcp_servers"][0]["env"][0]["value"] == "tool-key"
    assert production["tts_voice_pool"][0]["voice"] == "alloy"


def test_disabled_overrides_preserve_defaults_and_apply_field_rules() -> None:
    defaults = {
        "chat_provider": "openai",
        "chat_model": "global-model",
        "chat_temperature": 1,
        "tts_strip_html": True,
        "image_quality": 80,
        "chat_use_tools": False,
    }
    extras = {
        "use_custom_model": False,
        "chat_model": "ignored-model",
        "chat_temperature": 0,
        "tts_strip_html": False,
        "image_quality": 0,
        "chat_use_tools": True,
        "tts_style": "Warm",
        "tts_language": "ja",
    }
    resolved = resolve_field_settings(defaults, extras)
    assert resolved["chat_model"] == "global-model"
    assert resolved["chat_temperature"] == 1
    assert resolved["tts_strip_html"] is True
    assert resolved["image_quality"] == 80
    assert resolved["chat_use_tools"] is True
    assert resolved["tts_style"] == "Warm"
    assert resolved["tts_language"] == "ja"


@pytest.mark.parametrize("fail", [False, True])
def test_widget_action_uses_ancestor_draft_and_restores_context(fail: bool) -> None:
    draft = {
        "openai_api_key": "unsaved-provider-key",
        "chat_temperature": 0,
        "custom_providers": [{"name": "Local", "api_key": "unsaved-local-key"}],
    }
    ancestor = SimpleNamespace(draft_settings=lambda: deepcopy(draft))
    child = cast("QWidget", SimpleNamespace(parentWidget=lambda: ancestor))
    captured: dict[str, Any] = {}

    @with_widget_settings
    def launch(widget: "QWidget") -> None:
        captured.update(config.snapshot())
        assert ui_job_owner.get() is widget
        if fail:
            raise ValueError("preview failed")

    previous = {"openai_api_key": "persisted-provider-key"}
    context_token = config_context.set(previous)
    owner_token = ui_job_owner.set(None)
    try:
        if fail:
            with pytest.raises(ValueError, match="preview failed"):
                launch(child)
        else:
            launch(child)
        assert config_context.get() is previous
        assert ui_job_owner.get() is None
    finally:
        ui_job_owner.reset(owner_token)
        config_context.reset(context_token)
    draft["openai_api_key"] = "later-edit"
    assert captured["openai_api_key"] == "unsaved-provider-key"
    assert captured["custom_providers"][0]["api_key"] == "unsaved-local-key"
