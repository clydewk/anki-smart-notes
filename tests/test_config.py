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

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import (
    PROFILE_CONFIG_NAME,
    Config,
    config_context,
    normalize_config,
    write_json_atomic,
)


@pytest.fixture
def profile_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> SimpleNamespace:
    defaults = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    shared = deepcopy(defaults)
    shared.update(
        chat_temperature=0,
        tts_provider="Custom Voice",
        tts_model="legacy-model-id",
        tts_voice="legacy-voice-id",
        tts_voice_pool=None,
        prompts_map={
            "note_types": {
                "Stale Type": {
                    "fields": {"Deleted Field": "Explain {{Old Source}}"},
                    "extras": {
                        "Deleted Field": {
                            "type": "tts",
                            "use_custom_model": True,
                            "tts_provider": "Custom Voice",
                            "tts_model": "legacy-model-id",
                            "tts_voice": "legacy-voice-id",
                            "image_quality": 0,
                        }
                    },
                }
            }
        },
    )
    names = ["A", "B"]
    for name in names:
        (tmp_path / name).mkdir()
    manager = MagicMock()
    manager.getConfig.side_effect = lambda _: deepcopy(shared)
    manager.addonConfigDefaults.side_effect = lambda _: deepcopy(defaults)

    def write(_: str, value: dict[str, Any]) -> None:
        shared.clear()
        shared.update(deepcopy(value))

    manager.writeConfig.side_effect = write
    pm = SimpleNamespace(name="A", base=str(tmp_path), profiles=lambda: list(names))
    pm.profileFolder = lambda: str(tmp_path / pm.name)
    monkeypatch.setattr("src.config.mw", SimpleNamespace(addonManager=manager, pm=pm))
    monkeypatch.setattr("src.config.get_file_path", lambda name: str(tmp_path / name))
    return SimpleNamespace(
        config=Config(),
        shared=shared,
        manager=manager,
        pm=pm,
        names=names,
        base=tmp_path,
        write=write,
    )


def test_profile_migration_normalizes_every_copy_and_preserves_ids(
    profile_environment: SimpleNamespace,
) -> None:
    env = profile_environment
    original = deepcopy(env.shared)
    env.config.initialize_profiles()
    backup = (env.base / "config_backup_before_profiles.json").read_bytes()
    assert json.loads(backup) == original
    a = json.loads((env.base / "A" / PROFILE_CONFIG_NAME).read_text())
    b = json.loads((env.base / "B" / PROFILE_CONFIG_NAME).read_text())
    assert a == b
    assert "openai_api_key" not in a
    extras = a["prompts_map"]["note_types"]["Stale Type"]["-1"]["extras"][
        "Deleted Field"
    ]
    assert extras["tts_voice_pool"][0]["provider"] == "Custom Voice"
    assert extras["tts_voice_pool"][0]["model"] == "legacy-model-id"
    assert extras["image_quality"] == 0
    env.config.setup_config()
    assert env.config.chat_temperature == 0
    env.config.save({"prompts_map": {"note_types": {}}})
    env.pm.name = "B"
    env.config.setup_config()
    assert env.config.prompts_map == b["prompts_map"]
    env.names.append("New")
    (env.base / "New").mkdir()
    env.pm.name = "New"
    env.config.initialize_profiles()
    assert env.config.prompts_map == {"note_types": {}}
    assert (env.base / "config_backup_before_profiles.json").read_bytes() == backup


def test_profile_migration_retry_preserves_initialized_profiles_and_original_source(
    profile_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = profile_environment
    original = deepcopy(env.shared)
    blocked = env.base / "B" / PROFILE_CONFIG_NAME

    def fail_b(path: Path, value: object) -> None:
        if path == blocked:
            raise OSError("profile B unavailable")
        write_json_atomic(path, value)

    monkeypatch.setattr("src.config.write_json_atomic", fail_b)
    with pytest.raises(OSError, match="profile B"):
        env.config.initialize_profiles()
    assert not env.shared.get("did_migrate_profiles")
    assert env.shared["profile_migration_names"] == ["A", "B"]
    customized = {"version": 1, "prompts_map": {"note_types": {}}}
    write_json_atomic(env.base / "A" / PROFILE_CONFIG_NAME, customized)
    env.shared["prompts_map"] = {"note_types": {}}
    env.names.append("New")
    (env.base / "New").mkdir()
    monkeypatch.setattr("src.config.write_json_atomic", write_json_atomic)
    env.config.initialize_profiles()
    assert json.loads((env.base / "A" / PROFILE_CONFIG_NAME).read_text()) == customized
    assert (
        json.loads(blocked.read_text())["prompts_map"]
        == normalize_config(original)["prompts_map"]
    )
    assert env.shared["did_migrate_profiles"] is True
    env.pm.name = "New"
    env.config.initialize_profiles()
    assert env.config.prompts_map == {"note_types": {}}


@pytest.mark.parametrize("partially_written", [False, True])
def test_save_failure_rolls_back_profile_and_shared_settings(
    profile_environment: SimpleNamespace, partially_written: bool
) -> None:
    env = profile_environment
    env.config.initialize_profiles()
    env.config.setup_config()
    original = deepcopy(env.shared)
    profile = env.base / "A" / PROFILE_CONFIG_NAME
    profile_before = json.loads(profile.read_text())
    calls = 0

    def fail_first(name: str, value: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            if partially_written:
                env.write(name, value)
            raise OSError("failed Save")
        env.write(name, value)

    env.manager.writeConfig.side_effect = fail_first
    with pytest.raises(OSError, match="failed Save"):
        env.config.save(
            {"openai_api_key": "new-secret", "prompts_map": {"note_types": {}}}
        )
    assert env.shared == original
    assert json.loads(profile.read_text()) == profile_before


def test_save_does_not_touch_shared_settings_when_profile_write_fails(
    profile_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = profile_environment
    env.config.initialize_profiles()
    before = deepcopy(env.shared)
    profile = env.base / "A" / PROFILE_CONFIG_NAME
    profile_before = profile.read_bytes()
    env.manager.writeConfig.reset_mock()
    monkeypatch.setattr(
        "src.config.write_json_atomic",
        MagicMock(side_effect=PermissionError("read-only profile")),
    )
    with pytest.raises(PermissionError, match="read-only"):
        env.config.save({"prompts_map": {"note_types": {}}})
    env.manager.writeConfig.assert_not_called()
    assert env.shared == before
    assert profile.read_bytes() == profile_before


def test_snapshot_is_detached_and_draft_cannot_persist(
    profile_environment: SimpleNamespace,
) -> None:
    env = profile_environment
    env.config.initialize_profiles()
    snapshot = env.config.snapshot()
    token = config_context.set(snapshot)
    env.manager.writeConfig.reset_mock()
    try:
        detached = env.config.snapshot()
        detached["prompts_map"]["note_types"].clear()
        assert env.config.prompts_map["note_types"]
        with pytest.raises(RuntimeError, match="background job or preview"):
            env.config.save({"openai_api_key": "unsaved-key"})
        env.manager.writeConfig.assert_not_called()
    finally:
        config_context.reset(token)


def test_normalization_preserves_explicit_empty_pool_and_is_idempotent() -> None:
    values = {
        "prompts_map": {"note_types": {}},
        "tts_provider": "openai",
        "tts_model": "tts-1",
        "tts_voice": "alloy",
        "tts_voice_pool": [],
        "chat_temperature": 0,
    }
    normalized = normalize_config(values)
    assert normalized["tts_voice_pool"] == []
    assert normalized["chat_temperature"] == 0
    assert normalize_config(normalized) == normalized
    assert "did_deck_filter_migration" not in values


@pytest.mark.parametrize("fast,expected", [(False, "auto"), (True, "fast")])
def test_normalization_migrates_legacy_openai_speed(fast: bool, expected: str) -> None:
    normalized = normalize_config({"openai_fast_mode_enabled": fast})
    assert normalized["openai_service_tier"] == expected
    assert normalize_config(normalized) == normalized
    for saved in ("default", "ultrafast", "future-tier"):
        normalized["openai_service_tier"] = saved
        assert normalize_config(normalized)["openai_service_tier"] == saved


def test_atomic_replace_failure_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"original": true}')
    monkeypatch.setattr(
        Path, "replace", MagicMock(side_effect=PermissionError("replace failed"))
    )
    with pytest.raises(PermissionError):
        write_json_atomic(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}
    assert not path.with_suffix(".json.tmp").exists()


def test_unknown_profile_version_cannot_be_overwritten(
    profile_environment: SimpleNamespace,
) -> None:
    env = profile_environment
    path = env.base / "A" / PROFILE_CONFIG_NAME
    write_json_atomic(path, {"version": 2, "future": "data"})
    with pytest.raises(ValueError, match="version"):
        env.config.snapshot()
    with pytest.raises(ValueError, match="version"):
        env.config.save({"prompts_map": {"note_types": {}}})
    assert json.loads(path.read_text()) == {"version": 2, "future": "data"}
