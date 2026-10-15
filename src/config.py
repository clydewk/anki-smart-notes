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
from collections.abc import Mapping
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

from aqt import mw

from .constants import DEFAULT_TEMPERATURE, GLOBAL_DECK_ID
from .logger import register_secrets
from .models import (
    BuiltInToolsConfig,
    ChatModels,
    ChatProviders,
    CustomProvider,
    ImageAspectRatio,
    ImageGenerationQuality,
    ImageModels,
    ImageOutputFormat,
    ImageProviders,
    ImageResolution,
    McpServerConfig,
    OpenAIModels,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    PromptMap,
    ProviderSettings,
    TTSModels,
    TTSProviders,
    TTSVoiceTarget,
    make_tts_voice_target,
    normalize_built_in_tools_config,
    normalize_field_extras,
    normalize_tts_voice_pool,
)
from .ui.rate_dialog import RateDialog
from .utils import USES_BEFORE_RATE_DIALOG, get_file_path

PROFILE_KEYS = frozenset(
    {
        "prompts_map",
        "generate_at_review",
        "regenerate_notes_when_batching",
        "allow_empty_fields",
    }
)
PROFILE_CONFIG_NAME = "smart-notes.json"
config_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "smart_notes_config", default=None
)


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy shapes without changing provider, model, or voice IDs."""
    normalized = deepcopy(dict(values))
    prompts = normalized.setdefault("prompts_map", {"note_types": {}})
    if not isinstance(prompts, dict) or not isinstance(prompts.get("note_types"), dict):
        raise ValueError(
            "Invalid Smart Notes prompts map; restore its backup before generating."
        )
    for note_type, decks in prompts["note_types"].items():
        if not isinstance(decks, dict):
            raise ValueError(f"Invalid Smart Notes prompts for note type {note_type}.")
        if "fields" in decks:
            decks = {str(GLOBAL_DECK_ID): decks}
            prompts["note_types"][note_type] = decks
        for field_options in decks.values():
            if not isinstance(field_options, dict) or not isinstance(
                field_options.get("fields"), dict
            ):
                raise ValueError(
                    f"Invalid Smart Notes fields for note type {note_type}."
                )
            extras = field_options.get("extras") or {}
            field_options["extras"] = {
                field: normalize_field_extras(extras.get(field))
                for field in dict.fromkeys([*field_options["fields"], *extras])
            }
    pool = normalized.get("tts_voice_pool")
    if pool is None:
        target = make_tts_voice_target(
            normalized.get("tts_provider"),
            normalized.get("tts_model"),
            normalized.get("tts_voice"),
        )
        normalized["tts_voice_pool"] = [target] if target else []
    else:
        normalized["tts_voice_pool"] = normalize_tts_voice_pool(pool)
    if not isinstance(normalized.get("chat_temperature"), (int, float)):
        normalized["chat_temperature"] = DEFAULT_TEMPERATURE
    if not isinstance(normalized.get("chat_use_tools"), bool):
        normalized["chat_use_tools"] = bool(normalized.get("chat_use_mcp", False))
    for key in ("openai_fast_mode_enabled", "openai_daily_token_budget_enabled"):
        if not isinstance(normalized.get(key), bool):
            normalized[key] = False
    if normalized.get("openai_service_tier") is None:
        normalized["openai_service_tier"] = (
            "fast" if normalized["openai_fast_mode_enabled"] else "auto"
        )
    if not isinstance(normalized.get("openai_daily_token_budget"), int):
        normalized["openai_daily_token_budget"] = 1_000_000
    normalized["built_in_tools"] = normalize_built_in_tools_config(
        normalized.get("built_in_tools")
    )
    normalized.update(
        did_deck_filter_migration=True,
        did_migrate_tts_voice_pool=True,
        did_cleanup_config_defaults=True,
    )
    return normalized


def initialize_profile_configs(
    base: Path, profiles: list[str], legacy: dict[str, Any]
) -> None:
    """Idempotently copy normalized legacy settings to the initial profiles."""
    values = normalize_config(legacy)
    for name in profiles:
        path = base / name / PROFILE_CONFIG_NAME
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                path,
                {
                    "version": 1,
                    **{key: deepcopy(values.get(key)) for key in PROFILE_KEYS},
                },
            )


class Config:
    """Fancy config class that uses the Anki addon manager to store config values."""

    openai_api_key: str | None
    anthropic_api_key: str | None
    deepseek_api_key: str | None
    google_api_key: str | None
    elevenlabs_api_key: str | None
    fish_api_key: str | None
    replicate_api_key: str | None

    custom_providers: list[CustomProvider]
    mcp_servers: list[McpServerConfig]
    built_in_tools: BuiltInToolsConfig
    provider_settings: dict[str, ProviderSettings]

    prompts_map: PromptMap
    generate_at_review: bool
    times_used: int
    last_seen_version: str | None
    openai_endpoint: str | None
    regenerate_notes_when_batching: bool
    allow_empty_fields: bool
    last_message_id: int
    debug: bool
    openai_fast_mode_enabled: bool
    openai_service_tier: OpenAIServiceTier
    openai_daily_token_budget_enabled: bool
    openai_daily_token_budget: int

    # Chat
    chat_provider: ChatProviders
    chat_model: ChatModels
    chat_temperature: float
    chat_reasoning_effort: OpenAIReasoningEffort | None
    chat_markdown_to_html: bool
    chat_use_tools: bool

    # TTS
    tts_voice_pool: list[TTSVoiceTarget]
    tts_strip_html: bool
    # Legacy single-voice settings retained for migration compatibility.
    tts_provider: TTSProviders
    tts_voice: str
    tts_model: TTSModels

    # Images
    image_provider: ImageProviders
    image_model: ImageModels
    image_aspect_ratio: ImageAspectRatio | None
    image_resolution: ImageResolution | None
    image_output_format: ImageOutputFormat | None
    image_quality: int | None
    image_generation_quality: ImageGenerationQuality | None

    # Dialogs / Migrations
    did_show_chained_error_dialog: bool
    did_show_rate_dialog: bool
    did_show_premium_tts_dialog: bool
    did_deck_filter_migration: bool
    did_migrate_tts_voice_pool: bool
    did_cleanup_config_defaults: bool
    did_click_rate_link: bool

    # Capacity alerts
    did_show_capacity_threshold_this_cycle: bool

    # Deprecated fields:
    legacy_openai_model: OpenAIModels
    legacy_openai_reasoning_effort: OpenAIReasoningEffort | None

    def setup_config(self) -> None:
        values = self.snapshot()
        normalized = normalize_config(values)
        changes = {
            key: value
            for key, value in normalized.items()
            if key not in values or value != values[key]
        }
        if changes:
            self.save(changes)

    def snapshot(self) -> dict[str, Any]:
        captured = config_context.get()
        if captured is not None:
            values = deepcopy(captured)
        else:
            if not mw:
                raise RuntimeError("Anki main window is not available")
            values = deepcopy(mw.addonManager.getConfig(__name__) or {})
            path = self.profile_path()
            if path is not None and path.exists():
                scoped = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(scoped, dict) or scoped.get("version") != 1:
                    raise ValueError("Unsupported Smart Notes profile settings version")
                values.update(
                    {key: scoped[key] for key in PROFILE_KEYS if key in scoped}
                )
        register_secrets(
            value
            for key, value in values.items()
            if isinstance(value, str) and ("key" in key or "token" in key)
        )
        for provider in values.get("custom_providers", []) or []:
            register_secrets([provider.get("api_key", "")])
        for server in values.get("mcp_servers", []) or []:
            register_secrets(
                pair["value"]
                for pair in [*server.get("env", []), *server.get("headers", [])]
                if isinstance(pair.get("value"), str)
            )
        return values

    def profile_path(self) -> Path | None:
        if not mw or not mw.pm.name:
            return None
        return Path(mw.pm.profileFolder()) / PROFILE_CONFIG_NAME

    def initialize_profiles(self) -> None:
        if not mw:
            return
        values = deepcopy(mw.addonManager.getConfig(__name__) or {})
        if not values.get("did_migrate_profiles"):
            backup = Path(get_file_path("config_backup_before_profiles.json"))
            if not backup.exists():
                write_json_atomic(backup, values)
            names = values.get("profile_migration_names")
            if names is None:
                names = list(mw.pm.profiles())
                values["profile_migration_names"] = names
                mw.addonManager.writeConfig(__name__, values)
            legacy = json.loads(backup.read_text(encoding="utf-8"))
            initialize_profile_configs(Path(mw.pm.base), names, legacy)
            values["did_migrate_profiles"] = True
            mw.addonManager.writeConfig(__name__, values)
        path = self.profile_path()
        if path is not None and not path.exists():
            defaults = self.defaults() or {}
            write_json_atomic(
                path,
                {
                    "version": 1,
                    **{key: defaults.get(key) for key in PROFILE_KEYS},
                    "prompts_map": {"note_types": {}},
                },
            )

    def __getattr__(self, key: str) -> Any:
        return self.snapshot().get(key)

    def __setattr__(self, name: str, value: object) -> None:
        self.save({name: value})

    def save(self, updates: Mapping[str, object]) -> None:
        if config_context.get() is not None:
            raise RuntimeError(
                "Cannot persist configuration from a background job or preview"
            )
        if not mw:
            raise RuntimeError("Anki main window is not available")
        previous = deepcopy(mw.addonManager.getConfig(__name__) or {})
        shared = {
            **previous,
            **{key: value for key, value in updates.items() if key not in PROFILE_KEYS},
        }
        path = self.profile_path()
        previous_profile = path.read_bytes() if path and path.exists() else None
        raw_scoped: object = (
            json.loads(previous_profile)
            if previous_profile is not None
            else {"version": 1}
        )
        if not isinstance(raw_scoped, dict) or raw_scoped.get("version") != 1:
            raise ValueError("Unsupported Smart Notes profile settings version")
        scoped: dict[str, Any] = raw_scoped
        scoped.update(
            {key: value for key, value in updates.items() if key in PROFILE_KEYS}
        )
        # Persist shared settings last; roll back the profile if Anki rejects the save.
        profile_written = False
        try:
            if path and PROFILE_KEYS.intersection(updates):
                write_json_atomic(path, scoped)
                profile_written = True
            elif path is None:
                shared.update(updates)
            mw.addonManager.writeConfig(__name__, shared)
        except Exception:
            if path and profile_written:
                if previous_profile is None:
                    path.unlink(missing_ok=True)
                else:
                    write_json_atomic(path, json.loads(previous_profile))
            if mw.addonManager.getConfig(__name__) != previous:
                mw.addonManager.writeConfig(__name__, previous)
            raise

    def restore_defaults(self) -> None:
        preserved = {
            "prompts_map",
            "custom_providers",
            "mcp_servers",
            "provider_settings",
        }
        self.save(
            {
                key: value
                for key, value in (self.defaults() or {}).items()
                if key not in preserved and "key" not in key and "token" not in key
            }
        )

    def defaults(self) -> dict[str, Any] | None:
        if not mw:
            return {}

        return mw.addonManager.addonConfigDefaults(__name__.split(".")[0])

    # Compatibility entry points for previous releases.
    def perform_deck_filter_migration(self) -> None:
        self.setup_config()

    def perform_tts_voice_pool_migration(self) -> None:
        self.setup_config()

    def perform_extras_cleanup(self) -> None:
        self.setup_config()


config = Config()


def bump_usage_counter() -> None:
    config.times_used += 1
    if config.times_used > USES_BEFORE_RATE_DIALOG and not config.did_show_rate_dialog:
        config.did_show_rate_dialog = True
        dialog = RateDialog()
        dialog.exec()


T = TypeVar("T")
M = TypeVar("M", bound=Mapping[str, object])


# TODO: this belongs in utils but ciruclar import
# TODO: make this use the none_defaulting (too much type golf for now tho)
def key_or_config_val(vals: M | None, k: str) -> T:  # type: ignore
    return (
        cast("T", vals[k])
        if (vals and vals.get(k) is not None)
        else cast("T", getattr(config, k))
    )
