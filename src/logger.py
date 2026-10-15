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

import logging
import os
import re
import sys
from collections.abc import Iterable

from aqt import mw

from .utils import get_file_path, is_production

SECRET_KEY_RE = re.compile(
    r"(?:api.?key|authorization|password|secret|token|credential|code_verifier)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"([\"']?(?:[\w-]*(?:api.?key|token|secret|password)|id_token_hint|code_verifier|authorization)[\"']?\s*[:=]\s*)"
    r"(?:[\"'][^\"']*[\"']|[^\s&,}]+)",
    re.IGNORECASE,
)
known_secrets: frozenset[str] = frozenset()


def register_secrets(values: Iterable[str]) -> None:
    global known_secrets
    known_secrets = known_secrets.union(value for value in values if value)


def redact_text(value: str) -> str:
    for secret in sorted(known_secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(https?://)[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(
        r"(Bearer\s+)[^\s\"',}]+", r"\1[REDACTED]", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"([?&](?:key|api_key|token|access_token|refresh_token|id_token|id_token_hint|code|code_verifier|secret|password)=)[^&#\s]*",
        r"\1[REDACTED]",
        value,
        flags=re.IGNORECASE,
    )
    return SECRET_VALUE_RE.sub(r"\1[REDACTED]", value)


def redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY_RE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def setup_logger() -> None:
    if not mw:
        return

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = RedactingFormatter("%(name)s/%(filename)s: [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Can't use config directly here to avoid cyclical import
    config = mw.addonManager.getConfig(__name__)
    if not config:
        return

    is_debug = config.get("debug")

    if is_production() and not is_debug:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)

        if not os.getenv("IS_TEST"):
            try:
                file_handler = logging.FileHandler(
                    get_file_path("smart-notes.log"), mode="w", encoding="utf-8"
                )
            except OSError:
                stream_handler.handle(
                    logger.makeRecord(
                        logger.name,
                        logging.WARNING,
                        __file__,
                        0,
                        "Cannot open the log file; using console logging.",
                        (),
                        None,
                    )
                )
            else:
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            if is_debug:
                logger.debug("Starting app in debug mode")

    logger.addHandler(stream_handler)


logger = logging.getLogger("smart_notes")
logger.propagate = False
