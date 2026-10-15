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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.logger import (
    RedactingFormatter,
    redact,
    redact_text,
    register_secrets,
    setup_logger,
)
from src.markdown import convert_markdown_to_html, convert_text_to_html

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("convert", [convert_markdown_to_html, convert_text_to_html])
def test_generated_sentence_hint_preserves_basic_html_formatting(
    convert: "Callable[[str], str]",
) -> None:
    sentence = "明日<b>食べます</b>。<br>田中さんの<strong>お子さん</strong>です。"
    rendered = convert(sentence)
    assert sentence in rendered
    assert "&lt;b&gt;" not in rendered
    assert "&lt;/b&gt;" not in rendered


@pytest.mark.parametrize("convert", [convert_markdown_to_html, convert_text_to_html])
def test_generated_html_formatting_does_not_allow_attributes_or_active_tags(
    convert: "Callable[[str], str]",
) -> None:
    rendered = convert('<b onclick="alert(1)">unsafe</b><script>alert(2)</script>')
    assert "<b onclick" not in rendered
    assert "<script>" not in rendered
    assert "&lt;b onclick=" in rendered
    assert "&lt;script&gt;" in rendered


def test_markdown_keeps_html_literal_in_code_and_escaped_text() -> None:
    rendered = convert_markdown_to_html(
        "`<b>inline code</b>`\n\n```html\n<b>block code</b>\n```\n\n"
        r"\<b>escaped\</b> &lt;b&gt;entity&lt;/b&gt;"
    )
    assert "<code>&lt;b&gt;inline code&lt;/b&gt;</code>" in rendered
    assert "&lt;b&gt;block code&lt;/b&gt;" in rendered
    assert "&lt;b&gt;escaped&lt;/b&gt;" in rendered
    assert "&lt;b&gt;entity&lt;/b&gt;" in rendered
    assert "<b>" not in rendered


def test_markdown_preserves_code_and_escapes_active_content() -> None:
    rendered = convert_markdown_to_html(
        "# Heading\n\nunder_score_name **bold**\nnext line\n\n"
        "```python\na_b = '<script>alert(1)</script>'\n```\n\n"
        "<img src=x onerror=alert(1)>\n[unsafe](javascript:alert%281%29)"
    )
    assert "<h1>Heading</h1>" in rendered
    assert "under_score_name <strong>bold</strong>" in rendered
    assert "<br />" in rendered
    assert "a_b = '&lt;script&gt;alert(1)&lt;/script&gt;'" in rendered
    assert "<img" not in rendered
    assert 'href="javascript:' not in rendered
    assert "&lt;img" in rendered


def test_plain_text_is_escaped_before_note_html() -> None:
    assert convert_text_to_html("<script>x</script>&\r\n**literal**") == (
        "&lt;script&gt;x&lt;/script&gt;&amp;<br>**literal**"
    )


def test_recursive_and_traceback_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.logger.known_secrets", frozenset())
    register_secrets(["private-draft-key"])
    value = {
        "providers": [{"api_key": "abc", "headers": {"Authorization": "Bearer xyz"}}]
    }
    assert redact(value) == {
        "providers": [
            {"api_key": "[REDACTED]", "headers": {"Authorization": "[REDACTED]"}}
        ]
    }
    diagnostic = redact_text(
        "https://user:pass@host/path?key=URLSECRET&safe=value "
        'Authorization: Bearer xyz api_key="quoted secret" private-draft-key'
    )
    for secret in (
        "user:pass",
        "URLSECRET",
        "xyz",
        "quoted secret",
        "private-draft-key",
    ):
        assert secret not in diagnostic
    try:
        raise ValueError("private-draft-key")
    except ValueError as error:
        record = logging.LogRecord(
            "test",
            logging.ERROR,
            "",
            0,
            "%s",
            (error,),
            (type(error), error, error.__traceback__),
        )
    rendered = RedactingFormatter().format(record)
    assert "private-draft-key" not in rendered
    assert "ValueError: [REDACTED]" in rendered


def test_chatgpt_oauth_diagnostics_redact_credentials() -> None:
    diagnostic = redact_text(
        "https://auth.openai.com/api/accounts/authorize?client_id=oaiapp_example"
        "&id_token_hint=identity-secret&code=callback-secret&code_verifier=pkce-secret "
        'id_token_hint="quoted-identity" code_verifier="quoted-verifier"'
    )
    for secret in (
        "identity-secret",
        "callback-secret",
        "pkce-secret",
        "quoted-identity",
        "quoted-verifier",
    ):
        assert secret not in diagnostic
    assert "client_id=oaiapp_example" in diagnostic
    assert redact({"id_token_hint": "identity", "code_verifier": "verifier"}) == {
        "id_token_hint": "[REDACTED]",
        "code_verifier": "[REDACTED]",
    }


def test_logger_survives_unwritable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IS_TEST")
    monkeypatch.setattr("src.logger.mw", MagicMock())
    monkeypatch.setattr("src.logger.logger", logging.Logger("isolated"))
    monkeypatch.setattr("src.logger.is_production", lambda: False)
    monkeypatch.setattr(
        "src.logger.logging.FileHandler", MagicMock(side_effect=PermissionError)
    )
    setup_logger()
