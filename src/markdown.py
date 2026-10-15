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

from __future__ import annotations

import re
from html import escape
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from markdown_it.rules_inline.state_inline import StateInline

SAFE_FORMATTING_HTML_TAG = re.compile(
    r"(</?(?:b|strong|i|em|u|s|sub|sup|br)[ \t]*/?>)", re.IGNORECASE
)


def parse_formatting_html(state: StateInline, silent: bool) -> bool:
    """Recognize only basic formatting tags without attributes."""
    match = SAFE_FORMATTING_HTML_TAG.match(state.src, state.pos)
    if match is None or match.end() > state.posMax:
        return False
    if not silent:
        token = state.push("html_inline", "", 0)
        token.content = match.group()
    state.pos = match.end()
    return True


markdown_parser = MarkdownIt("commonmark", {"html": False, "breaks": True})
markdown_parser.inline.ruler.before(
    "html_inline", "formatting_html", parse_formatting_html
)


def convert_markdown_to_html(markdown: str) -> str:
    return markdown_parser.render(markdown)


def convert_text_to_html(text: str) -> str:
    formatted = "".join(
        part if SAFE_FORMATTING_HTML_TAG.fullmatch(part) else escape(part)
        for part in SAFE_FORMATTING_HTML_TAG.split(text)
    )
    return formatted.replace("\r\n", "\n").replace("\n", "<br>")
