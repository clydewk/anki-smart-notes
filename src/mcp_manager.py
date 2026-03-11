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

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .logger import logger
from .mcp_runtime import (
    McpServerProbeResult,
    McpToolCallResult,
    McpToolSpec,
    mcp_runtime,
)

if TYPE_CHECKING:
    from .models import McpServerConfig

MCP_EXPOSED_TOOL_PREFIX = "mcp"
MCP_MAX_TOOL_NAME_LENGTH = 64


@dataclass(frozen=True)
class McpExposedTool:
    server: McpServerConfig
    original_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]


def sanitize_tool_segment(raw_value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", raw_value).strip("_").lower()
    return clean or "tool"


def build_exposed_tool_name(server_id: str, tool_name: str) -> str:
    server_segment = sanitize_tool_segment(server_id)
    tool_segment = sanitize_tool_segment(tool_name)
    base = f"{MCP_EXPOSED_TOOL_PREFIX}_{server_segment}_{tool_segment}"
    if len(base) <= MCP_MAX_TOOL_NAME_LENGTH:
        return base

    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    trimmed = base[: MCP_MAX_TOOL_NAME_LENGTH - len(digest) - 1].rstrip("_")
    return f"{trimmed}_{digest}"


def format_tool_result(result: McpToolCallResult) -> str:
    parts: list[str] = []
    if result.text:
        parts.append(result.text)

    if result.structured_content:
        structured = json.dumps(result.structured_content, ensure_ascii=True)
        if not parts or parts[-1] != structured:
            parts.append(structured)

    if not parts:
        parts.append("")

    output = "\n".join(part for part in parts if part).strip()
    if result.is_error:
        output = f"Tool error: {output}" if output else "Tool error"
    return output


class McpManager:
    def __init__(self, server_configs: list[McpServerConfig]) -> None:
        self._server_configs = server_configs
        self._tool_map: dict[str, McpExposedTool] = {}

    def enabled_servers(self) -> list[McpServerConfig]:
        return [
            server for server in self._server_configs if server.get("enabled", True)
        ]

    async def build_tool_registry(self) -> tuple[list[McpExposedTool], list[str]]:
        tools: list[McpExposedTool] = []
        warnings: list[str] = []
        self._tool_map = {}

        for server in self.enabled_servers():
            try:
                probe = await mcp_runtime.probe_server(server)
            except Exception as exc:
                warning = f"{server['name']}: {exc}"
                warnings.append(warning)
                logger.warning("Skipping MCP server %s: %s", server["name"], exc)
                continue

            server_tools, server_warnings = self._build_server_tools(server, probe)
            warnings.extend(server_warnings)
            for tool in server_tools:
                if tool.exposed_name in self._tool_map:
                    warning = (
                        f"{server['name']}:{tool.original_name}: duplicate exposed tool name "
                        f"{tool.exposed_name}"
                    )
                    warnings.append(warning)
                    logger.warning(warning)
                    continue
                self._tool_map[tool.exposed_name] = tool
                tools.append(tool)
        return tools, warnings

    async def execute_tool_call(
        self, exposed_name: str, arguments: dict[str, Any]
    ) -> str:
        tool = self._tool_map.get(exposed_name)
        if tool is None:
            raise Exception(f"Unknown MCP tool: {exposed_name}")

        result = await mcp_runtime.call_tool(tool.server, tool.original_name, arguments)
        return format_tool_result(result)

    def _build_server_tools(
        self, server: McpServerConfig, probe: McpServerProbeResult
    ) -> tuple[list[McpExposedTool], list[str]]:
        tools: list[McpExposedTool] = []
        warnings: list[str] = []

        for tool in probe.tools:
            try:
                tools.append(self._build_tool(server, probe, tool))
            except Exception as exc:
                warning = f"{server['name']}:{tool.name}: {exc}"
                warnings.append(warning)
                logger.warning(
                    "Skipping MCP tool %s on %s: %s", tool.name, server["name"], exc
                )

        return tools, warnings

    def _build_tool(
        self,
        server: McpServerConfig,
        probe: McpServerProbeResult,
        tool: McpToolSpec,
    ) -> McpExposedTool:
        exposed_name = build_exposed_tool_name(server["id"], tool.name)
        if tool.input_schema.get("type") != "object":
            raise ValueError("inputSchema.type must be object")

        description = tool.description.strip() or (
            f"Call the {tool.name} tool from MCP server {probe.server_info.name}."
        )
        return McpExposedTool(
            server=server,
            original_name=tool.name,
            exposed_name=exposed_name,
            description=description,
            input_schema=tool.input_schema,
        )
