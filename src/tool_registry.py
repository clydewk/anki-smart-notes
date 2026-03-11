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

from typing import TYPE_CHECKING, Any

from .built_in_tools import BuiltInToolContext, BuiltInToolProvider
from .chat_provider import TextToolDefinition
from .mcp_manager import McpManager

if TYPE_CHECKING:
    from .models import BuiltInToolsConfig, McpServerConfig


class ToolRegistry:
    def __init__(
        self,
        *,
        context: BuiltInToolContext,
        built_in_tools: BuiltInToolsConfig,
        mcp_servers: list[McpServerConfig],
    ) -> None:
        self._built_in_provider = BuiltInToolProvider(context, built_in_tools)
        self._mcp_manager = McpManager(mcp_servers)

    async def build_tool_registry(self) -> tuple[list[TextToolDefinition], list[str]]:
        built_in_tools = self._built_in_provider.build_tool_registry()
        mcp_tools, warnings = await self._mcp_manager.build_tool_registry()

        tools = list(built_in_tools)
        tools.extend(
            [
                TextToolDefinition(
                    name=tool.exposed_name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                )
                for tool in mcp_tools
            ]
        )
        return tools, warnings

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._built_in_provider.handles(tool_name):
            return await self._built_in_provider.execute_tool_call(tool_name, arguments)
        return await self._mcp_manager.execute_tool_call(tool_name, arguments)
