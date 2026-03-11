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
import sys
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from src.mcp_manager import McpManager
from src.mcp_runtime import (
    MCP_PROTOCOL_VERSION,
    McpRuntime,
    McpServerInfo,
    McpServerProbeResult,
    McpToolCallResult,
    McpToolSpec,
)

if TYPE_CHECKING:
    from src.models import McpServerConfig


def make_stdio_server_script(path: Path) -> None:
    script = dedent(
        """
        import json
        import sys

        def read_message():
            content_length = None
            while True:
                line = sys.stdin.buffer.readline()
                if not line:
                    return None
                if line == b"\\r\\n":
                    break
                name, value = line.decode("ascii").split(":", 1)
                if name.lower() == "content-length":
                    content_length = int(value.strip())
            if content_length is None:
                return None
            body = sys.stdin.buffer.read(content_length)
            return json.loads(body)

        def send_message(payload):
            body = json.dumps(payload).encode("utf-8")
            sys.stdout.buffer.write(
                f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("ascii") + body
            )
            sys.stdout.buffer.flush()

        while True:
            message = read_message()
            if message is None:
                break
            method = message.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "serverInfo": {"name": "fake-stdio", "version": "1.0"},
                            "capabilities": {},
                        },
                    }
                )
            elif method == "tools/list":
                send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": "Echo input",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                )
            elif method == "tools/call":
                text = message["params"]["arguments"].get("text", "")
                send_message(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": {
                            "content": [{"type": "text", "text": text}],
                        },
                    }
                )
        """
    ).strip()
    with open(path, "w", encoding="utf-8") as file:
        file.write(script)


@pytest.mark.asyncio
async def test_stdio_mcp_runtime_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.mcp_runtime.get_version", lambda: "test-version")

    server_script = tmp_path / "fake_stdio_server.py"
    make_stdio_server_script(server_script)

    runtime = McpRuntime()
    server = cast(
        "McpServerConfig",
        {
            "id": "stdio-test",
            "name": "Fake Stdio",
            "enabled": True,
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_script)],
            "env": [],
            "env_passthrough": [],
            "cwd": "",
            "url": "",
            "headers": [],
            "header_env_vars": [],
        },
    )

    probe = await runtime.probe_server(server)
    assert probe.server_info == McpServerInfo(name="fake-stdio", version="1.0")
    assert [tool.name for tool in probe.tools] == ["echo"]

    result = await runtime.call_tool(server, "echo", {"text": "hello"})
    assert result.text == "hello"

    await runtime.close_current_session()


@pytest.mark.asyncio
async def test_streamable_http_mcp_runtime_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.mcp_runtime.get_version", lambda: "test-version")

    requests: list[httpx.Request] = []
    session_id = "session-123"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)

        payload = json.loads(request.content.decode("utf-8"))
        assert request.headers["MCP-Protocol-Version"] == MCP_PROTOCOL_VERSION

        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "serverInfo": {"name": "fake-http", "version": "2.0"},
                        "capabilities": {},
                    },
                },
            )

        if payload["method"] == "notifications/initialized":
            assert request.headers["Mcp-Session-Id"] == session_id
            return httpx.Response(
                202,
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": session_id,
                },
            )

        if payload["method"] == "tools/list":
            assert request.headers["Mcp-Session-Id"] == session_id
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo input",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )

        assert request.headers["Mcp-Session-Id"] == session_id
        return httpx.Response(
            200,
            headers={
                "Content-Type": "application/json",
                "Mcp-Session-Id": session_id,
            },
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": payload["params"]["arguments"]["text"],
                        }
                    ]
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def fake_get_http_client() -> httpx.AsyncClient:
        return client

    monkeypatch.setattr(
        "src.mcp_runtime.provider_runtime.get_http_client", fake_get_http_client
    )

    runtime = McpRuntime()
    server = cast(
        "McpServerConfig",
        {
            "id": "http-test",
            "name": "Fake HTTP",
            "enabled": True,
            "transport": "streamable_http",
            "command": "",
            "args": [],
            "env": [],
            "env_passthrough": [],
            "cwd": "",
            "url": "https://example.com/mcp",
            "headers": [],
            "header_env_vars": [],
        },
    )

    probe = await runtime.probe_server(server)
    assert probe.server_info == McpServerInfo(name="fake-http", version="2.0")
    result = await runtime.call_tool(server, "echo", {"text": "hello"})
    assert result.text == "hello"

    await runtime.close_current_session()
    assert requests[-1].method == "DELETE"
    assert requests[-1].headers["Mcp-Session-Id"] == session_id
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_manager_builds_registry_and_executes_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe_server(server: dict[str, Any]) -> McpServerProbeResult:
        del server
        return McpServerProbeResult(
            server_info=McpServerInfo(name="fake", version="1.0"),
            tools=[
                McpToolSpec(
                    name="echo",
                    description="Echo input",
                    input_schema={"type": "object"},
                ),
                McpToolSpec(
                    name="bad",
                    description="Bad schema",
                    input_schema={"type": "string"},
                ),
            ],
        )

    async def fake_call_tool(
        server: dict[str, Any], tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        del server
        return McpToolCallResult(
            text=f"{tool_name}:{arguments['value']}",
            structured_content=None,
            is_error=False,
        )

    monkeypatch.setattr("src.mcp_manager.mcp_runtime.probe_server", fake_probe_server)
    monkeypatch.setattr("src.mcp_manager.mcp_runtime.call_tool", fake_call_tool)

    manager = McpManager(
        [
            {
                "id": "server-1",
                "name": "Server 1",
                "enabled": True,
                "transport": "stdio",
                "command": "cmd",
                "args": [],
                "env": [],
                "env_passthrough": [],
                "cwd": "",
                "url": "",
                "headers": [],
                "header_env_vars": [],
            }
        ]
    )

    tools, warnings = await manager.build_tool_registry()
    assert [tool.exposed_name for tool in tools] == ["mcp_server_1_echo"]
    assert warnings == ["Server 1:bad: inputSchema.type must be object"]

    result = await manager.execute_tool_call("mcp_server_1_echo", {"value": "hello"})
    assert result == "echo:hello"


@pytest.mark.asyncio
async def test_mcp_manager_formats_error_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe_server(server: dict[str, Any]) -> McpServerProbeResult:
        del server
        return McpServerProbeResult(
            server_info=McpServerInfo(name="fake", version="1.0"),
            tools=[
                McpToolSpec(
                    name="broken",
                    description="Broken tool",
                    input_schema={"type": "object"},
                )
            ],
        )

    async def fake_call_tool(
        server: dict[str, Any], tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        del server, tool_name, arguments
        return McpToolCallResult(
            text="failed",
            structured_content={"detail": "boom"},
            is_error=True,
        )

    monkeypatch.setattr("src.mcp_manager.mcp_runtime.probe_server", fake_probe_server)
    monkeypatch.setattr("src.mcp_manager.mcp_runtime.call_tool", fake_call_tool)

    manager = McpManager(
        [
            {
                "id": "server-2",
                "name": "Server 2",
                "enabled": True,
                "transport": "stdio",
                "command": "cmd",
                "args": [],
                "env": [],
                "env_passthrough": [],
                "cwd": "",
                "url": "",
                "headers": [],
                "header_env_vars": [],
            }
        ]
    )

    await manager.build_tool_registry()
    result = await manager.execute_tool_call("mcp_server_2_broken", {})
    assert result.startswith("Tool error:")
