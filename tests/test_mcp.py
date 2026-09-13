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

import asyncio
import json
import os
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from src.mcp_manager import McpManager
from src.mcp_runtime import (
    MCP_MAX_STDERR_BYTES,
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    McpRuntime,
    McpServerInfo,
    McpServerProbeResult,
    McpToolCallResult,
    McpToolSpec,
    StdioMcpSession,
    StreamableHttpMcpSession,
    build_stdio_env,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from src.models import McpServerConfig


@pytest.fixture(autouse=True)
def mcp_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.mcp_runtime.get_version", lambda: "test-version")


def make_stdio_server_script(path: Path) -> None:
    script = dedent(
        """
        import json
        import sys
        import os
        import time
        import signal

        if len(sys.argv) > 1 and sys.argv[1] == "ignore-term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

        def read_message():
            line = sys.stdin.buffer.readline()
            return json.loads(line) if line else None

        def send_message(payload):
            sys.stdout.buffer.write(json.dumps(payload).encode("utf-8") + b"\\n")
            sys.stdout.buffer.flush()

        while True:
            message = read_message()
            if message is None:
                break
            method = message.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                sys.stderr.buffer.write(b"diagnostic" * 20000)
                sys.stderr.buffer.flush()
                send_message({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
                send_message({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"})
                assert read_message() == {"jsonrpc": "2.0", "id": "server-ping", "result": {}}
                send_message({"jsonrpc": "2.0", "id": "server-request", "method": "sampling/createMessage"})
                reply = read_message()
                assert reply["id"] == "server-request" and reply["error"]["code"] == -32601
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
                if message["params"]["arguments"].get("noisy"):
                    sys.stderr.buffer.write(b"busy" * 50000)
                    sys.stderr.buffer.flush()
                if message["params"]["name"] == "hang":
                    time.sleep(600)
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


def make_mcp_config(
    command: str = "", args: list[str] | None = None
) -> McpServerConfig:
    return {
        "id": "test",
        "name": "Test",
        "enabled": True,
        "transport": "stdio" if command else "streamable_http",
        "command": command,
        "args": args or [],
        "env": [],
        "env_passthrough": [],
        "cwd": "",
        "url": "https://example.com/mcp",
        "headers": [],
        "header_env_vars": [],
    }


def test_stdio_environment_requires_explicit_secret_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("UNRELATED_API_KEY", "not-inherited")
    monkeypatch.setenv("ALLOWED_TOKEN", "explicit-token")
    server = make_mcp_config(sys.executable)
    server["env_passthrough"] = ["ALLOWED_TOKEN"]
    server["env"] = [{"key": "PATH", "value": "/custom/bin"}]
    environment = build_stdio_env(server)
    assert "UNRELATED_API_KEY" not in environment
    assert environment["ALLOWED_TOKEN"] == "explicit-token"
    assert environment["PATH"] == "/custom/bin"


@pytest.mark.asyncio
async def test_changed_config_and_environment_retire_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = McpRuntime()
    server = make_mcp_config()
    first = runtime.get_session(server)
    assert runtime.get_session(server) is first
    server["url"] = "https://other.example.com/mcp"
    second = runtime.get_session(server)
    assert second is not first
    assert first.config["url"] == "https://example.com/mcp"
    server["header_env_vars"] = [{"key": "Authorization", "value": "MCP_TOKEN"}]
    monkeypatch.setenv("MCP_TOKEN", "first")
    third = runtime.get_session(server)
    monkeypatch.setenv("MCP_TOKEN", "second")
    fourth = runtime.get_session(server)
    assert fourth is not third
    assert third.headers["Authorization"] == "first"
    await runtime.close_current_session()
    assert all(
        not session.matches(session.config)
        for session in (first, second, third, fourth)
    )


class InspectableStdioSession(StdioMcpSession):
    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def diagnostics(self) -> bytearray:
        return self._stderr_tail

    @property
    def stderr_task(self) -> asyncio.Task[None] | None:
        return self._stderr_task


@pytest.mark.asyncio
async def test_stdio_cancellation_and_timeout_reap_process_and_allow_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server_script = tmp_path / "server.py"
    make_stdio_server_script(server_script)
    server = make_mcp_config(sys.executable, [str(server_script)])
    runtime = McpRuntime()
    monkeypatch.setattr("src.mcp_runtime.StdioMcpSession", InspectableStdioSession)
    session = cast("InspectableStdioSession", runtime.get_session(server))
    await session.initialize()
    assert len(session.diagnostics) <= MCP_MAX_STDERR_BYTES
    assert session.diagnostics
    process = session.process
    assert process is not None
    task = asyncio.create_task(session.call_tool("hang", {}))
    await asyncio.sleep(0.02)
    queued = asyncio.create_task(session.call_tool("echo", {"text": "queued"}))
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert process.returncode is None and session.matches(server)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode is not None
    assert session.stderr_task is None
    result = await runtime.call_tool(server, "echo", {"text": "after cancel"})
    assert result.text == "after cancel"
    monkeypatch.setattr("src.mcp_runtime.MCP_DEFAULT_READ_TIMEOUT_SEC", 0.05)
    with pytest.raises(TimeoutError):
        await runtime.call_tool(server, "hang", {})
    monkeypatch.setattr("src.mcp_runtime.MCP_DEFAULT_READ_TIMEOUT_SEC", 60.0)
    assert (
        await runtime.call_tool(server, "echo", {"text": "after timeout"})
    ).text == "after timeout"
    await runtime.close_current_session()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal handling")
async def test_stdio_shutdown_kills_process_ignoring_termination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server_script = tmp_path / "server.py"
    make_stdio_server_script(server_script)
    server = make_mcp_config(sys.executable, [str(server_script), "ignore-term"])
    session = InspectableStdioSession(server)
    await session.initialize()
    process = session.process
    assert process is not None
    monkeypatch.setattr("src.mcp_runtime.MCP_CLOSE_TIMEOUT_SEC", 0.02)
    async with asyncio.timeout(1):
        await session.close()
    assert process.returncode is not None


class McpEventStream(httpx.AsyncByteStream):
    def __init__(
        self, events: list[dict[str, Any]], newline: str, hold_open: bool
    ) -> None:
        self.events = events
        self.newline = newline
        self.hold_open = hold_open
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            encoded = (
                f"data: {json.dumps(event)}{self.newline}{self.newline}"
            ).encode()
            for offset in range(0, len(encoded), 7):
                yield encoded[offset : offset + 7]
        if self.hold_open:
            await asyncio.sleep(600)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_http_stream_matches_id_handles_server_requests_and_paginates(
    monkeypatch: pytest.MonkeyPatch, newline: str
) -> None:
    streams: list[McpEventStream] = []
    replies: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        method = payload.get("method")
        if method is None:
            replies.append(payload)
            return httpx.Response(202)
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {"name": "stream-test"},
                "capabilities": {"tools": {}},
            }
        else:
            pages.append(payload["params"])
            result = {
                "tools": [
                    {
                        "name": "second" if payload["params"] else "first",
                        "inputSchema": {"type": "object"},
                    }
                ],
            }
            if not payload["params"]:
                result["nextCursor"] = "page-2"
        stream = McpEventStream(
            [
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {"progress": 1},
                },
                {"jsonrpc": "2.0", "id": "server-ping", "method": "ping"},
                {"jsonrpc": "2.0", "id": "unsupported", "method": "roots/list"},
                {"jsonrpc": "2.0", "id": 999, "result": {"ignored": True}},
                {"jsonrpc": "2.0", "id": payload["id"], "result": result},
            ],
            newline,
            hold_open=True,
        )
        streams.append(stream)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=stream
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def get_client() -> httpx.AsyncClient:
            return client

        monkeypatch.setattr(
            "src.mcp_runtime.provider_runtime.get_http_client", get_client
        )
        runtime = McpRuntime()
        async with asyncio.timeout(1):
            probe = await runtime.probe_server(make_mcp_config())
        assert [tool.name for tool in probe.tools] == ["first", "second"]
        assert pages == [{}, {"cursor": "page-2"}]
        assert all(stream.closed for stream in streams)
        assert all(
            reply.get("result") == {}
            for reply in replies
            if reply["id"] == "server-ping"
        )
        assert all(
            reply["error"]["code"] == -32601
            for reply in replies
            if reply["id"] == "unsupported"
        )
        await runtime.close_current_session()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["eof", "oversize", "timeout"])
async def test_http_stream_failures_are_bounded_and_close_transport(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    event: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"text": "x" * 256},
    }
    stream = McpEventStream([event], "\r\n", hold_open=failure == "timeout")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=stream
        )

    if failure == "oversize":
        monkeypatch.setattr("src.mcp_runtime.MCP_MAX_MESSAGE_BYTES", 256)
    if failure == "timeout":
        monkeypatch.setattr("src.mcp_runtime.MCP_DEFAULT_READ_TIMEOUT_SEC", 0.02)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

        async def get_client() -> httpx.AsyncClient:
            return client

        monkeypatch.setattr(
            "src.mcp_runtime.provider_runtime.get_http_client", get_client
        )
        session = StreamableHttpMcpSession(make_mcp_config())
        with pytest.raises(TimeoutError if failure == "timeout" else McpProtocolError):
            await session.initialize()
        assert stream.closed
        assert not session.matches(session.config)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "streamable_http"])
async def test_independent_sdk_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, transport: str
) -> None:
    executable = os.environ.get("MCP_TEST_PYTHON")
    if not executable:
        pytest.skip(
            "Set MCP_TEST_PYTHON to a Python environment containing the test-only MCP SDK"
        )
    script = tmp_path / "sdk_server.py"
    script.write_text(
        dedent("""
        import socket
        import sys
        import uvicorn
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("Independent SDK", stateless_http=True)
        @server.tool()
        def echo(text: str) -> str:
            return text
        if sys.argv[1] == "stdio":
            server.run(transport="stdio")
        else:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            print(sock.getsockname()[1], flush=True)
            uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="error")).run(sockets=[sock])
    """),
        encoding="utf-8",
    )
    runtime = McpRuntime()
    server = make_mcp_config(executable, [str(script), transport])
    process: asyncio.subprocess.Process | None = None
    async with httpx.AsyncClient() as client:

        async def get_client() -> httpx.AsyncClient:
            return client

        monkeypatch.setattr(
            "src.mcp_runtime.provider_runtime.get_http_client", get_client
        )
        try:
            if transport == "streamable_http":
                process = await asyncio.create_subprocess_exec(
                    executable,
                    str(script),
                    transport,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                assert process.stdout is not None
                async with asyncio.timeout(10):
                    port = int(await process.stdout.readline())
                    server = make_mcp_config()
                    server["url"] = f"http://127.0.0.1:{port}/mcp"
                    while True:
                        try:
                            await client.get(server["url"])
                            break
                        except httpx.TransportError:
                            await asyncio.sleep(0.02)
            probe = await runtime.probe_server(server)
            assert probe.server_info.name == "Independent SDK"
            assert [tool.name for tool in probe.tools] == ["echo"]
            assert (
                await runtime.call_tool(server, "echo", {"text": "SDK round trip"})
            ).text == "SDK round trip"
        finally:
            await runtime.close_current_session()
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except TimeoutError:
                    process.kill()
                    await process.wait()


@pytest.mark.asyncio
async def test_isolated_draft_probe_preserves_production_session(
    tmp_path: Path,
) -> None:
    script = tmp_path / "stdio_server.py"
    make_stdio_server_script(script)
    server = make_mcp_config(sys.executable, [str(script)])
    runtime = McpRuntime()
    await runtime.probe_server(server)
    production = runtime.get_session(server)
    draft = {**server, "env": [{"key": "DRAFT_SETTING", "value": "unsaved"}]}
    probe = await runtime.probe_server(cast("McpServerConfig", draft), isolated=True)
    assert probe.server_info.name == "fake-stdio"
    assert runtime.get_session(server) is production
    assert production.matches(server)
    assert (
        await runtime.call_tool(server, "echo", {"text": "still production"})
    ).text == "still production"
    await runtime.close_current_session()
