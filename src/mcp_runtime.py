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
from contextlib import asynccontextmanager, suppress
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx

from .logger import logger
from .provider_runtime import RequestTimeouts, build_httpx_timeout, provider_runtime
from .utils import get_version

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from .models import McpKeyValuePair, McpServerConfig

MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_DEFAULT_CONNECT_TIMEOUT_SEC = 10.0
MCP_DEFAULT_READ_TIMEOUT_SEC = 60.0
MCP_MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MCP_MAX_STDERR_BYTES = 64 * 1024
MCP_CLOSE_TIMEOUT_SEC = 2.0
MCP_STDIO_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
)


class McpError(Exception):
    """Base class for MCP runtime errors."""


class McpProtocolError(McpError):
    """Raised when an MCP server returns an invalid response."""


class McpInitializationError(McpError):
    """Raised when an MCP server cannot be initialized."""


@dataclass(frozen=True)
class McpServerInfo:
    name: str
    version: str | None


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class McpToolCallResult:
    text: str
    structured_content: dict[str, Any] | None
    is_error: bool


@dataclass(frozen=True)
class McpServerProbeResult:
    server_info: McpServerInfo
    tools: list[McpToolSpec]


def pairs_to_dict(pairs: list[McpKeyValuePair]) -> dict[str, str]:
    return {pair["key"]: pair["value"] for pair in pairs if pair["key"]}


def build_stdio_env(config: McpServerConfig) -> dict[str, str]:
    env = {key: os.environ[key] for key in MCP_STDIO_ENV if key in os.environ}

    for env_name in config.get("env_passthrough", []):
        if env_name in os.environ:
            env[env_name] = os.environ[env_name]

    for key, value in pairs_to_dict(config.get("env", [])).items():
        env[key] = value

    return env


def build_http_headers(config: McpServerConfig) -> dict[str, str]:
    headers = pairs_to_dict(config.get("headers", []))

    for header_name, env_name in pairs_to_dict(
        config.get("header_env_vars", [])
    ).items():
        env_value = os.getenv(env_name)
        if env_value:
            headers[header_name] = env_value

    return headers


def parse_server_info(payload: Any) -> McpServerInfo:
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP initialize response was not a JSON object.")

    server_info = payload.get("serverInfo", {})
    if not isinstance(server_info, dict):
        raise McpProtocolError("MCP initialize response did not include serverInfo.")

    name = server_info.get("name")
    if not isinstance(name, str) or not name:
        raise McpProtocolError("MCP serverInfo.name was missing.")

    version = server_info.get("version")
    return McpServerInfo(
        name=name, version=version if isinstance(version, str) else None
    )


def parse_tools(payload: Any) -> list[McpToolSpec]:
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP tools/list response was not a JSON object.")

    raw_tools = payload.get("tools", [])
    if not isinstance(raw_tools, list):
        raise McpProtocolError("MCP tools/list response did not include tools.")

    tools: list[McpToolSpec] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            logger.warning("Skipping invalid MCP tool entry")
            continue

        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            logger.warning("Skipping MCP tool without a valid name")
            continue

        description = raw_tool.get("description")
        input_schema = raw_tool.get("inputSchema")
        if not isinstance(input_schema, dict):
            logger.warning("Skipping MCP tool %s with invalid inputSchema", name)
            continue

        tools.append(
            McpToolSpec(
                name=name,
                description=description if isinstance(description, str) else "",
                input_schema=cast("dict[str, Any]", input_schema),
            )
        )

    return tools


def parse_tool_call_result(payload: Any) -> McpToolCallResult:
    if not isinstance(payload, dict):
        raise McpProtocolError("MCP tools/call response was not a JSON object.")

    content = payload.get("content", [])
    text_parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif item_type == "json" and item.get("json") is not None:
                text_parts.append(json.dumps(item["json"], ensure_ascii=True))

    structured_content = payload.get("structuredContent")
    structured = structured_content if isinstance(structured_content, dict) else None
    if structured and not text_parts:
        text_parts.append(json.dumps(structured, ensure_ascii=True))

    return McpToolCallResult(
        text="\n".join(part for part in text_parts if part).strip(),
        structured_content=structured,
        is_error=bool(payload.get("isError")),
    )


def encode_message(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    if len(body) > MCP_MAX_MESSAGE_BYTES:
        raise McpProtocolError("MCP message exceeded the 8 MiB limit.")
    return body


def decode_messages(body: bytes) -> list[dict[str, Any]]:
    if len(body) > MCP_MAX_MESSAGE_BYTES:
        raise McpProtocolError("MCP message exceeded the 8 MiB limit.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise McpProtocolError("MCP response was not valid JSON.") from exc
    messages = payload if isinstance(payload, list) else [payload]
    if not messages or any(
        not isinstance(message, dict) or message.get("jsonrpc") != "2.0"
        for message in messages
    ):
        raise McpProtocolError("MCP response was not a JSON-RPC 2.0 message.")
    return cast("list[dict[str, Any]]", messages)


class BaseMcpSession:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = deepcopy(config)
        self.environment = build_stdio_env(config)
        self.headers = build_http_headers(config)
        self._next_id = 1
        self._lock = asyncio.Lock()
        self.server_info: McpServerInfo | None = None
        self._initialized = False
        self._closed = False

    def matches(self, config: McpServerConfig) -> bool:
        return (
            not self._closed
            and self.config == config
            and self.environment == build_stdio_env(config)
            and self.headers == build_http_headers(config)
        )

    def _take_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    @asynccontextmanager
    async def _operation(self) -> AsyncGenerator[None]:
        async with asyncio.timeout(MCP_DEFAULT_READ_TIMEOUT_SEC), self._lock:
            if self._closed:
                raise McpError(
                    "MCP session is closed; retry with the current configuration."
                )
            try:
                yield
            except (asyncio.CancelledError, McpProtocolError, httpx.HTTPError, OSError):
                # A cancelled request can leave an unread response in the pipe.
                await self.close()
                raise

    async def initialize(self) -> McpServerInfo:
        async with self._operation():
            return await self._initialize_locked()

    async def _initialize_locked(self) -> McpServerInfo:
        if self._initialized and self.server_info is not None:
            return self.server_info
        result = await self._send_request_locked(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "anki-smart-notes",
                    "version": get_version(),
                },
            },
        )
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                "MCP server selected an unsupported protocol version."
            )
        self.server_info = parse_server_info(result)
        await self._send_notification_locked("notifications/initialized", {})
        self._initialized = True
        return self.server_info

    async def list_tools(self) -> list[McpToolSpec]:
        async with self._operation():
            await self._initialize_locked()
            tools: list[McpToolSpec] = []
            params: dict[str, Any] = {}
            seen_cursors: set[str] = set()
            while True:
                result = await self._send_request_locked("tools/list", params)
                tools.extend(parse_tools(result))
                cursor = result.get("nextCursor")
                if cursor is None:
                    return tools
                if not isinstance(cursor, str) or cursor in seen_cursors:
                    raise McpProtocolError(
                        "MCP tools/list returned an invalid pagination cursor."
                    )
                seen_cursors.add(cursor)
                params = {"cursor": cursor}

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        async with self._operation():
            await self._initialize_locked()
            result = await self._send_request_locked(
                "tools/call", {"name": tool_name, "arguments": arguments}
            )
            return parse_tool_call_result(result)

    async def close(self) -> None:
        self._initialized = False
        self._closed = True

    async def _handle_message(
        self, message: dict[str, Any], request_id: int
    ) -> dict[str, Any] | None:
        method = message.get("method")
        if method is not None:
            if "id" in message:
                reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
                if method == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "Method not supported"}
                await self._send_message_locked(reply)
            # Progress/log notifications need no reply. tools/list is never cached.
            return None
        if message.get("id") != request_id:
            return None
        error = message.get("error")
        if isinstance(error, dict):
            raise McpProtocolError(str(error.get("message") or "MCP request failed."))
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError("MCP response did not include a result object.")
        return cast("dict[str, Any]", result)

    async def _send_notification_locked(
        self, method: str, params: dict[str, Any]
    ) -> None:
        await self._send_message_locked(
            {"jsonrpc": "2.0", "method": method, "params": params}
        )

    async def _send_message_locked(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError


class StdioMcpSession(BaseMcpSession):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = bytearray()

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._closed:
            raise McpError("MCP session is closed.")
        if self._process is not None:
            if self._process.returncode is None:
                return self._process
            raise McpProtocolError("MCP stdio server exited unexpectedly.")

        command = self.config.get("command", "").strip()
        if not command:
            raise McpInitializationError("MCP stdio server command is required.")

        self._process = await asyncio.create_subprocess_exec(
            command,
            *self.config.get("args", []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.get("cwd") or None,
            env=self.environment,
            limit=MCP_MAX_MESSAGE_BYTES + 1,
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._process), context=Context()
        )
        return self._process

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while chunk := await process.stderr.read(8192):
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-MCP_MAX_STDERR_BYTES]

    async def close(self) -> None:
        process, self._process = self._process, None
        stderr_task, self._stderr_task = self._stderr_task, None
        await super().close()
        try:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), MCP_CLOSE_TIMEOUT_SEC)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            process.kill()
                        await process.wait()
        finally:
            if process is not None and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)

    async def _send_message_locked(self, payload: dict[str, Any]) -> None:
        process = await self._ensure_process()
        if process.stdin is None:
            raise McpProtocolError("MCP stdio server stdin was unavailable.")
        process.stdin.write(encode_message(payload) + b"\n")
        await process.stdin.drain()

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        request_id = self._take_id()
        await self._send_message_locked(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        process = await self._ensure_process()
        if process.stdout is None:
            raise McpProtocolError("MCP stdio server stdout was unavailable.")
        while True:
            try:
                body = await process.stdout.readline()
            except ValueError as exc:
                raise McpProtocolError("MCP message exceeded the 8 MiB limit.") from exc
            if not body:
                raise McpProtocolError("MCP stdio server closed unexpectedly.")
            if not body.endswith(b"\n"):
                raise McpProtocolError(
                    "MCP stdio server returned an incomplete message."
                )
            for message in decode_messages(body.rstrip(b"\r\n")):
                result = await self._handle_message(message, request_id)
                if result is not None:
                    return result


class StreamableHttpMcpSession(BaseMcpSession):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._session_id: str | None = None

    async def close(self) -> None:
        session_id, self._session_id = self._session_id, None
        await super().close()
        if session_id is None:
            return
        try:
            async with asyncio.timeout(MCP_CLOSE_TIMEOUT_SEC):
                client = await provider_runtime.get_http_client()
                headers = self._request_headers()
                headers["Mcp-Session-Id"] = session_id
                async with client.stream("DELETE", self.config["url"], headers=headers):
                    pass
        except (httpx.HTTPError, TimeoutError):
            logger.debug("Ignoring MCP session close failure")

    async def _send_message_locked(self, payload: dict[str, Any]) -> None:
        await self._request_locked(payload, request_id=None)

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        request_id = self._take_id()
        return await self._request_locked(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            request_id=request_id,
        )

    async def _request_locked(
        self, payload: dict[str, Any], *, request_id: int | None
    ) -> dict[str, Any]:
        if self._closed:
            raise McpError("MCP session is closed.")
        client = await provider_runtime.get_http_client()
        headers = self._request_headers()
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        async with client.stream(
            "POST",
            self.config["url"],
            headers=headers,
            content=encode_message(payload),
            timeout=build_httpx_timeout(
                RequestTimeouts(
                    connect_timeout_sec=MCP_DEFAULT_CONNECT_TIMEOUT_SEC,
                    sock_read_timeout_sec=MCP_DEFAULT_READ_TIMEOUT_SEC,
                )
            ),
        ) as response:
            response.raise_for_status()
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._session_id = session_id
            if request_id is None:
                return {}
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                async for message in self._sse_messages(response):
                    result = await self._handle_message(message, request_id)
                    if result is not None:
                        return result
            else:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MCP_MAX_MESSAGE_BYTES:
                        raise McpProtocolError("MCP message exceeded the 8 MiB limit.")
                for message in decode_messages(bytes(body)):
                    result = await self._handle_message(message, request_id)
                    if result is not None:
                        return result
            raise McpProtocolError(
                "MCP stream ended before the requested response arrived."
            )

    def _request_headers(self) -> dict[str, str]:
        return {
            **self.headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }

    async def _sse_messages(
        self, response: httpx.Response
    ) -> AsyncIterator[dict[str, Any]]:
        pending = bytearray()
        data: list[bytes] = []
        event_size = 0
        async for chunk in response.aiter_bytes():
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, remainder = pending.partition(b"\n")
                pending = remainder
                line = line.rstrip(b"\r")
                event_size += len(line)
                if event_size > MCP_MAX_MESSAGE_BYTES:
                    raise McpProtocolError("MCP SSE event exceeded the 8 MiB limit.")
                if not line:
                    if data:
                        for message in decode_messages(b"\n".join(data)):
                            yield message
                    data = []
                    event_size = 0
                elif line.startswith(b"data:"):
                    data.append(bytes(line[5:]).removeprefix(b" "))
            if len(pending) + event_size > MCP_MAX_MESSAGE_BYTES:
                raise McpProtocolError("MCP SSE event exceeded the 8 MiB limit.")


class McpRuntime:
    def __init__(self) -> None:
        self._sessions: dict[tuple[asyncio.AbstractEventLoop, str], BaseMcpSession] = {}
        self._closing: set[asyncio.Task[None]] = set()

    def _session_key(
        self, config: McpServerConfig
    ) -> tuple[asyncio.AbstractEventLoop, str]:
        return (asyncio.get_running_loop(), config["id"])

    def _build_session(self, config: McpServerConfig) -> BaseMcpSession:
        if config["transport"] == "stdio":
            return StdioMcpSession(config)
        if config["transport"] == "streamable_http":
            return StreamableHttpMcpSession(config)
        raise McpInitializationError(
            f"Unsupported MCP transport: {config['transport']}"
        )

    async def _close_session(self, session: BaseMcpSession) -> None:
        try:
            await session.close()
        except Exception:
            logger.warning("Failed to close MCP session")

    def get_session(self, config: McpServerConfig) -> BaseMcpSession:
        key = self._session_key(config)
        session = self._sessions.get(key)
        if session is not None and not session.matches(config):
            closing = asyncio.create_task(
                self._close_session(session), context=Context()
            )
            self._closing.add(closing)
            closing.add_done_callback(self._closing.discard)
            session = None
        if session is None:
            session = self._build_session(config)
            self._sessions[key] = session
        return session

    async def probe_server(
        self, config: McpServerConfig, *, isolated: bool = False
    ) -> McpServerProbeResult:
        session = self._build_session(config) if isolated else self.get_session(config)
        try:
            server_info = await session.initialize()
            tools = await session.list_tools()
            return McpServerProbeResult(server_info=server_info, tools=tools)
        finally:
            if isolated:
                await session.close()

    async def call_tool(
        self, config: McpServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        session = self.get_session(config)
        return await session.call_tool(tool_name, arguments)

    async def close_current_session(self) -> None:
        loop = asyncio.get_running_loop()
        keys = [key for key in self._sessions if key[0] is loop]
        sessions = [self._sessions.pop(key) for key in keys]
        closing = [task for task in self._closing if task.get_loop() is loop]
        await asyncio.gather(
            *(self._close_session(session) for session in sessions), *closing
        )


mcp_runtime = McpRuntime()
