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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx

from .logger import logger
from .provider_runtime import RequestTimeouts, build_httpx_timeout, provider_runtime
from .utils import get_version

if TYPE_CHECKING:
    from .models import McpKeyValuePair, McpServerConfig

MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_DEFAULT_CONNECT_TIMEOUT_SEC = 10.0
MCP_DEFAULT_READ_TIMEOUT_SEC = 60.0


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
    env = os.environ.copy()

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
            logger.warning("Skipping invalid MCP tool entry: %s", raw_tool)
            continue

        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            logger.warning("Skipping MCP tool without a valid name: %s", raw_tool)
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


class BaseMcpSession:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._next_id = 1
        self._lock = asyncio.Lock()
        self.server_info: McpServerInfo | None = None
        self._initialized = False

    def _take_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    async def initialize(self) -> McpServerInfo:
        async with self._lock:
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
            self.server_info = parse_server_info(result)
            await self._send_notification_locked("notifications/initialized", {})
            self._initialized = True
            return self.server_info

    async def list_tools(self) -> list[McpToolSpec]:
        await self.initialize()
        async with self._lock:
            result = await self._send_request_locked("tools/list", {})
            return parse_tools(result)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        await self.initialize()
        async with self._lock:
            result = await self._send_request_locked(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )
            return parse_tool_call_result(result)

    async def close(self) -> None:
        self._initialized = False

    async def _send_notification_locked(
        self, method: str, params: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError


class StdioMcpSession(BaseMcpSession):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._process: asyncio.subprocess.Process | None = None

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process

        command = self.config.get("command", "").strip()
        if not command:
            raise McpInitializationError("MCP stdio server command is required.")

        logger.debug("Starting MCP stdio server %s", self.config["name"])
        self._process = await asyncio.create_subprocess_exec(
            command,
            *self.config.get("args", []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.get("cwd") or None,
            env=build_stdio_env(self.config),
        )
        return self._process

    async def close(self) -> None:
        process = self._process
        self._process = None
        await super().close()

        if process is None:
            return

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    async def _send_notification_locked(
        self, method: str, params: dict[str, Any]
    ) -> None:
        process = await self._ensure_process()
        await self._write_message(
            process, {"jsonrpc": "2.0", "method": method, "params": params}
        )

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        process = await self._ensure_process()
        request_id = self._take_id()
        await self._write_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        response = await self._read_response(process)
        if response.get("id") != request_id:
            raise McpProtocolError(
                f"MCP stdio server returned mismatched id {response.get('id')}"
            )

        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message") or json.dumps(error, ensure_ascii=True)
            raise McpProtocolError(str(message))

        result = response.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError(
                "MCP stdio response did not include a result object."
            )
        return cast("dict[str, Any]", result)

    async def _write_message(
        self, process: asyncio.subprocess.Process, payload: dict[str, Any]
    ) -> None:
        if process.stdin is None:
            raise McpProtocolError("MCP stdio server stdin was unavailable.")

        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        process.stdin.write(headers + body)
        await process.stdin.drain()

    async def _read_response(
        self, process: asyncio.subprocess.Process
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise McpProtocolError("MCP stdio server stdout was unavailable.")

        content_length: int | None = None
        while True:
            line = await process.stdout.readline()
            if not line:
                stderr_text = await self._read_stderr(process)
                raise McpProtocolError(
                    f"MCP stdio server closed unexpectedly. {stderr_text}".strip()
                )

            if line == b"\r\n":
                break

            try:
                header_name, header_value = line.decode("ascii").split(":", 1)
            except ValueError as exc:
                raise McpProtocolError(f"Invalid MCP stdio header: {line!r}") from exc

            if header_name.lower() == "content-length":
                content_length = int(header_value.strip())

        if content_length is None:
            raise McpProtocolError("MCP stdio response was missing Content-Length.")

        body = await process.stdout.readexactly(content_length)
        try:
            response = json.loads(body)
        except json.JSONDecodeError as exc:
            raise McpProtocolError(
                "MCP stdio response body was not valid JSON."
            ) from exc

        if not isinstance(response, dict):
            raise McpProtocolError("MCP stdio response body was not a JSON object.")
        return cast("dict[str, Any]", response)

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> str:
        if process.stderr is None:
            return ""

        try:
            data = await asyncio.wait_for(process.stderr.read(), timeout=0.1)
        except asyncio.TimeoutError:
            return ""
        return data.decode("utf-8", errors="replace").strip()


class StreamableHttpMcpSession(BaseMcpSession):
    def __init__(self, config: McpServerConfig) -> None:
        super().__init__(config)
        self._session_id: str | None = None

    async def close(self) -> None:
        session_id = self._session_id
        self._session_id = None
        await super().close()

        if session_id is None:
            return

        client = await provider_runtime.get_http_client()
        headers = self._request_headers()
        headers["Mcp-Session-Id"] = session_id
        try:
            await client.request(
                "DELETE",
                self.config["url"],
                headers=headers,
                timeout=build_httpx_timeout(
                    RequestTimeouts(
                        connect_timeout_sec=MCP_DEFAULT_CONNECT_TIMEOUT_SEC,
                        sock_read_timeout_sec=MCP_DEFAULT_READ_TIMEOUT_SEC,
                    )
                ),
            )
        except httpx.HTTPError:
            logger.debug(
                "Ignoring MCP session close failure for %s", self.config["name"]
            )

    async def _send_notification_locked(
        self, method: str, params: dict[str, Any]
    ) -> None:
        await self._request_locked(
            {"jsonrpc": "2.0", "method": method, "params": params},
            expect_response=False,
        )

    async def _send_request_locked(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        request_id = self._take_id()
        response = await self._request_locked(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
            expect_response=True,
        )
        if response.get("id") != request_id:
            raise McpProtocolError(
                f"MCP HTTP server returned mismatched id {response.get('id')}"
            )

        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message") or json.dumps(error, ensure_ascii=True)
            raise McpProtocolError(str(message))

        result = response.get("result")
        if not isinstance(result, dict):
            raise McpProtocolError("MCP HTTP response did not include a result object.")
        return cast("dict[str, Any]", result)

    async def _request_locked(
        self, payload: dict[str, Any], *, expect_response: bool
    ) -> dict[str, Any]:
        client = await provider_runtime.get_http_client()
        headers = self._request_headers()
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        response = await client.request(
            "POST",
            self.config["url"],
            headers=headers,
            json=payload,
            timeout=build_httpx_timeout(
                RequestTimeouts(
                    connect_timeout_sec=MCP_DEFAULT_CONNECT_TIMEOUT_SEC,
                    sock_read_timeout_sec=MCP_DEFAULT_READ_TIMEOUT_SEC,
                )
            ),
        )
        response.raise_for_status()

        response_session_id = response.headers.get("Mcp-Session-Id")
        if response_session_id:
            self._session_id = response_session_id

        if not expect_response and not response.content:
            return {}

        return self._parse_http_response(response)

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        headers.update(build_http_headers(self.config))
        return headers

    def _parse_http_response(self, response: httpx.Response) -> dict[str, Any]:
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse_response(response.text)

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise McpProtocolError(
                "MCP HTTP response body was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise McpProtocolError("MCP HTTP response body was not a JSON object.")
        return cast("dict[str, Any]", payload)

    def _parse_sse_response(self, body: str) -> dict[str, Any]:
        events = []
        for chunk in body.split("\n\n"):
            lines = [line for line in chunk.splitlines() if line.startswith("data:")]
            if not lines:
                continue
            payload = "\n".join(line[5:].strip() for line in lines)
            if payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise McpProtocolError(
                    "MCP HTTP SSE payload was not valid JSON."
                ) from exc
            if isinstance(event, dict):
                events.append(event)

        if not events:
            raise McpProtocolError("MCP HTTP SSE response did not contain any events.")
        return cast("dict[str, Any]", events[-1])


class McpRuntime:
    def __init__(self) -> None:
        self._sessions: dict[tuple[asyncio.AbstractEventLoop, str], BaseMcpSession] = {}

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

    def get_session(self, config: McpServerConfig) -> BaseMcpSession:
        key = self._session_key(config)
        session = self._sessions.get(key)
        if session is None:
            session = self._build_session(config)
            self._sessions[key] = session
        return session

    async def probe_server(self, config: McpServerConfig) -> McpServerProbeResult:
        session = self.get_session(config)
        server_info = await session.initialize()
        tools = await session.list_tools()
        return McpServerProbeResult(server_info=server_info, tools=tools)

    async def call_tool(
        self, config: McpServerConfig, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        session = self.get_session(config)
        return await session.call_tool(tool_name, arguments)

    async def close_current_session(self) -> None:
        loop = asyncio.get_running_loop()
        keys_to_remove = [key for key in self._sessions if key[0] is loop]
        sessions = [self._sessions.pop(key) for key in keys_to_remove]
        for session in sessions:
            try:
                await session.close()
            except Exception as exc:
                logger.warning("Failed to close MCP session: %s", exc)


mcp_runtime = McpRuntime()
