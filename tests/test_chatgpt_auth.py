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
import base64
import hashlib
import json
import socket
import stat
import time
from threading import Event
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from src.chatgpt_auth import (
    CLIENT_ID,
    ChatGPTAuth,
    ChatGPTAuthError,
    account_epoch,
    parse_credentials,
)
from src.job_context import JobCancelledError, JobContext, current_job

if TYPE_CHECKING:
    from pathlib import Path


def jwt(claims: dict[str, Any]) -> str:
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"header.{payload}.signature"


def token_response() -> dict[str, Any]:
    return {
        "access_token": "fake-access-secret",
        "refresh_token": "fake-refresh-secret",
        "expires_in": 3600,
        "id_token": jwt(
            {
                "email": "test@example.test",
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "account-test",
                    "chatgpt_plan_type": "plus",
                },
            }
        ),
    }


def write_credentials(path: Path, *, expired: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "account": {
                    "email": "test@example.test",
                    "account_id": "account-test",
                    "plan": "plus",
                },
                "access_token": "old-access-secret",
                "refresh_token": "old-refresh-secret",
                "expires_at": time.time() + (-10 if expired else 3600),
            }
        )
    )


@pytest.fixture
def oauth_port(monkeypatch: pytest.MonkeyPatch) -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    monkeypatch.setattr("src.chatgpt_auth.CALLBACK_PORT", port)
    return port


@pytest.fixture
def token_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    client.stream.return_value = AsyncMock()
    client.stream.return_value.__aenter__.return_value = httpx.Response(
        200, json=token_response()
    )
    monkeypatch.setattr(
        "src.chatgpt_auth.provider_runtime.get_http_client",
        AsyncMock(return_value=client),
    )
    return client


async def callback_request(port: int, state: str, code: str = "test-code") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    query = urlencode({"state": state, "code": code})
    writer.write(
        f"GET /auth/callback?{query} HTTP/1.1\r\nHost: localhost:{port}\r\n\r\n".encode()
    )
    await writer.drain()
    result = await reader.read()
    writer.close()
    await writer.wait_closed()
    return result


def test_account_metadata_and_secrets_do_not_appear_in_repr() -> None:
    credentials = parse_credentials(token_response())
    assert credentials.account.account_id == "account-test"
    assert credentials.account.email == "test@example.test"
    assert credentials.account.plan == "plus"
    assert "fake-access-secret" not in repr(credentials)
    assert "fake-refresh-secret" not in repr(credentials)


def test_sign_out_only_removes_own_credentials(tmp_path: Path) -> None:
    own, other = tmp_path / "chatgpt.json", tmp_path / "other-app.json"
    write_credentials(own)
    write_credentials(other)
    auth = ChatGPTAuth(own)
    assert auth.status() is not None
    auth.sign_out()
    assert auth.status() is None
    assert not own.exists()
    assert other.exists()


async def test_browser_pkce_state_and_private_persistence(
    tmp_path: Path, oauth_port: int, token_client: MagicMock
) -> None:
    auth = ChatGPTAuth(tmp_path / "private/chatgpt.json")
    opened: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(auth.sign_in(opened.set_result))
    url = await opened
    query = parse_qs(urlsplit(url).query)
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == [f"http://localhost:{oauth_port}/auth/callback"]
    assert query["originator"] == ["anki_smart_notes"]
    assert query["code_challenge_method"] == ["S256"]
    invalid = await callback_request(oauth_port, "wrong-state")
    assert b"400 Bad Request" in invalid
    assert not task.done()
    valid = await callback_request(oauth_port, query["state"][0])
    assert b"200 OK" in valid
    account = await task
    assert account.email == "test@example.test"
    data = token_client.stream.call_args.kwargs["data"]
    expected = (
        base64.urlsafe_b64encode(
            hashlib.sha256(data["code_verifier"].encode()).digest()
        )
        .rstrip(b"=")
        .decode()
    )
    assert query["code_challenge"] == [expected]
    assert token_client.stream.call_args.kwargs["follow_redirects"] is False
    path = tmp_path / "private/chatgpt.json"
    assert ChatGPTAuth(path).status() == account
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


async def test_cancel_sign_in_invalidates_and_releases_listener(
    tmp_path: Path, oauth_port: int, token_client: MagicMock
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    opened: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(auth.sign_in(opened.set_result))
    await opened
    previous = auth.generation
    auth.cancel_sign_in()
    assert auth.generation > previous
    with pytest.raises(asyncio.CancelledError):
        await task
    server = await asyncio.start_server(
        lambda _reader, writer: writer.close(), "127.0.0.1", oauth_port
    )
    server.close()
    await server.wait_closed()
    token_client.stream.assert_not_called()
    assert auth.status() is None


async def test_sign_in_timeout_releases_listener(
    tmp_path: Path, oauth_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.chatgpt_auth.LOGIN_TIMEOUT", 0.01)
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    with pytest.raises(ChatGPTAuthError, match="timed out"):
        await auth.sign_in(lambda _url: None)
    server = await asyncio.start_server(
        lambda _reader, writer: writer.close(), "127.0.0.1", oauth_port
    )
    server.close()
    await server.wait_closed()


async def test_concurrent_refresh_is_serialized_and_rotation_persisted(
    tmp_path: Path, token_client: MagicMock
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, expired=True)
    auth = ChatGPTAuth(path)
    first, second = await asyncio.gather(auth.headers(), auth.headers())
    assert first == second
    assert first["Authorization"] == "Bearer fake-access-secret"
    token_client.stream.assert_called_once()
    saved = json.loads(path.read_text())
    assert saved["refresh_token"] == "fake-refresh-secret"
    assert "fake-access-secret" not in repr(auth.status())


async def test_refresh_error_never_exposes_body_or_replays(
    tmp_path: Path, token_client: MagicMock
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, expired=True)
    auth = ChatGPTAuth(path)
    token_client.stream.return_value.__aenter__.return_value = httpx.Response(
        401, text="secret-token-in-upstream-error"
    )
    with pytest.raises(ChatGPTAuthError) as error:
        await auth.headers()
    assert "secret-token" not in str(error.value)
    token_client.stream.assert_called_once()
    assert auth.status() is None
    assert not path.exists()


async def test_sign_out_cancels_active_session_and_prevents_token_resurrection(
    tmp_path: Path, token_client: MagicMock
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, expired=True)
    auth = ChatGPTAuth(path)
    dispatched = asyncio.Event()

    async def pending(*_args: Any, **_kwargs: Any) -> httpx.Response:
        dispatched.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json=token_response())

    token_client.stream.return_value.__aenter__.side_effect = pending

    async def request() -> None:
        async with auth.session():
            await auth.headers()

    task = asyncio.create_task(request())
    await dispatched.wait()
    auth.sign_out()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert auth.status() is None
    assert not path.exists()


async def test_queued_generation_cannot_use_changed_account(tmp_path: Path) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    token = account_epoch.set(auth.generation)
    try:
        auth.sign_out()
        with pytest.raises(JobCancelledError):
            async with auth.session():
                pytest.fail("A stale account reached the provider")
    finally:
        account_epoch.reset(token)


async def test_completed_job_context_stays_invalid_after_sign_out(
    tmp_path: Path,
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    active = Event()
    active.set()
    context = JobContext(object(), 1, active)
    token = current_job.set(context)
    try:
        async with auth.session():
            assert context.valid()
            assert account_epoch.get() == auth.generation
        assert context.valid()
        auth.sign_out()
        assert not context.valid()
        with pytest.raises(JobCancelledError):
            context.check()
    finally:
        current_job.reset(token)


async def test_sign_in_cannot_save_after_sign_out_during_exchange(
    tmp_path: Path, oauth_port: int, token_client: MagicMock
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    opened: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    dispatched = asyncio.Event()

    async def pending(*_args: Any, **_kwargs: Any) -> httpx.Response:
        dispatched.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json=token_response())

    token_client.stream.return_value.__aenter__.side_effect = pending
    task = asyncio.create_task(auth.sign_in(opened.set_result))
    url = await opened
    await callback_request(oauth_port, parse_qs(urlsplit(url).query)["state"][0])
    await dispatched.wait()
    auth.sign_out()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert auth.status() is None
    assert not (tmp_path / "chatgpt.json").exists()


async def test_many_field_sessions_do_not_chain_context_predicates(
    tmp_path: Path,
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    active = Event()
    active.set()
    context = JobContext(object(), 1, active)
    token = current_job.set(context)
    try:
        for _ in range(1200):
            async with auth.session():
                assert context.valid()
        auth.sign_out()
        assert not context.valid()
    finally:
        current_job.reset(token)


async def test_oversized_callback_headers_are_rejected_without_ending_login(
    tmp_path: Path, oauth_port: int, token_client: MagicMock
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    opened: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(auth.sign_in(opened.set_result))
    url = await opened
    reader, writer = await asyncio.open_connection("127.0.0.1", oauth_port)
    writer.write(b"GET / HTTP/1.1\r\nX-Large: " + b"x" * 18000 + b"\r\n\r\n")
    await writer.drain()
    assert await reader.read() == b""
    writer.close()
    await writer.wait_closed()
    assert not task.done()
    await callback_request(oauth_port, parse_qs(urlsplit(url).query)["state"][0])
    await task
    token_client.stream.assert_called_once()


async def test_expired_session_surfaces_auth_error_instead_of_canceling_caller(
    tmp_path: Path, token_client: MagicMock
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, expired=True)
    auth = ChatGPTAuth(path)
    token_client.stream.return_value.__aenter__.return_value = httpx.Response(
        401, json={"error": "invalid_grant"}
    )
    with pytest.raises(ChatGPTAuthError):
        async with auth.session():
            await auth.headers()
    await asyncio.sleep(0)
    assert auth.status() is None


async def test_token_response_body_is_bounded_before_parsing(
    tmp_path: Path, token_client: MagicMock
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, expired=True)
    auth = ChatGPTAuth(path)
    token_client.stream.return_value.__aenter__.return_value = httpx.Response(
        200, content=b"x" * 65537
    )
    with pytest.raises(ChatGPTAuthError, match="invalid sign-in response"):
        await auth.headers()
    token_client.stream.assert_called_once()
    token_client.stream.return_value.__aexit__.assert_awaited_once()
    assert auth.status() is None


@pytest.mark.parametrize(
    "claims,expected",
    [
        ({}, None),
        ({"chatgpt_compute_residency": "no_constraint"}, None),
        ({"chatgpt_compute_residency": "custom-region"}, "custom-region"),
        (
            {
                "chatgpt_compute_residency": "us",
                "https://api.openai.com/auth": {"chatgpt_compute_residency": "eu"},
            },
            "eu",
        ),
    ],
)
async def test_access_token_residency_is_forwarded_without_changing_region(
    tmp_path: Path, claims: dict[str, Any], expected: str | None
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path)
    value = json.loads(path.read_text())
    value["access_token"] = jwt(claims)
    path.write_text(json.dumps(value))
    headers = await ChatGPTAuth(path).headers()
    assert headers.get("x-openai-internal-codex-residency") == expected


def test_partial_identity_claims_do_not_hide_access_token_account() -> None:
    value = token_response()
    value["id_token"] = jwt(
        {
            "email": "test@example.test",
            "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"},
        }
    )
    value["access_token"] = jwt(
        {"https://api.openai.com/auth": {"chatgpt_account_id": "access-account"}}
    )
    assert parse_credentials(value).account.account_id == "access-account"
