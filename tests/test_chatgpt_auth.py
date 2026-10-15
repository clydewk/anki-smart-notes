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
import stat
import time
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
import rsa
from jose import jwt

from src.chatgpt_auth import (
    DYNAMIC_CLIENT_ID,
    ISSUER,
    PLAN_SCOPE,
    RESOURCE,
    SCOPES,
    ChatGPTAuth,
    ChatGPTAuthError,
    account_epoch,
)
from src.job_context import JobCancelledError, JobContext, current_job

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture(scope="module")
def signing_key() -> rsa.PrivateKey:
    return rsa.newkeys(2048)[1]


def encoded_integer(value: int) -> str:
    return (
        base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big"))
        .rstrip(b"=")
        .decode()
    )


@dataclass
class OAuthPeer:
    key: rsa.PrivateKey
    nonce: str = "nonce"
    client_id: str = "oaiapp_test"
    claims: dict[str, Any] = field(default_factory=dict)
    token_fields: dict[str, Any] = field(default_factory=dict)
    status: int = 200
    error: dict[str, Any] | None = None
    calls: list[httpx.Request] = field(default_factory=list)
    exchange_started: asyncio.Event = field(default_factory=asyncio.Event)
    exchange_resume: asyncio.Event | None = None
    revoke_status: int = 200
    kid: str = "key-1"
    omit_fields: set[str] = field(default_factory=set)
    body: bytes | None = None
    transport_error: httpx.HTTPError | None = None

    def tokens(self) -> dict[str, Any]:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": self.client_id,
            "sub": "subject-1",
            "email": "test@example.test",
            "iat": now,
            "exp": now + 3600,
            "nonce": self.nonce,
            **self.claims,
        }
        identity = jwt.encode(
            claims,
            self.key.save_pkcs1().decode(),
            algorithm="RS256",
            headers={"kid": self.kid},
        )
        tokens = {
            "access_token": "fake-access-secret",
            "refresh_token": "fake-refresh-secret",
            "id_token": identity,
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": SCOPES,
            **self.token_fields,
        }
        return {
            key: value for key, value in tokens.items() if key not in self.omit_fields
        }

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": ISSUER + "/api/accounts/authorize",
                    "token_endpoint": ISSUER + "/api/accounts/oauth/token",
                    "revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke",
                    "jwks_uri": ISSUER + "/.well-known/jwks.json",
                },
            )
        if request.url.path == "/.well-known/jwks.json":
            return httpx.Response(
                200,
                json={
                    "keys": [
                        {
                            "kid": self.kid,
                            "kty": "RSA",
                            "alg": "RS256",
                            "use": "sig",
                            "n": encoded_integer(self.key.n),
                            "e": encoded_integer(self.key.e),
                        }
                    ]
                },
            )
        if request.url.path == "/api/accounts/oauth/revoke":
            return httpx.Response(
                self.revoke_status,
                json={} if self.revoke_status == 200 else {"error": "server_error"},
            )
        assert request.url.path == "/api/accounts/oauth/token"
        self.exchange_started.set()
        if self.exchange_resume is not None:
            await self.exchange_resume.wait()
        if self.transport_error is not None:
            raise self.transport_error
        if self.body is not None:
            return httpx.Response(self.status, content=self.body)
        return httpx.Response(
            self.status, json=self.error if self.error is not None else self.tokens()
        )

    def exchanges(self) -> list[dict[str, list[str]]]:
        return [
            parse_qs(request.content.decode())
            for request in self.calls
            if request.url.path.endswith("/oauth/token")
        ]


@pytest.fixture
async def peer(
    signing_key: rsa.PrivateKey, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[OAuthPeer]:
    server = OAuthPeer(signing_key)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handle)
    ) as client:
        monkeypatch.setattr(
            "src.chatgpt_auth.provider_runtime.get_http_client",
            AsyncMock(return_value=client),
        )
        yield server


def write_credentials(path: Path, peer: OAuthPeer, *, expired: bool = False) -> None:
    tokens = peer.tokens()
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "ext_agent_host_id": uuid4().urn,
                "selected": peer.client_id,
                "registrations": [
                    {
                        "client_id": peer.client_id,
                        "subject": "subject-1",
                        "email": "test@example.test",
                        "credentials": {
                            "access_token": "old-access-secret",
                            "refresh_token": "old-refresh-secret",
                            "id_token": tokens["id_token"],
                            "expires_at": time.time() + (-10 if expired else 3600),
                            "scopes": SCOPES.split(),
                        },
                    }
                ],
            }
        )
    )


async def callback_request(
    url: str,
    *,
    state: str | None = None,
    client_id: str | None = "oaiapp_test",
    error: str | None = None,
) -> bytes:
    query = parse_qs(urlsplit(url).query)
    redirect = urlsplit(query["redirect_uri"][0])
    reader, writer = await asyncio.open_connection("127.0.0.1", redirect.port)
    parameters = {
        "state": state if state is not None else query["state"][0],
        "code": "test-code",
    }
    if client_id is not None:
        parameters["client_id"] = client_id
    if error is not None:
        parameters["error"] = error
    writer.write(
        f"GET {redirect.path}?{urlencode(parameters)} HTTP/1.1\r\nHost: {redirect.netloc}\r\n\r\n".encode()
    )
    await writer.drain()
    result = await reader.read()
    writer.close()
    await writer.wait_closed()
    return result


async def begin_sign_in(
    auth: ChatGPTAuth, peer: OAuthPeer, *, new_account: bool = False
) -> tuple[asyncio.Task[Any], str]:
    opened: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(auth.sign_in(opened.set_result, new_account=new_account))
    url = await asyncio.wait_for(opened, 3)
    peer.nonce = parse_qs(urlsplit(url).query)["nonce"][0]
    return task, url


async def complete_sign_in(
    auth: ChatGPTAuth, peer: OAuthPeer, *, new_account: bool = False
) -> str:
    task, url = await begin_sign_in(auth, peer, new_account=new_account)
    await callback_request(url, client_id=peer.client_id)
    await task
    return url


async def test_new_registration_pkce_nonce_discovery_and_private_persistence(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "private/chatgpt.json"
    auth = ChatGPTAuth(path)
    task, url = await begin_sign_in(auth, peer)
    query = parse_qs(urlsplit(url).query)
    assert url.startswith(ISSUER + "/api/accounts/authorize?")
    assert query["client_id"] == [DYNAMIC_CLIENT_ID]
    assert query["agent_name_hint"] == ["Anki Smart Notes"]
    assert query["resource"] == [RESOURCE]
    assert query["scope"] == [SCOPES]
    assert UUID(query["ext_agent_host_id"][0]).urn == query["ext_agent_host_id"][0]
    assert query["redirect_uri"][0].startswith("http://127.0.0.1:")
    assert "client_secret" not in query
    assert b"400 Bad Request" in await callback_request(url, state="wrong-state")
    assert not task.done()
    assert b"200 OK" in await callback_request(url)
    account = await task
    assert account.email == "test@example.test"
    assert account.account_id == peer.client_id
    assert account.connected and account.sharing
    data = peer.exchanges()[0]
    assert data["client_id"] == [peer.client_id]
    assert data["redirect_uri"] == query["redirect_uri"]
    assert data["resource"] == [RESOURCE]
    expected = (
        base64.urlsafe_b64encode(
            hashlib.sha256(data["code_verifier"][0].encode()).digest()
        )
        .rstrip(b"=")
        .decode()
    )
    assert query["code_challenge"] == [expected]
    assert ChatGPTAuth(path).status() == account
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert "fake-access-secret" not in repr(account)
    assert await auth.headers() == {
        "Authorization": "Bearer fake-access-secret",
        "User-Agent": "AnkiSmartNotes",
    }
    assert auth.welcome_needed
    auth.mark_welcome_seen()
    assert not ChatGPTAuth(path).welcome_needed


@pytest.mark.parametrize(
    "change",
    [
        {"iss": "https://attacker.example"},
        {"aud": "another-client"},
        {"exp": 1},
        {"nonce": "wrong"},
        {"sub": ""},
        {"iat": 99999999999},
        {"iat": True},
        {"aud": ["oaiapp_test", "other"]},
        {"azp": "other"},
    ],
)
async def test_id_token_claims_are_verified(
    tmp_path: Path, peer: OAuthPeer, change: dict[str, Any]
) -> None:
    peer.claims = change
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url)
    with pytest.raises(ChatGPTAuthError, match="identity could not be verified"):
        await task
    account = auth.status()
    assert account is not None and not account.connected


async def test_id_token_signature_is_verified(tmp_path: Path, peer: OAuthPeer) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    token = peer.tokens()["id_token"]
    header, claims, signature = token.split(".")
    signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    peer.token_fields["id_token"] = f"{header}.{claims}.{signature}"
    await callback_request(url)
    with pytest.raises(ChatGPTAuthError, match="identity could not be verified"):
        await task


async def test_unfamiliar_key_refreshes_jwks(tmp_path: Path, peer: OAuthPeer) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    await complete_sign_in(auth, peer)
    peer.kid = "rotated-key"
    await complete_sign_in(auth, peer)
    assert sum(request.url.path.endswith("jwks.json") for request in peer.calls) == 2


async def test_returning_registration_hints_and_identity_match(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    original = parse_qs(urlsplit(await complete_sign_in(auth, peer)).query)
    task, url = await begin_sign_in(auth, peer)
    query = parse_qs(urlsplit(url).query)
    assert query["client_id"] == [peer.client_id]
    assert query["ext_agent_host_id"] == original["ext_agent_host_id"]
    assert query["login_hint"] == ["test@example.test"]
    assert query["id_token_hint"]
    assert "agent_name_hint" not in query and "prompt" not in query
    peer.claims["sub"] = "different-subject"
    await callback_request(url, client_id=None)
    with pytest.raises(ChatGPTAuthError, match="does not match"):
        await task
    assert await auth.headers() == {
        "Authorization": "Bearer fake-access-secret",
        "User-Agent": "AnkiSmartNotes",
    }


@pytest.mark.parametrize("client_id", [None, DYNAMIC_CLIENT_ID])
async def test_new_callback_requires_issued_client(
    tmp_path: Path, peer: OAuthPeer, client_id: str | None
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url, client_id=client_id)
    with pytest.raises(ChatGPTAuthError, match="registration"):
        await task
    assert not peer.exchanges()


async def test_reauthorization_rejects_changed_callback_client(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    await complete_sign_in(auth, peer)
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url, client_id="another-client")
    with pytest.raises(ChatGPTAuthError, match="registration"):
        await task
    assert len(peer.exchanges()) == 1


async def test_failed_exchange_retains_new_registration_for_retry(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    auth = ChatGPTAuth(path)
    task, url = await begin_sign_in(auth, peer)
    peer.status, peer.error = 400, {"error": "invalid_grant"}
    await callback_request(url)
    with pytest.raises(ChatGPTAuthError):
        await task
    assert auth.status() is not None
    peer.status, peer.error = 200, None
    restarted = ChatGPTAuth(path)
    retried = await complete_sign_in(restarted, peer)
    assert parse_qs(urlsplit(retried).query)["client_id"] == [peer.client_id]


@pytest.mark.parametrize("identity_only", [False, True])
async def test_identity_without_plan_permission_is_retained_and_cannot_infer(
    tmp_path: Path, peer: OAuthPeer, identity_only: bool
) -> None:
    peer.token_fields = {"scope": "openid profile email"}
    if identity_only:
        peer.omit_fields = {"access_token", "refresh_token", "expires_in", "token_type"}
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    await complete_sign_in(auth, peer)
    account = auth.status()
    assert account and account.connected and not account.sharing
    with pytest.raises(ChatGPTAuthError, match="plan usage is disabled"):
        await auth.headers()
    task, url = await begin_sign_in(auth, peer)
    query = parse_qs(urlsplit(url).query)
    assert query["prompt"] == ["consent"]
    assert PLAN_SCOPE in query["scope"][0]
    peer.token_fields = {}
    peer.omit_fields.clear()
    await callback_request(url)
    await task
    assert auth.welcome_needed


async def test_multiple_accounts_same_email_remain_independent_and_logout_retains_registration(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    auth = ChatGPTAuth(path)
    await complete_sign_in(auth, peer)
    first = peer.client_id
    peer.client_id = "oaiapp_second"
    await complete_sign_in(auth, peer, new_account=True)
    assert len(auth.list_accounts()) == 2
    assert all(account.email == "test@example.test" for account in auth.list_accounts())
    auth.select_account(first)
    selected = auth.status()
    assert selected is not None and selected.account_id == first
    generation = auth.generation
    revoke = auth.sign_out()
    assert auth.generation > generation
    selected = auth.status()
    assert selected is not None and not selected.connected
    restored = ChatGPTAuth(path).status()
    assert restored is not None and not restored.connected
    await revoke
    request = next(
        request for request in peer.calls if request.url.path.endswith("revoke")
    )
    assert parse_qs(request.content.decode()) == {
        "token": ["fake-refresh-secret"],
        "token_type_hint": ["refresh_token"],
        "client_id": [first],
    }
    auth.select_account(peer.client_id)
    selected = auth.status()
    assert selected is not None and selected.connected


async def test_revocation_failure_clears_locally_and_warns(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    auth = ChatGPTAuth(path)
    await complete_sign_in(auth, peer)
    peer.revoke_status = 503
    with pytest.raises(ChatGPTAuthError, match="revocation was not confirmed"):
        await auth.sign_out()
    assert "fake-access-secret" not in path.read_text()
    assert (
        len([request for request in peer.calls if request.url.path.endswith("revoke")])
        == 2
    )


async def test_legacy_credentials_are_never_sent_or_migrated(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "access_token": "legacy-secret",
                "refresh_token": "legacy-refresh",
            }
        )
    )
    auth = ChatGPTAuth(path)
    assert auth.needs_reauthorization and auth.status() is None
    with pytest.raises(ChatGPTAuthError, match="Sign in"):
        await auth.headers()
    await complete_sign_in(auth, peer)
    assert not auth.needs_reauthorization
    assert "legacy" not in path.read_text()
    assert all(b"legacy" not in request.content for request in peer.calls)


async def test_concurrent_refreshes_across_instances_use_rotated_token_once(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    first, second = ChatGPTAuth(path), ChatGPTAuth(path)
    assert first.status() == second.status()
    headers = await asyncio.gather(first.headers(), second.headers())
    assert headers[0] == headers[1]
    assert headers[0]["Authorization"] == "Bearer fake-access-secret"
    assert len(peer.exchanges()) == 1
    assert peer.exchanges()[0] == {
        "grant_type": ["refresh_token"],
        "client_id": [peer.client_id],
        "refresh_token": ["old-refresh-secret"],
        "resource": [RESOURCE],
    }
    assert (
        json.loads(path.read_text())["registrations"][0]["credentials"]["refresh_token"]
        == "fake-refresh-secret"
    )


@pytest.mark.parametrize(
    "code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
async def test_terminal_refresh_clears_tokens_only(
    tmp_path: Path, peer: OAuthPeer, code: str
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    auth = ChatGPTAuth(path)
    peer.status, peer.error = 400, {"error": code, "error_description": "fake-secret"}
    with pytest.raises(ChatGPTAuthError) as error:
        async with auth.session():
            await auth.headers()
    assert "fake-secret" not in str(error.value)
    selected = auth.status()
    assert selected is not None and not selected.connected
    assert len(auth.list_accounts()) == 1 and len(peer.exchanges()) == 1
    assert "old-refresh-secret" not in path.read_text()


@pytest.mark.parametrize(
    "code,status", [("invalid_client", 400), ("server_error", 503), ("unknown", 401)]
)
async def test_nonterminal_refresh_preserves_tokens(
    tmp_path: Path, peer: OAuthPeer, code: str, status: int
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    auth = ChatGPTAuth(path)
    original = path.read_text()
    peer.status, peer.error = status, {"error": code, "error_description": "secret"}
    with pytest.raises(ChatGPTAuthError):
        await auth.headers()
    assert path.read_text() == original
    assert len(peer.exchanges()) == 1


async def test_cancel_refresh_preserves_tokens_and_releases_lock(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    original = path.read_text()
    auth = ChatGPTAuth(path)
    peer.exchange_resume = asyncio.Event()
    task = asyncio.create_task(auth.headers())
    await peer.exchange_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.read_text() == original
    peer.exchange_resume.set()
    assert (await auth.headers())["Authorization"] == "Bearer fake-access-secret"


async def test_sign_out_cancels_refresh_and_prevents_resurrection(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    auth = ChatGPTAuth(path)
    peer.exchange_resume = asyncio.Event()

    async def request() -> None:
        async with auth.session():
            await auth.headers()

    task = asyncio.create_task(request())
    await peer.exchange_started.wait()
    await auth.sign_out()
    with pytest.raises(asyncio.CancelledError):
        await task
    selected = auth.status()
    assert selected is not None and not selected.connected
    assert "old-access-secret" not in path.read_text()


async def test_queued_and_completed_job_contexts_stay_invalid_after_sign_out(
    tmp_path: Path,
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    active = Event()
    active.set()
    context = JobContext(object(), 1, active)
    token = current_job.set(context)
    epoch = account_epoch.set(auth.generation)
    try:
        for _ in range(1200):
            async with auth.session():
                assert context.valid()
        await auth.sign_out()
        assert not context.valid()
        with pytest.raises(JobCancelledError):
            async with auth.session():
                pytest.fail("A stale account reached the provider")
    finally:
        account_epoch.reset(epoch)
        current_job.reset(token)


async def test_cancel_and_timeout_release_ephemeral_callback(
    tmp_path: Path, peer: OAuthPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    port = urlsplit(parse_qs(urlsplit(url).query)["redirect_uri"][0]).port
    auth.cancel_sign_in()
    with pytest.raises(asyncio.CancelledError):
        await task
    server = await asyncio.start_server(
        lambda _reader, writer: writer.close(), "127.0.0.1", port
    )
    server.close()
    await server.wait_closed()
    monkeypatch.setattr("src.chatgpt_auth.LOGIN_TIMEOUT", 0.01)
    with pytest.raises(ChatGPTAuthError, match="timed out"):
        await auth.sign_in(lambda _url: None)
    assert not peer.exchanges()


async def test_oversized_callback_does_not_consume_login(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    port = urlsplit(parse_qs(urlsplit(url).query)["redirect_uri"][0]).port
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET / HTTP/1.1\r\nX-Large: " + b"x" * 18000 + b"\r\n\r\n")
    await writer.drain()
    assert await reader.read() == b""
    writer.close()
    await writer.wait_closed()
    assert not task.done()
    await callback_request(url)
    await task


async def test_declined_callback_does_not_exchange(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url, error="access_denied")
    with pytest.raises(ChatGPTAuthError, match="declined"):
        await task
    assert not peer.exchanges()


async def test_corrupt_store_errors_persist_and_failed_session_does_not_leak(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    path.write_text("invalid json")
    auth = ChatGPTAuth(path)
    for _ in range(2):
        with pytest.raises(ChatGPTAuthError, match="Could not read"):
            auth.status()
    ready, resume = asyncio.Event(), asyncio.Event()

    async def failed_session() -> None:
        with pytest.raises(ChatGPTAuthError):
            async with auth.session():
                pytest.fail("Corrupt credentials admitted a session")
        ready.set()
        await resume.wait()

    task = asyncio.create_task(failed_session())
    await ready.wait()
    write_credentials(path, peer)
    await auth.sign_out()
    await asyncio.sleep(0)
    assert not task.done()
    resume.set()
    await task


async def test_atomic_write_failure_preserves_connection(
    tmp_path: Path, peer: OAuthPeer, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer)
    auth = ChatGPTAuth(path)
    previous = auth.status()
    contents = path.read_text()

    def fail_replace(*_args: Any) -> None:
        raise OSError("not writable")

    monkeypatch.setattr("src.chatgpt_auth.os.replace", fail_replace)
    with pytest.raises(ChatGPTAuthError, match="Could not save"):
        auth.select_account(peer.client_id)
    assert auth.status() == previous
    assert path.read_text() == contents
    assert not list(tmp_path.glob(".chatgpt-*"))


async def test_sign_out_during_code_exchange_cannot_resurrect_tokens(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    auth = ChatGPTAuth(path)
    peer.exchange_resume = asyncio.Event()
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url)
    await peer.exchange_started.wait()
    await auth.sign_out()
    with pytest.raises(asyncio.CancelledError):
        await task
    account = auth.status()
    assert account is not None and not account.connected
    assert "fake-access-secret" not in path.read_text()


@pytest.mark.parametrize("body", [b"not json", b"[]", b"x" * 262145])
async def test_malformed_or_oversized_token_response_preserves_saved_credentials(
    tmp_path: Path, peer: OAuthPeer, body: bytes
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    original = path.read_text()
    peer.body = body
    with pytest.raises(ChatGPTAuthError):
        await ChatGPTAuth(path).headers()
    assert path.read_text() == original
    assert len(peer.exchanges()) == 1


async def test_refresh_transport_failure_is_safe_and_not_replayed(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    write_credentials(path, peer, expired=True)
    original = path.read_text()
    peer.transport_error = httpx.ReadError("secret-access-token in diagnostic")
    with pytest.raises(ChatGPTAuthError) as error:
        await ChatGPTAuth(path).headers()
    assert "secret-access-token" not in str(error.value)
    assert path.read_text() == original
    assert len(peer.exchanges()) == 1


@pytest.mark.parametrize(
    "missing",
    ["access_token", "refresh_token", "expires_in", "token_type", "scope", "id_token"],
)
async def test_direct_grant_requires_complete_credentials(
    tmp_path: Path, peer: OAuthPeer, missing: str
) -> None:
    peer.omit_fields = {missing}
    auth = ChatGPTAuth(tmp_path / "chatgpt.json")
    task, url = await begin_sign_in(auth, peer)
    await callback_request(url)
    with pytest.raises(ChatGPTAuthError):
        await task
    account = auth.status()
    assert account is not None and not account.connected


async def test_cross_process_account_switch_never_changes_captured_request(
    tmp_path: Path, peer: OAuthPeer
) -> None:
    path = tmp_path / "chatgpt.json"
    auth = ChatGPTAuth(path)
    await complete_sign_in(auth, peer)
    first = peer.client_id
    peer.client_id = "oaiapp_other"
    await complete_sign_in(auth, peer, new_account=True)
    auth.select_account(first)
    other_process = ChatGPTAuth(path)
    with pytest.raises(JobCancelledError):
        async with auth.session():
            other_process.select_account(peer.client_id)
            await auth.headers()
