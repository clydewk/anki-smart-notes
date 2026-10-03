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
import os
import secrets
import sys
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit
from weakref import WeakKeyDictionary

import httpx

from .job_context import JobCancelledError, current_job, deliver_future
from .provider_runtime import provider_runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from .job_context import JobContext


CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CALLBACK_PORT = 1455
LOGIN_TIMEOUT = 300
account_epoch: ContextVar[int | None] = ContextVar(
    "chatgpt_account_epoch", default=None
)


class ChatGPTAuthError(Exception):
    """A sign-in failure safe to display without credential redaction."""


@dataclass(frozen=True)
class ChatGPTAccount:
    email: str | None
    account_id: str
    plan: str | None


@dataclass(frozen=True)
class ChatGPTCredentials:
    account: ChatGPTAccount
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float


def credential_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return root / "AnkiSmartNotes/chatgpt.json"


def token_claims(token: object) -> dict[str, Any]:
    # Claims are display/routing metadata, never proof of authentication.
    if not isinstance(token, str):
        return {}
    try:
        encoded = token.split(".")[1]
        value = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        return value if isinstance(value, dict) else {}
    except (ValueError, IndexError, UnicodeError):
        return {}


def parse_credentials(
    value: dict[str, Any], previous: ChatGPTCredentials | None = None
) -> ChatGPTCredentials:
    access = value.get("access_token")
    refresh = value.get("refresh_token") or (
        previous.refresh_token if previous else None
    )
    access_claims = token_claims(access)
    identity_claims = token_claims(value.get("id_token"))
    claims = {**access_claims, **identity_claims}
    auth: dict[str, Any] = {}
    account_id: object = None
    for candidate in (access_claims, identity_claims):
        nested = candidate.get("https://api.openai.com/auth", {})
        nested = nested if isinstance(nested, dict) else {}
        auth.update(nested)
        organizations = candidate.get("organizations", [])
        organization = (
            organizations[0]
            if isinstance(organizations, list) and organizations
            else {}
        )
        organization = organization if isinstance(organization, dict) else {}
        account_id = (
            candidate.get("chatgpt_account_id")
            or nested.get("chatgpt_account_id")
            or organization.get("id")
            or account_id
        )
    account_id = account_id or (previous.account.account_id if previous else None)
    expires = value.get("expires_in", 3600)
    if not (
        isinstance(access, str)
        and access
        and isinstance(refresh, str)
        and refresh
        and isinstance(account_id, str)
        and account_id
        and isinstance(expires, (int, float))
        and not isinstance(expires, bool)
        and 0 < expires <= 31 * 24 * 3600
    ):
        raise ChatGPTAuthError(
            "ChatGPT returned an invalid sign-in response. Please sign in again."
        )
    email = claims.get("email")
    plan = auth.get("chatgpt_plan_type")
    account = ChatGPTAccount(
        email
        if isinstance(email, str)
        else previous.account.email
        if previous
        else None,
        account_id,
        plan if isinstance(plan, str) else previous.account.plan if previous else None,
    )
    return ChatGPTCredentials(account, access, refresh, time.time() + expires)


class ChatGPTAuth:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or credential_path()
        self._lock = RLock()
        self._generation = 0
        self._loaded = False
        self._credentials: ChatGPTCredentials | None = None
        self._sessions: dict[asyncio.Task[Any], int] = {}
        self._contexts: WeakKeyDictionary[JobContext, int] = WeakKeyDictionary()
        self._login_task: asyncio.Task[Any] | None = None
        self._refresh_lock: asyncio.Lock | None = None
        self._refresh_loop: asyncio.AbstractEventLoop | None = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def _read(self) -> ChatGPTCredentials | None:
        with self._lock:
            if not self._loaded:
                self._loaded = True
                try:
                    if self._path.stat().st_size > 65536:
                        return None
                    value = json.loads(self._path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict) or value.get("version") != 1:
                        return None
                    account = value.get("account", {})
                    if not isinstance(account, dict):
                        return None
                    if not all(
                        isinstance(item, str) and item
                        for item in (
                            value.get("access_token"),
                            value.get("refresh_token"),
                            account.get("account_id"),
                        )
                    ):
                        return None
                    if any(
                        item is not None and not isinstance(item, str)
                        for item in (account.get("email"), account.get("plan"))
                    ):
                        return None
                    credentials = ChatGPTCredentials(
                        ChatGPTAccount(
                            account.get("email"),
                            account["account_id"],
                            account.get("plan"),
                        ),
                        value["access_token"],
                        value["refresh_token"],
                        float(value["expires_at"]),
                    )
                    self._credentials = credentials
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            return self._credentials

    def status(self) -> ChatGPTAccount | None:
        credentials = self._read()
        return credentials.account if credentials else None

    def _check(self, generation: int) -> None:
        if generation != self.generation:
            raise JobCancelledError(
                "The ChatGPT account changed. Run generation again."
            )

    def _save(self, credentials: ChatGPTCredentials, generation: int) -> None:
        with self._lock:
            self._check(generation)
            temporary: str | None = None
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._path.parent.chmod(0o700)
                descriptor, temporary = tempfile.mkstemp(
                    prefix=".chatgpt-", dir=self._path.parent
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(
                        {
                            "version": 1,
                            "account": {
                                "email": credentials.account.email,
                                "account_id": credentials.account.account_id,
                                "plan": credentials.account.plan,
                            },
                            "access_token": credentials.access_token,
                            "refresh_token": credentials.refresh_token,
                            "expires_at": credentials.expires_at,
                        },
                        output,
                    )
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self._path)
                self._credentials = credentials
                self._loaded = True
            except OSError:
                raise ChatGPTAuthError(
                    "Could not save the ChatGPT connection. Check application-data folder permissions."
                ) from None
            finally:
                if temporary is not None:
                    with suppress(OSError):
                        Path(temporary).unlink(missing_ok=True)

    def cancel_sign_in(self) -> None:
        with self._lock:
            task = self._login_task
        if task is not None and not task.done():
            self._invalidate()
            deliver_future(task.get_loop(), task.cancel)

    def _invalidate(self) -> int:
        with self._lock:
            self._generation += 1
            tasks = tuple(self._sessions)
            generation = self._generation
        try:
            caller = asyncio.current_task()
        except RuntimeError:
            caller = None
        for task in tasks:
            if task is not caller:
                deliver_future(task.get_loop(), task.cancel)
        return generation

    def sign_out(self) -> None:
        self._invalidate()
        self.cancel_sign_in()
        with self._lock:
            self._credentials = None
            self._loaded = True
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                raise ChatGPTAuthError(
                    "ChatGPT is disconnected, but its saved credentials could not be removed. Check application-data folder permissions."
                ) from None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None]:
        generation = account_epoch.get()
        generation = self.generation if generation is None else generation
        task = asyncio.current_task()
        assert task is not None
        with self._lock:
            self._check(generation)
            context = current_job.get()
            if context is not None:
                if context in self._contexts:
                    self._check(self._contexts[context])
                else:
                    previous = context.connection_valid
                    context.connection_valid = lambda: (
                        generation == self.generation
                        and (previous is None or previous())
                    )
                    self._contexts[context] = generation
            self._sessions[task] = self._sessions.get(task, 0) + 1
        token = account_epoch.set(generation)
        try:
            yield
            self._check(generation)
        finally:
            account_epoch.reset(token)
            with self._lock:
                count = self._sessions.get(task, 1) - 1
                if count:
                    self._sessions[task] = count
                else:
                    self._sessions.pop(task, None)

    async def _exchange(self, data: dict[str, str]) -> dict[str, Any]:
        try:
            async with asyncio.timeout(60):
                client = await provider_runtime.get_http_client()
                async with client.stream(
                    "POST",
                    TOKEN_URL,
                    data=data,
                    headers={"Accept": "application/json"},
                    timeout=60,
                    follow_redirects=False,
                ) as response:
                    if response.status_code != 200:
                        raise ChatGPTAuthError(
                            "ChatGPT sign-in expired or was rejected. Please sign in again."
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        if len(body) + len(chunk) > 65536:
                            raise ChatGPTAuthError(
                                "ChatGPT returned an invalid sign-in response. Please sign in again."
                            )
                        body.extend(chunk)
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (httpx.HTTPError, TimeoutError, ValueError):
            raise ChatGPTAuthError(
                "Could not complete ChatGPT authentication. Please sign in again."
            ) from None

    async def headers(self, expected_generation: int | None = None) -> dict[str, str]:
        generation = (
            expected_generation
            if expected_generation is not None
            else account_epoch.get()
        )
        generation = self.generation if generation is None else generation
        self._check(generation)
        loop = asyncio.get_running_loop()
        if self._refresh_loop is not loop:
            self._refresh_loop, self._refresh_lock = loop, asyncio.Lock()
        assert self._refresh_lock is not None
        async with self._refresh_lock:
            self._check(generation)
            credentials = self._read()
            if credentials is None:
                raise ChatGPTAuthError(
                    "Sign in with ChatGPT in Providers before generating text."
                )
            if credentials.expires_at <= time.time() + 60:
                try:
                    result = await self._exchange(
                        {
                            "grant_type": "refresh_token",
                            "client_id": CLIENT_ID,
                            "refresh_token": credentials.refresh_token,
                        }
                    )
                    refreshed = parse_credentials(result, credentials)
                    if refreshed.account.account_id != credentials.account.account_id:
                        raise ChatGPTAuthError(
                            "The ChatGPT account changed. Please sign in again."
                        )
                    self._save(refreshed, generation)
                    credentials = refreshed
                except (ChatGPTAuthError, asyncio.CancelledError):
                    # Rotation may already have happened. Never replay an ambiguous refresh.
                    with self._lock:
                        if generation == self._generation:
                            with suppress(ChatGPTAuthError):
                                self.sign_out()
                    raise
            self._check(generation)
            headers = {
                "Authorization": f"Bearer {credentials.access_token}",
                "ChatGPT-Account-Id": credentials.account.account_id,
                "User-Agent": "AnkiSmartNotes",
                "originator": "anki_smart_notes",
            }
            claims = token_claims(credentials.access_token)
            auth = claims.get("https://api.openai.com/auth", {})
            auth = auth if isinstance(auth, dict) else {}
            residency = auth.get(
                "chatgpt_compute_residency", claims.get("chatgpt_compute_residency")
            )
            if residency and residency != "no_constraint":
                if not isinstance(residency, str) or any(
                    not 32 < ord(char) < 127 for char in residency
                ):
                    raise ChatGPTAuthError(
                        "ChatGPT returned invalid account routing information. Please sign in again."
                    )
                headers["x-openai-internal-codex-residency"] = residency
            return headers

    async def sign_in(self, open_browser: Callable[[str], None]) -> ChatGPTAccount:
        task = asyncio.current_task()
        assert task is not None
        with self._lock:
            if self._login_task is not None:
                raise ChatGPTAuthError("ChatGPT sign-in is already in progress.")
            self._login_task = task
        generation = self._invalidate()
        verifier = secrets.token_urlsafe(32)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        state = secrets.token_urlsafe(32)
        redirect = f"http://localhost:{CALLBACK_PORT}/auth/callback"
        callback: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        connections: set[asyncio.Task[Any]] = set()

        async def receive(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            current = asyncio.current_task()
            assert current is not None
            connections.add(current)
            status, message = "400 Bad Request", "Invalid sign-in callback."
            try:
                async with asyncio.timeout(10):
                    request = await reader.readuntil(b"\r\n\r\n")
                method, target, _ = (
                    request.split(b"\r\n", 1)[0].decode("ascii").split(" ")
                )
                url = urlsplit(target)
                query = parse_qs(url.query, max_num_fields=12)
                states = query.get("state", [])
                if (
                    method == "GET"
                    and url.path == "/auth/callback"
                    and len(states) == 1
                    and secrets.compare_digest(states[0], state)
                    and not callback.done()
                ):
                    codes = query.get("code", [])
                    if "error" in query:
                        callback.set_exception(
                            ChatGPTAuthError(
                                "ChatGPT sign-in was declined. Please try again."
                            )
                        )
                    elif len(codes) == 1 and codes[0]:
                        callback.set_result(codes[0])
                        status, message = (
                            "200 OK",
                            "Sign-in received. Return to Anki Smart Notes.",
                        )
                writer.write(
                    (
                        f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                        f"Content-Length: {len(message.encode())}\r\nConnection: close\r\n"
                        f"Cache-Control: no-store\r\n\r\n{message}"
                    ).encode()
                )
                await writer.drain()
            except (
                ValueError,
                UnicodeError,
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                TimeoutError,
                OSError,
            ):
                pass
            finally:
                writer.close()
                with suppress(OSError):
                    await writer.wait_closed()
                connections.discard(current)

        try:
            try:
                server = await asyncio.start_server(
                    receive, "127.0.0.1", CALLBACK_PORT, limit=16384
                )
            except OSError:
                raise ChatGPTAuthError(
                    "Could not start ChatGPT sign-in on localhost:1455. Close other sign-in windows and try again."
                ) from None
            async with server:
                try:
                    open_browser(
                        AUTHORIZE_URL
                        + "?"
                        + urlencode(
                            {
                                "response_type": "code",
                                "client_id": CLIENT_ID,
                                "redirect_uri": redirect,
                                "scope": "openid profile email offline_access",
                                "code_challenge": challenge,
                                "code_challenge_method": "S256",
                                "state": state,
                                "id_token_add_organizations": "true",
                                "codex_cli_simplified_flow": "true",
                                "originator": "anki_smart_notes",
                            }
                        )
                    )
                except Exception:
                    raise ChatGPTAuthError(
                        "Could not open your browser for ChatGPT sign-in."
                    ) from None
                try:
                    code = await asyncio.wait_for(callback, LOGIN_TIMEOUT)
                except TimeoutError:
                    raise ChatGPTAuthError(
                        "ChatGPT sign-in timed out. Please try again."
                    ) from None
            self._check(generation)
            result = await self._exchange(
                {
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect,
                }
            )
            credentials = parse_credentials(result)
            self._save(credentials, generation)
            return credentials.account
        finally:
            for connection in tuple(connections):
                connection.cancel()
            if connections:
                await asyncio.gather(*connections, return_exceptions=True)
            with self._lock:
                if self._login_task is task:
                    self._login_task = None


chatgpt_auth = ChatGPTAuth()
