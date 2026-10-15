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
import math
import os
import secrets
import sys
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary

import httpx
from jose import JWTError, jwt

from .job_context import JobCancelledError, current_job, deliver_future
from .logger import register_secrets
from .provider_runtime import provider_runtime

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator

    from .job_context import JobContext


ISSUER = "https://auth.openai.com"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT_ID = "dynamic_agent_client"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
LOGIN_TIMEOUT = 300
MAX_AUTH_BYTES = 262144
TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
account_epoch: ContextVar[int | None] = ContextVar(
    "chatgpt_account_epoch", default=None
)
account_registration: ContextVar[str | None] = ContextVar(
    "chatgpt_registration", default=None
)


class ChatGPTAuthError(Exception):
    """Authentication failure containing no untrusted response text or credentials."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ChatGPTAccount:
    email: str | None
    account_id: str
    plan: str | None = None
    connected: bool = True
    sharing: bool = True


@dataclass(frozen=True)
class ChatGPTCredentials:
    access_token: str | None = field(repr=False)
    refresh_token: str | None = field(repr=False)
    id_token: str = field(repr=False)
    expires_at: float
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class ChatGPTRegistration:
    client_id: str
    subject: str | None = None
    email: str | None = None
    credentials: ChatGPTCredentials | None = field(default=None, repr=False)

    @property
    def account(self) -> ChatGPTAccount:
        return ChatGPTAccount(
            self.email,
            self.client_id,
            connected=self.credentials is not None,
            sharing=self.credentials is not None
            and PLAN_SCOPE in self.credentials.scopes,
        )


def credential_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming")))
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return root / "AnkiSmartNotes/chatgpt.json"


def nonempty_string(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value) and not any(ord(c) < 32 for c in value)
    )


def parse_registration(value: dict[str, Any]) -> ChatGPTRegistration:
    if (
        not nonempty_string(value.get("client_id"))
        or value["client_id"] == DYNAMIC_CLIENT_ID
    ):
        raise ValueError
    if any(
        value.get(key) is not None and not nonempty_string(value[key])
        for key in ("email", "subject")
    ):
        raise ValueError
    credentials = None
    saved = value.get("credentials")
    if saved is not None:
        if not isinstance(saved, dict) or not value.get("subject"):
            raise ValueError
        if not nonempty_string(saved.get("id_token")) or any(
            saved.get(key) is not None and not nonempty_string(saved[key])
            for key in ("access_token", "refresh_token")
        ):
            raise ValueError
        expiry = saved.get("expires_at")
        scopes = saved.get("scopes")
        if (
            not isinstance(expiry, (float, int))
            or isinstance(expiry, bool)
            or not math.isfinite(expiry)
        ):
            raise ValueError
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) for scope in scopes
        ):
            raise ValueError
        credentials = ChatGPTCredentials(
            saved["access_token"],
            saved["refresh_token"],
            saved["id_token"],
            expiry,
            tuple(scopes),
        )
    return ChatGPTRegistration(
        value["client_id"], value.get("subject"), value.get("email"), credentials
    )


class ChatGPTAuth:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or credential_path()
        self._lock = RLock()
        self._generation = 0
        self._loaded = False
        self._host_id = uuid4().urn
        self._registrations: dict[str, ChatGPTRegistration] = {}
        self._selected: str | None = None
        self._migration_required = False
        self._welcomed = False
        self._sessions: dict[asyncio.Task[Any], int] = {}
        self._contexts: WeakKeyDictionary[JobContext, int] = WeakKeyDictionary()
        self._login_task: asyncio.Task[Any] | None = None
        self._discovery: dict[str, Any] | None = None
        self._jwks: dict[str, Any] | None = None
        self._jwks_at = 0.0
        self._storage_descriptor: int | None = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def needs_reauthorization(self) -> bool:
        self._read()
        return self._migration_required

    @property
    def welcome_needed(self) -> bool:
        account = self.status()
        return bool(account and account.sharing and not self._welcomed)

    def _read(self, *, reload: bool = False) -> None:
        with self._lock:
            if self._loaded and not reload:
                return
            self._loaded = False
            try:
                if self._path.stat().st_size > MAX_AUTH_BYTES:
                    raise ValueError
                value = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError
                if value.get("version") == 1:
                    self._registrations, self._selected = {}, None
                    self._migration_required = True
                    self._loaded = True
                    return
                if value.get("version") != 2:
                    raise ValueError
                host = value["ext_agent_host_id"]
                if not isinstance(host, str) or UUID(host).urn != host:
                    raise ValueError
                records = value["registrations"]
                if not isinstance(records, list):
                    raise ValueError
                registrations = {
                    item.client_id: item
                    for item in (parse_registration(record) for record in records)
                }
                if len(records) != len(registrations):
                    raise ValueError
                selected = value.get("selected")
                if selected is not None and selected not in registrations:
                    raise ValueError
                self._host_id, self._registrations, self._selected = (
                    host,
                    registrations,
                    selected,
                )
                self._welcomed = value.get("welcomed") is True
                self._migration_required = value.get("migration_required") is True
                for registration in registrations.values():
                    if registration.credentials:
                        register_secrets(
                            token
                            for token in (
                                registration.credentials.access_token,
                                registration.credentials.refresh_token,
                                registration.credentials.id_token,
                            )
                            if token
                        )
            except FileNotFoundError:
                self._registrations, self._selected = {}, None
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                raise ChatGPTAuthError(
                    "Could not read the ChatGPT connection. Check its application-data file and permissions."
                ) from None
            self._loaded = True

    @contextmanager
    def _storage_lock(self, *, invalidate: bool = False) -> Generator[None]:
        with self._lock:
            # Account actions may cancel an in-flight refresh on this instance.
            # Reuse its lock while it unwinds; the generation check prevents its save.
            existing = self._storage_descriptor
            if invalidate and existing is not None:
                yield
                return
        descriptor: int | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._path.parent.chmod(0o700)
            descriptor = os.open(
                self._path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600
            )
            if sys.platform == "win32":
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            raise ChatGPTAuthError(
                "Another Anki process may be updating the ChatGPT connection, or its application-data folder is not writable. Try again."
            ) from None
        with self._lock:
            self._storage_descriptor = descriptor
        try:
            yield
        finally:
            with self._lock:
                self._storage_descriptor = None
            os.close(descriptor)

    def _write(
        self,
        registrations: dict[str, ChatGPTRegistration],
        selected: str | None,
        *,
        welcomed: bool | None = None,
        migration_required: bool | None = None,
    ) -> None:
        temporary: str | None = None
        welcomed = self._welcomed if welcomed is None else welcomed
        migration_required = (
            self._migration_required
            if migration_required is None
            else migration_required
        )
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".chatgpt-", dir=self._path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "version": 2,
                        "ext_agent_host_id": self._host_id,
                        "selected": selected,
                        "registrations": [
                            asdict(item) for item in registrations.values()
                        ],
                        "welcomed": welcomed,
                        "migration_required": migration_required,
                    },
                    output,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self._path)
        except OSError:
            raise ChatGPTAuthError(
                "Could not save the ChatGPT connection. Check application-data folder permissions."
            ) from None
        finally:
            if temporary is not None:
                with suppress(OSError):
                    Path(temporary).unlink(missing_ok=True)
        self._loaded = True
        self._registrations, self._selected = registrations, selected
        self._welcomed, self._migration_required = welcomed, migration_required

    def list_accounts(self) -> list[ChatGPTAccount]:
        self._read()
        return [item.account for item in self._registrations.values()]

    def status(self) -> ChatGPTAccount | None:
        self._read()
        registration = self._registrations.get(self._selected or "")
        return registration.account if registration else None

    def select_account(self, account_id: str) -> None:
        self.cancel_sign_in()
        with self._lock, self._storage_lock(invalidate=True):
            self._read(reload=True)
            if account_id not in self._registrations:
                raise ChatGPTAuthError(
                    "That ChatGPT account is no longer saved. Sign in again."
                )
            self._invalidate()
            self._write(self._registrations, account_id)

    def mark_welcome_seen(self) -> None:
        with self._lock, self._storage_lock():
            self._read(reload=True)
            self._write(self._registrations, self._selected, welcomed=True)

    def _check(self, generation: int) -> None:
        if generation != self.generation:
            raise JobCancelledError(
                "The ChatGPT account changed. Run generation again."
            )

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

    def sign_out(self) -> Awaitable[None]:
        self._invalidate()
        self.cancel_sign_in()
        with self._lock, self._storage_lock(invalidate=True):
            self._read(reload=True)
            registration = self._registrations.get(self._selected or "")
            if registration is not None:
                records = dict(self._registrations)
                records[registration.client_id] = ChatGPTRegistration(
                    registration.client_id, registration.subject, registration.email
                )
                self._write(records, self._selected)
        return self._revoke(registration)

    async def _revoke(self, registration: ChatGPTRegistration | None) -> None:
        if (
            registration is None
            or registration.credentials is None
            or registration.credentials.refresh_token is None
        ):
            return
        try:
            async with asyncio.timeout(10):
                discovery = await self._configuration()
                for attempt in range(2):
                    try:
                        await self._request(
                            discovery["revocation_endpoint"],
                            data={
                                "token": registration.credentials.refresh_token,
                                "token_type_hint": "refresh_token",
                                "client_id": registration.client_id,
                            },
                            empty=True,
                        )
                        return
                    except ChatGPTAuthError:
                        if attempt:
                            raise
                        await asyncio.sleep(0.5)
        except (ChatGPTAuthError, TimeoutError):
            raise ChatGPTAuthError(
                "Signed out locally. Remote revocation was not confirmed; disconnect Smart Notes in ChatGPT Settings → Usage."
            ) from None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None]:
        self._read()
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
        identity = account_registration.get() or self._selected
        registration_token = account_registration.set(identity)
        token = account_epoch.set(generation)
        try:
            yield
            self._check(generation)
        finally:
            account_epoch.reset(token)
            account_registration.reset(registration_token)
            with self._lock:
                count = self._sessions.get(task, 1) - 1
                if count:
                    self._sessions[task] = count
                else:
                    self._sessions.pop(task, None)

    async def _request(
        self, url: str, *, data: dict[str, str] | None = None, empty: bool = False
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(60):
                client = await provider_runtime.get_http_client()
                async with client.stream(
                    "POST" if data is not None else "GET",
                    url,
                    data=data,
                    headers={"Accept": "application/json"},
                    timeout=60,
                    follow_redirects=False,
                ) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        if len(body) + len(chunk) > MAX_AUTH_BYTES:
                            raise ValueError
                        body.extend(chunk)
                    if empty and response.status_code == 200:
                        return {}
                    value = json.loads(body)
                    if not isinstance(value, dict):
                        raise ValueError
                    if response.status_code != 200:
                        error = value.get("error")
                        code = error.get("code") if isinstance(error, dict) else error
                        code = (
                            code
                            if isinstance(code, str)
                            and code in TERMINAL_REFRESH_ERRORS | {"invalid_client"}
                            else None
                        )
                        raise ChatGPTAuthError(
                            "The ChatGPT client registration is invalid. Check the integration configuration."
                            if code == "invalid_client"
                            else "ChatGPT authentication was rejected. Please sign in again."
                            if code
                            else "ChatGPT authentication is temporarily unavailable. Try again later.",
                            code=code,
                        )
                    return value
        except (httpx.HTTPError, TimeoutError, ValueError):
            raise ChatGPTAuthError(
                "Could not complete ChatGPT authentication. Check your connection and try again."
            ) from None

    async def _configuration(self) -> dict[str, Any]:
        if self._discovery is None:
            value = await self._request(DISCOVERY_URL)
            if value.get("issuer") != ISSUER:
                raise ChatGPTAuthError(
                    "ChatGPT returned invalid authentication configuration."
                )
            for name in (
                "authorization_endpoint",
                "token_endpoint",
                "jwks_uri",
                "revocation_endpoint",
            ):
                endpoint = value.get(name)
                url = urlsplit(endpoint) if isinstance(endpoint, str) else None
                if (
                    url is None
                    or url.scheme != "https"
                    or url.netloc != "auth.openai.com"
                    or url.query
                    or url.fragment
                ):
                    raise ChatGPTAuthError(
                        "ChatGPT returned invalid authentication configuration."
                    )
            self._discovery = value
        return self._discovery

    async def _identity(
        self, token: str, client_id: str, nonce: str | None
    ) -> dict[str, Any]:
        discovery = await self._configuration()
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not nonempty_string(header.get("kid")):
                raise ValueError
            keys = self._jwks.get("keys", []) if self._jwks else []
            if time.monotonic() - self._jwks_at > 3600 or not any(
                isinstance(key, dict) and key.get("kid") == header["kid"]
                for key in keys
            ):
                self._jwks = await self._request(discovery["jwks_uri"])
                self._jwks_at = time.monotonic()
                keys = self._jwks.get("keys")
            if not isinstance(keys, list):
                raise ValueError
            keys = [
                key
                for key in keys
                if isinstance(key, dict)
                and key.get("kid") == header["kid"]
                and key.get("kty") == "RSA"
                and key.get("use", "sig") == "sig"
                and key.get("alg", "RS256") == "RS256"
            ]
            if len(keys) != 1:
                raise ValueError
            claims = jwt.decode(
                token,
                keys[0],
                algorithms=["RS256"],
                audience=client_id,
                issuer=ISSUER,
                options={
                    "require_sub": True,
                    "require_exp": True,
                    "require_iat": True,
                    "require_aud": True,
                    "require_iss": True,
                    "leeway": 5,
                    "verify_at_hash": False,
                },
            )
            if (
                not nonempty_string(claims.get("sub"))
                or not isinstance(claims.get("iat"), int)
                or isinstance(claims["iat"], bool)
                or claims["iat"] > time.time() + 5
            ):
                raise ValueError
            if (
                isinstance(claims.get("aud"), list)
                and len(claims["aud"]) > 1
                and claims.get("azp") != client_id
            ):
                raise ValueError
            if claims.get("azp", client_id) != client_id:
                raise ValueError
            if nonce is not None and claims.get("nonce") != nonce:
                raise ValueError
            return claims
        except (JWTError, ValueError, TypeError, OverflowError):
            raise ChatGPTAuthError(
                "ChatGPT identity could not be verified. Please sign in again."
            ) from None

    async def _tokens(
        self,
        value: dict[str, Any],
        registration: ChatGPTRegistration,
        nonce: str | None,
    ) -> ChatGPTRegistration:
        old = registration.credentials
        access, refresh = value.get("access_token"), value.get("refresh_token")
        identity = value.get("id_token")
        expires, scope = value.get("expires_in"), value.get("scope")
        if (
            any(
                item is not None and not nonempty_string(item)
                for item in (access, refresh)
            )
            or not isinstance(value.get("token_type", "Bearer"), str)
            or value.get("token_type", "Bearer").lower() != "bearer"
        ):
            raise ChatGPTAuthError(
                "ChatGPT returned an invalid token response. Please sign in again."
            )
        if not isinstance(scope, str):
            raise ChatGPTAuthError(
                "ChatGPT returned an invalid token response. Please sign in again."
            )
        scopes = tuple(scope.split())
        if (
            expires is not None
            and (
                not isinstance(expires, (int, float))
                or isinstance(expires, bool)
                or not math.isfinite(expires)
                or not 0 < expires <= 86400
            )
        ) or (
            PLAN_SCOPE in scopes
            and (expires is None or value.get("token_type") != "Bearer")
        ):
            raise ChatGPTAuthError(
                "ChatGPT returned an invalid token response. Please sign in again."
            )
        if identity is None and nonce is None and old is not None:
            identity_expiry = old.expires_at
            identity, subject, email = (
                old.id_token,
                registration.subject,
                registration.email,
            )
        else:
            if not isinstance(identity, str):
                raise ChatGPTAuthError(
                    "ChatGPT returned an invalid identity token. Please sign in again."
                )
            claims = await self._identity(identity, registration.client_id, nonce)
            identity_expiry = float(claims["exp"])
            subject = claims["sub"]
            email = claims.get("email")
            email = email if isinstance(email, str) else registration.email
            if registration.subject is not None and subject != registration.subject:
                raise ChatGPTAuthError(
                    "The returned ChatGPT identity does not match the selected account. Sign in with that account again."
                )
        if PLAN_SCOPE in scopes and not all(
            nonempty_string(item) for item in (access, refresh)
        ):
            raise ChatGPTAuthError(
                "ChatGPT returned an invalid token response. Please sign in again."
            )
        credentials = ChatGPTCredentials(
            access,
            refresh,
            identity,
            time.time() + expires if expires is not None else identity_expiry,
            scopes,
        )
        register_secrets(token for token in (access, refresh, identity) if token)
        return ChatGPTRegistration(registration.client_id, subject, email, credentials)

    async def headers(self, expected_generation: int | None = None) -> dict[str, str]:
        generation = (
            expected_generation
            if expected_generation is not None
            else account_epoch.get()
        )
        generation = self.generation if generation is None else generation
        self._check(generation)
        # Lock the shared file across the refresh and reload after acquiring it. A
        # second Anki process must see the rotated token, not refresh the old one.
        deadline = time.monotonic() + 60
        while True:
            guard = self._storage_lock()
            try:
                guard.__enter__()
                break
            except ChatGPTAuthError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.05)
                self._check(generation)
        try:
            with self._lock:
                self._check(generation)
                selected = account_registration.get() or self._selected
                self._read(reload=True)
                if selected is not None and selected != self._selected:
                    self._invalidate()
                    self._check(generation)
                registration = self._registrations.get(self._selected or "")
            if registration is None or registration.credentials is None:
                raise ChatGPTAuthError(
                    "Sign in with ChatGPT in Providers before generating text."
                )
            credentials = registration.credentials
            if PLAN_SCOPE not in credentials.scopes:
                raise ChatGPTAuthError(
                    "ChatGPT plan usage is disabled. Enable it in Providers or choose an API provider."
                )
            if credentials.expires_at <= time.time() + 60:
                if credentials.refresh_token is None:
                    raise ChatGPTAuthError(
                        "Sign in with ChatGPT again to renew this connection."
                    )
                discovery = await self._configuration()
                try:
                    value = await self._request(
                        discovery["token_endpoint"],
                        data={
                            "grant_type": "refresh_token",
                            "client_id": registration.client_id,
                            "refresh_token": credentials.refresh_token,
                            "resource": RESOURCE,
                        },
                    )
                    refreshed = await self._tokens(value, registration, None)
                    with self._lock:
                        self._check(generation)
                        records = dict(self._registrations)
                        records[registration.client_id] = refreshed
                        self._write(records, self._selected)
                    assert refreshed.credentials is not None
                    credentials = refreshed.credentials
                except ChatGPTAuthError as error:
                    if error.code in TERMINAL_REFRESH_ERRORS:
                        with self._lock:
                            self._check(generation)
                            records = dict(self._registrations)
                            records[registration.client_id] = ChatGPTRegistration(
                                registration.client_id,
                                registration.subject,
                                registration.email,
                            )
                            self._write(records, self._selected)
                            self._invalidate()
                    raise
            self._check(generation)
            if PLAN_SCOPE not in credentials.scopes:
                raise ChatGPTAuthError(
                    "ChatGPT plan usage is disabled. Enable it in Providers or choose an API provider."
                )
            return {
                "Authorization": f"Bearer {credentials.access_token}",
                "User-Agent": "AnkiSmartNotes",
            }
        finally:
            guard.__exit__(None, None, None)

    async def sign_in(
        self, open_browser: Callable[[str], None], *, new_account: bool = False
    ) -> ChatGPTAccount:
        task = asyncio.current_task()
        assert task is not None
        with self._lock:
            if self._login_task is not None:
                raise ChatGPTAuthError("ChatGPT sign-in is already in progress.")
            self._login_task = task
        generation = self._invalidate()
        connections: set[asyncio.Task[Any]] = set()
        try:
            with self._lock, self._storage_lock():
                self._read(reload=True)
                self._write(self._registrations, self._selected)
                registration = (
                    None
                    if new_account
                    else self._registrations.get(self._selected or "")
                )
            discovery = await self._configuration()
            verifier, state, nonce = (secrets.token_urlsafe(32) for _ in range(3))
            challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .rstrip(b"=")
                .decode()
            )
            callback: asyncio.Future[tuple[str, str]] = (
                asyncio.get_running_loop().create_future()
            )

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
                        codes, clients = (
                            query.get("code", []),
                            query.get("client_id", []),
                        )
                        client_id = (
                            clients[0]
                            if len(clients) == 1
                            else registration.client_id
                            if not clients and registration
                            else None
                        )
                        if "error" in query:
                            callback.set_exception(
                                ChatGPTAuthError(
                                    "ChatGPT sign-in was declined. Please try again."
                                )
                            )
                        elif (
                            not nonempty_string(client_id)
                            or client_id == DYNAMIC_CLIENT_ID
                            or (
                                registration is not None
                                and client_id != registration.client_id
                            )
                        ):
                            callback.set_exception(
                                ChatGPTAuthError(
                                    "ChatGPT returned an invalid account registration. Please sign in again."
                                )
                            )
                        elif (
                            len(codes) == 1 and codes[0] and isinstance(client_id, str)
                        ):
                            callback.set_result((codes[0], client_id))
                            status, message = (
                                "200 OK",
                                "Sign-in received. Return to Anki Smart Notes.",
                            )
                    writer.write(
                        (
                            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(message.encode())}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{message}"
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
                server = await asyncio.start_server(
                    receive, "127.0.0.1", 0, limit=16384
                )
            except OSError:
                raise ChatGPTAuthError(
                    "Could not start the local ChatGPT sign-in callback. Please try again."
                ) from None
            async with server:
                redirect = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/auth/callback"
                parameters = {
                    "response_type": "code",
                    "client_id": registration.client_id
                    if registration
                    else DYNAMIC_CLIENT_ID,
                    "ext_agent_host_id": self._host_id,
                    "redirect_uri": redirect,
                    "scope": SCOPES,
                    "resource": RESOURCE,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": state,
                    "nonce": nonce,
                }
                if registration is None:
                    parameters["agent_name_hint"] = "Anki Smart Notes"
                else:
                    if registration.email:
                        parameters["login_hint"] = registration.email
                    if registration.credentials:
                        parameters["id_token_hint"] = registration.credentials.id_token
                        if PLAN_SCOPE not in registration.credentials.scopes:
                            parameters["prompt"] = "consent"
                try:
                    open_browser(
                        discovery["authorization_endpoint"]
                        + "?"
                        + urlencode(parameters)
                    )
                except Exception:
                    raise ChatGPTAuthError(
                        "Could not open your browser for ChatGPT sign-in."
                    ) from None
                try:
                    code, client_id = await asyncio.wait_for(callback, LOGIN_TIMEOUT)
                except TimeoutError:
                    raise ChatGPTAuthError(
                        "ChatGPT sign-in timed out. Please try again."
                    ) from None
            self._check(generation)
            if registration is None:
                with self._lock, self._storage_lock():
                    self._read(reload=True)
                    if client_id in self._registrations:
                        raise ChatGPTAuthError(
                            "ChatGPT returned an existing registration for a new account. Select that saved account instead."
                        )
                    registration = ChatGPTRegistration(client_id)
                    records = dict(self._registrations)
                    records[client_id] = registration
                    self._write(records, self._selected or client_id)
            result = await self._request(
                discovery["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect,
                    "resource": RESOURCE,
                },
            )
            authenticated = await self._tokens(result, registration, nonce)
            with self._lock, self._storage_lock():
                self._check(generation)
                self._read(reload=True)
                records = dict(self._registrations)
                records[client_id] = authenticated
                self._write(records, client_id, migration_required=False)
            return authenticated.account
        finally:
            for connection in tuple(connections):
                connection.cancel()
            if connections:
                await asyncio.gather(*connections, return_exceptions=True)
            with self._lock:
                if self._login_task is task:
                    self._login_task = None


chatgpt_auth = ChatGPTAuth()
