from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from urllib.parse import unquote

from .config import ApiAuthConfig, ApiScope, WebAuthConfig, WebRole, WebUserConfig
from .redact import redact_secret_text


SESSION_COOKIE_NAME = "survng_session"
DEFAULT_SESSION_DAYS = 14
SESSION_TTL_SECONDS = DEFAULT_SESSION_DAYS * 24 * 60 * 60
_SESSION_REGISTRY_LIMIT = 1000
_SESSION_REGISTRY_LOCK = threading.RLock()


@dataclass(slots=True)
class WebSessionRecord:
    """Non-secret metadata for a signed browser session."""

    token_digest: str
    user_id: str
    issued_at: int
    expires_at: int
    last_seen_at: int
    client_ip: str


_WEB_SESSIONS: dict[str, WebSessionRecord] = {}
_REVOKED_WEB_SESSIONS: dict[str, int] = {}


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register_web_session(token: str, user_id: str, issued_at: int, expires_at: int, client_ip: str = "") -> None:
    digest = _session_digest(token)
    now = int(time.time())
    with _SESSION_REGISTRY_LOCK:
        _prune_web_sessions(now)
        _WEB_SESSIONS[digest] = WebSessionRecord(
            token_digest=digest,
            user_id=user_id,
            issued_at=issued_at,
            expires_at=expires_at,
            last_seen_at=now,
            client_ip=client_ip,
        )
        if len(_WEB_SESSIONS) > _SESSION_REGISTRY_LIMIT:
            oldest = min(_WEB_SESSIONS.values(), key=lambda item: item.last_seen_at)
            _WEB_SESSIONS.pop(oldest.token_digest, None)


def touch_web_session(token: str, client_ip: str = "") -> None:
    now = int(time.time())
    digest = _session_digest(token)
    with _SESSION_REGISTRY_LOCK:
        _prune_web_sessions(now)
        record = _WEB_SESSIONS.get(digest)
        if record is not None:
            record.last_seen_at = now
            if client_ip:
                record.client_ip = client_ip


def revoke_web_session(session_id: str) -> bool:
    with _SESSION_REGISTRY_LOCK:
        for digest, record in tuple(_WEB_SESSIONS.items()):
            if digest == session_id or digest.startswith(session_id):
                _WEB_SESSIONS.pop(digest, None)
                _REVOKED_WEB_SESSIONS[digest] = record.expires_at
                return True
    return False


def revoke_web_sessions_for_user(user_id: str) -> int:
    with _SESSION_REGISTRY_LOCK:
        matching = [digest for digest, record in _WEB_SESSIONS.items() if record.user_id == user_id]
        for digest in matching:
            record = _WEB_SESSIONS.pop(digest)
            _REVOKED_WEB_SESSIONS[digest] = record.expires_at
        return len(matching)


def list_web_sessions(now: int | None = None) -> list[dict[str, Any]]:
    current = int(time.time() if now is None else now)
    with _SESSION_REGISTRY_LOCK:
        _prune_web_sessions(current)
        return [
            {
                "id": record.token_digest[:16],
                "user_id": record.user_id,
                "issued_at": record.issued_at,
                "last_seen_at": record.last_seen_at,
                "expires_at": record.expires_at,
                "client_ip": record.client_ip,
                "duration_seconds": max(0, current - record.issued_at),
            }
            for record in sorted(_WEB_SESSIONS.values(), key=lambda item: item.last_seen_at, reverse=True)
        ]


def _prune_web_sessions(now: int) -> None:
    for digest, record in tuple(_WEB_SESSIONS.items()):
        if record.expires_at <= now:
            _WEB_SESSIONS.pop(digest, None)
    for digest, expires_at in tuple(_REVOKED_WEB_SESSIONS.items()):
        if expires_at <= now:
            _REVOKED_WEB_SESSIONS.pop(digest, None)


def is_web_session_revoked(token: str) -> bool:
    digest = _session_digest(token)
    with _SESSION_REGISTRY_LOCK:
        _prune_web_sessions(int(time.time()))
        return digest in _REVOKED_WEB_SESSIONS
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAX_N = 2**16
_SCRYPT_MAX_R = 16
_SCRYPT_MAX_P = 4
_SCRYPT_MAX_DKLEN = 64
_SCRYPT_MAX_SALT_BYTES = 64
_DUMMY_PASSWORD_HASH = ""


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    token_id: str
    name: str
    scopes: frozenset[ApiScope]
    kind: str = "token"
    role: str | None = None
    username: str | None = None

    def permits(self, required_scope: ApiScope) -> bool:
        return "admin" in self.scopes or required_scope in self.scopes


def scopes_for_web_role(role: WebRole) -> frozenset[ApiScope]:
    if role == "admin":
        return frozenset({"admin"})
    return frozenset({"read"})


def public_user_payload(user: WebUserConfig) -> dict[str, str]:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "role": user.role,
    }


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return (
        f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, n_text, r_text, p_text, salt_hex, digest_hex = password_hash.split("$")
        if scheme != "scrypt":
            return False
        n = int(n_text)
        r = int(r_text)
        p = int(p_text)
        salt = bytes.fromhex(salt_hex)
        digest = bytes.fromhex(digest_hex)
        if (
            n < 2
            or n > _SCRYPT_MAX_N
            or n & (n - 1)
            or r < 1
            or r > _SCRYPT_MAX_R
            or p < 1
            or p > _SCRYPT_MAX_P
            or not salt
            or len(salt) > _SCRYPT_MAX_SALT_BYTES
            or not digest
            or len(digest) > _SCRYPT_MAX_DKLEN
        ):
            return False
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(digest),
        )
    except (AttributeError, ValueError):
        return False
    return hmac.compare_digest(candidate, digest)


def _dummy_password_hash() -> str:
    global _DUMMY_PASSWORD_HASH
    if not _DUMMY_PASSWORD_HASH:
        _DUMMY_PASSWORD_HASH = hash_password("survng-dummy-password")
    return _DUMMY_PASSWORD_HASH


def session_ttl_seconds(session_days: int = DEFAULT_SESSION_DAYS) -> int:
    return int(session_days) * 24 * 60 * 60


def encode_session(
    user_id: str,
    session_key: str,
    *,
    now: int | None = None,
    ttl_seconds: int | None = None,
    session_epoch: int = 0,
    nonce: str | None = None,
) -> str:
    issued = int(time.time() if now is None else now)
    lifetime = SESSION_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
    expires = issued + max(1, lifetime)
    epoch = max(0, int(session_epoch))
    token_nonce = nonce or secrets.token_hex(16)
    payload = f"{user_id}:{issued}:{expires}:{epoch}:{token_nonce}"
    digest = hmac.new(
        session_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{digest}"


def decode_session(
    token: str,
    session_key: str,
    *,
    now: int | None = None,
) -> tuple[str, int] | None:
    try:
        parts = token.split(":")
        if len(parts) == 6:
            user_id, issued_text, expires_text, epoch_text, nonce, digest = parts
            epoch = int(epoch_text)
            if not nonce:
                return None
            payload = f"{user_id}:{issued_text}:{expires_text}:{epoch}:{nonce}"
        elif len(parts) == 5:
            user_id, issued_text, expires_text, epoch_text, digest = parts
            epoch = int(epoch_text)
            payload = f"{user_id}:{issued_text}:{expires_text}:{epoch}"
        elif len(parts) == 4:
            user_id, issued_text, expires_text, digest = parts
            epoch = 0
            payload = f"{user_id}:{issued_text}:{expires_text}"
        else:
            return None
        issued = int(issued_text)
        expires = int(expires_text)
    except ValueError:
        return None
    if not user_id or expires <= int(time.time() if now is None else now) or issued > expires:
        return None
    expected = hmac.new(
        session_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(digest, expected):
        return None
    return user_id, epoch


def session_cookie_value(cookie_header: str) -> str:
    for part in cookie_header.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name == SESSION_COOKIE_NAME:
            return unquote(value.strip())
    return ""


def session_times(token: str) -> tuple[int, int] | None:
    """Return signed session timestamps without exposing or validating secrets."""
    try:
        parts = token.split(":")
        if len(parts) not in {4, 5, 6}:
            return None
        issued = int(parts[1])
        expires = int(parts[2])
    except (IndexError, ValueError):
        return None
    if not parts[0] or issued > expires:
        return None
    return issued, expires


def principal_for_web_user(user: WebUserConfig) -> ApiPrincipal:
    return ApiPrincipal(
        user.id,
        user.display_name or user.username,
        scopes_for_web_role(user.role),
        kind="user",
        role=user.role,
        username=user.username,
    )


def authenticate_session(
    cookie_header: str,
    auth_config: WebAuthConfig,
) -> ApiPrincipal | None:
    if not auth_config.enabled or not auth_config.session_key:
        return None
    token = session_cookie_value(cookie_header)
    if not token or is_web_session_revoked(token):
        return None
    decoded = decode_session(token, auth_config.session_key)
    if decoded is None:
        return None
    user_id, epoch = decoded
    matched = next((user for user in auth_config.users if user.id == user_id), None)
    if matched is None or int(getattr(matched, "session_epoch", 0) or 0) != epoch:
        return None
    return principal_for_web_user(matched)


def find_web_user(auth_config: WebAuthConfig, username: str) -> WebUserConfig | None:
    wanted = username.strip().casefold()
    matched = None
    for user in auth_config.users:
        if user.username.casefold() == wanted:
            matched = user
    return matched


def authenticate_password(
    username: str,
    password: str,
    auth_config: WebAuthConfig,
) -> WebUserConfig | None:
    user = find_web_user(auth_config, username)
    password_hash = user.password_hash if user is not None else _dummy_password_hash()
    if not verify_password(password, password_hash):
        return None
    return user


def is_public_api_path(method: str, path: str) -> bool:
    normalized = method.upper()
    if path == "/api/health":
        return True
    if path == "/api/auth/session" and normalized in {"GET", "HEAD", "OPTIONS"}:
        return True
    if path == "/api/auth/login" and normalized == "POST":
        return True
    if path == "/api/auth/logout" and normalized == "POST":
        return True
    if path == "/api/auth/bootstrap" and normalized == "POST":
        return True
    return False


def hash_api_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def authenticate_api_token(
    authorization: str,
    auth_config: ApiAuthConfig,
) -> ApiPrincipal | None:
    scheme, separator, raw_token = authorization.strip().partition(" ")
    if not separator or scheme.lower() != "bearer" or not raw_token.strip():
        return None
    candidate = hash_api_token(raw_token.strip())
    matched = None
    # Compare every configured digest so timing does not reveal its position.
    for token in auth_config.tokens:
        if hmac.compare_digest(candidate, token.token_hash):
            matched = token
    if matched is None:
        return None
    return ApiPrincipal(matched.id, matched.name, frozenset(matched.scopes))


_CAMERA_CONTROL_SUFFIXES = frozenset({
    "/camera/start",
    "/camera/stop",
    "/recording/start",
    "/recording/stop",
    "/recording",
    "/detection",
})

# These POST endpoints only query existing media/index data. They use POST to
# carry bounded search filters and crop geometry, not to mutate SurvNG state.
_READ_ONLY_POST_PATHS = frozenset({
    "/api/semantic-search/visual",
    "/api/semantic-search/visual-frame",
})


def required_api_scope(method: str, path: str) -> ApiScope:
    normalized_method = method.upper()
    # Process logs can contain operationally sensitive context even after
    # redaction, so they are not part of the integration-facing read surface.
    if path == "/api/logs":
        return "admin"
    if path == "/api/support-bundle":
        return "admin"
    if path.startswith("/api/tls"):
        return "admin"
    if path.startswith("/api/auth/") and path not in {"/api/auth/session", "/api/auth/login", "/api/auth/logout", "/api/auth/bootstrap"}:
        return "admin"
    if normalized_method in {"GET", "HEAD", "OPTIONS"}:
        return "read"
    if normalized_method == "POST" and path in _READ_ONLY_POST_PATHS:
        return "read"
    if path.startswith("/api/cameras/") and any(
        path.endswith(suffix) for suffix in _CAMERA_CONTROL_SUFFIXES
    ):
        return "camera:control"
    return "admin"
