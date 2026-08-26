from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass

from urllib.parse import unquote

from .config import ApiAuthConfig, ApiScope, WebAuthConfig, WebRole, WebUserConfig


_SECRET_URL_RE = re.compile(
    r"(\b(?:rtsp|rtsps|rtmp|http|https)://)([^:/@\s]+):([^@\s]+)@",
    re.IGNORECASE,
)
_SECRET_FIELD_RE = re.compile(
    r'''(?ix)
    (\b(?:
        api[_-]?key|
        authorization|
        password|
        token|
        access[_-]?token|
        client[_-]?secret|
        refresh[_-]?token|
        id[_-]?token|
        cookie|
        set-cookie
    )\b["']?\s*[=:]\s*)
    (?:"[^"]*"|'[^']*'|[^,;\s}\]]+)
    ''',
)
_AUTHORIZATION_RE = re.compile(
    r"(?i)(\bauthorization\s*[=:]\s*(?:bearer|basic)\s+)[^\s,;]+"
)
_COOKIE_RE = re.compile(r"(?i)(\b(?:cookie|set-cookie)\s*:\s*)[^\r\n]*")


def redact_secret_text(value: object) -> str:
    redacted = _SECRET_URL_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}:***@",
        str(value),
    )
    redacted = _AUTHORIZATION_RE.sub(lambda match: f"{match.group(1)}***", redacted)
    redacted = _COOKIE_RE.sub(lambda match: f"{match.group(1)}***", redacted)
    return _SECRET_FIELD_RE.sub(lambda match: f"{match.group(1)}***", redacted)


SESSION_COOKIE_NAME = "survng_session"
DEFAULT_SESSION_DAYS = 14
SESSION_TTL_SECONDS = DEFAULT_SESSION_DAYS * 24 * 60 * 60
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
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
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n_text),
            r=int(r_text),
            p=int(p_text),
            dklen=len(bytes.fromhex(digest_hex)),
        )
    except (AttributeError, ValueError):
        return False
    return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))


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
) -> str:
    issued = int(time.time() if now is None else now)
    lifetime = SESSION_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
    expires = issued + max(1, lifetime)
    payload = f"{user_id}:{issued}:{expires}"
    digest = hmac.new(
        session_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{digest}"


def decode_session(token: str, session_key: str, *, now: int | None = None) -> str | None:
    try:
        user_id, issued_text, expires_text, digest = token.split(":")
        issued = int(issued_text)
        expires = int(expires_text)
    except ValueError:
        return None
    if not user_id or expires <= int(time.time() if now is None else now) or issued > expires:
        return None
    payload = f"{user_id}:{issued_text}:{expires_text}"
    expected = hmac.new(
        session_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(digest, expected):
        return None
    return user_id


def session_cookie_value(cookie_header: str) -> str:
    for part in cookie_header.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name == SESSION_COOKIE_NAME:
            return unquote(value.strip())
    return ""


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
    if not token:
        return None
    user_id = decode_session(token, auth_config.session_key)
    if user_id is None:
        return None
    matched = next((user for user in auth_config.users if user.id == user_id), None)
    if matched is None:
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


def required_api_scope(method: str, path: str) -> ApiScope:
    normalized_method = method.upper()
    # Process logs can contain operationally sensitive context even after
    # redaction, so they are not part of the integration-facing read surface.
    if path == "/api/logs":
        return "admin"
    if path.startswith("/api/tls"):
        return "admin"
    if path.startswith("/api/auth/") and path not in {"/api/auth/session", "/api/auth/login", "/api/auth/logout", "/api/auth/bootstrap"}:
        return "admin"
    if normalized_method in {"GET", "HEAD", "OPTIONS"}:
        return "read"
    if path.startswith("/api/cameras/") and any(
        path.endswith(suffix) for suffix in _CAMERA_CONTROL_SUFFIXES
    ):
        return "camera:control"
    return "admin"
