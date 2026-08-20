from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .config import ApiAuthConfig, ApiScope


_SECRET_URL_RE = re.compile(
    r"(\b(?:rtsp|rtsps|rtmp|http|https|reolink)://)([^:/@\s]+):([^@\s]+)@",
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


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    token_id: str
    name: str
    scopes: frozenset[ApiScope]

    def permits(self, required_scope: ApiScope) -> bool:
        return "admin" in self.scopes or required_scope in self.scopes


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
    if normalized_method in {"GET", "HEAD", "OPTIONS"}:
        return "read"
    if path.startswith("/api/cameras/") and any(
        path.endswith(suffix) for suffix in _CAMERA_CONTROL_SUFFIXES
    ):
        return "camera:control"
    return "admin"
