"""Stdlib-only secret redaction for logs and isolated child processes."""

from __future__ import annotations

import re


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
