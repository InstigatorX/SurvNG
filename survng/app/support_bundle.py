"""Privacy-safe, bounded support-bundle export for administrator troubleshooting."""

from __future__ import annotations

import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from survng import __version__ as SURVNG_VERSION

from .security import redact_secret_text


SCHEMA_VERSION = 1
MAX_LOG_ROWS = 300
MAX_OPERATIONAL_EVENTS = 200
MAX_DIAGNOSTIC_SESSIONS = 50
MAX_BUNDLE_BYTES = 4 * 1024 * 1024

# Keys which can contain credentials, private material, media, or host layout.
_SENSITIVE_KEYS = {
    "api_key", "api_token", "authorization", "cookie", "password", "passwd",
    "secret", "token", "access_token", "refresh_token", "client_secret",
    "private_key", "privkey", "password_hash", "session_key", "web_session_key",
    "snapshot_path", "recording_path", "media_path", "storage_path", "database_path",
    "certificate_path", "key_path", "fullchain_path", "config_path", "log_path",
    "storage_dir", "database_dir", "config_dir", "working_dir", "home_dir", "cwd",
}
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_:/])/(?:home|root|srv|var|tmp|etc|opt|mnt|media|data|run)/[^\s,;]+"
)
_STREAM_URL_RE = re.compile(r"\b(?:rtsp|rtsps|rtmp|http|https)://[^\s,;]+", re.IGNORECASE)


def _safe_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Return JSON-safe bounded data without secrets or filesystem disclosure."""
    normalized = key.lower().replace("-", "_")
    if normalized in _SENSITIVE_KEYS or any(
        part in normalized
        for part in ("password", "credential", "private_key", "secret", "token", "authorization", "cookie")
    ):
        return "[redacted]"
    if depth > 8:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        if (
            normalized.endswith("url")
            or normalized.endswith("_path")
            or normalized.endswith("_dir")
            or normalized in {"url", "uri", "path", "file"}
        ):
            return "[redacted]"
        redacted = redact_secret_text(value)
        redacted = _STREAM_URL_RE.sub("[redacted-url]", redacted)
        return _ABSOLUTE_PATH_RE.sub("[redacted-path]", redacted)[:1000]
    if isinstance(value, Mapping):
        return {
            str(item_key)[:96]: _safe_value(item, key=str(item_key), depth=depth + 1)
            for item_key, item in list(value.items())[:500]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth=depth + 1) for item in list(value)[:500]]
    return redact_secret_text(str(value))[:1000]


def _collect(label: str, function: Callable[[], Any]) -> Any:
    try:
        return _safe_value(function())
    except Exception as error:  # support export must remain available during failures
        return {"collection_error": label, "message": redact_secret_text(error)[:300]}


@dataclass(frozen=True, slots=True)
class SupportBundleDependencies:
    get_manager: Callable[[], Any]
    get_config: Callable[[], Any]
    system_status: Callable[[Any], dict[str, Any]]
    log_rows: Callable[[], Sequence[dict[str, Any]]]


def create_support_bundle_router(deps: SupportBundleDependencies) -> APIRouter:
    router = APIRouter(tags=["support"])

    @router.get("/api/support-bundle")
    def support_bundle() -> JSONResponse:
        collected_at = datetime.now(timezone.utc).isoformat()
        try:
            manager_object = deps.get_manager()
        except Exception as error:
            manager_object = None
            manager_error = {"collection_error": "manager", "message": redact_secret_text(error)[:300]}
        else:
            manager_error = None

        def config_payload() -> Any:
            config = deps.get_config()
            dumped = config.model_dump(mode="json") if hasattr(config, "model_dump") else vars(config)
            return dumped

        def camera_status() -> Any:
            if manager_object is None:
                return {"collection_error": "manager unavailable"}
            return manager_object.statuses()

        def system_status() -> Any:
            if manager_object is None:
                return {"collection_error": "manager unavailable"}
            return deps.system_status(manager_object)

        def logs() -> Any:
            return list(deps.log_rows())[-MAX_LOG_ROWS:]

        def operational_events() -> Any:
            if manager_object is None:
                return {"collection_error": "manager unavailable"}
            telemetry = getattr(manager_object, "telemetry", None)
            reader = getattr(telemetry, "operational_event_history", None)
            return reader(hours=24)[0:MAX_OPERATIONAL_EVENTS] if callable(reader) else []

        def diagnostic_sessions() -> Any:
            if manager_object is None:
                return {"collection_error": "manager unavailable"}
            telemetry = getattr(manager_object, "telemetry", None)
            reader = getattr(telemetry, "diagnostic_sessions", None)
            return reader(active_only=False)[:MAX_DIAGNOSTIC_SESSIONS] if callable(reader) else []

        payload = {
            "manifest": {
                "schema_version": SCHEMA_VERSION,
                "product": "SurvNG",
                "version": SURVNG_VERSION,
                "collected_at": collected_at,
                "privacy": {
                    "media_included": False,
                    "credentials_included": False,
                    "filesystem_paths_included": False,
                },
            },
            "runtime": manager_error or _collect("system status", system_status),
            "system": {
                "python_version": platform.python_version(),
                "platform": platform.platform(aliased=True),
                "implementation": platform.python_implementation(),
                "architecture": platform.machine(),
                "process": {"python": sys.version.split()[0]},
            },
            "config": _collect("configuration", config_payload),
            "cameras": _collect("camera status", camera_status),
            "logs": _collect("logs", logs),
            "operational_events": _collect("operational events", operational_events),
            "diagnostic_sessions": _collect("diagnostic sessions", diagnostic_sessions),
        }
        body = _safe_value(payload)
        # The individual sections are bounded, but retain a final hard ceiling.
        encoded = JSONResponse(content=body).body
        if len(encoded) > MAX_BUNDLE_BYTES:
            body = {
                "manifest": {
                    **payload["manifest"],
                    "bundle_truncated": True,
                    "maximum_bytes": MAX_BUNDLE_BYTES,
                },
                "system": payload["system"],
                "runtime": {"truncated": True},
                "config": {"truncated": True},
                "cameras": {"truncated": True},
                "logs": {"truncated": True, "count": MAX_LOG_ROWS},
                "operational_events": {"truncated": True},
                "diagnostic_sessions": {"truncated": True},
            }
        stamp = collected_at.replace(":", "").replace("+00:00", "Z")
        return JSONResponse(
            content=body,
            headers={
                "Content-Disposition": f'attachment; filename="survng-support-{stamp}.json"',
                "Cache-Control": "no-store",
            },
        )

    return router
