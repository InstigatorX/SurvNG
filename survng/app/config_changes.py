"""Recent configuration changes, for the Admin Overview's "Recently changed".

Every configuration save is compared with the configuration it replaces and
the differences are appended to a small JSON-lines log next to config.json.
Values are redacted before they are stored: secret fields record only that
they changed, and URLs lose their credentials. Recording is best effort and
never fails or delays a save beyond the diff itself.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, config_path
from .config_secrets import mask_url_password

LOGGER = logging.getLogger(__name__)

LOG_FILENAME = "config-changes.jsonl"
MAX_ENTRIES = 200
MAX_CHANGES_PER_ENTRY = 40
MAX_VALUE_LENGTH = 120
PRIVATE_FILE_MODE = 0o600

_SECRET_KEY = re.compile(
    r"(password|passwd|secret|token|api_?key|private_?key|session_key|credential)",
    re.IGNORECASE,
)
# Bookkeeping that changes on its own (sign-outs) is not a configuration change.
_IGNORED_PATHS: tuple[tuple[str, ...], ...] = (("web_auth", "revoked_sessions"),)
_ACRONYMS = {
    "url": "URL", "rtsp": "RTSP", "onvif": "ONVIF", "fps": "FPS", "mqtt": "MQTT",
    "api": "API", "id": "ID", "ptz": "PTZ", "hls": "HLS", "gpu": "GPU",
    "cpu": "CPU", "ema": "EMA", "tls": "TLS", "http": "HTTP", "https": "HTTPS",
    "homekit": "HomeKit", "ai": "AI",
}

_actor: ContextVar[str] = ContextVar("survng_config_change_actor", default="")


def _humanize(key: str) -> str:
    words = [word for word in re.split(r"[_\s-]+", str(key)) if word]
    return " ".join(
        _ACRONYMS.get(word.lower(), word.lower().capitalize() if index == 0 else word.lower())
        for index, word in enumerate(words)
    )


def _secret_path(path: Iterable[str]) -> bool:
    return any(_SECRET_KEY.search(str(segment)) for segment in path)


def _display(value: Any, path: tuple[str, ...]) -> str:
    if value is None:
        return "not set"
    if _secret_path(path):
        return "hidden" if value not in ("", None) else "empty"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) else str(value)
    if isinstance(value, str):
        if not value:
            return "empty"
        text = mask_url_password(value) if "://" in value else value
        text = str(text)
        return text if len(text) <= MAX_VALUE_LENGTH else f"{text[:MAX_VALUE_LENGTH - 1]}…"
    if isinstance(value, list):
        if not value:
            return "none"
        if all(isinstance(item, (str, int, float)) and not isinstance(item, bool) for item in value):
            return _display(", ".join(str(item) for item in value), path)
        return f"{len(value)} item{'s' if len(value) != 1 else ''}"
    return "changed"


_IDENTITY_FIELDS = ("id", "name", "username")
_LABEL_FIELDS = ("name", "display_name", "username")


def _item_key(item: Any, index: int) -> str:
    if isinstance(item, dict):
        for field in _IDENTITY_FIELDS:
            if item.get(field):
                return str(item[field])
    return f"#{index}"


def _item_label(item: Any, key: str) -> str:
    if isinstance(item, dict):
        for field in _LABEL_FIELDS:
            if item.get(field):
                return str(item[field])
    return key


def _keyed_list(values: Any) -> bool:
    """Lists of identifiable objects (cameras, zones, users) diff per item;
    anything else, such as zone points, is reported as one value."""
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(item, dict) and any(item.get(field) for field in _IDENTITY_FIELDS) for item in values)
    )


def _diff(
    before: Any,
    after: Any,
    path: tuple[str, ...],
    labels: tuple[str, ...],
    out: list[dict[str, Any]],
) -> None:
    if before == after or path in _IGNORED_PATHS:
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in dict.fromkeys([*before.keys(), *after.keys()]):
            _diff(before.get(key), after.get(key), (*path, key), (*labels, _humanize(key)), out)
        return
    if (_keyed_list(before) or _keyed_list(after)) and isinstance(before or [], list) and isinstance(after or [], list):
        start = len(out)
        before_items = {_item_key(item, index): item for index, item in enumerate(before or [])}
        after_items = {_item_key(item, index): item for index, item in enumerate(after or [])}
        for key in dict.fromkeys([*after_items.keys(), *before_items.keys()]):
            previous = before_items.get(key)
            current = after_items.get(key)
            label = _item_label(current if current is not None else previous, key)
            if previous is None:
                out.append({"path": [*path, key], "label": " › ".join((*labels, label)), "kind": "added", "before": "not set", "after": "added"})
            elif current is None:
                out.append({"path": [*path, key], "label": " › ".join((*labels, label)), "kind": "removed", "before": "present", "after": "removed"})
            else:
                _diff(previous, current, (*path, key), (*labels, label), out)
        if len(out) == start and list(before_items) != list(after_items):
            out.append({"path": list(path), "label": " › ".join(labels), "kind": "value", "before": "previous order", "after": "new order"})
        return
    out.append({
        "path": list(path),
        "label": " › ".join(labels),
        "kind": "added" if before is None else "removed" if after is None else "value",
        "before": _display(before, path),
        "after": _display(after, path),
    })


def describe_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Redacted leaf changes between two configuration dumps.

    Camera changes are labelled with the camera's name instead of the list
    position, and carry ``camera_id``.
    """
    out: list[dict[str, Any]] = []
    previous_cameras = {str(camera.get("id")): camera for camera in previous.get("cameras") or []}
    current_cameras = {str(camera.get("id")): camera for camera in current.get("cameras") or []}
    for camera_id in dict.fromkeys([*current_cameras.keys(), *previous_cameras.keys()]):
        before = previous_cameras.get(camera_id)
        after = current_cameras.get(camera_id)
        name = str((after or before or {}).get("name") or camera_id)
        if before is None:
            out.append({"camera_id": camera_id, "path": ["cameras", camera_id], "label": f"{name} › Camera", "kind": "added", "before": "not set", "after": "added"})
            continue
        if after is None:
            out.append({"camera_id": camera_id, "path": ["cameras", camera_id], "label": f"{name} › Camera", "kind": "removed", "before": "present", "after": "removed"})
            continue
        camera_changes: list[dict[str, Any]] = []
        _diff(before, after, ("cameras", camera_id), (name,), camera_changes)
        out.extend({**change, "camera_id": camera_id} for change in camera_changes)
    if [key for key in current_cameras if key in previous_cameras] != [key for key in previous_cameras if key in current_cameras]:
        out.append({"path": ["cameras"], "label": "Camera order", "kind": "value", "before": "previous order", "after": "new order"})
    system_before = {key: value for key, value in previous.items() if key != "cameras"}
    system_after = {key: value for key, value in current.items() if key != "cameras"}
    _diff(system_before, system_after, (), (), out)
    return out


def actor_from_scope(scope: dict[str, Any]) -> str:
    principal = scope.get("survng_principal")
    if principal is None:
        return ""
    if getattr(principal, "kind", None) == "user":
        return str(getattr(principal, "username", None) or getattr(principal, "name", "") or "")
    name = str(getattr(principal, "name", "") or "")
    return f"API token {name}".strip()


class ConfigChangeActorMiddleware:
    """Expose the signed-in user or API token to configuration saves on this request."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        token = _actor.set(actor_from_scope(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            _actor.reset(token)


class ConfigChangeLog:
    """Bounded, append-only log of configuration changes."""

    def __init__(self, path_factory: Callable[[], Path] | None = None) -> None:
        self._path_factory = path_factory or (lambda: config_path().expanduser().resolve().parent / LOG_FILENAME)
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        # The configuration most recently written through this log. Diffing
        # against it (rather than the in-memory active config) keeps a failed
        # apply that is rolled back on disk visible as two entries.
        self._last_saved: AppConfig | None = None

    @property
    def path(self) -> Path:
        return self._path_factory()

    def record(self, previous: AppConfig | None, current: AppConfig) -> dict[str, Any] | None:
        """Append the differences between two configurations. Never raises."""
        try:
            with self._state_lock:
                base = self._last_saved or previous
                self._last_saved = current.model_copy(deep=True)
            if base is None:
                return None
            changes = describe_changes(base.model_dump(mode="json"), current.model_dump(mode="json"))
            if not changes:
                return None
            entry = {
                "id": uuid.uuid4().hex,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "actor": _actor.get(),
                "changes": changes[:MAX_CHANGES_PER_ENTRY],
                "more": max(0, len(changes) - MAX_CHANGES_PER_ENTRY),
            }
            self._append(entry)
            return entry
        except Exception:
            LOGGER.exception("could not record configuration change history")
            return None

    def _append(self, entry: dict[str, Any]) -> None:
        path = self.path
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read_lines(path)
            lines = [*existing, json.dumps(entry, separators=(",", ":"))][-MAX_ENTRIES:]
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        try:
            return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except FileNotFoundError:
            return []

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Newest first. Unreadable lines are skipped; an unreadable log reads as empty."""
        bounded = max(1, min(int(limit), MAX_ENTRIES))
        try:
            with self._lock:
                lines = self._read_lines(self.path)
        except OSError:
            LOGGER.warning("could not read configuration change history", exc_info=True)
            return []
        entries: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
            if len(entries) >= bounded:
                break
        return entries
