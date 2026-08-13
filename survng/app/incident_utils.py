from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SNAPSHOT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SNAPSHOT_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MEDIA_DIRECTORIES = ("snapshots", "motion_samples", "recordings")
DEFAULT_INCIDENT_GAP_SECONDS = 45


def snapshot_media_type(path: Path) -> str:
    """Return a deterministic MIME type for every accepted evidence format."""
    return SNAPSHOT_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def stable_incident_key(camera_id: str, first_event_id: Any) -> str:
    return f"{camera_id}-{first_event_id}"


def stable_incident_id(camera_id: str, first_event_id: Any) -> str:
    return f"incident-{stable_incident_key(camera_id, first_event_id)}"


def event_epoch(event: dict[str, Any]) -> float:
    try:
        parsed = datetime.fromisoformat(str(event["created_at"]))
    except (KeyError, TypeError, ValueError):
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def incident_event_groups(
    rows: list[dict[str, Any]],
    gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
) -> list[tuple[str, list[dict[str, Any]]]]:
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for event in rows:
        by_camera.setdefault(str(event.get("camera_id") or ""), []).append(event)

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for camera_id, camera_events in by_camera.items():
        ordered = sorted(camera_events, key=event_epoch)
        current: list[dict[str, Any]] = []
        current_end = 0.0
        for event in ordered:
            created_epoch = event_epoch(event)
            if current and created_epoch - current_end > gap_seconds:
                groups.append((camera_id, current))
                current = []
            current.append(event)
            current_end = created_epoch
        if current:
            groups.append((camera_id, current))

    groups.sort(key=lambda item: event_epoch(item[1][-1]), reverse=True)
    return groups


def portable_media_path(storage_dir: Path, path_value: object) -> str:
    """Return a storage-root-relative database value when it can be verified safely."""
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        return path.as_posix()

    storage_root = storage_dir.resolve()
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(storage_root).as_posix()
    except ValueError:
        pass

    # Migrate media written under a different deployment mount (for example,
    # systemd's /mnt/... versus Docker's /media). Only accept a known SurvNG
    # media subtree whose equivalent file is present beneath the current root.
    for directory in MEDIA_DIRECTORIES:
        if directory not in path.parts:
            continue
        relative = Path(*path.parts[path.parts.index(directory) :])
        candidate = (storage_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(storage_root)
        except ValueError:
            continue
        if candidate.is_file():
            return relative.as_posix()
    return raw_path


def stored_media_path(
    storage_dir: Path,
    path_value: object,
    media_storage: MediaStorageRegistry | None = None,
) -> Path:
    """Resolve a stored media reference without allowing escape from storage."""
    raw_path = str(path_value or "").strip()
    if not raw_path:
        raise FileNotFoundError("stored media is unavailable")

    storage_root = storage_dir.resolve()
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else storage_root / path).resolve()
    try:
        resolved.relative_to(storage_root)
    except ValueError as exc:
        if media_storage is None or not media_storage.contains(resolved):
            raise PermissionError("stored media is outside configured media storage") from exc
    if not resolved.is_file():
        raise FileNotFoundError("stored media is unavailable")
    return resolved


def event_snapshot_path(
    storage_dir: Path,
    event: dict[str, Any],
    media_storage: MediaStorageRegistry | None = None,
) -> Path:
    raw_path = str(event.get("snapshot_path") or "")
    if not raw_path:
        raise FileNotFoundError("event snapshot is unavailable")

    try:
        snapshot = stored_media_path(storage_dir, raw_path, media_storage)
    except FileNotFoundError as exc:
        raise FileNotFoundError("event snapshot is unavailable") from exc
    except PermissionError as exc:
        raise PermissionError("event snapshot is outside storage directory") from exc

    if snapshot.suffix.lower() not in SNAPSHOT_SUFFIXES:
        raise PermissionError("event snapshot is not an image")
    return snapshot
from .media_storage import MediaStorageRegistry
