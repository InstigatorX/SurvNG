from __future__ import annotations

from pathlib import Path
from typing import Any


SNAPSHOT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def stable_incident_key(camera_id: str, first_event_id: Any) -> str:
    return f"{camera_id}-{first_event_id}"


def stable_incident_id(camera_id: str, first_event_id: Any) -> str:
    return f"incident-{stable_incident_key(camera_id, first_event_id)}"


def event_snapshot_path(storage_dir: Path, event: dict[str, Any]) -> Path:
    raw_path = str(event.get("snapshot_path") or "")
    if not raw_path:
        raise FileNotFoundError("event snapshot is unavailable")

    storage_root = storage_dir.resolve()
    snapshot = Path(raw_path).resolve()
    try:
        snapshot.relative_to(storage_root)
    except ValueError as exc:
        raise PermissionError("event snapshot is outside storage directory") from exc

    if snapshot.suffix.lower() not in SNAPSHOT_SUFFIXES:
        raise PermissionError("event snapshot is not an image")
    if not snapshot.is_file():
        raise FileNotFoundError("event snapshot is unavailable")
    return snapshot
