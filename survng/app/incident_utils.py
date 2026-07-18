from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


SNAPSHOT_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INCIDENT_GAP_SECONDS = 45


def stable_incident_key(camera_id: str, first_event_id: Any) -> str:
    return f"{camera_id}-{first_event_id}"


def stable_incident_id(camera_id: str, first_event_id: Any) -> str:
    return f"incident-{stable_incident_key(camera_id, first_event_id)}"


def event_epoch(event: dict[str, Any]) -> float:
    return datetime.fromisoformat(str(event["created_at"])).timestamp()


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
