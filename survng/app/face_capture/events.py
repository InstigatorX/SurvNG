"""What face capture reads from the event store: person events and their timing."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .moments import Box


def motion_qualification(objects: Any) -> dict[str, Any]:
    for item in objects if isinstance(objects, list) else ():
        if isinstance(item, dict) and item.get("status") == "motion_qualification":
            value = item.get("motion_qualification")
            if isinstance(value, dict):
                return value
    return {}


def has_confirmed_person(objects: Any) -> bool:
    """A person confirmed on the recording (temporal consensus)."""
    return any(
        isinstance(item, dict)
        and str(item.get("label") or "").strip().lower() == "person"
        and item.get("temporal_consensus") is True
        for item in (objects if isinstance(objects, list) else ())
    )


def event_epoch(created_at: str, qualification: dict[str, Any]) -> float:
    """The epoch the event's recorded sampling was anchored to."""
    timing = qualification.get("event_timing")
    if isinstance(timing, dict) and timing.get("sampling_at"):
        try:
            return datetime.fromisoformat(str(timing["sampling_at"])).timestamp()
        except ValueError:
            pass
    value = qualification.get("evidence_frame_at_epoch")
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(created_at).timestamp()


def person_events(
    connection: sqlite3.Connection,
    *,
    since: str,
    until: str,
    cameras: list[str] | None = None,
    limit: int = 200,
    before_id: int | None = None,
) -> list[tuple[int, str, float, dict[str, Any]]]:
    """Recorded person events created in ``[since, until]``, newest first.

    ``before_id`` continues a listing below the last id it returned.
    """
    query = (
        "select id, camera_id, objects_json, created_at from events"
        " where created_at >= ? and created_at <= ? and objects_json like '%\"person\"%'"
    )
    args: list[Any] = [since, until]
    if before_id is not None:
        query += " and id < ?"
        args.append(int(before_id))
    if cameras:
        query += f" and camera_id in ({','.join('?' * len(cameras))})"
        args += cameras
    query += " order by id desc"
    found = []
    for event_id, camera_id, objects_json, created_at in connection.execute(query, args):
        try:
            objects = json.loads(objects_json or "[]")
        except ValueError:
            continue  # one malformed row must not stop every later pass
        if not has_confirmed_person(objects):
            continue
        qualification = motion_qualification(objects)
        found.append((int(event_id), str(camera_id), event_epoch(str(created_at), qualification), qualification))
        if len(found) >= limit:
            break
    return found


def plan_offsets(config: Any, qualification: dict[str, Any], epoch: float, camera_id: str) -> list[float]:
    from ..motion_pipeline.object_detection import _flatten_refinement_stages, resolve_recorded_refinement_plan

    stages = resolve_recorded_refinement_plan(
        config, qualification=qualification, event_at=datetime.fromtimestamp(epoch, timezone.utc),
        camera_id=camera_id,
    )[0]
    return list(_flatten_refinement_stages(stages))


def stored_people(
    connection: sqlite3.Connection, event_id: int, epoch: float, main_size: tuple[int, int] | None,
) -> list[tuple[float, list[Box]]]:
    """The event's stored person boxes by time, in main-stream pixels when the size is known."""
    by_time: dict[float, list[Box]] = {}
    for payload, captured in connection.execute(
        "select payload_json, captured_epoch from scene_observations where event_id = ?", (event_id,)
    ):
        item = json.loads(payload or "{}")
        box = item.get("box") or {}
        if item.get("label") != "person" or not box:
            continue
        sx = sy = 1.0
        if main_size:
            sx = main_size[0] / float(item.get("detection_frame_width") or main_size[0])
            sy = main_size[1] / float(item.get("detection_frame_height") or main_size[1])
        try:
            scaled = (float(box["x1"]) * sx, float(box["y1"]) * sy, float(box["x2"]) * sx, float(box["y2"]) * sy)
        except (KeyError, TypeError, ValueError):
            continue
        by_time.setdefault(round(float(captured) - epoch, 3), []).append(scaled)
    return sorted(by_time.items())
