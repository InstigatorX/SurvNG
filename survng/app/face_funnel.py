"""Where faces are lost between "person recorded" and "person named".

Reads the SurvNG database read-only and prints, per camera, how many recorded
person events reached each face stage. It changes nothing and loads no model.

python -m survng.app.face_funnel /path/to/database-dir --days 14
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# A standing adult's face is roughly this share of their height. Used only to
# estimate whether a camera's people are large enough for a usable face.
FACE_WIDTH_PER_PERSON_HEIGHT = 0.09

SEARCH_COUNTERS = (
    "people",
    "people_over_cap",
    "windows_too_small",
    "crops",
    "crops_empty",
    "faces",
)

STAGES = (
    ("person_events", "Recorded person events"),
    ("face_size_possible", "…person large enough for a face (estimate)"),
    ("face_found", "…face found"),
    ("face_stored", "…face kept for recognition"),
    ("face_embedded", "…face embedded"),
    ("face_suggested", "…name suggested"),
    ("face_named", "…named (confirmed or automatic)"),
)


def _box_size(box: Any) -> tuple[float, float] | None:
    if not isinstance(box, dict):
        return None
    try:
        width = float(box["x2"]) - float(box["x1"])
        height = float(box["y2"]) - float(box["y1"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _label(item: Any) -> str:
    return str(item.get("label") or "").strip().lower() if isinstance(item, dict) else ""


def summarize_event(objects: Any) -> dict[str, Any] | None:
    """Reduce one event's stored objects to its face-funnel facts.

    Returns None when the event holds no person confirmed on the recording.
    """
    if not isinstance(objects, list):
        return None
    person_heights = [
        size[1]
        for item in objects
        if _label(item) == "person"
        and item.get("temporal_consensus") is True
        and (size := _box_size(item.get("box"))) is not None
    ]
    if not person_heights:
        return None
    face_sides = [
        min(size)
        for item in objects
        if _label(item) == "face" and (size := _box_size(item.get("box"))) is not None
    ]
    phases: dict[str, Any] = {}
    for item in objects:
        if isinstance(item, dict) and item.get("status") == "motion_qualification":
            qualification = item.get("motion_qualification")
            if isinstance(qualification, dict) and isinstance(
                qualification.get("object_detection_phases_ms"), dict
            ):
                phases = qualification["object_detection_phases_ms"]

    def phase(name: str) -> float:
        value = phases.get(name)
        return float(value) if isinstance(value, (int, float)) else 0.0

    return {
        "person_height": max(person_heights),
        "face_sides": face_sides,
        # Events stored before the counters existed have none of these keys.
        "search_recorded": any(f"face_search_{name}" in phases for name in SEARCH_COUNTERS),
        "search": {name: phase(f"face_search_{name}") for name in SEARCH_COUNTERS},
        "deadline_reached": phase("face_evidence_deadline_reached") > 0,
        "recording_unavailable": phase("face_evidence_recording_unavailable") > 0,
        "evidence_failed": phase("face_evidence_failed") > 0,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 1)


def _new_camera() -> dict[str, Any]:
    return {
        **{key: 0 for key, _ in STAGES},
        "person_heights": [],
        "face_sides": [],
        "search_events": 0,
        "search": {name: 0 for name in SEARCH_COUNTERS},
        "search_found_nothing": 0,
        "deadline_reached": 0,
        "recording_unavailable": 0,
        "evidence_failed": 0,
        "observations": 0,
        "outcomes": {},
        "review": {},
    }


def build_report(
    connection: sqlite3.Connection,
    *,
    since: str,
    face_min_size: int = 48,
) -> dict[str, Any]:
    """Aggregate the funnel for events created at or after `since` (ISO UTC)."""
    connection.row_factory = sqlite3.Row
    cameras: dict[str, dict[str, Any]] = {}
    person_events: dict[int, str] = {}
    found_events: set[int] = set()
    min_person_height = face_min_size / FACE_WIDTH_PER_PERSON_HEIGHT
    for row in connection.execute(
        "select id, camera_id, objects_json from events where created_at >= ? order by id",
        (since,),
    ):
        try:
            summary = summarize_event(json.loads(row["objects_json"] or "[]"))
        except (TypeError, ValueError):
            continue
        if summary is None:
            continue
        camera = cameras.setdefault(str(row["camera_id"]), _new_camera())
        person_events[int(row["id"])] = str(row["camera_id"])
        camera["person_events"] += 1
        camera["person_heights"].append(summary["person_height"])
        camera["face_sides"].extend(summary["face_sides"])
        if summary["person_height"] >= min_person_height:
            camera["face_size_possible"] += 1
        if summary["face_sides"]:
            camera["face_found"] += 1
            found_events.add(int(row["id"]))
        if summary["search_recorded"]:
            camera["search_events"] += 1
            for name in SEARCH_COUNTERS:
                camera["search"][name] += int(summary["search"][name])
            if summary["search"]["crops"] > 0 and summary["search"]["faces"] == 0:
                camera["search_found_nothing"] += 1
        for key in ("deadline_reached", "recording_unavailable", "evidence_failed"):
            camera[key] += int(summary[key])

    has_faces = connection.execute(
        "select 1 from sqlite_master where type = 'table' and name = 'face_observations'"
    ).fetchone()
    oldest_observation = None
    if has_faces:
        oldest_observation = connection.execute(
            "select min(observed_at) from face_observations"
        ).fetchone()[0]
        per_event: dict[int, dict[str, bool]] = {}
        for row in connection.execute(
            "select event_id, recognition_outcome, review_status, person_id, "
            "candidate_person_id from face_observations"
        ):
            camera_id = person_events.get(int(row["event_id"]))
            if camera_id is None:
                continue
            camera = cameras[camera_id]
            camera["observations"] += 1
            outcome = str(row["recognition_outcome"] or "pending")
            review = str(row["review_status"] or "unknown")
            camera["outcomes"][outcome] = camera["outcomes"].get(outcome, 0) + 1
            camera["review"][review] = camera["review"].get(review, 0) + 1
            flags = per_event.setdefault(int(row["event_id"]), {})
            flags["face_stored"] = True
            if outcome == "embedded":
                flags["face_embedded"] = True
            if row["candidate_person_id"] is not None or row["person_id"] is not None:
                flags["face_suggested"] = True
            if row["person_id"] is not None and review in {"confirmed", "auto_identified"}:
                flags["face_named"] = True
        for event_id, flags in per_event.items():
            camera = cameras[person_events[event_id]]
            for key in flags:
                camera[key] += 1
            # Faces from the extra frames are stored without a cover-frame object.
            if event_id not in found_events:
                camera["face_found"] += 1

    for camera in cameras.values():
        heights = camera.pop("person_heights")
        sides = camera.pop("face_sides")
        camera["person_height_px"] = {
            "median": _percentile(heights, 0.5),
            "p90": _percentile(heights, 0.9),
        }
        camera["face_min_side_px"] = {
            "count": len(sides),
            "median": _percentile(sides, 0.5),
            "p90": _percentile(sides, 0.9),
            "below_min_size": sum(1 for side in sides if side < face_min_size),
        }
    return {
        "since": since,
        "face_min_size": face_min_size,
        "oldest_face_observation": oldest_observation,
        "cameras": dict(sorted(cameras.items())),
    }


def _share(part: int, whole: int) -> str:
    return f"{part:>6}  {100 * part / whole:5.1f}%" if whole else f"{part:>6}       -"


def format_report(report: dict[str, Any]) -> str:
    lines = [f"Face funnel for events since {report['since']}"]
    if not report["cameras"]:
        lines.append("No recorded person events in this period.")
        return "\n".join(lines)
    oldest = report["oldest_face_observation"]
    if oldest and oldest > report["since"]:
        lines.append(
            f"Stored faces only reach back to {oldest}: older faces were pruned, "
            "so the last four stages undercount before that time."
        )
    for camera_id, camera in report["cameras"].items():
        total = camera["person_events"]
        lines += ["", f"Camera {camera_id}"]
        lines += [f"  {_share(camera[key], total)}  {title}" for key, title in STAGES]
        heights = camera["person_height_px"]
        sides = camera["face_min_side_px"]
        lines.append(
            f"  Person height: median {heights['median']} px, 90th percentile {heights['p90']} px "
            f"(a {report['face_min_size']} px face needs about "
            f"{report['face_min_size'] / FACE_WIDTH_PER_PERSON_HEIGHT:.0f} px)"
        )
        if sides["count"]:
            lines.append(
                f"  Faces found: {sides['count']}, smaller side median {sides['median']} px, "
                f"90th percentile {sides['p90']} px, {sides['below_min_size']} under "
                f"{report['face_min_size']} px"
            )
        if camera["observations"]:
            lines.append(f"  Stored faces by outcome: {_counts(camera['outcomes'])}")
            lines.append(f"  Stored faces by review: {_counts(camera['review'])}")
        blocked = {
            "deadline reached": camera["deadline_reached"],
            "recording unavailable": camera["recording_unavailable"],
            "search failed": camera["evidence_failed"],
        }
        if any(blocked.values()):
            lines.append(f"  Extra-frame search cut short: {_counts(blocked)}")
        if camera["search_events"]:
            search = camera["search"]
            lines.append(
                f"  Search detail ({camera['search_events']} events with counters): "
                f"{search['people']} people examined, {search['people_over_cap']} skipped over the cap, "
                f"{search['windows_too_small']} too small to crop, {search['crops']} crops searched, "
                f"{search['crops_empty']} with no face, {search['faces']} faces; "
                f"{camera['search_found_nothing']} events searched and found nothing"
            )
        else:
            lines.append("  Search detail: no events with counters yet (recorded from this version on)")
    return "\n".join(lines)


def _counts(values: dict[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(values.items()) if count) or "none"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("database_dir", type=Path, help="directory holding survng.sqlite3")
    parser.add_argument("--days", type=float, default=14.0, help="how far back to look (default 14)")
    parser.add_argument("--face-min-size", type=int, default=48, help="detector.face_min_size in use")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    database = args.database_dir / "survng.sqlite3"
    if not database.is_file():
        parser.error(f"{database} does not exist")
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        report = build_report(connection, since=since, face_min_size=args.face_min_size)
    finally:
        connection.close()
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
