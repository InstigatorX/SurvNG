"""Tie captured faces to the incident's tracked people (scene objects).

A face belongs to the scene person whose box holds the face's centre on the
same camera at the face's moment. Scene observations are sparse (a median of
1.7 s from a face, measured here), so the person's box is interpolated between
two of its observations at most 4 s apart, or taken from its nearest
observation within 2 s. When no box holds the face but only one person was
tracked on that camera in the incident, the face is theirs. Scene boxes are in
the detector's frame and face boxes in the main-stream frame, so scene boxes
are scaled first.

Binding is worked out when an incident is shown, not stored: an incident's
scene observations keep arriving while it is open.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

MATCH_SECONDS = 2.0
INTERPOLATE_SECONDS = 4.0
# The face centre may fall this far (a share of the person box) outside it.
BOX_MARGIN = 0.1


def _epoch(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _box(value: Any) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(value, dict):
            box = (float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))
        else:
            box = tuple(float(v) for v in value)[:4]
    except (KeyError, TypeError, ValueError):
        return None
    return box if len(box) == 4 and box[2] > box[0] and box[3] > box[1] else None


def person_observations(scene_objects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each person observation as {object_id, event_id, camera_id, epoch, box, frame}."""
    found = []
    for subject in scene_objects or []:
        if str(subject.get("label") or "").lower() != "person":
            continue
        for observation in subject.get("observations") or []:
            box, epoch = _box(observation.get("box")), _epoch(observation.get("captured_at"))
            if box is None or epoch is None:
                continue
            found.append({
                "object_id": subject.get("id"),
                "event_id": observation.get("event_id"),
                "camera_id": observation.get("camera_id"),
                "epoch": epoch,
                "box": box,
                "frame": (observation.get("detection_frame_width"), observation.get("detection_frame_height")),
            })
    return found


def _scaled(observation: dict[str, Any], frame: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = observation["frame"]
    sx = float(frame[0]) / float(width or frame[0])
    sy = float(frame[1]) / float(height or frame[1])
    x1, y1, x2, y2 = observation["box"]
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def _box_at(track: list[dict[str, Any]], epoch: float, frame: tuple[int, int]) -> tuple[tuple, float] | None:
    """The object's box at ``epoch`` on one camera, and how far (s) it is from an observation."""
    before = [o for o in track if o["epoch"] <= epoch]
    after = [o for o in track if o["epoch"] > epoch]
    if before and after:
        a, b = before[-1], after[0]
        if b["epoch"] - a["epoch"] <= INTERPOLATE_SECONDS:
            w = (epoch - a["epoch"]) / max(b["epoch"] - a["epoch"], 1e-6)
            box = tuple(p + (q - p) * w for p, q in zip(_scaled(a, frame), _scaled(b, frame)))
            return box, min(epoch - a["epoch"], b["epoch"] - epoch)
    nearest = min(track, key=lambda o: abs(o["epoch"] - epoch))
    gap = abs(nearest["epoch"] - epoch)
    return (_scaled(nearest, frame), gap) if gap <= MATCH_SECONDS else None


def bind_faces(
    faces: Iterable[dict[str, Any]],
    scene_objects: Iterable[dict[str, Any]],
    main_size: Callable[[str], tuple[int, int] | None],
) -> dict[int, Any]:
    """face id -> scene object id, for faces that match a tracked person.

    ``faces`` rows carry id, event_id, camera_id, captured_epoch, box_json (or
    box), and frame_width/frame_height when known; ``main_size(camera)`` gives
    the main-stream size for faces stored without it.
    """
    observations = person_observations(scene_objects)
    # camera -> object -> its observations on that camera, in time order
    tracks: dict[str, dict[Any, list[dict[str, Any]]]] = {}
    for observation in observations:
        tracks.setdefault(str(observation["camera_id"]), {}).setdefault(observation["object_id"], []).append(observation)
    for by_object in tracks.values():
        for track in by_object.values():
            track.sort(key=lambda o: o["epoch"])
    bound: dict[int, Any] = {}
    for face in faces:
        raw = face.get("box_json")
        box = _box(json.loads(raw) if isinstance(raw, str) else face.get("box"))
        epoch = _epoch(face.get("captured_epoch"))
        camera = str(face.get("camera_id"))
        if box is None or epoch is None or camera not in tracks:
            continue
        frame = (face.get("frame_width"), face.get("frame_height"))
        if not all(frame):
            frame = main_size(camera)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best: tuple[tuple, Any] | None = None
        if frame and all(frame):
            for object_id, track in tracks[camera].items():
                placed = _box_at(track, epoch, frame)
                if placed is None:
                    continue
                (x1, y1, x2, y2), gap = placed
                mx, my = (x2 - x1) * BOX_MARGIN, (y2 - y1) * BOX_MARGIN
                if not (x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my):
                    continue
                same_event = any(o["event_id"] == face.get("event_id") for o in track)
                # Same event first, then nearest in time, then the tighter box.
                rank = (not same_event, gap, (x2 - x1) * (y2 - y1))
                if best is None or rank < best[0]:
                    best = (rank, object_id)
        if best is None and len(tracks[camera]) == 1:
            # The only person tracked on this camera in the incident.
            best = ((True,), next(iter(tracks[camera])))
        if best is not None:
            bound[int(face["id"])] = best[1]
    return bound
