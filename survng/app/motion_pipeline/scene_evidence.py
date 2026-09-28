"""Evidence acquisition independent of presentation and notification policy."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def scene_observation(
    detected: dict[str, Any], *, captured_at_epoch: float | None,
    frame_source: str, recording_path: str = "", offset_seconds: float | None = None,
    frame_timestamp_exact: bool = False, scene_track_key: str | None = None,
) -> dict[str, Any] | None:
    """Copy a usable observation without treating alert policy as evidence policy.

    A source frame reference is authoritative. Off-cover observations never
    inherit the representative image's path or dimensions.
    """
    if not detected.get("label") or not isinstance(detected.get("box"), dict):
        return None
    try:
        confidence = float(detected.get("confidence", 0))
        floor = float(detected.get("temporal_candidate_threshold", 0))
        box = {key: float(detected["box"][key]) for key in ("x1", "y1", "x2", "y2")}
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(value) for value in (*box.values(), confidence, floor)):
        return None
    if confidence < floor or box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        return None
    observation = {
        key: value for key, value in detected.items()
        if not key.startswith("_") and key not in {"snapshot_path", "image_path"}
    }
    observation.update({
        "label": str(detected["label"]), "confidence": confidence, "box": box,
        "frame_source": frame_source,
        "frame_timestamp_exact": bool(frame_timestamp_exact),
    })
    if captured_at_epoch is not None and math.isfinite(captured_at_epoch):
        observation["captured_at_epoch"] = round(captured_at_epoch, 6)
        observation["frame_captured_at_epoch"] = round(captured_at_epoch, 6)
    if recording_path:
        observation["recording_path"] = recording_path
    if offset_seconds is not None:
        observation["offset_seconds"] = round(offset_seconds, 6)
    if scene_track_key:
        observation["scene_track_key"] = scene_track_key
    identity = [frame_source, recording_path, captured_at_epoch, offset_seconds,
                observation["label"], box]
    observation["observation_key"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]
    return observation


def scene_batches(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [observation for item in objects
            if item.get("status") == "scene_observations"
            for observation in item.get("observations", [])
            if isinstance(observation, dict) and observation.get("label")]
