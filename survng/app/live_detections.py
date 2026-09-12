"""Bounded, session-qualified detector evidence shared by motion and tracking.

PTS is media time, never wall time. Missing evidence is distinct from a completed
inference with no objects. No image allocations or model calls belong here.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    source_pts: float
    inference_sequence: int
    width: int
    height: int
    objects: tuple[dict[str, Any], ...]
    session: str = ""

    @classmethod
    def parse(cls, payload: dict[str, Any], *, session: str = "") -> DetectionSnapshot:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported detection snapshot schema")
        pts = payload.get("source_pts")
        if isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts) or pts < 0:
            raise ValueError("invalid detection PTS")
        for key in ("inference_sequence", "width", "height"):
            if type(payload.get(key)) is not int or payload[key] <= 0:
                raise ValueError(f"invalid detection {key}")
        objects = payload.get("objects")
        if not isinstance(objects, list):
            raise ValueError("detection objects must be a list")
        for obj in objects:
            if not isinstance(obj, dict) or not str(obj.get("label") or "").strip():
                raise ValueError("invalid detection label")
            box = obj.get("box")
            if not isinstance(box, dict):
                raise ValueError("invalid detection box")
            values = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
            confidence = obj.get("confidence")
            if not all(type(v) in (int, float) and math.isfinite(v) for v in [*values, confidence]):
                raise ValueError("nonfinite detection geometry or confidence")
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1 or not 0 <= confidence <= 1:
                raise ValueError("invalid detection geometry or confidence")
        return cls(float(pts), payload["inference_sequence"], payload["width"], payload["height"],
                   tuple(deepcopy(objects)), session)

    def scaled_objects(self, width: int, height: int) -> list[dict[str, Any]]:
        if width <= 0 or height <= 0:
            return []
        result = []
        for raw in self.objects:
            obj = deepcopy(raw)
            box = obj["box"]
            for key in ("x1", "x2"):
                box[key] = max(0.0, min(float(width), box[key] * width / self.width))
            for key in ("y1", "y2"):
                box[key] = max(0.0, min(float(height), box[key] * height / self.height))
            if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
                result.append(obj)
        return result


class DetectionHistory:
    """Caller owns locking; history never spans a reconnect or PTS reset."""

    def __init__(self) -> None:
        self.snapshots: deque[DetectionSnapshot] = deque(maxlen=32)
        self.session = ""
        self.counts: Counter[str] = Counter()

    def reset(self, session: str = "") -> None:
        self.snapshots.clear()
        self.session = session
        self.counts["resets"] += 1

    def add(self, snapshot: DetectionSnapshot) -> None:
        if not snapshot.session or snapshot.session != self.session:
            self.counts["wrong_session"] += 1
            return
        if self.snapshots and (
            snapshot.source_pts <= self.snapshots[-1].source_pts
            or snapshot.inference_sequence <= self.snapshots[-1].inference_sequence
        ):
            self.counts["out_of_order"] += 1
            return
        self.snapshots.append(snapshot)
        self.counts["snapshots"] += 1
        self.counts["empty_snapshots"] += not snapshot.objects

    def match(self, *, pts: float, session: str, detect_fps: float) -> DetectionSnapshot | None:
        if not session or session != self.session:
            self.counts["wrong_session"] += 1
            return None
        if not math.isfinite(pts) or pts < 0:
            self.counts["invalid_pts"] += 1
            return None
        fps = detect_fps if math.isfinite(detect_fps) and detect_fps > 0 else 5.0
        tolerance = min(0.5, 1.0 / fps + 0.05)
        snapshot = next((item for item in reversed(self.snapshots) if item.source_pts <= pts), None)
        if snapshot is None:
            self.counts["unmatched"] += 1
            return None
        if pts - snapshot.source_pts > tolerance:
            self.counts["stale"] += 1
            return None
        self.counts["matched"] += 1
        return snapshot

    def status(self) -> dict[str, int]:
        return {**self.counts, "history_size": len(self.snapshots)}
