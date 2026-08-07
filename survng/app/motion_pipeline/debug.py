from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .context import Frame, MotionContext


DEBUG_LAYER_LABELS = {
    "overlay": "Annotated motion",
    "original": "Original frame",
    "processed": "Processed frame",
    "background": "Learned background",
    "difference": "Frame difference",
    "threshold": "Threshold mask",
    "motion_mask": "Clean motion mask",
    "ema_exclusion": "EMA excluded area",
}


def _display_frame(frame: Frame) -> Frame:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def _point(value: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return (
        max(0, min(width - 1, round(value[0] * width))),
        max(0, min(height - 1, round(value[1] * height))),
    )


def _box(
    value: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    return _point((value[0], value[1]), width, height), _point(
        (value[2], value[3]), width, height
    )


def _overlay(context: MotionContext) -> Frame | None:
    base = (
        context.processed_frame
        if context.processed_frame is not None
        else context.original_frame
    )
    if base is None:
        return None
    image = _display_frame(base)
    height, width = image.shape[:2]
    latest_blobs = (
        context.filtered_blob_history[-1].blobs
        if context.filtered_blob_history
        else tuple(context.blobs)
    )
    for blob in latest_blobs:
        top_left, bottom_right = _box(blob.box, width, height)
        cv2.rectangle(image, top_left, bottom_right, (62, 203, 116), 1)
        cv2.circle(image, _point(blob.centroid, width, height), 2, (62, 203, 116), -1)
    track = context.dominant_track
    if track is not None and track.path:
        points = np.asarray(
            [_point(point, width, height) for point in track.path],
            dtype=np.int32,
        )
        if len(points) > 1:
            cv2.polylines(image, [points], False, (29, 161, 242), 2)
        cv2.circle(
            image,
            (int(points[-1][0]), int(points[-1][1])),
            3,
            (29, 161, 242),
            -1,
        )
    cv2.putText(
        image,
        f"score {context.scoring.score:.3f} / {context.scoring.threshold:.3f}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _encode_jpeg(frame: Frame) -> bytes:
    display = _display_frame(frame)
    success, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if not success:
        raise ValueError("could not encode motion debug image")
    return bytes(encoded)


@dataclass(frozen=True, slots=True)
class MotionDebugSnapshot:
    captured_at: float
    accepted: bool
    score: float
    threshold: float
    reason: str
    frame_count: int
    blob_count: int
    track_points: int
    event_state: str
    timings: dict[str, float]
    images: dict[str, bytes]

    @classmethod
    def from_context(cls, context: MotionContext) -> "MotionDebugSnapshot":
        candidates: dict[str, Frame | None] = {
            "original": context.original_frame,
            "processed": context.processed_frame,
            "background": context.background_image,
            "difference": context.difference_image,
            "threshold": (
                context.threshold_mask_history[-1]
                if context.threshold_mask_history
                else None
            ),
            "motion_mask": context.binary_motion_mask,
            "ema_exclusion": context.motion_exclusion_mask,
            "overlay": _overlay(context),
        }
        images = {
            name: _encode_jpeg(frame)
            for name, frame in candidates.items()
            if frame is not None
        }
        return cls(
            captured_at=context.captured_at,
            accepted=context.scoring.accepted,
            score=context.scoring.score,
            threshold=context.scoring.threshold,
            reason=context.scoring.reason,
            frame_count=context.scoring.frame_count,
            blob_count=len(context.blobs),
            track_points=len(context.dominant_track.path) if context.dominant_track else 0,
            event_state=context.event_state.phase.value,
            timings={
                stage_id: round(timing.duration_ms, 3)
                for stage_id, timing in context.timings.items()
            },
            images=images,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "accepted": self.accepted,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "frame_count": self.frame_count,
            "blob_count": self.blob_count,
            "track_points": self.track_points,
            "event_state": self.event_state,
            "timings": dict(self.timings),
            "layers": [
                {"id": layer, "label": DEBUG_LAYER_LABELS[layer]}
                for layer in DEBUG_LAYER_LABELS
                if layer in self.images
            ],
        }


class MotionDebugSnapshotStore:
    def __init__(self, lease_seconds: float = 120.0) -> None:
        self.lease_seconds = max(10.0, float(lease_seconds))
        self._enabled_until = 0.0
        self._snapshot: MotionDebugSnapshot | None = None
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled_until = (
                time.monotonic() + self.lease_seconds if enabled else 0.0
            )
            if not enabled:
                self._snapshot = None

    def enabled(self) -> bool:
        with self._lock:
            return time.monotonic() < self._enabled_until

    def capture(self, context: MotionContext) -> MotionDebugSnapshot | None:
        if not self.enabled():
            return None
        snapshot = MotionDebugSnapshot.from_context(context)
        with self._lock:
            if time.monotonic() < self._enabled_until:
                self._snapshot = snapshot
                return snapshot
        return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            enabled_until = self._enabled_until
            enabled = time.monotonic() < enabled_until
            snapshot = self._snapshot
        return {
            "enabled": enabled,
            "expires_in_seconds": (
                round(max(0.0, enabled_until - time.monotonic()), 1)
                if enabled
                else 0.0
            ),
            "snapshot": snapshot.metadata() if snapshot is not None else None,
        }

    def image(self, layer: str) -> bytes | None:
        with self._lock:
            return self._snapshot.images.get(layer) if self._snapshot is not None else None
