"""Motion qualification result type and shared image helpers.

The fixed-threshold frame-difference scorer that used to live here was
retired with its pipeline stages; EMA scoring is in
``motion_pipeline.adaptive_stages``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


MOTION_SCORE_THRESHOLDS = {
    "high": 0.36,
    "balanced": 0.48,
    "low": 0.60,
}


@dataclass(frozen=True)
class MotionQualificationResult:
    accepted: bool
    score: float
    threshold: float
    reason: str
    frame_count: int
    features: dict[str, Any]
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocess_motion_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(frame, (5, 5), 0)


def preprocess_motion_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [preprocess_motion_frame(frame) for frame in frames]


def morphology_motion_masks(
    masks: list[np.ndarray],
    kernel_size: int = 3,
    close_iterations: int = 2,
) -> list[np.ndarray]:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed: list[np.ndarray] = []
    for mask in masks:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed.append(
            cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        )
    return processed
