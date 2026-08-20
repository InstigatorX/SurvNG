from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ..visual_quality import image_quality


LOGGER = logging.getLogger("survng.app.faces")
FACE_QUALITY_VERSION = 2
FACE_OUTCOME_PENDING = "pending"
FACE_OUTCOME_EMBEDDED = "embedded"
FACE_OUTCOME_TOO_SMALL = "too_small"
FACE_OUTCOME_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FaceQuality:
    score: float
    sharpness: float
    exposure: float
    contrast: float
    size: float
    edge_detail: float


@dataclass(frozen=True, slots=True)
class FaceMatch:
    person_id: int | None
    score: float | None
    runner_up_score: float | None
    margin: float | None
    reference_ids: tuple[int, ...]
    reference_scores: tuple[float, ...]


class FaceTooSmallError(ValueError):
    """The detected crop cannot produce a reliable face embedding."""


def _face_crop(
    frame: np.ndarray,
    box: dict[str, float],
    *,
    padding: float = 0.12,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1 = float(box["x1"]), float(box["y1"])
    x2, y2 = float(box["x2"]), float(box["y2"])
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _face_quality(face: np.ndarray, detector_confidence: float) -> FaceQuality:
    if face.size == 0:
        return FaceQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    height, width = face.shape[:2]
    visual = image_quality(face, max_dimension=256)
    size = max(0.0, min(1.0, min(height, width) / 160.0))
    confidence = max(0.0, min(1.0, float(detector_confidence)))
    score = (
        0.35 * visual.sharpness
        + 0.20 * visual.exposure
        + 0.15 * visual.contrast
        + 0.20 * size
        + 0.10 * confidence
    )
    return FaceQuality(
        round(score, 4),
        round(visual.sharpness, 4),
        round(visual.exposure, 4),
        round(visual.contrast, 4),
        round(size, 4),
        round(visual.edge_detail, 4),
    )


def parse_face_box(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        box = {name: float(value[name]) for name in ("x1", "y1", "x2", "y2")}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in box.values()):
        return None
    if box["x1"] < 0 or box["y1"] < 0:
        return None
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        return None
    return box


