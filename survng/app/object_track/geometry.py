from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import ObjectTrackingConfig
from .types import AppearanceEncoder, Box, ObjectDetectorBackend

LOGGER = logging.getLogger("survng.app.object_tracking")


def _rescale_detection_boxes(
    objects: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> None:
    if (
        source_width <= 0
        or source_height <= 0
        or target_width <= 0
        or target_height <= 0
        or (source_width == target_width and source_height == target_height)
    ):
        return
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    for detected in objects:
        box = detected.get("box")
        if not isinstance(box, dict):
            continue
        try:
            box["x1"] = float(box["x1"]) * scale_x
            box["y1"] = float(box["y1"]) * scale_y
            box["x2"] = float(box["x2"]) * scale_x
            box["y2"] = float(box["y2"]) * scale_y
        except (KeyError, TypeError, ValueError):
            continue


def _labeled_tracking_objects(raw: object) -> list[dict[str, Any]]:
    """Copy labeled sidecar boxes for live tracking without calling OpenVINO."""
    if not isinstance(raw, list):
        return []
    objects: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or _box(item.get("box")) is None:
            continue
        copied = dict(item)
        copied["label"] = label
        box = item.get("box")
        if isinstance(box, dict):
            copied["box"] = dict(box)
        objects.append(copied)
    return objects


def _detect_tracking_objects(
    detector: ObjectDetectorBackend,
    frame: np.ndarray,
    confidence_threshold: float,
    *,
    enrichment: bool = False,
) -> list[dict[str, Any]]:
    method_name = "detect_enrichment" if enrichment else "detect_tracking"
    method = getattr(detector, method_name, detector.detect)
    return list(method(frame, confidence_threshold=confidence_threshold))

def _inference_deferred(objects: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(item, dict) and item.get("status") == "inference_deferred"
        for item in objects
    )

def _encoder_supports_label(
    encoder: AppearanceEncoder,
    config: ObjectTrackingConfig,
    label: str,
) -> bool:
    if not config.reid_enabled_for_label(label):
        return False
    supports = getattr(encoder, "supports_label", None)
    return bool(supports(label)) if callable(supports) else label == "person"

def _encode_appearance(
    encoder: AppearanceEncoder,
    label: str,
    crop: np.ndarray,
) -> np.ndarray:
    embed_for_label = getattr(encoder, "embed_for_label", None)
    if callable(embed_for_label):
        return np.asarray(embed_for_label(label, crop), dtype=np.float32)
    return np.asarray(encoder.embed(crop), dtype=np.float32)

def _box(value: object) -> Box | None:
    if not isinstance(value, dict):
        return None
    try:
        x1 = float(value["x1"])
        y1 = float(value["y1"])
        x2 = float(value["x2"])
        y2 = float(value["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(item) for item in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _confidence(detection: dict[str, Any]) -> float:
    try:
        value = float(detection.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0

def _iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)

def _appearance(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.all(np.isfinite(vector)) or norm <= 1e-9:
        return None
    return vector / norm

def _ensure_detection_appearance(
    detection: dict[str, Any],
    reason: str = "unspecified",
) -> np.ndarray | None:
    """Resolve a detection's appearance at most once, only when association needs it."""
    embedding = _appearance(detection.get("_tracking_embedding"))
    if embedding is not None:
        return embedding
    provider = detection.pop("_tracking_embedding_provider", None)
    if not callable(provider):
        return None
    detection["_tracking_embedding_reason"] = reason
    try:
        embedding = _appearance(provider())
    except Exception:
        # Providers normally report sanitized details through session telemetry.
        # Keep this defensive boundary free of exception text in case a future
        # provider includes a credential-bearing model or camera path.
        LOGGER.warning("lazy appearance provider failed")
        return None
    if embedding is not None:
        detection["_tracking_embedding"] = embedding
    return embedding
