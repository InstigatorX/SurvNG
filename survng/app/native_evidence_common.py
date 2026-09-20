"""Shared immutable-ish evidence values and geometry helpers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Candidate:
    epoch: float
    image: object
    objects: list
    score: float


# Live persist historically wrote "gvatrack"; some recovered/older rows use "native".
NATIVE_TRACKING_IMPLEMENTATIONS = frozenset({"gvatrack", "native"})


def is_native_tracking_implementation(value: object) -> bool:
    """True when object_tracking.implementation is the native cover/evidence path."""
    return value in NATIVE_TRACKING_IMPLEMENTATIONS


def image_quality(image):
    """Reject uniform/corrupt-looking frames without rejecting night exposure."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (min(320, gray.shape[1]), min(180, gray.shape[0])),
    )
    spread = float(np.percentile(small, 98) - np.percentile(small, 2))
    sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
    if spread < 10 or sharp < 2:
        return None
    return min(sharp, 500) / 500 + min(spread, 100) / 100


def matches_object_extent(expected, actual):
    """Return True when ``actual`` confirms the nominated object extent.

    Rejects same-class fragments and oversized blobs: the detection must mostly
    lie inside the expected box, share meaningful IoU, and be similar in area
    (at least 40% of the larger box). Cover/crop verification used to accept a
    small OD patch inside a large projected car; that made incident annotations
    much smaller than the vehicle.
    """
    try:
        expected_box = {
            key: float(expected[key]) for key in ("x1", "y1", "x2", "y2")
        }
        actual_box = {
            key: float(actual[key]) for key in ("x1", "y1", "x2", "y2")
        }
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        expected_box["x1"] < expected_box["x2"]
        and expected_box["y1"] < expected_box["y2"]
        and actual_box["x1"] < actual_box["x2"]
        and actual_box["y1"] < actual_box["y2"]
    ):
        return False

    intersection = (
        max(0.0, min(expected_box["x2"], actual_box["x2"]) - max(expected_box["x1"], actual_box["x1"]))
        * max(0.0, min(expected_box["y2"], actual_box["y2"]) - max(expected_box["y1"], actual_box["y1"]))
    )

    def area(box: dict[str, float]) -> float:
        return max(1.0, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))

    expected_area = area(expected_box)
    actual_area = area(actual_box)
    union = expected_area + actual_area - intersection
    size_ratio = min(expected_area, actual_area) / max(expected_area, actual_area)
    return (
        intersection / actual_area >= 0.5
        and intersection / union >= 0.3
        and size_ratio >= 0.4
    )


def resize_objects(objects, from_size, to_size):
    fw, fh = from_size
    tw, th = to_size
    result = deepcopy(objects)
    for obj in result:
        box = obj.get("box") or {}
        obj["box"] = {
            key: float(box.get(key, 0))
            * (tw / fw if key.startswith("x") else th / fh)
            for key in ("x1", "y1", "x2", "y2")
        }
        obj.update(
            detection_frame_width=tw,
            detection_frame_height=th,
        )
    return result
