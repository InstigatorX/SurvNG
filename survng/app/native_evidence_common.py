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
    """A same-class fragment is not confirmation of the nominated object."""
    intersection = (
        max(0, min(expected["x2"], actual["x2"]) - max(expected["x1"], actual["x1"]))
        * max(0, min(expected["y2"], actual["y2"]) - max(expected["y1"], actual["y1"]))
    )

    def area(box):
        return max(1, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))

    return (
        intersection / area(actual) >= .5
        and intersection / (area(actual) + area(expected) - intersection) >= .3
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
