"""Upper-body window used by the dedicated face detector.

The live path stretch-resizes this crop. Training crops must use the same
window or the exported boxes will not match what the worker sees.
"""
from __future__ import annotations

import math
from typing import Any


def box_tuple(box: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not isinstance(box, dict):
        return None
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def scale_box_to_frame(
    box: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    dest_width: int,
    dest_height: int,
) -> tuple[float, float, float, float]:
    """Map a box from the detection coordinate plane onto a decoded frame."""
    if source_width <= 0 or source_height <= 0 or dest_width <= 0 or dest_height <= 0:
        return box
    if source_width == dest_width and source_height == dest_height:
        return box
    scale_x = dest_width / source_width
    scale_y = dest_height / source_height
    x1, y1, x2, y2 = box
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def upper_body_window(
    person_box: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Return the inclusive-exclusive crop the face detector already uses.

    Padding is 8% of the person width on each side, 5% of the height above
    the box, and the window ends at 68% of the person height. Crops under
    24 pixels on either side are rejected.
    """
    if frame_width <= 0 or frame_height <= 0:
        return None
    x1, y1, x2, y2 = person_box
    person_width = x2 - x1
    person_height = y2 - y1
    if person_width <= 0 or person_height <= 0:
        return None
    left = max(0, min(frame_width, int(math.floor(x1 - person_width * 0.08))))
    right = max(left, min(frame_width, int(math.ceil(x2 + person_width * 0.08))))
    top = max(0, min(frame_height, int(math.floor(y1 - person_height * 0.05))))
    bottom = max(top, min(frame_height, int(math.ceil(y1 + person_height * 0.68))))
    if right - left < 24 or bottom - top < 24:
        return None
    return left, top, right, bottom


def window_dict(window: tuple[int, int, int, int]) -> dict[str, int]:
    left, top, right, bottom = window
    return {"x1": left, "y1": top, "x2": right, "y2": bottom}
