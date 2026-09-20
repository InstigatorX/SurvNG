from __future__ import annotations

from dataclasses import dataclass


DEEP_SORT_IMPLEMENTATIONS = frozenset({"dlstreamer_deep_sort", "deep-sort"})
DEFAULT_DEEP_SORT_CONFIG = (
    "max_iou_distance=0.7,max_age=60,n_init=3,"
    "max_cosine_distance=0.3,nn_budget=100,"
    "object_class=person,reid_max_age=30"
)


@dataclass(frozen=True, slots=True)
class NativeTrackingPlan:
    mode: str
    tracking_classes: tuple[str, ...] | None
    reid_model_path: str = ""
    reid_device: str = "CPU"
    deep_sort_config: str = DEFAULT_DEEP_SORT_CONFIG


def _tracking_classes(native) -> tuple[str, ...] | None:
    values = getattr(native, "tracking_classes", None)
    if values is None:
        return None
    normalized = tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        )
    )
    return normalized


def resolve_native_tracking(detector) -> NativeTrackingPlan:
    """Resolve live-graph tracking. The incident path no longer inserts gvatrack.

    ``tracking_classes`` still selects which labels may admit/extend activity in
    the application. Deep SORT remains rejected because it requires gvatrack.
    """
    native = detector.native
    tracking = native.tracking
    classes = _tracking_classes(native)
    if tracking.mode == "deep-sort":
        raise ValueError(
            "DL Streamer Deep SORT is unavailable; "
            "native live detection runs without gvatrack"
        )
    return NativeTrackingPlan(
        mode="off",
        tracking_classes=classes,
    )
