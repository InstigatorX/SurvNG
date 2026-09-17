from __future__ import annotations

from dataclasses import dataclass


DEEP_SORT_IMPLEMENTATIONS = frozenset({"dlstreamer_deep_sort", "deep-sort"})
DEFAULT_DEEP_SORT_CONFIG = (
    "max_iou_distance=0.7,max_age=30,n_init=3,"
    "max_cosine_distance=0.2,nn_budget=100"
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
    """Map existing detector.tracking settings onto the native GStreamer tracker.

    Deep SORT is intentionally an opt-in, person-only experiment. The default
    native-first behavior remains short-term-imageless and keeps its existing
    tracking-class filter unchanged.
    """
    native = detector.native
    tracking = detector.tracking
    classes = _tracking_classes(native)
    implementation = str(getattr(tracking, "implementation", "")).strip().lower()
    if implementation not in DEEP_SORT_IMPLEMENTATIONS:
        return NativeTrackingPlan(
            mode="short-term-imageless",
            tracking_classes=classes,
        )

    if not bool(getattr(tracking, "reid_enabled", False)):
        raise ValueError(
            "DL Streamer Deep SORT requires detector.tracking.reid_enabled=true"
        )
    model_path = str(getattr(tracking, "reid_model_path", "") or "").strip()
    if not model_path:
        raise ValueError(
            "DL Streamer Deep SORT requires detector.tracking.reid_model_path"
        )
    if int(getattr(native, "inference_interval", 1)) != 1:
        raise ValueError(
            "DL Streamer Deep SORT requires detector.native.inference_interval=1"
        )
    if classes is not None and classes != ("person",):
        raise ValueError(
            "DL Streamer Deep SORT experiment is person-only; "
            "detector.native.tracking_classes must be ['person'] or omitted"
        )
    resolve_device = getattr(tracking, "resolved_reid_device", None)
    if callable(resolve_device):
        device = str(resolve_device()).strip().upper() or "CPU"
    else:
        device = str(getattr(tracking, "reid_device", "CPU") or "CPU").strip().upper()
    return NativeTrackingPlan(
        mode="deep-sort",
        tracking_classes=("person",),
        reid_model_path=model_path,
        reid_device=device or "CPU",
    )
