from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NativeTrackingPlan:
    """Resolved live-graph tracking plan.

    SurvNG never inserts a GStreamer tracker. ``mode`` remains ``off``; callers
    still consume ``tracking_classes`` for application-side activity filtering.
    """

    mode: str
    tracking_classes: tuple[str, ...] | None


def _tracking_classes(native) -> tuple[str, ...] | None:
    values = getattr(native, "tracking_classes", None)
    if values is None:
        return None
    return tuple(
        dict.fromkeys(
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        )
    )


def resolve_native_tracking(detector) -> NativeTrackingPlan:
    """Resolve live-graph tracking. SurvNG never inserts gvatrack.

    ``tracking_classes`` still selects which labels may admit/extend activity in
    the application. Config modes that formerly selected a tracker normalize to
    ``off`` before this helper runs.
    """
    return NativeTrackingPlan(
        mode="off",
        tracking_classes=_tracking_classes(detector.native),
    )
