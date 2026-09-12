from __future__ import annotations

from typing import Any


def snapshot_has_luma_only_pixels(detected: dict[str, Any]) -> bool:
    """Use saved-cover provenance, not BGR shape or admission provenance.

    Native live evidence contains EMA luma, even when expanded to three
    channels. Cover promotion refreshes snapshot_source while deliberately
    retaining the original detection's frame_source/provisional status.
    Unknown legacy sources remain compatible; genuine grayscale night
    images from main/recorded capture must not be rejected by pixel tests.
    """
    snapshot_source = str(detected.get("snapshot_source") or "")
    if snapshot_source in {"recorded_main", "recorded_refinement", "object_tracking"}:
        return False
    return (
        snapshot_source in {"live_fast_path", "live_fallback"}
        or str(detected.get("frame_source") or "") in {"live_fast_path", "live_fallback"}
    )
