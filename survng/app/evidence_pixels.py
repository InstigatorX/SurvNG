from __future__ import annotations

from typing import Any


def snapshot_has_luma_only_pixels(detected: dict[str, Any]) -> bool:
    """Use saved-cover provenance, not BGR shape or admission provenance.

    Legacy native live evidence contains EMA luma, even when expanded to three
    channels. Current capture retains its original pixel format through EMA.
    Cover promotion refreshes snapshot_source while deliberately
    retaining the original detection's frame_source/provisional status.
    Unknown legacy sources remain compatible; genuine grayscale night
    images from main/recorded capture must not be rejected by pixel tests.
    """
    snapshot_source = str(detected.get("snapshot_source") or "")
    if snapshot_source in {"recorded_main", "recorded_refinement", "object_tracking"}:
        return False
    frame_source = str(detected.get("frame_source") or "")
    # Detection provenance describes the saved pixels only until a different
    # cover replaces them. Unknown legacy live evidence remains conservative.
    if not snapshot_source or snapshot_source == frame_source:
        pixel_format = detected.get("frame_pixel_format")
        if pixel_format in {"BGR", "GRAY8"}:
            return pixel_format == "GRAY8"
    return (
        snapshot_source in {"live_fast_path", "live_fallback"}
        or frame_source in {"live_fast_path", "live_fallback"}
    )
