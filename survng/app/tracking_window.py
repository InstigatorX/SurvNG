"""Recorded tracking windows are independent of representative cover selection."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def recorded_tracking_window(event: dict[str, Any], event_at: datetime, *, before: float,
                             after: float, activity_seconds: float) -> tuple[float, float]:
    anchor = event_at.timestamp()
    qualification = next((item.get("motion_qualification", {}) for item in event.get("objects", [])
                          if item.get("status") == "motion_qualification"), {})
    raw_persistence = qualification.get("features", {}).get("persistence_seconds", 0)
    try:
        persistence = float(raw_persistence)
    except (TypeError, ValueError):
        persistence = 0.0
    # Motion history describes a bounded trigger episode, not unlimited lookback.
    persistence = min(120.0, max(0.0, persistence)) if math.isfinite(persistence) else 0.0
    return anchor - persistence - before, anchor + activity_seconds + after
