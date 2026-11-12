"""Transcript retention (option B).

* Speech younger than ``speech.retention_days`` is never touched.
* Older speech that is not attached to an incident is deleted.
* Older speech attached to an incident keeps its transcript while a recording
  (main or sub stream) still covers that moment. Once the recording is gone,
  the text and its search entry are removed but the timing row stays, so the
  incident still shows that speech happened.

"Attached" is the same definition incident pages use (speech/attach.py).
Incidents are never deleted, so attachment cannot be the retention bound by
itself; the recording is.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from .attach import ATTACH_PAD_SECONDS, attached_to_any
from .store import SpeechStore

LOGGER = logging.getLogger("uvicorn.error")
BATCH_SIZE = 1000

Ranges = list[tuple[float, float]]


@dataclass(slots=True)
class RetentionResult:
    deleted: int = 0
    expired: int = 0
    kept: int = 0


def _covered(start: float, end: float, ranges: Ranges) -> bool:
    return any(end >= low and start <= high for low, high in ranges)


def apply_speech_retention(
    store: SpeechStore,
    *,
    cutoff: float,
    episode_windows: Callable[[str, float, float], Ranges] | None,
    recording_ranges: Callable[[str, float, float], Ranges] | None,
    batch_size: int = BATCH_SIZE,
) -> RetentionResult:
    result = RetentionResult()
    after_id = 0
    while True:
        candidates = store.retention_candidates(cutoff, after_id=after_id, limit=batch_size)
        if not candidates:
            return result
        after_id = candidates[-1][0]
        by_camera: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for segment_id, camera_id, start, end in candidates:
            by_camera[camera_id].append((segment_id, start, end))
        delete: list[int] = []
        expire: list[int] = []
        for camera_id, rows in by_camera.items():
            low = min(start for _, start, _ in rows) - ATTACH_PAD_SECONDS
            high = max(end for _, _, end in rows) + ATTACH_PAD_SECONDS
            windows = episode_windows(camera_id, low, high) if episode_windows else []
            ranges: Ranges | None = None
            for segment_id, start, end in rows:
                if not attached_to_any(start, end, windows):
                    delete.append(segment_id)
                    continue
                if ranges is None:
                    try:
                        ranges = recording_ranges(camera_id, low, high) if recording_ranges else []
                    except Exception:
                        # Unknown recording state must never destroy a
                        # transcript an incident still points at.
                        LOGGER.warning("Speech retention could not read recordings for %s", camera_id, exc_info=True)
                        ranges = [(float("-inf"), float("inf"))]
                if _covered(start, end, ranges):
                    result.kept += 1
                else:
                    expire.append(segment_id)
        result.deleted += store.delete_segments(delete)
        result.expired += store.expire_content(expire)
