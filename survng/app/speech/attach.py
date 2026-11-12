"""The single definition of "speech attached to an incident".

A speech segment is attached to an incident when it overlaps one of that
incident's per-camera episode windows on the same camera, widened by
ATTACH_PAD_SECONDS on both sides. The pad covers speech that began just
before the episode (segments include pre-roll) or continued after it, and
the wall-clock anchoring of audio. Incident display and transcript
retention both use these functions, so they cannot disagree.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol

ATTACH_PAD_SECONDS = 5.0

Window = tuple[str, float, float]  # camera_id, start_epoch, end_epoch


class SpeechTimingSource(Protocol):
    def timing_rows(self, camera_id: str, start: float, end: float) -> list[tuple[int, float, float, str, bool]]: ...


def padded(start: float, end: float, pad: float = ATTACH_PAD_SECONDS) -> tuple[float, float]:
    return float(start) - pad, float(end) + pad


def overlaps(segment_start: float, segment_end: float, window_start: float, window_end: float) -> bool:
    start, end = padded(window_start, window_end)
    return segment_end >= start and segment_start <= end


def attached_to_any(segment_start: float, segment_end: float, windows: Iterable[tuple[float, float]]) -> bool:
    return any(overlaps(segment_start, segment_end, start, end) for start, end in windows)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def incident_speech(
    source: SpeechTimingSource,
    windows_by_incident: Mapping[str, list[Window]],
) -> dict[str, dict[str, Any]]:
    """Speech counts per incident, with one timing query per camera.

    Returns only incidents that have attached speech. The payload carries
    counts and the padded windows the UI may use to request transcripts;
    it never carries transcript text.
    """
    spans: dict[str, list[float]] = {}
    for windows in windows_by_incident.values():
        for camera_id, start, end in windows:
            low, high = padded(start, end)
            if camera_id in spans:
                spans[camera_id][0] = min(spans[camera_id][0], low)
                spans[camera_id][1] = max(spans[camera_id][1], high)
            else:
                spans[camera_id] = [low, high]
    rows_by_camera = {
        camera_id: source.timing_rows(camera_id, low, high)
        for camera_id, (low, high) in spans.items()
    }
    result: dict[str, dict[str, Any]] = {}
    for incident_id, windows in windows_by_incident.items():
        attached: dict[int, tuple[float, float, bool]] = {}
        padded_windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for camera_id, start, end in windows:
            for segment_id, seg_start, seg_end, _status, has_text in rows_by_camera.get(camera_id, ()):
                if overlaps(seg_start, seg_end, start, end):
                    attached[segment_id] = (seg_start, seg_end, has_text)
                    padded_windows[camera_id].append(padded(start, end))
        if not attached:
            continue
        result[incident_id] = {
            "segments": len(attached),
            "seconds": round(sum(end - start for start, end, _ in attached.values()), 1),
            "transcribed": sum(1 for *_, has_text in attached.values() if has_text),
            "windows": [
                {"camera_id": camera_id, "start_at": _iso(low), "end_at": _iso(high)}
                for camera_id, items in sorted(padded_windows.items())
                for low, high in _merge(items)
            ],
        }
    return result


def _merge(items: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(set(items)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
