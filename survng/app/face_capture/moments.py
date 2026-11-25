"""Choose the moments worth searching for faces from person boxes over time."""
from __future__ import annotations

from dataclasses import dataclass, field

TRACK_MIN_IOU = 0.3
TRACK_MAX_GAP_SECONDS = 1.0
MOMENTS_PER_TRACK = 4
MOMENT_MIN_SPACING = 0.5
MAX_MOMENTS = 8

Box = tuple[float, float, float, float]


@dataclass
class Track:
    samples: list[tuple[float, Box]] = field(default_factory=list)


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_tracks(samples: list[tuple[float, list[Box]]]) -> list[Track]:
    """Greedy overlap linking of per-sample person boxes into tracks."""
    tracks: list[Track] = []
    for offset, boxes in sorted(samples, key=lambda item: item[0]):
        open_tracks = [t for t in tracks if offset - t.samples[-1][0] <= TRACK_MAX_GAP_SECONDS]
        pairs = sorted(
            ((iou(t.samples[-1][1], box), ti, bi) for ti, t in enumerate(open_tracks) for bi, box in enumerate(boxes)),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for overlap, ti, bi in pairs:
            if overlap < TRACK_MIN_IOU or ti in used_tracks or bi in used_boxes:
                continue
            open_tracks[ti].samples.append((offset, boxes[bi]))
            used_tracks.add(ti)
            used_boxes.add(bi)
        for bi, box in enumerate(boxes):
            if bi not in used_boxes:
                tracks.append(Track([(offset, box)]))
    return tracks


def choose_moments(
    tracks: list[Track],
    *,
    per_track: int = MOMENTS_PER_TRACK,
    spacing: float = MOMENT_MIN_SPACING,
    limit: int = MAX_MOMENTS,
) -> list[float]:
    """Each track's tallest samples (person height), spaced apart, best first."""
    candidates: list[tuple[float, float]] = []
    for track in tracks:
        chosen: list[float] = []
        for offset, box in sorted(track.samples, key=lambda item: item[1][3] - item[1][1], reverse=True):
            if len(chosen) >= per_track:
                break
            if all(abs(offset - other) >= spacing for other in chosen):
                chosen.append(offset)
                candidates.append((box[3] - box[1], offset))
    moments: list[float] = []
    for _, offset in sorted(candidates, reverse=True):
        if len(moments) >= limit:
            break
        if all(abs(offset - other) >= spacing / 2 for other in moments):
            moments.append(round(offset, 3))
    return sorted(moments)


def merge_windows(moments: list[float], radius: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for moment in sorted(moments):
        start, end = moment - radius, moment + radius
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows
