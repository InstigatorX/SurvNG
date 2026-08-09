"""Bounded face-candidate collection across recorded detection samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


MAX_CANDIDATES_PER_TRACK = 4
MAX_CANDIDATES_PER_EVENT = 12
MIN_CANDIDATE_OFFSET_GAP_SECONDS = 0.4
MIN_FACE_ASSOCIATION_IOU = 0.04
MAX_FACE_ASSOCIATION_DISTANCE_RATIO = 2.0


@dataclass(frozen=True, slots=True)
class FaceCandidateSample:
    """One already-decoded frame and its detector observations."""

    offset_seconds: float
    frame: np.ndarray
    objects: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    """A ranked face observation retained only for the current decision."""

    track_id: str
    rank: int
    offset_seconds: float
    frame: np.ndarray
    box: dict[str, float]
    confidence: float
    quality_score: float
    sharpness_score: float
    exposure_score: float
    edge_clearance_ratio: float
    detection_source: str


@dataclass(slots=True)
class _CandidateTrack:
    observations: list[tuple[FaceCandidateSample, dict[str, Any]]]

    @property
    def latest(self) -> dict[str, Any]:
        return self.observations[-1][1]


def collect_face_candidates(
    samples: Iterable[FaceCandidateSample],
    *,
    max_per_track: int = MAX_CANDIDATES_PER_TRACK,
    max_per_event: int = MAX_CANDIDATES_PER_EVENT,
) -> tuple[FaceCandidate, ...]:
    """Associate faces across samples and retain bounded, useful evidence.

    This function deliberately consumes frames that object detection has already
    decoded and faces that it has already detected.  It performs no inference
    and retains references only until the detection result is handled.
    """
    ordered_samples = tuple(samples)
    tracks: list[_CandidateTrack] = []
    for sample in ordered_samples:
        available = set(range(len(tracks)))
        faces = [
            item
            for item in sample.objects
            if str(item.get("label") or "").strip().lower() == "face"
            and _box(item) is not None
        ]
        for detected in sorted(faces, key=_confidence, reverse=True):
            matches = [
                (score, index)
                for index in available
                if (
                    score := _association_score(tracks[index].latest, detected)
                ) is not None
            ]
            if matches:
                _score, index = max(matches)
                track = tracks[index]
                available.remove(index)
            else:
                track = _CandidateTrack([])
                tracks.append(track)
            track.observations.append((sample, detected))

    retained: list[FaceCandidate] = []
    per_track_limit = max(1, int(max_per_track))
    for track_index, track in enumerate(tracks, start=1):
        ranked = sorted(
            track.observations,
            key=lambda item: _candidate_score(item[0], item[1]),
            reverse=True,
        )
        selected: list[tuple[FaceCandidateSample, dict[str, Any]]] = []
        for observation in ranked:
            sample, _detected = observation
            if any(
                abs(sample.offset_seconds - chosen.offset_seconds)
                < MIN_CANDIDATE_OFFSET_GAP_SECONDS
                for chosen, _item in selected
            ):
                continue
            selected.append(observation)
            if len(selected) >= per_track_limit:
                break
        for rank, (sample, detected) in enumerate(selected, start=1):
            box = _box(detected)
            if box is None:
                continue
            crop_result = _padded_crop(sample.frame, box)
            if crop_result is None:
                continue
            crop, crop_box = crop_result
            retained.append(
                FaceCandidate(
                    track_id=f"face-{track_index}",
                    rank=rank,
                    offset_seconds=float(sample.offset_seconds),
                    frame=crop,
                    box=crop_box,
                    confidence=_confidence(detected),
                    quality_score=_finite_score(detected.get("face_quality_score")),
                    sharpness_score=_finite_score(detected.get("face_sharpness_score")),
                    exposure_score=_finite_score(detected.get("face_exposure_score")),
                    edge_clearance_ratio=_edge_clearance(box, sample.frame),
                    detection_source=str(detected.get("detection_source") or "object_detector"),
                )
            )
    retained.sort(
        key=lambda item: (
            item.rank,
            -item.quality_score,
            -item.confidence,
            item.track_id,
        )
    )
    return tuple(retained[: max(1, int(max_per_event))])


def _candidate_score(
    sample: FaceCandidateSample,
    detected: dict[str, Any],
) -> tuple[float, float, float, float]:
    box = _box(detected)
    clearance = _edge_clearance(box, sample.frame) if box is not None else 0.0
    quality = _finite_score(detected.get("face_quality_score"))
    sharpness = _finite_score(detected.get("face_sharpness_score"))
    exposure = _finite_score(detected.get("face_exposure_score"))
    utility = (
        0.48 * quality
        + 0.18 * sharpness
        + 0.12 * exposure
        + 0.12 * _confidence(detected)
        + 0.10 * min(1.0, clearance / 0.08)
    )
    return utility, quality, _confidence(detected), -abs(sample.offset_seconds)


def _association_score(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> float | None:
    previous_parent = previous.get("parent_person_box")
    current_parent = current.get("parent_person_box")
    if isinstance(previous_parent, dict) and isinstance(current_parent, dict):
        parent_score = _association_score(
            {"box": previous_parent},
            {"box": current_parent},
        )
        # A dedicated face belongs to its detected person. Refusing a distant
        # parent match prevents two nearby or crossing faces from swapping IDs.
        return 4.0 + parent_score if parent_score is not None else None
    previous_box = _box(previous)
    current_box = _box(current)
    if previous_box is None or current_box is None:
        return None
    px1, py1, px2, py2 = previous_box
    cx1, cy1, cx2, cy2 = current_box
    intersection = max(0.0, min(px2, cx2) - max(px1, cx1)) * max(
        0.0, min(py2, cy2) - max(py1, cy1)
    )
    union = (px2 - px1) * (py2 - py1) + (cx2 - cx1) * (cy2 - cy1) - intersection
    iou = intersection / union if union > 0.0 else 0.0
    if iou >= MIN_FACE_ASSOCIATION_IOU:
        return 2.0 + iou
    previous_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
    current_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
    distance = math.dist(previous_center, current_center)
    scale = max(
        math.hypot(px2 - px1, py2 - py1),
        math.hypot(cx2 - cx1, cy2 - cy1),
        1.0,
    )
    ratio = distance / scale
    if ratio > MAX_FACE_ASSOCIATION_DISTANCE_RATIO:
        return None
    return 1.0 - ratio / MAX_FACE_ASSOCIATION_DISTANCE_RATIO


def _edge_clearance(
    box: tuple[float, float, float, float],
    frame: np.ndarray,
) -> float:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    x1, y1, x2, y2 = box
    return round(
        max(
            0.0,
            min(x1 / width, y1 / height, (width - x2) / width, (height - y2) / height),
        ),
        5,
    )


def _padded_crop(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    padding: float = 0.2,
) -> tuple[np.ndarray, dict[str, float]] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, min(width, int(math.floor(x1 - pad_x))))
    top = max(0, min(height, int(math.floor(y1 - pad_y))))
    right = max(left, min(width, int(math.ceil(x2 + pad_x))))
    bottom = max(top, min(height, int(math.ceil(y2 + pad_y))))
    if right <= left or bottom <= top:
        return None
    # The explicit copy establishes a small immutable ownership boundary and
    # allows the much larger decoded sample frame to be released immediately.
    crop = np.ascontiguousarray(frame[top:bottom, left:right]).copy()
    crop.setflags(write=False)
    return crop, {
        "x1": float(x1 - left),
        "y1": float(y1 - top),
        "x2": float(x2 - left),
        "y2": float(y2 - top),
    }


def _confidence(detected: dict[str, Any]) -> float:
    return _finite_score(detected.get("confidence"))


def _finite_score(value: object) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _box(value: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = value.get("box")
    if not isinstance(raw, dict):
        return None
    try:
        box = tuple(float(raw[name]) for name in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not all(math.isfinite(item) for item in box) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
