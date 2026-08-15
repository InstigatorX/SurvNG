from __future__ import annotations

import logging
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from ..config import CameraConfig
from ..face_candidates import FaceCandidate, FaceCandidateSample, collect_face_candidates
from ..ffmpeg_hw import recorded_frame_hw_args
from ..visual_quality import VisualQuality, image_quality
from ..zones import apply_detection_zones, detection_threshold
from .context import Frame


LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_STAGES = (
    (-1.0, -0.5, 0.0, 0.5, 1.0),
    (4.0, 4.5),
    (8.0, 8.5),
    (12.0, 12.5),
)
RECORDED_EVENT_FRAME_OFFSETS = tuple(
    offset
    for stage in RECORDED_EVENT_FRAME_STAGES
    for offset in stage
)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 24.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0
RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS = 6.0
FAST_LIVE_FRAME_MAX_AGE_SECONDS = 1.0
FAST_LIVE_FRAME_FUTURE_TOLERANCE_SECONDS = 0.5
TEMPORAL_ASSOCIATION_MIN_IOU = 0.05
TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO = 2.5
REPRESENTATIVE_DYNAMIC_DISPLACEMENT_RATIO = 0.01
REPRESENTATIVE_DYNAMIC_PATH_RATIO = 0.025
REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO = 0.01
REPRESENTATIVE_MIN_SUBJECT_AREA_RATIO = 0.0025
REPRESENTATIVE_MIN_QUALITY_SCORE = 0.25
SEMANTIC_RESCUE_THRESHOLD_FRACTION = 0.5


@dataclass(frozen=True)
class _RecordedDetectionSample:
    offset: float
    frame: Frame
    objects: list[dict[str, Any]]
    recording_path: str
    requested_offset: float | None = None
    exact_timestamp: bool = False


@dataclass(frozen=True)
class RecordedDetectionResult:
    """Recorded evidence plus truthful phase timings for one sampling pass."""

    frame: Frame | None
    objects: list[dict[str, Any]]
    recording_path: str
    timings_ms: dict[str, float]
    refinement_pending: bool = False
    face_candidates: tuple[FaceCandidate, ...] = ()

    def __iter__(self):
        # Preserve the historical three-value provider contract for callers
        # that do not need refinement or timing metadata.
        yield self.frame
        yield self.objects
        yield self.recording_path


@dataclass(frozen=True, slots=True)
class TimestampedLiveFrame:
    """Provenance required before live evidence can become provisional truth."""

    frame: Frame
    captured_at_epoch: float
    captured_at_monotonic: float
    sequence: int
    camera_generation: int
    capture_generation: int
    source: str = "live"


@dataclass(frozen=True)
class _DecodedRecordedFrame:
    frame: Frame
    actual_offset: float
    exact_timestamp: bool


@dataclass
class _TemporalDetectionEvidence:
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, sample_index: int, detected: dict[str, Any]) -> None:
        self.observations[sample_index] = detected

    @property
    def latest(self) -> dict[str, Any]:
        return self.observations[max(self.observations)]

    @property
    def label_votes(self) -> dict[str, int]:
        votes: dict[str, int] = {}
        for detected in self.observations.values():
            label = str(detected.get("label") or "").strip()
            if label:
                votes[label] = votes.get(label, 0) + 1
        return votes

    @property
    def winning_label(self) -> str:
        def label_score(label: str) -> tuple[int, float, float, str]:
            confidences = [
                _confidence(item)
                for item in self.observations.values()
                if str(item.get("label") or "").strip() == label
            ]
            return (
                len(confidences),
                float(median(confidences)),
                max(confidences, default=0.0),
                label,
            )

        return max(self.label_votes, key=label_score, default="")

    @property
    def winning_observations(self) -> list[dict[str, Any]]:
        label = self.winning_label
        return [
            item
            for item in self.observations.values()
            if str(item.get("label") or "").strip() == label
        ]

    @property
    def aggregate_confidence(self) -> float:
        return float(median(_confidence(item) for item in self.winning_observations))

    @property
    def peak_confidence(self) -> float:
        return max((_confidence(item) for item in self.winning_observations), default=0.0)


_ImageQuality = VisualQuality


def _image_quality(frame: Frame) -> _ImageQuality:
    return image_quality(frame)


def _sample_image_quality(
    sample: _RecordedDetectionSample,
    visible: list[_TemporalDetectionEvidence],
    sample_index: int,
) -> _ImageQuality:
    """Prefer object-crop quality so timestamps and static scenery cannot win."""
    regions: list[Frame] = []
    height, width = sample.frame.shape[:2]
    for track in visible:
        detected = track.observations.get(sample_index)
        box = _box(detected) if detected is not None else None
        if box is None:
            continue
        x1, y1, x2, y2 = box
        left = max(0, min(width, int(math.floor(x1))))
        top = max(0, min(height, int(math.floor(y1))))
        right = max(left, min(width, int(math.ceil(x2))))
        bottom = max(top, min(height, int(math.ceil(y2))))
        if right - left >= 8 and bottom - top >= 8:
            regions.append(sample.frame[top:bottom, left:right])
    qualities = [_image_quality(region) for region in regions]
    if not qualities:
        return _image_quality(sample.frame)
    count = float(len(qualities))
    return _ImageQuality(
        score=sum(item.score for item in qualities) / count,
        sharpness=sum(item.sharpness for item in qualities) / count,
        exposure=sum(item.exposure for item in qualities) / count,
        contrast=sum(item.contrast for item in qualities) / count,
        edge_detail=sum(item.edge_detail for item in qualities) / count,
    )

def _confidence(detected: dict[str, Any]) -> float:
    try:
        confidence = float(detected.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _box(detected: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = detected.get("box")
    if not isinstance(box, dict):
        return None
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _temporal_association_score(
    previous: dict[str, Any],
    detected: dict[str, Any],
) -> float | None:
    previous_box = _box(previous)
    detected_box = _box(detected)
    if previous_box is None or detected_box is None:
        return None
    px1, py1, px2, py2 = previous_box
    dx1, dy1, dx2, dy2 = detected_box
    intersection = max(0.0, min(px2, dx2) - max(px1, dx1)) * max(0.0, min(py2, dy2) - max(py1, dy1))
    union = (px2 - px1) * (py2 - py1) + (dx2 - dx1) * (dy2 - dy1) - intersection
    iou = intersection / union if union > 0 else 0.0
    if iou >= TEMPORAL_ASSOCIATION_MIN_IOU:
        return 2.0 + iou
    previous_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
    detected_center = ((dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0)
    center_distance = math.hypot(
        detected_center[0] - previous_center[0],
        detected_center[1] - previous_center[1],
    )
    scale = max(
        math.hypot(px2 - px1, py2 - py1),
        math.hypot(dx2 - dx1, dy2 - dy1),
        1.0,
    )
    distance_ratio = center_distance / scale
    if distance_ratio > TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO:
        return None
    return 1.0 - distance_ratio / TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO


_TEMPORAL_LABEL_FAMILIES = (
    frozenset({
        "car", "truck", "bus", "van", "motorcycle", "bicycle",
        "robot_lawnmower",
    }),
    frozenset({"person", "child"}),
    frozenset({"dog", "cat", "horse", "deer", "bird", "animal"}),
)


def _temporally_compatible_labels(previous: dict[str, Any], detected: dict[str, Any]) -> bool:
    left = str(previous.get("label") or "").strip().lower()
    right = str(detected.get("label") or "").strip().lower()
    if not left or not right or left == right:
        return True
    return any(left in family and right in family for family in _TEMPORAL_LABEL_FAMILIES)


def _eligible_detection(detected: dict[str, Any]) -> bool:
    return bool(detected.get("label") and detected.get("incident_eligible") is not False)


def _semantic_rescue_threshold(detected: dict[str, Any]) -> float:
    candidate = float(detected.get("temporal_candidate_threshold") or 0.0)
    standard = float(detected.get("confidence_threshold") or 1.0)
    return min(
        0.99,
        candidate + SEMANTIC_RESCUE_THRESHOLD_FRACTION * (standard - candidate),
    )


def _candidate_detection(detected: dict[str, Any]) -> bool:
    """Return whether a detector result can corroborate a temporal object track.

    Zone admission is intentionally not required here. An object that enters an
    incident zone and then crosses its boundary is still the same object; the
    completed track only needs one admitted observation to become eligible.
    """
    return bool(
        detected.get("label")
        and detected.get(
            "temporal_candidate_eligible",
            detected.get("confidence_eligible"),
        ) is not False
        and _box(detected) is not None
    )


def _temporal_motion_metrics(
    track: _TemporalDetectionEvidence,
    samples: list[_RecordedDetectionSample],
) -> tuple[float, float]:
    """Measure detector-box movement in resolution-independent frame units."""
    centers: list[tuple[float, float]] = []
    for sample_index, detected in sorted(track.observations.items()):
        box = _box(detected)
        if box is None or sample_index >= len(samples):
            continue
        height, width = samples[sample_index].frame.shape[:2]
        if width <= 0 or height <= 0:
            continue
        x1, y1, x2, y2 = box
        centers.append(((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height)))
    if len(centers) < 2:
        return 0.0, 0.0
    displacement = math.dist(centers[0], centers[-1])
    path = sum(math.dist(previous, current) for previous, current in zip(centers, centers[1:]))
    return displacement, path


def _normalized_box_metrics(
    detected: dict[str, Any],
    frame: Frame,
) -> tuple[float, float]:
    """Return edge clearance and area in resolution-independent units."""
    box = _box(detected)
    height, width = frame.shape[:2]
    if box is None or width <= 0 or height <= 0:
        return 0.0, 0.0
    x1, y1, x2, y2 = box
    clearance = min(
        x1 / width,
        y1 / height,
        (width - x2) / width,
        (height - y2) / height,
    )
    area = ((x2 - x1) * (y2 - y1)) / float(width * height)
    return (
        max(0.0, min(0.5, clearance)),
        max(0.0, min(1.0, area)),
    )


def _representative_needs_refinement(objects: list[dict[str, Any]]) -> bool:
    """Identify an avoidably weak cover frame after detection is confirmed.

    Refinement is deliberately limited to the active/new subject rather than a
    stationary corroborating object. This keeps a parked vehicle from making a
    clipped person look like an acceptable two-object representative frame.
    """
    primary = [
        item
        for item in objects
        if item.get("temporal_consensus") is True
        and item.get("snapshot_primary_subject") is True
    ]
    if not primary:
        return False
    return any(
        float(item.get("snapshot_edge_clearance_ratio") or 0.0)
        < REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO
        or float(item.get("snapshot_subject_area_ratio") or 0.0)
        < REPRESENTATIVE_MIN_SUBJECT_AREA_RATIO
        or float(item.get("snapshot_quality_score") or 0.0)
        < REPRESENTATIVE_MIN_QUALITY_SCORE
        for item in primary
    )


def _temporal_consensus(
    samples: list[_RecordedDetectionSample],
    minimum_confirmations: int,
    class_confirmations: dict[str, int] | None = None,
) -> tuple[_RecordedDetectionSample, list[dict[str, Any]]]:
    """Select one recorded frame and retain only repeatable object evidence."""
    evidence: list[_TemporalDetectionEvidence] = []
    assignments: dict[tuple[int, int], _TemporalDetectionEvidence] = {}
    for sample_index, sample in enumerate(samples):
        available = set(range(len(evidence)))
        candidates = [item for item in sample.objects if _candidate_detection(item)]
        pair_scores = sorted(
            (
                (score, evidence_index, object_index)
                for evidence_index in available
                for object_index, detected in enumerate(candidates)
                if _temporally_compatible_labels(evidence[evidence_index].latest, detected)
                and (score := _temporal_association_score(
                    evidence[evidence_index].latest,
                    detected,
                )) is not None
            ),
            reverse=True,
        )
        matched_objects: dict[int, int] = {}
        for _score, evidence_index, object_index in pair_scores:
            if evidence_index not in available or object_index in matched_objects:
                continue
            matched_objects[object_index] = evidence_index
            available.remove(evidence_index)
        for object_index, detected in enumerate(candidates):
            evidence_index = matched_objects.get(object_index)
            if evidence_index is None:
                track = _TemporalDetectionEvidence()
                evidence.append(track)
            else:
                track = evidence[evidence_index]
            track.add(sample_index, detected)
            assignments[(sample_index, id(detected))] = track

    default_required = max(1, min(5, int(minimum_confirmations)))
    normalized_class_confirmations = {
        str(label).strip().lower(): max(1, min(5, int(confirmations)))
        for label, confirmations in (class_confirmations or {}).items()
    }
    required_by_track = {
        id(track): normalized_class_confirmations.get(track.winning_label.lower(), default_required)
        for track in evidence
    }
    normally_confirmed_ids = {
        id(track)
        for track in evidence
        if len(track.winning_observations) >= required_by_track[id(track)]
        and any(_eligible_detection(item) for item in track.winning_observations)
    }
    rescue_threshold_by_track = {
        id(track): min(
            0.99,
            max(_semantic_rescue_threshold(item) for item in track.winning_observations),
        )
        for track in evidence
        if track.winning_observations
    }
    rescue_candidate_ids = {
        id(track)
        for track in evidence
        if id(track) not in normally_confirmed_ids
        and len(track.winning_observations) >= max(3, required_by_track[id(track)])
        and track.aggregate_confidence >= rescue_threshold_by_track[id(track)]
        and any(
            item.get("spatial_zone_eligible") is True
            for item in track.winning_observations
        )
    }
    confirmed_ids = normally_confirmed_ids | rescue_candidate_ids
    if confirmed_ids:
        confirmed_ids.update(
            id(track)
            for track in evidence
            if track.winning_label.strip().lower() == "face"
            and len(track.winning_observations) >= required_by_track[id(track)]
        )

    motion_by_track = {
        id(track): _temporal_motion_metrics(track, samples)
        for track in evidence
        if id(track) in confirmed_ids
    }
    primary_ids = {
        id(track)
        for track in evidence
        if id(track) in confirmed_ids
        and (
            min(track.observations, default=0) > 0
            or motion_by_track[id(track)][0]
            >= REPRESENTATIVE_DYNAMIC_DISPLACEMENT_RATIO
            or motion_by_track[id(track)][1] >= REPRESENTATIVE_DYNAMIC_PATH_RATIO
        )
    }
    if not primary_ids:
        primary_ids = set(confirmed_ids)

    quality_by_sample: dict[int, _ImageQuality] = {}

    def sample_score(
        item: tuple[int, _RecordedDetectionSample],
    ) -> tuple[int, int, float, float, int, int, float, float, float, float, float]:
        sample_index, sample = item
        visible = [
            track
            for track in evidence
            if id(track) in confirmed_ids
            and sample_index in track.observations
            and str(track.observations[sample_index].get("label") or "").strip() == track.winning_label
        ]
        admitted_visible = [
            track for track in visible
            if _eligible_detection(track.observations[sample_index])
        ]
        primary_visible = [track for track in visible if id(track) in primary_ids]
        primary_metrics = [
            _normalized_box_metrics(track.observations[sample_index], sample.frame)
            for track in primary_visible
        ]
        fully_framed_primary = sum(
            clearance >= REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO
            for clearance, _area in primary_metrics
        )
        best_primary_clearance = max(
            (clearance for clearance, _area in primary_metrics),
            default=0.0,
        )
        best_primary_area = max(
            (area for _clearance, area in primary_metrics),
            default=0.0,
        )
        quality = _sample_image_quality(sample, visible, sample_index)
        quality_by_sample[sample_index] = quality
        face_quality = max(
            (
                float(track.observations[sample_index].get("face_quality_score") or 0.0)
                for track in visible
                if track.winning_label.strip().lower() == "face"
            ),
            default=0.0,
        )
        raw_peak = max((_confidence(detected) for detected in sample.objects if _candidate_detection(detected)), default=0.0)
        return (
            int(bool(primary_visible)),
            fully_framed_primary,
            best_primary_area,
            best_primary_clearance,
            len(admitted_visible),
            len(visible),
            face_quality,
            sum(track.aggregate_confidence for track in visible),
            quality.score,
            raw_peak,
            -abs(sample.offset),
        )

    selected_index, selected = max(enumerate(samples), key=sample_score)
    selected_quality = quality_by_sample[selected_index]
    annotated: list[dict[str, Any]] = []
    for detected in selected.objects:
        if not _candidate_detection(detected):
            annotated.append(dict(detected))
            continue
        track = assignments.get((selected_index, id(detected)))
        if track is None:
            continue
        confirmed = id(track) in confirmed_ids
        label_confirmed_here = str(detected.get("label") or "").strip() == track.winning_label
        confirmed = confirmed and label_confirmed_here
        incident_confirmed = (
            id(track) in normally_confirmed_ids
            and label_confirmed_here
            and not bool(detected.get("auxiliary_detection"))
        )
        rescue_candidate = id(track) in rescue_candidate_ids
        zone_eligible = any(
            _eligible_detection(item) for item in track.winning_observations
        ) or bool(
            rescue_candidate
            and any(
                item.get("spatial_zone_eligible") is True
                for item in track.winning_observations
            )
        )
        edge_clearance, subject_area = _normalized_box_metrics(detected, selected.frame)
        enriched = {
            **detected,
            "label": track.winning_label,
            # Rescue candidates remain evidence here. Only the downstream
            # decision policy may promote them after causal motion and scene
            # context have been evaluated.
            "incident_eligible": incident_confirmed,
            "zone_eligible": zone_eligible,
            "temporal_eligible": incident_confirmed,
            "incident_ineligible_reasons": (
                [] if incident_confirmed else
                ["auxiliary_detection"] if detected.get("auxiliary_detection") else
                ["outside_incident_zone"] if not zone_eligible else
                ["pending_causal_confirmation"] if rescue_candidate else
                ["temporal_unconfirmed"]
            ),
            "temporal_consensus": confirmed,
            "temporal_low_confidence_confirmation": rescue_candidate,
            "temporal_rescue_candidate": rescue_candidate,
            "semantic_tier": (
                "standard" if incident_confirmed else
                "rescue_candidate" if rescue_candidate else
                "evidence"
            ),
            "semantic_candidate_threshold": round(
                max(
                    float(item.get("temporal_candidate_threshold") or 0.0)
                    for item in track.winning_observations
                ),
                4,
            ),
            "semantic_standard_threshold": round(
                max(
                    float(item.get("confidence_threshold") or 1.0)
                    for item in track.winning_observations
                ),
                4,
            ),
            "semantic_rescue_threshold": round(
                rescue_threshold_by_track.get(id(track), 0.0),
                4,
            ),
            "semantic_rescue_threshold_fraction": SEMANTIC_RESCUE_THRESHOLD_FRACTION,
            "semantic_median_confidence": round(track.aggregate_confidence, 4),
            "semantic_min_confidence": round(
                min((_confidence(item) for item in track.winning_observations), default=0.0),
                4,
            ),
            "semantic_max_confidence": round(track.peak_confidence, 4),
            "temporal_sample_offset_seconds": selected.offset,
            "temporal_requested_sample_offset_seconds": (
                selected.requested_offset
                if selected.requested_offset is not None
                else selected.offset
            ),
            "temporal_sample_timestamp_source": (
                "source_pts" if selected.exact_timestamp else "requested_offset"
            ),
            "temporal_observations": len(track.winning_observations),
            "temporal_track_observations": len(track.observations),
            "temporal_incident_observations": sum(
                1 for item in track.winning_observations if _eligible_detection(item)
            ),
            "temporal_required_observations": required_by_track[id(track)],
            "temporal_samples": len(samples),
            "temporal_peak_confidence": round(track.peak_confidence, 4),
            "temporal_label_votes": track.label_votes,
            "snapshot_quality_score": round(selected_quality.score, 4),
            "snapshot_sharpness_score": round(selected_quality.sharpness, 4),
            "snapshot_exposure_score": round(selected_quality.exposure, 4),
            "snapshot_contrast_score": round(selected_quality.contrast, 4),
            "snapshot_edge_detail_score": round(selected_quality.edge_detail, 4),
            "snapshot_primary_subject": id(track) in primary_ids,
            "snapshot_edge_clearance_ratio": round(edge_clearance, 5),
            "snapshot_subject_area_ratio": round(subject_area, 5),
        }
        observation_indices = sorted(track.observations)
        if observation_indices:
            peak_observation_index = max(
                (
                    index
                    for index, item in track.observations.items()
                    if str(item.get("label") or "").strip() == track.winning_label
                ),
                key=lambda index: _confidence(track.observations[index]),
            )
            enriched["temporal_peak_confidence_offset_seconds"] = samples[
                peak_observation_index
            ].offset
            first_observation_index = observation_indices[0]
            last_observation_index = observation_indices[-1]
            enriched["temporal_first_observation_offset_seconds"] = samples[
                first_observation_index
            ].offset
            enriched["temporal_last_observation_offset_seconds"] = samples[
                last_observation_index
            ].offset
            enriched["temporal_newly_appeared"] = first_observation_index > 0
            pretrigger_indices = [
                index
                for index in observation_indices
                if samples[index].offset < 0.0
            ]
            posttrigger_indices = [
                index
                for index in observation_indices
                if samples[index].offset >= 0.0
            ]
            pretrigger_sample_count = sum(sample.offset < 0.0 for sample in samples)
            same_label_seen_pretrigger = any(
                str(candidate.get("label") or "").strip() == track.winning_label
                for sample in samples
                if sample.offset < 0.0
                for candidate in sample.objects
                if _candidate_detection(candidate)
            )
            enriched["temporal_pretrigger_observations"] = len(pretrigger_indices)
            enriched["temporal_posttrigger_observations"] = len(posttrigger_indices)
            enriched["temporal_pretrigger_samples"] = pretrigger_sample_count
            enriched["temporal_same_label_seen_pretrigger"] = same_label_seen_pretrigger
            enriched["temporal_robust_new_appearance"] = bool(
                not pretrigger_indices
                and pretrigger_sample_count >= 2
                and len(posttrigger_indices) >= 2
                and not same_label_seen_pretrigger
            )
            first_zones = set(
                str(value)
                for value in (
                    track.observations[first_observation_index].get("spatial_zones")
                    or track.observations[first_observation_index].get("zones")
                    or []
                )
                if str(value)
            )
            later_zones = {
                str(value)
                for index in observation_indices[1:]
                for value in (
                    track.observations[index].get("spatial_zones")
                    or track.observations[index].get("zones")
                    or []
                )
                if str(value)
            }
            enriched["temporal_zone_entry"] = bool(later_zones - first_zones)
        displacement, path = motion_by_track.get(
            id(track),
            _temporal_motion_metrics(track, samples),
        )
        enriched["temporal_center_displacement_ratio"] = round(displacement, 5)
        enriched["temporal_center_path_ratio"] = round(path, 5)
        if confirmed:
            enriched["confidence"] = round(track.aggregate_confidence, 4)
        annotated.append(enriched)

    LOGGER.debug(
        "recorded object consensus retained %d/%d candidates across %d frames (minimum %d)",
        len(confirmed_ids),
        len(evidence),
        len(samples),
        default_required,
    )
    return selected, annotated


class MotionObjectDetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: Frame,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def detect_faces(self, frame: Frame) -> list[dict[str, Any]]:
        ...


class MotionRecordingProvider(Protocol):
    ffmpeg_path: str
    hardware_acceleration: str

    def recording_at(self, camera_id: str, epoch: float) -> dict[str, Any] | None:
        ...


LiveFrameProvider = Callable[[], Frame | None]
TimestampedLiveFrameProvider = Callable[[], TimestampedLiveFrame | tuple[Frame, float] | None]
StopRequested = Callable[[], bool]


class _EventRecordedSampler:
    """Reuse recording discovery and decoded samples within one event workflow."""

    def __init__(
        self,
        *,
        camera_id: str,
        recorder: MotionRecordingProvider,
        frame_reader: Callable[..., Frame | None],
        batch_frame_reader: Callable[
            ..., tuple[dict[float, _DecodedRecordedFrame], int]
        ] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.recorder = recorder
        self.frame_reader = frame_reader
        self.batch_frame_reader = batch_frame_reader
        self._rows: list[dict[str, Any]] = sorted(
            (dict(row) for row in (rows or [])),
            key=lambda row: float(row.get("start_epoch", 0.0)),
        )
        self._frames: dict[tuple[str, float], _DecodedRecordedFrame | None] = {}

    def recording_at(self, epoch: float) -> dict[str, Any] | None:
        # Match Recorder.recording_at(): at a shared segment boundary the
        # newest segment is preferred, avoiding an EOF decode from its predecessor.
        for row in reversed(self._rows):
            start = float(row.get("start_epoch", 0.0))
            end = row.get("end_epoch")
            if end is None:
                duration = row.get("duration_seconds")
                end = start + float(duration) if duration is not None else start
            if start <= epoch <= float(end):
                return row
        row = self.recorder.recording_at(self.camera_id, epoch)
        if row is not None and row.get("start_epoch") is not None:
            normalized = dict(row)
            if normalized.get("end_epoch") is None:
                duration = normalized.get("duration_seconds")
                if duration is not None:
                    normalized["end_epoch"] = float(normalized["start_epoch"]) + float(duration)
            self._rows.append(normalized)
            self._rows.sort(
                key=lambda item: float(item.get("start_epoch", 0.0))
            )
            return normalized
        return row

    def frame_at(
        self,
        path: Path,
        offset_seconds: float,
        *,
        deadline: float | None,
    ) -> _DecodedRecordedFrame | None:
        key = (str(path), round(max(0.0, offset_seconds), 3))
        if key not in self._frames:
            frame = self.frame_reader(
                path,
                offset_seconds,
                deadline=deadline,
            )
            if frame is not None:
                self._frames[key] = _DecodedRecordedFrame(
                    frame=frame,
                    actual_offset=max(0.0, offset_seconds),
                    exact_timestamp=False,
                )
            return self._frames.get(key)
        return self._frames[key]

    def frames_at(
        self,
        path: Path,
        offsets_seconds: list[float],
        *,
        deadline: float | None,
    ) -> tuple[dict[float, _DecodedRecordedFrame], int, int]:
        """Return cached/batched frames and bounded fallback counts.

        The tuple contains frames keyed by the caller's original offsets,
        batch process count, and single-frame fallback count.
        """
        requested = list(dict.fromkeys(max(0.0, value) for value in offsets_seconds))
        frames: dict[float, _DecodedRecordedFrame] = {}
        missing: list[float] = []
        for offset in requested:
            key = (str(path), round(offset, 3))
            cached = self._frames.get(key)
            if cached is not None:
                frames[offset] = cached
            else:
                missing.append(offset)

        batch_processes = 0
        if missing and self.batch_frame_reader is not None:
            decoded, batch_processes = self.batch_frame_reader(
                path, missing, deadline=deadline
            )
            decoded_by_key = {
                round(max(0.0, offset), 3): frame
                for offset, frame in decoded.items()
            }
            for offset in missing:
                normalized_offset = round(offset, 3)
                frame = decoded_by_key.get(normalized_offset)
                if frame is None:
                    continue
                key = (str(path), normalized_offset)
                self._frames[key] = frame
                frames[offset] = frame

        fallback_count = 0
        for offset in missing:
            if offset in frames:
                continue
            fallback_count += 1
            frame = self.frame_at(path, offset, deadline=deadline)
            if frame is not None:
                frames[offset] = frame
        return frames, batch_processes, fallback_count


class RecordedMotionObjectDetector:
    """Combines recorded event-time detections into repeatable object evidence."""

    def __init__(
        self,
        camera: CameraConfig,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
        live_frame_provider: LiveFrameProvider,
        timestamped_live_frame_provider: TimestampedLiveFrameProvider | None = None,
        stop_requested: StopRequested = lambda: False,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.recorder = recorder
        self.live_frame_provider = live_frame_provider
        self.timestamped_live_frame_provider = timestamped_live_frame_provider
        self.stop_requested = stop_requested

    def detect(self, event_at: datetime) -> RecordedDetectionResult:
        return self._detect(
            event_at,
            stages=RECORDED_EVENT_FRAME_STAGES,
            retry_seconds=RECORDED_EVENT_RETRY_SECONDS,
            allow_representative_refinement=True,
            refinement_pending=False,
        )

    def detect_initial(self, event_at: datetime) -> RecordedDetectionResult:
        """Run one strictly fresh live-frame check and always refine later.

        Finalized recordings intentionally lag the live edge by a segment. The
        initial security path must not wait for that boundary. A stale or
        unavailable live frame therefore produces a provisional no-frame result
        rather than cancelling the authoritative recorded refinement.
        """
        workflow_started = time.monotonic()
        timing = {
            "recording_wait_ms": 0.0,
            "frame_decode_ms": 0.0,
            "detector_request_ms": 0.0,
            "detection_enrichment_ms": 0.0,
            "temporal_confirmation_wait_ms": 0.0,
            "recording_batch_processes": 0.0,
            "recording_fallback_samples": 0.0,
            "recording_samples_requested": 0.0,
            "recording_samples_decoded": 0.0,
        }
        provider = self.timestamped_live_frame_provider
        sample = provider() if provider is not None else None
        if sample is None:
            return self._result(
                None,
                [{"status": "fast_frame_unavailable", "frame_source": "live_fast_path"}],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        if isinstance(sample, TimestampedLiveFrame):
            frame = sample.frame
            captured_at = float(sample.captured_at_epoch)
            sequence = int(sample.sequence)
            generation = int(sample.camera_generation)
            capture_generation = int(sample.capture_generation)
            source = str(sample.source or "live")
            provenance_valid = bool(
                source == "live"
                and sequence > 0
                and generation > 0
                and capture_generation > 0
                and math.isfinite(float(sample.captured_at_monotonic))
            )
        else:
            # Compatibility for external factories/tests. It remains fresh but
            # cannot claim capture-generation provenance.
            frame, captured_at = sample
            sequence = 0
            generation = 0
            capture_generation = 0
            source = "live"
            provenance_valid = True
        if not math.isfinite(float(captured_at)) or not provenance_valid:
            return self._result(
                None,
                [{
                    "status": "fast_frame_invalid_provenance",
                    "frame_source": "live_fast_path",
                    "frame_sequence": sequence,
                    "camera_generation": generation,
                    "capture_generation": capture_generation,
                }],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        frame_age = time.time() - float(captured_at)
        if (
            frame_age > FAST_LIVE_FRAME_MAX_AGE_SECONDS
            or frame_age < -FAST_LIVE_FRAME_FUTURE_TOLERANCE_SECONDS
        ):
            return self._result(
                None,
                [{
                    "status": "fast_frame_stale",
                    "frame_source": "live_fast_path",
                    "frame_age_ms": round(frame_age * 1000.0, 3),
                }],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        objects = self._detect_objects(
            frame,
            timing=timing,
            enrich_faces=False,
            workload="initial",
        )
        for detected in objects:
            if isinstance(detected, dict):
                detected.update({
                    "frame_source": "live_fast_path",
                    "provisional_detection": True,
                    "frame_captured_at_epoch": round(float(captured_at), 6),
                    "frame_age_ms": round(max(0.0, frame_age) * 1000.0, 3),
                    "frame_sequence": sequence,
                    "camera_generation": generation,
                    "capture_generation": capture_generation,
                })
        return self._result(
            frame,
            objects,
            "",
            timing,
            workflow_started,
            refinement_pending=True,
        )

    def _detect(
        self,
        event_at: datetime,
        *,
        stages: tuple[tuple[float, ...], ...],
        retry_seconds: float,
        allow_representative_refinement: bool,
        refinement_pending: bool,
    ) -> RecordedDetectionResult:
        workflow_started = time.monotonic()
        timing = {
            "recording_wait_ms": 0.0,
            "frame_decode_ms": 0.0,
            "detector_request_ms": 0.0,
            "detection_enrichment_ms": 0.0,
            "temporal_confirmation_wait_ms": 0.0,
            "recording_batch_processes": 0.0,
            "recording_fallback_samples": 0.0,
            "recording_samples_requested": 0.0,
            "recording_samples_decoded": 0.0,
        }
        event_epoch = event_at.timestamp()
        initial_offsets = stages[0]
        newest_needed = event_epoch + max(initial_offsets) + RECORDED_EVENT_SETTLE_SECONDS
        wait_seconds = max(0.0, newest_needed - time.time())
        if wait_seconds > 0:
            slept = min(wait_seconds, 3.0)
            time.sleep(slept)
            timing["temporal_confirmation_wait_ms"] += slept * 1000.0

        deadline = time.monotonic() + max(0.0, retry_seconds)
        refinement_deadline: float | None = None
        samples_by_offset: dict[float, _RecordedDetectionSample] = {}
        default_required = max(
            1,
            min(
                len(RECORDED_EVENT_FRAME_OFFSETS),
                int(getattr(self.detector.config, "event_confirmation_frames", 2)),
            ),
        )
        class_confirmations = {
            str(label).strip().lower(): max(
                1,
                min(len(RECORDED_EVENT_FRAME_OFFSETS), int(confirmations)),
            )
            for label, confirmations in dict(
                getattr(self.detector.config, "event_class_confirmation_frames", {}) or {}
            ).items()
        }
        prefetched_rows: list[dict[str, Any]] = []
        rows_between = getattr(self.recorder, "recording_rows_between", None)
        if callable(rows_between):
            lookup_started = time.monotonic()
            try:
                prefetched_rows = list(rows_between(
                    self.camera.id,
                    event_epoch + min(offset for stage in stages for offset in stage) - 1.0,
                    event_epoch + max(offset for stage in stages for offset in stage) + 1.0,
                    "main",
                    discover_missing=False,
                ))
            except Exception:
                LOGGER.debug(
                    "recording range prefetch unavailable for %s",
                    self.camera.id,
                    exc_info=True,
                )
            timing["recording_wait_ms"] += (
                time.monotonic() - lookup_started
            ) * 1000.0
        sampler = _EventRecordedSampler(
            camera_id=self.camera.id,
            recorder=self.recorder,
            frame_reader=self._read_recorded_frame,
            batch_frame_reader=self._read_recorded_frames,
            rows=prefetched_rows,
        )

        for stage_index, stage_offsets in enumerate(stages):
            while True:
                if self.stop_requested():
                    return self._result(
                        None,
                        [{"status": "cancelled"}],
                        "",
                        timing,
                        workflow_started,
                        refinement_pending=False,
                    )
                stage_deadline = min(
                    deadline,
                    refinement_deadline
                    if refinement_deadline is not None
                    else deadline,
                )
                recording_missing = False
                future_sample = False
                pending_by_path: dict[Path, list[tuple[float, float, dict[str, Any]]]] = {}
                for sample_offset in stage_offsets:
                    if sample_offset in samples_by_offset:
                        continue
                    if time.monotonic() >= stage_deadline:
                        break
                    target_epoch = event_epoch + sample_offset
                    if target_epoch + RECORDED_EVENT_SETTLE_SECONDS > time.time():
                        future_sample = True
                        continue
                    lookup_started = time.monotonic()
                    row = sampler.recording_at(target_epoch)
                    timing["recording_wait_ms"] += (
                        time.monotonic() - lookup_started
                    ) * 1000.0
                    if row is None:
                        recording_missing = True
                        continue
                    start_epoch = row.get("start_epoch")
                    if start_epoch is None:
                        continue
                    frame_offset = max(0.0, target_epoch - float(start_epoch))
                    pending_by_path.setdefault(Path(str(row["path"])), []).append(
                        (sample_offset, frame_offset, row)
                    )

                for path, pending in pending_by_path.items():
                    decode_started = time.monotonic()
                    frames, batch_processes, fallback_count = sampler.frames_at(
                        path,
                        [frame_offset for _, frame_offset, _ in pending],
                        deadline=stage_deadline,
                    )
                    timing["frame_decode_ms"] += (
                        time.monotonic() - decode_started
                    ) * 1000.0
                    timing["recording_batch_processes"] += batch_processes
                    timing["recording_fallback_samples"] += fallback_count
                    timing["recording_samples_requested"] += len(pending)
                    timing["recording_samples_decoded"] += len(frames)
                    for sample_offset, frame_offset, row in pending:
                        decoded = frames.get(frame_offset)
                        if decoded is None:
                            continue
                        objects = self._detect_objects(
                            decoded.frame,
                            timing=timing,
                            workload="refinement",
                        )
                        row_start = float(row.get("start_epoch") or 0.0)
                        actual_offset = (
                            row_start + decoded.actual_offset - event_epoch
                        )
                        samples_by_offset[sample_offset] = _RecordedDetectionSample(
                            offset=actual_offset,
                            requested_offset=sample_offset,
                            exact_timestamp=decoded.exact_timestamp,
                            frame=decoded.frame,
                            objects=objects,
                            recording_path=str(row["path"]),
                        )

                samples = [
                    samples_by_offset[offset]
                    for offset in RECORDED_EVENT_FRAME_OFFSETS
                    if offset in samples_by_offset
                ]
                if samples:
                    selected, objects = _temporal_consensus(
                        samples,
                        default_required,
                        class_confirmations,
                    )
                    if any(item.get("temporal_consensus") is True for item in objects):
                        # Detection is already proven. If its cover image clips,
                        # obscures, or barely shows the active/new subject, spend
                        # one existing +4 second stage looking for a better frame.
                        # This is bounded and does not wait through the full
                        # tracking lifecycle or add a second inference pipeline.
                        representative_needs_refinement = bool(
                            stage_index == 0
                            and _representative_needs_refinement(objects)
                        )
                        if allow_representative_refinement and representative_needs_refinement:
                            refinement_deadline = min(
                                deadline,
                                time.monotonic()
                                + RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS,
                            )
                            LOGGER.info(
                                "recorded object representative refinement requested for %s",
                                self.camera.id,
                            )
                            break
                        if stage_index:
                            LOGGER.info(
                                "recorded object selection completed for %s at stage +%.1fs",
                                self.camera.id,
                                max(stage_offsets),
                            )
                        return self._result(
                            selected.frame,
                            objects,
                            selected.recording_path,
                            timing,
                            workflow_started,
                            refinement_pending=bool(
                                refinement_pending and representative_needs_refinement
                            ),
                            face_candidates=self._face_candidates(samples),
                        )
                if all(offset in samples_by_offset for offset in stage_offsets):
                    break
                remaining = deadline - time.monotonic()
                if refinement_deadline is not None:
                    remaining = min(
                        remaining,
                        refinement_deadline - time.monotonic(),
                    )
                if remaining <= 0:
                    break
                slept = min(RECORDED_EVENT_RETRY_INTERVAL_SECONDS, remaining)
                time.sleep(slept)
                wait_key = (
                    "recording_wait_ms"
                    if recording_missing and not future_sample
                    else "temporal_confirmation_wait_ms"
                )
                timing[wait_key] += slept * 1000.0
            if (
                refinement_deadline is not None
                and time.monotonic() >= refinement_deadline
            ):
                break
            if time.monotonic() >= deadline:
                break

        if samples:
            selected, objects = _temporal_consensus(
                samples,
                default_required,
                class_confirmations,
            )
            return self._result(
                selected.frame,
                objects,
                selected.recording_path,
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
                face_candidates=self._face_candidates(samples),
            )

        fallback = self.live_frame_provider()
        if fallback is None:
            return self._result(
                None,
                [{"status": "no_recorded_frame"}],
                "",
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
                face_candidates=(),
            )
        objects = self._detect_objects(
            fallback,
            timing=timing,
            workload="refinement",
        )
        if objects:
            for detected in objects:
                detected["frame_source"] = "live_fallback"
                detected["recording_status"] = "no_recorded_frame"
            return self._result(
                fallback,
                objects,
                "",
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
            )
        return self._result(
            fallback,
            [{"status": "no_recorded_frame", "frame_source": "live_fallback"}],
            "",
            timing,
            workflow_started,
            refinement_pending=refinement_pending,
        )

    @staticmethod
    def _face_candidates(
        samples: list[_RecordedDetectionSample],
    ) -> tuple[FaceCandidate, ...]:
        return collect_face_candidates(
            FaceCandidateSample(
                offset_seconds=sample.offset,
                frame=sample.frame,
                objects=tuple(sample.objects),
            )
            for sample in samples
        )

    @staticmethod
    def _result(
        frame: Frame | None,
        objects: list[dict[str, Any]],
        recording_path: str,
        timing: dict[str, float],
        workflow_started: float,
        *,
        refinement_pending: bool,
        face_candidates: tuple[FaceCandidate, ...] = (),
    ) -> RecordedDetectionResult:
        normalized = {key: round(max(0.0, value), 3) for key, value in timing.items()}
        normalized["workflow_ms"] = round(
            max(0.0, (time.monotonic() - workflow_started) * 1000.0),
            3,
        )
        return RecordedDetectionResult(
            frame=frame,
            objects=objects,
            recording_path=recording_path,
            timings_ms=normalized,
            refinement_pending=refinement_pending,
            face_candidates=face_candidates,
        )

    def _detect_objects(
        self,
        frame: Frame,
        *,
        timing: dict[str, float] | None = None,
        enrich_faces: bool = True,
        workload: str = "refinement",
    ) -> list[dict[str, Any]]:
        enrichment_started = time.monotonic()
        configured_threshold = float(self.detector.config.confidence_threshold)
        class_thresholds = dict(
            getattr(
                self.detector.config,
                "event_class_confidence_thresholds",
                {},
            )
            or {}
        )
        threshold = detection_threshold(
            self.camera,
            configured_threshold,
            class_thresholds,
        )
        candidate_threshold = min(
            threshold,
            float(getattr(
                self.detector.config,
                "event_candidate_confidence_threshold",
                threshold,
            )),
        )
        detector_started = time.monotonic()
        detector_method = getattr(
            self.detector,
            "detect_initial" if workload == "initial" else "detect_refinement",
            self.detector.detect,
        )
        objects = detector_method(
            frame,
            confidence_threshold=candidate_threshold,
        )
        detector_ms = (time.monotonic() - detector_started) * 1000.0
        if timing is not None:
            timing["detector_request_ms"] += detector_ms
        frame_height, frame_width = frame.shape[:2]
        detect_faces = getattr(self.detector, "detect_faces", None)
        if enrich_faces and callable(detect_faces):
            dedicated_faces = self._detect_faces_in_people(
                frame,
                objects,
                detect_faces,
            )
            objects = self._merge_dedicated_faces(objects, dedicated_faces)
        for detected in objects:
            if isinstance(detected, dict) and detected.get("label"):
                detected["detection_frame_width"] = int(frame_width)
                detected["detection_frame_height"] = int(frame_height)
                if str(detected.get("label") or "").strip().lower() == "face":
                    box = _box(detected)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        left = max(0, min(frame_width, int(math.floor(x1))))
                        top = max(0, min(frame_height, int(math.floor(y1))))
                        right = max(left, min(frame_width, int(math.ceil(x2))))
                        bottom = max(top, min(frame_height, int(math.ceil(y2))))
                        quality = _image_quality(frame[top:bottom, left:right])
                        detected["face_quality_score"] = round(quality.score, 4)
                        detected["face_sharpness_score"] = round(quality.sharpness, 4)
                        detected["face_exposure_score"] = round(quality.exposure, 4)
        apply_detection_zones(
            self.camera,
            objects,
            int(frame_width),
            int(frame_height),
            configured_threshold,
            bool(getattr(self.detector.config, "require_incident_zone", True)),
            class_thresholds,
        )
        for detected in objects:
            if isinstance(detected, dict):
                detected["temporal_candidate_threshold"] = candidate_threshold
                detected["temporal_candidate_eligible"] = bool(
                    detected.get("label")
                    and _confidence(detected) >= candidate_threshold
                )
            if isinstance(detected, dict) and str(
                detected.get("label") or ""
            ).strip().lower() == "face":
                if detected.get("detection_source") == "dedicated_face":
                    detected["confidence_eligible"] = True
                detected["incident_eligible"] = False
                detected["auxiliary_detection"] = True
        if timing is not None:
            timing["detection_enrichment_ms"] += max(
                0.0,
                (time.monotonic() - enrichment_started) * 1000.0 - detector_ms,
            )
        return objects

    @staticmethod
    def _detect_faces_in_people(
        frame: Frame,
        objects: list[dict[str, Any]],
        detect_faces: Callable[[Frame], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Run the 300px face model where CCTV faces retain useful resolution."""
        frame_height, frame_width = frame.shape[:2]
        detected_faces: list[dict[str, Any]] = []
        people = [
            item
            for item in objects
            if isinstance(item, dict)
            and str(item.get("label") or "").strip().lower() == "person"
            and _box(item) is not None
        ]
        for person in sorted(people, key=_confidence, reverse=True)[:8]:
            box = _box(person)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            person_width, person_height = x2 - x1, y2 - y1
            left = max(0, min(frame_width, int(math.floor(x1 - person_width * 0.08))))
            right = max(left, min(frame_width, int(math.ceil(x2 + person_width * 0.08))))
            top = max(0, min(frame_height, int(math.floor(y1 - person_height * 0.05))))
            bottom = max(top, min(frame_height, int(math.ceil(y1 + person_height * 0.68))))
            if right - left < 24 or bottom - top < 24:
                continue
            for detected in detect_faces(frame[top:bottom, left:right]):
                if not isinstance(detected, dict):
                    continue
                face_box = _box(detected)
                if face_box is None:
                    continue
                fx1, fy1, fx2, fy2 = face_box
                detected_faces.append(
                    {
                        **detected,
                        "box": {
                            "x1": fx1 + left,
                            "y1": fy1 + top,
                            "x2": fx2 + left,
                            "y2": fy2 + top,
                        },
                        "parent_person_box": person.get("box"),
                    }
                )
        return detected_faces

    @staticmethod
    def _merge_dedicated_faces(
        objects: list[dict[str, Any]],
        dedicated_faces: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prefer dedicated face boxes while preserving unrelated detector output."""
        merged = [
            dict(item)
            for item in objects
            if isinstance(item, dict)
            and str(item.get("label") or "").strip().lower() != "face"
        ]
        candidates = [
            dict(item)
            for item in dedicated_faces
            if isinstance(item, dict) and _box(item) is not None
        ]
        valid_dedicated: list[dict[str, Any]] = []
        for item in sorted(candidates, key=_confidence, reverse=True):
            item_box = _box(item)
            if item_box is None:
                continue
            if any(
                RecordedMotionObjectDetector._box_iou(item_box, _box(existing)) >= 0.4
                for existing in valid_dedicated
            ):
                continue
            valid_dedicated.append(item)
        merged.extend(valid_dedicated)
        for item in objects:
            if (
                not isinstance(item, dict)
                or str(item.get("label") or "").strip().lower() != "face"
            ):
                continue
            generic_box = _box(item)
            if generic_box is None:
                continue
            overlaps = [
                RecordedMotionObjectDetector._box_iou(generic_box, _box(face))
                for face in valid_dedicated
                if _box(face) is not None
            ]
            if max(overlaps, default=0.0) < 0.25:
                merged.append({**item, "detection_source": "object_detector"})
        return merged

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float] | None,
    ) -> float:
        if second is None:
            return 0.0
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
        return intersection / union if union > 0 else 0.0

    def _read_recorded_frame(
        self,
        path: Path,
        offset_seconds: float,
        *,
        deadline: float | None = None,
    ) -> Frame | None:
        if not path.exists():
            return None
        attempts = [0.0, -0.25, 0.25, -0.75, 0.75]
        last_error = ""
        hw_input_args, hw_filter_args = recorded_frame_hw_args(self.recorder.hardware_acceleration)
        decode_plans = [("hardware", hw_input_args, hw_filter_args)] if hw_input_args else []
        decode_plans.append(("cpu", [], []))
        for nudge in attempts:
            sample_at = max(0.0, offset_seconds + nudge)
            for backend, input_args, filter_args in decode_plans:
                timeout = 8.0
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        last_error = "recorded-frame deadline expired"
                        break
                    timeout = min(timeout, remaining)
                command = [
                    self.recorder.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-fflags",
                    "+discardcorrupt",
                    "-err_detect",
                    "ignore_err",
                    *input_args,
                    "-ss",
                    f"{sample_at:.3f}",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-frames:v",
                    "1",
                    *filter_args,
                    "-pix_fmt",
                    "bgr24",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "bmp",
                    "pipe:1",
                ]
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"{backend} timed out"
                    continue
                if result.returncode != 0 or not result.stdout:
                    detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[0:2]
                    last_error = f"{backend}: {' '.join(detail)[:180]}"
                    continue
                array = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                last_error = f"{backend}: lossless frame decode returned no frame"
            if deadline is not None and time.monotonic() >= deadline:
                break
        LOGGER.debug(
            "skipped unreadable recording sample for %s at %.2fs: %s%s",
            self.camera.id,
            offset_seconds,
            path,
            f" ({last_error})" if last_error else "",
        )
        return None

    def _read_recorded_frames(
        self,
        path: Path,
        offsets_seconds: list[float],
        *,
        deadline: float | None = None,
    ) -> tuple[dict[float, _DecodedRecordedFrame], int]:
        """Decode several ordered samples from one segment in one process.

        The select expression emits exactly the first decoded frame at or
        after each requested segment-relative timestamp. Missing outputs are
        intentionally left to the established nudged single-frame fallback.
        """
        if not path.exists():
            return {}, 0
        offsets = sorted(dict.fromkeys(round(max(0.0, value), 3) for value in offsets_seconds))
        if not offsets:
            return {}, 0

        expression = "0"
        for index, offset in reversed(list(enumerate(offsets))):
            expression = (
                f"if(eq(selected_n\\,{index})\\,gte(t\\,{offset:.3f})\\,{expression})"
            )
        hw_input_args, hw_filter_args = recorded_frame_hw_args(
            self.recorder.hardware_acceleration
        )
        decode_plans = [("hardware", hw_input_args, hw_filter_args)] if hw_input_args else []
        decode_plans.append(("cpu", [], []))
        last_error = ""
        process_count = 0
        for backend, input_args, filter_args in decode_plans:
            timeout = 8.0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            filter_prefix = ""
            if filter_args:
                try:
                    filter_prefix = f"{filter_args[filter_args.index('-vf') + 1]},"
                except (ValueError, IndexError):
                    filter_prefix = ""
            command = [
                self.recorder.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "info",
                "-fflags",
                "+discardcorrupt",
                "-err_detect",
                "ignore_err",
                *input_args,
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"{filter_prefix}select={expression},showinfo@event_sample",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(len(offsets)),
                "-pix_fmt",
                "bgr24",
                "-f",
                "image2pipe",
                "-vcodec",
                "bmp",
                "pipe:1",
            ]
            try:
                process_count += 1
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_error = f"{backend} timed out"
                continue
            if result.returncode != 0 or not result.stdout:
                detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[0:2]
                last_error = f"{backend}: {' '.join(detail)[:180]}"
                continue
            encoded_frames = self._split_bmp_stream(result.stdout)
            actual_offsets = [
                float(match.group(1))
                for match in re.finditer(
                    rb"showinfo@event_sample[^\r\n]*\bpts_time:([-+0-9.eE]+)",
                    result.stderr,
                )
            ]
            decoded: dict[float, _DecodedRecordedFrame] = {}
            for offset, encoded, actual_offset in zip(
                offsets,
                encoded_frames,
                actual_offsets,
                strict=False,
            ):
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    decoded[offset] = _DecodedRecordedFrame(
                        frame=frame,
                        actual_offset=actual_offset,
                        exact_timestamp=True,
                    )
            if decoded:
                return decoded, process_count
            last_error = f"{backend}: batch frame decode returned no frames"
        LOGGER.debug(
            "recorded batch decode failed for %s camera=%s samples=%d%s",
            path,
            self.camera.id,
            len(offsets),
            f" ({last_error})" if last_error else "",
        )
        return {}, process_count

    @staticmethod
    def _split_bmp_stream(payload: bytes) -> list[bytes]:
        """Split concatenated image2pipe BMP records using their file headers."""
        frames: list[bytes] = []
        cursor = 0
        payload_length = len(payload)
        while cursor + 6 <= payload_length:
            if payload[cursor:cursor + 2] != b"BM":
                break
            frame_size = int.from_bytes(payload[cursor + 2:cursor + 6], "little")
            if frame_size < 54 or cursor + frame_size > payload_length:
                break
            frames.append(payload[cursor:cursor + frame_size])
            cursor += frame_size
        return frames

class RecordedMotionObjectDetectorFactory:
    def __init__(
        self,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
    ) -> None:
        self.detector = detector
        self.recorder = recorder

    def create(
        self,
        camera: CameraConfig,
        live_frame_provider: LiveFrameProvider,
        timestamped_live_frame_provider: TimestampedLiveFrameProvider | None = None,
        stop_requested: StopRequested = lambda: False,
    ) -> RecordedMotionObjectDetector:
        return RecordedMotionObjectDetector(
            camera=camera,
            detector=self.detector,
            recorder=self.recorder,
            live_frame_provider=live_frame_provider,
            timestamped_live_frame_provider=timestamped_live_frame_provider,
            stop_requested=stop_requested,
        )
