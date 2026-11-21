"""Association, confirmation, and representative frame selection for recorded motion."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..object_motion import ObjectMotionEstimate, estimate_object_motion
from ..visual_quality import VisualQuality, image_quality
from .context import Frame


# Preserve the detector's log category for evidence diagnostics.
LOGGER = logging.getLogger(f"{__package__}.object_detection")
TEMPORAL_ASSOCIATION_MIN_IOU = 0.05
TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO = 2.5
REPRESENTATIVE_DYNAMIC_DISPLACEMENT_RATIO = 0.01
REPRESENTATIVE_DYNAMIC_PATH_RATIO = 0.025
REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO = 0.01
REPRESENTATIVE_MIN_SUBJECT_AREA_RATIO = 0.0025
REPRESENTATIVE_MIN_QUALITY_SCORE = 0.25
SEMANTIC_RESCUE_THRESHOLD_FRACTION = 0.5


@dataclass
class _RecordedDetectionSample:
    offset: float
    frame: Frame | None
    objects: list[dict[str, Any]]
    recording_path: str
    requested_offset: float | None = None
    exact_timestamp: bool = False


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


def _track_image_qualities(
    sample: _RecordedDetectionSample,
    visible: list[_TemporalDetectionEvidence],
    sample_index: int,
) -> dict[int, _ImageQuality]:
    """Crop quality of each visible track's box, keyed by ``id(track)``."""
    qualities: dict[int, _ImageQuality] = {}
    if sample.frame is None:
        return qualities
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
            qualities[id(track)] = _image_quality(sample.frame[top:bottom, left:right])
    return qualities


def _combined_image_quality(
    sample: _RecordedDetectionSample,
    track_qualities: dict[int, _ImageQuality],
) -> _ImageQuality:
    """Prefer object-crop quality so timestamps and static scenery cannot win."""
    qualities = list(track_qualities.values())
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


def _meets_configured_confidence_floor(detected: dict[str, Any]) -> bool:
    threshold = detected.get("confidence_threshold")
    if threshold is None:
        return detected.get("confidence_eligible") is not False
    try:
        confidence = float(detected.get("confidence") or 0.0)
        required = float(threshold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(confidence) and math.isfinite(required) and confidence >= required


def _is_confirming_detection(detected: dict[str, Any]) -> bool:
    """Return whether one observation can satisfy a frame confirmation.

    The lower temporal candidate threshold only associates weak detections into
    a track.  Each configured confirmation must independently meet the normal
    class confidence floor; otherwise one strong frame plus weak candidates
    could create an incident.
    """
    return bool(
        detected.get("label")
        and not detected.get("auxiliary_detection")
        and _meets_configured_confidence_floor(detected)
    )


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
) -> ObjectMotionEstimate:
    """Measure shared motion evidence in normalized frame units."""
    centers: list[tuple[float, float, float]] = []
    for sample_index, detected in sorted(track.observations.items()):
        box = _box(detected)
        if box is None or sample_index >= len(samples):
            continue
        frame = samples[sample_index].frame
        if frame is None:
            continue
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            continue
        x1, y1, x2, y2 = box
        centers.append((
            samples[sample_index].offset,
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
        ))
    return estimate_object_motion(centers)


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


def _subject_view_score(area_ratio: float, quality_score: float) -> float:
    """Rank a subject view by resolved detail rather than box size alone.

    Linear subject size times crop quality: a blurred view must be much larger
    than a sharp one to win, while equally sharp views still order by size.
    """
    return math.sqrt(max(0.0, area_ratio)) * max(0.0, min(1.0, quality_score))


def _unconfirmed_cover_track_ids(
    evidence: list[_TemporalDetectionEvidence],
) -> set[int]:
    """Tracks that may anchor the cover when no track met its confirmation count.

    Scene establishment can accept a zone-admitted subject that was sampled too
    few times to confirm. Presentation only: these ids never change admission.
    The best-supported tracks win so one stray detection cannot outrank a
    subject seen in several samples.
    """
    support = {
        id(track): sum(
            _eligible_detection(item) and _is_confirming_detection(item)
            for item in track.winning_observations
        )
        for track in evidence
    }
    best = max(support.values(), default=0)
    if best <= 0:
        return set()
    return {track_id for track_id, count in support.items() if count == best}


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
    if any(item.get("snapshot_visible") is False for item in primary):
        return True
    return any(
        float(item.get("snapshot_edge_clearance_ratio") or 0.0)
        < REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO
        or float(item.get("snapshot_subject_area_ratio") or 0.0)
        < REPRESENTATIVE_MIN_SUBJECT_AREA_RATIO
        or float(item.get("snapshot_quality_score") or 0.0)
        < REPRESENTATIVE_MIN_QUALITY_SCORE
        for item in primary
    )


def _collect_temporal_evidence(
    samples: list[_RecordedDetectionSample],
) -> list[_TemporalDetectionEvidence]:
    """Associate candidate detections across samples into temporal tracks."""
    evidence: list[_TemporalDetectionEvidence] = []
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
    return evidence


def _refinement_stage_complete(
    samples: list[_RecordedDetectionSample],
    minimum_confirmations: int,
    class_confirmations: dict[str, int] | None = None,
) -> bool:
    """Whether every associated track is closed and at least one is confirmed.

    Uses full-sample track association, not the selected-frame annotation list,
    so a newly appeared subject on a later offset remains visible to early-exit.
    """
    evidence = _collect_temporal_evidence(samples)
    if not evidence:
        return False
    default_required = max(1, min(5, int(minimum_confirmations)))
    normalized_class_confirmations = {
        str(label).strip().lower(): max(1, min(5, int(confirmations)))
        for label, confirmations in (class_confirmations or {}).items()
    }
    required_by_track = {
        id(track): normalized_class_confirmations.get(
            track.winning_label.lower(),
            default_required,
        )
        for track in evidence
    }
    normally_confirmed_ids = {
        id(track)
        for track in evidence
        if len(track.winning_observations) >= required_by_track[id(track)]
        and sum(
            _is_confirming_detection(item)
            for item in track.winning_observations
        ) >= required_by_track[id(track)]
        and any(_eligible_detection(item) for item in track.winning_observations)
    }
    if not normally_confirmed_ids:
        return False
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
    return all(id(track) in confirmed_ids for track in evidence)


def _refinement_early_exit_ready(
    *,
    stage_index: int,
    stage_offsets: tuple[float, ...],
    samples_by_offset: dict[float, _RecordedDetectionSample],
    samples: list[_RecordedDetectionSample],
    minimum_confirmations: int,
    class_confirmations: dict[str, int] | None = None,
) -> bool:
    """Allow skipping remaining stage offsets only after the core window is seen.

    Stage 0 must still observe the near-event neighborhood (|offset| <= 0.5)
    before early-exit, so a pre-trigger confirmation cannot hide a subject that
    enters at event time. Delayed discovery stages may stop as soon as every
    observed track is confirmed.
    """
    if not _refinement_stage_complete(
        samples,
        minimum_confirmations,
        class_confirmations,
    ):
        return False
    if stage_index > 0:
        return True
    core_offsets = tuple(
        offset for offset in stage_offsets if abs(float(offset)) <= 0.5
    )
    if not core_offsets:
        return True
    return all(offset in samples_by_offset for offset in core_offsets)


def _temporal_trigger_evidence(track, samples) -> dict[str, Any]:
    """Use the same arrival and zone-entry evidence for scenes and alerts."""
    evidence = {}
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
        evidence["temporal_peak_confidence_offset_seconds"] = samples[
            peak_observation_index
        ].offset
        first_observation_index = observation_indices[0]
        last_observation_index = observation_indices[-1]
        evidence["temporal_first_observation_offset_seconds"] = samples[
            first_observation_index
        ].offset
        evidence["temporal_last_observation_offset_seconds"] = samples[
            last_observation_index
        ].offset
        evidence["temporal_newly_appeared"] = first_observation_index > 0
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
        evidence["temporal_pretrigger_observations"] = len(pretrigger_indices)
        evidence["temporal_posttrigger_observations"] = len(posttrigger_indices)
        evidence["temporal_pretrigger_samples"] = pretrigger_sample_count
        evidence["temporal_same_label_seen_pretrigger"] = same_label_seen_pretrigger
        evidence["temporal_robust_new_appearance"] = bool(
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
        evidence["temporal_zone_entry"] = bool(later_zones - first_zones)
    return evidence


def _temporal_consensus(
    samples: list[_RecordedDetectionSample],
    minimum_confirmations: int,
    class_confirmations: dict[str, int] | None = None,
) -> tuple[_RecordedDetectionSample, list[dict[str, Any]]]:
    """Select one recorded frame and retain only repeatable object evidence."""
    evidence = _collect_temporal_evidence(samples)
    assignments: dict[tuple[int, int], _TemporalDetectionEvidence] = {}
    for sample_index, sample in enumerate(samples):
        for detected in sample.objects:
            if not _candidate_detection(detected):
                continue
            for track in evidence:
                if (
                    sample_index in track.observations
                    and track.observations[sample_index] is detected
                ):
                    assignments[(sample_index, id(detected))] = track
                    break

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
        and sum(
            _is_confirming_detection(item)
            for item in track.winning_observations
        ) >= required_by_track[id(track)]
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
            or motion_by_track[id(track)].displacement_ratio
            >= REPRESENTATIVE_DYNAMIC_DISPLACEMENT_RATIO
            or motion_by_track[id(track)].excursion_ratio >= REPRESENTATIVE_DYNAMIC_PATH_RATIO
        )
    }
    if not primary_ids:
        primary_ids = set(confirmed_ids)

    # Without a confirmed track every object-based key below would tie and
    # whole-frame quality would pick the cover, which can be an empty scene.
    cover_ids = set(confirmed_ids)
    cover_primary_ids = set(primary_ids)
    if not cover_ids:
        cover_ids = _unconfirmed_cover_track_ids(evidence)
        cover_primary_ids = set(cover_ids)

    quality_by_sample: dict[int, _ImageQuality] = {}

    def sample_score(
        item: tuple[int, _RecordedDetectionSample],
    ) -> tuple[int, int, int, float, float, float, int, float, float, float, float, float]:
        sample_index, sample = item
        if sample.frame is None:
            raise ValueError("recorded consensus samples must retain frames")
        visible = [
            track
            for track in evidence
            if id(track) in cover_ids
            and sample_index in track.observations
            and str(track.observations[sample_index].get("label") or "").strip() == track.winning_label
        ]
        admitted_visible = [
            track for track in visible
            if _eligible_detection(track.observations[sample_index])
        ]
        primary_visible = [track for track in visible if id(track) in cover_primary_ids]
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
        track_qualities = _track_image_qualities(sample, visible, sample_index)
        quality = _combined_image_quality(sample, track_qualities)
        quality_by_sample[sample_index] = quality
        # A crop too small to measure borrows the sample's quality.
        best_primary_view = max(
            (
                _subject_view_score(
                    area,
                    track_qualities.get(id(track), quality).score,
                )
                for track, (_clearance, area) in zip(primary_visible, primary_metrics)
            ),
            default=0.0,
        )
        best_primary_area = max(
            (area for _clearance, area in primary_metrics),
            default=0.0,
        )
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
            len(admitted_visible),
            fully_framed_primary,
            best_primary_view,
            best_primary_area,
            best_primary_clearance,
            len(visible),
            face_quality,
            sum(track.aggregate_confidence for track in visible),
            quality.score,
            raw_peak,
            -abs(sample.offset),
        )

    selected_index, selected = max(enumerate(samples), key=sample_score)
    if selected.frame is None:
        raise ValueError("recorded consensus samples must retain frames")

    def enrich_observation(
        detected: dict[str, Any],
        track: _TemporalDetectionEvidence,
        observation_index: int,
        *,
        snapshot_visible: bool,
    ) -> dict[str, Any]:
        observation_sample = samples[observation_index]
        if observation_sample.frame is None:
            raise ValueError("recorded consensus samples must retain frames")
        observation_quality = quality_by_sample[observation_index]
        confirmed = id(track) in confirmed_ids
        label_confirmed_here = str(detected.get("label") or "").strip() == track.winning_label
        confirmed = confirmed and label_confirmed_here
        incident_confirmed = (
            id(track) in normally_confirmed_ids
            and label_confirmed_here
            and _eligible_detection(detected)
            and _meets_configured_confidence_floor(detected)
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
        edge_clearance, subject_area = _normalized_box_metrics(
            detected,
            observation_sample.frame,
        )
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
            "temporal_sample_offset_seconds": observation_sample.offset,
            "offset_seconds": observation_sample.offset,
            "recording_path": observation_sample.recording_path,
            "frame_timestamp_exact": observation_sample.exact_timestamp,
            "temporal_requested_sample_offset_seconds": (
                observation_sample.requested_offset
                if observation_sample.requested_offset is not None
                else observation_sample.offset
            ),
            "temporal_sample_timestamp_source": (
                "source_pts"
                if observation_sample.exact_timestamp
                else "requested_offset"
            ),
            "temporal_observations": len(track.winning_observations),
            "temporal_track_observations": len(track.observations),
            "temporal_incident_observations": sum(
                1
                for item in track.winning_observations
                if _eligible_detection(item)
                and _meets_configured_confidence_floor(item)
            ),
            "temporal_required_observations": required_by_track[id(track)],
            "temporal_samples": len(samples),
            "temporal_peak_confidence": round(track.peak_confidence, 4),
            "temporal_label_votes": track.label_votes,
            "snapshot_quality_score": round(observation_quality.score, 4),
            "snapshot_sharpness_score": round(observation_quality.sharpness, 4),
            "snapshot_exposure_score": round(observation_quality.exposure, 4),
            "snapshot_contrast_score": round(observation_quality.contrast, 4),
            "snapshot_edge_detail_score": round(observation_quality.edge_detail, 4),
            "snapshot_primary_subject": id(track) in cover_primary_ids,
            "snapshot_edge_clearance_ratio": round(edge_clearance, 5),
            "snapshot_subject_area_ratio": round(subject_area, 5),
        }
        enriched.update(_temporal_trigger_evidence(track, samples))
        motion = motion_by_track.get(id(track))
        if motion is None:
            motion = _temporal_motion_metrics(track, samples)
        # Preserve original aggregate diagnostics for historical comparisons.
        # Admission consumers read the versioned estimate instead.
        enriched["temporal_center_displacement_ratio"] = round(motion.raw_displacement_ratio, 5)
        enriched["temporal_center_path_ratio"] = round(motion.raw_path_ratio, 5)
        enriched["temporal_motion"] = motion.as_dict()
        if confirmed:
            enriched["confidence"] = round(track.aggregate_confidence, 4)
        if not snapshot_visible:
            enriched["snapshot_visible"] = False
        return enriched

    annotated: list[dict[str, Any]] = []
    represented_confirmed_ids: set[int] = set()
    for detected in selected.objects:
        if not _candidate_detection(detected):
            annotated.append(dict(detected))
            continue
        track = assignments.get((selected_index, id(detected)))
        if track is None:
            continue
        annotated.append(
            enrich_observation(
                detected,
                track,
                selected_index,
                snapshot_visible=True,
            )
        )
        if (
            id(track) in confirmed_ids
            and str(detected.get("label") or "").strip() == track.winning_label
        ):
            represented_confirmed_ids.add(id(track))

    # The best presentation frame need not contain every confirmed active
    # subject. Preserve one source observation for each missing primary track
    # so incident admission does not depend on the cover-frame composition.
    for track in evidence:
        if id(track) not in primary_ids or id(track) in represented_confirmed_ids:
            continue
        observation_indices = [
            index
            for index, item in track.observations.items()
            if str(item.get("label") or "").strip() == track.winning_label
        ]
        if not observation_indices:
            continue

        def observation_score(index: int) -> tuple[int, int, float, float, float, float, float]:
            frame = samples[index].frame
            if frame is None:
                raise ValueError("recorded consensus samples must retain frames")
            item = track.observations[index]
            clearance, area = _normalized_box_metrics(item, frame)
            return (
                int(_eligible_detection(item)),
                int(clearance >= REPRESENTATIVE_MIN_EDGE_CLEARANCE_RATIO),
                area,
                clearance,
                _confidence(item),
                quality_by_sample[index].score,
                -abs(samples[index].offset),
            )

        observation_index = max(observation_indices, key=observation_score)
        annotated.append(
            enrich_observation(
                track.observations[observation_index],
                track,
                observation_index,
                snapshot_visible=False,
            )
        )

    LOGGER.debug(
        "recorded object consensus retained %d/%d candidates across %d frames (minimum %d)",
        len(confirmed_ids),
        len(evidence),
        len(samples),
        default_required,
    )
    return selected, annotated
