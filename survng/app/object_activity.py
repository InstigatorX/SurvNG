"""Typed object activity attribution between detection and incident policy."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Literal, Mapping

from .object_motion import (
    TemporalObjectMotionEvidence,
    estimate_object_motion,
    temporal_object_motion_evidence,
)
from .scene_context_memory import InMemorySceneContextMemory, SceneContextMemory, normalized_box
from .stationary_policy import StationaryObjectPolicy, stationary_object_policy


AttributionMode = Literal["off", "shadow", "enforce"]


class ObjectActivityRole(StrEnum):
    ACTIVE = "active"
    SCENE_CONTEXT = "scene_context"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class ObjectActivityEvidence:
    displacement_ratio: float
    path_ratio: float
    excursion_ratio: float | None
    movement_threshold: float
    track_observations: int
    pretrigger_observations: int
    posttrigger_observations: int
    robust_new_appearance: bool
    zone_entry: bool
    ema_region_overlap: bool
    ema_alignment_reliable: bool
    credible_movement: bool
    stable_across_trigger: bool
    scene_context_memory_match: bool
    scene_context_memory_sightings: int
    scene_context_memory_age_seconds: float | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "displacement_ratio": round(self.displacement_ratio, 5),
            "path_ratio": round(self.path_ratio, 5),
            "excursion_ratio": self.excursion_ratio,
            "motion_evidence_method": "legacy_path" if self.excursion_ratio is None else "bounded_excursion",
            "movement_threshold": round(self.movement_threshold, 5),
            "track_observations": self.track_observations,
            "pretrigger_observations": self.pretrigger_observations,
            "posttrigger_observations": self.posttrigger_observations,
            "robust_new_appearance": self.robust_new_appearance,
            "zone_entry": self.zone_entry,
            "ema_region_overlap": self.ema_region_overlap,
            "ema_alignment_reliable": self.ema_alignment_reliable,
            "credible_movement": self.credible_movement,
            "stable_across_trigger": self.stable_across_trigger,
            "scene_context_memory_match": self.scene_context_memory_match,
            "scene_context_memory_sightings": self.scene_context_memory_sightings,
            "scene_context_memory_age_seconds": (
                round(self.scene_context_memory_age_seconds, 3)
                if self.scene_context_memory_age_seconds is not None
                else None
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ObjectActivityAttribution:
    observation: dict[str, Any]
    role: ObjectActivityRole
    confidence: float
    evidence: ObjectActivityEvidence

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObjectIncidentAdmission:
    attribution: ObjectActivityAttribution
    detector_eligible: bool
    admitted: bool
    counterfactual_suppressed: bool
    reason: str

    def stored_observation(self) -> dict[str, Any]:
        if not self.attribution.observation.get("label"):
            return dict(self.attribution.observation)
        stored = dict(self.observation)
        stored["detector_incident_eligible"] = self.detector_eligible
        stored["activity_role"] = self.attribution.role.value
        stored["activity_confidence"] = round(self.attribution.confidence, 4)
        stored["activity_evidence"] = self.attribution.evidence.as_dict()
        stored["activity_counterfactual_suppressed"] = self.counterfactual_suppressed
        stored["activity_admission_reason"] = self.reason
        stored["activity_eligible"] = self.admitted
        if not self.admitted:
            stored["incident_eligible"] = False
            if self.counterfactual_suppressed:
                stored["incident_ineligible_reason"] = "stationary_scene_context"
                existing = stored.get("incident_ineligible_reasons")
                reasons = (
                    [str(value) for value in existing]
                    if isinstance(existing, list)
                    else [str(existing)] if existing else []
                )
                stored["incident_ineligible_reasons"] = list(dict.fromkeys([
                    *reasons,
                    "stationary_scene_context",
                ]))
        return stored

    @property
    def observation(self) -> dict[str, Any]:
        return self.attribution.observation


def apply_stationary_alert(observation: dict[str, Any], *, presence: str, mode: str) -> None:
    """Clear alert eligibility for an enforced stationary subject.

    An unconfirmed track keeps its temporal reason. Shadow mode records the
    counterfactual and leaves eligibility unchanged.
    """
    effect = stationary_subject_effect(str(observation.get("activity_role") or ""), presence, mode)
    if effect == "shadow":
        observation["alert_shadow_reasons"] = ["stationary_scene_context"]
    if effect != "enforce":
        return
    if observation.get("alert_reasons") == ["temporal_unconfirmed"]:
        return
    if str(observation.get("label") or "").lower() == "face":
        return
    observation["alert_eligible"] = False
    observation["alert_reasons"] = ["stationary_scene_context"]


def stationary_subject_effect(role: str, presence: str, mode: str) -> str:
    """How a proven stationary subject affects establishment and alerts.

    ``enforce`` suppresses presence. ``shadow`` records the counterfactual.
    Anything else, including a missing role, fails open.
    """
    if str(role or "") != "scene_context" or presence != "ignore":
        return ""
    if mode == "enforce":
        return "enforce"
    if mode == "shadow":
        return "shadow"
    return ""


def annotate_box_history_motion(
    observation: dict[str, Any],
    box_history: Any,
    *,
    width: float,
    height: float,
) -> None:
    """Attach bounded-excursion motion from a track's pixel box history."""
    if width <= 0 or height <= 0 or not isinstance(box_history, list):
        return
    centers: list[tuple[float, float, float]] = []
    for sample in box_history:
        if not isinstance(sample, (list, tuple)) or len(sample) < 5:
            continue
        try:
            timestamp, x1, y1, x2, y2 = (float(value) for value in sample[:5])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (timestamp, x1, y1, x2, y2)):
            continue
        centers.append((timestamp, (x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height)))
    estimate = estimate_object_motion(centers)
    observation["temporal_track_observations"] = estimate.samples
    observation["temporal_consensus"] = estimate.samples >= 2
    observation["temporal_pretrigger_observations"] = 1 if estimate.samples >= 2 else 0
    observation["temporal_posttrigger_observations"] = 1 if estimate.samples >= 2 else 0
    observation["temporal_center_displacement_ratio"] = estimate.raw_displacement_ratio
    observation["temporal_center_path_ratio"] = estimate.filtered_path_ratio
    observation["temporal_motion"] = estimate.as_dict()
    observation["temporal_robust_new_appearance"] = False
    observation.setdefault("temporal_zone_entry", False)


class ObjectActivityAttributor:
    """Class-agnostic attribution using bounded temporal evidence only.

    The initial enforcement boundary is intentionally conservative. It can
    prove an object is stable scene context, but incomplete evidence always
    remains indeterminate and therefore fail-open.
    """

    CONTEXT_MEMORY_TTL_SECONDS = 2 * 60 * 60
    CONTEXT_MEMORY_MAX_ENTRIES = 128
    CONTEXT_MEMORY_MAX_SIGHTINGS = 16

    def __init__(
        self,
        mode: AttributionMode = "enforce",
        stationary_tolerance: str = "balanced",
        *,
        memory: SceneContextMemory | None = None,
        camera_id: str = "",
    ) -> None:
        self.mode: AttributionMode = mode
        self.camera_id = camera_id
        self.memory: SceneContextMemory = memory or InMemorySceneContextMemory()
        self.stationary_policy: StationaryObjectPolicy = stationary_object_policy(
            stationary_tolerance
        )
        self._lock = threading.Lock()
        self._counts = {
            "evaluated": 0,
            "active": 0,
            "scene_context": 0,
            "indeterminate": 0,
            "counterfactual_suppressions": 0,
            "enforced_suppressions": 0,
            "zone_rejections": 0,
            "confidence_rejections": 0,
            "temporal_rejections": 0,
            "detector_admissions": 0,
            "semantic_rescue_candidates": 0,
            "semantic_rescue_admissions": 0,
            "semantic_rescue_rejections": 0,
        }
        self._reasons: dict[str, int] = {}
        self._semantic_rescue_by_source: dict[str, dict[str, int]] = {}
    def attribute(
        self,
        objects: Iterable[dict[str, Any]],
        qualification: Mapping[str, Any],
        *,
        event_key: str = "",
        observed_at_epoch: float | None = None,
    ) -> tuple[ObjectActivityAttribution, ...]:
        now = float(observed_at_epoch if observed_at_epoch is not None else time.time())
        key = event_key or f"runtime:{now:.6f}"
        attributed = tuple(
            self._attribute_one(item, qualification, event_key=key, observed_at_epoch=now)
            for item in objects
            if isinstance(item, dict)
        )
        self._record(attributed)
        return attributed

    def admit(
        self,
        objects: Iterable[dict[str, Any]],
        qualification: Mapping[str, Any],
        *,
        event_key: str = "",
        observed_at_epoch: float | None = None,
    ) -> tuple[ObjectIncidentAdmission, ...]:
        results = self.attribute(
            objects,
            qualification,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
        )
        return tuple(self._admit_one(result) for result in results)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                **self._counts,
                "reasons": dict(self._reasons),
                "semantic_rescue_by_source": {
                    source: dict(counts)
                    for source, counts in self._semantic_rescue_by_source.items()
                },
                "scene_context_memory_entries": self.memory.count(self.camera_id),
                "scene_context_memory_ttl_seconds": self.stationary_policy.scene_memory_ttl_seconds,
                "stationary_policy": self.stationary_policy.as_dict(),
            }

    def record_semantic_rescue(
        self,
        *,
        source: str,
        candidates: int,
        admitted: int,
        rejected: int,
    ) -> None:
        """Record final rescue outcomes after the admission policy runs."""
        if candidates <= 0:
            return
        normalized_source = str(source or "unknown").strip().lower() or "unknown"
        with self._lock:
            self._counts["semantic_rescue_candidates"] += max(0, int(candidates))
            self._counts["semantic_rescue_admissions"] += max(0, int(admitted))
            self._counts["semantic_rescue_rejections"] += max(0, int(rejected))
            counts = self._semantic_rescue_by_source.setdefault(
                normalized_source,
                {"candidates": 0, "admitted": 0, "rejected": 0},
            )
            counts["candidates"] += max(0, int(candidates))
            counts["admitted"] += max(0, int(admitted))
            counts["rejected"] += max(0, int(rejected))

    def reconfigure(self, mode: AttributionMode) -> None:
        with self._lock:
            self.mode = mode
            if mode == "off":
                self.memory.clear(self.camera_id)

    def reconfigure_stationary_tolerance(self, tolerance: str) -> None:
        with self._lock:
            self.stationary_policy = stationary_object_policy(tolerance)

    def stamp_recorded(
        self,
        observations: Iterable[dict[str, Any]],
        *,
        event_key: str,
        observed_at_epoch: float,
    ) -> None:
        """Record the same activity role on recorded observations.

        Enforcement stays with establishment and alerts. This only writes the
        role, and it does not remember subjects while attribution is off.
        """
        if self.mode == "off":
            return
        for item in observations:
            if not isinstance(item, dict) or not item.get("label"):
                continue
            result = self._attribute_one(
                item,
                {},
                event_key=event_key,
                observed_at_epoch=observed_at_epoch,
            )
            item["activity_role"] = result.role.value
            item["activity_confidence"] = round(result.confidence, 4)
            item["activity_evidence"] = result.evidence.as_dict()

    def _attribute_one(
        self,
        observation: dict[str, Any],
        qualification: Mapping[str, Any],
        *,
        event_key: str,
        observed_at_epoch: float,
    ) -> ObjectActivityAttribution:
        if not observation.get("label"):
            return self._result(
                observation,
                ObjectActivityRole.INDETERMINATE,
                0.0,
                qualification,
                reasons=("not_object_detection",),
            )
        motion = temporal_object_motion_evidence(observation)
        stable = bool(
            observation.get("temporal_consensus") is True
            and motion.stable(
                maximum_displacement_ratio=self.stationary_policy.scene_stable_displacement_ratio,
                maximum_path_ratio=self.stationary_policy.scene_stable_path_ratio,
                require_trigger_span=True,
            )
        )
        stable_observation = bool(
            observation.get("temporal_consensus") is True
            and motion.temporal_evidence_available
            and motion.displacement_ratio
            <= self.stationary_policy.scene_stable_displacement_ratio
            and motion.movement_extent_ratio <= self.stationary_policy.scene_stable_path_ratio
            and not motion.zone_entry
        )
        with self.memory.hold():
            return self._attribute_stable(
                observation,
                qualification,
                motion=motion,
                stable=stable,
                stable_observation=stable_observation,
                event_key=event_key,
                observed_at_epoch=observed_at_epoch,
            )

    def _attribute_stable(
        self,
        observation: dict[str, Any],
        qualification: Mapping[str, Any],
        *,
        motion: TemporalObjectMotionEvidence,
        stable: bool,
        stable_observation: bool,
        event_key: str,
        observed_at_epoch: float,
    ) -> ObjectActivityAttribution:
        memory_match, memory_sightings, memory_age = self._context_memory_evidence(
            observation,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
        )
        ema_overlap = self._ema_overlap(observation, qualification)
        if motion.credible_movement:
            role = ObjectActivityRole.ACTIVE
            confidence = self._bounded(
                0.72
                + min(
                    0.25,
                    motion.displacement_ratio * 4.0 + motion.movement_extent_ratio * 2.0,
                )
            )
            reasons = ("credible_temporal_movement",)
        elif motion.zone_entry:
            role = ObjectActivityRole.ACTIVE
            confidence = 0.86
            reasons = ("zone_entry",)
        elif (
            stable_observation
            and memory_match
            and memory_sightings >= self.stationary_policy.scene_memory_min_prior_sightings
        ):
            role = ObjectActivityRole.SCENE_CONTEXT
            confidence = self._bounded(0.86 + min(0.1, memory_sightings * 0.02))
            reasons = ("repeated_stable_scene_context",)
        else:
            role = ObjectActivityRole.INDETERMINATE
            confidence = 0.35 if motion.track_observations >= 2 else 0.15
            reasons = (
                ("robust_new_appearance_without_causal_motion",)
                if motion.robust_new_appearance
                else ("insufficient_causal_evidence",)
            )
        result = self._result(
            observation,
            role,
            confidence,
            qualification,
            motion=motion,
            ema_overlap=ema_overlap,
            stable=stable,
            memory_match=memory_match,
            memory_sightings=memory_sightings,
            memory_age=memory_age,
            reasons=reasons,
        )
        if stable_observation:
            self._remember_scene_context(
                observation,
                event_key=event_key,
                observed_at_epoch=observed_at_epoch,
            )
        elif role is ObjectActivityRole.ACTIVE:
            self._forget_scene_context_observation(
                observation,
                event_key=event_key,
                invalidate_location=bool(
                    motion.zone_entry
                    or motion.displacement_ratio
                    >= max(0.02, motion.movement_threshold * 2.0)
                    or motion.movement_extent_ratio
                    >= max(0.04, motion.movement_threshold * 4.0)
                ),
            )
        return result

    def _result(
        self,
        observation: dict[str, Any],
        role: ObjectActivityRole,
        confidence: float,
        qualification: Mapping[str, Any],
        *,
        motion: TemporalObjectMotionEvidence | None = None,
        ema_overlap: bool | None = None,
        stable: bool = False,
        memory_match: bool = False,
        memory_sightings: int = 0,
        memory_age: float | None = None,
        reasons: tuple[str, ...] = (),
    ) -> ObjectActivityAttribution:
        motion = motion or temporal_object_motion_evidence(observation)
        overlap = self._ema_overlap(observation, qualification) if ema_overlap is None else ema_overlap
        return ObjectActivityAttribution(
            observation=dict(observation),
            role=role,
            confidence=self._bounded(confidence),
            evidence=ObjectActivityEvidence(
                displacement_ratio=motion.displacement_ratio,
                path_ratio=motion.path_ratio,
                excursion_ratio=motion.excursion_ratio,
                movement_threshold=motion.movement_threshold,
                track_observations=motion.track_observations,
                pretrigger_observations=motion.pretrigger_observations,
                posttrigger_observations=motion.posttrigger_observations,
                robust_new_appearance=motion.robust_new_appearance,
                zone_entry=motion.zone_entry,
                ema_region_overlap=overlap,
                # Main/substream registration is not currently calibrated.
                # Keep overlap diagnostic until an explicit mapping exists.
                ema_alignment_reliable=False,
                credible_movement=motion.credible_movement,
                stable_across_trigger=stable,
                scene_context_memory_match=memory_match,
                scene_context_memory_sightings=memory_sightings,
                scene_context_memory_age_seconds=memory_age,
                reasons=reasons,
            ),
        )

    def _context_memory_evidence(
        self,
        observation: Mapping[str, Any],
        *,
        event_key: str,
        observed_at_epoch: float,
    ) -> tuple[bool, int, float | None]:
        label = str(observation.get("label") or "").strip().lower()
        box = self._normalized_box(observation)
        if not label or box is None:
            return False, 0, None
        return self.memory.match(
            camera_id=self.camera_id,
            label=label,
            box=box,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
            min_iou=self.stationary_policy.scene_memory_min_iou,
            ttl_seconds=self.stationary_policy.scene_memory_ttl_seconds,
        )

    def _remember_scene_context(
        self,
        observation: Mapping[str, Any],
        *,
        event_key: str,
        observed_at_epoch: float,
    ) -> None:
        label = str(observation.get("label") or "").strip().lower()
        box = self._normalized_box(observation)
        if not label or box is None:
            return
        self.memory.remember(
            camera_id=self.camera_id,
            label=label,
            box=box,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
            min_iou=self.stationary_policy.scene_memory_min_iou,
            ttl_seconds=self.stationary_policy.scene_memory_ttl_seconds,
            max_entries=self.CONTEXT_MEMORY_MAX_ENTRIES,
            max_sightings=self.CONTEXT_MEMORY_MAX_SIGHTINGS,
        )

    def _forget_scene_context_observation(
        self,
        observation: Mapping[str, Any],
        *,
        event_key: str,
        invalidate_location: bool,
    ) -> None:
        label = str(observation.get("label") or "").strip().lower()
        box = self._normalized_box(observation)
        if not label or box is None:
            return
        self.memory.forget(
            camera_id=self.camera_id,
            label=label,
            box=box,
            event_key=event_key,
            invalidate_location=invalidate_location,
            min_iou=self.stationary_policy.scene_memory_min_iou,
        )

    def _admit_one(
        self,
        attribution: ObjectActivityAttribution,
    ) -> ObjectIncidentAdmission:
        observation = attribution.observation
        detector_eligible = observation.get("incident_eligible") is not False
        counterfactual = bool(
            detector_eligible
            and attribution.role is ObjectActivityRole.SCENE_CONTEXT
        )
        enforced = self.mode == "enforce" and counterfactual
        return ObjectIncidentAdmission(
            attribution=attribution,
            detector_eligible=detector_eligible,
            admitted=bool(detector_eligible and not enforced),
            counterfactual_suppressed=counterfactual,
            reason=(
                "stationary_scene_context"
                if enforced
                else "shadow_scene_context"
                if counterfactual
                else "detector_ineligible"
                if not detector_eligible
                else "activity_eligible"
            ),
        )

    def _record(self, results: tuple[ObjectActivityAttribution, ...]) -> None:
        with self._lock:
            for result in results:
                if not result.observation.get("label"):
                    continue
                self._counts["evaluated"] += 1
                self._counts[result.role.value] += 1
                if result.observation.get("incident_eligible") is not False:
                    self._counts["detector_admissions"] += 1
                elif result.observation.get("confidence_eligible") is False:
                    self._counts["confidence_rejections"] += 1
                elif result.observation.get("zone_eligible") is False:
                    self._counts["zone_rejections"] += 1
                elif result.observation.get("temporal_eligible") is False:
                    self._counts["temporal_rejections"] += 1
                counterfactual = bool(
                    result.observation.get("incident_eligible") is not False
                    and result.role is ObjectActivityRole.SCENE_CONTEXT
                )
                if counterfactual:
                    self._counts["counterfactual_suppressions"] += 1
                    if self.mode == "enforce":
                        self._counts["enforced_suppressions"] += 1
                for reason in result.evidence.reasons:
                    self._reasons[reason] = self._reasons.get(reason, 0) + 1

    @staticmethod
    def _normalized_box(
        observation: Mapping[str, Any],
    ) -> tuple[float, float, float, float] | None:
        return normalized_box(observation)

    @staticmethod
    def _box_iou(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = intersection_width * intersection_height
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _ema_overlap(
        observation: Mapping[str, Any],
        qualification: Mapping[str, Any],
    ) -> bool:
        features = qualification.get("features")
        regions = features.get("motion_regions") if isinstance(features, Mapping) else None
        box = observation.get("box")
        try:
            width = float(observation.get("detection_frame_width") or 0.0)
            height = float(observation.get("detection_frame_height") or 0.0)
            if not isinstance(box, Mapping) or width <= 0 or height <= 0:
                return False
            normalized = (
                float(box["x1"]) / width,
                float(box["y1"]) / height,
                float(box["x2"]) / width,
                float(box["y2"]) / height,
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in normalized):
            return False
        if not isinstance(regions, list):
            return False
        x1, y1, x2, y2 = normalized
        for region in regions:
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                continue
            try:
                rx1, ry1, rx2, ry2 = (float(value) for value in region)
            except (TypeError, ValueError):
                continue
            if min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1):
                return True
        return False

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
