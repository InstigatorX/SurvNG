"""Typed object activity attribution between detection and incident policy."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Literal, Mapping


AttributionMode = Literal["off", "shadow", "enforce"]


class ObjectActivityRole(StrEnum):
    ACTIVE = "active"
    SCENE_CONTEXT = "scene_context"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class ObjectActivityEvidence:
    displacement_ratio: float
    path_ratio: float
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
        if not self.admitted:
            stored["incident_eligible"] = False
            if self.counterfactual_suppressed:
                stored["incident_ineligible_reason"] = "stationary_scene_context"
        return stored

    @property
    def observation(self) -> dict[str, Any]:
        return self.attribution.observation


@dataclass(slots=True)
class _SceneContextEntry:
    label: str
    box: tuple[float, float, float, float]
    last_seen_epoch: float
    stable_event_keys: list[str] = field(default_factory=list)

    @property
    def stable_sightings(self) -> int:
        return len(self.stable_event_keys)


class ObjectActivityAttributor:
    """Class-agnostic attribution using bounded temporal evidence only.

    The initial enforcement boundary is intentionally conservative. It can
    prove an object is stable scene context, but incomplete evidence always
    remains indeterminate and therefore fail-open.
    """

    STABLE_DISPLACEMENT_RATIO = 0.0025
    STABLE_PATH_RATIO = 0.006
    MINIMUM_MOVEMENT_RATIO = 0.003
    MAXIMUM_MOVEMENT_RATIO = 0.02
    MOVEMENT_BOX_SCALE = 0.04
    CONTEXT_MEMORY_TTL_SECONDS = 2 * 60 * 60
    CONTEXT_MEMORY_MAX_ENTRIES = 128
    CONTEXT_MEMORY_MIN_IOU = 0.72
    CONTEXT_MEMORY_MIN_PRIOR_SIGHTINGS = 2

    def __init__(self, mode: AttributionMode = "enforce") -> None:
        self.mode: AttributionMode = mode
        self._lock = threading.Lock()
        self._counts = {
            "evaluated": 0,
            "active": 0,
            "scene_context": 0,
            "indeterminate": 0,
            "counterfactual_suppressions": 0,
            "enforced_suppressions": 0,
        }
        self._reasons: dict[str, int] = {}
        self._context_memory: list[_SceneContextEntry] = []

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
                "scene_context_memory_entries": len(self._context_memory),
                "scene_context_memory_ttl_seconds": self.CONTEXT_MEMORY_TTL_SECONDS,
            }

    def reconfigure(self, mode: AttributionMode) -> None:
        with self._lock:
            self.mode = mode
            if mode == "off":
                self._context_memory.clear()

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
        displacement = self._finite(observation.get("temporal_center_displacement_ratio"))
        path = self._finite(observation.get("temporal_center_path_ratio"))
        track_observations = self._integer(observation.get("temporal_track_observations"))
        pretrigger = self._integer(observation.get("temporal_pretrigger_observations"))
        posttrigger = self._integer(observation.get("temporal_posttrigger_observations"))
        # Older persisted detections predate the exact counters. Their bounded
        # first/last offsets still provide a conservative replay signal.
        first_offset = self._finite_signed(
            observation.get("temporal_first_observation_offset_seconds")
        )
        last_offset = self._finite_signed(
            observation.get("temporal_last_observation_offset_seconds")
        )
        if pretrigger == 0 and first_offset is not None and first_offset < 0.0:
            pretrigger = 1
        if posttrigger == 0 and last_offset is not None and last_offset >= 0.0:
            posttrigger = 1
        robust_appearance = bool(observation.get("temporal_robust_new_appearance"))
        zone_entry = bool(observation.get("temporal_zone_entry"))
        movement_threshold = self._movement_threshold(observation)
        credible_movement = bool(
            displacement >= movement_threshold
            or path >= max(0.01, movement_threshold * 2.5)
        )
        stable = bool(
            observation.get("temporal_consensus") is True
            and track_observations >= 2
            and pretrigger >= 1
            and posttrigger >= 1
            and displacement <= self.STABLE_DISPLACEMENT_RATIO
            and path <= self.STABLE_PATH_RATIO
            and not robust_appearance
            and not zone_entry
        )
        stable_observation = bool(
            observation.get("temporal_consensus") is True
            and track_observations >= 2
            and displacement <= self.STABLE_DISPLACEMENT_RATIO
            and path <= self.STABLE_PATH_RATIO
            and not robust_appearance
            and not zone_entry
        )
        memory_match, memory_sightings, memory_age = self._context_memory_evidence(
            observation,
            event_key=event_key,
            observed_at_epoch=observed_at_epoch,
        )
        ema_overlap = self._ema_overlap(observation, qualification)
        if credible_movement:
            role = ObjectActivityRole.ACTIVE
            confidence = self._bounded(0.72 + min(0.25, displacement * 4.0 + path * 2.0))
            reasons = ("credible_temporal_movement",)
        elif robust_appearance:
            role = ObjectActivityRole.ACTIVE
            confidence = 0.88
            reasons = ("robust_new_appearance",)
        elif zone_entry:
            role = ObjectActivityRole.ACTIVE
            confidence = 0.86
            reasons = ("zone_entry",)
        elif (
            stable_observation
            and memory_match
            and memory_sightings >= self.CONTEXT_MEMORY_MIN_PRIOR_SIGHTINGS
        ):
            role = ObjectActivityRole.SCENE_CONTEXT
            confidence = self._bounded(0.86 + min(0.1, memory_sightings * 0.02))
            reasons = ("repeated_stable_scene_context",)
        else:
            role = ObjectActivityRole.INDETERMINATE
            confidence = 0.35 if track_observations >= 2 else 0.15
            reasons = ("insufficient_causal_evidence",)
        result = self._result(
            observation,
            role,
            confidence,
            qualification,
            displacement=displacement,
            path=path,
            movement_threshold=movement_threshold,
            track_observations=track_observations,
            pretrigger=pretrigger,
            posttrigger=posttrigger,
            robust_appearance=robust_appearance,
            zone_entry=zone_entry,
            ema_overlap=ema_overlap,
            credible_movement=credible_movement,
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
                    robust_appearance
                    or zone_entry
                    or displacement >= max(0.02, movement_threshold * 2.0)
                    or path >= max(0.04, movement_threshold * 4.0)
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
        displacement: float = 0.0,
        path: float = 0.0,
        movement_threshold: float = MINIMUM_MOVEMENT_RATIO,
        track_observations: int = 0,
        pretrigger: int = 0,
        posttrigger: int = 0,
        robust_appearance: bool = False,
        zone_entry: bool = False,
        ema_overlap: bool | None = None,
        credible_movement: bool = False,
        stable: bool = False,
        memory_match: bool = False,
        memory_sightings: int = 0,
        memory_age: float | None = None,
        reasons: tuple[str, ...] = (),
    ) -> ObjectActivityAttribution:
        overlap = self._ema_overlap(observation, qualification) if ema_overlap is None else ema_overlap
        return ObjectActivityAttribution(
            observation=dict(observation),
            role=role,
            confidence=self._bounded(confidence),
            evidence=ObjectActivityEvidence(
                displacement_ratio=displacement,
                path_ratio=path,
                movement_threshold=movement_threshold,
                track_observations=track_observations,
                pretrigger_observations=pretrigger,
                posttrigger_observations=posttrigger,
                robust_new_appearance=robust_appearance,
                zone_entry=zone_entry,
                ema_region_overlap=overlap,
                # Main/substream registration is not currently calibrated.
                # Keep overlap diagnostic until an explicit mapping exists.
                ema_alignment_reliable=False,
                credible_movement=credible_movement,
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
        with self._lock:
            self._prune_context_memory(observed_at_epoch)
            matches = [
                entry
                for entry in self._context_memory
                if entry.label == label
                and event_key not in entry.stable_event_keys
                and entry.last_seen_epoch <= observed_at_epoch
                and self._box_iou(entry.box, box) >= self.CONTEXT_MEMORY_MIN_IOU
            ]
            if not matches:
                return False, 0, None
            match = max(matches, key=lambda entry: self._box_iou(entry.box, box))
            return (
                True,
                match.stable_sightings,
                max(0.0, observed_at_epoch - match.last_seen_epoch),
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
        with self._lock:
            self._prune_context_memory(observed_at_epoch)
            match = next(
                (
                    entry
                    for entry in self._context_memory
                    if entry.label == label
                    and entry.last_seen_epoch <= observed_at_epoch
                    and self._box_iou(entry.box, box) >= self.CONTEXT_MEMORY_MIN_IOU
                ),
                None,
            )
            if match is not None:
                if event_key not in match.stable_event_keys:
                    match.stable_event_keys.append(event_key)
                match.box = box
                match.last_seen_epoch = observed_at_epoch
            else:
                self._context_memory.append(
                    _SceneContextEntry(label, box, observed_at_epoch, [event_key])
                )
            if len(self._context_memory) > self.CONTEXT_MEMORY_MAX_ENTRIES:
                self._context_memory.sort(key=lambda entry: entry.last_seen_epoch)
                del self._context_memory[:-self.CONTEXT_MEMORY_MAX_ENTRIES]

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
        with self._lock:
            retained: list[_SceneContextEntry] = []
            for entry in self._context_memory:
                matches = bool(
                    entry.label == label
                    and self._box_iou(entry.box, box) >= self.CONTEXT_MEMORY_MIN_IOU
                )
                if matches and invalidate_location:
                    continue
                if matches and event_key in entry.stable_event_keys:
                    entry.stable_event_keys.remove(event_key)
                if entry.stable_event_keys:
                    retained.append(entry)
            self._context_memory[:] = retained

    def _prune_context_memory(self, observed_at_epoch: float) -> None:
        cutoff = observed_at_epoch - self.CONTEXT_MEMORY_TTL_SECONDS
        self._context_memory[:] = [
            entry for entry in self._context_memory if entry.last_seen_epoch >= cutoff
        ]

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

    @classmethod
    def _movement_threshold(cls, observation: Mapping[str, Any]) -> float:
        box = observation.get("box")
        try:
            width = float(observation.get("detection_frame_width") or 0.0)
            height = float(observation.get("detection_frame_height") or 0.0)
            if not isinstance(box, Mapping) or width <= 0 or height <= 0:
                raise ValueError
            diagonal = math.hypot(
                (float(box["x2"]) - float(box["x1"])) / width,
                (float(box["y2"]) - float(box["y1"])) / height,
            )
        except (KeyError, TypeError, ValueError):
            return cls.MAXIMUM_MOVEMENT_RATIO
        return min(
            cls.MAXIMUM_MOVEMENT_RATIO,
            max(cls.MINIMUM_MOVEMENT_RATIO, diagonal * cls.MOVEMENT_BOX_SCALE),
        )

    @staticmethod
    def _normalized_box(
        observation: Mapping[str, Any],
    ) -> tuple[float, float, float, float] | None:
        box = observation.get("box")
        try:
            width = float(observation.get("detection_frame_width") or 0.0)
            height = float(observation.get("detection_frame_height") or 0.0)
            if not isinstance(box, Mapping) or width <= 0 or height <= 0:
                return None
            normalized = (
                float(box["x1"]) / width,
                float(box["y1"]) / height,
                float(box["x2"]) / width,
                float(box["y2"]) / height,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            return None
        return normalized

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
    def _finite(value: object) -> float:
        try:
            result = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, result) if math.isfinite(result) else 0.0

    @staticmethod
    def _finite_signed(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _integer(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
