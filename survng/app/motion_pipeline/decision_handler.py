from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import time
from typing import Any, Callable, Protocol

from ..detector import detection_failure
from ..domain_events import IncidentCreated, ObjectDetected
from ..face_candidates import FaceCandidate
from ..object_activity import AttributionMode, ObjectActivityAttributor
from ..object_motion import (
    MAXIMUM_MOVEMENT_RATIO,
    MINIMUM_MOVEMENT_RATIO,
    temporal_object_motion_evidence,
)
from .context import Frame


MotionDetectionProvider = Callable[[datetime], Any]
MotionSnapshotWriter = Callable[[Frame, datetime], str]
MotionEventCallback = Callable[[str, dict[str, Any]], None]
MotionObjectSerializer = Callable[[list[dict[str, Any]]], str]
FaceCandidateSink = Callable[[int, str, str, list[dict[str, Any]]], int]


LOGGER = logging.getLogger(__name__)
MOTION_REGION_MARGIN_RATIO = 0.035


def _intersects_motion_region(
    box: tuple[float, float, float, float],
    regions: list[object],
) -> bool:
    x1, y1, x2, y2 = box
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            continue
        try:
            rx1, ry1, rx2, ry2 = (float(value) for value in region)
        except (TypeError, ValueError):
            continue
        rx1 -= MOTION_REGION_MARGIN_RATIO
        ry1 -= MOTION_REGION_MARGIN_RATIO
        rx2 += MOTION_REGION_MARGIN_RATIO
        ry2 += MOTION_REGION_MARGIN_RATIO
        if min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1):
            return True
    return False


def motion_correlated_objects(
    frame: Frame,
    objects: list[dict[str, Any]],
    qualification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep objects that spatially or temporally explain an EMA trigger."""
    features = qualification.get("features")
    regions = features.get("motion_regions", []) if isinstance(features, dict) else []
    if not isinstance(regions, list):
        regions = []
    correlated: list[dict[str, Any]] = []
    spatial_matches = 0
    temporal_matches = 0
    temporal_path_matches = 0
    new_appearance_matches = 0
    stationary_spatial_rejections = 0
    height, width = frame.shape[:2]
    for detected in objects:
        evidence = temporal_object_motion_evidence(
            detected,
            frame_width=width,
            frame_height=height,
        )
        box = evidence.normalized_box
        spatial = bool(box is not None and _intersects_motion_region(box, regions))
        temporal = evidence.displacement_ratio >= evidence.movement_threshold
        spatial_fallback = spatial and not evidence.temporal_evidence_available
        appearance_match = spatial and evidence.newly_appeared
        # A short recorded sequence can begin and end at nearly the same point
        # while a real object walks through the EMA region.  Permit that only
        # when spatial evidence also agrees and the travelled path is well
        # beyond ordinary detector-box jitter.  A stationary object therefore
        # still cannot explain unrelated motion beside or behind it.
        spatial_path = bool(
            spatial
            and evidence.temporal_evidence_available
            and evidence.path_ratio >= evidence.path_threshold
        )
        motion_correlated = bool(
            temporal or spatial_path or spatial_fallback or appearance_match
        )
        detected["motion_correlated"] = motion_correlated
        detected["motion_correlation"] = (
            "temporal" if temporal else
            "spatial_path" if spatial_path else
            "appearance" if appearance_match else
            "spatial" if spatial_fallback else
            "none"
        )
        detected["motion_correlation_threshold"] = round(
            evidence.movement_threshold,
            5,
        )
        detected["motion_correlation_eligible"] = motion_correlated
        detected["motion_temporal_evidence_available"] = (
            evidence.temporal_evidence_available
        )
        if motion_correlated:
            correlated.append(detected)
            spatial_matches += int(spatial)
            temporal_matches += int(temporal)
            temporal_path_matches += int(spatial_path)
            new_appearance_matches += int(appearance_match)
        else:
            # Preserve the detection as diagnostic evidence without allowing
            # an unrelated stationary object to become an incident label.
            detected["incident_eligible"] = False
            existing_reasons = detected.get("incident_ineligible_reasons")
            reasons = (
                [str(value) for value in existing_reasons]
                if isinstance(existing_reasons, list)
                else [str(existing_reasons)] if existing_reasons else []
            )
            detected["incident_ineligible_reasons"] = list(dict.fromkeys([
                *reasons,
                "object_not_motion_correlated",
            ]))
            stationary_spatial_rejections += int(
                spatial and evidence.temporal_evidence_available
            )
    return correlated, {
        "required": True,
        "motion_region_count": len(regions),
        "eligible_object_count": len(objects),
        "correlated_object_count": len(correlated),
        "spatial_match_count": spatial_matches,
        "temporal_match_count": temporal_matches,
        "temporal_path_match_count": temporal_path_matches,
        "new_appearance_match_count": new_appearance_matches,
        "stationary_spatial_rejection_count": stationary_spatial_rejections,
        "minimum_temporal_movement_ratio": MAXIMUM_MOVEMENT_RATIO,
        "adaptive_minimum_temporal_movement_ratio": MINIMUM_MOVEMENT_RATIO,
        "region_margin_ratio": MOTION_REGION_MARGIN_RATIO,
    }


class MotionEventStore(Protocol):
    def add_event(
        self,
        camera_id: str,
        kind: str,
        topic: str = "",
        message: str = "",
        snapshot_path: str = "",
        recording_path: str = "",
        objects_json: str = "[]",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        ...

    def add_motion_audit(
        self,
        camera_id: str,
        snapshot_path: str,
        created_at: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        ...

    def refine_event_evidence(
        self,
        event_id: int,
        *,
        snapshot_path: str,
        recording_path: str,
        objects_json: str,
    ) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True, slots=True)
class MotionDecisionOutcome:
    event_id: int | None
    snapshot_path: str
    object_detected: bool | None
    detected_objects: tuple[dict[str, Any], ...] = ()
    rejection_reason: str = ""
    motion_correlation: dict[str, Any] | None = None
    refinement_pending: bool = False
    processing_timing: dict[str, Any] | None = None
    object_activity: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "snapshot_path": self.snapshot_path,
            "object_detected": self.object_detected,
            "rejection_reason": self.rejection_reason,
            "motion_correlation": self.motion_correlation,
            "refinement_pending": self.refinement_pending,
            "processing_timing": self.processing_timing,
            "object_activity": self.object_activity,
        }


class MotionDecisionHandler:
    """Consumes a motion decision and owns downstream event side effects."""

    def __init__(
        self,
        camera_id: str,
        events: MotionEventStore,
        detection_provider: MotionDetectionProvider,
        snapshot_writer: MotionSnapshotWriter,
        object_serializer: MotionObjectSerializer,
        initial_detection_provider: MotionDetectionProvider | None = None,
        event_callback: MotionEventCallback | None = None,
        activity_attributor: ObjectActivityAttributor | None = None,
        face_candidate_sink: FaceCandidateSink | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.detection_provider = detection_provider
        self.initial_detection_provider = initial_detection_provider
        self.snapshot_writer = snapshot_writer
        self.object_serializer = object_serializer
        self.event_callback = event_callback
        self.activity_attributor = activity_attributor
        self.face_candidate_sink = face_candidate_sink

    def activity_status(self) -> dict[str, Any]:
        if self.activity_attributor is None:
            return {"mode": "off", "evaluated": 0}
        return self.activity_attributor.status()

    def reconfigure_activity_attribution(self, mode: AttributionMode) -> None:
        if self.activity_attributor is not None:
            self.activity_attributor.reconfigure(mode)

    def handle(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        return self._handle_with_provider(
            self.initial_detection_provider or self.detection_provider,
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )

    def refine(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        existing_event_id: int | None,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        """Run delayed evidence and atomically enrich or create the incident."""
        return self._handle_with_provider(
            self.detection_provider,
            topic,
            message,
            event_at,
            qualification,
            existing_event_id=existing_event_id,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )

    def _handle_with_provider(
        self,
        provider: MotionDetectionProvider,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        existing_event_id: int | None = None,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        detection_started = time.monotonic()
        detection_started_epoch = time.time()
        provider_result = provider(event_at)
        frame, objects, recording_path = provider_result
        processed_at = datetime.now(timezone.utc)
        normalized_event_at = (
            event_at.replace(tzinfo=timezone.utc)
            if event_at.tzinfo is None
            else event_at.astimezone(timezone.utc)
        )
        workflow_ms = round(
            (time.monotonic() - detection_started) * 1000,
            3,
        )
        qualification["object_detection_workflow_ms"] = workflow_ms
        timings = getattr(provider_result, "timings_ms", None)
        if isinstance(timings, dict):
            qualification["object_detection_phases_ms"] = {
                str(key): round(max(0.0, float(value)), 3)
                for key, value in timings.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        received_at_epoch = qualification.get("trigger_received_at_epoch")
        if isinstance(received_at_epoch, (int, float)):
            qualification["decision_queue_wait_ms"] = round(
                max(0.0, (detection_started_epoch - float(received_at_epoch)) * 1000.0),
                3,
            )
        qualification["processed_at"] = processed_at.isoformat()
        qualification["event_processing_delay_seconds"] = round(
            max(0.0, (processed_at - normalized_event_at).total_seconds()),
            3,
        )
        if frame is None:
            objects = [{"status": "no_recorded_frame"}]
        processing_timing = {
            "workflow_ms": workflow_ms,
            "decision_queue_wait_ms": qualification.get("decision_queue_wait_ms"),
            "phases_ms": qualification.get("object_detection_phases_ms", {}),
        }

        detection_completed = frame is not None and not detection_failure(objects)

        activity_summary: dict[str, Any] | None = None
        if (
            self.activity_attributor is not None
            and self.activity_attributor.mode != "off"
        ):
            admissions = self.activity_attributor.admit(
                objects,
                qualification,
                event_key=normalized_event_at.isoformat(),
                observed_at_epoch=normalized_event_at.timestamp(),
            )
            objects = [admission.stored_observation() for admission in admissions]
            activity_objects = [
                {
                    "label": admission.observation.get("label"),
                    "role": admission.attribution.role.value,
                    "confidence": round(admission.attribution.confidence, 4),
                    "admitted": admission.admitted,
                    "reason": admission.reason,
                }
                for admission in admissions
                if admission.observation.get("label")
            ]
            activity_summary = {
                "mode": self.activity_attributor.mode,
                "observed": len(activity_objects),
                "detector_admitted": sum(
                    admission.detector_eligible for admission in admissions
                    if admission.observation.get("label")
                ),
                "zone_rejected": sum(
                    admission.observation.get("confidence_eligible") is not False
                    and admission.observation.get("zone_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "confidence_rejected": sum(
                    admission.observation.get("confidence_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "temporal_rejected": sum(
                    admission.observation.get("zone_eligible") is not False
                    and admission.observation.get("temporal_eligible") is False
                    for admission in admissions
                    if admission.observation.get("label")
                ),
                "active": sum(item["role"] == "active" for item in activity_objects),
                "scene_context": sum(
                    item["role"] == "scene_context" for item in activity_objects
                ),
                "indeterminate": sum(
                    item["role"] == "indeterminate" for item in activity_objects
                ),
                "admitted": sum(bool(item["admitted"]) for item in activity_objects),
                "scene_context_suppressed": sum(
                    item["reason"] == "stationary_scene_context"
                    for item in activity_objects
                ),
                "objects": activity_objects,
            }
            qualification["object_activity_attribution"] = activity_summary

        eligible_objects = [
            detected
            for detected in objects
            if detected.get("label") and detected.get("incident_eligible") is not False
        ]
        correlation: dict[str, Any] | None = None
        uncorrelated_eligible_objects = 0
        if require_motion_correlation and frame is not None:
            uncorrelated_eligible_objects = len(eligible_objects)
            eligible_objects, correlation = motion_correlated_objects(
                frame,
                eligible_objects,
                qualification,
            )
            uncorrelated_eligible_objects -= len(eligible_objects)
            qualification["motion_correlation"] = correlation
        verification_candidate = bool(qualification.get("suppression_verification_candidate"))
        if qualification.get("borderline_candidate"):
            qualification["rescued_by_object"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)
        if verification_candidate:
            qualification["suppression_verification_rescued"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)

        snapshot_path = ""
        if frame is not None:
            snapshot_path = self.snapshot_writer(frame, event_at)

        if require_eligible_object and not eligible_objects:
            rejection_reason = (
                "object_not_motion_correlated"
                if require_motion_correlation and uncorrelated_eligible_objects > 0
                else "no_eligible_object"
            )
            return MotionDecisionOutcome(
                event_id=existing_event_id,
                snapshot_path=snapshot_path,
                object_detected=False if detection_completed else None,
                detected_objects=tuple(eligible_objects),
                rejection_reason=rejection_reason,
                motion_correlation=correlation,
                refinement_pending=bool(
                    getattr(provider_result, "refinement_pending", False)
                ),
                processing_timing=processing_timing,
                object_activity=activity_summary,
            )

        stored_objects = [
            *objects,
            {"status": "motion_qualification", "motion_qualification": qualification},
        ]
        if existing_event_id is None:
            event = self.events.add_event(
                camera_id=self.camera_id,
                kind="motion",
                topic=topic,
                message=message,
                snapshot_path=snapshot_path,
                recording_path=recording_path,
                objects_json=self.object_serializer(stored_objects),
                created_at=event_at.isoformat(),
            )
            event_id = int(event["id"])
        else:
            event_id = int(existing_event_id)
            self.events.refine_event_evidence(
                event_id,
                snapshot_path=snapshot_path,
                recording_path=recording_path,
                objects_json=self.object_serializer(stored_objects),
            )
        self._persist_face_candidates(
            event_id,
            event_at,
            tuple(getattr(provider_result, "face_candidates", ()) or ()),
        )
        if self.event_callback and existing_event_id is None:
            self._publish(
                "incident",
                IncidentCreated(
                    event_id=event_id,
                    camera_id=self.camera_id,
                    timestamp=event_at.isoformat(),
                ).to_payload(),
            )

        detected_objects = [detected for detected in stored_objects if detected.get("label")]
        if self.event_callback and detected_objects:
            self._publish(
                "object",
                ObjectDetected(
                    event_id=event_id,
                    camera_id=self.camera_id,
                    timestamp=event_at.isoformat(),
                    snapshot_path=snapshot_path,
                    recording_path=recording_path,
                    objects=tuple(detected_objects),
                    incident_objects=tuple(eligible_objects),
                ).to_payload(),
            )
        return MotionDecisionOutcome(
            event_id=event_id,
            snapshot_path=snapshot_path,
            object_detected=bool(eligible_objects) if detection_completed else None,
            detected_objects=tuple(eligible_objects),
            motion_correlation=correlation,
            refinement_pending=bool(
                getattr(provider_result, "refinement_pending", False)
            ),
            processing_timing=processing_timing,
            object_activity=activity_summary,
        )

    def _persist_face_candidates(
        self,
        event_id: int,
        event_at: datetime,
        candidates: tuple[FaceCandidate, ...],
    ) -> None:
        if self.face_candidate_sink is None or not candidates:
            return
        persisted: list[dict[str, Any]] = []
        for candidate in candidates:
            frame = candidate.frame
            height, width = frame.shape[:2]
            box = candidate.box
            face_width = float(box["x2"] - box["x1"])
            face_height = float(box["y2"] - box["y1"])
            pad_x, pad_y = face_width * 0.2, face_height * 0.2
            left = max(0, int(math.floor(box["x1"] - pad_x)))
            top = max(0, int(math.floor(box["y1"] - pad_y)))
            right = min(width, int(math.ceil(box["x2"] + pad_x)))
            bottom = min(height, int(math.ceil(box["y2"] + pad_y)))
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            candidate_at = event_at + timedelta(seconds=candidate.offset_seconds)
            path = self.snapshot_writer(crop, candidate_at)
            if not path:
                continue
            persisted.append({
                "snapshot_path": path,
                "box": {
                    "x1": float(box["x1"] - left),
                    "y1": float(box["y1"] - top),
                    "x2": float(box["x2"] - left),
                    "y2": float(box["y2"] - top),
                },
                "confidence": candidate.confidence,
                "track_id": candidate.track_id,
                "rank": candidate.rank,
                "offset_seconds": candidate.offset_seconds,
                "quality_score": candidate.quality_score,
                "quality": {
                    "sharpness": candidate.sharpness_score,
                    "exposure": candidate.exposure_score,
                    "edge_clearance": candidate.edge_clearance_ratio,
                },
                "detection_source": candidate.detection_source,
            })
        if persisted:
            try:
                self.face_candidate_sink(
                    event_id,
                    self.camera_id,
                    event_at.isoformat(),
                    persisted,
                )
            except Exception:
                LOGGER.exception(
                    "face candidate persistence failed for camera %s event %s",
                    self.camera_id,
                    event_id,
                )

    def record_audit(
        self,
        *,
        event_at: datetime,
        snapshot_path: str,
        mode: str,
        sensitivity: str,
        score: float,
        threshold: float,
        reason: str,
        object_detected: bool | None,
        trigger_count: int,
        features: dict[str, Any],
        category: str = "qualification",
        event_id: int | None = None,
        related_event_id: int | None = None,
        decision_id: str = "",
    ) -> dict[str, Any]:
        audit = self.events.add_motion_audit(
            event_id=event_id,
            related_event_id=related_event_id,
            decision_id=decision_id,
            camera_id=self.camera_id,
            snapshot_path=snapshot_path,
            created_at=event_at.isoformat(),
            mode=mode,
            sensitivity=sensitivity,
            score=score,
            threshold=threshold,
            reason=reason,
            object_detected=object_detected,
            trigger_count=trigger_count,
            features=features,
            category=category,
        )
        if self.event_callback:
            self._publish("motion_audit", audit)
        return audit

    def _publish(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(event_type, payload)
        except Exception:
            LOGGER.exception(
                "post-persistence %s notification failed for camera %s",
                event_type,
                self.camera_id,
            )


class MotionDecisionHandlerFactory:
    def __init__(
        self,
        events: MotionEventStore,
        object_serializer: MotionObjectSerializer,
        face_candidate_sink: FaceCandidateSink | None = None,
    ) -> None:
        self.events = events
        self.object_serializer = object_serializer
        self.face_candidate_sink = face_candidate_sink

    def create(
        self,
        camera_id: str,
        detection_provider: MotionDetectionProvider,
        snapshot_writer: MotionSnapshotWriter,
        initial_detection_provider: MotionDetectionProvider | None = None,
        event_callback: MotionEventCallback | None = None,
        activity_attributor: ObjectActivityAttributor | None = None,
    ) -> MotionDecisionHandler:
        return MotionDecisionHandler(
            camera_id=camera_id,
            events=self.events,
            detection_provider=detection_provider,
            initial_detection_provider=initial_detection_provider,
            snapshot_writer=snapshot_writer,
            object_serializer=self.object_serializer,
            event_callback=event_callback,
            activity_attributor=activity_attributor,
            face_candidate_sink=self.face_candidate_sink,
        )
