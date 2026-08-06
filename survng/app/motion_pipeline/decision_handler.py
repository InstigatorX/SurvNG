from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import time
from typing import Any, Callable, Protocol

from ..detector import detection_failure
from .context import Frame


MotionDetectionProvider = Callable[[datetime], tuple[Frame | None, list[dict[str, Any]], str]]
MotionSnapshotWriter = Callable[[Frame, datetime], str]
MotionEventCallback = Callable[[str, dict[str, Any]], None]
MotionObjectSerializer = Callable[[list[dict[str, Any]]], str]


LOGGER = logging.getLogger(__name__)
MOTION_REGION_MARGIN_RATIO = 0.035
TEMPORAL_OBJECT_MOVEMENT_RATIO = 0.02
MINIMUM_TEMPORAL_OBJECT_MOVEMENT_RATIO = 0.003
TEMPORAL_OBJECT_BOX_MOVEMENT_SCALE = 0.04


def _detection_box_ratio(
    detected: dict[str, Any],
    frame: Frame,
) -> tuple[float, float, float, float] | None:
    box = detected.get("box")
    if not isinstance(box, dict):
        return None
    try:
        height, width = frame.shape[:2]
        x1 = float(box["x1"]) / width
        y1 = float(box["y1"]) / height
        x2 = float(box["x2"]) / width
        y2 = float(box["y2"]) / height
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


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


def _temporal_movement_threshold(
    box: tuple[float, float, float, float] | None,
) -> float:
    if box is None:
        return TEMPORAL_OBJECT_MOVEMENT_RATIO
    x1, y1, x2, y2 = box
    object_diagonal = math.hypot(x2 - x1, y2 - y1)
    return min(
        TEMPORAL_OBJECT_MOVEMENT_RATIO,
        max(
            MINIMUM_TEMPORAL_OBJECT_MOVEMENT_RATIO,
            object_diagonal * TEMPORAL_OBJECT_BOX_MOVEMENT_SCALE,
        ),
    )


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
    for detected in objects:
        box = _detection_box_ratio(detected, frame)
        spatial = bool(box is not None and _intersects_motion_region(box, regions))
        try:
            displacement = float(detected.get("temporal_center_displacement_ratio") or 0.0)
        except (TypeError, ValueError):
            displacement = 0.0
        try:
            path = float(detected.get("temporal_center_path_ratio") or 0.0)
        except (TypeError, ValueError):
            path = 0.0
        movement_threshold = _temporal_movement_threshold(box)
        temporal_evidence_available = bool(
            int(detected.get("temporal_track_observations") or 0) >= 2
        )
        newly_appeared = bool(detected.get("temporal_newly_appeared"))
        temporal = displacement >= movement_threshold
        spatial_fallback = spatial and not temporal_evidence_available
        appearance_match = spatial and newly_appeared
        # A short recorded sequence can begin and end at nearly the same point
        # while a real object walks through the EMA region.  Permit that only
        # when spatial evidence also agrees and the travelled path is well
        # beyond ordinary detector-box jitter.  A stationary object therefore
        # still cannot explain unrelated motion beside or behind it.
        spatial_path = bool(
            spatial
            and temporal_evidence_available
            and path >= max(0.01, movement_threshold * 2.5)
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
        detected["motion_correlation_threshold"] = round(movement_threshold, 5)
        detected["motion_temporal_evidence_available"] = temporal_evidence_available
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
            stationary_spatial_rejections += int(spatial and temporal_evidence_available)
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
        "minimum_temporal_movement_ratio": TEMPORAL_OBJECT_MOVEMENT_RATIO,
        "adaptive_minimum_temporal_movement_ratio": MINIMUM_TEMPORAL_OBJECT_MOVEMENT_RATIO,
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


@dataclass(frozen=True, slots=True)
class MotionDecisionOutcome:
    event_id: int | None
    snapshot_path: str
    object_detected: bool | None
    detected_objects: tuple[dict[str, Any], ...] = ()
    rejection_reason: str = ""
    motion_correlation: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "snapshot_path": self.snapshot_path,
            "object_detected": self.object_detected,
            "rejection_reason": self.rejection_reason,
            "motion_correlation": self.motion_correlation,
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
        event_callback: MotionEventCallback | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.events = events
        self.detection_provider = detection_provider
        self.snapshot_writer = snapshot_writer
        self.object_serializer = object_serializer
        self.event_callback = event_callback

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
        detection_started = time.monotonic()
        frame, objects, recording_path = self.detection_provider(event_at)
        processed_at = datetime.now(timezone.utc)
        normalized_event_at = (
            event_at.replace(tzinfo=timezone.utc)
            if event_at.tzinfo is None
            else event_at.astimezone(timezone.utc)
        )
        qualification["object_detection_duration_ms"] = round(
            (time.monotonic() - detection_started) * 1000,
            3,
        )
        qualification["processed_at"] = processed_at.isoformat()
        qualification["event_processing_delay_seconds"] = round(
            max(0.0, (processed_at - normalized_event_at).total_seconds()),
            3,
        )
        if frame is None:
            objects = [{"status": "no_recorded_frame"}]

        detection_completed = frame is not None and not detection_failure(objects)

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
                event_id=None,
                snapshot_path=snapshot_path,
                object_detected=False if detection_completed else None,
                detected_objects=tuple(eligible_objects),
                rejection_reason=rejection_reason,
                motion_correlation=correlation,
            )

        stored_objects = [
            *objects,
            {"status": "motion_qualification", "motion_qualification": qualification},
        ]
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
        if self.event_callback:
            self._publish(
                "incident",
                {
                    "event_id": event_id,
                    "camera_id": self.camera_id,
                    "timestamp": event_at.isoformat(),
                    "kind": "motion",
                },
            )

        detected_objects = [detected for detected in stored_objects if detected.get("label")]
        if self.event_callback and detected_objects:
            self._publish(
                "object",
                {
                    "event_id": event_id,
                    "camera_id": self.camera_id,
                    "timestamp": event_at.isoformat(),
                    "snapshot_path": snapshot_path,
                    "recording_path": recording_path,
                    "objects": detected_objects,
                    "incident_objects": eligible_objects,
                },
            )
        return MotionDecisionOutcome(
            event_id=event_id,
            snapshot_path=snapshot_path,
            object_detected=bool(eligible_objects) if detection_completed else None,
            detected_objects=tuple(eligible_objects),
            motion_correlation=correlation,
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
    ) -> None:
        self.events = events
        self.object_serializer = object_serializer

    def create(
        self,
        camera_id: str,
        detection_provider: MotionDetectionProvider,
        snapshot_writer: MotionSnapshotWriter,
        event_callback: MotionEventCallback | None = None,
    ) -> MotionDecisionHandler:
        return MotionDecisionHandler(
            camera_id=camera_id,
            events=self.events,
            detection_provider=detection_provider,
            snapshot_writer=snapshot_writer,
            object_serializer=self.object_serializer,
            event_callback=event_callback,
        )
