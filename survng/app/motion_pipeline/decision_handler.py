from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable, Protocol

from ..detector import detection_failure
from .context import Frame


MotionDetectionProvider = Callable[[datetime], tuple[Frame | None, list[dict[str, Any]], str]]
MotionSnapshotWriter = Callable[[Frame, datetime], str]
MotionEventCallback = Callable[[str, dict[str, Any]], None]
MotionObjectSerializer = Callable[[list[dict[str, Any]]], str]


LOGGER = logging.getLogger(__name__)


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "snapshot_path": self.snapshot_path,
            "object_detected": self.object_detected,
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
        verification_candidate = bool(qualification.get("suppression_verification_candidate"))
        if qualification.get("borderline_candidate"):
            qualification["rescued_by_object"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)
        if verification_candidate:
            qualification["suppression_verification_rescued"] = bool(eligible_objects)
            qualification["effective_accepted"] = bool(eligible_objects)
            qualification["would_suppress"] = not bool(eligible_objects)

        if require_eligible_object and not eligible_objects:
            return MotionDecisionOutcome(
                event_id=None,
                snapshot_path="",
                object_detected=False if detection_completed else None,
                detected_objects=tuple(eligible_objects),
            )

        snapshot_path = ""
        if frame is not None:
            snapshot_path = self.snapshot_writer(frame, event_at)

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
