from __future__ import annotations

from datetime import datetime, timezone
import logging
import threading
from typing import Any, Callable, Protocol

import numpy as np

from .motion_pipeline.decision_handler import MotionDecisionOutcome
from .security import redact_secret_text


LOGGER = logging.getLogger(__name__)


class MotionDecisionProcessor(Protocol):
    def handle(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome: ...


TrackingPrewarmer = Callable[[], object | None]
ImageReader = Callable[[str], np.ndarray | None]
TrackingEnabled = Callable[[], bool]
TrackableObjects = Callable[[list[dict[str, Any]]], bool]
TrackingStarter = Callable[
    [int, datetime, list[dict[str, Any]], np.ndarray | None],
    bool | None,
]


class MotionIncidentService:
    """Persists a qualified incident and hands durable results to tracking.

    Detection and persistence remain the decision processor's responsibility.
    Tracking is a best-effort post-persistence consumer: its failure must never
    cause the motion worker to replay detection and create a duplicate incident.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        decision_processor: MotionDecisionProcessor,
        tracking_enabled: TrackingEnabled,
        has_trackable_objects: TrackableObjects,
        start_tracking: TrackingStarter,
        prewarm_tracking: TrackingPrewarmer,
        image_reader: ImageReader,
    ) -> None:
        self.camera_id = camera_id
        self.decision_processor = decision_processor
        self.tracking_enabled = tracking_enabled
        self.has_trackable_objects = has_trackable_objects
        self.start_tracking = start_tracking
        self.prewarm_tracking = prewarm_tracking
        self.image_reader = image_reader
        self._status_lock = threading.Lock()
        self._prewarm_failures = 0
        self._last_prewarm_failure: dict[str, Any] | None = None
        self._handoff_failures = 0
        self._last_handoff_failure: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return {
                "prewarm_failures": self._prewarm_failures,
                "last_prewarm_failure": (
                    dict(self._last_prewarm_failure)
                    if self._last_prewarm_failure is not None
                    else None
                ),
                "handoff_failures": self._handoff_failures,
                "last_handoff_failure": (
                    dict(self._last_handoff_failure)
                    if self._last_handoff_failure is not None
                    else None
                ),
            }

    def _record_prewarm_failure(self, error: Exception) -> None:
        error_text = redact_secret_text(error)[:500]
        with self._status_lock:
            self._prewarm_failures += 1
            self._last_prewarm_failure = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_text,
                "error_type": type(error).__name__,
            }

    def _record_handoff_failure(
        self,
        event_id: int,
        error_type: str,
        error: object,
    ) -> None:
        error_text = redact_secret_text(error)[:500]
        with self._status_lock:
            self._handoff_failures += 1
            self._last_handoff_failure = {
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_text,
                "error_type": error_type,
            }

    def process(
        self,
        topic: str,
        message: str,
        event_at: datetime,
        qualification: dict[str, Any],
        *,
        require_eligible_object: bool = False,
        require_motion_correlation: bool = False,
    ) -> MotionDecisionOutcome:
        try:
            if self.tracking_enabled():
                # Start opening main concurrently with recorded validation so
                # the tracking handoff does not pay another RTSP startup delay.
                self.prewarm_tracking()
        except Exception as error:
            # Prewarming is an optimization. It must never prevent the core
            # detection and incident-persistence path from running.
            self._record_prewarm_failure(error)
            LOGGER.warning(
                "tracking prewarm failed for %s: %s: %s",
                self.camera_id,
                type(error).__name__,
                redact_secret_text(error)[:500],
            )
        outcome = self.decision_processor.handle(
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )
        if outcome.event_id is None or not outcome.object_detected:
            return outcome

        try:
            detected_objects = list(outcome.detected_objects)
            if not self.has_trackable_objects(detected_objects):
                return outcome
            initial_frame = None
            if outcome.snapshot_path:
                initial_frame = self.image_reader(outcome.snapshot_path)
            started = self.start_tracking(
                outcome.event_id,
                event_at,
                detected_objects,
                initial_frame,
            )
            if started is None:
                return outcome
            if not started:
                self._record_handoff_failure(
                    outcome.event_id,
                    "TrackingDeclined",
                    "tracking session declined the incident",
                )
                LOGGER.warning(
                    "post-persistence tracking handoff was declined for %s event %d",
                    self.camera_id,
                    outcome.event_id,
                )
        except Exception as error:
            self._record_handoff_failure(
                outcome.event_id,
                type(error).__name__,
                error,
            )
            LOGGER.error(
                "post-persistence tracking handoff failed for %s event %d: %s: %s",
                self.camera_id,
                outcome.event_id,
                type(error).__name__,
                redact_secret_text(error)[:500],
            )
        return outcome
