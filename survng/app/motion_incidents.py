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


class TrackingConfig(Protocol):
    enabled: bool

    def tracks_label(self, label: object) -> bool: ...


class IncidentTrackingSession(Protocol):
    config: TrackingConfig

    def start(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None = None,
    ) -> bool: ...


TrackingPrewarmer = Callable[[], object | None]
ImageReader = Callable[[str], np.ndarray | None]
TrackingProvider = Callable[[], IncidentTrackingSession]


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
        tracking_provider: TrackingProvider,
        prewarm_tracking: TrackingPrewarmer,
        image_reader: ImageReader,
    ) -> None:
        self.camera_id = camera_id
        self.decision_processor = decision_processor
        self.tracking_provider = tracking_provider
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
            if self.tracking_provider().config.enabled:
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
            tracking = self.tracking_provider()
            if not tracking.config.enabled:
                return outcome
            trackable_objects = [
                item
                for item in outcome.detected_objects
                if (
                    item.get("label")
                    and item.get("incident_eligible") is not False
                    and tracking.config.tracks_label(item.get("label"))
                )
            ]
            if not trackable_objects:
                return outcome
            initial_frame = None
            if outcome.snapshot_path:
                initial_frame = self.image_reader(outcome.snapshot_path)
            started = tracking.start(
                outcome.event_id,
                event_at,
                trackable_objects,
                initial_frame,
            )
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
