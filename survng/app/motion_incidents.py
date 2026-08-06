from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable, Protocol

import numpy as np

from .motion_pipeline.decision_handler import MotionDecisionOutcome


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
        tracking: IncidentTrackingSession,
        prewarm_tracking: TrackingPrewarmer,
        image_reader: ImageReader,
    ) -> None:
        self.camera_id = camera_id
        self.decision_processor = decision_processor
        self.tracking = tracking
        self.prewarm_tracking = prewarm_tracking
        self.image_reader = image_reader

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
        if self.tracking.config.enabled:
            # Start opening main concurrently with recorded validation so the
            # tracking handoff does not pay another RTSP startup delay.
            self.prewarm_tracking()
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
            initial_frame = None
            if self.tracking.config.enabled and outcome.snapshot_path:
                initial_frame = self.image_reader(outcome.snapshot_path)
            self.tracking.start(
                outcome.event_id,
                event_at,
                list(outcome.detected_objects),
                initial_frame,
            )
        except Exception:
            LOGGER.exception(
                "post-persistence tracking handoff failed for %s event %d",
                self.camera_id,
                outcome.event_id,
            )
        return outcome
