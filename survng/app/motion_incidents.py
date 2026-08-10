from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import copy
import logging
import queue
import threading
from typing import Any, Callable, Protocol

import numpy as np

from .motion_pipeline.decision_handler import MotionDecisionOutcome
from .security import redact_secret_text


LOGGER = logging.getLogger(__name__)


class MotionDecisionProcessor(Protocol):
    def activity_status(self) -> dict[str, Any]: ...

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
    ) -> MotionDecisionOutcome: ...


TrackingPrewarmer = Callable[[], object | None]
ImageReader = Callable[[str], np.ndarray | None]
TrackingEnabled = Callable[[], bool]
TrackableObjects = Callable[[list[dict[str, Any]]], bool]
TrackingStarter = Callable[
    [int, datetime, list[dict[str, Any]], np.ndarray | None],
    bool | None,
]
RefinementCallback = Callable[[MotionDecisionOutcome], None]


def _compact_refinement_qualification(
    qualification: dict[str, Any],
) -> dict[str, Any]:
    """Copy decision evidence without repeating the large pipeline graph."""
    compact = copy.deepcopy(qualification)
    telemetry = compact.get("telemetry")
    if isinstance(telemetry, dict):
        compact["telemetry"] = {
            "schema_version": telemetry.get("schema_version"),
            "origins": copy.deepcopy(telemetry.get("origins", {})),
            "compacted_for_refinement": True,
        }
    compact["refinement_payload_compacted"] = True
    return compact


@dataclass(frozen=True, slots=True)
class _RefinementJob:
    topic: str
    message: str
    event_at: datetime
    qualification: dict[str, Any]
    existing_event_id: int | None
    require_eligible_object: bool
    require_motion_correlation: bool
    callback: RefinementCallback | None
    initial_outcome: MotionDecisionOutcome

    def key(self) -> tuple[str, int | float]:
        if self.existing_event_id is not None:
            return ("event", self.existing_event_id)
        sequence = self.qualification.get("motion_episode_sequence")
        if isinstance(sequence, int):
            return ("episode", sequence)
        return ("time", round(self.event_at.timestamp(), 3))


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
        self._refinement_queue: queue.Queue[_RefinementJob | None] = queue.Queue(maxsize=3)
        self._pending_refinement_keys: set[tuple[str, int | float]] = set()
        self._refinement_thread: threading.Thread | None = None
        self._refinement_stop: threading.Event | None = None
        self._refinements_queued = 0
        self._refinements_completed = 0
        self._refinements_dropped = 0
        self._refinements_coalesced = 0
        self._refinements_superseded = 0
        self._refinement_failures = 0
        self._last_refinement_failure: dict[str, Any] | None = None
        self._refinement_callback_failures = 0
        self._last_refinement_callback_failure: dict[str, Any] | None = None
        self._timing_samples = 0
        self._timing_totals_ms: dict[str, float] = {}
        self._timing_counts: dict[str, int] = {}
        self._last_timing_ms: dict[str, float] = {}

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
                "refinements_queued": self._refinements_queued,
                "refinements_completed": self._refinements_completed,
                "refinements_dropped": self._refinements_dropped,
                "refinements_coalesced": self._refinements_coalesced,
                "refinements_superseded": self._refinements_superseded,
                "refinement_failures": self._refinement_failures,
                "last_refinement_failure": (
                    dict(self._last_refinement_failure)
                    if self._last_refinement_failure is not None
                    else None
                ),
                "refinement_callback_failures": self._refinement_callback_failures,
                "last_refinement_callback_failure": (
                    dict(self._last_refinement_callback_failure)
                    if self._last_refinement_callback_failure is not None
                    else None
                ),
                "refinement_queue_depth": self._refinement_queue.qsize(),
                "refinement_pending_episodes": len(self._pending_refinement_keys),
                "refinement_worker_alive": bool(
                    self._refinement_thread and self._refinement_thread.is_alive()
                ),
                "object_detection_timing": {
                    "samples": self._timing_samples,
                    "last_ms": dict(self._last_timing_ms),
                    "average_ms": {
                        key: round(value / max(1, self._timing_counts.get(key, 0)), 3)
                        for key, value in self._timing_totals_ms.items()
                    },
                },
                "object_activity_attribution": self.decision_processor.activity_status(),
            }

    def start(self, stop_event: threading.Event) -> None:
        if self._refinement_thread is not None and self._refinement_thread.is_alive():
            return
        self._refinement_stop = stop_event
        thread = threading.Thread(
            target=self._run_refinements,
            name=f"motion-refine-{self.camera_id}",
            daemon=False,
        )
        self._refinement_thread = thread
        thread.start()

    def request_stop(self) -> None:
        try:
            self._refinement_queue.put_nowait(None)
        except queue.Full:
            try:
                self._refinement_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._refinement_queue.put_nowait(None)
            except queue.Full:
                pass

    def wait_stopped(self, timeout: float) -> bool:
        thread = self._refinement_thread
        if thread is None:
            return True
        thread.join(max(0.0, timeout))
        if thread.is_alive():
            return False
        self._refinement_thread = None
        self._refinement_stop = None
        self._clear_refinements()
        return True

    def running(self) -> bool:
        return bool(self._refinement_thread and self._refinement_thread.is_alive())

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
        refinement_callback: RefinementCallback | None = None,
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
        self._record_timing(outcome)
        if outcome.refinement_pending:
            refinement_admission = self._queue_refinement(
                _RefinementJob(
                    topic=topic,
                    message=message,
                    event_at=event_at,
                    qualification=_compact_refinement_qualification(qualification),
                    existing_event_id=outcome.event_id,
                    require_eligible_object=require_eligible_object,
                    require_motion_correlation=require_motion_correlation,
                    callback=refinement_callback,
                    initial_outcome=outcome,
                )
            )
            if refinement_admission == "dropped":
                # Initial detection is already valid evidence. Capacity loss
                # must not also discard its tracking handoff.
                self._handoff(outcome, event_at)
        else:
            self._handoff(outcome, event_at)
        return outcome

    def _handoff(self, outcome: MotionDecisionOutcome, event_at: datetime) -> None:
        if outcome.event_id is None or not outcome.object_detected:
            return

        try:
            detected_objects = list(outcome.detected_objects)
            if not self.has_trackable_objects(detected_objects):
                return
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
                return
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

    def _queue_refinement(self, job: _RefinementJob) -> str:
        if not self.running():
            with self._status_lock:
                self._refinements_dropped += 1
            LOGGER.warning("motion refinement worker unavailable for %s", self.camera_id)
            return "dropped"
        superseded: _RefinementJob | None = None
        with self._status_lock:
            key = job.key()
            if key in self._pending_refinement_keys:
                self._refinements_coalesced += 1
                LOGGER.debug(
                    "coalesced duplicate motion refinement for %s %s",
                    self.camera_id,
                    key,
                )
                return "coalesced"
            try:
                self._refinement_queue.put_nowait(job)
            except queue.Full:
                removed = False
                try:
                    candidate = self._refinement_queue.get_nowait()
                    removed = True
                except queue.Empty:
                    candidate = None
                if removed and candidate is None:
                    # Preserve a shutdown sentinel; admission is already closed.
                    try:
                        self._refinement_queue.put_nowait(None)
                    except queue.Full:
                        pass
                    self._refinements_dropped += 1
                    return "dropped"
                if candidate is not None:
                    superseded = candidate
                    self._pending_refinement_keys.discard(candidate.key())
                try:
                    self._refinement_queue.put_nowait(job)
                except queue.Full:
                    self._refinements_dropped += 1
                    return "dropped"
                if superseded is not None:
                    self._refinements_superseded += 1
            self._pending_refinement_keys.add(key)
            self._refinements_queued += 1
        if superseded is not None:
            # Refinement is optional enrichment. Preserve the displaced
            # decision's already-valid tracking handoff.
            self._handoff(superseded.initial_outcome, superseded.event_at)
            LOGGER.info(
                "superseded stale motion refinement for %s %s with %s",
                self.camera_id,
                superseded.key(),
                job.key(),
            )
        return "queued"

    def _run_refinements(self) -> None:
        while True:
            stop = self._refinement_stop
            if stop is None or stop.is_set():
                return
            try:
                job = self._refinement_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None or stop.is_set():
                return
            try:
                try:
                    outcome = self.decision_processor.refine(
                        job.topic,
                        job.message,
                        job.event_at,
                        job.qualification,
                        existing_event_id=job.existing_event_id,
                        require_eligible_object=job.require_eligible_object,
                        require_motion_correlation=job.require_motion_correlation,
                    )
                except Exception as error:
                    failure = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": redact_secret_text(error)[:500],
                        "error_type": type(error).__name__,
                        "event_at": job.event_at.isoformat(),
                        "existing_event_id": job.existing_event_id,
                    }
                    with self._status_lock:
                        self._refinement_failures += 1
                        self._last_refinement_failure = failure
                    self._handoff(job.initial_outcome, job.event_at)
                    LOGGER.exception(
                        "motion refinement failed for %s: %s: %s",
                        self.camera_id,
                        type(error).__name__,
                        redact_secret_text(error)[:500],
                    )
                    continue

                self._record_timing(outcome)
                self._handoff(outcome, job.event_at)
                if job.callback is not None and not stop.is_set():
                    try:
                        job.callback(outcome)
                    except Exception as error:
                        failure = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "error": redact_secret_text(error)[:500],
                            "error_type": type(error).__name__,
                            "event_at": job.event_at.isoformat(),
                            "existing_event_id": job.existing_event_id,
                        }
                        with self._status_lock:
                            self._refinement_callback_failures += 1
                            self._last_refinement_callback_failure = failure
                        LOGGER.exception(
                            "motion refinement callback failed for %s: %s: %s",
                            self.camera_id,
                            type(error).__name__,
                            redact_secret_text(error)[:500],
                        )
                with self._status_lock:
                    self._refinements_completed += 1
            finally:
                with self._status_lock:
                    self._pending_refinement_keys.discard(job.key())

    def _record_timing(self, outcome: MotionDecisionOutcome) -> None:
        timing = outcome.processing_timing
        if not isinstance(timing, dict):
            return
        flattened: dict[str, float] = {}
        for key in ("workflow_ms", "decision_queue_wait_ms"):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                flattened[key] = max(0.0, float(value))
        phases = timing.get("phases_ms")
        if isinstance(phases, dict):
            for key, value in phases.items():
                if isinstance(value, (int, float)):
                    flattened[str(key)] = max(0.0, float(value))
        if not flattened:
            return
        with self._status_lock:
            self._timing_samples += 1
            self._last_timing_ms = dict(flattened)
            for key, value in flattened.items():
                self._timing_totals_ms[key] = self._timing_totals_ms.get(key, 0.0) + value
                self._timing_counts[key] = self._timing_counts.get(key, 0) + 1

    def _clear_refinements(self) -> None:
        while True:
            try:
                self._refinement_queue.get_nowait()
            except queue.Empty:
                with self._status_lock:
                    self._pending_refinement_keys.clear()
                return
