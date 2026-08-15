from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import copy
from collections import deque
import hashlib
import json
import logging
import queue
import threading
import time
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


class DetectionJobStore(Protocol):
    def enqueue_detection_job(
        self, *, job_id: str, camera_id: str, dedupe_key: str,
        payload: dict[str, Any],
    ) -> str: ...
    def claim_detection_job(
        self, camera_id: str, *, lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None: ...
    def complete_detection_job(self, job_id: str, event_id: int | None) -> None: ...
    def retry_detection_job(
        self, job_id: str, error: str, *, retry_delay_seconds: float = 2.0,
        maximum_attempts: int = 5,
    ) -> bool: ...
    def detection_job_status(self, camera_id: str) -> dict[str, int]: ...

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

    def dedupe_key(self) -> str:
        kind, value = self.key()
        return f"{kind}:{value}"

    def job_id(self, camera_id: str) -> str:
        return hashlib.sha256(
            f"{camera_id}\0{self.dedupe_key()}".encode("utf-8")
        ).hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "message": self.message,
            "event_at": self.event_at.isoformat(),
            "qualification": json.loads(json.dumps(self.qualification, default=str)),
            "existing_event_id": self.existing_event_id,
            "require_eligible_object": self.require_eligible_object,
            "require_motion_correlation": self.require_motion_correlation,
            "initial_outcome": {
                **self.initial_outcome.as_dict(),
                "detected_objects": list(self.initial_outcome.detected_objects),
            },
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        callback: RefinementCallback | None,
    ) -> "_RefinementJob":
        initial = dict(payload.get("initial_outcome") or {})
        return cls(
            topic=str(payload.get("topic") or ""),
            message=str(payload.get("message") or ""),
            event_at=datetime.fromisoformat(str(payload["event_at"])),
            qualification=dict(payload.get("qualification") or {}),
            existing_event_id=(
                int(payload["existing_event_id"])
                if payload.get("existing_event_id") is not None else None
            ),
            require_eligible_object=bool(payload.get("require_eligible_object")),
            require_motion_correlation=bool(payload.get("require_motion_correlation")),
            callback=callback,
            initial_outcome=MotionDecisionOutcome(
                event_id=(int(initial["event_id"]) if initial.get("event_id") is not None else None),
                snapshot_path=str(initial.get("snapshot_path") or ""),
                object_detected=initial.get("object_detected"),
                detected_objects=tuple(initial.get("detected_objects") or ()),
                rejection_reason=str(initial.get("rejection_reason") or ""),
                motion_correlation=initial.get("motion_correlation"),
                refinement_pending=bool(initial.get("refinement_pending")),
                processing_timing=initial.get("processing_timing"),
                object_activity=initial.get("object_activity"),
            ),
        )


class _MemoryDetectionJobStore:
    """Test/local fallback with the same no-capacity-drop semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def enqueue_detection_job(self, *, job_id, camera_id, dedupe_key, payload):
        with self._lock:
            if job_id in self._jobs:
                return "coalesced"
            self._jobs[job_id] = {
                "id": job_id, "camera_id": camera_id, "dedupe_key": dedupe_key,
                "payload": copy.deepcopy(payload), "state": "queued", "attempts": 0,
                "available_at": time.monotonic(),
            }
            return "queued"

    def claim_detection_job(self, camera_id, *, lease_seconds=60.0):
        del lease_seconds
        with self._lock:
            for job in self._jobs.values():
                if (
                    job["camera_id"] == camera_id
                    and job["state"] == "queued"
                    and job["available_at"] <= time.monotonic()
                ):
                    job["state"] = "running"
                    job["attempts"] += 1
                    return copy.deepcopy(job)
        return None

    def complete_detection_job(self, job_id, event_id):
        with self._lock:
            self._jobs[job_id]["state"] = "completed"
            self._jobs[job_id]["event_id"] = event_id

    def retry_detection_job(
        self, job_id, error, *, retry_delay_seconds=2.0, maximum_attempts=5,
    ):
        del error
        with self._lock:
            job = self._jobs[job_id]
            retry = job["attempts"] < maximum_attempts
            job["state"] = "queued" if retry else "failed"
            job["available_at"] = time.monotonic() + retry_delay_seconds
            return retry

    def detection_job_status(self, camera_id):
        with self._lock:
            result: dict[str, int] = {}
            for job in self._jobs.values():
                if job["camera_id"] == camera_id:
                    result[job["state"]] = result.get(job["state"], 0) + 1
            return result


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
        refinement_store: DetectionJobStore | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.decision_processor = decision_processor
        self.tracking_enabled = tracking_enabled
        self.has_trackable_objects = has_trackable_objects
        self.start_tracking = start_tracking
        self.prewarm_tracking = prewarm_tracking
        self.image_reader = image_reader
        self.refinement_store = refinement_store or _MemoryDetectionJobStore()
        self._status_lock = threading.Lock()
        self._prewarm_failures = 0
        self._last_prewarm_failure: dict[str, Any] | None = None
        self._handoff_failures = 0
        self._last_handoff_failure: dict[str, Any] | None = None
        self._handoff_lock = threading.Lock()
        self._handed_off_event_ids: set[int] = set()
        self._handoff_event_order: deque[int] = deque()
        self._refinement_queue: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._refinement_callbacks: dict[str, RefinementCallback] = {}
        self._refinement_thread: threading.Thread | None = None
        self._refinement_stop: threading.Event | None = None
        self._refinement_accepting = False
        self._refinements_queued = 0
        self._refinements_completed = 0
        self._refinements_coalesced = 0
        self._refinement_failures = 0
        self._last_refinement_failure: dict[str, Any] | None = None
        self._refinement_callback_failures = 0
        self._last_refinement_callback_failure: dict[str, Any] | None = None
        self._timing_samples = 0
        self._timing_totals_ms: dict[str, float] = {}
        self._timing_counts: dict[str, int] = {}
        self._last_timing_ms: dict[str, float] = {}

    def status(self) -> dict[str, Any]:
        durable = self.refinement_store.detection_job_status(self.camera_id)
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
                "refinements_coalesced": self._refinements_coalesced,
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
                "refinement_queue_depth": int(durable.get("queued", 0)),
                "refinement_pending_episodes": int(durable.get("queued", 0))
                + int(durable.get("running", 0)),
                "refinement_durable": True,
                "refinement_jobs": durable,
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
        with self._status_lock:
            if self._refinement_thread is not None and self._refinement_thread.is_alive():
                return
            self._refinement_stop = stop_event
            self._refinement_accepting = not stop_event.is_set()
            thread = threading.Thread(
                target=self._run_refinements,
                name=f"motion-refine-{self.camera_id}",
                daemon=False,
            )
            self._refinement_thread = thread
        try:
            thread.start()
        except Exception:
            with self._status_lock:
                if self._refinement_thread is thread:
                    self._refinement_thread = None
                    self._refinement_stop = None
                    self._refinement_accepting = False
            raise

    def request_stop(self) -> None:
        # Close admission before publishing the sentinel. Otherwise a producer
        # can enqueue behind the sentinel while the worker is still alive and
        # lose the initial tracking handoff when shutdown drains that job.
        with self._status_lock:
            self._refinement_accepting = False
        try:
            self._refinement_queue.put_nowait(True)
        except queue.Full:
            pass

    def wait_stopped(self, timeout: float) -> bool:
        thread = self._refinement_thread
        if thread is None:
            return True
        thread.join(max(0.0, timeout))
        if thread.is_alive():
            return False
        with self._status_lock:
            self._refinement_thread = None
            self._refinement_stop = None
            self._refinement_accepting = False
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
        outcome = self.decision_processor.handle(
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )
        self._record_timing(outcome)
        refinement_admission = "not_needed"
        if outcome.refinement_pending:
            # Mandatory delayed discovery is admitted before any optional
            # capture prewarm or tracking handoff can consume time/resources.
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
        # Strong provisional evidence can begin tracking immediately. Handoff
        # is idempotent per event, so later refined evidence cannot duplicate it.
        if not outcome.refinement_pending or outcome.object_detected is True:
            self._handoff(outcome, event_at)
        try:
            if self.tracking_enabled():
                # Main-stream startup is optional enrichment. It must happen
                # only after the protected initial inference and persistence
                # path has completed.
                self.prewarm_tracking()
        except Exception as error:
            self._record_prewarm_failure(error)
            LOGGER.warning(
                "tracking prewarm failed for %s: %s: %s",
                self.camera_id,
                type(error).__name__,
                redact_secret_text(error)[:500],
            )
        if outcome.refinement_pending and refinement_admission == "dropped":
            # Initial detection is already valid evidence. Capacity loss must
            # not also discard its tracking handoff.
            self._handoff(outcome, event_at)
        return outcome

    def _handoff(self, outcome: MotionDecisionOutcome, event_at: datetime) -> bool:
        if outcome.event_id is None or not outcome.object_detected:
            return False

        with self._handoff_lock:
            event_id = int(outcome.event_id)
            if event_id in self._handed_off_event_ids:
                return True
            try:
                detected_objects = list(outcome.detected_objects)
                if not self.has_trackable_objects(detected_objects):
                    return False
                initial_frame = None
                if outcome.snapshot_path:
                    initial_frame = self.image_reader(outcome.snapshot_path)
                started = self.start_tracking(
                    event_id,
                    event_at,
                    detected_objects,
                    initial_frame,
                )
                if started is None:
                    return False
                if started:
                    self._handed_off_event_ids.add(event_id)
                    self._handoff_event_order.append(event_id)
                    while len(self._handoff_event_order) > 256:
                        expired = self._handoff_event_order.popleft()
                        self._handed_off_event_ids.discard(expired)
                    return True
                self._record_handoff_failure(
                    event_id,
                    "TrackingDeclined",
                    "tracking session declined the incident",
                )
                LOGGER.warning(
                    "post-persistence tracking handoff was declined for %s event %d",
                    self.camera_id,
                    event_id,
                )
                return False
            except Exception as error:
                self._record_handoff_failure(
                    event_id,
                    type(error).__name__,
                    error,
                )
                LOGGER.error(
                    "post-persistence tracking handoff failed for %s event %d: %s: %s",
                    self.camera_id,
                    event_id,
                    type(error).__name__,
                    redact_secret_text(error)[:500],
                )
                return False

    def _queue_refinement(self, job: _RefinementJob) -> str:
        with self._status_lock:
            stop = self._refinement_stop
            worker_available = bool(
                self._refinement_accepting
                and stop is not None
                and not stop.is_set()
                and self._refinement_thread is not None
                and self._refinement_thread.is_alive()
            )
            job_id = job.job_id(self.camera_id)
            admission = self.refinement_store.enqueue_detection_job(
                job_id=job_id,
                camera_id=self.camera_id,
                dedupe_key=job.dedupe_key(),
                payload=job.payload(),
            )
            if admission == "coalesced":
                self._refinements_coalesced += 1
                return admission
            if job.callback is not None:
                self._refinement_callbacks[job_id] = job.callback
            self._refinements_queued += 1
        if worker_available:
            try:
                self._refinement_queue.put_nowait(True)
            except queue.Full:
                pass
        return "queued"

    def _run_refinements(self) -> None:
        while True:
            stop = self._refinement_stop
            if stop is None or stop.is_set():
                return
            claimed = self.refinement_store.claim_detection_job(self.camera_id)
            if claimed is None:
                try:
                    self._refinement_queue.get(timeout=0.5)
                except queue.Empty:
                    pass
                continue
            job_id = str(claimed["id"])
            with self._status_lock:
                callback = self._refinement_callbacks.get(job_id)
            job = _RefinementJob.from_payload(dict(claimed["payload"]), callback)
            job.qualification["detection_intent_id"] = job_id
            completed = False
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
                    retrying = self.refinement_store.retry_detection_job(
                        job_id,
                        failure["error"],
                    )
                    self._handoff(job.initial_outcome, job.event_at)
                    LOGGER.exception(
                        "motion refinement failed for %s (%s): %s: %s",
                        self.camera_id,
                        "retrying" if retrying else "terminal",
                        type(error).__name__,
                        redact_secret_text(error)[:500],
                    )
                    if retrying:
                        try:
                            self._refinement_queue.put_nowait(True)
                        except queue.Full:
                            pass
                    else:
                        with self._status_lock:
                            self._refinement_callbacks.pop(job_id, None)
                    continue

                self._record_timing(outcome)
                if outcome.object_detected is None:
                    retrying = self.refinement_store.retry_detection_job(
                        job_id,
                        outcome.rejection_reason or "refinement returned no terminal evidence",
                    )
                    if retrying:
                        try:
                            self._refinement_queue.put_nowait(True)
                        except queue.Full:
                            pass
                    else:
                        with self._status_lock:
                            self._refinement_callbacks.pop(job_id, None)
                    continue
                self._handoff(
                    outcome,
                    job.event_at,
                )
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
                self.refinement_store.complete_detection_job(job_id, outcome.event_id)
                completed = True
            finally:
                with self._status_lock:
                    if completed:
                        self._refinement_callbacks.pop(job_id, None)

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
                return
