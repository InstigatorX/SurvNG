from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, replace
import copy
from collections import deque
import errno
import hashlib
import json
import logging
import math
import queue
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Protocol

import numpy as np

from .durable_payload import durable_json_copy
from .evidence_work import EvidenceWorkPreempted, cancellable_evidence_work
from .event_store.jobs import (
    DETECTION_COMPLETION_JOB_MAXIMUM_AGE_SECONDS,
    DETECTION_EVENT_JOB_MAXIMUM_AGE_SECONDS,
    DETECTION_JOB_MAXIMUM_AGE_SECONDS,
    detection_job_occurrence_equivalent,
)
from .motion_pipeline.decision_handler import MotionDecisionOutcome
from .perf_samples import RollingLatencySamples
from .security import redact_secret_text


LOGGER = logging.getLogger(__name__)
REFINEMENT_COMPLETION_RETRY_SECONDS = 2.0
REFINEMENT_COMPLETION_MAXIMUM_ATTEMPTS = 2_147_483_647
REFINEMENT_STORE_RETRY_SECONDS = 2.0


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
        payload: dict[str, Any], available_at: float | None = None,
    ) -> str: ...
    def claim_detection_job(
        self, camera_id: str, *, lease_seconds: float = 60.0, lease_owner: str = "",
        maximum_age_seconds: float = DETECTION_JOB_MAXIMUM_AGE_SECONDS,
        event_maximum_age_seconds: float | None = None,
    ) -> dict[str, Any] | None: ...
    def complete_detection_job(self, job_id: str, event_id: int | None, *, lease_owner: str = "") -> bool | None: ...
    def checkpoint_detection_job(
        self, job_id: str, payload: dict[str, Any], *, lease_owner: str = "",
    ) -> bool: ...
    def defer_detection_job(self, job_id: str, *, lease_owner: str) -> bool: ...
    def reschedule_detection_job(
        self, job_id: str, payload: dict[str, Any], *, available_at: float, lease_owner: str,
    ) -> bool: ...
    def retry_detection_job(
        self, job_id: str, error: str, *, retry_delay_seconds: float = 2.0,
        maximum_attempts: int = 5,
        lease_owner: str = "",
    ) -> bool | None: ...
    def expire_stale_detection_jobs(
        self, camera_id: str, *, maximum_age_seconds: float,
        event_maximum_age_seconds: float | None = None,
    ) -> int: ...
    def detection_job_status(self, camera_id: str) -> dict[str, int | float]: ...
    def pending_detection_job_ids(self, camera_id: str) -> set[str]: ...

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
RefinementCompletionHandler = Callable[[MotionDecisionOutcome, dict[str, Any]], None]
REFINEMENT_MAX_QUEUE_AGE_SECONDS = DETECTION_JOB_MAXIMUM_AGE_SECONDS
REFINEMENT_EVENT_MAX_QUEUE_AGE_SECONDS = DETECTION_EVENT_JOB_MAXIMUM_AGE_SECONDS
REFINEMENT_STALE_EXPIRY_INTERVAL_SECONDS = 5.0
# Matches the retired durable discovery expiry. The retained live frame stays
# usable, but an older sample no longer describes the current scene.
SCENE_DISCOVERY_MAX_AGE_SECONDS = 60.0
REFINEMENT_WORKER_STAGES = (
    "claim", "refine", "checkpoint", "handoff", "completion", "complete",
    "reschedule", "verification", "cover", "discovery",
)


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


def _outcome_payload(outcome: MotionDecisionOutcome) -> dict[str, Any]:
    return {
        **outcome.as_dict(),
        "detected_objects": list(outcome.detected_objects),
    }


def _outcome_from_payload(payload: dict[str, Any]) -> MotionDecisionOutcome:
    return MotionDecisionOutcome(
        event_id=(int(payload["event_id"]) if payload.get("event_id") is not None else None),
        snapshot_path=str(payload.get("snapshot_path") or ""),
        object_detected=payload.get("object_detected"),
        detected_objects=tuple(payload.get("detected_objects") or ()),
        rejection_reason=str(payload.get("rejection_reason") or ""),
        motion_correlation=payload.get("motion_correlation"),
        refinement_pending=bool(payload.get("refinement_pending")),
        processing_timing=payload.get("processing_timing"),
        object_activity=payload.get("object_activity"),
        depth_attribution=payload.get("depth_attribution"),
        cover_promoted=bool(payload.get("cover_promoted")),
        cover_promotion_reason=str(payload.get("cover_promotion_reason") or ""),
        refinement_event_id=(
            int(payload["refinement_event_id"])
            if payload.get("refinement_event_id") is not None
            else None
        ),
    )


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
    completion_context: dict[str, Any] | None
    initial_outcome: MotionDecisionOutcome
    refined_outcome: MotionDecisionOutcome | None = None
    handoff_completed: bool = False
    handoff_disposition: str = "pending"
    # When recorded evidence is predicted to exist. Claim and freshness are
    # measured from here, not from admission, so segment closure is not
    # charged against the job's age.
    evidence_ready_epoch: float | None = None

    def key(self) -> tuple[str, str | int | float]:
        # Once the fast path persists an incident, that canonical event—not the
        # route probe that happened to discover it—owns recorded refinement.
        # This keeps an earlier no-object attempt for the same route intent
        # from coalescing with or colliding against the event's cover job.
        if self.existing_event_id is not None:
            return ("event", self.existing_event_id)
        intent_id = str(self.qualification.get("detection_intent_id") or "").strip()
        if intent_id:
            return ("intent", intent_id)
        # Legacy/manual callers without a durable intent may still refine, but
        # a runtime-local episode sequence is never a safe durable identity.
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
            "qualification": durable_json_copy(self.qualification),
            "existing_event_id": self.existing_event_id,
            "require_eligible_object": self.require_eligible_object,
            "require_motion_correlation": self.require_motion_correlation,
            "completion_context": durable_json_copy(self.completion_context or {}),
            "initial_outcome": _outcome_payload(self.initial_outcome),
            "refined_outcome": (
                _outcome_payload(self.refined_outcome)
                if self.refined_outcome is not None
                else None
            ),
            "handoff_completed": self.handoff_completed,
            "handoff_disposition": self.handoff_disposition,
            **(
                {"evidence_ready_epoch": self.evidence_ready_epoch}
                if self.evidence_ready_epoch is not None
                else {}
            ),
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
            completion_context=(dict(payload.get("completion_context") or {}) or None),
            initial_outcome=_outcome_from_payload(initial),
            refined_outcome=(
                _outcome_from_payload(dict(payload["refined_outcome"]))
                if isinstance(payload.get("refined_outcome"), dict)
                else None
            ),
            handoff_completed=bool(payload.get("handoff_completed")),
            handoff_disposition=str(payload.get("handoff_disposition") or "pending"),
            evidence_ready_epoch=(
                float(payload["evidence_ready_epoch"])
                if payload.get("evidence_ready_epoch") is not None
                else None
            ),
        )


class _MemoryDetectionJobStore:
    """Test/local fallback with the same no-capacity-drop semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def enqueue_detection_job(self, *, job_id, camera_id, dedupe_key, payload, available_at=None):
        delay = 0.0 if available_at is None else max(0.0, float(available_at) - time.time())
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None:
                existing = next(
                    (
                        job
                        for job in self._jobs.values()
                        if job["camera_id"] == camera_id
                        and job["dedupe_key"] == dedupe_key
                    ),
                    None,
                )
            if existing is not None:
                if (
                    existing["id"] != job_id
                    or existing["camera_id"] != camera_id
                    or existing["dedupe_key"] != dedupe_key
                ):
                    raise RuntimeError(
                        "detection job identity collision with different occurrence"
                    )
                if not detection_job_occurrence_equivalent(
                    existing["payload"], payload
                ):
                    raise RuntimeError(
                        "detection job identity collision with different occurrence"
                    )
                return "coalesced"
            self._jobs[job_id] = {
                "id": job_id, "camera_id": camera_id, "dedupe_key": dedupe_key,
                "payload": copy.deepcopy(payload), "state": "queued", "attempts": 0,
                "available_at": time.monotonic() + delay,
                "created_at_monotonic": time.monotonic(),
                "ready_at_monotonic": time.monotonic() + delay if delay else None,
                "lease_owner": "",
                "lease_expires_at": None,
            }
            return "queued"

    def claim_detection_job(
        self,
        camera_id,
        *,
        lease_seconds=60.0,
        lease_owner="",
        maximum_age_seconds=DETECTION_JOB_MAXIMUM_AGE_SECONDS,
        event_maximum_age_seconds=None,
    ):
        now = time.monotonic()
        with self._lock:
            self._expire_stale_detection_jobs_locked(
                camera_id,
                maximum_age_seconds=maximum_age_seconds,
                event_maximum_age_seconds=event_maximum_age_seconds,
                now=now,
            )
            reclaimable = [
                job
                for job in self._jobs.values()
                if (
                    job["camera_id"] == camera_id
                    and job["state"] == "running"
                    and (
                        (
                            job.get("lease_expires_at") is not None
                            and float(job["lease_expires_at"]) <= now
                        )
                        or str(job.get("lease_owner") or "") == str(lease_owner)
                    )
                )
            ]
            if reclaimable:
                job = min(
                    reclaimable,
                    key=lambda item: (
                        float(item["created_at_monotonic"]),
                        str(item["id"]),
                    ),
                )
            else:
                queued = [
                    candidate
                    for candidate in self._jobs.values()
                    if candidate["camera_id"] == camera_id
                    and candidate["state"] == "queued"
                    and candidate["available_at"] <= now
                ]
                def claim_rank(candidate):
                    qualification = candidate["payload"].get("qualification")
                    if isinstance(qualification, dict) and qualification.get("scene_discovery") is True:
                        return 2
                    return 1 if candidate["payload"].get("existing_event_id") is not None else 0

                job = min(queued, key=claim_rank, default=None)
                if job is None:
                    return None
            job["state"] = "running"
            job["attempts"] += 1
            job["lease_owner"] = str(lease_owner or "")
            job["lease_expires_at"] = now + max(1.0, float(lease_seconds))
            return copy.deepcopy(job)

    def expire_stale_detection_jobs(
        self,
        camera_id,
        *,
        maximum_age_seconds,
        event_maximum_age_seconds=None,
    ):
        with self._lock:
            return self._expire_stale_detection_jobs_locked(
                camera_id,
                maximum_age_seconds=maximum_age_seconds,
                event_maximum_age_seconds=event_maximum_age_seconds,
                now=time.monotonic(),
            )

    def _expire_stale_detection_jobs_locked(
        self,
        camera_id,
        *,
        maximum_age_seconds,
        event_maximum_age_seconds,
        now,
    ):
        probe_cutoff = now - max(0.0, float(maximum_age_seconds))
        event_maximum_age = (
            float(maximum_age_seconds)
            if event_maximum_age_seconds is None
            else max(0.0, float(event_maximum_age_seconds))
        )
        event_cutoff = now - event_maximum_age
        expired = 0
        for job in self._jobs.values():
            checkpointed = isinstance(job["payload"].get("refined_outcome"), dict)
            qualification = job["payload"].get("qualification")
            discovery = isinstance(qualification, dict) and qualification.get("scene_discovery") is True
            cutoff = (
                now - DETECTION_COMPLETION_JOB_MAXIMUM_AGE_SECONDS
                if checkpointed
                else event_cutoff
                if job["payload"].get("existing_event_id") is not None or discovery
                else probe_cutoff
            )
            reference = (
                job["created_at_monotonic"]
                if checkpointed
                else max(job["created_at_monotonic"], job.get("ready_at_monotonic") or 0.0)
            )
            if job["camera_id"] != camera_id or reference > cutoff:
                continue
            lease_expires_at = job.get("lease_expires_at")
            reclaimable_running = (
                job["state"] == "running"
                and lease_expires_at is not None
                and float(lease_expires_at) <= now
            )
            if job["state"] == "queued" or reclaimable_running:
                job["state"] = "failed"
                job["last_error"] = (
                    "expired_refinement_completion" if checkpointed else "stale_refinement"
                )
                job["lease_owner"] = ""
                job["lease_expires_at"] = None
                expired += 1
        return expired

    def complete_detection_job(self, job_id, event_id, *, lease_owner=""):
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] != "running":
                return False
            if lease_owner and str(job.get("lease_owner") or "") != lease_owner:
                return False
            job["state"] = "completed"
            job["event_id"] = event_id
            job["lease_owner"] = ""
            job["lease_expires_at"] = None
            return True

    def checkpoint_detection_job(self, job_id, payload, *, lease_owner=""):
        with self._lock:
            job = self._jobs[job_id]
            if lease_owner and str(job.get("lease_owner") or "") != lease_owner:
                return False
            if job["state"] != "running":
                return False
            job["payload"] = copy.deepcopy(payload)
            return True

    def defer_detection_job(self, job_id: str, *, lease_owner: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] != "running" or job["lease_owner"] != lease_owner:
                return False
            job.update(state="queued", attempts=max(0, job["attempts"] - 1),
                       available_at=time.monotonic(), lease_expires_at=None,
                       lease_owner="", last_error="detection_stopped")
            return True

    def reschedule_detection_job(self, job_id, payload, *, available_at, lease_owner):
        delay = max(0.0, float(available_at) - time.time())
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] != "running" or job["lease_owner"] != lease_owner:
                return False
            ready = time.monotonic() + delay
            job.update(state="queued", attempts=max(0, job["attempts"] - 1),
                       payload=copy.deepcopy(payload), available_at=ready,
                       ready_at_monotonic=ready, lease_expires_at=None, lease_owner="")
            return True

    def retry_detection_job(
        self, job_id, error, *, retry_delay_seconds=2.0, maximum_attempts=5,
        lease_owner="",
    ):
        with self._lock:
            job = self._jobs[job_id]
            if job["state"] != "running":
                return None
            if lease_owner and str(job.get("lease_owner") or "") != lease_owner:
                return None
            retry = job["attempts"] < maximum_attempts
            job["state"] = "queued" if retry else "failed"
            job["last_error"] = str(error)[:1000]
            job["available_at"] = time.monotonic() + retry_delay_seconds
            job["lease_owner"] = ""
            job["lease_expires_at"] = None
            return retry

    def cancel_queued_detection_jobs(self, camera_id, *, reason="detection_disabled"):
        with self._lock:
            cancelled = 0
            for job in self._jobs.values():
                if (
                    job["camera_id"] == camera_id and job["state"] == "queued"
                    and not isinstance(job["payload"].get("refined_outcome"), dict)
                ):
                    job["state"] = "failed"
                    job["last_error"] = reason
                    cancelled += 1
            return cancelled

    def pending_detection_job_ids(self, camera_id):
        with self._lock:
            return {
                job_id for job_id, job in self._jobs.items()
                if job["camera_id"] == camera_id
                and job["state"] in {"queued", "running"}
            }

    def detection_job_status(self, camera_id):
        with self._lock:
            result: dict[str, int | float] = {}
            now = time.monotonic()
            oldest_created_at: float | None = None
            for job in self._jobs.values():
                if job["camera_id"] == camera_id:
                    result[job["state"]] = result.get(job["state"], 0) + 1
                    if job["state"] in {"queued", "running"}:
                        created_at = float(job["created_at_monotonic"])
                        oldest_created_at = (
                            created_at
                            if oldest_created_at is None
                            else min(oldest_created_at, created_at)
                        )
            result["oldest_age_ms"] = round(
                max(0.0, (now - oldest_created_at) * 1000.0)
                if oldest_created_at is not None
                else 0.0,
                3,
            )
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
        scene_analysis_enabled: TrackingEnabled | None = None,
        resume_scene_analysis: Callable[[], bool] | None = None,
        evidence_ready_at: Callable[[datetime, dict[str, Any]], float | None] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.decision_processor = decision_processor
        self.evidence_ready_at = evidence_ready_at
        self.tracking_enabled = tracking_enabled
        self.scene_analysis_enabled = scene_analysis_enabled or (lambda: False)
        self.resume_scene_analysis = resume_scene_analysis
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
        self._lease_owner = uuid.uuid4().hex
        self._security_work_pending = threading.Event()
        self._refinement_callbacks: dict[str, RefinementCallback] = {}
        self._refinement_progress: dict[str, _RefinementJob] = {}
        self._refinement_completion_handler: RefinementCompletionHandler | None = None
        self._refinement_thread: threading.Thread | None = None
        self._refinement_stop: threading.Event | None = None
        self._active_refinement: dict[str, Any] | None = None
        self._refinement_accepting = False
        self._refinements_queued = 0
        self._refinements_completed = 0
        self._refinements_coalesced = 0
        self._refinements_resumed = 0
        self._pending_scene_discovery: datetime | None = None
        self._scene_discovery_counts = {
            "offered": 0, "superseded": 0, "stale": 0, "completed": 0, "failed": 0,
        }
        self._last_scene_discovery_failure_log = float("-inf")
        self._refinement_failures = 0
        self._refinement_timeouts = 0
        self._last_refinement_failure: dict[str, Any] | None = None
        self._refinement_callback_failures = 0
        self._last_refinement_callback_failure: dict[str, Any] | None = None
        self._timing_samples = 0
        self._timing_totals_ms: dict[str, float] = {}
        self._timing_counts: dict[str, int] = {}
        self._last_timing_ms: dict[str, float] = {}
        self._live_workflow_samples = RollingLatencySamples()
        self._refine_workflow_samples = RollingLatencySamples()
        self._worker_stage_samples = {
            stage: RollingLatencySamples() for stage in REFINEMENT_WORKER_STAGES
        }

    def status(self) -> dict[str, Any]:
        durable = self.refinement_store.detection_job_status(self.camera_id)
        lock_status = getattr(self.refinement_store, "lock_wait_status", None)
        lock_wait = lock_status() if callable(lock_status) else None
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
                "refinements_resumed": self._refinements_resumed,
                "worker_stage_ms": {
                    stage: samples.snapshot()
                    for stage, samples in self._worker_stage_samples.items()
                },
                "store_lock_wait": lock_wait,
                "scene_discovery": {
                    **self._scene_discovery_counts,
                    "pending": self._pending_scene_discovery is not None,
                },
                "refinement_failures": self._refinement_failures,
                "refinement_timeouts": self._refinement_timeouts,
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
                "oldest_refinement_age_ms": float(durable.get("oldest_age_ms", 0.0)),
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
                    "live_workflow_ms_p95": self._live_workflow_samples.percentile(95),
                    "refine_workflow_ms_p95": self._refine_workflow_samples.percentile(95),
                    "refinement_timeouts": self._refinement_timeouts,
                    "oldest_refinement_age_ms": float(
                        durable.get("oldest_age_ms", 0.0)
                    ),
                },
                "object_activity_attribution": self.decision_processor.activity_status(),
            }

    def set_refinement_completion_handler(
        self,
        handler: RefinementCompletionHandler | None,
    ) -> None:
        """Register finalization that can be rebuilt from a durable payload."""
        with self._status_lock:
            self._refinement_completion_handler = handler

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

    def _refinement_worker_accepting(self) -> bool:
        with self._status_lock:
            stop = self._refinement_stop
            return bool(
                self._refinement_accepting
                and stop is not None
                and not stop.is_set()
                and self._refinement_thread is not None
                and self._refinement_thread.is_alive()
            )

    def request_stop(self) -> None:
        # Close admission before publishing the sentinel. A producer that
        # arrives after this sees no accepting worker and hands tracking off
        # from live evidence instead of waiting for a job that will not run.
        with self._status_lock:
            self._refinement_accepting = False
            self._security_work_pending.set()
        try:
            self._refinement_queue.put_nowait(True)
        except queue.Full:
            pass

    def _set_refinement_stage(self, stage: str) -> None:
        with self._status_lock:
            if self._active_refinement is not None:
                self._active_refinement["stage"] = stage

    def wait_stopped(self, timeout: float) -> bool:
        thread = self._refinement_thread
        if thread is None:
            return True
        thread.join(max(0.0, timeout))
        if thread.is_alive():
            with self._status_lock:
                active = dict(self._active_refinement or {})
            started = active.pop("started_monotonic", time.monotonic())
            LOGGER.error(
                "refinement shutdown timeout camera=%s job=%s stage=%s elapsed_seconds=%.3f",
                self.camera_id, active.get("job_id", "unknown"),
                active.get("stage", "idle"), time.monotonic() - started,
            )
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

    def discard_queued_work(self, reason: str = "detection_disabled") -> int:
        """Drop unstarted inference after detection is turned off for this camera."""
        with self._status_lock:
            self._pending_scene_discovery = None
        cancel = getattr(self.refinement_store, "cancel_queued_detection_jobs", None)
        if not callable(cancel):
            return 0
        cancelled = int(cancel(self.camera_id, reason=reason))
        self._forget_terminal_refinements()
        if cancelled:
            LOGGER.info(
                "discarded %d queued detection job(s) for %s: %s",
                cancelled, self.camera_id, reason,
            )
        return cancelled

    def offer_scene_discovery(self, event_at: datetime) -> str:
        """Hold one whole-scene sample for the worker's idle time.

        Discovery is opportunistic: only the newest offer is kept, it is never
        persisted, and every durable job runs first.
        """
        with self._status_lock:
            self._scene_discovery_counts["offered"] += 1
            admission = "queued"
            if self._pending_scene_discovery is not None:
                self._scene_discovery_counts["superseded"] += 1
                admission = "superseded"
            self._pending_scene_discovery = event_at
            wake = self._refinement_accepting
        if wake:
            try:
                self._refinement_queue.put_nowait(True)
            except queue.Full:
                pass
        return admission

    def _run_scene_discovery(self) -> bool:
        if self._security_work_pending.is_set():
            return False
        with self._status_lock:
            event_at = self._pending_scene_discovery
            self._pending_scene_discovery = None
        if event_at is None:
            return False
        if time.time() - event_at.timestamp() > SCENE_DISCOVERY_MAX_AGE_SECONDS:
            with self._status_lock:
                self._scene_discovery_counts["stale"] += 1
            return True
        qualification = {
            "scene_discovery": True,
            "trigger_source": "scene_discovery",
            "detection_intent_id": f"scene:{self.camera_id}:{event_at.timestamp():.6f}",
        }
        try:
            with cancellable_evidence_work(lambda: bool(self._refinement_stop and self._refinement_stop.is_set())):
                outcome = self.decision_processor.refine(
                    "scene/discovery", "Scene observation", event_at, qualification,
                    existing_event_id=None, require_eligible_object=False,
                    require_motion_correlation=False,
                )
        except EvidenceWorkPreempted:
            return True
        except Exception:
            with self._status_lock:
                self._scene_discovery_counts["failed"] += 1
            now = time.monotonic()
            if now - self._last_scene_discovery_failure_log >= 60.0:
                self._last_scene_discovery_failure_log = now
                LOGGER.exception("scene discovery failed for %s", self.camera_id)
            return True
        self._record_timing(outcome, kind="refine")
        if outcome.event_id is not None:
            self._handoff(outcome, event_at)
        with self._status_lock:
            self._scene_discovery_counts["completed"] += 1
        return True

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
        refinement_completion_context: dict[str, Any] | None = None,
    ) -> MotionDecisionOutcome:
        outcome = self.decision_processor.handle(
            topic,
            message,
            event_at,
            qualification,
            require_eligible_object=require_eligible_object,
            require_motion_correlation=require_motion_correlation,
        )
        self._record_timing(outcome, kind="live")
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
                    existing_event_id=(
                        outcome.refinement_event_id
                        if outcome.refinement_event_id is not None
                        else outcome.event_id
                    ),
                    require_eligible_object=require_eligible_object,
                    require_motion_correlation=require_motion_correlation,
                    callback=refinement_callback,
                    completion_context=refinement_completion_context,
                    initial_outcome=outcome,
                )
            )
        # Tracking is identification/cover enrichment, not admission. Wait for
        # recorded confirmation whenever a refinement worker can actually run
        # this job. If refinement cannot run now, preserve a live handoff so a
        # confirmed object is not left without a session.
        defer_tracking = (
            outcome.refinement_pending
            and refinement_admission in {"queued", "coalesced"}
            and self._refinement_worker_accepting()
        )
        if not defer_tracking:
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
        return outcome

    def _handoff(self, outcome: MotionDecisionOutcome, event_at: datetime) -> bool:
        if outcome.event_id is None or (not outcome.object_detected and not self.scene_analysis_enabled()):
            return False

        with self._handoff_lock:
            event_id = int(outcome.event_id)
            if event_id in self._handed_off_event_ids:
                return True
            try:
                detected_objects = list(outcome.detected_objects)
                if not self.scene_analysis_enabled() and not self.has_trackable_objects(detected_objects):
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
        if self.evidence_ready_at is not None and job.evidence_ready_epoch is None:
            try:
                ready = self.evidence_ready_at(job.event_at, job.qualification)
            except Exception:
                LOGGER.debug("evidence readiness unavailable for %s", self.camera_id, exc_info=True)
                ready = None
            if ready is not None and math.isfinite(ready):
                job = replace(job, evidence_ready_epoch=float(ready))
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
                **(
                    {"available_at": job.evidence_ready_epoch}
                    if job.evidence_ready_epoch is not None
                    else {}
                ),
            )
            if admission == "coalesced":
                self._refinements_coalesced += 1
                return admission
            # Coalescing can refer to a terminal job; only new admission is
            # a wakeup. Existing durable retries are checked by the worker.
            self._security_work_pending.set()
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
        """Recover ledger I/O without starting a second consumer or losing leases."""
        last_log: float | None = None
        try:
            while (stop := self._refinement_stop) is not None and not stop.is_set():
                try:
                    self._run_refinements_until_error()
                    return
                except Exception as error:
                    retryable = self._retryable_store_error(error)
                    with self._status_lock:
                        self._refinement_failures += 1
                        self._last_refinement_failure = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "error": redact_secret_text(error)[:500],
                            "error_type": type(error).__name__,
                        }
                    now = time.monotonic()
                    if not retryable or last_log is None or now - last_log >= 60.0:
                        last_log = now
                        LOGGER.exception(
                            "motion refinement worker %s for %s",
                            "retrying ledger I/O" if retryable else "stopped after unexpected failure",
                            self.camera_id,
                        )
                    if not retryable or stop.wait(REFINEMENT_STORE_RETRY_SECONDS):
                        return
        finally:
            with self._status_lock:
                self._refinement_accepting = False

    @staticmethod
    def _retryable_store_error(error: Exception) -> bool:
        if isinstance(error, sqlite3.OperationalError):
            code = getattr(error, "sqlite_errorcode", 0) & 0xFF
            return code in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
                sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL, sqlite3.SQLITE_PROTOCOL,
            } or (not code and str(error).lower() in {
                "database is locked", "database table is locked", "disk i/o error",
                "unable to open database file", "database or disk is full",
            })
        return isinstance(error, OSError) and error.errno in {
            errno.EAGAIN, errno.EINTR, errno.EIO, errno.ENOSPC,
            errno.ESTALE, errno.ETIMEDOUT,
        }

    def _forget_terminal_refinements(self) -> None:
        with self._status_lock:
            local_ids = set(self._refinement_callbacks) | set(self._refinement_progress)
        if not local_ids:
            return
        pending = self.refinement_store.pending_detection_job_ids(self.camera_id)
        with self._status_lock:
            for job_id in local_ids:
                if job_id not in pending:
                    self._refinement_callbacks.pop(job_id, None)
                    self._refinement_progress.pop(job_id, None)

    def _run_scene_candidate(self) -> bool:
        """Verify a retained acquisition without inventing an event to own work."""
        claim = getattr(self.refinement_store, "claim_scene_candidate", None)
        if not callable(claim) or self._security_work_pending.is_set():
            return False
        job = claim(self.camera_id, lease_owner=self._lease_owner)
        if job is None:
            return False
        owner = {"lease_owner": self._lease_owner, "lease_token": job["lease_token"]}
        try:
            seed = self.refinement_store.scene_sample(job["seed_sample_ids"][0])
            if seed is None:
                raise LookupError("candidate acquisition unavailable")
            qualification = {
                "scene_confirmation": True, "trigger_source": "scene_confirmation",
                "scene_candidate_id": job["id"], "scene_candidate_lease_owner": self._lease_owner,
                "scene_candidate_lease_token": job["lease_token"],
                "scene_seed_sample_ids": job["seed_sample_ids"],
                "scene_confirmation_window": {"start_epoch": job["start_epoch"], "end_epoch": job["end_epoch"]},
                "detection_intent_id": "scene-confirmation:" + job["id"] + ":" + str(job["lease_generation"]),
            }
            event_at = datetime.fromtimestamp(seed["captured_epoch"], timezone.utc)
            admitted = self.refinement_store.scene_candidate_admission(job["id"], job["lease_generation"])
            if admitted:
                outcome = MotionDecisionOutcome(
                    event_id=admitted["id"],
                    snapshot_path=str(self.refinement_store._snapshot_path_for_retention(admitted["snapshot_path"])) if admitted["snapshot_path"] else "",
                    object_detected=True,
                    detected_objects=tuple(o for o in json.loads(admitted["objects_json"] or "[]") if o.get("label")),
                    scene_activity_decision_id=admitted["decision_id"],
                )
                event_at = datetime.fromisoformat(admitted["created_at"].replace("Z", "+00:00"))
            else:
                with cancellable_evidence_work(lambda: bool(self._refinement_stop and self._refinement_stop.is_set())):
                    outcome = self.decision_processor.refine(
                        "scene/confirmation", "Verification of observed activity", event_at,
                        qualification, existing_event_id=None,
                        require_eligible_object=False, require_motion_correlation=False,
                    )
            decision_id = outcome.scene_activity_decision_id
            if not decision_id:
                raise RuntimeError("verification did not persist an activity decision")
            decision = self.refinement_store.scene_activity_decision(decision_id)
            if decision["verdict"] in {"pending", "incomplete"}:
                self.refinement_store.defer_scene_candidate(job["id"], reason=decision["reason"], **owner)
            else:
                if outcome.event_id is not None and not self._handoff(outcome, event_at):
                    self.refinement_store.defer_scene_candidate(job["id"], reason="tracking_handoff_pending", **owner)
                else:
                    self.refinement_store.finish_scene_candidate(job["id"], decision_id=decision_id, **owner)
            self._record_timing(outcome, kind="refine")
        except EvidenceWorkPreempted:
            self.refinement_store.defer_scene_candidate(job["id"], reason="verification_preempted", **owner)
        except Exception as error:
            self.refinement_store.defer_scene_candidate(job["id"], reason="verification_failed:" + type(error).__name__, **owner)
            LOGGER.exception("scene verification failed for %s candidate %s", self.camera_id, job["id"])
        return True

    def _run_cover_requirement(self) -> bool:
        claim = getattr(self.refinement_store, "claim_cover_requirement", None)
        if not callable(claim):
            return False
        if self._security_work_pending.is_set():
            return False
        requirement = claim(self.camera_id, lease_owner=self._lease_owner)
        if requirement is None:
            return False
        event_id = int(requirement["event_id"])
        reason = "cover_attempt_failed"
        next_priority_check = 0.0

        def preempted() -> bool:
            nonlocal next_priority_check
            if self._security_work_pending.is_set() or bool(self._refinement_stop and self._refinement_stop.is_set()):
                return True
            # A durable retry can become due without a new in-memory enqueue.
            # Bound the read frequency while decoding/awaiting capacity.
            now = time.monotonic()
            if now >= next_priority_check:
                next_priority_check = now + 0.25
                due = getattr(self.refinement_store, "has_due_detection_job", None)
                if callable(due) and due(self.camera_id):
                    self._security_work_pending.set()
                    return True
            return False

        try:
            payload = requirement["payload"]
            qualification = copy.deepcopy(payload.get("qualification") or {})
            qualification["cover_only"] = True
            qualification["cover_requirement_lease_owner"] = self._lease_owner
            qualification["evidence_sampling"] = {
                "minimum_last_offset_seconds": min(12.0, 4.0 * int(requirement["attempts"])),
                "timeout_seconds": max(0.1, min(20.0, float(requirement["deadline_epoch"]) - time.time())),
            }
            with cancellable_evidence_work(preempted):
                outcome = self.decision_processor.refine(
                    str(payload.get("topic") or ""),
                    str(payload.get("message") or ""),
                    datetime.fromisoformat(payload["event_at"]),
                    qualification,
                    existing_event_id=event_id,
                    require_eligible_object=True,
                    require_motion_correlation=True,
                )
            reason = outcome.cover_promotion_reason or outcome.rejection_reason or "cover_not_satisfied"
        except EvidenceWorkPreempted:
            self.refinement_store.defer_cover_requirement(
                event_id, lease_owner=self._lease_owner, reason="security_work_preempted",
            )
            return True
        except Exception as error:
            reason = f"cover_attempt_failed:{type(error).__name__}"
            LOGGER.exception("cover recovery failed for %s event %d", self.camera_id, event_id)
        self.refinement_store.finish_cover_attempt(
            event_id, lease_owner=self._lease_owner, reason=reason,
        )
        return True

    def _run_refinements_until_error(self) -> None:
        last_prune = 0.0
        last_stale_expiry = 0.0
        last_scene_recovery = 0.0
        last_scene_recovery_log = float("-inf")
        while True:
            stop = self._refinement_stop
            if stop is None or stop.is_set():
                return
            prune = getattr(self.refinement_store, "prune_detection_jobs", None)
            now = time.monotonic()
            if self.resume_scene_analysis is not None and now - last_scene_recovery >= 1.0:
                last_scene_recovery = now
                try:
                    self.resume_scene_analysis()
                except Exception as error:
                    with self._status_lock:
                        self._refinement_failures += 1
                        self._last_refinement_failure = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "stage": "scene_analysis_recovery",
                            "error": redact_secret_text(error)[:500],
                            "error_type": type(error).__name__,
                        }
                    if now - last_scene_recovery_log >= 60:
                        last_scene_recovery_log = now
                        LOGGER.exception("scene analysis recovery failed for %s", self.camera_id)
            if callable(prune) and now - last_prune >= 60.0:
                last_prune = now
                try:
                    prune()
                except Exception:
                    LOGGER.exception(
                        "terminal detection-job pruning failed for %s",
                        self.camera_id,
                    )
            if now - last_stale_expiry >= REFINEMENT_STALE_EXPIRY_INTERVAL_SECONDS:
                last_stale_expiry = now
                expire_stale = getattr(
                    self.refinement_store,
                    "expire_stale_detection_jobs",
                    None,
                )
                if callable(expire_stale):
                    try:
                        expired = int(expire_stale(
                            self.camera_id,
                            maximum_age_seconds=REFINEMENT_MAX_QUEUE_AGE_SECONDS,
                            event_maximum_age_seconds=(
                                REFINEMENT_EVENT_MAX_QUEUE_AGE_SECONDS
                            ),
                        ))
                        self._forget_terminal_refinements()
                        if expired:
                            LOGGER.info(
                                "expired %d stale motion refinement job(s) for %s",
                                expired,
                                self.camera_id,
                            )
                    except Exception:
                        LOGGER.exception(
                            "stale motion refinement expiration failed for %s",
                            self.camera_id,
                        )
            # Clear before inspecting the durable queue. Admission after this
            # point sets the token, including the race between claim and cover.
            with self._status_lock:
                self._security_work_pending.clear()
            started = time.monotonic()
            claimed = self.refinement_store.claim_detection_job(
                self.camera_id,
                lease_owner=self._lease_owner,
                maximum_age_seconds=REFINEMENT_MAX_QUEUE_AGE_SECONDS,
                event_maximum_age_seconds=(
                    REFINEMENT_EVENT_MAX_QUEUE_AGE_SECONDS
                ),
            )
            if claimed is None:
                started = time.monotonic()
                if self._run_scene_candidate():
                    self._record_worker_stage("verification", started)
                    continue
                # The event owns cover completion independently of terminal
                # security jobs. Poll its durable requirements only after
                # ordinary refinement has had first access to this worker.
                started = time.monotonic()
                if self._run_cover_requirement():
                    self._record_worker_stage("cover", started)
                    continue
                started = time.monotonic()
                if self._run_scene_discovery():
                    self._record_worker_stage("discovery", started)
                    continue
                try:
                    self._refinement_queue.get(timeout=0.5)
                except queue.Empty:
                    pass
                continue
            self._record_worker_stage("claim", started)
            job_id = str(claimed["id"])
            with self._status_lock:
                callback = self._refinement_callbacks.get(job_id)
                completion_handler = self._refinement_completion_handler
            try:
                durable_job = _RefinementJob.from_payload(
                    dict(claimed["payload"]), callback
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                # This claimed occurrence cannot be processed. Quarantine it
                # under our lease so later valid work can proceed immediately.
                self.refinement_store.retry_detection_job(
                    job_id,
                    f"invalid_refinement_payload: {redact_secret_text(error)[:500]}",
                    maximum_attempts=1,
                    lease_owner=self._lease_owner,
                )
                with self._status_lock:
                    self._refinement_callbacks.pop(job_id, None)
                    self._refinement_progress.pop(job_id, None)
                    self._refinement_failures += 1
                    self._last_refinement_failure = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": "invalid_refinement_payload",
                        "error_type": type(error).__name__,
                    }
                LOGGER.exception("invalid refinement job %s for %s", job_id, self.camera_id)
                continue
            with self._status_lock:
                local_progress = self._refinement_progress.get(job_id)
            job = durable_job
            if local_progress is not None:
                job = replace(
                    durable_job,
                    refined_outcome=(
                        durable_job.refined_outcome
                        if durable_job.refined_outcome is not None
                        else local_progress.refined_outcome
                    ),
                    handoff_completed=(
                        durable_job.handoff_completed
                        or local_progress.handoff_completed
                    ),
                    handoff_disposition=(
                        durable_job.handoff_disposition if durable_job.handoff_completed
                        else local_progress.handoff_disposition
                    ),
                )
            refinement_qualification = dict(job.qualification)
            refinement_qualification["detection_intent_id"] = job_id
            reschedule = getattr(self.refinement_store, "reschedule_detection_job", None)
            if callable(reschedule) and job.refined_outcome is None:
                refinement_qualification["refinement_split_allowed"] = True
            completed = False
            with self._status_lock:
                self._active_refinement = {
                    "job_id": job_id, "stage": "recorded_refinement",
                    "started_monotonic": time.monotonic(),
                }
            try:
                try:
                    if job.refined_outcome is not None:
                        outcome = job.refined_outcome
                    else:
                        started = time.monotonic()
                        with cancellable_evidence_work(
                            stop.is_set, optional=False,
                            stage_reporter=self._set_refinement_stage,
                        ):
                            outcome = self.decision_processor.refine(
                                job.topic,
                                job.message,
                                job.event_at,
                                refinement_qualification,
                                existing_event_id=job.existing_event_id,
                                require_eligible_object=job.require_eligible_object,
                                require_motion_correlation=job.require_motion_correlation,
                            )
                        self._record_worker_stage("refine", started)
                except EvidenceWorkPreempted:
                    self.refinement_store.defer_detection_job(
                        job_id, lease_owner=self._lease_owner,
                    )
                    return
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
                        if self._is_refinement_timeout_error(error):
                            self._refinement_timeouts += 1
                        self._last_refinement_failure = failure
                    retrying = self.refinement_store.retry_detection_job(
                        job_id,
                        failure["error"],
                        lease_owner=self._lease_owner,
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

                if outcome.refinement_resume_at is not None and callable(reschedule):
                    # Earlier stages were empty and the next recording is not
                    # indexed yet: free the worker instead of sleeping on it.
                    resumed = replace(
                        job,
                        qualification={
                            **job.qualification,
                            "refinement_resume_stage": outcome.refinement_resume_stage,
                        },
                        evidence_ready_epoch=outcome.refinement_resume_at,
                    )
                    started = time.monotonic()
                    rescheduled = reschedule(
                        job_id, resumed.payload(),
                        available_at=outcome.refinement_resume_at,
                        lease_owner=self._lease_owner,
                    )
                    self._record_worker_stage("reschedule", started)
                    if rescheduled:
                        with self._status_lock:
                            self._refinements_resumed += 1
                    else:
                        LOGGER.warning(
                            "paused motion refinement %s for %s lost its lease",
                            job_id, self.camera_id,
                        )
                    continue

                with self._status_lock:
                    self._active_refinement["stage"] = "completion"
                if stop.is_set():
                    self.refinement_store.defer_detection_job(job_id, lease_owner=self._lease_owner)
                    return
                reused_checkpoint = job.refined_outcome is not None
                if not reused_checkpoint:
                    self._record_timing(outcome, kind="refine")
                if self._is_refinement_timeout_outcome(outcome):
                    with self._status_lock:
                        self._refinement_timeouts += 1
                if outcome.object_detected is None:
                    retrying = self.refinement_store.retry_detection_job(
                        job_id,
                        outcome.rejection_reason or "refinement returned no terminal evidence",
                        lease_owner=self._lease_owner,
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
                if not reused_checkpoint:
                    job = replace(job, refined_outcome=outcome)
                    with self._status_lock:
                        self._refinement_progress[job_id] = job
                if (
                    durable_job.refined_outcome is None
                    and not self._checkpoint_refinement_progress(job_id, job)
                ):
                    continue
                if not job.handoff_completed:
                    handoff_started = time.monotonic()
                    started = self._handoff(outcome, job.event_at)
                    self._record_worker_stage("handoff", handoff_started)
                    disposition = (
                        "started" if started else
                        "not_applicable" if not outcome.object_detected else "declined"
                    )
                    job = replace(job, handoff_completed=True, handoff_disposition=disposition)
                    with self._status_lock:
                        self._refinement_progress[job_id] = job
                if (
                    not durable_job.handoff_completed
                    and not self._checkpoint_refinement_progress(job_id, job)
                ):
                    continue
                durable_completion = job.completion_context is not None
                if durable_completion:
                    try:
                        if stop.is_set():
                            raise RuntimeError(
                                "refinement completion deferred during shutdown"
                            )
                        if completion_handler is None:
                            raise RuntimeError(
                                "refinement completion handler is unavailable"
                            )
                        completion_started = time.monotonic()
                        completion_handler(outcome, job.completion_context or {})
                        self._record_worker_stage("completion", completion_started)
                    except Exception as error:
                        self._record_refinement_completion_failure(job, error)
                        self._retry_refinement_completion(job_id, error)
                        continue
                elif job.callback is not None and not stop.is_set():
                    # Compatibility-only callbacks have no durable replay
                    # context. Preserve their historical best-effort behavior.
                    try:
                        job.callback(outcome)
                    except Exception as error:
                        self._record_refinement_completion_failure(job, error)
                        LOGGER.exception(
                            "motion refinement callback failed for %s: %s: %s",
                            self.camera_id,
                            type(error).__name__,
                            redact_secret_text(error)[:500],
                        )
                try:
                    complete_started = time.monotonic()
                    transitioned = self.refinement_store.complete_detection_job(
                        job_id,
                        outcome.event_id,
                        lease_owner=self._lease_owner,
                    )
                    self._record_worker_stage("complete", complete_started)
                    if transitioned is False:
                        raise RuntimeError("refinement completion lost its owner lease")
                except Exception as error:
                    self._record_refinement_completion_failure(job, error)
                    self._retry_refinement_completion(job_id, error)
                    continue
                completed = True
                with self._status_lock:
                    self._refinements_completed += 1
            finally:
                with self._status_lock:
                    self._active_refinement = None
                    if completed:
                        self._refinement_callbacks.pop(job_id, None)
                        self._refinement_progress.pop(job_id, None)

    def _checkpoint_refinement_progress(
        self,
        job_id: str,
        job: _RefinementJob,
    ) -> bool:
        try:
            started = time.monotonic()
            checkpointed = self.refinement_store.checkpoint_detection_job(
                job_id,
                job.payload(),
                lease_owner=self._lease_owner,
            )
            self._record_worker_stage("checkpoint", started)
        except Exception as error:
            self._record_refinement_completion_failure(job, error)
            retrying = self._retry_refinement_completion(job_id, error)
            if not retrying:
                with self._status_lock:
                    self._refinement_progress.pop(job_id, None)
            return False
        if checkpointed:
            return True
        error = RuntimeError("refinement result checkpoint lost its lease")
        self._record_refinement_completion_failure(job, error)
        with self._status_lock:
            self._refinement_progress.pop(job_id, None)
        LOGGER.error(
            "motion refinement checkpoint rejected for %s: %s",
            self.camera_id,
            error,
        )
        return False

    def _record_refinement_completion_failure(
        self,
        job: _RefinementJob,
        error: Exception,
    ) -> None:
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

    def _retry_refinement_completion(
        self,
        job_id: str,
        error: Exception,
    ) -> bool:
        try:
            retrying = self.refinement_store.retry_detection_job(
                job_id,
                redact_secret_text(error)[:500],
                retry_delay_seconds=REFINEMENT_COMPLETION_RETRY_SECONDS,
                maximum_attempts=REFINEMENT_COMPLETION_MAXIMUM_ATTEMPTS,
                lease_owner=self._lease_owner,
            )
        except Exception as retry_error:
            if not self._retryable_store_error(retry_error):
                raise
            # Keep the successful result in memory. The same owner can reclaim
            # its running lease and retry the checkpoint without inference.
            LOGGER.exception(
                "motion refinement retry write failed for %s",
                self.camera_id,
            )
            stop = self._refinement_stop
            if stop is not None:
                stop.wait(REFINEMENT_COMPLETION_RETRY_SECONDS)
            return True
        LOGGER.exception(
            "motion refinement completion failed for %s (%s): %s: %s",
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
                self._refinement_progress.pop(job_id, None)
        return retrying

    @staticmethod
    def _is_refinement_timeout_error(error: Exception) -> bool:
        text = str(error).strip().lower()
        return any(
            marker in text
            for marker in (
                "timed out",
                "timeout",
                "deadline expired",
                "decode process budget unavailable",
                "decode budget unavailable",
            )
        )

    @staticmethod
    def _is_refinement_timeout_outcome(outcome: MotionDecisionOutcome) -> bool:
        timeout_values = {
            "decode_budget_timeout",
            "deadline_expired",
            "refinement_timeout",
        }
        if str(outcome.rejection_reason or "").strip().lower() in timeout_values:
            return True
        for item in outcome.detected_objects:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            reason = str(item.get("reason") or "").strip().lower()
            if status in timeout_values or reason in timeout_values:
                return True
            if status == "cancelled" and reason == "deadline_expired":
                return True
        return False

    def _record_timing(
        self,
        outcome: MotionDecisionOutcome,
        *,
        kind: str,
    ) -> None:
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
        workflow_ms = flattened.get("workflow_ms")
        if workflow_ms is not None:
            (
                self._refine_workflow_samples
                if kind == "refine"
                else self._live_workflow_samples
            ).add(workflow_ms)
        with self._status_lock:
            self._timing_samples += 1
            self._last_timing_ms = dict(flattened)
            for key, value in flattened.items():
                self._timing_totals_ms[key] = self._timing_totals_ms.get(key, 0.0) + value
                self._timing_counts[key] = self._timing_counts.get(key, 0) + 1

    def _record_worker_stage(self, stage: str, started: float) -> None:
        self._worker_stage_samples[stage].add((time.monotonic() - started) * 1000.0)

    def _clear_refinements(self) -> None:
        while True:
            try:
                self._refinement_queue.get_nowait()
            except queue.Empty:
                return
