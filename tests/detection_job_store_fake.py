"""In-memory DetectionJobStore for tests.

Mirrors the SQLite store's job lifecycle so MotionIncidentService tests can
run without a database, and so parity tests can exercise both stores.
Production always uses the event store.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

from survng.app.event_store.jobs import (
    DETECTION_COMPLETION_JOB_MAXIMUM_AGE_SECONDS,
    DETECTION_JOB_MAXIMUM_AGE_SECONDS,
    detection_job_occurrence_equivalent,
)


class MemoryDetectionJobStore:
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

    def next_camera_work_delay_seconds(self, camera_id):
        """Return when this fallback store's next durable job can be claimed."""
        now = time.monotonic()
        with self._lock:
            due = [
                float(job.get("available_at") or now)
                for job in self._jobs.values()
                if job["camera_id"] == camera_id and job["state"] == "queued"
            ]
            due.extend(
                float(job.get("lease_expires_at") or now)
                for job in self._jobs.values()
                if job["camera_id"] == camera_id and job["state"] == "running"
            )
        return max(0.0, min(due) - now) if due else None
