"""Ownership boundary for one camera's motion runtime."""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .motion import MotionQualificationResult
from .motion_analysis_service import MotionAnalysisService
from .motion_decisions import MotionDecisionOrchestrator
from .motion_incidents import MotionIncidentService
from .motion_events import MotionEventCoordinator
from .motion_ingress import MotionEventIngressService
from .motion_pipeline import MotionEvidenceRepository, MotionPipeline
from .motion_qualification_service import MotionQualificationService
from .security import redact_secret_text

if TYPE_CHECKING:
    from .camera_lifecycle import CameraRuntimeState

LOGGER = logging.getLogger(__name__)


class CameraMotionState:
    """Own mutable motion telemetry and camera-facing event state.

    This object is deliberately independent of ``CameraWorker``. Motion
    services share it as a small synchronized state boundary instead of
    calling back through their composition owner.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        camera_state: CameraRuntimeState,
        event_callback: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        self.camera_id = camera_id
        self.camera_state = camera_state
        self._event_callback = event_callback
        self._lock = threading.Lock()
        self._ingress_idle = threading.Condition(camera_state.lock)
        self._ingress_by_generation: dict[int, int] = {}
        self._last_motion_at = ""
        self._analysis_wait_samples_ms: deque[float] = deque(maxlen=600)
        self._stats: dict[str, Any] = {
            "triggers": 0,
            "bursts": 0,
            "passed": 0,
            "audit_rejected": 0,
            "suppressed": 0,
            "priority_bypasses": 0,
            "insufficient_frames": 0,
            "inconclusive": 0,
            "dropped_triggers": 0,
            "analysis_frames_dropped": 0,
            "analysis_wait_ms_total": 0.0,
            "analysis_wait_ms_max": 0.0,
            "continuous_frames": 0,
            "continuous_candidates": 0,
            "adaptive_triggers_deferred": 0,
            "active_followup_candidates": 0,
            "active_followup_deduplicated": 0,
            "active_followup_rate_limited": 0,
            "active_followup_episode_limited": 0,
            "active_followup_queue_rejected": 0,
            "active_followup_commit_races": 0,
            "active_followup_triggers": 0,
            "active_followup_objects": 0,
            "active_followup_no_object": 0,
            "visual_backup_candidates": 0,
            "visual_backup_triggers": 0,
            "visual_backup_onvif_matches": 0,
            "visual_backup_rate_limited": 0,
            "visual_backup_not_ready": 0,
            "visual_backup_not_promoted": 0,
            "visual_backup_uncorrelated_objects": 0,
            "illumination_evaluations": 0,
            "illumination_candidates": 0,
            "illumination_filtered": 0,
            "illumination_verification_probes": 0,
            "illumination_verification_rescues": 0,
            "analysis_worker_errors": 0,
            "event_worker_errors": 0,
            "event_callback_errors": 0,
            "event_retries": 0,
            "event_retry_drops": 0,
            "stale_fusion_samples": 0,
            "validation_failures": 0,
            "validation_fail_opens": 0,
            "audit_object_matches": 0,
            "suppression_verification_checks": 0,
            "suppression_verification_rescues": 0,
            "last_result": None,
        }

    def enabled(self) -> bool:
        with self.camera_state.lock:
            return self.camera_state.enabled

    def detection_enabled(self) -> bool:
        with self.camera_state.lock:
            return self.camera_state.detection_enabled

    def lifecycle_generation(self) -> int:
        with self.camera_state.lock:
            return self.camera_state.generation

    def accepting_events(self) -> bool:
        with self.camera_state.lock:
            return self.camera_state.accepting_motion_events

    def begin_ingress(self) -> int | None:
        """Atomically admit one callback into the active camera generation."""
        with self._ingress_idle:
            if (
                not self.camera_state.accepting_motion_events
                or not self.camera_state.detection_enabled
            ):
                return None
            generation = self.camera_state.generation
            self._ingress_by_generation[generation] = (
                self._ingress_by_generation.get(generation, 0) + 1
            )
            return generation

    def end_ingress(self, generation: int) -> None:
        with self._ingress_idle:
            remaining = self._ingress_by_generation.get(generation, 0) - 1
            if remaining > 0:
                self._ingress_by_generation[generation] = remaining
            else:
                self._ingress_by_generation.pop(generation, None)
            if not self._ingress_by_generation:
                self._ingress_idle.notify_all()

    def wait_ingress_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._ingress_idle:
            while self._ingress_by_generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._ingress_idle.wait(remaining)
            return True

    def ingress_in_flight(self) -> int:
        with self._ingress_idle:
            return sum(self._ingress_by_generation.values())

    def last_motion_at(self) -> str:
        with self._lock:
            return self._last_motion_at

    def set_last_motion_at(self, value: str) -> None:
        with self._lock:
            self._last_motion_at = value

    def publish_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(event_type, payload)
        except Exception:
            self.increment_stat("event_callback_errors")
            LOGGER.exception(
                "camera event callback failed for %s event=%s",
                self.camera_id,
                event_type,
            )

    def increment_stat(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[name] = int(self._stats.get(name) or 0) + amount

    def record_analysis_wait(self, wait_ms: float) -> None:
        with self._lock:
            self._stats["analysis_wait_ms_total"] += wait_ms
            self._stats["analysis_wait_ms_max"] = max(
                float(self._stats["analysis_wait_ms_max"]),
                wait_ms,
            )
            self._analysis_wait_samples_ms.append(max(0.0, wait_ms))

    def record_decision(
        self,
        *,
        result: MotionQualificationResult,
        qualification: dict[str, Any],
        retry_attempt: bool,
        priority: bool,
        mode: str,
        borderline_candidate: bool,
        suppression_verification_candidate: bool,
    ) -> None:
        with self._lock:
            if not retry_attempt:
                self._stats["bursts"] += 1
                if priority:
                    self._stats["priority_bypasses"] += 1
                if result.reason == "insufficient_frames":
                    self._stats["insufficient_frames"] += 1
                if result.reason == "no_temporal_signal":
                    self._stats["inconclusive"] += 1
                if result.accepted:
                    self._stats["passed"] += 1
                elif (
                    mode in {"camera", "camera_rescue", "adaptive", "enforce"}
                    and not borderline_candidate
                    and not suppression_verification_candidate
                ):
                    self._stats["suppressed"] += 1
                elif mode == "audit":
                    self._stats["audit_rejected"] += 1
            self._stats["last_result"] = copy.deepcopy(qualification)

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._stats)
            samples = sorted(self._analysis_wait_samples_ms)
        result["analysis_wait_ms_p95"] = self._percentile(samples, 0.95)
        result["analysis_wait_ms_p99"] = self._percentile(samples, 0.99)
        return result

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = min(
            len(values) - 1,
            max(0, round((len(values) - 1) * percentile)),
        )
        return round(float(values[index]), 3)


class MotionRuntimeService:
    """Own both motion workers and every resource tied to their generation."""

    def __init__(
        self,
        *,
        camera_id: str,
        state: CameraMotionState,
        events: MotionEventCoordinator,
        analysis: MotionAnalysisService,
        decisions: MotionDecisionOrchestrator,
        incidents: MotionIncidentService,
        ingress: MotionEventIngressService,
        qualification: MotionQualificationService,
        evidence: MotionEvidenceRepository,
        pipelines: Iterable[tuple[str, MotionPipeline]],
    ) -> None:
        self.camera_id = camera_id
        self.state = state
        self.events = events
        self.analysis = analysis
        self.decisions = decisions
        self.incidents = incidents
        self.ingress = ingress
        self.qualification = qualification
        self.evidence = evidence
        self.pipelines = tuple(pipelines)
        self._operation_lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._generation_clean = True

    def start(self, stop_event: threading.Event) -> None:
        """Start a fresh analysis/decision generation transactionally."""
        with self._operation_lock:
            if not self._generation_clean:
                raise RuntimeError(
                    f"cannot start motion runtime for {self.camera_id} after "
                    "incomplete generation cleanup"
                )
            residual = self.active_workers()
            if residual:
                raise RuntimeError(
                    f"cannot start motion runtime for {self.camera_id} while "
                    f"workers remain: {', '.join(residual)}"
                )
            self.events.clear()
            self.events.episode_controller.configure_rescue_policy(
                self.qualification.visual_backup_policy()
            )
            self.events.episode_controller.start_generation(
                self.state.lifecycle_generation()
            )
            self._stop_event = stop_event
            try:
                self.incidents.start(stop_event)
                self.analysis.start(stop_event)
                self.decisions.start(stop_event)
            except BaseException as start_error:
                rollback_errors: list[BaseException] = []
                stop_event.set()
                self._attempt(self.analysis.request_stop, rollback_errors)
                self._attempt(self.decisions.request_stop, rollback_errors)
                self._attempt(self.incidents.request_stop, rollback_errors)
                self._attempt(
                    lambda: self.decisions.wait_stopped(1.0),
                    rollback_errors,
                )
                self._attempt(
                    lambda: self.analysis.wait_stopped(1.0),
                    rollback_errors,
                )
                self._attempt(
                    lambda: self.incidents.wait_stopped(1.0),
                    rollback_errors,
                )
                self._stop_event = None
                for rollback_error in rollback_errors:
                    LOGGER.error(
                        "motion runtime start rollback failed for %s: %s",
                        self.camera_id,
                        redact_secret_text(rollback_error),
                    )
                self._generation_clean = (
                    not rollback_errors and not self.active_workers()
                )
                raise start_error

    def request_stop(self) -> None:
        """Stop admission and wake both workers without blocking."""
        failures: list[BaseException] = []
        stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()
        self._attempt(self.analysis.request_stop, failures)
        self._attempt(self.decisions.request_stop, failures)
        self._attempt(self.incidents.request_stop, failures)
        self._raise_failures("request stop", failures)

    def wait_stopped(
        self,
        *,
        analysis_timeout: float,
        decision_timeout: float,
    ) -> bool:
        """Join both workers and reset generation state only after they exit."""
        with self._operation_lock:
            failures: list[BaseException] = []
            deadline = time.monotonic() + max(
                0.0,
                analysis_timeout,
                decision_timeout,
            )
            decision_stopped = self._attempt_result(
                lambda: self.decisions.wait_stopped(min(
                    max(0.0, decision_timeout),
                    max(0.0, deadline - time.monotonic()),
                )),
                failures,
            )
            analysis_stopped = self._attempt_result(
                lambda: self.analysis.wait_stopped(min(
                    max(0.0, analysis_timeout),
                    max(0.0, deadline - time.monotonic()),
                )),
                failures,
            )
            refinement_stopped = self._attempt_result(
                lambda: self.incidents.wait_stopped(min(
                    max(0.0, decision_timeout),
                    max(0.0, deadline - time.monotonic()),
                )),
                failures,
            )
            ingress_stopped = self._attempt_result(
                lambda: self.ingress.wait_idle(
                    max(0.0, deadline - time.monotonic())
                ),
                failures,
            )
            workers_stopped = (
                decision_stopped
                and analysis_stopped
                and refinement_stopped
                and ingress_stopped
            )
            if workers_stopped and not failures:
                self._attempt(self.analysis.reset, failures)
                self._attempt(self.evidence.clear, failures)
                self._attempt(self.qualification.reset_runtime, failures)
                self._attempt(self.events.reset, failures)
            stopped = workers_stopped and not failures
            self._generation_clean = stopped
            if stopped:
                self._stop_event = None
            elif not failures:
                LOGGER.error(
                    "preserving motion runtime for %s because workers remain: %s",
                    self.camera_id,
                    ", ".join(self.active_workers()) or "unknown",
                )
            self._raise_failures("wait for stop", failures)
            return stopped

    def active_workers(self) -> list[str]:
        workers: list[str] = []
        if self.decisions.running():
            workers.append("motion events")
        if self.analysis.running():
            workers.append("motion analysis")
        if self.incidents.running():
            workers.append("motion refinement")
        if self.ingress.in_flight():
            workers.append("motion ingress")
        return workers

    def close(self) -> None:
        with self._operation_lock:
            residual = self.active_workers()
            if residual:
                raise RuntimeError(
                    f"cannot close motion runtime for {self.camera_id} while "
                    f"workers remain: {', '.join(residual)}"
                )
            failures: list[BaseException] = []
            for label, pipeline in self.pipelines:
                try:
                    pipeline.close()
                except BaseException as error:
                    failures.append(error)
                    LOGGER.error(
                        "%s motion pipeline cleanup failed for %s: %s",
                        label,
                        self.camera_id,
                        redact_secret_text(error),
                    )
            self._raise_failures("close motion pipelines", failures)

    def handle_event(
        self,
        topic: str = "manual",
        message: str = "",
        event_at: datetime | None = None,
    ) -> None:
        self.ingress.handle(topic, message, event_at)

    def submit_frame(
        self,
        image: Any,
        captured_at_monotonic: float,
        captured_at_epoch: float,
        *,
        capture_sequence: int = 0,
        capture_generation: int = 0,
        lifecycle_generation: int = 0,
        source_pts: float = float("nan"),
    ) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return
        self.analysis.submit_frame(
            image,
            captured_at_monotonic,
            stop_event,
            captured_at_epoch,
            capture_sequence=capture_sequence,
            capture_generation=capture_generation,
            lifecycle_generation=lifecycle_generation,
            source_pts=source_pts,
        )

    def runtime_status(self) -> dict[str, Any]:
        return {
            "active_workers": self.active_workers(),
            "event_worker_running": self.decisions.running(),
            "event_queue_depth": self.events.queue.qsize(),
            "event_ingress_in_flight": self.ingress.in_flight(),
            "event_time_model": self.ingress.timing_status(),
            "retry_queue_depth": self.events.retry_queue_depth(),
            "events": self.events.runtime_status(),
            "generation_clean": self._generation_clean,
        }

    @staticmethod
    def _attempt(operation: Callable[[], Any], failures: list[BaseException]) -> None:
        try:
            operation()
        except BaseException as error:
            failures.append(error)

    @classmethod
    def _attempt_result(
        cls,
        operation: Callable[[], bool],
        failures: list[BaseException],
    ) -> bool:
        try:
            return bool(operation())
        except BaseException as error:
            failures.append(error)
            return False

    def _raise_failures(
        self,
        action: str,
        failures: list[BaseException],
    ) -> None:
        if not failures:
            return
        first = failures[0]
        if not isinstance(first, Exception):
            raise first
        raise RuntimeError(
            f"motion runtime failed to {action} for {self.camera_id}: "
            f"{redact_secret_text(first)}"
        ) from None
