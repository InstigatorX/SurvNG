from __future__ import annotations

import hashlib
import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from .motion import MotionQualificationResult
from .motion_events import (
    MotionEventCoordinator,
    MotionTrigger,
    MotionTriggerBatch,
    RetryDisposition,
)

LOGGER = logging.getLogger(__name__)

INCIDENT_ACTIVITY_REASONS = frozenset({"event_state_active", "event_state_cooldown"})
ENFORCING_MODES = frozenset({"camera", "camera_rescue", "adaptive", "enforce"})
AUDITED_MODES = frozenset({"audit", *ENFORCING_MODES})


class MotionAuditRecorder(Protocol):
    def record_audit(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MotionDecisionHooks:
    motion_settings: Callable[[], tuple[str, str, int]]
    rescue_settings: Callable[[], tuple[bool, float]]
    suppression_verification_rate: Callable[[], float]
    matches_recent_priority_motion: Callable[[float], bool]
    qualify_motion_burst: Callable[
        [datetime, float, str],
        tuple[MotionQualificationResult, dict[str, Any]],
    ]
    with_pipeline_telemetry: Callable[
        [MotionQualificationResult], MotionQualificationResult
    ]
    process_incident: Callable[..., dict[str, Any]]
    sample_rejected_motion: Callable[[datetime, MotionQualificationResult], str]
    related_incident_event_id: Callable[[MotionQualificationResult], int | None]
    reset_motion_fusion_runtime: Callable[[], None]
    record_visual_camera_match: Callable[[float], bool]
    complete_adaptive_trigger: Callable[[MotionTriggerBatch], None]
    set_active_incident_event_id: Callable[[int], None]
    publish_event: Callable[[str, dict[str, Any]], None]
    record_decision_stats: Callable[..., None]
    increment_stat: Callable[[str, int], None]


def priority_motion_topic(topic: str) -> bool:
    searchable = topic.lower()
    return searchable.startswith("manual") or any(
        word in searchable
        for word in ("person", "people", "human", "vehicle", "animal", "face")
    )


def should_verify_suppression(decision_id: str, rate: float) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    sample = int.from_bytes(
        hashlib.sha256(decision_id.encode("utf-8")).digest()[:8],
        "big",
    ) / float(2**64)
    return sample < rate


def is_borderline_candidate(
    result: MotionQualificationResult,
    enabled: bool,
    margin: float,
) -> bool:
    return bool(
        enabled
        and not result.accepted
        and result.reason == "low_score"
        and result.score >= max(0.0, result.threshold - margin)
    )


def audit_features(result: MotionQualificationResult) -> dict[str, Any]:
    features = dict(result.features)
    if result.telemetry:
        features["pipeline_telemetry"] = result.telemetry
    return features


class MotionDecisionOrchestrator:
    """Own the trigger-to-audit/incident policy for one camera.

    Frame production and qualification remain injected capabilities. This class
    owns ordering, retries, suppression/rescue policy, durable-event boundaries,
    and audit correlation without knowing how any of those capabilities work.
    """

    def __init__(
        self,
        *,
        camera_id: str,
        events: MotionEventCoordinator,
        audit_recorder: MotionAuditRecorder,
        hooks: MotionDecisionHooks,
        burst_quiet_seconds: Callable[[], float],
    ) -> None:
        self._camera_id = camera_id
        self._events = events
        self._audit_recorder = audit_recorder
        self._hooks = hooks
        self._burst_quiet_seconds = burst_quiet_seconds

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.run_until_error(stop_event)
                return
            except Exception:
                failed_triggers = self._events.take_failed_active()
                self._hooks.increment_stat("event_worker_errors", 1)
                LOGGER.exception("motion event cycle failed for %s", self._camera_id)
                if failed_triggers and not stop_event.is_set():
                    disposition = self.retry_batch(failed_triggers, stop_event)
                    if disposition == RetryDisposition.DROPPED:
                        self._hooks.complete_adaptive_trigger(failed_triggers)

    def retry_batch(
        self,
        triggers: MotionTriggerBatch | list[MotionTrigger | dict[str, Any]],
        stop_event: threading.Event,
    ) -> RetryDisposition:
        return self._events.schedule_retry(
            triggers,
            stop_event=stop_event,
            on_retry=lambda name: self._hooks.increment_stat(name, 1),
            on_drop=lambda name: self._hooks.increment_stat(name, 1),
        )

    def run_until_error(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                first = self._events.next_trigger(timeout=0.5)
            except queue.Empty:
                continue
            if first is None or stop_event.is_set():
                return
            triggers = self._events.coalesce(
                first,
                quiet_seconds=self._burst_quiet_seconds(),
                stop_event=stop_event,
            )
            if triggers is None:
                return
            self._events.set_active(triggers)
            self._process_batch(triggers, stop_event)

    def _process_batch(
        self,
        triggers: MotionTriggerBatch,
        stop_event: threading.Event,
    ) -> None:
        if stop_event.is_set():
            self._hooks.complete_adaptive_trigger(triggers)
            self._events.set_active(None)
            return
        priority_triggers = [
            item for item in triggers if priority_motion_topic(item.topic)
        ]
        representative = min(
            priority_triggers or triggers,
            key=lambda item: item.event_at,
        )
        event_at = representative.event_at
        received_at = min(item.received_at for item in triggers)
        decision_id = next(
            (item.decision_id for item in triggers if item.decision_id),
            "",
        ) or uuid.uuid4().hex
        for item in triggers:
            item.decision_id = decision_id

        mode, sensitivity, frame_width = self._hooks.motion_settings()
        rescue_enabled, rescue_margin = self._hooks.rescue_settings()
        priority = bool(priority_triggers)
        adaptive_only = all(item.topic.startswith("adaptive/") for item in triggers)
        visual_backup_queued = any(
            item.topic == "adaptive/visual_backup" for item in triggers
        )
        visual_backup = adaptive_only and visual_backup_queued
        if visual_backup_queued and not visual_backup:
            matched_camera_at = self._events.latest_camera_motion()
            if self._hooks.record_visual_camera_match(matched_camera_at):
                self._hooks.increment_stat("visual_backup_onvif_matches", 1)

        result, diagnostics = self._qualification_result(
            triggers=triggers,
            event_at=event_at,
            received_at=received_at,
            mode=mode,
            sensitivity=sensitivity,
            priority=priority,
            adaptive_only=adaptive_only,
        )
        if stop_event.is_set():
            self._hooks.complete_adaptive_trigger(triggers)
            self._events.set_active(None)
            return

        borderline_candidate = is_borderline_candidate(
            result,
            rescue_enabled,
            rescue_margin,
        )
        verification_rate = self._hooks.suppression_verification_rate()
        suppression_verification_candidate = bool(
            mode in ENFORCING_MODES
            and not result.accepted
            and not borderline_candidate
            and not result.reason.startswith("event_state_")
            and should_verify_suppression(decision_id, verification_rate)
        )
        qualification = {
            **result.as_dict(),
            **diagnostics,
            "mode": mode,
            "sensitivity": sensitivity,
            "frame_width": frame_width,
            "borderline_rescue_enabled": rescue_enabled,
            "borderline_margin": rescue_margin,
            "borderline_candidate": borderline_candidate,
            "suppression_verification_rate": verification_rate,
            "suppression_verification_candidate": suppression_verification_candidate,
            "trigger_count": len(triggers),
            "trigger_source": (
                "visual_backup"
                if visual_backup
                else "adaptive" if adaptive_only else "camera"
            ),
            "retry_count": max((item.retry_count for item in triggers), default=0),
            "would_suppress": bool(mode in AUDITED_MODES and not result.accepted),
        }
        effective_accepted = bool(
            mode in {"off", "audit"}
            or result.accepted
            or borderline_candidate
            or suppression_verification_candidate
        )
        qualification["effective_accepted"] = effective_accepted
        retry_attempt = qualification["retry_count"] > 0
        for item in triggers:
            item.retry_qualification_result = result
            item.retry_diagnostics = dict(diagnostics)
        self._hooks.record_decision_stats(
            result=result,
            qualification=qualification,
            retry_attempt=retry_attempt,
            priority=priority,
            mode=mode,
            borderline_candidate=borderline_candidate,
            suppression_verification_candidate=suppression_verification_candidate,
        )

        if not retry_attempt:
            self._hooks.publish_event(
                "motion_qualification",
                {
                    "camera_id": self._camera_id,
                    "timestamp": event_at.isoformat(),
                    **qualification,
                },
            )
        if not effective_accepted:
            self._record_rejected_audit(
                triggers,
                decision_id=decision_id,
                event_at=event_at,
                mode=mode,
                sensitivity=sensitivity,
                result=result,
            )
            self._hooks.complete_adaptive_trigger(triggers)
            self._events.set_active(None)
            return

        self._process_accepted(
            triggers=triggers,
            representative=representative,
            event_at=event_at,
            decision_id=decision_id,
            mode=mode,
            sensitivity=sensitivity,
            result=result,
            qualification=qualification,
            visual_backup=visual_backup,
            borderline_candidate=borderline_candidate,
            suppression_verification_candidate=suppression_verification_candidate,
        )

    def _qualification_result(
        self,
        *,
        triggers: MotionTriggerBatch,
        event_at: datetime,
        received_at: float,
        mode: str,
        sensitivity: str,
        priority: bool,
        adaptive_only: bool,
    ) -> tuple[MotionQualificationResult, dict[str, Any]]:
        prequalified = [
            item.prequalified for item in triggers if item.prequalified is not None
        ]
        retry_results = [
            item.retry_qualification_result
            for item in triggers
            if item.retry_qualification_result is not None
        ]
        diagnostics: dict[str, Any] = {
            "windows_evaluated": 0,
            "event_receipt_delta_seconds": round(
                received_at - event_at.timestamp(), 3
            ),
        }
        retry_diagnostics = next(
            (
                dict(item.retry_diagnostics)
                for item in triggers
                if item.retry_diagnostics is not None
            ),
            None,
        )
        if adaptive_only and self._hooks.matches_recent_priority_motion(
            event_at.timestamp()
        ):
            result = MotionQualificationResult(
                False,
                0.0,
                1.0,
                "priority_event_deduplicated",
                0,
                {"primary_motion_source": "adaptive_background"},
            )
        elif mode == "off":
            result = MotionQualificationResult(True, 1.0, 0.0, "disabled", 0, {})
        elif priority:
            result = MotionQualificationResult(
                True,
                1.0,
                0.0,
                "priority_topic",
                0,
                {"primary_motion_source": "onvif_priority"},
            )
        elif retry_results:
            result = max(retry_results, key=lambda item: item.score)
            if retry_diagnostics is not None:
                diagnostics = retry_diagnostics
        elif prequalified:
            result = self._hooks.with_pipeline_telemetry(
                max(prequalified, key=lambda item: item.score)
            )
        else:
            result, diagnostics = self._hooks.qualify_motion_burst(
                event_at, received_at, sensitivity
            )
        return result, diagnostics

    def _record_rejected_audit(
        self,
        triggers: MotionTriggerBatch,
        *,
        decision_id: str,
        event_at: datetime,
        mode: str,
        sensitivity: str,
        result: MotionQualificationResult,
    ) -> None:
        snapshot_path = next(
            (
                item.audit_snapshot_path
                for item in triggers
                if item.audit_snapshot_path is not None
            ),
            None,
        )
        if snapshot_path is None:
            snapshot_path = (
                ""
                if result.reason in INCIDENT_ACTIVITY_REASONS
                else self._hooks.sample_rejected_motion(event_at, result)
            )
            for item in triggers:
                item.audit_snapshot_path = snapshot_path
        self._audit_recorder.record_audit(
            decision_id=decision_id,
            snapshot_path=snapshot_path,
            event_at=event_at,
            mode=mode,
            sensitivity=sensitivity,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            object_detected=None,
            trigger_count=len(triggers),
            features=audit_features(result),
            related_event_id=self._hooks.related_incident_event_id(result),
        )

    def _process_accepted(
        self,
        *,
        triggers: MotionTriggerBatch,
        representative: MotionTrigger,
        event_at: datetime,
        decision_id: str,
        mode: str,
        sensitivity: str,
        result: MotionQualificationResult,
        qualification: dict[str, Any],
        visual_backup: bool,
        borderline_candidate: bool,
        suppression_verification_candidate: bool,
    ) -> None:
        durable_incident = False
        try:
            outcome = self._hooks.process_incident(
                representative.topic,
                representative.message,
                event_at,
                qualification,
                require_eligible_object=bool(
                    visual_backup
                    or borderline_candidate
                    or suppression_verification_candidate
                ),
                require_motion_correlation=visual_backup,
            )
            event_id = outcome.get("event_id")
            if event_id is not None:
                durable_incident = True
                # A persisted incident is the idempotency boundary. Failures in
                # later audits or notifications must never replay detection.
                self._events.set_active(None)
                self._hooks.set_active_incident_event_id(int(event_id))
            object_outcome = outcome.get("object_detected")
            found_object = object_outcome is True
            if borderline_candidate and found_object:
                self._hooks.increment_stat("borderline_rescues", 1)
            elif mode in ENFORCING_MODES and borderline_candidate:
                self._hooks.increment_stat("suppressed", 1)
            if suppression_verification_candidate:
                self._hooks.increment_stat("suppression_verification_checks", 1)
                self._hooks.increment_stat(
                    "suppression_verification_rescues" if found_object else "suppressed",
                    1,
                )
            if mode == "audit" and not result.accepted and found_object:
                self._hooks.increment_stat("audit_object_matches", 1)
            if visual_backup:
                self._record_visual_backup_audit(
                    triggers=triggers,
                    decision_id=decision_id,
                    event_at=event_at,
                    mode=mode,
                    sensitivity=sensitivity,
                    result=result,
                    outcome=outcome,
                    event_id=event_id,
                    object_outcome=object_outcome,
                    found_object=found_object,
                )
            elif mode in AUDITED_MODES and not result.accepted:
                audit_snapshot_path = str(outcome.get("snapshot_path") or "")
                if not audit_snapshot_path and event_id is None:
                    audit_snapshot_path = self._hooks.sample_rejected_motion(
                        event_at, result
                    )
                self._audit_recorder.record_audit(
                    event_id=int(event_id) if event_id is not None else None,
                    decision_id=decision_id if event_id is None else "",
                    snapshot_path=audit_snapshot_path,
                    event_at=event_at,
                    mode=mode,
                    sensitivity=sensitivity,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    object_detected=object_outcome,
                    trigger_count=len(triggers),
                    features={
                        **audit_features(result),
                        "suppression_verification": suppression_verification_candidate,
                    },
                )
        except Exception:
            if durable_incident:
                self._hooks.complete_adaptive_trigger(triggers)
            raise
        else:
            self._hooks.complete_adaptive_trigger(triggers)
            self._events.set_active(None)

    def _record_visual_backup_audit(
        self,
        *,
        triggers: MotionTriggerBatch,
        decision_id: str,
        event_at: datetime,
        mode: str,
        sensitivity: str,
        result: MotionQualificationResult,
        outcome: dict[str, Any],
        event_id: Any,
        object_outcome: Any,
        found_object: bool,
    ) -> None:
        if result.features.get("illumination_verification_probe") and found_object:
            self._hooks.increment_stat("illumination_verification_rescues", 1)
        correlation = outcome.get("motion_correlation")
        if (
            outcome.get("rejection_reason") == "object_not_motion_correlated"
            and isinstance(correlation, dict)
        ):
            self._hooks.increment_stat(
                "visual_backup_uncorrelated_objects",
                int(correlation.get("eligible_object_count") or 0),
            )
        if not found_object:
            self._hooks.reset_motion_fusion_runtime()
        self._audit_recorder.record_audit(
            event_id=int(event_id) if event_id is not None else None,
            decision_id=decision_id if event_id is None else "",
            snapshot_path=str(outcome.get("snapshot_path") or ""),
            event_at=event_at,
            mode=mode,
            sensitivity=sensitivity,
            score=result.score,
            threshold=result.threshold,
            reason=str(outcome.get("rejection_reason") or "visual_backup_trigger"),
            object_detected=object_outcome,
            trigger_count=len(triggers),
            features={
                **audit_features(result),
                "visual_backup_original_reason": result.reason,
                "motion_correlation": correlation,
            },
            category="visual_backup",
        )
