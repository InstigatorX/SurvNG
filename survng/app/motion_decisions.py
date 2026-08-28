from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from .config import MotionQualificationConfig
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


class MotionDecisionQualification(Protocol):
    def settings(self) -> tuple[str, str, int]: ...
    def rescue_settings(self) -> tuple[bool, float]: ...
    def suppression_verification_rate(self) -> float: ...
    def qualify_burst(
        self,
        event_at: datetime,
        received_at: float,
        sensitivity: str,
        sample_source: Any,
    ) -> tuple[MotionQualificationResult, dict[str, Any]]: ...
    def with_pipeline_telemetry(
        self, result: MotionQualificationResult
    ) -> MotionQualificationResult: ...

class MotionDecisionIncidents(Protocol):
    def process(self, *args: Any, **kwargs: Any) -> Any: ...


class MotionDecisionMedia(Protocol):
    def sample_rejected_motion(
        self, event_at: datetime, result: MotionQualificationResult
    ) -> str: ...


class MotionDecisionState(Protocol):
    def publish_event(self, event_type: str, payload: dict[str, Any]) -> None: ...
    def record_decision(self, **kwargs: Any) -> None: ...
    def increment_stat(self, name: str, amount: int = 1) -> None: ...


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


def is_confident_nuisance(result: MotionQualificationResult) -> bool:
    """Return true only for EMA rejections backed by categorical evidence."""
    features = result.features
    if result.reason == "global_illumination_change":
        return float(features.get("global_change", 0.0)) >= float(
            features.get("global_change_threshold", 1.0)
        )
    if result.reason == "stationary_region":
        return bool(
            int(features.get("stationary_region_count", 0)) > 0
            and float(features.get("motion_progress", 1.0)) < 0.55
            and float(features.get("robust_displacement", 1.0))
            < float(features.get("stationary_max_displacement_threshold", 0.0))
        )
    if result.reason == "persistent_scene_motion":
        return bool(
            float(features.get("motion_progress", 1.0)) < 0.45
            and float(features.get("direction_consistency", 1.0)) < 0.55
            and float(features.get("containment_radius", 1.0))
            < max(
                0.05,
                float(features.get("stationary_containment_threshold", 0.0)) * 2.0,
            )
        )
    return False


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
        config: MotionQualificationConfig,
        qualification: MotionDecisionQualification,
        incidents: MotionDecisionIncidents,
        media: MotionDecisionMedia,
        analysis: MotionDecisionAnalysis,
        state: MotionDecisionState,
    ) -> None:
        self._camera_id = camera_id
        self._events = events
        self._audit_recorder = audit_recorder
        self._config = config
        self._qualification = qualification
        self._incidents = incidents
        self._media = media
        self._analysis = analysis
        self._state = state
        self._thread: threading.Thread | None = None
        self._refinement_completion_lock = threading.Lock()
        self._completed_refinement_contexts: set[str] = set()
        completion_setter = getattr(
            self._incidents,
            "set_refinement_completion_handler",
            None,
        )
        if callable(completion_setter):
            completion_setter(self._record_refined_outcome_from_context)

    def start(self, stop_event: threading.Event) -> None:
        """Start the sole decision worker owned by this orchestrator."""
        if self.running():
            return
        thread = threading.Thread(
            target=self.run,
            args=(stop_event,),
            name=f"motion-{self._camera_id}",
            daemon=False,
        )
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._thread = None
            raise

    def request_stop(self) -> None:
        self._events.signal_stop()

    def wait_stopped(self, timeout: float) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        self._thread = None
        return True

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.run_until_error(stop_event)
                return
            except Exception:
                failed_triggers = self._events.take_failed_active()
                self._state.increment_stat("event_worker_errors", 1)
                LOGGER.exception("motion event cycle failed for %s", self._camera_id)
                if failed_triggers:
                    if stop_event.is_set():
                        self._abort_episode_intents(failed_triggers)
                        self._events.release_deliveries(failed_triggers)
                    else:
                        disposition = self.retry_batch(failed_triggers, stop_event)
                        if disposition == RetryDisposition.DROPPED:
                            self._fail_episode_intents(failed_triggers)
                            self._events.fail_deliveries(
                                failed_triggers,
                                "motion decision retry limit exceeded",
                            )
                        elif disposition == RetryDisposition.STOPPED:
                            self._abort_episode_intents(failed_triggers)
                            self._events.release_deliveries(failed_triggers)

    def retry_batch(
        self,
        triggers: MotionTriggerBatch | list[MotionTrigger | dict[str, Any]],
        stop_event: threading.Event,
    ) -> RetryDisposition:
        return self._events.schedule_retry(
            triggers,
            stop_event=stop_event,
            on_retry=lambda name: self._state.increment_stat(name, 1),
            on_drop=lambda name: self._state.increment_stat(name, 1),
        )

    def run_until_error(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                first = self._events.next_trigger(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                return
            if stop_event.is_set():
                claimed = MotionTriggerBatch((first,))
                self._abort_episode_intents(claimed)
                self._events.release_deliveries(claimed)
                return
            mode, _sensitivity, _frame_width = self._qualification.settings()
            immediate_camera_primary = bool(
                mode == "camera_rescue"
                and not first.topic.startswith("adaptive/")
                and first.retry_batch is None
            )
            triggers = self._events.coalesce(
                first,
                quiet_seconds=(
                    0.0
                    if immediate_camera_primary
                    else self._config.burst_quiet_seconds
                ),
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
            if not any(item.retry_count > 0 for item in triggers):
                self._abort_episode_intents(triggers)
            self._events.release_deliveries(triggers)
            self._complete_adaptive_trigger(triggers, complete_delivery=False)
            self._events.set_active(None)
            return
        self._mark_episode_intents_running(triggers)
        authoritative_sources = {
            source.value
            for item in triggers
            if item.detection_intent_id
            for intent in (self._events.episode_controller.intent(item.detection_intent_id),)
            if intent is not None
            for source in intent.sources
        }
        priority_triggers = [
            item for item in triggers if priority_motion_topic(item.topic)
        ]
        representative = min(
            priority_triggers or triggers,
            key=lambda item: item.event_at,
        )
        route_trigger = next(
            (
                item
                for item in triggers
                if item.prequalified is not None
                and isinstance(
                    item.prequalified.features.get("route_detection_watch"), dict
                )
            ),
            None,
        )
        event_at = representative.event_at
        received_at = min(item.received_at for item in triggers)
        decision_id = next(
            (item.decision_id for item in triggers if item.decision_id),
            "",
        ) or uuid.uuid4().hex
        for item in triggers:
            item.decision_id = decision_id

        mode, sensitivity, frame_width = self._qualification.settings()
        rescue_enabled, rescue_margin = self._qualification.rescue_settings()
        priority = bool(priority_triggers)
        adaptive_only = bool(
            all(item.topic.startswith("adaptive/") for item in triggers)
            and not authoritative_sources.intersection({"camera", "manual"})
        )
        visual_backup_queued = any(
            item.topic == "adaptive/visual_backup" for item in triggers
        )
        visual_backup = adaptive_only and visual_backup_queued
        active_followup_queued = any(
            item.topic == "adaptive/active_followup" for item in triggers
        )
        active_followup = adaptive_only and active_followup_queued
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
            self._abort_episode_intents(triggers)
            self._events.release_deliveries(triggers)
            self._complete_adaptive_trigger(triggers, complete_delivery=False)
            self._events.set_active(None)
            return

        borderline_candidate = is_borderline_candidate(
            result,
            rescue_enabled,
            rescue_margin,
        )
        verification_rate = self._qualification.suppression_verification_rate()
        confident_nuisance = is_confident_nuisance(result)
        suppression_verification_candidate = bool(
            mode in ENFORCING_MODES
            and not result.accepted
            and not borderline_candidate
            and not result.reason.startswith("event_state_")
            and (
                (
                    not adaptive_only
                    and not confident_nuisance
                )
                or should_verify_suppression(decision_id, verification_rate)
            )
        )
        camera_uncertainty_verification = bool(
            suppression_verification_candidate
            and not adaptive_only
            and not confident_nuisance
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
            "camera_uncertainty_verification": camera_uncertainty_verification,
            "confident_nuisance": confident_nuisance,
            "trigger_count": len(triggers),
            "trigger_source": (
                "visual_backup"
                if visual_backup
                else "active_followup"
                if active_followup
                else "adaptive" if adaptive_only else "camera"
            ),
            "retry_count": max((item.retry_count for item in triggers), default=0),
            "trigger_received_at_epoch": received_at,
            "event_timing": (
                representative.event_timing.to_payload()
                if representative.event_timing is not None
                else {
                    "sampling_at": event_at.isoformat(),
                    "received_at": datetime.fromtimestamp(
                        received_at, timezone.utc
                    ).isoformat(),
                    "camera_event_at": None,
                    "selection_reason": "legacy_trigger",
                }
            ),
            "would_suppress": bool(mode in AUDITED_MODES and not result.accepted),
            "motion_episode_sequence": self._events.current_episode_sequence(),
            "motion_episode_managed": bool(
                representative.detection_intent_id
                and self._events.episode_controller.intent(
                    representative.detection_intent_id
                )
                is not None
            ),
            "detection_intent_id": representative.delivery_job_id,
            "evidence_frame_at_epoch": representative.evidence_frame_at_epoch,
            "evidence_frame_sequence": representative.evidence_frame_sequence,
            "evidence_capture_generation": representative.evidence_capture_generation,
            "evidence_lifecycle_generation": representative.lifecycle_generation,
            "evidence_frame_required": bool(
                representative.prequalified is not None
                and representative.prequalified.features.get("ema_v2")
            ),
        }
        if route_trigger is not None and route_trigger.prequalified is not None:
            # The representative controls the camera-primary presentation and
            # timing, but route provenance controls the durable occurrence
            # identity and target admission.  Do not lose it when sources
            # coalesce into the same decision batch.
            route_features = route_trigger.prequalified.features
            qualification["features"] = {
                **qualification["features"],
                "route_detection_watch": dict(
                    route_features["route_detection_watch"]
                ),
            }
            route_intent_id = (
                route_trigger.delivery_job_id or route_trigger.detection_intent_id
            )
            if route_intent_id:
                qualification["detection_intent_id"] = route_intent_id
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
        self._state.record_decision(
            result=result,
            qualification=qualification,
            retry_attempt=retry_attempt,
            priority=priority,
            mode=mode,
            borderline_candidate=borderline_candidate,
            suppression_verification_candidate=suppression_verification_candidate,
        )

        if not retry_attempt:
            self._state.publish_event(
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
            self._complete_adaptive_trigger(triggers)
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
            active_followup=active_followup,
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
        if mode == "off":
            result = MotionQualificationResult(True, 1.0, 0.0, "disabled", 0, {})
        elif mode == "camera_rescue" and not adaptive_only:
            # Camera + EMA Backup makes the camera notice authoritative. EMA
            # remains an independent rescue source and must never delay or
            # suppress the primary camera-triggered object check.
            result = MotionQualificationResult(
                True,
                1.0,
                0.0,
                "camera_primary_fast_path",
                0,
                {"primary_motion_source": "camera"},
            )
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
            result = self._qualification.with_pipeline_telemetry(
                max(prequalified, key=lambda item: item.score)
            )
        else:
            result, diagnostics = self._qualification.qualify_burst(
                event_at, received_at, sensitivity, self._analysis
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
                else self._media.sample_rejected_motion(event_at, result)
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
            related_event_id=self._related_incident_event_id(result),
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
        active_followup: bool,
        borderline_candidate: bool,
        suppression_verification_candidate: bool,
    ) -> None:
        durable_incident = False
        episode_sequence = int(qualification["motion_episode_sequence"])
        try:
            outcome = self._incidents.process(
                representative.topic,
                representative.message,
                event_at,
                qualification,
                require_eligible_object=bool(
                    visual_backup
                    or active_followup
                    or borderline_candidate
                    or suppression_verification_candidate
                ),
                require_motion_correlation=visual_backup or active_followup,
                refinement_callback=lambda refined: self._record_refined_outcome(
                    refined,
                    decision_id=decision_id,
                    event_at=event_at,
                    mode=mode,
                    sensitivity=sensitivity,
                    result=result,
                    trigger_count=len(triggers),
                    visual_backup=visual_backup,
                    active_followup=active_followup,
                    borderline_candidate=borderline_candidate,
                    suppression_verification_candidate=suppression_verification_candidate,
                    episode_sequence=episode_sequence,
                    episode_managed=bool(
                        qualification.get("motion_episode_managed")
                    ),
                ),
                refinement_completion_context={
                    "completion_id": f"{decision_id}:refinement",
                    "decision_id": decision_id,
                    "event_at": event_at.isoformat(),
                    "mode": mode,
                    "sensitivity": sensitivity,
                    "result": result.as_dict(),
                    "trigger_count": len(triggers),
                    "visual_backup": visual_backup,
                    "active_followup": active_followup,
                    "borderline_candidate": borderline_candidate,
                    "suppression_verification_candidate": (
                        suppression_verification_candidate
                    ),
                    "episode_sequence": episode_sequence,
                    "episode_managed": bool(
                        qualification.get("motion_episode_managed")
                    ),
                },
            ).as_dict()
            event_id = outcome.get("event_id")
            if event_id is not None:
                durable_incident = True
                # A persisted incident is the idempotency boundary. Failures in
                # later audits or notifications must never replay detection.
                self._events.set_active(None)
                if qualification.get("motion_episode_managed"):
                    self._events.link_incident(
                        int(event_id),
                        expected_sequence=episode_sequence,
                    )
            object_outcome = outcome.get("object_detected")
            found_object = object_outcome is True
            if borderline_candidate and found_object:
                self._state.increment_stat("borderline_rescues", 1)
            elif mode in ENFORCING_MODES and borderline_candidate:
                self._state.increment_stat("suppressed", 1)
            if suppression_verification_candidate:
                self._state.increment_stat("suppression_verification_checks", 1)
                self._state.increment_stat(
                    "suppression_verification_rescues" if found_object else "suppressed",
                    1,
                )
                if qualification.get("camera_uncertainty_verification"):
                    self._state.increment_stat("camera_uncertainty_checks", 1)
                    if found_object:
                        self._state.increment_stat("camera_uncertainty_rescues", 1)
            if mode == "audit" and not result.accepted and found_object:
                self._state.increment_stat("audit_object_matches", 1)
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
            elif active_followup:
                self._record_active_followup_audit(
                    triggers=triggers,
                    decision_id=decision_id,
                    event_at=event_at,
                    mode=mode,
                    sensitivity=sensitivity,
                    result=result,
                    outcome=outcome,
                    event_id=event_id,
                    object_outcome=object_outcome,
                )
            elif mode in AUDITED_MODES and not result.accepted:
                audit_snapshot_path = str(outcome.get("snapshot_path") or "")
                if not audit_snapshot_path and event_id is None:
                    audit_snapshot_path = self._media.sample_rejected_motion(
                        event_at, result
                    )
                self._audit_recorder.record_audit(
                    event_id=int(event_id) if event_id is not None else None,
                    decision_id=decision_id,
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
                        "object_detection_timing": outcome.get("processing_timing"),
                        "object_activity_attribution": outcome.get("object_activity"),
                    },
                )
        except Exception:
            if durable_incident:
                self._complete_adaptive_trigger(triggers)
            raise
        else:
            self._complete_adaptive_trigger(triggers)
            self._events.set_active(None)

    def _record_refined_outcome(
        self,
        refined: Any,
        *,
        decision_id: str,
        event_at: datetime,
        mode: str,
        sensitivity: str,
        result: MotionQualificationResult,
        trigger_count: int,
        visual_backup: bool,
        active_followup: bool,
        borderline_candidate: bool,
        suppression_verification_candidate: bool,
        episode_sequence: int,
        episode_managed: bool,
    ) -> None:
        outcome = refined.as_dict()
        event_id = outcome.get("event_id")
        object_outcome = outcome.get("object_detected")
        found_object = object_outcome is True
        correlation = outcome.get("motion_correlation")
        reason = str(outcome.get("rejection_reason") or result.reason)
        features = {
            **audit_features(result),
            "late_object_refinement": True,
            "motion_correlation": correlation,
            "suppression_verification": suppression_verification_candidate,
            "object_detection_timing": outcome.get("processing_timing"),
            "object_activity_attribution": outcome.get("object_activity"),
        }
        if visual_backup:
            features["visual_backup_original_reason"] = result.reason
            reason = str(outcome.get("rejection_reason") or "visual_backup_trigger")
        elif active_followup:
            reason = str(outcome.get("rejection_reason") or "active_event_followup")
        if visual_backup or active_followup or (mode in AUDITED_MODES and not result.accepted):
            self._audit_recorder.record_audit(
                event_id=int(event_id) if event_id is not None else None,
                decision_id=decision_id,
                snapshot_path=str(outcome.get("snapshot_path") or ""),
                event_at=event_at,
                mode=mode,
                sensitivity=sensitivity,
                score=result.score,
                threshold=result.threshold,
                reason=reason,
                object_detected=object_outcome,
                trigger_count=trigger_count,
                features=features,
                category=(
                    "visual_backup"
                    if visual_backup
                    else "active_followup"
                    if active_followup
                    else "qualification"
                ),
            )
        # Complete all fallible durable writes before mutating process-local
        # counters. Durable retries may safely replay the audit by decision_id
        # and incident linkage without double-counting runtime statistics.
        if event_id is not None and episode_managed:
            self._events.link_incident(
                int(event_id),
                expected_sequence=episode_sequence,
            )
        if found_object:
            self._state.increment_stat("late_object_rescues", 1)
            if borderline_candidate:
                self._state.increment_stat("borderline_rescues", 1)
            if suppression_verification_candidate:
                self._state.increment_stat("suppression_verification_rescues", 1)
        if active_followup:
            if object_outcome is True:
                self._state.increment_stat("active_followup_objects", 1)
            elif object_outcome is False:
                self._state.increment_stat("active_followup_no_object", 1)

    def _record_refined_outcome_from_context(
        self,
        refined: Any,
        context: dict[str, Any],
    ) -> None:
        """Rebuild refinement finalization after a durable-job restart."""
        completion_id = str(
            context.get("completion_id") or context.get("decision_id") or ""
        )
        if not completion_id:
            raise ValueError("durable refinement completion is missing an id")
        with self._refinement_completion_lock:
            if completion_id in self._completed_refinement_contexts:
                return
            self._record_refined_outcome_from_context_once(refined, context)
            self._completed_refinement_contexts.add(completion_id)

    def _record_refined_outcome_from_context_once(
        self,
        refined: Any,
        context: dict[str, Any],
    ) -> None:
        raw_result = dict(context.get("result") or {})
        result = MotionQualificationResult(
            accepted=bool(raw_result.get("accepted")),
            score=float(raw_result.get("score", 0.0)),
            threshold=float(raw_result.get("threshold", 0.0)),
            reason=str(raw_result.get("reason") or "refinement"),
            frame_count=int(raw_result.get("frame_count", 0)),
            features=dict(raw_result.get("features") or {}),
            telemetry=dict(raw_result.get("telemetry") or {}),
        )
        event_at = datetime.fromisoformat(str(context["event_at"]))
        self._record_refined_outcome(
            refined,
            decision_id=str(context.get("decision_id") or ""),
            event_at=event_at,
            mode=str(context.get("mode") or ""),
            sensitivity=str(context.get("sensitivity") or ""),
            result=result,
            trigger_count=int(context.get("trigger_count", 0)),
            visual_backup=bool(context.get("visual_backup")),
            active_followup=bool(context.get("active_followup")),
            borderline_candidate=bool(context.get("borderline_candidate")),
            suppression_verification_candidate=bool(
                context.get("suppression_verification_candidate")
            ),
            episode_sequence=int(context.get("episode_sequence", 0)),
            episode_managed=bool(context.get("episode_managed")),
        )

    def _record_active_followup_audit(
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
    ) -> None:
        correlation = outcome.get("motion_correlation")
        if not bool(outcome.get("refinement_pending")):
            if object_outcome is True:
                self._state.increment_stat("active_followup_objects", 1)
            elif object_outcome is False:
                self._state.increment_stat("active_followup_no_object", 1)
        self._audit_recorder.record_audit(
            event_id=int(event_id) if event_id is not None else None,
            related_event_id=(
                None if event_id is not None else self._events.active_incident_event_id()
            ),
            decision_id=decision_id,
            snapshot_path=str(outcome.get("snapshot_path") or ""),
            event_at=event_at,
            mode=mode,
            sensitivity=sensitivity,
            score=result.score,
            threshold=result.threshold,
            reason=str(outcome.get("rejection_reason") or "active_event_followup"),
            object_detected=object_outcome,
            trigger_count=len(triggers),
            features={
                **audit_features(result),
                "motion_correlation": correlation,
                "object_detection_timing": outcome.get("processing_timing"),
                "object_activity_attribution": outcome.get("object_activity"),
            },
            category="active_followup",
        )

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
            self._state.increment_stat("illumination_verification_rescues", 1)
        correlation = outcome.get("motion_correlation")
        if (
            outcome.get("rejection_reason") == "object_not_motion_correlated"
            and isinstance(correlation, dict)
        ):
            self._state.increment_stat(
                "visual_backup_uncorrelated_objects",
                int(correlation.get("eligible_object_count") or 0),
            )
        self._audit_recorder.record_audit(
            event_id=int(event_id) if event_id is not None else None,
            decision_id=decision_id,
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
                "object_detection_timing": outcome.get("processing_timing"),
                "object_activity_attribution": outcome.get("object_activity"),
                "cover_promotion": {
                    "promoted": bool(outcome.get("cover_promoted")),
                    "reason": str(outcome.get("cover_promotion_reason") or ""),
                },
            },
            category="visual_backup",
        )

    def _complete_adaptive_trigger(
        self,
        triggers: MotionTriggerBatch,
        *,
        complete_delivery: bool = True,
    ) -> None:
        completed_monotonic = time.monotonic()
        for intent_id in {
            item.detection_intent_id for item in triggers if item.detection_intent_id
        }:
            try:
                self._events.episode_controller.complete(
                    intent_id,
                    occurred_monotonic=completed_monotonic,
                )
            except ValueError:
                # A stopped generation may already have aborted the reservation.
                pass
        if complete_delivery:
            self._events.complete_deliveries(triggers)

    def _mark_episode_intents_running(self, triggers: MotionTriggerBatch) -> None:
        started_monotonic = time.monotonic()
        for intent_id in {
            item.detection_intent_id for item in triggers if item.detection_intent_id
        }:
            try:
                self._events.episode_controller.mark_running(
                    intent_id,
                    occurred_monotonic=started_monotonic,
                )
            except ValueError:
                pass

    def _fail_episode_intents(self, triggers: MotionTriggerBatch) -> None:
        failed_monotonic = time.monotonic()
        for intent_id in {
            item.detection_intent_id for item in triggers if item.detection_intent_id
        }:
            try:
                self._events.episode_controller.fail(
                    intent_id,
                    occurred_monotonic=failed_monotonic,
                )
            except ValueError:
                pass

    def _abort_episode_intents(self, triggers: MotionTriggerBatch) -> None:
        aborted_monotonic = time.monotonic()
        for intent_id in {
            item.detection_intent_id for item in triggers if item.detection_intent_id
        }:
            try:
                self._events.episode_controller.abort(
                    intent_id,
                    occurred_monotonic=aborted_monotonic,
                )
            except ValueError:
                pass

    def _related_incident_event_id(
        self,
        result: MotionQualificationResult,
    ) -> int | None:
        if result.reason not in INCIDENT_ACTIVITY_REASONS:
            return None
        return self._events.active_incident_event_id()
