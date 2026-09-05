from __future__ import annotations

import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from survng.app.motion import MotionQualificationResult
from survng.app.ema_v2 import CameraNotice
from survng.app.events import EventStore
from survng.app.motion_decisions import (
    COMPLETED_REFINEMENT_MEMORY,
    MotionDecisionOrchestrator,
    audit_features,
    is_confident_nuisance,
    is_borderline_candidate,
    priority_motion_topic,
    should_verify_suppression,
)
from survng.app.config import MotionQualificationConfig
from survng.app.motion_events import (
    MotionEventCoordinator,
    MotionTrigger,
    MotionTriggerBatch,
)


def test_motion_policy_helpers_are_deterministic_and_categorical() -> None:
    result = MotionQualificationResult(False, 0.47, 0.48, "low_score", 4, {})

    assert priority_motion_topic("manual")
    assert priority_motion_topic("rule/PersonDetected")
    assert not priority_motion_topic("onvif/motion")
    assert is_borderline_candidate(result, True, 0.03)
    assert not is_borderline_candidate(result, False, 0.03)
    assert should_verify_suppression("stable-decision", 0.5) == (
        should_verify_suppression("stable-decision", 0.5)
    )
    assert not should_verify_suppression("stable-decision", 0.0)
    assert should_verify_suppression("stable-decision", 1.0)


def test_confident_nuisance_requires_categorical_supporting_evidence() -> None:
    uncertain_stationary = MotionQualificationResult(
        False, 0.2, 0.5, "stationary_foreground", 3, {"motion_progress": 0.1}
    )
    learned_stationary = MotionQualificationResult(
        False,
        0.2,
        0.5,
        "stationary_region",
        3,
        {
            "stationary_region_count": 2,
            "motion_progress": 0.2,
            "robust_displacement": 0.01,
            "stationary_max_displacement_threshold": 0.03,
        },
    )
    illumination = MotionQualificationResult(
        False,
        0.1,
        0.5,
        "global_illumination_change",
        3,
        {"global_change": 0.7, "global_change_threshold": 0.55},
    )

    assert not is_confident_nuisance(uncertain_stationary)
    assert is_confident_nuisance(learned_stationary)
    assert is_confident_nuisance(illumination)


def test_audit_features_copies_inputs_and_attaches_pipeline_telemetry() -> None:
    result = MotionQualificationResult(
        False,
        0.2,
        0.5,
        "low_score",
        2,
        {"blob_count": 1},
        telemetry={"schema_version": 1},
    )

    features = audit_features(result)

    assert features == {
        "blob_count": 1,
        "pipeline_telemetry": {"schema_version": 1},
    }
    features["blob_count"] = 2
    assert result.features["blob_count"] == 1


def test_orchestrator_executes_an_off_mode_trigger_and_clears_active_batch() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2, camera_id="gate")
    stop_event = threading.Event()
    process_incident = Mock()

    def process(*_args: object, **_kwargs: object) -> Mock:
        stop_event.set()
        return Mock(as_dict=Mock(return_value={
            "event_id": 42,
            "object_detected": True,
            "snapshot_path": "",
        }))

    process_incident.side_effect = process
    incidents = Mock()
    incidents.process = process_incident
    qualification = Mock()
    qualification.settings.return_value = ("off", "default", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    state = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=state,
        model_labels=lambda: ["person", "car"],
    )
    now = datetime.now(timezone.utc)
    assert events.enqueue(
        MotionTrigger(
            topic="RuleEngine/VehicleDetect",
            message="",
            event_at=now,
            received_at=now.timestamp(),
        )
    )

    orchestrator.run_until_error(stop_event)

    process_incident.assert_called_once()
    durable_qualification = process_incident.call_args.args[3]
    assert durable_qualification["camera_semantics"]["reports"][0]["category"] == "vehicle"
    state.publish_event.assert_called_once()
    state.record_decision.assert_called_once()
    assert events.active_triggers is None


def test_camera_primary_batch_preserves_route_provenance_for_admission() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    qualification = Mock()
    qualification.settings.return_value = ("off", "default", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    incidents = Mock()
    incidents.process.return_value.as_dict.return_value = {
        "event_id": None,
        "object_detected": False,
        "snapshot_path": "",
    }
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=Mock(),
    )
    now = datetime.now(timezone.utc)
    route_id = "route:gate:back-left:58395"
    route = MotionTrigger(
        topic="adaptive/visual_backup",
        message="route watch",
        event_at=now,
        received_at=now.timestamp(),
        detection_intent_id=route_id,
        delivery_job_id=route_id,
        prequalified=MotionQualificationResult(
            True, 0.8, 0.5, "route_watch", 1,
            {"route_detection_watch": {
                "origin_camera_id": "back-left", "origin_event_id": 58395,
            }},
        ),
    )
    primary = MotionTrigger(
        topic="rule/person",
        message="camera primary",
        event_at=now,
        received_at=now.timestamp(),
    )

    orchestrator._process_batch(MotionTriggerBatch((route, primary)), threading.Event())

    persisted_qualification = incidents.process.call_args.args[3]
    assert persisted_qualification["detection_intent_id"] == route_id
    assert persisted_qualification["features"]["route_detection_watch"] == {
        "origin_camera_id": "back-left", "origin_event_id": 58395,
    }


def test_retained_episode_camera_notice_supplies_semantic_provenance() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2, camera_id="gate")
    now = datetime.now(timezone.utc)
    reserved = events.episode_controller.observe_camera(
        CameraNotice(
            "gate", now.timestamp(), 100.0, "RuleEngine/VehicleDetect", ""
        ),
        generation=0,
    )
    assert reserved.intent is not None
    events.episode_controller.acknowledge_admission(
        reserved.intent.intent_id, admitted=True, occurred_monotonic=100.1
    )
    trigger = MotionTrigger(
        topic="adaptive/visual_backup",
        message="ema",
        event_at=now,
        received_at=now.timestamp(),
        detection_intent_id=reserved.intent.intent_id,
    )
    qualification = Mock()
    qualification.settings.return_value = ("off", "balanced", 320)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    incidents = Mock()
    incidents.process.return_value = Mock(as_dict=Mock(return_value={
        "event_id": 9, "object_detected": False, "snapshot_path": "",
    }))
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate", events=events, audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification, incidents=incidents, media=Mock(),
        analysis=Mock(), state=Mock(), model_labels=lambda: ["car", "truck"],
    )

    orchestrator._process_batch(MotionTriggerBatch((trigger,)), threading.Event())

    reports = incidents.process.call_args.args[3]["camera_semantics"]["reports"]
    assert reports[0]["candidate_model_classes"] == ["car", "truck"]


def test_stopping_batch_skips_qualification_and_releases_adaptive_state() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    stop_event = threading.Event()
    stop_event.set()
    adaptive = MotionTrigger(
        topic="adaptive/motion",
        message="motion",
        event_at=datetime.now(timezone.utc),
        received_at=100.0,
    )
    batch = MotionTriggerBatch((adaptive,))
    events.set_active(batch)
    qualification = Mock()
    qualification.settings.return_value = ("camera", "default", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    state = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=state,
    )

    orchestrator._process_batch(batch, stop_event)

    qualification.qualify_burst.assert_not_called()
    assert events.active_triggers is None


def test_uncertain_camera_rejection_receives_bounded_object_verification() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    now = datetime.now(timezone.utc)
    trigger = MotionTrigger(
        topic="onvif/motion",
        message="motion",
        event_at=now,
        received_at=now.timestamp(),
        decision_id="camera-uncertain",
    )
    result = MotionQualificationResult(
        False,
        0.18,
        0.48,
        "low_persistence",
        2,
        {"persistence_seconds": 0.3},
    )
    qualification = Mock()
    qualification.settings.return_value = ("camera", "balanced", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.qualify_burst.return_value = (result, {})
    incidents = Mock()
    incidents.process.return_value = Mock(as_dict=Mock(return_value={
        "event_id": None,
        "object_detected": False,
        "snapshot_path": "",
        "rejection_reason": "no_eligible_object",
    }))
    state = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=state,
    )

    orchestrator._process_batch(MotionTriggerBatch((trigger,)), threading.Event())

    assert incidents.process.call_count == 1
    process_kwargs = incidents.process.call_args.kwargs
    assert process_kwargs["require_eligible_object"] is True
    assert process_kwargs["require_motion_correlation"] is False
    published = state.publish_event.call_args.args[1]
    assert published["camera_uncertainty_verification"] is True
    assert published["confident_nuisance"] is False
    state.increment_stat.assert_any_call("camera_uncertainty_checks", 1)


def test_camera_rescue_camera_notice_bypasses_ema_qualification() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    now = datetime.now(timezone.utc)
    qualification = Mock()
    qualification.settings.return_value = ("camera_rescue", "balanced", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    incidents = Mock()
    incidents.process.return_value = Mock(as_dict=Mock(return_value={
        "event_id": 42,
        "object_detected": True,
        "snapshot_path": "",
    }))
    state = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.5),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=state,
    )

    orchestrator._process_batch(MotionTriggerBatch((MotionTrigger(
        topic="onvif/motion",
        message="motion",
        event_at=now,
        received_at=now.timestamp(),
    ),)), threading.Event())

    qualification.qualify_burst.assert_not_called()
    published = state.publish_event.call_args.args[1]
    assert published["reason"] == "camera_primary_fast_path"
    assert published["effective_accepted"] is True


def test_camera_rescue_camera_notice_skips_burst_coalescing_delay() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    now = datetime.now(timezone.utc)
    assert events.enqueue(MotionTrigger(
        topic="onvif/motion",
        message="motion",
        event_at=now,
        received_at=now.timestamp(),
    ))
    original_coalesce = events.coalesce
    events.coalesce = Mock(side_effect=original_coalesce)
    qualification = Mock()
    qualification.settings.return_value = ("camera_rescue", "balanced", 640)
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.5),
        qualification=qualification,
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=Mock(),
    )
    stop = threading.Event()
    events.coalesce.side_effect = lambda *args, **kwargs: (
        stop.set() or None
    )

    orchestrator.run_until_error(stop)

    assert events.coalesce.call_args.kwargs["quiet_seconds"] == 0.0


def test_claimed_trigger_is_released_when_shutdown_arrives_after_claim() -> None:
    stop = threading.Event()
    now = datetime.now(timezone.utc)
    trigger = MotionTrigger(
        topic="adaptive/visual_backup",
        message="motion",
        event_at=now,
        received_at=now.timestamp(),
        delivery_job_id="claimed-1",
    )
    events = Mock()

    def claim(*_args: object, **_kwargs: object) -> MotionTrigger:
        stop.set()
        return trigger

    events.next_trigger.side_effect = claim
    events.episode_controller = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(),
        qualification=Mock(),
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=Mock(),
    )

    orchestrator.run_until_error(stop)

    events.release_deliveries.assert_called_once()
    released = events.release_deliveries.call_args.args[0]
    assert released.triggers == (trigger,)


def test_shutdown_after_durable_claim_returns_row_to_queued_state() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        stop = threading.Event()
        store = EventStore(Path(tmpdir))
        events = MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        now = datetime.now(timezone.utc)
        assert events.enqueue(MotionTrigger(
            topic="adaptive/visual_backup",
            message="motion",
            event_at=now,
            received_at=now.timestamp(),
            detection_intent_id="durable-claim-1",
        ))
        original_claim = store.claim_motion_trigger

        def claim(*args: object, **kwargs: object) -> dict | None:
            claimed = original_claim(*args, **kwargs)
            stop.set()
            return claimed

        store.claim_motion_trigger = claim  # type: ignore[method-assign]
        qualification = Mock()
        orchestrator = MotionDecisionOrchestrator(
            camera_id="gate",
            events=events,
            audit_recorder=Mock(),
            config=MotionQualificationConfig(),
            qualification=qualification,
            incidents=Mock(),
            media=Mock(),
            analysis=Mock(),
            state=Mock(),
        )

        orchestrator.run_until_error(stop)

        assert store.motion_trigger_status("gate") == {"queued": 1}


def test_durable_delivery_uses_one_persisted_retry_budget() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        events = MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        reserved = events.episode_controller.observe_camera(
            CameraNotice("gate", 100.0, 100.0, "onvif/motion"),
            generation=0,
        )
        assert reserved.intent is not None
        events.episode_controller.acknowledge_admission(
            reserved.intent.intent_id,
            admitted=True,
            occurred_monotonic=100.1,
        )
        now = datetime.now(timezone.utc)
        assert events.enqueue(MotionTrigger(
            topic="onvif/motion",
            message="motion",
            event_at=now,
            received_at=now.timestamp(),
            detection_intent_id=reserved.intent.intent_id,
        ))
        original_fail = store.fail_motion_trigger

        def fail_and_make_due(*args: object, **kwargs: object) -> bool:
            retrying = original_fail(*args, **kwargs)
            if retrying:
                with store._connect_jobs() as connection:
                    connection.execute(
                        "update motion_trigger_jobs set available_at = 0 where id = ?",
                        (args[0],),
                    )
            return retrying

        store.fail_motion_trigger = fail_and_make_due  # type: ignore[method-assign]
        qualification = Mock()
        qualification.settings.return_value = ("off", "default", 640)
        qualification.rescue_settings.return_value = (False, 0.0)
        qualification.suppression_verification_rate.return_value = 0.0
        incidents = Mock()
        incidents.process.side_effect = RuntimeError("detector unavailable")
        state = Mock()
        orchestrator = MotionDecisionOrchestrator(
            camera_id="gate",
            events=events,
            audit_recorder=Mock(),
            config=MotionQualificationConfig(burst_quiet_seconds=0.1),
            qualification=qualification,
            incidents=incidents,
            media=Mock(),
            analysis=Mock(),
            state=state,
        )
        stop = threading.Event()
        thread = threading.Thread(target=orchestrator.run, args=(stop,))
        thread.start()
        waiter = threading.Event()
        for _ in range(100):
            if store.motion_trigger_status("gate") == {"failed": 1}:
                break
            waiter.wait(0.01)
        stop.set()
        events.signal_stop()
        thread.join(1.0)

        assert not thread.is_alive()
        assert incidents.process.call_count == 3
        assert events.retry_queue_depth() == 0
        assert events.episode_snapshot()["request_status"] == "failed"
        assert store.motion_trigger_status("gate") == {"failed": 1}
        retry_stats = [
            call.args for call in state.increment_stat.call_args_list
            if call.args and call.args[0] in {"event_retries", "event_retry_drops"}
        ]
        assert retry_stats == [
            ("event_retries", 1),
            ("event_retries", 1),
            ("event_retry_drops", 1),
        ]


def test_learned_camera_nuisance_remains_suppressed_without_forced_check() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    now = datetime.now(timezone.utc)
    result = MotionQualificationResult(
        False,
        0.2,
        0.48,
        "stationary_region",
        4,
        {
            "stationary_region_count": 1,
            "motion_progress": 0.2,
            "robust_displacement": 0.01,
            "stationary_max_displacement_threshold": 0.03,
        },
    )
    qualification = Mock()
    qualification.settings.return_value = ("camera", "balanced", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.qualify_burst.return_value = (result, {})
    incidents = Mock()
    state = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=state,
    )
    trigger = MotionTrigger(
        topic="onvif/motion",
        message="motion",
        event_at=now,
        received_at=now.timestamp(),
        decision_id="camera-nuisance",
    )

    orchestrator._process_batch(MotionTriggerBatch((trigger,)), threading.Event())

    incidents.process.assert_not_called()
    published = state.publish_event.call_args.args[1]
    assert published["camera_uncertainty_verification"] is False
    assert published["confident_nuisance"] is True


def test_active_followup_requires_correlated_object_and_records_audit() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    now = datetime.now(timezone.utc)
    result = MotionQualificationResult(
        True,
        0.82,
        0.48,
        "active_event_new_motion",
        3,
        {
            "active_event_followup": True,
            "active_event_followup_anchor": 1,
            "motion_regions": [[0.55, 0.55, 0.85, 0.9]],
        },
    )
    trigger = MotionTrigger(
        topic="adaptive/active_followup",
        message="new credible motion during active EMA event",
        event_at=now,
        received_at=now.timestamp(),
        prequalified=result,
    )
    batch = MotionTriggerBatch((trigger,))
    qualification = Mock()
    qualification.settings.return_value = ("adaptive", "balanced", 320)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.with_pipeline_telemetry.return_value = result
    incidents = Mock()
    incidents.process.return_value = Mock(as_dict=Mock(return_value={
        "event_id": None,
        "object_detected": False,
        "snapshot_path": "followup.webp",
        "rejection_reason": "no_object_detected",
        "motion_correlation": {"required": True},
    }))
    audit = Mock()
    state = Mock()
    events.episode_controller.observe_camera(
        CameraNotice("camera", now.timestamp(), 100.0, "onvif/motion"),
        generation=0,
    )
    events.link_incident(42)
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=audit,
        config=MotionQualificationConfig(burst_quiet_seconds=0.1),
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=Mock(),
        state=state,
    )

    orchestrator._process_batch(batch, threading.Event())

    process_kwargs = incidents.process.call_args.kwargs
    assert process_kwargs["require_eligible_object"] is True
    assert process_kwargs["require_motion_correlation"] is True
    audit_kwargs = audit.record_audit.call_args.kwargs
    assert audit_kwargs["category"] == "active_followup"
    assert audit_kwargs["related_event_id"] == 42
    assert audit_kwargs["object_detected"] is False
    state.increment_stat.assert_any_call("active_followup_no_object", 1)


def test_refined_active_followup_updates_completed_outcome_telemetry() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    state = Mock()
    audit = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=audit,
        config=MotionQualificationConfig(),
        qualification=Mock(),
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=state,
    )
    refined = Mock()
    refined.as_dict.return_value = {
        "event_id": 55,
        "object_detected": True,
        "snapshot_path": "followup.webp",
        "rejection_reason": "",
        "motion_correlation": {"required": True},
        "depth_attribution": {"evaluated_count": 1, "would_admit_count": 1},
    }
    result = MotionQualificationResult(
        True,
        0.82,
        0.48,
        "active_event_new_motion",
        3,
        {},
    )

    orchestrator._record_refined_outcome(
        refined,
        decision_id="followup-refined",
        event_at=datetime.now(timezone.utc),
        mode="adaptive",
        sensitivity="balanced",
        result=result,
        trigger_count=1,
        visual_backup=False,
        active_followup=True,
        borderline_candidate=False,
        suppression_verification_candidate=False,
        episode_sequence=events.current_episode_sequence(),
        episode_managed=True,
    )

    state.increment_stat.assert_any_call("active_followup_objects", 1)
    audit_kwargs = audit.record_audit.call_args.kwargs
    assert audit_kwargs["category"] == "active_followup"
    assert audit_kwargs["event_id"] == 55
    assert audit_kwargs["features"]["depth_attribution"]["would_admit_count"] == 1


def test_durable_refinement_completion_is_retry_safe_and_idempotent() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    state = Mock()
    audit = Mock()
    audit.record_audit.side_effect = [RuntimeError("audit unavailable"), None]
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=audit,
        config=MotionQualificationConfig(),
        qualification=Mock(),
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=state,
    )
    refined = Mock()
    refined.as_dict.return_value = {
        "event_id": 55,
        "object_detected": True,
        "snapshot_path": "followup.webp",
        "rejection_reason": "",
        "motion_correlation": {"required": True},
    }
    event_at = datetime.now(timezone.utc)
    context = {
        "completion_id": "followup-refined:refinement",
        "decision_id": "followup-refined",
        "event_at": event_at.isoformat(),
        "mode": "adaptive",
        "sensitivity": "balanced",
        "result": MotionQualificationResult(
            True,
            0.82,
            0.48,
            "active_event_new_motion",
            3,
            {},
        ).as_dict(),
        "trigger_count": 1,
        "active_followup": True,
        "episode_sequence": events.current_episode_sequence(),
        "episode_managed": False,
    }

    with pytest.raises(RuntimeError, match="audit unavailable"):
        orchestrator._record_refined_outcome_from_context(refined, context)
    state.increment_stat.assert_not_called()

    orchestrator._record_refined_outcome_from_context(refined, context)
    orchestrator._record_refined_outcome_from_context(refined, context)

    assert audit.record_audit.call_count == 2
    assert state.increment_stat.call_count == 2
    state.increment_stat.assert_any_call("late_object_rescues", 1)
    state.increment_stat.assert_any_call("active_followup_objects", 1)


def test_durable_refinement_dedupe_memory_stays_bounded() -> None:
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=MotionQualificationConfig(),
        qualification=Mock(),
        incidents=Mock(),
        media=Mock(),
        analysis=Mock(),
        state=Mock(),
    )
    refined = Mock()
    refined.as_dict.return_value = {
        "event_id": 55,
        "object_detected": True,
        "snapshot_path": "followup.webp",
        "rejection_reason": "",
        "motion_correlation": {"required": True},
    }
    event_at = datetime.now(timezone.utc).isoformat()

    def context_for(index: int) -> dict:
        return {
            "completion_id": f"decision-{index}:refinement",
            "decision_id": f"decision-{index}",
            "event_at": event_at,
            "mode": "adaptive",
            "sensitivity": "balanced",
            "result": MotionQualificationResult(
                True,
                0.82,
                0.48,
                "active_event_new_motion",
                3,
                {},
            ).as_dict(),
            "trigger_count": 1,
            "active_followup": True,
            "episode_sequence": events.current_episode_sequence(),
            "episode_managed": False,
        }

    overflow = COMPLETED_REFINEMENT_MEMORY + 25
    for index in range(overflow):
        orchestrator._record_refined_outcome_from_context(refined, context_for(index))

    assert len(orchestrator._completed_refinement_contexts) == COMPLETED_REFINEMENT_MEMORY
    assert len(orchestrator._completed_refinement_order) == COMPLETED_REFINEMENT_MEMORY
    # Recent completions stay deduplicated so durable replays remain idempotent.
    assert "decision-0:refinement" not in orchestrator._completed_refinement_contexts
    assert f"decision-{overflow - 1}:refinement" in orchestrator._completed_refinement_contexts
