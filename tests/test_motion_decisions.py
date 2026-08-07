from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import Mock

from survng.app.motion import MotionQualificationResult
from survng.app.motion_decisions import (
    MotionDecisionOrchestrator,
    audit_features,
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
    events = MotionEventCoordinator(queue_size=4, retry_limit=2)
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
    state.active_incident_event_id.return_value = None
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
    now = datetime.now(timezone.utc)
    assert events.enqueue(
        MotionTrigger(
            topic="onvif/motion",
            message="motion",
            event_at=now,
            received_at=now.timestamp(),
        )
    )

    orchestrator.run_until_error(stop_event)

    process_incident.assert_called_once()
    state.publish_event.assert_called_once()
    state.record_decision.assert_called_once()
    assert events.active_triggers is None


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
    events.adaptive_trigger_pending = True
    qualification = Mock()
    qualification.settings.return_value = ("camera", "default", 640)
    qualification.rescue_settings.return_value = (False, 0.0)
    qualification.suppression_verification_rate.return_value = 0.0
    state = Mock()
    state.active_incident_event_id.return_value = None
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
    assert not events.adaptive_trigger_pending
    assert events.active_triggers is None
