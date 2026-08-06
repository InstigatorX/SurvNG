from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import Mock

from survng.app.motion import MotionQualificationResult
from survng.app.motion_decisions import (
    MotionDecisionHooks,
    MotionDecisionOrchestrator,
    audit_features,
    is_borderline_candidate,
    priority_motion_topic,
    should_verify_suppression,
)
from survng.app.motion_events import MotionEventCoordinator, MotionTrigger


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

    def process(*_args: object, **_kwargs: object) -> dict[str, object]:
        stop_event.set()
        return {"event_id": 42, "object_detected": True, "snapshot_path": ""}

    process_incident.side_effect = process
    publish_event = Mock()
    record_stats = Mock()
    complete_adaptive = Mock()
    orchestrator = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        burst_quiet_seconds=lambda: 0.0,
        hooks=MotionDecisionHooks(
            motion_settings=lambda: ("off", "default", 640),
            rescue_settings=lambda: (False, 0.0),
            suppression_verification_rate=lambda: 0.0,
            matches_recent_priority_motion=lambda _observed_at: False,
            qualify_motion_burst=Mock(),
            with_pipeline_telemetry=lambda result: result,
            process_incident=process_incident,
            sample_rejected_motion=lambda _event_at, _result: "",
            related_incident_event_id=lambda _result: None,
            reset_motion_fusion_runtime=Mock(),
            record_visual_camera_match=lambda _observed_at: False,
            complete_adaptive_trigger=complete_adaptive,
            set_active_incident_event_id=Mock(),
            publish_event=publish_event,
            record_decision_stats=record_stats,
            increment_stat=Mock(),
        ),
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
    publish_event.assert_called_once()
    record_stats.assert_called_once()
    complete_adaptive.assert_called_once()
    assert events.active_triggers is None
