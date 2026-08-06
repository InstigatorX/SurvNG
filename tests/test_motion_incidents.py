from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np

from survng.app.motion_incidents import MotionIncidentService
from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome


def _service(
    outcome: MotionDecisionOutcome,
    *,
    tracking_enabled: bool = True,
) -> tuple[MotionIncidentService, Mock, Mock, Mock, Mock]:
    decision = Mock()
    decision.handle.return_value = outcome
    tracking = Mock()
    tracking.config.enabled = tracking_enabled
    prewarm = Mock()
    image_reader = Mock()
    service = MotionIncidentService(
        camera_id="gate",
        decision_processor=decision,
        tracking_provider=lambda: tracking,
        prewarm_tracking=prewarm,
        image_reader=image_reader,
    )
    return service, decision, tracking, prewarm, image_reader


def test_process_prewarms_and_seeds_tracking_from_persisted_snapshot() -> None:
    seed = np.ones((90, 160, 3), dtype=np.uint8)
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="snapshot.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    service, decision, tracking, prewarm, image_reader = _service(outcome)
    image_reader.return_value = seed
    event_at = datetime.now(timezone.utc)

    result = service.process("motion", "person", event_at, {"accepted": True})

    assert result is outcome
    prewarm.assert_called_once_with()
    image_reader.assert_called_once_with("snapshot.webp")
    assert tracking.start.call_args.args[:3] == (
        42,
        event_at,
        [{"label": "person", "incident_eligible": True}],
    )
    assert tracking.start.call_args.args[3] is seed
    decision.handle.assert_called_once()


def test_process_does_not_track_rejected_or_objectless_outcome() -> None:
    outcome = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="snapshot.webp",
        object_detected=False,
    )
    service, _decision, tracking, prewarm, image_reader = _service(outcome)

    service.process("motion", "none", datetime.now(timezone.utc), {})

    prewarm.assert_called_once_with()
    image_reader.assert_not_called()
    tracking.start.assert_not_called()


def test_disabled_tracking_avoids_prewarm_and_snapshot_read() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="snapshot.webp",
        object_detected=True,
        detected_objects=({"label": "car"},),
    )
    service, _decision, tracking, prewarm, image_reader = _service(
        outcome,
        tracking_enabled=False,
    )

    service.process("motion", "car", datetime.now(timezone.utc), {})

    prewarm.assert_not_called()
    image_reader.assert_not_called()
    tracking.start.assert_not_called()
    assert service.status()["handoff_failures"] == 0


def test_tracking_failure_after_persistence_does_not_escape_or_replay() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="snapshot.webp",
        object_detected=True,
        detected_objects=({"label": "person"},),
    )
    service, decision, tracking, _prewarm, _image_reader = _service(outcome)
    tracking.start.side_effect = RuntimeError("capacity transition failed")

    result = service.process("motion", "person", datetime.now(timezone.utc), {})

    assert result is outcome
    decision.handle.assert_called_once()
    status = service.status()
    assert status["handoff_failures"] == 1
    assert status["last_handoff_failure"]["event_id"] == 42
    assert status["last_handoff_failure"]["error_type"] == "RuntimeError"
    assert status["last_handoff_failure"]["error"] == "capacity transition failed"


def test_process_resolves_current_tracking_session_for_hot_reload() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=True,
        detected_objects=({"label": "person"},),
    )
    decision = Mock()
    decision.handle.return_value = outcome
    first = Mock()
    first.config.enabled = False
    second = Mock()
    second.config.enabled = True
    second.start.return_value = True
    current = [first]
    service = MotionIncidentService(
        camera_id="gate",
        decision_processor=decision,
        tracking_provider=lambda: current[0],
        prewarm_tracking=Mock(),
        image_reader=Mock(),
    )

    current[0] = second
    service.process("motion", "person", datetime.now(timezone.utc), {})

    first.start.assert_not_called()
    second.start.assert_called_once()


def test_enabled_tracking_decline_is_visible_in_telemetry() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=True,
        detected_objects=({"label": "person"},),
    )
    service, _decision, tracking, _prewarm, _image_reader = _service(outcome)
    tracking.start.return_value = False

    service.process("motion", "person", datetime.now(timezone.utc), {})

    failure = service.status()["last_handoff_failure"]
    assert failure["error_type"] == "TrackingDeclined"
    assert failure["event_id"] == 42


def test_handoff_failure_telemetry_redacts_stream_credentials() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=True,
        detected_objects=({"label": "person"},),
    )
    service, _decision, tracking, _prewarm, _image_reader = _service(outcome)
    tracking.start.side_effect = RuntimeError(
        "failed rtsp://camera-admin:secret@example.invalid/live"
    )

    service.process("motion", "person", datetime.now(timezone.utc), {})

    failure = service.status()["last_handoff_failure"]
    assert "secret" not in failure["error"]
    assert "camera-admin:***@" in failure["error"]
