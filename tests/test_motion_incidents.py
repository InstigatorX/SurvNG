from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock
import threading

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
    tracking.config.tracks_label.side_effect = lambda label: label != "face"
    prewarm = Mock()
    image_reader = Mock()

    def has_trackable(objects: list[dict]) -> bool:
        return bool(
            tracking.config.enabled
            and any(
                item.get("label")
                and item.get("incident_eligible") is not False
                and tracking.config.tracks_label(item.get("label"))
                for item in objects
            )
        )

    def start_tracking(event_id, event_at, objects, initial_frame):
        trackable = [
            item
            for item in objects
            if (
                item.get("label")
                and item.get("incident_eligible") is not False
                and tracking.config.tracks_label(item.get("label"))
            )
        ]
        return tracking.start(event_id, event_at, trackable, initial_frame)

    service = MotionIncidentService(
        camera_id="gate",
        decision_processor=decision,
        tracking_enabled=lambda: bool(tracking.config.enabled),
        has_trackable_objects=has_trackable,
        start_tracking=start_tracking,
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


def test_intentionally_untracked_objects_do_not_report_handoff_failure() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="snapshot.webp",
        object_detected=True,
        detected_objects=({"label": "face", "incident_eligible": True},),
    )
    service, _decision, tracking, prewarm, image_reader = _service(outcome)

    service.process("motion", "face", datetime.now(timezone.utc), {})

    prewarm.assert_called_once_with()
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
    second.config.tracks_label.return_value = True
    second.start.return_value = True
    current = [first]

    def has_trackable(objects: list[dict]) -> bool:
        tracking = current[0]
        return bool(
            tracking.config.enabled
            and any(tracking.config.tracks_label(item.get("label")) for item in objects)
        )

    def start_tracking(event_id, event_at, objects, initial_frame):
        tracking = current[0]
        trackable = [
            item
            for item in objects
            if tracking.config.tracks_label(item.get("label"))
        ]
        return tracking.start(event_id, event_at, trackable, initial_frame)

    service = MotionIncidentService(
        camera_id="gate",
        decision_processor=decision,
        tracking_enabled=lambda: bool(current[0].config.enabled),
        has_trackable_objects=has_trackable,
        start_tracking=start_tracking,
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


def test_post_persistence_tracking_configuration_failure_cannot_replay_incident() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=True,
        detected_objects=({"label": "person"},),
    )
    service, decision, tracking, _prewarm, _image_reader = _service(outcome)
    tracking.config.tracks_label.side_effect = RuntimeError("invalid tracking labels")

    result = service.process("motion", "person", datetime.now(timezone.utc), {})

    assert result is outcome
    decision.handle.assert_called_once()
    tracking.start.assert_not_called()
    failure = service.status()["last_handoff_failure"]
    assert failure["event_id"] == 42
    assert failure["error_type"] == "RuntimeError"


def test_prewarm_failure_does_not_prevent_detection_or_persistence() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=False,
    )
    service, decision, _tracking, prewarm, _image_reader = _service(outcome)
    prewarm.side_effect = RuntimeError("main capture unavailable")

    result = service.process("motion", "person", datetime.now(timezone.utc), {})

    assert result is outcome
    decision.handle.assert_called_once()
    status = service.status()
    assert status["prewarm_failures"] == 1
    assert status["last_prewarm_failure"]["error_type"] == "RuntimeError"
    assert status["handoff_failures"] == 0


def test_late_refinement_runs_off_decision_path_and_completes_before_shutdown() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="initial.webp",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=84,
        snapshot_path="refined.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    service, decision, tracking, _prewarm, image_reader = _service(initial)
    decision.refine.return_value = refined
    tracking.start.return_value = True
    image_reader.return_value = np.ones((20, 20, 3), dtype=np.uint8)
    completed = threading.Event()
    stop = threading.Event()
    service.start(stop)

    outcome = service.process(
        "adaptive/visual_backup",
        "motion",
        datetime.now(timezone.utc),
        {
            "accepted": True,
            "telemetry": {
                "schema_version": 1,
                "origins": {"qualification": "global"},
                "graphs": {"qualification": {"configuration": ["large"]}},
            },
        },
        require_eligible_object=True,
        refinement_callback=lambda value: completed.set() if value is refined else None,
    )

    assert outcome is initial
    assert completed.wait(1.0)
    decision.refine.assert_called_once()
    refinement_qualification = decision.refine.call_args.args[3]
    assert refinement_qualification["refinement_payload_compacted"] is True
    assert "graphs" not in refinement_qualification["telemetry"]
    assert refinement_qualification["telemetry"]["origins"] == {
        "qualification": "global"
    }
    assert decision.refine.call_args.kwargs["existing_event_id"] is None
    tracking.start.assert_called_once()
    assert service.status()["refinements_completed"] == 1
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)
    assert not service.running()


def test_duplicate_refinement_for_same_episode_is_coalesced() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
    )
    service, decision, _tracking, _prewarm, _image_reader = _service(initial)
    entered = threading.Event()
    release = threading.Event()

    def refine(*_args, **_kwargs):
        entered.set()
        assert release.wait(1.0)
        return refined

    decision.refine.side_effect = refine
    stop = threading.Event()
    service.start(stop)
    event_at = datetime.now(timezone.utc)
    service.process("motion", "first", event_at, {"motion_episode_sequence": 7})
    assert entered.wait(1.0)
    service.process("motion", "duplicate", event_at, {"motion_episode_sequence": 7})

    status = service.status()
    assert status["refinements_coalesced"] == 1
    assert status["refinement_pending_episodes"] == 1
    release.set()
    for _ in range(100):
        if service.status()["refinements_completed"]:
            break
        threading.Event().wait(0.01)
    assert decision.refine.call_count == 1
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_coalesced_refinement_does_not_duplicate_initial_tracking_handoff() -> None:
    initial = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="initial.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="refined.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    service, decision, tracking, _prewarm, image_reader = _service(initial)
    image_reader.return_value = np.ones((10, 10, 3), dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()

    def refine(*_args, **_kwargs):
        entered.set()
        assert release.wait(1.0)
        return refined

    decision.refine.side_effect = refine
    tracking.start.return_value = True
    stop = threading.Event()
    service.start(stop)
    event_at = datetime.now(timezone.utc)
    service.process("motion", "first", event_at, {"motion_episode_sequence": 9})
    assert entered.wait(1.0)
    service.process("motion", "duplicate", event_at, {"motion_episode_sequence": 9})
    tracking.start.assert_not_called()

    release.set()
    for _ in range(100):
        if service.status()["refinements_completed"]:
            break
        threading.Event().wait(0.01)
    tracking.start.assert_called_once()
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_full_refinement_queue_supersedes_oldest_optional_episode() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
    )
    service, decision, _tracking, _prewarm, _image_reader = _service(initial)
    entered = threading.Event()
    release = threading.Event()
    refined_sequences: list[int] = []

    def refine(_topic, _message, _event_at, qualification, **_kwargs):
        sequence = qualification["motion_episode_sequence"]
        refined_sequences.append(sequence)
        if sequence == 1:
            entered.set()
            assert release.wait(1.0)
        return refined

    decision.refine.side_effect = refine
    stop = threading.Event()
    service.start(stop)
    event_at = datetime.now(timezone.utc)
    service.process("motion", "1", event_at, {"motion_episode_sequence": 1})
    assert entered.wait(1.0)
    for sequence in (2, 3, 4, 5):
        service.process(
            "motion",
            str(sequence),
            event_at,
            {"motion_episode_sequence": sequence},
        )

    assert service.status()["refinements_superseded"] == 1
    release.set()
    for _ in range(100):
        if service.status()["refinements_completed"] == 4:
            break
        threading.Event().wait(0.01)
    assert refined_sequences == [1, 3, 4, 5]
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_unavailable_refinement_worker_preserves_initial_tracking_handoff() -> None:
    initial = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="initial.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
        refinement_pending=True,
    )
    service, decision, tracking, _prewarm, image_reader = _service(initial)
    image_reader.return_value = np.ones((20, 20, 3), dtype=np.uint8)

    outcome = service.process(
        "motion",
        "person",
        datetime.now(timezone.utc),
        {},
    )

    assert outcome is initial
    decision.refine.assert_not_called()
    tracking.start.assert_called_once()
    assert service.status()["refinements_dropped"] == 1


def test_failed_refinement_preserves_initial_handoff_and_reports_cause() -> None:
    initial = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="initial.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
        refinement_pending=True,
    )
    service, decision, tracking, _prewarm, image_reader = _service(initial)
    decision.refine.side_effect = RuntimeError("recorded frame unavailable")
    image_reader.return_value = np.ones((20, 20, 3), dtype=np.uint8)
    stop = threading.Event()
    service.start(stop)

    service.process("motion", "person", datetime.now(timezone.utc), {})

    deadline = threading.Event()
    for _ in range(100):
        if service.status()["refinement_failures"]:
            break
        deadline.wait(0.01)
    status = service.status()
    assert status["refinement_failures"] == 1
    assert status["last_refinement_failure"]["error_type"] == "RuntimeError"
    assert status["last_refinement_failure"]["error"] == "recorded frame unavailable"
    tracking.start.assert_called_once()
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_refinement_callback_failure_does_not_reclassify_completed_refinement() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="initial.webp",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=84,
        snapshot_path="refined.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    service, decision, _tracking, _prewarm, _image_reader = _service(initial)
    decision.refine.return_value = refined
    stop = threading.Event()
    service.start(stop)

    service.process(
        "motion",
        "person",
        datetime.now(timezone.utc),
        {},
        refinement_callback=lambda _value: (_ for _ in ()).throw(
            RuntimeError("audit callback failed")
        ),
    )

    waiter = threading.Event()
    for _ in range(100):
        if service.status()["refinements_completed"]:
            break
        waiter.wait(0.01)
    status = service.status()
    assert status["refinements_completed"] == 1
    assert status["refinement_failures"] == 0
    assert status["refinement_callback_failures"] == 1
    assert status["last_refinement_callback_failure"]["error_type"] == "RuntimeError"
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)
