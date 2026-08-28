from __future__ import annotations

import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.events import EventStore
from survng.app.event_store.jobs import DETECTION_JOB_MAXIMUM_AGE_SECONDS
from survng.app.motion_incidents import (
    REFINEMENT_MAX_QUEUE_AGE_SECONDS,
    MotionIncidentService,
    _MemoryDetectionJobStore,
)
from survng.app.motion_pipeline.decision_handler import MotionDecisionOutcome


def _service(
    outcome: MotionDecisionOutcome,
    *,
    tracking_enabled: bool = True,
    refinement_store: EventStore | None = None,
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
        refinement_store=refinement_store,
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


def test_initial_detection_completes_before_tracking_prewarm() -> None:
    outcome = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    service, decision, _tracking, prewarm, _image_reader = _service(outcome)
    order: list[str] = []
    decision.handle.side_effect = lambda *_args, **_kwargs: (
        order.append("detect") or outcome
    )
    prewarm.side_effect = lambda: order.append("prewarm")

    service.process("motion", "none", datetime.now(timezone.utc), {})

    assert order[:2] == ["detect", "prewarm"]


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


def test_mandatory_refinement_is_admitted_before_optional_prewarm() -> None:
    initial = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="initial.webp",
        object_detected=False,
        refinement_pending=True,
    )
    service, _decision, _tracking, prewarm, _image_reader = _service(initial)
    stop = threading.Event()
    service.start(stop)

    prewarm.side_effect = lambda: (
        service.status()["refinements_queued"] == 1
        or (_ for _ in ()).throw(AssertionError("prewarm ran before refinement admission"))
    )
    service.process("motion", "person", datetime.now(timezone.utc), {})

    prewarm.assert_called_once()
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


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
    qualification = {
        "motion_episode_sequence": 7,
        "detection_intent_id": "gate:iruntime-a:g1:e7:request:1",
    }
    service.process("motion", "first", event_at, qualification)
    assert entered.wait(1.0)
    service.process("motion", "first", event_at, qualification)

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
    qualification = {
        "motion_episode_sequence": 9,
        "detection_intent_id": "gate:iruntime-a:g1:e9:request:1",
    }
    service.process("motion", "first", event_at, qualification)
    assert entered.wait(1.0)
    service.process("motion", "first", event_at, qualification)
    # Strong provisional evidence starts tracking immediately; the duplicate
    # and later refinement must not start a second session.
    tracking.start.assert_called_once()

    release.set()
    for _ in range(100):
        if service.status()["refinements_completed"]:
            break
        threading.Event().wait(0.01)
    tracking.start.assert_called_once()
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_refinement_burst_is_durable_and_does_not_supersede_episodes() -> None:
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
    service.process(
        "motion",
        "1",
        event_at,
        {
            "motion_episode_sequence": 1,
            "detection_intent_id": "gate:iruntime-a:g1:e1:request:1",
        },
    )
    assert entered.wait(1.0)
    for sequence in (2, 3, 4, 5):
        service.process(
            "motion",
            str(sequence),
            event_at,
            {
                "motion_episode_sequence": sequence,
                "detection_intent_id": (
                    f"gate:iruntime-a:g1:e{sequence}:request:1"
                ),
            },
        )

    assert service.status()["refinement_pending_episodes"] == 5
    release.set()
    for _ in range(100):
        if service.status()["refinements_completed"] == 5:
            break
        threading.Event().wait(0.01)
    assert refined_sequences == [1, 2, 3, 4, 5]
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


def test_refinement_admission_closes_before_shutdown_sentinel() -> None:
    initial = MotionDecisionOutcome(
        event_id=42,
        snapshot_path="initial.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
        refinement_pending=True,
    )
    service, decision, tracking, _prewarm, image_reader = _service(initial)
    decision.refine.return_value = initial
    image_reader.return_value = np.ones((10, 10, 3), dtype=np.uint8)
    tracking.start.return_value = True
    stop = threading.Event()
    service.start(stop)

    service.request_stop()
    service.process(
        "motion",
        "person",
        datetime.now(timezone.utc),
        {"motion_episode_sequence": 12},
    )

    tracking.start.assert_called_once()
    assert service.status()["refinement_pending_episodes"] == 1
    stop.set()
    assert service.wait_stopped(1.0)


def test_refinement_thread_start_failure_restores_stopped_state() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
    )
    service, _decision, _tracking, _prewarm, _image_reader = _service(initial)

    with patch("survng.app.motion_incidents.threading.Thread.start", side_effect=RuntimeError("no thread")):
        with pytest.raises(RuntimeError, match="no thread"):
            service.start(threading.Event())

    assert not service.running()
    assert service.wait_stopped(0.01)


def test_memory_refinement_store_coalesces_route_capture_time_drift() -> None:
    store = _MemoryDetectionJobStore()
    route_id = "route:back-right:back-middle:57993"
    payload = {
        "topic": "adaptive/visual_backup",
        "event_at": "2026-08-27T23:41:16.884471+00:00",
        "qualification": {"detection_intent_id": route_id},
        "existing_event_id": None,
        "require_eligible_object": True,
        "require_motion_correlation": True,
    }

    assert store.enqueue_detection_job(
        job_id="job-1",
        camera_id="back-right",
        dedupe_key=f"intent:{route_id}",
        payload=payload,
    ) == "queued"
    assert store.enqueue_detection_job(
        job_id="job-1",
        camera_id="back-right",
        dedupe_key=f"intent:{route_id}",
        payload={
            **payload,
            "event_at": "2026-08-27T23:41:20.321719+00:00",
        },
    ) == "coalesced"


def test_replacement_runtime_same_episode_sequence_queues_distinct_refinement() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        first, *_ = _service(initial, refinement_store=store)
        replacement, *_ = _service(initial, refinement_store=store)
        event_at = datetime(2026, 8, 16, 14, 44, 17, tzinfo=timezone.utc)

        first.process(
            "adaptive/visual_backup",
            "credible motion",
            event_at,
            {
                "motion_episode_sequence": 3,
                "detection_intent_id": "gate:iruntime-a:g1:e3:request:1",
            },
        )
        replacement.process(
            "adaptive/visual_backup",
            "credible motion",
            event_at,
            {
                "motion_episode_sequence": 3,
                "detection_intent_id": "gate:iruntime-b:g1:e3:request:1",
            },
        )

        job_status = store.detection_job_status("gate")
        assert job_status["queued"] == 2
        assert job_status["oldest_age_ms"] >= 0.0
        with store._connect_jobs() as connection:
            dedupe_keys = {
                str(row[0])
                for row in connection.execute(
                    "select dedupe_key from detection_jobs where camera_id = 'gate'"
                )
            }
        assert dedupe_keys == {
            "intent:gate:iruntime-a:g1:e3:request:1",
            "intent:gate:iruntime-b:g1:e3:request:1",
        }


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


def test_live_refine_timing_and_oldest_refinement_age_are_reported() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
        processing_timing={"workflow_ms": 12.5, "phases_ms": {}},
    )
    refined = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        processing_timing={"workflow_ms": 48.0, "phases_ms": {"frame_decode_ms": 8.0}},
    )
    service, *_ = _service(initial)

    service.process("motion", "none", datetime.now(timezone.utc), {})
    service._record_timing(refined, kind="refine")
    timing = service.status()["object_detection_timing"]

    assert timing["live_workflow_ms_p95"] == 12.5
    assert timing["refine_workflow_ms_p95"] == 48.0
    assert timing["oldest_refinement_age_ms"] >= 0.0


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


def test_recovered_refinement_runs_persisted_completion_context() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=84,
        snapshot_path="refined.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        first, *_ = _service(initial, refinement_store=store)
        event_at = datetime.now(timezone.utc)
        first.process(
            "motion",
            "person",
            event_at,
            {"detection_intent_id": "gate:restart-safe-refinement"},
            refinement_completion_context={
                "decision_id": "decision-7",
                "event_at": event_at.isoformat(),
            },
        )

        replacement, decision, *_ = _service(initial, refinement_store=store)
        decision.refine.return_value = refined
        completion = Mock()
        replacement.set_refinement_completion_handler(completion)
        stop = threading.Event()
        replacement.start(stop)

        waiter = threading.Event()
        for _ in range(100):
            if replacement.status()["refinements_completed"]:
                break
            waiter.wait(0.01)

        completion.assert_called_once_with(
            refined,
            {
                "decision_id": "decision-7",
                "event_at": event_at.isoformat(),
            },
        )
        stop.set()
        replacement.request_stop()
        assert replacement.wait_stopped(1.0)


def test_durable_completion_failure_retries_until_handler_succeeds() -> None:
    initial = MotionDecisionOutcome(
        event_id=None,
        snapshot_path="",
        object_detected=False,
        refinement_pending=True,
    )
    refined = MotionDecisionOutcome(
        event_id=84,
        snapshot_path="refined.webp",
        object_detected=True,
        detected_objects=({"label": "person", "incident_eligible": True},),
    )
    service, decision, *_ = _service(initial)
    decision.refine.return_value = refined
    completion = Mock(side_effect=[RuntimeError("audit unavailable"), None])
    service.set_refinement_completion_handler(completion)
    stop = threading.Event()

    with patch(
        "survng.app.motion_incidents.REFINEMENT_COMPLETION_RETRY_SECONDS",
        0.0,
    ):
        service.start(stop)
        service.process(
            "motion",
            "person",
            datetime.now(timezone.utc),
            {"detection_intent_id": "gate:retry-completion"},
            refinement_completion_context={"completion_id": "decision-8:refinement"},
        )
        waiter = threading.Event()
        for _ in range(100):
            if service.status()["refinements_completed"]:
                break
            waiter.wait(0.01)

    assert completion.call_count == 2
    assert decision.refine.call_count == 2
    assert service.status()["refinements_completed"] == 1
    assert service.status()["refinement_callback_failures"] == 1
    refinement_jobs = service.status()["refinement_jobs"]
    assert refinement_jobs["completed"] == 1
    assert refinement_jobs["oldest_age_ms"] == 0.0
    stop.set()
    service.request_stop()
    assert service.wait_stopped(1.0)


def test_refinement_age_constant_matches_detection_job_store() -> None:
    assert REFINEMENT_MAX_QUEUE_AGE_SECONDS == DETECTION_JOB_MAXIMUM_AGE_SECONDS


def test_memory_claim_expires_stale_expired_lease_before_fresh_job() -> None:
    store = _MemoryDetectionJobStore()
    store.enqueue_detection_job(
        job_id="zombie",
        camera_id="gate",
        dedupe_key="episode:zombie",
        payload={"event_at": "2026-08-22T19:00:00+00:00"},
    )
    store.enqueue_detection_job(
        job_id="fresh",
        camera_id="gate",
        dedupe_key="episode:fresh",
        payload={"event_at": "2026-08-22T20:00:00+00:00"},
    )
    with store._lock:
        zombie = store._jobs["zombie"]
        zombie["state"] = "running"
        zombie["lease_owner"] = "dead-worker"
        zombie["lease_expires_at"] = 0.0
        zombie["created_at_monotonic"] = zombie["created_at_monotonic"] - 30.0

    claimed = store.claim_detection_job(
        "gate",
        lease_owner="refiner-a",
        maximum_age_seconds=1.0,
    )

    assert claimed is not None
    assert claimed["id"] == "fresh"
    assert store._jobs["zombie"]["state"] == "failed"
    assert store._jobs["zombie"]["last_error"] == "stale_refinement"
