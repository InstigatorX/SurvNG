from __future__ import annotations

import queue
import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.motion import MotionQualificationResult
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_analysis_service import (
    MotionAnalysisHooks,
    MotionAnalysisService,
)
from survng.app.motion_coordinator import (
    VisualBackupAction,
    VisualBackupCoordinator,
    VisualBackupDecision,
    VisualBackupPolicy,
)
from survng.app.motion_events import MotionEventCoordinator
from survng.app.motion_pipeline import MotionDebugSnapshotStore


def _hooks(
    *,
    run_pipeline: Mock | None = None,
    with_source_evidence: Mock | None = None,
    publish_event: Mock | None = None,
    set_last_motion_at: Mock | None = None,
    increment_stat: Mock | None = None,
    trigger_mode: str = "camera_rescue",
    execute_continuous: Mock | None = None,
) -> MotionAnalysisHooks:
    return MotionAnalysisHooks(
        frame_analysis_required=lambda: True,
        sample_fps=lambda: 5.0,
        frame_width=lambda: 320,
        motion_settings=lambda: (trigger_mode, "balanced", 320),
        continuous_primary_required=lambda: True,
        continuous_primary_due=lambda _captured_at, _last_at: True,
        execute_continuous=execute_continuous or Mock(),
        execute_debug_capture=Mock(),
        adaptive_rearm_seconds=lambda: 5.0,
        priority_dedup_seconds=lambda: 2.0,
        run_pipeline=run_pipeline or Mock(),
        illumination_filter_enabled=lambda: False,
        trigger_mode=lambda: trigger_mode,
        detection_enabled=lambda: True,
        with_source_evidence=with_source_evidence or Mock(),
        visual_backup_settings=lambda: {
            "grace_seconds": 2.0,
            "minimum_score": 0.7,
            "minimum_consecutive": 3,
            "cooldown_seconds": 20.0,
            "maximum_triggers_5m": 3,
        },
        visual_backup_policy=lambda: VisualBackupPolicy(
            warmup_seconds=10.0,
            grace_seconds=2.0,
            minimum_score=0.7,
            score_margin=0.15,
            minimum_consecutive=3,
            cooldown_seconds=20.0,
            maximum_triggers_5m=3,
            sample_fps=5.0,
            background_fps=2.0,
        ),
        suppression_verification_rate=lambda: 0.0,
        visual_backup_warmup_seconds=lambda: 10.0,
        sample_rejected_motion=lambda _event_at, _result: "",
        publish_event=publish_event or Mock(),
        set_last_motion_at=set_last_motion_at or Mock(),
        increment_stat=increment_stat or Mock(),
        record_analysis_wait=Mock(),
    )


def _service(hooks: MotionAnalysisHooks, queue_size: int = 1) -> MotionAnalysisService:
    observation = Mock()
    observation.handles_observation.return_value = False
    return MotionAnalysisService(
        camera_id="gate",
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=8,
        queue_size=queue_size,
        limiter=FairMotionAnalysisLimiter(1),
        observation_pipeline=observation,
        events=MotionEventCoordinator(queue_size=4, retry_limit=2),
        visual_backup=VisualBackupCoordinator(),
        audit_recorder=Mock(),
        debug_store=MotionDebugSnapshotStore(),
        hooks=hooks,
    )


def test_frame_sampling_keeps_compact_gray_and_color_buffers() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    frame = np.zeros((180, 640, 3), dtype=np.uint8)

    service.remember_frame(frame, 10.0, stop_event, 100.0)

    assert service.frames[-1][0] == 100.0
    assert service.frames[-1][1].shape == (90, 320)
    assert service.color_frames[-1][1].shape == (90, 320, 3)
    assert service.queue.get_nowait() == 100.0
    telemetry = service.telemetry_snapshot()
    assert telemetry["frames_sampled"] == 1
    assert telemetry["derived_frame_count"] == 2
    assert telemetry["derived_frame_bytes"] == 90 * 320 * 4
    assert telemetry["preprocess_count"] == 1


def test_latest_only_schedule_replaces_stale_pending_work() -> None:
    increment_stat = Mock()
    service = _service(_hooks(increment_stat=increment_stat))
    stop_event = threading.Event()

    service.schedule(100.0, stop_event)
    service.schedule(101.0, stop_event)

    assert service.queue.get_nowait() == 101.0
    increment_stat.assert_called_once_with("analysis_frames_dropped", 1)
    telemetry = service.telemetry_snapshot()
    assert telemetry["mailbox_high_water"] == 1
    assert telemetry["mailbox_replacements"] == 1


def test_schedule_does_not_report_drop_when_consumer_wins_full_queue_race() -> None:
    increment_stat = Mock()
    service = _service(_hooks(increment_stat=increment_stat))
    stop_event = threading.Event()
    service.queue.put_nowait(100.0)
    original_get = service.queue.get_nowait

    def drained_before_eviction() -> float:
        original_get()
        raise queue.Empty

    service.queue.get_nowait = Mock(side_effect=drained_before_eviction)

    service.schedule(101.0, stop_event)

    assert service.queue.queue[0] == 101.0
    increment_stat.assert_not_called()


def test_schedule_rejects_work_after_stop_requested() -> None:
    service = _service(_hooks())
    service.request_stop()

    service.schedule(100.0, threading.Event())

    assert list(service.queue.queue) == [None]


def test_thread_start_failure_disables_frame_admission_until_restarted() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()

    with patch("survng.app.motion_analysis_service.threading.Thread.start", side_effect=RuntimeError("start failed")):
        with pytest.raises(RuntimeError, match="start failed"):
            service.start(stop_event)

    service.schedule(100.0, stop_event)
    assert service.thread is None
    assert service.queue.empty()
    assert service._stop_requested.is_set()
    assert service._accepting_frames is False


def test_worker_loop_uses_injected_execution_boundary_and_stops_cleanly() -> None:
    stop_event = threading.Event()
    execute_continuous = Mock(side_effect=lambda _at: stop_event.set())
    service = _service(_hooks(execute_continuous=execute_continuous))
    with service.frame_lock:
        service.frames.append((100.0, np.zeros((90, 160), dtype=np.uint8)))
    service.queue.put_nowait(100.0)

    service.run(stop_event)

    execute_continuous.assert_called_once_with(100.0)
    assert service.last_processed_at == 100.0
    telemetry = service.telemetry_snapshot()
    assert telemetry["capture_to_analysis_count"] == 1
    assert telemetry["analysis_cycle_count"] == 1
    assert telemetry["copy_count"] == 1
    assert telemetry["copies_by_reason"]["analysis_latest"]["count"] == 1


def test_adaptive_analysis_promotes_accepted_fused_motion() -> None:
    accepted = MotionQualificationResult(True, 0.8, 0.48, "qualified", 3, {})
    run_pipeline = Mock(return_value=accepted)
    with_source_evidence = Mock(return_value=accepted)
    publish_event = Mock()
    set_last_motion_at = Mock()
    service = _service(
        _hooks(
            run_pipeline=run_pipeline,
            with_source_evidence=with_source_evidence,
            publish_event=publish_event,
            set_last_motion_at=set_last_motion_at,
            trigger_mode="adaptive",
        )
    )
    with service.frame_lock:
        service.color_frames.extend(
            [
                (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
                (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
            ]
        )

    service.analyze_continuous(100.0)

    trigger = service.events.next_trigger(timeout=0.01)
    assert trigger is not None
    assert trigger.topic == "adaptive/motion"
    assert trigger.prequalified is accepted
    set_last_motion_at.assert_called_once()
    publish_event.assert_called_once()
    assert service.events.adaptive_trigger_pending


def test_request_stop_replaces_pending_work_with_sentinel() -> None:
    service = _service(_hooks())
    service.queue.put_nowait(100.0)

    service.request_stop()

    assert service.queue.get_nowait() is None
    try:
        service.queue.get_nowait()
    except queue.Empty:
        pass
    else:
        raise AssertionError("stop queue should contain only the sentinel")


def test_stopped_service_rejects_late_capture_callbacks() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    stop_event.set()
    service.request_stop()

    service.remember_frame(
        np.zeros((180, 320, 3), dtype=np.uint8),
        10.0,
        stop_event,
        100.0,
    )

    assert not service.frames
    assert service.queue.get_nowait() is None
    assert service.queue.empty()


def test_adaptive_enqueue_failure_releases_reservation() -> None:
    accepted = MotionQualificationResult(True, 0.8, 0.48, "qualified", 3, {})
    service = _service(
        _hooks(
            run_pipeline=Mock(return_value=accepted),
            with_source_evidence=Mock(return_value=accepted),
            trigger_mode="adaptive",
        )
    )
    with service.frame_lock:
        service.color_frames.extend(
            [
                (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
                (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
            ]
        )
    service.events.enqueue = Mock(side_effect=RuntimeError("queue unavailable"))

    try:
        service.analyze_continuous(100.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("enqueue failure should remain visible to the worker")

    assert not service.events.adaptive_trigger_pending


def test_stop_requested_during_adaptive_fusion_prevents_publication() -> None:
    accepted = MotionQualificationResult(True, 0.8, 0.48, "qualified", 3, {})
    publish_event = Mock()
    fusion = Mock()
    service = _service(
        _hooks(
            run_pipeline=Mock(return_value=accepted),
            with_source_evidence=fusion,
            publish_event=publish_event,
            trigger_mode="adaptive",
        )
    )

    def stop_during_fusion(*_args: object, **_kwargs: object) -> MotionQualificationResult:
        service.request_stop()
        return accepted

    fusion.side_effect = stop_during_fusion
    with service.frame_lock:
        service.color_frames.extend(
            [
                (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
                (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
            ]
        )

    service.analyze_continuous(100.0)

    assert not service.events.adaptive_trigger_pending
    assert list(service.events.queue.queue) == []
    publish_event.assert_not_called()


def test_visual_fusion_failure_releases_reservation_and_candidate() -> None:
    accepted = MotionQualificationResult(True, 0.9, 0.48, "qualified", 3, {})
    service = _service(
        _hooks(
            with_source_evidence=Mock(side_effect=RuntimeError("fusion unavailable")),
        )
    )
    service.visual_backup.evaluate = Mock(
        return_value=VisualBackupDecision(
            VisualBackupAction.READY,
            accepted,
            required_score=0.7,
            consecutive=3,
            scene_ready=True,
        )
    )
    samples = [
        (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
        (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
    ]

    try:
        service.consider_visual_backup(accepted, samples, 100.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("fusion failure should remain visible to the worker")

    assert not service.events.adaptive_trigger_pending
    assert service.visual_backup.consecutive == 0
