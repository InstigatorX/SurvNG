from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.motion import MotionQualificationResult
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_analysis_service import (
    ANALYSIS_SLOT_WAKEUP,
    MotionAnalysisService,
    MotionFrameSubmission,
)
from survng.app.config import CameraTransitionRoute, MotionQualificationConfig
from survng.app.detection_watch import RouteDetectionWatch
from survng.app.ema_v2 import EmaPolicy
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
    preprocessor_implementation: str = "gray_blur",
    reset_temporal_runtime: Mock | None = None,
) -> SimpleNamespace:
    qualification = Mock()
    qualification.frame_analysis_required.return_value = True
    qualification.settings.return_value = (trigger_mode, "balanced", 320)
    qualification.preprocessor_implementation.return_value = preprocessor_implementation
    qualification.continuous_primary_required.return_value = True
    qualification.continuous_primary_due.return_value = True
    qualification.run_pipeline = run_pipeline or Mock()
    qualification.illumination_filter_enabled.return_value = False
    qualification.trigger_mode.return_value = trigger_mode
    qualification.with_source_evidence = with_source_evidence or Mock()
    qualification.visual_backup_settings.return_value = {
            "grace_seconds": 2.0,
            "minimum_score": 0.7,
            "minimum_consecutive": 3,
            "cooldown_seconds": 20.0,
            "maximum_triggers_5m": 3,
        }
    qualification.visual_backup_policy.return_value = EmaPolicy(
            warmup_seconds=10.0,
            grace_seconds=2.0,
            minimum_score=0.7,
            score_margin=0.15,
            minimum_consecutive=3,
            cooldown_seconds=20.0,
            maximum_triggers_5m=3,
            sample_fps=5.0,
            background_fps=2.0,
        )
    qualification.suppression_verification_rate.return_value = 0.0
    qualification.reset_runtime = reset_temporal_runtime or Mock()
    state = Mock()
    state.detection_enabled.return_value = True
    state.lifecycle_generation.return_value = 0
    state.publish_event = publish_event or Mock()
    state.set_last_motion_at = set_last_motion_at or Mock()
    state.increment_stat = increment_stat or Mock()
    media = Mock()
    media.sample_rejected_motion.return_value = ""
    media.sample_rejected_motion_frame.return_value = ""
    return SimpleNamespace(
        config=MotionQualificationConfig(
            sample_fps=5.0,
            frame_width=320,
            visual_backup_warmup_seconds=10.0,
        ),
        qualification=qualification,
        state=state,
        media=media,
        execute_continuous=execute_continuous,
    )


def _service(
    dependencies: SimpleNamespace,
    queue_size: int = 1,
    *,
    camera_id: str = "gate",
) -> MotionAnalysisService:
    service = MotionAnalysisService(
        camera_id=camera_id,
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=8,
        queue_size=queue_size,
        limiter=FairMotionAnalysisLimiter(1),
        events=MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id=camera_id,
        ),
        evidence=Mock(),
        audit_recorder=Mock(),
        debug_store=MotionDebugSnapshotStore(),
        config=dependencies.config,
        qualification=dependencies.qualification,
        media=dependencies.media,
        state=dependencies.state,
    )
    if dependencies.execute_continuous is not None:
        service.analyze_continuous = dependencies.execute_continuous
    return service


def test_frame_sampling_keeps_compact_gray_and_color_buffers() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    frame = np.zeros((180, 640, 3), dtype=np.uint8)

    service.remember_frame(frame, 10.0, stop_event, 100.0)

    assert service.frames[-1][0] == 100.0
    assert service.frames[-1][1].shape == (90, 320)
    assert service.color_frames[-1][1].shape == (90, 320, 3)
    assert service.processed_frames[-1][1].shape == (90, 320)
    assert not service.frames[-1][1].flags.writeable
    assert not service.color_frames[-1][1].flags.writeable
    assert not service.processed_frames[-1][1].flags.writeable
    assert service.queue.get_nowait() == 100.0
    telemetry = service.telemetry_snapshot()
    assert telemetry["frames_sampled"] == 1
    assert telemetry["derived_frame_count"] == 3
    assert telemetry["derived_frame_bytes"] == 90 * 320 * 5
    assert telemetry["preprocess_count"] == 1


def test_evidence_frame_selection_is_nearest_and_generation_bounded() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    first = np.full((18, 32, 3), 11, dtype=np.uint8)
    second = np.full((18, 32, 3), 22, dtype=np.uint8)

    service.submit_frame(
        first,
        10.0,
        stop_event,
        100.0,
        capture_sequence=41,
        capture_generation=7,
        lifecycle_generation=3,
    )
    submitted = service.queue.get_nowait()
    assert isinstance(submitted, MotionFrameSubmission)
    service._preprocess_frame(
        submitted.image,
        submitted.captured_at_epoch,
        captured_at_monotonic=submitted.captured_at_monotonic,
        capture_sequence=submitted.capture_sequence,
        capture_generation=submitted.capture_generation,
        lifecycle_generation=submitted.lifecycle_generation,
    )
    service.submit_frame(
        second,
        10.3,
        stop_event,
        100.3,
        capture_sequence=42,
        capture_generation=7,
        lifecycle_generation=3,
    )
    submitted = service.queue.get_nowait()
    assert isinstance(submitted, MotionFrameSubmission)
    service._preprocess_frame(
        submitted.image,
        submitted.captured_at_epoch,
        captured_at_monotonic=submitted.captured_at_monotonic,
        capture_sequence=submitted.capture_sequence,
        capture_generation=submitted.capture_generation,
        lifecycle_generation=submitted.lifecycle_generation,
    )

    selected = service.evidence_frame_near(
        100.01,
        sequence=41,
        capture_generation=7,
        lifecycle_generation=3,
    )
    assert selected is not None
    assert selected.sequence == 41
    assert int(selected.image[0, 0, 0]) == 11
    assert service.evidence_frame_near(
        100.0,
        sequence=41,
        capture_generation=8,
        lifecycle_generation=3,
    ) is None
    assert service.evidence_frame_near(
        100.0,
        sequence=41,
        capture_generation=7,
        lifecycle_generation=4,
    ) is None
    assert service.evidence_frame_near(
        105.0,
        sequence=41,
        capture_generation=7,
        lifecycle_generation=3,
    ) is None


def test_portrait_sampling_caps_the_long_edge_instead_of_expanding_pixels() -> None:
    service = _service(_hooks())

    service.remember_frame(
        np.zeros((640, 480, 3), dtype=np.uint8),
        10.0,
        threading.Event(),
        100.0,
    )

    assert service.color_frames[-1][1].shape == (320, 240, 3)
    assert service.telemetry_snapshot()["derived_frame_bytes"] == 320 * 240 * 5


def test_extreme_aspect_sampling_preserves_geometry() -> None:
    service = _service(_hooks())

    service.remember_frame(
        np.zeros((1000, 100, 3), dtype=np.uint8),
        10.0,
        threading.Event(),
        100.0,
    )

    assert service.color_frames[-1][1].shape == (320, 32, 3)


def test_cached_derivatives_are_bounded_to_the_reusable_three_frame_window() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()

    for index in range(6):
        service.remember_frame(
            np.full((180, 640, 3), index, dtype=np.uint8),
            10.0 + index,
            stop_event,
            100.0 + index,
        )

    assert len(service.frames) == 6
    assert len(service.color_frames) == 3
    assert len(service.processed_frames) == 3
    assert [timestamp for timestamp, _frame in service.processed_frames] == [
        103.0,
        104.0,
        105.0,
    ]


def test_unrecognized_preprocessor_does_not_create_gray_blur_cache() -> None:
    service = _service(_hooks(preprocessor_implementation="future_gpu"))
    stop_event = threading.Event()

    service.remember_frame(
        np.zeros((180, 640, 3), dtype=np.uint8),
        10.0,
        stop_event,
        100.0,
    )

    assert len(service.frames) == 1
    assert not service.processed_frames
    telemetry = service.telemetry_snapshot()
    assert telemetry["derived_frame_count"] == 2
    assert telemetry["derived_frame_bytes"] == 90 * 320 * 4


def test_raw_submission_defers_preprocessing_to_analysis_worker() -> None:
    stop_event = threading.Event()
    execute_continuous = Mock(side_effect=lambda _at: stop_event.set())
    service = _service(_hooks(execute_continuous=execute_continuous))
    captured_at = time.time()

    service.submit_frame(
        np.zeros((180, 640, 3), dtype=np.uint8),
        10.0,
        stop_event,
        captured_at,
    )

    assert not service.frames
    assert service.telemetry_snapshot()["preprocess_count"] == 0
    queued = service.queue.queue[0]
    assert isinstance(queued, MotionFrameSubmission)
    assert queued.captured_at_epoch == captured_at

    service.run(stop_event)

    assert service.frames[-1][0] == captured_at
    assert service.frames[-1][1].shape == (90, 320)
    assert service.color_frames[-1][1].shape == (90, 320, 3)
    telemetry = service.telemetry_snapshot()
    assert telemetry["raw_frames_submitted"] == 1
    assert telemetry["preprocess_count"] == 1
    execute_continuous.assert_called_once_with(captured_at)


def test_raw_submission_mailbox_replaces_stale_frame_before_preprocessing() -> None:
    increment_stat = Mock()
    service = _service(_hooks(increment_stat=increment_stat))
    stop_event = threading.Event()
    stale = np.zeros((180, 320, 3), dtype=np.uint8)
    latest = np.ones((180, 320, 3), dtype=np.uint8)

    service.submit_frame(stale, 10.0, stop_event, 100.0)
    service.submit_frame(latest, 11.0, stop_event, 101.0)

    queued = service.queue.get_nowait()
    assert isinstance(queued, MotionFrameSubmission)
    assert queued.captured_at_epoch == 101.0
    assert queued.image is not latest
    assert not queued.image.flags.writeable
    latest.fill(7)
    assert int(queued.image[0, 0, 0]) == 1
    assert not service.frames
    increment_stat.assert_called_once_with("analysis_frames_dropped", 1)
    telemetry = service.telemetry_snapshot()
    assert telemetry["mailbox_replacements"] == 1
    assert telemetry["copies_by_reason"]["submission_safety"]["count"] == 2


def test_read_only_view_is_copied_before_async_submission() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    owner = np.ones((90, 160, 3), dtype=np.uint8)
    view = owner[:, :, :]
    view.setflags(write=False)

    service.submit_frame(view, 10.0, stop_event, 100.0)
    queued = service.queue.get_nowait()
    owner.fill(9)

    assert isinstance(queued, MotionFrameSubmission)
    assert queued.image.flags.owndata
    assert not queued.image.flags.writeable
    assert int(queued.image[0, 0, 0]) == 1
    assert service.telemetry_snapshot()["copy_count"] == 1


def test_worker_preserves_temporal_history_while_analysis_slot_is_busy() -> None:
    increment_stat = Mock()
    stop_event = threading.Event()
    execute_continuous = Mock(side_effect=lambda _at: stop_event.set())
    limiter = FairMotionAnalysisLimiter(1)
    service = _service(_hooks(
        increment_stat=increment_stat,
        execute_continuous=execute_continuous,
    ))
    service.limiter = limiter
    first = np.zeros((180, 320, 3), dtype=np.uint8)
    latest = np.ones((180, 320, 3), dtype=np.uint8)

    with limiter.acquire("blocker"):
        first_at = time.time()
        service.submit_frame(first, 10.0, stop_event, first_at)
        worker = threading.Thread(target=service.run, args=(stop_event,))
        worker.start()
        deadline = time.monotonic() + 1.0
        while limiter.status()["pending"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert limiter.status()["pending"] == 1
        latest_at = time.time()
        service.submit_frame(latest, 11.0, stop_event, latest_at)

    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert [captured_at for captured_at, _frame in service.frames] == [
        first_at,
        latest_at,
    ]
    assert len(service.frames) == 2
    assert int(service.color_frames[-1][1][0, 0, 0]) == 1
    increment_stat.assert_not_called()
    telemetry = service.telemetry_snapshot()
    assert telemetry["mailbox_replacements"] == 0
    assert telemetry["analysis_slot_deferrals"] >= 1


def test_released_analysis_slot_wakes_worker_without_polling_delay() -> None:
    limiter = FairMotionAnalysisLimiter(1)
    stop_event = threading.Event()
    executed = threading.Event()
    service = _service(_hooks(
        execute_continuous=Mock(side_effect=lambda _at: executed.set()),
    ))
    service.limiter = limiter
    worker = threading.Thread(target=service.run, args=(stop_event,))

    with limiter.acquire("blocker"):
        worker.start()
        service.submit_frame(
            np.zeros((90, 160, 3), dtype=np.uint8),
            10.0,
            stop_event,
            100.0,
        )
        deadline = time.monotonic() + 1.0
        while limiter.status()["pending"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert limiter.status()["pending"] == 1
        assert not executed.is_set()

    # The queue fallback is 500 ms; completion well before that proves the
    # limiter release callback woke the frame lane directly.
    assert executed.wait(timeout=0.25)
    service.request_stop()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    telemetry = service.telemetry_snapshot()
    assert telemetry["analysis_slot_deferrals"] == 1
    assert telemetry["qualification_count"] == 1


def test_fleet_contention_preserves_each_cameras_sampled_history() -> None:
    limiter = FairMotionAnalysisLimiter(2)
    blocker_release = threading.Event()
    blockers_ready = threading.Barrier(3)

    def hold_slot(camera_id: str) -> None:
        with limiter.acquire(camera_id):
            blockers_ready.wait()
            assert blocker_release.wait(2.0)

    blocker_threads = [
        threading.Thread(target=hold_slot, args=(f"blocker-{index}",))
        for index in range(2)
    ]
    for thread in blocker_threads:
        thread.start()
    blockers_ready.wait()

    stop_events = [threading.Event() for _index in range(12)]
    services = [_service(_hooks()) for _index in range(12)]
    workers = []
    for index, service in enumerate(services):
        service.camera_id = f"camera-{index}"
        service.limiter = limiter
        worker = threading.Thread(target=service.run, args=(stop_events[index],))
        worker.start()
        workers.append(worker)

    for sample in range(4):
        for index, service in enumerate(services):
            service.submit_frame(
                np.full((90, 160, 3), sample, dtype=np.uint8),
                10.0 + sample,
                stop_events[index],
                100.0 + sample,
            )
        deadline = time.monotonic() + 2.0
        while (
            any(len(service.frames) < sample + 1 for service in services)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert all(len(service.frames) == sample + 1 for service in services)

    assert all(
        service.telemetry_snapshot()["analysis_slot_deferrals"] >= 1
        for service in services
    )
    blocker_release.set()
    for thread in blocker_threads:
        thread.join(timeout=1.0)
    for service in services:
        service.request_stop()
    for worker in workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()


def test_backward_wall_clock_step_resets_runtime_without_suspending_frames() -> None:
    stop_event = threading.Event()
    reset_runtime = Mock()
    analyzed: list[float] = []

    def execute(captured_at: float) -> None:
        analyzed.append(captured_at)
        if len(analyzed) == 2:
            stop_event.set()

    service = _service(_hooks(
        execute_continuous=Mock(side_effect=execute),
        reset_temporal_runtime=reset_runtime,
    ))
    worker = threading.Thread(target=service.run, args=(stop_event,))
    worker.start()
    service.submit_frame(
        np.zeros((90, 160, 3), dtype=np.uint8),
        10.0,
        stop_event,
        100.0,
    )
    deadline = time.monotonic() + 1.0
    while len(analyzed) < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    stale_result = MotionQualificationResult(True, 0.8, 0.5, "qualified", 4, {})
    service.qualification_results.append((100.0, stale_result))
    service.last_continuous_result = stale_result
    service.submit_frame(
        np.ones((90, 160, 3), dtype=np.uint8),
        11.0,
        stop_event,
        90.0,
    )
    worker.join(timeout=1.0)

    assert analyzed == [100.0, 90.0]
    assert service.last_processed_sequence == 2
    assert service.last_processed_at == 90.0
    assert [timestamp for timestamp, _frame in service.frames] == [90.0]
    assert service.qualification_results_since(0.0) == []
    assert service.last_continuous_result is None
    reset_runtime.assert_called_once()
    assert callable(reset_runtime.call_args.kwargs["clear_observation_evidence"])
    assert service.telemetry_snapshot()["clock_discontinuity_resets"] == 1


def test_runtime_reset_clears_both_ema_conditioners_and_route_replay_history() -> None:
    service = _service(_hooks())
    service.ema_v2.scene_ready = True
    service.ema_verification.scene_ready = True
    service.recent_accepted_results.append(
        (100.0, MotionQualificationResult(True, 0.6, 0.48, "qualified", 3, {}))
    )

    service.reset()

    assert service.ema_v2.scene_ready is False
    assert service.ema_verification.scene_ready is False
    assert list(service.recent_accepted_results) == []


def test_failed_deferred_analysis_is_not_retried_in_idle_loop() -> None:
    failure = RuntimeError("pipeline failed")
    execute = Mock(side_effect=failure)
    service = _service(_hooks(execute_continuous=execute))
    service._pending_analysis_at = 100.0

    with pytest.raises(RuntimeError, match="pipeline failed"):
        service._try_execute_pending_analysis()

    assert service._pending_analysis_at == 0.0
    execute.assert_called_once_with(100.0)
    assert service.telemetry_snapshot()["qualification_count"] == 1


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


def test_raw_frame_replacing_slot_wakeup_is_not_counted_as_frame_drop() -> None:
    increment_stat = Mock()
    service = _service(_hooks(increment_stat=increment_stat))
    stop_event = threading.Event()
    service.queue.put_nowait(ANALYSIS_SLOT_WAKEUP)

    service.schedule(101.0, stop_event)

    assert service.queue.get_nowait() == 101.0
    increment_stat.assert_not_called()
    assert service.telemetry_snapshot()["mailbox_replacements"] == 0


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
    assert telemetry["copy_count"] == 0
    assert telemetry["shared_read_count"] == 1
    assert telemetry["shared_reads_by_reason"]["analysis_latest"]["count"] == 1


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
        first_processed = np.zeros((90, 160), dtype=np.uint8)
        second_processed = np.zeros((90, 160), dtype=np.uint8)
        service.color_frames.extend(
            [
                (99.5, np.full((90, 160, 3), 10, dtype=np.uint8)),
                (100.0, np.full((90, 160, 3), 60, dtype=np.uint8)),
            ]
        )
        service.processed_frames.extend(
            [(99.5, first_processed), (100.0, second_processed)]
        )

    service.analyze_continuous(100.0)

    trigger = service.events.next_trigger(timeout=0.01)
    assert trigger is not None
    assert trigger.topic == "adaptive/motion"
    assert trigger.prequalified.features["ema_v2"] is True
    assert trigger.prequalified.score == accepted.score
    assert trigger.detection_intent_id
    set_last_motion_at.assert_called_once()
    publish_event.assert_called_once()
    assert service.events.episode_controller.snapshot()["request_status"] == "admitted"
    cached = run_pipeline.call_args.kwargs["processed_frames"]
    assert cached[0] is first_processed
    assert cached[1] is second_processed
    assert (
        run_pipeline.call_args.kwargs["processed_frame_implementation"]
        == "gray_blur"
    )
    telemetry = service.telemetry_snapshot()
    assert telemetry["cached_derivative_reuse_count"] == 2
    assert telemetry["cached_derivative_reuse_bytes"] == 90 * 160 * 2


def test_adaptive_analysis_anchors_new_track_during_active_episode() -> None:
    activation = MotionQualificationResult(
        True,
        0.8,
        0.48,
        "qualified",
        3,
        {
            "event_state_phase": "active",
            "event_state_key": "gate:100000",
            "event_state_transition": "activation_threshold",
            "motion_region_track_id": 1,
            "motion_regions": [[0.05, 0.05, 0.25, 0.2]],
        },
    )
    active_new_track = MotionQualificationResult(
        True,
        0.82,
        0.48,
        "qualified",
        3,
        {
            "event_state_phase": "active",
            "event_state_key": "gate:100000",
            "event_state_transition": "active_confirmed",
            "motion_region_track_id": 7,
            "motion_regions": [[0.55, 0.55, 0.85, 0.9]],
        },
    )
    stats = Mock()
    service = _service(_hooks(
        run_pipeline=Mock(side_effect=[activation, active_new_track]),
        increment_stat=stats,
        trigger_mode="adaptive",
    ))
    service.events.episode_controller.minimum_followup_interval_seconds = 0.0
    with service.frame_lock:
        service.color_frames.extend([
            (99.5, np.full((90, 160, 3), 15, dtype=np.uint8)),
            (100.0, np.full((90, 160, 3), 80, dtype=np.uint8)),
        ])

    service.analyze_continuous(100.0)
    first = service.events.next_trigger(timeout=0.01)
    now_monotonic = time.monotonic()
    service.events.episode_controller.mark_running(
        first.detection_intent_id, occurred_monotonic=now_monotonic
    )
    service.events.episode_controller.complete(
        first.detection_intent_id, occurred_monotonic=now_monotonic
    )
    service.analyze_continuous(103.0)

    second = service.events.next_trigger(timeout=0.01)
    assert first is not None and first.topic == "adaptive/motion"
    assert second is not None and second.topic == "adaptive/active_followup"
    assert second.event_at.timestamp() == 103.0
    assert second.prequalified is not None
    assert second.prequalified.accepted
    assert second.prequalified.reason == "ema_v2_qualified"
    assert second.prequalified.features["active_event_followup"] is True
    stats.assert_any_call("active_followup_triggers", 1)


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


def test_reset_cancels_pending_fair_limiter_request() -> None:
    limiter = FairMotionAnalysisLimiter(1)
    service = _service(_hooks())
    service.limiter = limiter
    service._pending_analysis_at = 100.0

    with limiter.acquire("holder"):
        assert not service._try_execute_pending_analysis()
        assert limiter.status()["pending"] == 1
        service.reset()
        assert limiter.status()["pending"] == 0

    with limiter.try_acquire("foyer") as waited:
        assert waited is not None


def test_unexpected_worker_exit_cancels_pending_fair_limiter_request() -> None:
    limiter = FairMotionAnalysisLimiter(1)
    service = _service(_hooks())
    service.limiter = limiter

    def fail_with_pending(_stop_event: threading.Event) -> None:
        service._pending_analysis_at = 100.0
        with limiter.try_acquire("gate") as waited:
            assert waited is None
        raise RuntimeError("unexpected worker failure")

    service._run = fail_with_pending
    with limiter.acquire("holder"):
        with pytest.raises(RuntimeError, match="unexpected worker failure"):
            service.run(threading.Event())
        assert limiter.status()["pending"] == 0

    with limiter.try_acquire("foyer") as waited:
        assert waited is not None


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


def test_stop_wins_over_frame_admitted_before_mailbox_enqueue() -> None:
    service = _service(_hooks())
    stop_event = threading.Event()
    admitted = threading.Event()
    release_submitter = threading.Event()
    original_admit = service._admit_frame

    def pause_after_admission(
        frame_clock: float,
        submitted_stop: threading.Event,
    ) -> bool:
        result = original_admit(frame_clock, submitted_stop)
        admitted.set()
        assert release_submitter.wait(1.0)
        return result

    service._admit_frame = pause_after_admission
    submitter = threading.Thread(
        target=service.submit_frame,
        args=(np.zeros((180, 320, 3), dtype=np.uint8), 10.0, stop_event, 100.0),
    )
    submitter.start()
    assert admitted.wait(1.0)

    service.request_stop()
    release_submitter.set()
    submitter.join(timeout=1.0)

    assert not submitter.is_alive()
    assert list(service.queue.queue) == [None]
    assert service.telemetry_snapshot()["raw_frames_submitted"] == 0


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
                (99.5, np.full((90, 160, 3), 18, dtype=np.uint8)),
                (100.0, np.full((90, 160, 3), 90, dtype=np.uint8)),
            ]
        )
    service.events.enqueue = Mock(side_effect=RuntimeError("queue unavailable"))

    try:
        service.analyze_continuous(100.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("enqueue failure should remain visible to the worker")

    assert service.events.episode_controller.snapshot()["request_status"] == "aborted"


def test_temporal_filter_skips_stable_scene() -> None:
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
                (99.5, np.full((90, 160, 3), 45, dtype=np.uint8)),
                (100.0, np.full((90, 160, 3), 45, dtype=np.uint8)),
            ]
        )

    service.analyze_continuous(100.0)

    assert list(service.events.queue.queue) == []
    assert service.telemetry_snapshot()["temporal_filter_skips"] == 1
    assert service.primary_last_processed_at == 100.0


def test_temporal_filter_leaves_motion_debug_capture_due() -> None:
    service = _service(_hooks(trigger_mode="adaptive"))
    service.debug_store.set_enabled(True)
    frame = np.full((90, 160, 3), 45, dtype=np.uint8)
    gray = np.full((90, 160), 45, dtype=np.uint8)
    with service.frame_lock:
        service.color_frames.extend([(99.5, frame), (100.0, frame.copy())])
        service.frames.extend([(99.5, gray), (100.0, gray.copy())])

    service.analyze_continuous(100.0)

    assert service.telemetry_snapshot()["temporal_filter_skips"] == 1
    assert service.debug_store.status()["snapshot"] is None
    assert service.debug_store.capture_due()
    service.qualification.run_pipeline.assert_not_called()


def test_worker_captures_motion_debug_while_analysis_slot_is_busy() -> None:
    accepted = MotionQualificationResult(False, 0.1, 0.48, "quiet", 2, {})
    run_pipeline = Mock(return_value=accepted)
    limiter = FairMotionAnalysisLimiter(1)
    stop_event = threading.Event()
    service = _service(_hooks(run_pipeline=run_pipeline))
    service.limiter = limiter
    gray = np.zeros((90, 160), dtype=np.uint8)
    gray.setflags(write=False)
    with service.frame_lock:
        service.frames.extend([(99.0, gray), (100.0, gray)])
    service.debug_store.set_enabled(True)

    with limiter.acquire("blocker"):
        worker = threading.Thread(target=service.run, args=(stop_event,))
        worker.start()
        service.submit_frame(
            np.zeros((90, 160, 3), dtype=np.uint8),
            12.0,
            stop_event,
            102.0,
        )
        deadline = time.monotonic() + 2.0
        while not run_pipeline.called and time.monotonic() < deadline:
            time.sleep(0.01)
        stop_event.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    run_pipeline.assert_called()
    assert run_pipeline.call_args.kwargs.get("capture_debug") is True
    assert run_pipeline.call_args.kwargs.get("isolated") is True


def test_temporal_filter_warms_visual_backup_during_quiet_scene() -> None:
    run_pipeline = Mock()
    service = _service(_hooks(run_pipeline=run_pipeline, trigger_mode="camera_rescue"))
    frame = np.full((90, 160, 3), 45, dtype=np.uint8)

    for captured_at in (100.0, 103.0, 106.0, 109.0, 112.0):
        with service.frame_lock:
            service.color_frames.extend(
                [(captured_at - 0.5, frame.copy()), (captured_at, frame.copy())]
            )
        service.analyze_continuous(captured_at)

    run_pipeline.assert_not_called()
    assert service.ema_v2.scene_ready is True
    assert service.ema_verification.scene_ready is True
    assert service.primary_last_processed_at == 112.0


def test_stop_requested_before_adaptive_admission_prevents_publication() -> None:
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

    service.request_stop()
    with service.frame_lock:
        service.color_frames.extend(
            [
                (99.5, np.full((90, 160, 3), 12, dtype=np.uint8)),
                (100.0, np.full((90, 160, 3), 70, dtype=np.uint8)),
            ]
        )

    service.analyze_continuous(100.0)

    assert list(service.events.queue.queue) == []
    publish_event.assert_not_called()


def test_ema_v2_qualified_edge_bypasses_legacy_fusion_gate() -> None:
    accepted = MotionQualificationResult(True, 0.9, 0.48, "qualified", 3, {})
    service = _service(
        _hooks(
            with_source_evidence=Mock(side_effect=RuntimeError("fusion unavailable")),
        )
    )
    service.ema_v2.scene_ready = True
    service.ema_v2.analysis_started_at = 90.0
    service.ema_v2.observation_count = 3
    samples = [
        (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
        (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
    ]

    for captured_at in (100.0, 101.0, 102.0):
        service.consider_visual_backup(accepted, samples, captured_at)

    service.qualification.with_source_evidence.assert_not_called()
    trigger = service.events.queue.get_nowait()
    assert trigger.prequalified.reason == "ema_v2_qualified"
    assert trigger.detection_intent_id


def test_persistent_accepted_ema_below_rescue_score_still_requests_analysis() -> None:
    service = _service(_hooks())
    for conditioner in (service.ema_v2, service.ema_verification):
        conditioner.scene_ready = True
        conditioner.analysis_started_at = 90.0
        conditioner.observation_count = 3
    accepted = MotionQualificationResult(True, 0.61, 0.48, "qualified", 3, {})
    samples = [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))]

    for captured_at in (100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0):
        service.consider_visual_backup(accepted, samples, captured_at)

    trigger = service.events.queue.get_nowait()
    assert trigger.prequalified.features["security_verification"] is True
    assert (
        trigger.prequalified.features["security_verification_reason"]
        == "persistent_ema"
    )
    assert trigger.prequalified.features["visual_backup_required_score"] == 0.48


def test_route_watch_accelerates_below_score_ema_verification() -> None:
    service = _service(_hooks())
    onvif_observer = Mock()
    service.set_onvif_effectiveness_observer(onvif_observer)
    watch = SimpleNamespace(as_dict=lambda: {
        "source_camera_id": "gate",
        "target_camera_id": "back-left",
        "source_event_id": 44,
    })
    consume = Mock(return_value=True)
    service.set_security_verification_context(
        route_watch=lambda _camera_id, _captured_at: watch,
        consume_route_watch=consume,
    )
    accepted = MotionQualificationResult(True, 0.61, 0.48, "qualified", 3, {})
    samples = [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))]

    service.consider_visual_backup(accepted, samples, 100.0)

    trigger = service.events.queue.get_nowait()
    features = trigger.prequalified.features
    assert features["security_verification_reason"] == "route_watch"
    assert features["route_detection_watch"]["source_event_id"] == 44
    assert features["security_verification_bypass_limits"] is True
    # Trigger admission is not proof of an eligible incident. The watch stays
    # available until the decision handler durably admits a target event.
    consume.assert_not_called()
    onvif_observer.assert_not_called()


def test_route_watch_replays_recent_accepted_ema_seen_before_confirmation() -> None:
    service = _service(_hooks())
    accepted = MotionQualificationResult(True, 0.56, 0.48, "qualified", 3, {})
    service.recent_accepted_results.append((105.0, accepted))
    watch = SimpleNamespace(
        target_camera_id="gate",
        eligible_at=100.0,
        expires_at=110.0,
        as_dict=lambda: {
            "source_camera_id": "lower-garage",
            "target_camera_id": "gate",
            "source_event_id": 45,
        },
    )
    service.set_security_verification_context(
        route_watch=lambda _camera_id, captured_at: (
            watch if 100.0 <= captured_at <= 110.0 else None
        ),
    )

    assert service.consider_route_watch(watch) is True

    trigger = service.events.queue.get_nowait()
    assert trigger.event_at.timestamp() == 105.0
    assert (
        trigger.prequalified.features["security_verification_reason"]
        == "route_watch"
    )


def test_route_watch_replays_durable_ema_after_service_restart() -> None:
    service = _service(_hooks(), camera_id="back-right")
    watch = SimpleNamespace(
        target_camera_id="back-right",
        eligible_at=100.0,
        expires_at=110.0,
        as_dict=lambda: {
            "source_camera_id": "back-middle",
            "target_camera_id": "back-right",
            "source_event_id": 44723,
        },
    )
    service.set_security_verification_context(
        route_watch=lambda _camera_id, captured_at: (
            watch if 100.0 <= captured_at <= 110.0 else None
        ),
        load_ema_candidates=lambda _camera_id, _start, _end: [(
            105.0,
            {
                "accepted": True,
                "score": 0.5545,
                "threshold": 0.48,
                "reason": "qualified",
                "frame_count": 3,
                "features": {"motion_regions": [[0.1, 0.2, 0.3, 0.4]]},
                "telemetry": {},
            },
        )],
    )

    assert service.consider_route_watch(watch) is True

    trigger = service.events.queue.get_nowait()
    assert trigger.event_at.timestamp() == 105.0
    assert trigger.detection_intent_id == "route:back-right:back-middle:44723"
    assert (
        trigger.prequalified.features["security_verification_reason"]
        == "route_watch"
    )


def test_confirmed_route_chain_replays_reported_multi_camera_vehicle_trace() -> None:
    watches = RouteDetectionWatch([
        CameraTransitionRoute(
            from_camera="lower-garage",
            to_camera="gate",
            max_seconds=30,
        ),
        CameraTransitionRoute(
            from_camera="lower-garage",
            to_camera="upper-garage",
            max_seconds=30,
        ),
        CameraTransitionRoute(
            from_camera="gate",
            to_camera="back-left",
            max_seconds=30,
        ),
        CameraTransitionRoute(
            from_camera="back-left",
            to_camera="back-middle",
            max_seconds=30,
        ),
        CameraTransitionRoute(
            from_camera="back-middle",
            to_camera="back-right",
            max_seconds=30,
        ),
    ])
    observations = {
        "gate": (1001.0, 0.7250),
        "upper-garage": (1004.0, 0.7651),
        "back-left": (1027.0, 0.7101),
        "back-middle": (1029.0, 0.6107),
        "back-right": (1048.0, 0.5545),
    }
    services: dict[str, MotionAnalysisService] = {}
    route_paths: dict[str, tuple[str, ...]] = {}
    route_origins: dict[str, tuple[str, int]] = {}
    for camera_id, (captured_at, score) in observations.items():
        service = _service(_hooks(), camera_id=camera_id)
        service.recent_accepted_results.append((
            captured_at,
            MotionQualificationResult(
                True,
                score,
                0.48,
                "qualified",
                3,
                {},
            ),
        ))
        service.set_security_verification_context(
            route_watch=watches.match,
            consume_route_watch=watches.consume,
        )
        services[camera_id] = service

    def confirm(camera_id: str, event_id: int, event_at: float) -> None:
        origin_camera_id, origin_event_id = route_origins.get(camera_id, ("", 0))
        for watch in watches.observe_incident(
            camera_id=camera_id,
            event_id=event_id,
            event_at=event_at,
            objects=[{"label": "car", "incident_eligible": True}],
            route_path=route_paths.get(camera_id, ()),
            origin_camera_id=origin_camera_id,
            origin_event_id=origin_event_id,
        ):
            route_paths[watch.target_camera_id] = watch.route_path
            route_origins[watch.target_camera_id] = (
                watch.origin_camera_id,
                watch.origin_event_id,
            )
            assert services[watch.target_camera_id].consider_route_watch(watch)

    confirm("lower-garage", 44720, 1000.0)
    confirm("gate", 44721, 1001.0)
    confirm("back-left", 44722, 1027.0)
    confirm("back-middle", 44723, 1029.0)

    for camera_id, service in services.items():
        trigger = service.events.queue.get_nowait()
        assert trigger.prequalified.features["security_verification_reason"] == "route_watch"
        assert trigger.prequalified.features["security_verification_bypass_limits"] is True
        assert trigger.event_at.timestamp() == observations[camera_id][0]
        watch_details = trigger.prequalified.features["route_detection_watch"]
        assert watch_details["origin_camera_id"] == "lower-garage"
        assert watch_details["origin_event_id"] == 44720


def test_route_verification_wins_when_ordinary_conditioner_also_qualifies() -> None:
    service = _service(_hooks())
    accepted = MotionQualificationResult(True, 0.9, 0.48, "qualified", 3, {})
    service.ema_v2.scene_ready = True
    service.ema_v2.analysis_started_at = 90.0
    service.ema_v2.observation_count = 3
    policy = service.qualification.visual_backup_policy()
    service.ema_v2.evaluate(accepted, 98.0, 998.0, policy, detection_enabled=True)
    service.ema_v2.evaluate(accepted, 99.0, 999.0, policy, detection_enabled=True)
    watch = SimpleNamespace(as_dict=lambda: {
        "source_camera_id": "lower-garage",
        "target_camera_id": "gate",
        "source_event_id": 46,
    })
    service.set_security_verification_context(
        route_watch=lambda _camera_id, _captured_at: watch,
    )

    service.consider_visual_backup(
        accepted,
        [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))],
        100.0,
    )

    trigger = service.events.queue.get_nowait()
    assert trigger.prequalified.features["security_verification_reason"] == "route_watch"


@pytest.mark.parametrize("terminal", [False, True])
def test_route_watch_gets_distinct_durable_intent_during_existing_episode(
    terminal: bool,
) -> None:
    service = _service(_hooks())
    service.ema_v2.scene_ready = True
    service.ema_v2.analysis_started_at = 80.0
    service.ema_v2.observation_count = 3
    strong = MotionQualificationResult(True, 0.9, 0.48, "qualified", 3, {})
    samples = [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))]
    for captured_at in (100.0, 101.0, 102.0):
        service.consider_visual_backup(strong, samples, captured_at)
    original = service.events.queue.get_nowait()
    if terminal:
        service.events.episode_controller.complete(
            original.detection_intent_id,
            occurred_monotonic=time.monotonic(),
        )

    watch = SimpleNamespace(as_dict=lambda: {
        "source_camera_id": "lower-garage",
        "target_camera_id": "gate",
        "source_event_id": 44720,
    })
    consume = Mock(return_value=True)
    service.set_security_verification_context(
        route_watch=lambda _camera_id, _captured_at: watch,
        consume_route_watch=consume,
    )
    route_result = MotionQualificationResult(
        True,
        0.55,
        0.48,
        "qualified",
        3,
        {},
    )

    service.consider_visual_backup(route_result, samples, 120.0)

    routed = service.events.queue.get_nowait()
    assert routed.detection_intent_id == "route:gate:lower-garage:44720"
    assert routed.detection_intent_id != original.detection_intent_id
    assert routed.event_at.timestamp() == 120.0
    consume.assert_not_called()


def test_degraded_onvif_accelerates_below_score_ema_verification() -> None:
    service = _service(_hooks())
    for conditioner in (service.ema_v2, service.ema_verification):
        conditioner.scene_ready = True
        conditioner.analysis_started_at = 90.0
        conditioner.observation_count = 3
    service.set_security_verification_context(
        onvif_effectiveness=lambda: {
            "signal_degraded": True,
            "signal_effectiveness_status": "degraded",
        },
    )
    accepted = MotionQualificationResult(True, 0.55, 0.48, "qualified", 3, {})
    samples = [(100.0, np.zeros((90, 160, 3), dtype=np.uint8))]

    for captured_at in (100.0, 101.0, 102.0):
        service.consider_visual_backup(accepted, samples, captured_at)

    trigger = service.events.queue.get_nowait()
    assert (
        trigger.prequalified.features["security_verification_reason"]
        == "onvif_degraded"
    )
    assert (
        trigger.prequalified.features["onvif_signal_effectiveness_status"]
        == "degraded"
    )


def test_visual_backup_audits_one_summary_for_sustained_below_gate_episode() -> None:
    dependencies = _hooks()
    service = _service(dependencies)
    below_gate = [
        MotionQualificationResult(True, score, 0.48, "qualified", 3, {})
        for score in (0.61, 0.69, 0.65)
    ]
    quiet = MotionQualificationResult(False, 0.1, 0.48, "low_score", 3, {})
    service.ema_v2.scene_ready = True
    service.ema_v2.analysis_started_at = 90.0
    service.ema_v2.observation_count = 3
    frames = [
        np.full((90, 160, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ]

    for index, result in enumerate([*below_gate, quiet]):
        service.consider_visual_backup(
            result,
            [(100.0 + index * 0.5, frames[index])],
            100.0 + index * 0.5,
        )

    service.audit_recorder.record_audit.assert_called_once()
    audit = service.audit_recorder.record_audit.call_args.kwargs
    assert audit["category"] == "visual_backup"
    assert audit["reason"] == "visual_backup_below_threshold"
    assert audit["score"] == 0.69
    assert audit["features"]["visual_backup_required_score"] == 0.7
    assert audit["features"]["visual_backup_credible_frames"] == 3
    assert audit["features"]["visual_backup_episode_duration_seconds"] == 1.0
    dependencies.media.sample_rejected_motion_frame.assert_called_once()
    stored_frame = dependencies.media.sample_rejected_motion_frame.call_args.args[2]
    assert stored_frame is frames[1]


def test_visual_backup_does_not_audit_one_frame_below_gate_noise() -> None:
    dependencies = _hooks()
    service = _service(dependencies)
    accepted = MotionQualificationResult(True, 0.69, 0.48, "qualified", 3, {})
    quiet = MotionQualificationResult(False, 0.1, 0.48, "low_score", 3, {})
    service.ema_v2.scene_ready = True
    service.ema_v2.analysis_started_at = 90.0
    service.ema_v2.observation_count = 3
    frame = np.zeros((90, 160, 3), dtype=np.uint8)

    service.consider_visual_backup(accepted, [(100.0, frame)], 100.0)
    service.consider_visual_backup(quiet, [(100.5, frame)], 100.5)

    service.audit_recorder.record_audit.assert_not_called()
    dependencies.media.sample_rejected_motion_frame.assert_not_called()
