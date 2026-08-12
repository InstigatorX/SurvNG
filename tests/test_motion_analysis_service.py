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
from survng.app.ema_v2 import EmaPolicy
from survng.app.motion_events import MotionEventCoordinator
from survng.app.motion_pipeline import MotionDebugSnapshotStore
from survng.app.config import MotionQualificationConfig


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


def _service(dependencies: SimpleNamespace, queue_size: int = 1) -> MotionAnalysisService:
    service = MotionAnalysisService(
        camera_id="gate",
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=8,
        queue_size=queue_size,
        limiter=FairMotionAnalysisLimiter(1),
        events=MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id="gate",
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
                (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
                (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
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
            (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
            (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
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

    assert service.events.episode_controller.snapshot()["request_status"] == "aborted"


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
                (99.5, np.zeros((90, 160, 3), dtype=np.uint8)),
                (100.0, np.zeros((90, 160, 3), dtype=np.uint8)),
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
