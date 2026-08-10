from __future__ import annotations

import queue
import threading
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from survng.app.motion_events import (
    MotionEventCoordinator,
    MotionTrigger,
    MotionTriggerBatch,
    RetryDisposition,
)


def _trigger(index: int = 1, *, topic: str = "manual/test") -> MotionTrigger:
    return MotionTrigger(
        topic=topic,
        message=str(index),
        event_at=datetime.now(timezone.utc),
        received_at=float(index),
    )


def test_full_queue_evicts_oldest_trigger_and_reports_drop() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=2)
    stats: list[str] = []
    assert coordinator.enqueue(_trigger(1), on_trigger=stats.append, on_drop=stats.append)
    assert coordinator.enqueue(_trigger(2), on_trigger=stats.append, on_drop=stats.append)
    assert coordinator.enqueue(_trigger(3), on_trigger=stats.append, on_drop=stats.append)

    assert coordinator.queue.get_nowait().message == "2"
    assert coordinator.queue.get_nowait().message == "3"
    assert stats == ["triggers", "triggers", "triggers", "dropped_triggers"]
    runtime = coordinator.runtime_status()
    assert runtime["enqueued"] == 3
    assert runtime["evicted"] == 1
    assert runtime["rejected"] == 0
    assert runtime["queue_high_water"] == 2


def test_full_queue_race_does_not_report_an_eviction_that_never_happened() -> None:
    coordinator = MotionEventCoordinator(queue_size=1, retry_limit=2)
    drops: list[str] = []
    coordinator.queue.put_nowait(_trigger(1))
    original_get = coordinator.queue.get_nowait

    def drained_before_eviction() -> MotionTrigger:
        original_get()
        raise queue.Empty

    coordinator.queue.get_nowait = Mock(side_effect=drained_before_eviction)

    assert coordinator.enqueue(_trigger(2), on_drop=drops.append)
    assert coordinator.queue.queue[0].received_at == 2.0
    assert drops == []


def test_successful_enqueue_records_minimum_sampled_high_water() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=2)
    coordinator.queue.qsize = Mock(return_value=0)

    assert coordinator.enqueue(_trigger())

    assert coordinator.runtime_status()["queue_high_water"] == 1


def test_coalesce_preserves_batch_order_and_stop_sentinel() -> None:
    coordinator = MotionEventCoordinator(queue_size=4, retry_limit=2)
    coordinator.queue.put_nowait(_trigger(2))
    coordinator.queue.put_nowait(None)

    assert coordinator.coalesce(
        _trigger(1),
        quiet_seconds=0.1,
        stop_event=threading.Event(),
    ) is None


def test_retry_batch_is_prioritized_and_bounded() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    retries: list[str] = []
    drops: list[str] = []
    stop = threading.Event()
    assert coordinator.schedule_retry(
        [_trigger()],
        stop_event=stop,
        on_retry=retries.append,
        on_drop=drops.append,
    ) == RetryDisposition.SCHEDULED
    wrapper = coordinator.next_trigger(timeout=0.01)
    assert wrapper is not None
    retry_batch = wrapper.retry_batch
    assert retry_batch[0].retry_count == 1
    assert coordinator.schedule_retry(
        retry_batch,
        stop_event=stop,
        on_retry=retries.append,
        on_drop=drops.append,
    ) == RetryDisposition.DROPPED
    assert retries == ["event_retries"]
    assert drops == ["event_retry_drops"]
    runtime = coordinator.runtime_status()
    assert runtime["retries_scheduled"] == 1
    assert runtime["retries_dropped"] == 1
    assert runtime["retry_high_water"] == 1


def test_adaptive_reservation_deduplicates_priority_and_rearms() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    coordinator.remember_priority(100.0)
    assert not coordinator.reserve_adaptive(
        101.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )
    assert coordinator.reserve_adaptive(
        110.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )
    coordinator.complete_adaptive(
        [_trigger(topic="adaptive/visual_backup")],
        completed_at=110.0,
    )
    assert not coordinator.reserve_adaptive(
        112.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )
    assert coordinator.reserve_adaptive(
        116.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )


def test_clear_removes_primary_and_retry_queues() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    coordinator.queue.put_nowait(_trigger())
    retry = _trigger(2)
    coordinator.retry_batches.append(MotionTrigger(
        topic="internal/retry_batch",
        message="retry",
        event_at=retry.event_at,
        received_at=retry.received_at,
        retry_batch=(retry,),
    ))
    coordinator.clear()
    assert coordinator.queue.empty()
    with pytest.raises(queue.Empty):
        coordinator.next_trigger(timeout=0.001)


def test_timebase_reset_preserves_queued_work_and_active_reservation() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    trigger = _trigger()
    coordinator.enqueue(trigger)
    coordinator.adaptive_trigger_pending = True
    coordinator.adaptive_last_completed_at = 100.0
    coordinator.remember_priority(101.0)
    coordinator.remember_camera_motion(102.0)

    coordinator.reset_timebase()

    assert coordinator.next_trigger(timeout=0.001) is trigger
    assert coordinator.adaptive_trigger_pending is True
    assert coordinator.adaptive_last_completed_at == 0.0
    assert tuple(coordinator.priority_motion_times) == ()
    assert coordinator.camera_motion_snapshot() == ()


def test_episode_ledger_unifies_source_history_and_incident_linkage() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    coordinator.remember_camera_motion(100.0)
    coordinator.remember_priority(101.0)
    coordinator.link_incident(42)

    snapshot = coordinator.episode_snapshot()

    assert snapshot["sequence"] == 1
    assert [item.source for item in snapshot["observations"]] == ["camera", "priority"]
    assert coordinator.camera_motion_snapshot() == (100.0,)
    assert coordinator.active_incident_event_id() == 42

    coordinator.remember_camera_motion(140.0)

    assert coordinator.episode_snapshot()["sequence"] == 2
    assert coordinator.active_incident_event_id() is None


def test_adaptive_reservation_starts_an_episode_without_camera_observation() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)

    assert coordinator.reserve_adaptive(
        100.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )

    snapshot = coordinator.episode_snapshot()
    assert snapshot["sequence"] == 1
    assert snapshot["observations"][0].source == "adaptive"


def test_retry_queue_depth_reports_coordinator_owned_state() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    retry = _trigger(2)
    coordinator.retry_batches.append(retry)

    assert coordinator.retry_queue_depth() == 1

    assert coordinator.next_trigger(timeout=0.001) is retry
    assert coordinator.retry_queue_depth() == 0


def test_coordinator_rejects_untyped_payloads() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    trigger = {
        "topic": "onvif/motion",
        "message": "motion",
        "event_at": datetime.now(timezone.utc),
        "received_at": 1.0,
        "typo_decison_id": "silently-lost-before",
    }

    with pytest.raises(TypeError, match="MotionTrigger"):
        coordinator.enqueue(trigger)


def test_typed_batch_assignments_do_not_change_trigger_order() -> None:
    coordinator = MotionEventCoordinator(queue_size=4, retry_limit=1)
    first = _trigger(1)
    second = _trigger(2)
    coordinator.queue.put_nowait(second)

    batch = coordinator.coalesce(
        first,
        quiet_seconds=0.001,
        stop_event=threading.Event(),
    )

    assert batch is not None
    assert tuple(item.message for item in batch) == ("1", "2")
    assert batch.triggers == (first, second)


def test_failed_adaptive_batch_remains_reserved_until_retry_finishes() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    adaptive = _trigger(topic="adaptive/visual_backup")
    batch = MotionTriggerBatch((adaptive,))
    coordinator.set_active(batch)
    coordinator.adaptive_trigger_pending = True

    assert coordinator.take_failed_active() is batch
    assert coordinator.adaptive_trigger_pending
    assert not coordinator.reserve_adaptive(
        adaptive.received_at + 10.0,
        rearm_seconds=5.0,
        priority_tolerance_seconds=2.0,
    )
