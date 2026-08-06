from __future__ import annotations

import queue
import threading
from datetime import datetime, timezone

import pytest

from survng.app.motion_events import MotionEventCoordinator


def _trigger(index: int = 1) -> dict[str, object]:
    return {
        "topic": "manual/test",
        "message": str(index),
        "event_at": datetime.now(timezone.utc),
        "received_at": float(index),
    }


def test_full_queue_evicts_oldest_trigger_and_reports_drop() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=2)
    stats: list[str] = []
    assert coordinator.enqueue(_trigger(1), on_trigger=stats.append, on_drop=stats.append)
    assert coordinator.enqueue(_trigger(2), on_trigger=stats.append, on_drop=stats.append)
    assert coordinator.enqueue(_trigger(3), on_trigger=stats.append, on_drop=stats.append)

    assert coordinator.queue.get_nowait()["message"] == "2"
    assert coordinator.queue.get_nowait()["message"] == "3"
    assert stats == ["triggers", "triggers", "triggers", "dropped_triggers"]


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
    coordinator.schedule_retry(
        [_trigger()],
        stop_event=stop,
        on_retry=retries.append,
        on_drop=drops.append,
    )
    wrapper = coordinator.next_trigger(timeout=0.01)
    assert wrapper is not None
    retry_batch = wrapper["_retry_batch"]
    assert retry_batch[0]["_event_retry_count"] == 1
    coordinator.schedule_retry(
        retry_batch,
        stop_event=stop,
        on_retry=retries.append,
        on_drop=drops.append,
    )
    assert retries == ["event_retries"]
    assert drops == ["event_retry_drops"]


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
        [{"topic": "adaptive/visual_backup"}],
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
    coordinator.retry_batches.append({**_trigger(2), "_retry_batch": [_trigger(2)]})
    coordinator.clear()
    assert coordinator.queue.empty()
    with pytest.raises(queue.Empty):
        coordinator.next_trigger(timeout=0.001)
