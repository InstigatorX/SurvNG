from __future__ import annotations

import queue
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from survng.app.ema_v2 import CameraNotice
from survng.app.events import EventStore
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


def test_durable_trigger_preserves_generation_qualified_evidence_token() -> None:
    trigger = MotionTrigger(
        topic="adaptive/motion",
        message="qualified EMA evidence",
        event_at=datetime.now(timezone.utc),
        received_at=100.25,
        lifecycle_generation=4,
        evidence_frame_at_epoch=100.125,
        evidence_frame_sequence=27,
        evidence_capture_generation=9,
    )

    recovered = MotionTrigger.from_durable_payload(
        trigger.durable_payload(),
        "motion-trigger:test",
    )

    assert recovered.evidence_frame_at_epoch == 100.125
    assert recovered.evidence_frame_sequence == 27
    assert recovered.evidence_capture_generation == 9
    assert recovered.lifecycle_generation == 4


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


def test_durable_trigger_overflow_survives_coordinator_restart() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        first = MotionEventCoordinator(
            queue_size=1,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        assert first.enqueue(_trigger(1), evict_oldest=False)
        assert first.enqueue(_trigger(2), evict_oldest=False)
        assert first.runtime_status()["durable_delivery"] == {"queued": 2}

        restored_store = EventStore(Path(tmpdir))
        restored = MotionEventCoordinator(
            queue_size=1,
            retry_limit=2,
            camera_id="gate",
            durable_store=restored_store,
        )
        recovered = restored.next_trigger(timeout=0.2)
        assert recovered is not None
        assert recovered.message == "1"
        restored.complete_deliveries(MotionTriggerBatch((recovered,)))
        second = restored.next_trigger(timeout=0.2)
        assert second is not None
        assert second.message == "2"


def test_route_trigger_replay_with_capture_drift_is_idempotent() -> None:
    """Repeated route replays must not raise or create a second durable job."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        coordinator = MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        route_id = "route:gate:lower-garage:53952"
        first = MotionTrigger(
            topic="adaptive/visual_backup",
            message="route watch replay",
            event_at=datetime(2026, 8, 22, 18, 10, 1, tzinfo=timezone.utc),
            received_at=100.0,
            episode_id=route_id,
            detection_intent_id=route_id,
            lifecycle_generation=4,
        )
        replay = MotionTrigger(
            topic="adaptive/visual_backup",
            message="route watch replay",
            event_at=datetime(2026, 8, 22, 18, 10, 7, tzinfo=timezone.utc),
            received_at=106.0,
            episode_id=route_id,
            detection_intent_id=route_id,
            lifecycle_generation=5,
        )
        assert coordinator.enqueue(first, evict_oldest=False)
        # Same route identity with drifted capture/lifecycle metadata.
        assert coordinator.enqueue(replay, evict_oldest=False)
        assert store.motion_trigger_status("gate") == {"queued": 1}
        with pytest.raises(RuntimeError, match="different occurrence"):
            store.enqueue_motion_trigger(
                job_id=route_id,
                camera_id="porch",
                payload=first.durable_payload(),
            )


def test_duplicate_durable_trigger_does_not_queue_a_second_wake() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        coordinator = MotionEventCoordinator(
            queue_size=4,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        route_id = "route:gate:back-left:58395"
        first = MotionTrigger(
            topic="adaptive/visual_backup",
            message="route watch",
            event_at=datetime(2026, 8, 28, 13, 26, 50, tzinfo=timezone.utc),
            received_at=100.0,
            detection_intent_id=route_id,
        )
        replay = MotionTrigger(
            topic="adaptive/visual_backup",
            message="route watch",
            event_at=datetime(2026, 8, 28, 13, 27, 4, tzinfo=timezone.utc),
            received_at=114.0,
            detection_intent_id=route_id,
        )

        assert coordinator.enqueue(first, evict_oldest=False)
        assert coordinator.enqueue(replay, evict_oldest=False)
        assert coordinator.queue.qsize() == 1
        assert coordinator.next_trigger(timeout=0.1) is first
        with pytest.raises(queue.Empty):
            coordinator.next_trigger(timeout=0.02)


def test_durable_wake_eviction_is_not_reported_as_trigger_loss() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EventStore(Path(tmpdir))
        coordinator = MotionEventCoordinator(
            queue_size=1,
            retry_limit=2,
            camera_id="gate",
            durable_store=store,
        )
        drops: list[str] = []
        assert coordinator.enqueue(_trigger(1), on_drop=drops.append)
        assert coordinator.enqueue(_trigger(2), on_drop=drops.append)

        status = coordinator.runtime_status()
        assert drops == []
        assert status["evicted"] == 0
        assert status["durable_wake_evictions"] == 1
        assert status["durable_delivery"] == {"queued": 2}


def test_stale_durable_wakeups_are_drained_iteratively() -> None:
    store = Mock()
    recovered = _trigger(9999).durable_payload()

    def claim(_camera_id, job_id=None, **_kwargs):
        if job_id is not None:
            return None
        return {"id": "recovered", "payload": recovered}

    store.claim_motion_trigger.side_effect = claim
    store.motion_trigger_status.return_value = {}
    coordinator = MotionEventCoordinator(
        queue_size=1500,
        retry_limit=1,
        camera_id="gate",
        durable_store=store,
    )
    for index in range(1200):
        trigger = _trigger(index)
        trigger.delivery_job_id = f"stale-{index}"
        coordinator.queue.put_nowait(trigger)

    result = coordinator.next_trigger(timeout=1.0)

    assert result is not None
    assert result.delivery_job_id == "recovered"
    assert coordinator.queue.empty()


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
    coordinator.episode_controller.observe_camera(
        CameraNotice("camera", 102.0, 102.0, "onvif/motion"), generation=0
    )
    sequence = coordinator.current_episode_sequence()
    coordinator.link_incident(42)

    coordinator.reset_timebase()

    assert coordinator.next_trigger(timeout=0.001) is trigger
    assert coordinator.current_episode_sequence() == sequence + 1
    assert coordinator.active_incident_event_id() is None
    assert not coordinator.link_incident(84, expected_sequence=sequence)


def test_episode_controller_owns_incident_linkage() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    coordinator.episode_controller.observe_camera(
        CameraNotice("camera", 100.0, 100.0, "onvif/motion"), generation=0
    )
    assert coordinator.link_incident(42)

    snapshot = coordinator.episode_snapshot()

    assert snapshot["sequence"] == 1
    assert coordinator.active_incident_event_id() == 42


def test_stale_refinement_cannot_link_into_a_new_episode() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    first = coordinator.episode_controller.observe_camera(
        CameraNotice("camera", 100.0, 100.0, "onvif/motion"), generation=0
    )
    assert first.intent is not None
    coordinator.episode_controller.acknowledge_admission(
        first.intent.intent_id,
        admitted=True,
        occurred_monotonic=100.1,
    )
    coordinator.episode_controller.complete(
        first.intent.intent_id,
        occurred_monotonic=100.2,
    )
    sequence = coordinator.current_episode_sequence()
    coordinator.episode_controller.observe_camera(
        CameraNotice("camera", 140.0, 140.0, "onvif/motion"), generation=0
    )

    assert not coordinator.link_incident(42, expected_sequence=sequence)
    assert coordinator.active_incident_event_id() is None


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


def test_failed_batch_is_coordinator_owned_until_retry_finishes() -> None:
    coordinator = MotionEventCoordinator(queue_size=2, retry_limit=1)
    batch = MotionTriggerBatch((_trigger(topic="adaptive/visual_backup"),))
    coordinator.set_active(batch)
    assert coordinator.take_failed_active() is batch
