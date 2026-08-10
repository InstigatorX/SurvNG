from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from survng.app.motion_ingress import MotionEventClock, MotionEventIngressService


def _service(
    *,
    accepting: bool = True,
    detection_enabled: bool = True,
    mode: str = "camera",
) -> tuple[MotionEventIngressService, Mock]:
    owned = Mock()
    owned.events.enqueue.return_value = True
    owned.state.accepting_events.return_value = accepting
    owned.state.detection_enabled.return_value = detection_enabled
    owned.state.begin_ingress.return_value = (
        1 if accepting and detection_enabled else None
    )
    owned.state.wait_ingress_idle.return_value = True
    owned.state.ingress_in_flight.return_value = 0
    owned.state.publish_event = owned.publish_event
    owned.state.set_last_motion_at = owned.set_last_motion_at
    owned.state.increment_stat = owned.increment_stat
    owned.qualification.settings.return_value = (mode, "balanced", 320)
    owned.qualification.observe_event = owned.observe_event
    service = MotionEventIngressService(
        camera_id="gate",
        events=owned.events,
        qualification=owned.qualification,
        state=owned.state,
        epoch_now=lambda: 1_700_000_000.0,
    )
    return service, owned


def test_camera_notice_is_normalized_observed_published_and_enqueued() -> None:
    service, owned = _service()
    local_time = datetime(
        2026,
        8,
        6,
        12,
        30,
        tzinfo=timezone(timedelta(hours=-4)),
    )

    service.handle("onvif/person", "person", local_time)

    camera_time = local_time.astimezone(timezone.utc)
    normalized = datetime.fromtimestamp(1_700_000_000.0, timezone.utc)
    owned.observe_event.assert_called_once_with(
        "onvif/person",
        "person",
        normalized,
        1_700_000_000.0,
    )
    owned.events.remember_priority.assert_called_once_with(1_700_000_000.0)
    owned.events.remember_camera_motion.assert_called_once_with(1_700_000_000.0)
    payload = owned.publish_event.call_args.args[1]
    assert payload["camera_id"] == "gate"
    assert payload["timestamp"] == normalized.isoformat()
    assert payload["source"] == "onvif"
    assert payload["event_timing"]["camera_event_at"] == camera_time.isoformat()
    assert payload["event_timing"]["selection_reason"] == "clock_model_warming"
    trigger = owned.events.enqueue.call_args.args[0]
    assert trigger.topic == "onvif/person"
    assert trigger.event_at == normalized
    assert trigger.received_at == 1_700_000_000.0
    assert trigger.event_timing.camera_event_at == camera_time
    owned.state.end_ingress.assert_called_once_with(1)


def test_event_clock_separates_stable_offset_from_delivery_delay() -> None:
    clock = MotionEventClock()
    base = 1_800_000_000.0
    for index in range(4):
        camera_at = datetime.fromtimestamp(base + index, timezone.utc)
        clock.resolve(camera_at, base + index + 60.0)

    camera_at = datetime.fromtimestamp(base + 10.0, timezone.utc)
    timing = clock.resolve(camera_at, base + 73.0)

    assert timing.selection_reason == "camera_clock_corrected"
    assert timing.estimated_clock_offset_seconds == 60.0
    assert timing.estimated_delivery_delay_seconds == 3.0
    assert timing.sampling_at.timestamp() == base + 70.0


def test_event_clock_uses_plausible_synchronized_camera_time_during_warmup() -> None:
    clock = MotionEventClock()
    camera_at = datetime.fromtimestamp(1_800_000_000.0, timezone.utc)

    timing = clock.resolve(camera_at, 1_800_000_000.4)

    assert timing.selection_reason == "plausible_camera_time"
    assert timing.sampling_at == camera_at
    assert timing.estimated_delivery_delay_seconds == 0.4


def test_adaptive_mode_retains_camera_evidence_without_queuing_detection() -> None:
    service, owned = _service(mode="adaptive")

    service.handle("onvif/motion", "motion")

    owned.observe_event.assert_called_once()
    owned.events.enqueue.assert_not_called()
    owned.events.remember_camera_motion.assert_not_called()
    owned.publish_event.assert_not_called()


def test_manual_topic_matching_is_case_insensitive() -> None:
    service, owned = _service(mode="adaptive")

    service.handle("Manual/test", "operator trigger")

    owned.events.enqueue.assert_called_once()
    owned.events.remember_priority.assert_called_once_with(1_700_000_000.0)
    owned.events.remember_camera_motion.assert_not_called()
    assert owned.publish_event.call_args.args[1]["source"] == "manual"


def test_disabled_or_stopped_ingress_has_no_side_effects() -> None:
    for accepting, detection_enabled in ((False, True), (True, False)):
        service, owned = _service(
            accepting=accepting,
            detection_enabled=detection_enabled,
        )

        service.handle("onvif/motion", "motion")

        owned.observe_event.assert_not_called()
        owned.events.enqueue.assert_not_called()
        owned.set_last_motion_at.assert_not_called()
        owned.state.end_ingress.assert_not_called()


def test_admitted_callback_is_counted_until_qualification_returns() -> None:
    service, owned = _service()
    entered = threading.Event()
    release = threading.Event()

    def observe(*_args: object) -> None:
        entered.set()
        assert release.wait(1.0)

    owned.qualification.observe_event.side_effect = observe
    thread = threading.Thread(
        target=service.handle,
        args=("onvif/motion", "motion"),
    )
    thread.start()
    assert entered.wait(1.0)
    owned.state.end_ingress.assert_not_called()

    release.set()
    thread.join(1.0)
    assert not thread.is_alive()
    owned.state.end_ingress.assert_called_once_with(1)


def test_failed_qualification_always_releases_ingress_admission() -> None:
    service, owned = _service()
    owned.qualification.observe_event.side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        service.handle("onvif/motion", "motion")

    owned.state.end_ingress.assert_called_once_with(1)


def test_enqueue_reports_trigger_and_drop_stats_through_injected_counter() -> None:
    service, owned = _service()
    owned.events.enqueue.side_effect = lambda _trigger, **kwargs: (
        kwargs["on_trigger"]("triggers"),
        kwargs["on_drop"]("dropped_triggers"),
        False,
    )[-1]
    trigger = {
        "topic": "manual/test",
        "message": "test",
        "event_at": datetime.now(timezone.utc),
        "received_at": 1.0,
    }

    assert service.enqueue(trigger, evict_oldest=False) is False

    assert owned.increment_stat.call_args_list[0].args == ("triggers", 1)
    assert owned.increment_stat.call_args_list[1].args == ("dropped_triggers", 1)
    assert owned.events.enqueue.call_args.kwargs["evict_oldest"] is False
