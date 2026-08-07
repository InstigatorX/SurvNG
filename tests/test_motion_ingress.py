from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from survng.app.motion_ingress import MotionEventIngressService


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

    normalized = local_time.astimezone(timezone.utc)
    owned.observe_event.assert_called_once_with(
        "onvif/person",
        "person",
        normalized,
        1_700_000_000.0,
    )
    owned.events.remember_priority.assert_called_once_with(1_700_000_000.0)
    owned.events.remember_camera_motion.assert_called_once_with(1_700_000_000.0)
    owned.publish_event.assert_called_once_with("motion", {
        "camera_id": "gate",
        "timestamp": normalized.isoformat(),
        "source": "onvif",
    })
    trigger = owned.events.enqueue.call_args.args[0]
    assert trigger.topic == "onvif/person"
    assert trigger.event_at == normalized
    assert trigger.received_at == 1_700_000_000.0


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
