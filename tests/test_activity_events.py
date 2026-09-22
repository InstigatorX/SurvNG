from __future__ import annotations

import threading
from datetime import datetime, timezone

from survng.app.activity_events import ActivityEventBus
from survng.app.domain_events import MotionObserved


def _observation(source: str = "onvif") -> MotionObserved:
    return MotionObserved(
        camera_id="gate",
        timestamp="2026-09-22T01:00:00+00:00",
        source=source,
    )


def test_activity_bus_emits_one_start_and_explicit_timeout_stop() -> None:
    transitions = []
    stopped = threading.Event()

    def publish(transition) -> None:
        transitions.append(transition)
        if transition.state.value == "inactive":
            stopped.set()

    bus = ActivityEventBus(
        publish,
        idle_after_seconds=0.05,
        utcnow=lambda: datetime(2026, 9, 22, tzinfo=timezone.utc),
    )
    bus.start_generation("gate", 7)

    bus.observe(_observation(), 7)
    bus.observe(_observation("visual"), 7)

    assert stopped.wait(1.0)
    assert [item.transition.value for item in transitions] == ["started", "stopped"]
    assert transitions[0].sequence == 1
    assert transitions[1].sequence == 2
    assert transitions[1].activity_id == transitions[0].activity_id
    assert transitions[1].sources == ("onvif", "visual")
    assert transitions[1].reason == "inactivity_timeout"


def test_activity_bus_generation_replacement_stops_old_activity() -> None:
    transitions = []
    bus = ActivityEventBus(transitions.append, idle_after_seconds=30)
    bus.start_generation("gate", 1)
    bus.observe(_observation(), 1)

    bus.start_generation("gate", 2)
    bus.observe(_observation(), 2)
    bus.stop_generation("gate", 1, reason="stale")

    assert [item.transition.value for item in transitions] == [
        "started",
        "stopped",
        "started",
    ]
    assert transitions[1].reason == "generation_replaced"
    assert transitions[2].generation == 2
    assert transitions[2].activity_id != transitions[0].activity_id
    bus.close()


def test_activity_bus_close_stops_each_active_camera_once() -> None:
    transitions = []
    bus = ActivityEventBus(transitions.append, idle_after_seconds=30)
    bus.start_generation("gate", 1)
    bus.observe(_observation(), 1)

    bus.close()
    bus.close()
    bus.stop_generation("gate", 1)

    assert [item.reason for item in transitions] == [
        "motion_observed",
        "service_stopped",
    ]


def test_activity_callback_failure_does_not_corrupt_later_state() -> None:
    calls = 0
    delivered = []

    def publish(transition) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("subscriber failed")
        delivered.append(transition)

    bus = ActivityEventBus(publish, idle_after_seconds=30)
    bus.start_generation("gate", 1)
    bus.observe(_observation(), 1)
    bus.stop_generation("gate", 1)

    assert calls == 2
    assert len(delivered) == 1
    assert delivered[0].transition.value == "stopped"
