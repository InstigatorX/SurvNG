from survng.app.config import AppConfig, CameraConfig, DetectionZone
from survng.app.manager import AppManager
from survng.app.state_events import StateEventBroker
from survng.app.zone_occupancy import (
    ZoneOccupancyBook,
    labeled_zone_observations,
    scene_look_publishes,
)


def camera() -> CameraConfig:
    return CameraConfig(
        id="lower-garage",
        name="Lower garage",
        stream_url="rtsp://camera/main",
        zones=[
            DetectionZone(name="Driveway", object_classes=["car"], notifications_enabled=False),
            DetectionZone(name="Road", object_classes=["car"]),
            DetectionZone(name="Grass", object_classes=["person"]),
            DetectionZone(name="NoMotion", behavior="ignore", object_classes=["car"]),
        ],
    )


def track(label, current, lifetime=None, *, active=True, timestamp="2026-10-16T19:32:27+00:00"):
    return {
        "camera_id": "lower-garage",
        "timestamp": timestamp,
        "source": "track",
        "active": active,
        "tracks": [{
            "label": label,
            "current_zones": list(current),
            "zones": list(lifetime if lifetime is not None else current),
        }],
    }


def look(objects, timestamp="2026-10-16T19:34:27+00:00"):
    return {
        "camera_id": "lower-garage",
        "timestamp": timestamp,
        "source": "scene",
        "objects": objects,
    }


def names(snapshot):
    return [item["zone"] for item in snapshot["zones"]]


def test_track_zone_change_publishes_once_until_the_zone_changes():
    book = ZoneOccupancyBook()
    gate = camera()
    arrived = book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240)
    assert names(arrived) == ["Driveway"]
    assert book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240) is None
    moved = book.observe(
        gate,
        track("car", ["Road"], ["Driveway", "Road"]),
        fail_safe_seconds=240,
    )
    assert names(moved) == ["Road"]


def test_lost_track_and_session_end_hold_until_a_look():
    book = ZoneOccupancyBook()
    gate = camera()
    book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240)
    lost = book.observe(gate, track("car", [], []), fail_safe_seconds=240)
    assert lost is None
    ended = book.observe(gate, {**track("car", [], []), "active": False}, fail_safe_seconds=240)
    assert ended is None
    still = book.observe(
        gate,
        look([{"label": "car", "zones": ["Driveway"], "activity_role": "scene_context"}]),
        fail_safe_seconds=240,
    )
    assert names(still) == ["Driveway"]
    assert still["fail_safe_seconds"] == 240


def test_later_look_that_misses_the_vehicle_clears_the_zone():
    book = ZoneOccupancyBook()
    gate = camera()
    book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240)
    book.observe(gate, {**track("car", ["Driveway"]), "active": False}, fail_safe_seconds=240)
    cleared = book.observe(gate, look([]), fail_safe_seconds=240)
    assert cleared["zones"] == []


def test_notification_disabled_zone_stays_occupied_and_filters_still_apply():
    book = ZoneOccupancyBook()
    snapshot = book.observe(
        camera(),
        look([
            {"label": "car", "zones": ["Driveway", "Grass", "NoMotion"]},
            {"label": "person", "zones": ["Grass"]},
        ]),
        fail_safe_seconds=240,
    )
    assert names(snapshot) == ["Driveway", "Grass"]


def test_a_later_look_can_see_a_zone_an_earlier_session_left():
    book = ZoneOccupancyBook()
    gate = camera()
    book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240)
    book.observe(gate, track("car", ["Road"], ["Driveway", "Road"]), fail_safe_seconds=240)
    book.observe(
        gate,
        {**track("person", ["Grass"]), "timestamp": "2026-10-16T19:33:00+00:00"},
        fail_safe_seconds=240,
    )
    restored = book.observe(
        gate,
        look([{"label": "car", "zones": ["Driveway"]}, {"label": "person", "zones": ["Grass"]}]),
        fail_safe_seconds=240,
    )
    assert names(restored) == ["Driveway", "Grass"]


def test_live_departure_wins_over_a_look_that_still_lists_the_old_zone():
    book = ZoneOccupancyBook()
    gate = camera()
    book.observe(gate, track("car", ["Driveway"]), fail_safe_seconds=240)
    book.observe(gate, track("car", ["Road"], ["Driveway", "Road"]), fail_safe_seconds=240)
    stayed = book.observe(
        gate,
        look([{"label": "car", "zones": ["Driveway", "Road"]}]),
        fail_safe_seconds=240,
    )
    assert names(stayed) == ["Road"]


def test_scene_verdict_gates_occupancy_publication():
    assert scene_look_publishes("pending", detection_completed=True)
    assert scene_look_publishes("supported", detection_completed=True)
    assert scene_look_publishes(None, detection_completed=True)
    assert not scene_look_publishes("unsupported", detection_completed=True)
    assert not scene_look_publishes("incomplete", detection_completed=True)
    assert not scene_look_publishes("pending", detection_completed=False)


def test_scene_observations_keep_their_zones():
    objects = labeled_zone_observations([
        {"status": "scene_observations", "observations": [
            {"payload": {"label": "car", "zones": ["Driveway"], "activity_role": "scene_context"}},
        ]},
    ])
    assert objects == [{"label": "car", "zones": ["Driveway"]}]


def test_manager_publishes_changes_and_scene_refreshes():
    manager = object.__new__(AppManager)
    manager.config = AppConfig()
    manager.config.cameras.append(camera())
    manager.config.detector.scene_discovery_heartbeat_seconds = 120
    manager.state_events = StateEventBroker()
    manager.publish_event("zone_occupancy_source", track("car", ["Driveway"]))
    manager.publish_event("zone_occupancy_source", track("car", ["Driveway"]))
    manager.publish_event("zone_occupancy_source", look([
        {"label": "car", "zones": ["Driveway"]},
    ]))
    events = [
        event for event in manager.state_events.events_after(f"{manager.state_events.instance_id}:0")
        if event.type == "zone_occupancy"
    ]
    assert len(events) == 2
    assert events[0].data["zones"] == [{"zone": "Driveway", "classes": ["car"]}]
    assert events[0].data["fail_safe_seconds"] == 240
    assert events[1].data["zones"] == [{"zone": "Driveway", "classes": ["car"]}]
    assert events[1].data["fail_safe_seconds"] == 240
