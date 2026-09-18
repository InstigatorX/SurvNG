import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from survng.app.config import CameraConfig, DetectorConfig
from survng.app.event_store import EventStore
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity


def detected(label, native_id, x, *, confidence=.9):
    return {
        "label": label,
        "confidence": confidence,
        "native_track_id": native_id,
        "detection_provenance": "native_fresh_detection",
        "box": {"x1": x, "y1": 10, "x2": x + 20, "y2": 50},
    }


def feed(activity, sequence, objects):
    pts = sequence / 5
    activity.consume(
        DetectionSnapshot(
            pts, sequence, 200, 100, tuple(objects), "session",
            "native_fresh_detection", 100 + pts,
        ),
        now=100 + pts,
        epoch=1000 + pts,
    )


def test_incident_inventory_keeps_confirmed_context_objects_without_using_them_for_admission():
    events = Mock()
    events.add_event.return_value = {"id": 1}
    config = DetectorConfig(
        enabled=True,
        native={"stationary": {"labels": ["car"]}},
    )
    activity = NativeActivity(
        CameraConfig(id="test", name="Test", stream_url="rtsp://example.test/live"),
        config, events, Mock(), Mock(return_value=""),
    )

    # Person starts the incident. The parked car is confirmed context but its
    # native stationary state is not allowed to drive activity.
    for sequence in (1, 2):
        feed(activity, sequence, [
            detected("person", 7, 10 + sequence),
            detected("car", 8, 80),
        ])

    assert activity.event_id == 1
    stored = json.loads(events.add_event.call_args.kwargs["objects_json"])
    assert {item["label"] for item in stored} == {"person", "car"}
    car = next(item for item in stored if item["label"] == "car")
    assert car["activity_eligible"] is False

    # A one-frame detector glitch is not temporally credible and never joins
    # the incident inventory.
    feed(activity, 3, [
        detected("person", 7, 13),
        detected("car", 8, 80),
        detected("bird", 9, 140),
    ])
    activity.persist("active", now=102)
    inventory = events.update_object_tracking.call_args.args[2]
    assert {item["label"] for item in inventory} == {"person", "car"}

    # A real secondary object appearing after incident admission is added once
    # it satisfies the same temporal confirmation requirement.
    feed(activity, 4, [
        detected("person", 7, 14),
        detected("car", 8, 80),
        detected("dog", 10, 120),
    ])
    feed(activity, 5, [
        detected("person", 7, 15),
        detected("car", 8, 80),
        detected("dog", 10, 121),
    ])
    activity.persist("active", now=103)
    inventory = events.update_object_tracking.call_args.args[2]
    assert {item["label"] for item in inventory} == {"person", "car", "dog"}
    assert events.update_object_tracking.call_args.kwargs["replace_objects"] is True


def test_event_store_inventory_merge_preserves_hi_res_cover_and_adds_new_objects(tmp_path):
    store = EventStore(Path(tmp_path))
    event = store.add_event(
        camera_id="test",
        kind="motion",
        created_at=datetime.now(timezone.utc).isoformat(),
        objects_json=json.dumps([
            {
                **detected("person", 7, 10),
                "track_id": 1,
                "box": {"x1": 200, "y1": 100, "x2": 400, "y2": 700},
                "detection_frame_width": 2560,
                "detection_frame_height": 1920,
                "frame_source": "recorded_main",
                "snapshot_visible": True,
            },
            {"status": "object_tracking", "object_tracking": {"state": "active"}},
        ]),
    )
    inventory = [
        {**detected("person", 7, 12), "track_id": 1, "track_state": "confirmed"},
        {**detected("car", 8, 80), "track_id": 2, "track_state": "confirmed"},
    ]
    updated = store.update_object_tracking(
        int(event["id"]),
        {"implementation": "gvatrack", "state": "active", "tracks": []},
        inventory,
        replace_objects=True,
    )
    objects = json.loads(updated["objects_json"])
    labels = [item for item in objects if item.get("label")]
    assert {item["label"] for item in labels} == {"person", "car"}
    person = next(item for item in labels if item["label"] == "person")
    car = next(item for item in labels if item["label"] == "car")
    assert person["box"]["x1"] == 200
    assert person["detection_frame_width"] == 2560
    assert person["frame_source"] == "recorded_main"
    assert person["snapshot_visible"] is True
    assert car["snapshot_visible"] is False
