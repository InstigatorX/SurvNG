import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from survng.app.config import CameraConfig, DetectorConfig
from survng.app.event_store import EventStore
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity
from survng.app.native_objects import NativeObjectRegistry
from survng.app.native_event_projection import (
    merge_cover_objects,
    merge_inventory_objects,
)


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
            pts,
            sequence,
            200,
            100,
            tuple(objects),
            "session",
            "native_fresh_detection",
            100 + pts,
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
        CameraConfig(
            id="test",
            name="Test",
            stream_url="rtsp://example.test/live",
        ),
        config,
        events,
        Mock(),
        Mock(return_value=""),
    )

    for sequence in (1, 2):
        feed(
            activity,
            sequence,
            [
                detected("person", 7, 10 + sequence),
                detected("car", 8, 80),
            ],
        )

    assert activity.event_id == 1
    stored = json.loads(events.add_event.call_args.kwargs["objects_json"])
    assert {item["label"] for item in stored} == {"person", "car"}
    car = next(item for item in stored if item["label"] == "car")
    assert car["activity_eligible"] is False

    feed(
        activity,
        3,
        [
            detected("person", 7, 13),
            detected("car", 8, 80),
            detected("bird", 9, 140),
        ],
    )
    activity.persist("active", now=102)
    inventory = events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in inventory} == {"person", "car"}

    feed(
        activity,
        4,
        [
            detected("person", 7, 14),
            detected("car", 8, 80),
            detected("dog", 10, 120),
        ],
    )
    feed(
        activity,
        5,
        [
            detected("person", 7, 15),
            detected("car", 8, 80),
            detected("dog", 10, 121),
        ],
    )
    activity.persist("active", now=103)
    inventory = events.update_native_incident_state.call_args.args[2]
    assert {item["label"] for item in inventory} == {"person", "car", "dog"}

    # Identity/history is owned by the registry, not duplicated by activity.
    assert not hasattr(activity, "inventory_tracks")
    assert not hasattr(activity, "_episode_tracks")
    assert len(activity.inventory.tracking_tracks()) == 3
    # The one-frame bird may remain tentative in the live registry, but it is
    # not promoted into the incident inventory.
    assert len(activity.registry.tracks) == 4


def test_registry_uses_weak_candidates_for_identity_but_not_confirmation():
    config = DetectorConfig(
        enabled=True,
        confidence_threshold=.6,
        event_candidate_confidence_threshold=.25,
        event_confirmation_frames=2,
    )
    registry = NativeObjectRegistry("test", config)

    def observation(confidence):
        item = detected("dog", 7, 10, confidence=confidence)
        item.update(
            confidence_threshold=.6,
            confidence_eligible=confidence >= .6,
            incident_eligible=confidence >= .6,
            zone_eligible=confidence >= .6,
        )
        return item

    seen = registry.observe(
        [observation(.72)],
        session="s",
        identity_epoch=0,
        epoch=100,
        now=10,
        dimensions=(200, 100),
        fresh_fps=5,
    )
    key = next(iter(seen))
    assert registry.export(key)["state"] == "tentative"

    # This weak observation preserves the same temporal identity but cannot
    # satisfy the second configured confirmation.
    registry.observe(
        [observation(.34)],
        session="s",
        identity_epoch=0,
        epoch=100.2,
        now=10.2,
        dimensions=(200, 100),
        fresh_fps=5,
    )
    track = registry.export(key)
    assert track["state"] == "tentative"
    assert track["confirming_observations"] == 1
    assert track["observations"] == 2

    registry.observe(
        [observation(.81)],
        session="s",
        identity_epoch=0,
        epoch=100.4,
        now=10.4,
        dimensions=(200, 100),
        fresh_fps=5,
    )
    track = registry.export(key)
    assert track["state"] == "confirmed"
    assert track["confirming_observations"] == 2
    assert track["observations"] == 3
    # Final confidence is temporal consensus, not the last/highest outlier.
    assert track["confidence"] == .72
    assert track["temporal_consensus"] is True
    assert track["temporal_observations"] == 3
    assert track["temporal_incident_observations"] == 2
    assert track["temporal_required_observations"] == 2
    assert track["temporal_peak_confidence"] == .81
    assert track["temporal_label_votes"] == {"dog": 3}


def test_registry_weak_candidates_alone_never_confirm():
    config = DetectorConfig(
        enabled=True,
        confidence_threshold=.6,
        event_candidate_confidence_threshold=.25,
        event_confirmation_frames=2,
    )
    registry = NativeObjectRegistry("test", config)
    for index, confidence in enumerate((.31, .37, .42), start=1):
        item = detected("dog", 7, 10 + index, confidence=confidence)
        item.update(
            confidence_threshold=.6,
            confidence_eligible=False,
            incident_eligible=False,
            zone_eligible=False,
        )
        seen = registry.observe(
            [item],
            session="s",
            identity_epoch=0,
            epoch=100 + index / 5,
            now=10 + index / 5,
            dimensions=(200, 100),
            fresh_fps=5,
        )
    track = registry.export(next(iter(seen)))
    assert track["state"] == "tentative"
    assert track["confirming_observations"] == 0
    assert track["observations"] == 3


def test_registry_uses_one_identity_when_native_label_changes_with_same_id():
    config = DetectorConfig(enabled=True)
    registry = NativeObjectRegistry("test", config)
    first = detected("car", 7, 10)
    second = detected("truck", 7, 12)
    for item in (first, second):
        item.update(
            confidence_eligible=True,
            incident_eligible=True,
            zone_eligible=True,
        )

    registry.observe(
        [first],
        session="s",
        identity_epoch=0,
        epoch=100,
        now=10,
        dimensions=(200, 100),
        fresh_fps=5,
    )
    seen = registry.observe(
        [second],
        session="s",
        identity_epoch=0,
        epoch=100.2,
        now=10.2,
        dimensions=(200, 100),
        fresh_fps=5,
    )

    assert len(registry.tracks) == 1
    key = next(iter(seen))
    track = registry.export(key)
    assert track["native_track_id"] == 7
    assert track["track_id"] == 1
    assert track["observations"] == 2
    assert len(track["box_history"]) == 2


def test_native_projection_separates_inventory_truth_from_cover_geometry():
    existing = [
        {
            "label": "person",
            "track_id": 1,
            "native_identity": "cam/s/0/7",
            "first_seen": "2026-09-18T10:00:00+00:00",
            "zones": ["door"],
            "activity_eligible": True,
            "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 80},
            "detection_frame_width": 100,
            "detection_frame_height": 100,
            "snapshot_visible": True,
        },
        {
            "label": "car",
            "track_id": 2,
            "native_identity": "cam/s/0/8",
            "zones": [],
            "activity_eligible": False,
            "box": {"x1": 40, "y1": 20, "x2": 80, "y2": 60},
            "snapshot_visible": True,
        },
        {"status": "object_tracking", "object_tracking": {"state": "active"}},
    ]
    cover = [{
        "label": "person",
        "track_id": 1,
        "native_identity": "cam/s/0/7",
        "box": {"x1": 200, "y1": 100, "x2": 500, "y2": 900},
        "detection_frame_width": 2560,
        "detection_frame_height": 1920,
        "frame_source": "recorded_main",
        "snapshot_visible": True,
        "native_cover_verified": True,
    }]
    projected = merge_cover_objects(existing, cover)
    person = next(item for item in projected if item.get("track_id") == 1)
    car = next(item for item in projected if item.get("track_id") == 2)
    assert person["first_seen"] == "2026-09-18T10:00:00+00:00"
    assert person["zones"] == ["door"]
    assert person["box"]["x1"] == 200
    assert person["detection_frame_width"] == 2560
    assert car["snapshot_visible"] is False
    assert next(item for item in projected if item.get("status") == "object_tracking")

    inventory = [{
        "label": "person",
        "track_id": 1,
        "native_identity": "cam/s/0/7",
        "first_seen": "2026-09-18T10:00:00+00:00",
        "zones": ["door", "porch"],
        "activity_eligible": True,
        "box": {"x1": 12, "y1": 10, "x2": 32, "y2": 80},
    }]
    merged = merge_inventory_objects(projected, inventory)
    person = next(item for item in merged if item.get("track_id") == 1)
    assert person["zones"] == ["door", "porch"]
    assert person["box"]["x1"] == 200
    assert person["detection_frame_width"] == 2560


def test_event_store_native_state_merge_preserves_hi_res_cover_and_adds_new_objects(tmp_path):
    store = EventStore(Path(tmp_path))
    event = store.add_event(
        camera_id="test",
        kind="motion",
        created_at=datetime.now(timezone.utc).isoformat(),
        objects_json=json.dumps(
            [
                {
                    **detected("person", 7, 10),
                    "track_id": 1,
                    "native_identity": "test/session/0/7",
                    "box": {
                        "x1": 200,
                        "y1": 100,
                        "x2": 400,
                        "y2": 700,
                    },
                    "detection_frame_width": 2560,
                    "detection_frame_height": 1920,
                    "frame_source": "recorded_main",
                    "snapshot_visible": True,
                },
                {
                    "status": "object_tracking",
                    "object_tracking": {"state": "active"},
                },
            ]
        ),
    )
    inventory = [
        {
            **detected("person", 7, 12),
            "track_id": 1,
            "native_identity": "test/session/0/7",
            "track_state": "confirmed",
        },
        {
            **detected("car", 8, 80),
            "track_id": 2,
            "native_identity": "test/session/0/8",
            "track_state": "confirmed",
        },
    ]
    updated = store.update_native_incident_state(
        int(event["id"]),
        {
            "implementation": "gvatrack",
            "state": "active",
            "tracks": [],
        },
        inventory,
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
