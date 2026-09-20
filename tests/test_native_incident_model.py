"""Durable multi-object time-range incidents without track-gated lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from survng.app.config import CameraConfig, DetectorConfig
from survng.app.event_store import EventStore
from survng.app.live_detections import DetectionSnapshot
from survng.app.native_activity import NativeActivity


def _detected(label, x, *, native_id=None, confidence=0.9):
    obj = {
        "label": label,
        "confidence": confidence,
        "detection_provenance": "native_fresh_detection",
        "box": {"x1": x, "y1": 10, "x2": x + 20, "y2": 50},
    }
    if native_id is not None:
        obj["native_track_id"] = native_id
    return obj


def _feed(activity, sequence, objects, *, now=None, epoch=None):
    pts = sequence / 5
    when = 100 + pts if now is None else now
    stamp = 1000 + pts if epoch is None else epoch
    activity.consume(
        DetectionSnapshot(
            pts,
            sequence,
            200,
            100,
            tuple(objects),
            "session",
            "native_fresh_detection",
            when,
        ),
        now=when,
        epoch=stamp,
    )


def _activity(tmp_path: Path) -> NativeActivity:
    events = EventStore(tmp_path)
    return NativeActivity(
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://example.test/live"),
        DetectorConfig(enabled=True, event_confirmation_frames=1, native={"stationary": {"labels": []}}),
        events,
        Mock(),
        Mock(return_value=""),
    )


def test_store_open_append_close_round_trip(tmp_path):
    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="yard",
        kind="motion",
        topic="native/object-presence",
        message="test",
        created_at="2026-09-20T12:00:00+00:00",
        objects_json="[]",
    )
    opened = store.open_incident(
        camera_id="yard",
        start_at="2026-09-20T12:00:00+00:00",
        participants=[{"label": "person", "box": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}}],
        observation_objects=[{"label": "person"}],
        seed_event_id=int(event["id"]),
    )
    assert opened["state"] == "active"
    assert opened["end_at"] is None
    assert opened["observation_count"] == 1
    store.append_incident_observation(
        opened["id"],
        observed_at="2026-09-20T12:00:02+00:00",
        objects=[{"label": "person"}, {"label": "dog"}],
        participants=[
            {"label": "person"},
            {"label": "dog"},
        ],
    )
    closed = store.close_incident(
        opened["id"],
        end_at="2026-09-20T12:00:07+00:00",
        state="complete",
        completion_reason="inactivity",
        participants=[{"label": "person"}, {"label": "dog"}],
    )
    assert closed["state"] == "complete"
    assert closed["end_at"] == "2026-09-20T12:00:07+00:00"
    assert closed["observation_count"] == 2
    observations = store.list_incident_observations(opened["id"])
    assert [item["seq"] for item in observations] == [1, 2]
    assert {obj["label"] for obj in observations[1]["objects"]} == {"person", "dog"}
    assert store.incident_for_event(int(event["id"]))["id"] == opened["id"]


def test_two_concurrent_labels_share_one_incident(tmp_path):
    activity = _activity(tmp_path)
    for sequence in (1, 2):
        _feed(
            activity,
            sequence,
            [
                _detected("person", 10, native_id=None),
                _detected("dog", 80, native_id=None),
            ],
        )
    assert activity.event_id == 1
    assert activity.incident_id is not None
    incident = activity.events.get_incident(activity.incident_id)
    labels = {item["label"] for item in incident["participants"]}
    assert labels == {"person", "dog"}
    assert incident["observation_count"] >= 1
    stored = json.loads(activity.events.get(activity.event_id)["objects_json"])
    assert not any(item.get("status") == "object_tracking" for item in stored)
    assert any(item.get("status") == "native_incident" for item in stored)


def test_mid_incident_object_joins_same_incident(tmp_path):
    activity = _activity(tmp_path)
    for sequence in (1, 2):
        _feed(activity, sequence, [_detected("person", 10 + sequence)])
    first_incident = activity.incident_id
    assert first_incident is not None
    for sequence in (3, 4):
        _feed(
            activity,
            sequence,
            [
                _detected("person", 14),
                _detected("car", 120),
            ],
        )
    assert activity.incident_id == first_incident
    incident = activity.events.get_incident(first_incident)
    labels = {item["label"] for item in incident["participants"]}
    assert labels == {"person", "car"}
    assert activity.event_id is not None
    with activity.events._connect() as conn:
        count = conn.execute("select count(*) as n from events").fetchone()["n"]
    assert int(count) == 1


def test_inactivity_close_then_reopen_creates_new_incident(tmp_path):
    activity = _activity(tmp_path)
    for sequence in (1, 2):
        _feed(activity, sequence, [_detected("person", 10 + sequence)])
    first = activity.incident_id
    for sequence in range(3, 30):
        _feed(activity, sequence, [], now=100 + sequence / 5)
    assert activity.incident_id is None
    closed = activity.events.get_incident(first)
    assert closed["state"] == "complete"
    assert closed["end_at"]
    for sequence in (40, 41):
        _feed(
            activity,
            sequence,
            [_detected("person", 30)],
            now=120 + sequence / 5,
            epoch=1200 + sequence / 5,
        )
    assert activity.incident_id is not None
    assert activity.incident_id != first


def test_registry_ignores_native_track_ids_for_association(tmp_path):
    activity = _activity(tmp_path)
    # Same box, flipping native IDs must stay one soft association.
    for sequence, native_id in ((1, 7), (2, 99), (3, 7)):
        _feed(
            activity,
            sequence,
            [_detected("person", 12, native_id=native_id)],
        )
    assert len(activity.registry.tracks) == 1
    person = next(iter(activity.registry.tracks.values()))
    assert person["observations"] == 3
    assert activity.incident_id is not None


def test_evidence_nominates_from_incident_observations(tmp_path):
    from survng.app.native_evidence import incident_observation_samples

    store = EventStore(tmp_path)
    event = store.add_event(
        camera_id="yard",
        kind="motion",
        topic="native/object-presence",
        created_at="2026-09-20T12:00:00+00:00",
        objects_json="[]",
    )
    opened = store.open_incident(
        camera_id="yard",
        start_at="2026-09-20T12:00:00+00:00",
        participants=[{"label": "person"}],
        observation_objects=[
            {
                "label": "person",
                "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 40},
            }
        ],
        seed_event_id=int(event["id"]),
    )
    store.append_incident_observation(
        opened["id"],
        observed_at="2026-09-20T12:00:02+00:00",
        objects=[
            {
                "label": "person",
                "box": {"x1": 12, "y1": 10, "x2": 32, "y2": 40},
            },
            {
                "label": "dog",
                "box": {"x1": 80, "y1": 10, "x2": 100, "y2": 40},
            },
        ],
    )
    store.update_native_incident_state(
        int(event["id"]),
        {
            "implementation": "native_observations",
            "state": "complete",
            "incident_id": opened["id"],
            "frame_width": 200,
            "frame_height": 100,
            "updated_at": "2026-09-20T12:00:02+00:00",
        },
        [{"label": "person"}, {"label": "dog"}],
    )
    samples = incident_observation_samples(
        store,
        store.get(int(event["id"])),
        {"implementation": "native_observations", "incident_id": opened["id"]},
    )
    assert len(samples) == 2
    assert {obj["label"] for obj in samples[1]["objects"]} == {"person", "dog"}
    assert not any("box_history" in obj for sample in samples for obj in sample["objects"])


def test_admission_without_native_track_ids(tmp_path):
    activity = _activity(tmp_path)
    objects = [_detected("person", 12), _detected("dog", 90)]
    _feed(activity, 1, objects)
    _feed(activity, 2, objects)
    assert activity.event_id is not None
    assert activity.incident_id is not None
    payload = json.loads(activity.events.get(activity.event_id)["objects_json"])
    lifecycle = next(item for item in payload if item.get("status") == "native_incident")
    assert lifecycle["native_incident"]["implementation"] == "native_observations"
    assert "object_tracking" not in lifecycle
