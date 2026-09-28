"""Track namespaces share subjects only when evidence and corrections permit it."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from survng.app.appearance_index import AppearanceIndex
from survng.app.events import EventStore

BASE = 1800000000


def observation(second, key="recorded:R", x=10, source="recorded_main", **extra):
    return {"label": "person", "confidence": .9, "captured_at_epoch": BASE + second,
            "scene_track_key": key, "frame_source": source, "frame_timestamp_exact": True,
            "recording_path": "recordings/gate.mp4", "detection_frame_width": 100,
            "detection_frame_height": 100, "box": {"x1": x, "y1": 10, "x2": x + 20, "y2": 60}, **extra}


def add(store, observations, camera="gate"):
    return store.add_event(camera, "motion", created_at=datetime.fromtimestamp(BASE, timezone.utc).isoformat(),
                           objects_json=json.dumps([{"status": "scene_observations", "observations": observations}]))


def aliases(store):
    with store._connect() as conn:
        return {row["track_key"]: row["object_id"] for row in conn.execute("select * from scene_track_aliases")}


@pytest.mark.parametrize("tracking_source", ["recorded_main", "tracking"])
def test_recorded_and_preroll_tracking_fuse_stably(tmp_path, tracking_source):
    store = EventStore(tmp_path)
    event = add(store, [observation(t) for t in (0, 1, 2)])
    original = store.scene_incident(event_id=event["id"])["scene_objects"][0]["id"]
    batch = [observation(t, "tracking:T", source=tracking_source) for t in (-5, -4, 0, 3)]
    store.record_scene_observations(event["id"], batch)
    current = store.scene_incident(event_id=event["id"])
    assert len(current["scene_objects"]) == 1
    subject = current["scene_objects"][0]
    assert subject["id"] == original
    assert subject["first_seen_at"] == datetime.fromtimestamp(BASE - 5, timezone.utc).isoformat()
    assert subject["last_seen_at"] == datetime.fromtimestamp(BASE + 3, timezone.utc).isoformat()
    assert set(aliases(store).values()) == {original}
    store.record_scene_observations(event["id"], batch)
    assert store.scene_incident(event_id=event["id"])["revision"] == current["revision"]
    restarted = EventStore(tmp_path)
    assert restarted.scene_incident(event_id=event["id"]) == current
    assert set(aliases(restarted).values()) == {original}


def test_simultaneous_second_person_stays_distinct(tmp_path):
    store = EventStore(tmp_path)
    raw = [observation(t, key, x) for t in (0, 1, 2) for key, x in (("recorded:R", 10), ("recorded:S", 70))]
    event = add(store, raw)
    store.record_scene_observations(event["id"], [observation(t, "tracking:T") for t in (-5, -4, 0, 3)])
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"]) == 2


def test_contradictory_simultaneous_history_blocks_tracker_switch_fusion(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(t) for t in (0, 1, 2)])
    store.record_scene_observations(event["id"], [observation(-5, "tracking:T", 70), observation(0, "tracking:T", 70)])
    before = aliases(store)
    store.record_scene_observations(event["id"], [observation(1, "tracking:T"), observation(3, "tracking:T")])
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"]) == 2
    assert aliases(store) == before


def test_imprecise_different_provenance_does_not_establish_shared_frame(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(t) for t in (0, 1, 2)])
    store.record_scene_observations(event["id"], [
        observation(t, "tracking:T", source="tracking", frame_timestamp_exact=False)
        for t in (-5, -4, 0, 3)
    ])
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"]) == 2


def test_label_correction_is_not_erased_by_shared_frame(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(t) for t in (0, 1, 2)])
    current = store.scene_incident(event_id=event["id"])
    object_id = current["scene_objects"][0]["id"]
    store.correct_scene_incident(current["id"], current["revision"], {"operation": "label", "object_id": object_id, "label": "dog"})
    store.record_scene_observations(event["id"], [observation(t, "tracking:T") for t in (-5, -4, 0, 3)])
    current = store.scene_incident(event_id=event["id"])
    assert len(current["scene_objects"]) == 2
    assert next(item for item in current["scene_objects"] if item["id"] == object_id)["label"] == "dog"


def test_separated_observations_and_ambiguous_alias_survive_replay(tmp_path):
    store = EventStore(tmp_path)
    raw = [observation(t) for t in (0, 1, 2)]
    event = add(store, raw)
    current = store.scene_incident(event_id=event["id"])
    selected = current["scene_objects"][0]["observations"][1]["id"]
    separated = store.correct_scene_incident(current["id"], current["revision"], {"operation": "separate", "observation_ids": [selected]})
    assert not aliases(store)
    store.record_scene_observations(event["id"], raw)
    assert store.scene_incident(event_id=event["id"]) == separated
    assert not aliases(store)


def test_manual_association_remaps_aliases_and_future_frames(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(0), observation(0, "recorded:S", 70)])
    current = store.scene_incident(event_id=event["id"])
    ids = [item["id"] for item in current["scene_objects"]]
    associated = store.correct_scene_incident(current["id"], current["revision"], {"operation": "associate", "object_ids": ids})
    assert set(aliases(store).values()) == {ids[0]}
    store.record_scene_observations(event["id"], [observation(3, "recorded:S", 70)])
    assert [item["id"] for item in store.scene_incident(event_id=event["id"])["scene_objects"]] == [associated["scene_objects"][0]["id"]]


def test_split_incident_remaps_cross_episode_subject_aliases(tmp_path):
    store = EventStore(tmp_path)
    left = add(store, [observation(0)])
    right = add(store, [observation(0, "recorded:S")], camera="drive")
    a, b = [store.scene_incident(event_id=e["id"]) for e in (left, right)]
    merged = store.correct_scene_incident(a["id"], a["revision"], {"operation": "merge", "incident_ids": [b["id"]], "expected_revisions": {b["id"]: b["revision"]}})
    ids = [item["id"] for item in merged["scene_objects"]]
    associated = store.correct_scene_incident(merged["id"], merged["revision"], {"operation": "associate", "object_ids": ids})
    selected = next(episode["id"] for episode in associated["episodes"] if episode["camera_id"] == "drive")
    store.correct_scene_incident(associated["id"], associated["revision"], {"operation": "split", "episode_ids": [selected]})
    right_scene = store.scene_incident(event_id=right["id"])
    right_object = right_scene["scene_objects"][0]["id"]
    assert aliases(store)[f"{right['id']}:recorded:S"] == right_object
    store.record_scene_observations(right["id"], [observation(3, "recorded:S")])
    assert len(store.scene_incident(event_id=right["id"])["scene_objects"]) == 1
    assert store.scene_incident(event_id=left["id"])["scene_objects"][0]["id"] != right_object


def test_fusion_remaps_appearance_subject_reference(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(t) for t in (0, 1, 2)])
    original = store.scene_incident(event_id=event["id"])["scene_objects"][0]["id"]
    store.record_scene_observations(event["id"], [observation(-5, "tracking:T")])
    current = store.scene_incident(event_id=event["id"])
    duplicate = next(item for item in current["scene_objects"] if item["id"] != original)
    index = AppearanceIndex(store.db_path)
    index.append_event(event["id"], "gate", [{"track_id": 100001, "label": "person", "model_kind": "person",
        "model_fingerprint": "test", "embedding": np.array([1., 0.]), "scene_object_id": duplicate["id"],
        "observation_id": duplicate["observations"][0]["id"]}])
    store.record_scene_observations(event["id"], [observation(0, "tracking:T")])
    current = store.scene_incident(event_id=event["id"])
    assert len(current["scene_objects"]) == 1
    assert index.scene_coverage(event["id"])[0] == {current["scene_objects"][0]["id"]}


def test_simultaneous_overlapping_people_are_not_temporal_matches(tmp_path):
    store = EventStore(tmp_path)
    event = add(store, [observation(0), observation(0, "recorded:S", x=12)])
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"]) == 2


@pytest.mark.parametrize("difference", ["none", "timestamp", "geometry", "event"])
def test_legacy_bridge_requires_same_event_exact_timestamp_and_box(tmp_path, difference):
    store = EventStore(tmp_path)
    legacy = observation(0)
    for key in ("scene_track_key", "frame_source", "frame_timestamp_exact", "recording_path", "captured_at_epoch"):
        legacy.pop(key)
    event = store.add_event("gate", "motion", created_at=datetime.fromtimestamp(BASE, timezone.utc).isoformat(),
                            objects_json=json.dumps([legacy]))
    tracked = observation(.005 if difference == "timestamp" else 0, "tracking:T",
                          x=10.1 if difference == "geometry" else 10, source="tracking")
    if difference == "event":
        add(store, [tracked])
    else:
        store.record_scene_observations(event["id"], [tracked])
    assert len(store.scene_incident(event_id=event["id"])["scene_objects"]) == (1 if difference == "none" else 2)
