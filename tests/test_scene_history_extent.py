"""Scene extent and certainty survive independent acquisition and policy changes."""
import json
from datetime import datetime, timezone

from survng.app.events import EventStore


def sample(at, x=10, **extra):
    return {"label":"person", "confidence":.75, "captured_at_epoch":at,
            "scene_track_key":"track-one", "box":{"x1":x,"y1":10,"x2":x+10,"y2":40}, **extra}


def event(store, at=1000, camera="gate", observations=()):
    return store.add_event(camera,"motion",created_at=datetime.fromtimestamp(at,timezone.utc).isoformat(),
                           objects_json=json.dumps([{"status":"scene_observations","observations":list(observations)}]))


def test_preroll_updates_stable_episode_extent_and_activity(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,observations=[sample(1000)])
    before=store.scene_incident(event_id=row["id"])
    store.record_scene_observations(row["id"],[sample(990),sample(1005,30,zones=["walkway"])])
    scene=store.scene_incident(event_id=row["id"])
    assert scene["id"]==before["id"]
    assert scene["start_epoch"]==990
    assert scene["episodes"][0]["event_ids"]==[row["id"]]
    assert {item["kind"] for item in scene["activity"]}=={"appeared","last_seen"}
    assert {item["object_id"] for item in scene["activity"]}=={scene["scene_objects"][0]["id"]}


def test_scene_certainty_uses_independent_confirmation_threshold(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,observations=[sample(1000,scene_confirmation_threshold=.8),sample(1001,scene_confirmation_threshold=.8)])
    scene=store.scene_incident(event_id=row["id"])
    assert scene["scene_objects"][0]["certainty"]=="possible"
    store.record_scene_observations(row["id"],[sample(1002,confidence=.85,scene_confirmation_threshold=.8),sample(1003,confidence=.85,scene_confirmation_threshold=.8)])
    assert store.scene_incident(event_id=row["id"])["scene_objects"][0]["certainty"]=="observed"


def test_simultaneous_identity_sightings_do_not_auto_connect(tmp_path):
    store=EventStore(tmp_path)
    first=event(store,observations=[sample(1000,temporal_newly_appeared=True)])
    second=event(store,camera="drive",observations=[sample(1000,temporal_newly_appeared=True)])
    identity=[{"identity_id":3,"status":"confirmed"}]
    store.update_scene_identities(first["id"],identity)
    store.update_scene_identities(second["id"],identity)
    assert store.scene_incident(event_id=first["id"])["id"]!=store.scene_incident(event_id=second["id"])["id"]


def test_refined_movement_can_connect_already_known_identity(tmp_path):
    store=EventStore(tmp_path)
    first=event(store,observations=[sample(1000),sample(1001)])
    second=event(store,at=1020,camera="drive",observations=[sample(1020),sample(1021)])
    identity=[{"identity_id":3,"status":"confirmed"}]
    store.update_scene_identities(first["id"],identity)
    store.update_scene_identities(second["id"],identity)
    assert store.scene_incident(event_id=first["id"])["id"]!=store.scene_incident(event_id=second["id"])["id"]
    from tests.scene_activity_helpers import record_person_motion
    record_person_motion(store,first['id'])
    record_person_motion(store,second['id'])
    store.update_scene_identities(second["id"],identity)
    assert store.scene_incident(event_id=first["id"])["id"]==store.scene_incident(event_id=second["id"])["id"]


def test_expired_supporting_snapshot_cannot_return_through_replayed_batch(tmp_path):
    store=EventStore(tmp_path)
    image=tmp_path/"snapshots"/"source.webp"
    image.parent.mkdir()
    image.write_bytes(b"evidence")
    observation=sample(1000,snapshot_path=str(image))
    row=event(store,observations=[observation])
    before=store.scene_incident(event_id=row["id"])
    assert before["scene_objects"][0]["observations"][0]["snapshot_available"]
    store.apply_snapshot_retention(2000,100)
    store.record_scene_observations(row["id"],[observation])
    after=store.scene_incident(event_id=row["id"])
    assert after["revision"]>before["revision"]
    assert not after["scene_objects"][0]["observations"][0]["snapshot_available"]
    assert not image.exists()


def test_box_motion_alone_cannot_extend_activity_boundaries(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,observations=[sample(1000)])
    store.record_scene_observations(row["id"],[sample(1000+i,10+i*.1) for i in range(1,61)])
    scene=store.scene_incident(event_id=row["id"])
    last=datetime.fromisoformat(scene["episodes"][0]["last_activity_at"]).timestamp()
    assert last==1000


def test_disconnected_historical_sightings_do_not_assert_unique_people(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,observations=[sample(1000,scene_track_key="first",temporal_observations=3)])
    store.record_scene_observations(row["id"],[sample(1020,scene_track_key="second",temporal_observations=3)])
    scene=store.scene_incident(event_id=row["id"])
    assert len(scene["scene_objects"])==2
    assert scene["continuity_uncertain"]
    assert scene["summary"] == "Person observed"
    assert "2 people" not in scene["summary"]
    assert all(s["continuity_uncertain"] for s in scene["scene_objects"])


def test_simultaneously_visible_people_can_be_counted(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,observations=[sample(1000,scene_track_key="first",temporal_observations=3),sample(1000,40,scene_track_key="second",temporal_observations=3)])
    scene=store.scene_incident(event_id=row["id"])
    assert scene["summary"]=="Person observed"
    assert not scene["continuity_uncertain"]


def test_historical_identity_enrichment_cannot_send_new_alert(tmp_path):
    from survng.app.manager import AppManager
    from survng.app.config import AppConfig
    store=EventStore(tmp_path)
    row=store.add_event("gate","motion",created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat(),
                        objects_json=json.dumps([sample(1000,incident_eligible=True)]))
    with store._connect() as conn:
        conn.execute("delete from scene_notification_outbox")
        conn.execute("update scene_incidents set historical=1")
    store.update_scene_identities(row["id"],[{"identity_id":3,"status":"confirmed"}])
    payload=store.scene_pending_notifications()[0]["payload"]
    manager=object.__new__(AppManager)
    manager.config=AppConfig(integration_notifications={"enabled":True})
    assert payload["historical"]
    assert not manager.incident_notification_allowed(payload)
