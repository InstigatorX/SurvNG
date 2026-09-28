import json
from datetime import datetime, timezone

from survng.app.event_store import EventStore


def observed(x=10, **extra):
    return {"label":"person", "confidence":.75,
            "box":{"x1":x,"y1":10,"x2":x+10,"y2":50},
            "captured_at_epoch":1000, "frame_source":"recorded_main", **extra}


def event(store, objects):
    return store.add_event(camera_id="gate",kind="motion",objects_json=json.dumps(objects),
                           created_at=datetime.fromtimestamp(1000,timezone.utc).isoformat())


def test_unconfirmed_discovery_cannot_alert_from_raw_zone_eligibility(tmp_path):
    store=EventStore(tmp_path)
    raw=observed(incident_eligible=True,temporal_observations=1)
    row=event(store,[{**raw,"incident_eligible":False,"alert_eligible":False,
                      "alert_reasons":["temporal_unconfirmed"]},
                     {"status":"scene_observations","observations":[raw]}])
    result=store.scene_incident(event_id=row["id"])
    assert len(result["scene_objects"])==1
    assert result["alert_decisions"][0]["eligible"] is False
    assert all(not item["eligible"] for item in result["alert_decisions"][0]["objects"])


def test_two_people_keep_separate_inventory_and_alert_policy(tmp_path):
    store=EventStore(tmp_path)
    inside=observed(incident_eligible=True,alert_eligible=True)
    outside=observed(60,incident_eligible=False,alert_eligible=False,
                     alert_reasons=["outside_incident_zone"])
    row=event(store,[inside,outside,{"status":"scene_observations","observations":[inside,outside]}])
    result=store.scene_incident(event_id=row["id"])
    assert len(result["scene_objects"])==2
    decisions=result["alert_decisions"][0]["objects"]
    assert len({item["object_id"] for item in decisions})==2
    assert [item["eligible"] for item in decisions]==[True,False]


def test_later_decision_for_same_observation_preserves_original_evidence(tmp_path):
    store=EventStore(tmp_path)
    raw=observed(scene_track_key="track-one")
    row=event(store,[{"status":"scene_observations","observations":[raw]}])
    with store._connect() as conn:
        original=conn.execute("select payload_json from scene_observations").fetchone()[0]
    store.record_scene_observations(row["id"],[{**raw,"alert_eligible":True,"alert_reasons":[]}])
    result=store.scene_incident(event_id=row["id"])
    assert result["alert_decisions"][0]["eligible"] is True
    with store._connect() as conn:
        assert conn.execute("select payload_json from scene_observations").fetchone()[0]==original
        assert conn.execute("select count(*) from scene_observations").fetchone()[0]==1
    # A later off-zone sighting does not erase the supported alert earlier in
    # this incident, and replaying the original frame cannot erase that decision.
    store.record_scene_observations(row["id"],[{**raw,"captured_at_epoch":1001,"alert_eligible":False,
                                               "alert_reasons":["outside_incident_zone"]}])
    store.record_scene_observations(row["id"],[raw])
    decisions=store.scene_incident(event_id=row["id"])["alert_decisions"][0]
    assert decisions["eligible"] is True
    assert {item["eligible"] for item in decisions["objects"]}=={True,False}


def test_unknown_legacy_policy_is_not_an_alert(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,[observed()])
    assert store.scene_incident(event_id=row["id"])["alert_decisions"][0]["eligible"] is False


def test_raw_acquisition_zone_acceptance_is_not_an_alert_decision(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,[{"status":"scene_observations","observations":[observed(incident_eligible=True)]}])
    scene=store.scene_incident(event_id=row["id"])
    assert len(scene["scene_objects"])==1
    assert scene["alert_decisions"][0]["eligible"] is False


def test_off_cover_confirmed_decision_uses_its_own_observation(tmp_path):
    store=EventStore(tmp_path)
    raw=observed(captured_at_epoch=1002)
    row=event(store,[{**raw,"snapshot_visible":False,"alert_eligible":True},
                     {"status":"scene_observations","observations":[raw]}])
    scene=store.scene_incident(event_id=row["id"])
    assert scene["alert_decisions"][0]["eligible"] is True
    assert scene["scene_objects"][0]["observations"][0]["snapshot_available"] is False


def test_alert_reference_follows_operator_association(tmp_path):
    store=EventStore(tmp_path)
    row=event(store,[observed(alert_eligible=True),observed(60,alert_eligible=False)])
    scene=store.scene_incident(event_id=row["id"])
    joined=store.correct_scene_incident(scene["id"],scene["revision"],
                                       {"operation":"associate","object_ids":[o["id"] for o in scene["scene_objects"]]})
    assert len(joined["scene_objects"])==1
    subject=joined["scene_objects"][0]["id"]
    assert {o["object_id"] for o in joined["alert_decisions"][0]["objects"]}=={subject}
