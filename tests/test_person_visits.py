import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from survng.app.appearance_index import AppearanceIndex
from survng.app.config import CameraTransitionRoute, ObjectTrackingConfig
from survng.app.events import EventStore
from survng.app.faces import FaceStore
from survng.app.face_routes import FaceRouteDependencies, create_face_router
from survng.app.person_visits import PersonVisitStore, project_visits


def config(**kwargs):
    return ObjectTrackingConfig(camera_transition_routes=[CameraTransitionRoute(from_camera="gate", to_camera="foyer", max_seconds=60)], **kwargs)


def node(key, camera, at, vector=(1., 0.), person=None, **kwargs):
    return dict(id=key, event_id=int(key), track_id=1, camera_id=camera, start=at, end=at+2, vector=np.array(vector)/np.linalg.norm(vector), model="v1", quality=.9, threshold=.7, observation_count=4, person_id=person, person_name=f"Person {person}" if person else "", anchor_status="confirmed" if person else None, **kwargs)


def test_default_suggests_without_assigning_name():
    result = project_visits([node("1", "gate", 0), node("2", "foyer", 10, person=7)], {}, config())
    assert len(result["visits"]) == 2
    assert len(result["suggestions"]) == 1
    assert result["visits"][1]["person_id"] is None


def test_opt_in_links_and_keeps_anchor_distinct_from_inference():
    result = project_visits([node("1", "gate", 0), node("2", "foyer", 10, person=7)], {}, config(visit_auto_link_enabled=True))
    assert len(result["visits"]) == 1
    visit = result["visits"][0]
    assert visit["person_id"] == 7
    assert visit["sightings"][0]["identity_status"] == "linked"
    assert visit["sightings"][0]["person_id"] is None
    assert visit["anchor_ids"] == ["2"]
    assert "vector" not in json.dumps(result)


@pytest.mark.parametrize("change", [dict(model="v2"), dict(quality=.1), dict(observation_count=1), dict(person_id=8)])
def test_weak_incompatible_or_conflicting_evidence_never_links(change):
    left = node("1", "gate", 0, person=7)
    right = node("2", "foyer", 10)
    right.update(change)
    result = project_visits([left, right], {}, config(visit_auto_link_enabled=True))
    assert len(result["visits"]) == 2


def test_competing_lookalikes_prevent_automatic_link():
    result = project_visits([node("1", "gate", 0), node("2", "gate", 0), node("3", "foyer", 10)], {}, config(visit_auto_link_enabled=True))
    assert len(result["visits"]) == 3
    assert all(item["ambiguous"] for item in result["suggestions"])


def test_route_direction_and_overlapping_times_are_not_guessed():
    for sightings in ([node("1", "foyer", 0), node("2", "gate", 10)], [node("1", "gate", 0), node("2", "foyer", 1)]):
        assert project_visits(sightings, {}, config(visit_auto_link_enabled=True))["suggestions"] == []


def test_rejected_pair_cannot_rejoin_through_an_intermediate():
    sightings = [node("1", "gate", 0), node("2", "foyer", 10), node("3", "drive", 20)]
    result = project_visits(sightings, {("1", "2"): "accept", ("2", "3"): "accept", ("1", "3"): "reject"}, config())
    assert len(result["visits"]) == 2
    assert any(link.get("blocked") for link in result["suggestions"])


def test_same_incident_people_cannot_be_merged():
    left, right = node("1", "gate", 0), node("2", "gate", 0)
    right["event_id"] = left["event_id"]
    result = project_visits([left, right], {("1", "2"): "accept"}, config())
    assert len(result["visits"]) == 2


def test_appearance_chain_cannot_drift():
    cfg = config(visit_auto_link_enabled=True)
    cfg.camera_transition_routes.append(CameraTransitionRoute(from_camera="foyer", to_camera="drive", max_seconds=60))
    nodes = [node("1", "gate", 0), node("2", "foyer", 10, (.94, .342)), node("3", "drive", 20, (.766, .643))]
    assert len(project_visits(nodes, {}, cfg)["visits"]) == 2


@pytest.fixture
def stores(tmp_path):
    events = EventStore(tmp_path)
    faces = FaceStore(tmp_path, start_recognition=False)
    index = AppearanceIndex(events.db_path)
    visits = PersonVisitStore(events.db_path)
    yield events, faces, index, visits
    faces.close()


def seed(stores, camera, at, person=None, people_count=1):
    events, faces, index, visits = stores
    when = datetime.fromtimestamp(at, timezone.utc).isoformat()
    event = events.add_event(camera, "motion", created_at=when, objects_json=json.dumps([{"label": "person"}] * people_count))
    event_id = event["id"]
    index.replace_event(event_id, camera, [dict(track_id=1, label="person", model_kind="person", model_fingerprint="v1", embedding=np.array([1., 0.]), match_threshold=.7, quality=.9, observation_count=4, first_seen=when, last_seen=when, created_at=when)])
    if person:
        with faces._connect() as db:
            db.execute("insert or ignore into face_people(id,name,created_at,updated_at) values (?,?,'','')", (person, f"Person {person}"))
            db.execute("""insert into face_observations(event_id,object_index,person_id,camera_id,snapshot_path,box_json,observed_at,created_at,review_status,canonical)
                values (?,0,?,?,'','{}',?,?,'confirmed',1)""", (event_id, person, camera, when, when))
    return event_id


def test_persist_review_restart_retraction_and_stale_revision(stores):
    _, faces, _, visits = stores
    seed(stores, "gate", 10000)
    seed(stores, "foyer", 10010, person=7)
    cfg = config()
    payload = visits.list(9990, 10100, cfg)
    nodes = {n["id"]: n for v in payload["visits"] for n in v["sightings"]}
    edge = payload["suggestions"][0]
    args = dict(left_id=edge["left"], right_id=edge["right"], left_revision=nodes[edge["left"]]["revision"], right_revision=nodes[edge["right"]]["revision"], decision="accept", start=9990, end=10100, config=cfg)
    visits.decide(**args)
    visits = PersonVisitStore(visits.database_path)
    assert len(visits.list(9990, 10100, cfg)["visits"]) == 1
    with faces._connect() as db:
        db.execute("update face_observations set person_id=null,review_status='unknown'")
    assert all(v["person_id"] is None for v in visits.list(9990, 10100, cfg)["visits"])
    with pytest.raises(ValueError, match="Evidence changed"):
        visits.decide(**args)


def test_multipeople_event_does_not_assign_face_to_arbitrary_body(stores):
    seed(stores, "gate", 10000, person=7, people_count=2)
    payload = stores[3].list(9990, 10100, config())
    assert len(payload["visits"]) == 2
    body = next(v for v in payload["visits"] if v["id"].startswith("track:"))
    assert body["person_id"] is None


def test_routes_reject_bad_windows_and_serialize_no_private_vectors(stores):
    seed(stores, "gate", 10000, person=7)
    manager = SimpleNamespace(person_visits=stores[3], config=SimpleNamespace(detector=SimpleNamespace(tracking=config())))
    app = FastAPI()
    app.include_router(create_face_router(FaceRouteDependencies(lambda: manager, threading.RLock(), lambda: None)).router)
    with TestClient(app) as client:
        response = client.get("/api/people/visits?start=9990&end=10100")
        assert response.status_code == 200
        assert "embedding" not in response.text and "snapshot_path" not in response.text
        for query in ("start=nan", "start=1&end=100000", "start=2&end=1"):
            assert client.get(f"/api/people/visits?{query}").status_code == 422


def test_retention_deletes_link_decisions(stores):
    seed(stores, "gate", 10000)
    seed(stores, "foyer", 10010)
    visits = stores[3]
    payload = visits.list(9990, 10100, config())
    nodes = [n for v in payload["visits"] for n in v["sightings"]]
    visits.decide(nodes[0]["id"], nodes[1]["id"], nodes[0]["revision"], nodes[1]["revision"], "reject", 9990, 10100, config())
    with visits._connect() as db:
        db.execute("delete from events where id=?", (nodes[0]["event_id"],))
        assert db.execute("select count(*) from person_visit_decisions").fetchone()[0] == 0


def test_people_without_embeddings_remain_visible(stores):
    stores[0].add_event("gate", "motion", created_at="2026-09-01T00:00:00Z", objects_json='[{"label":"person"},{"label":"person"}]')
    at = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    payload = stores[3].list(at-1, at+60, config())
    assert len(payload["visits"]) == 2
    assert all(item["status"] == "unresolved" for item in payload["visits"])


def test_truncation_suspends_automatic_links(stores):
    seed(stores, "gate", 10000)
    seed(stores, "gate", 10010)
    seed(stores, "foyer", 10020, person=7)
    stores[3].MAX_SIGHTINGS = 2
    result = stores[3].list(9990, 10100, config(visit_auto_link_enabled=True))
    assert result["truncated"]
    assert len(result["visits"]) == 2


def test_confirmed_rejection_survives_identity_rename(stores):
    seed(stores, "gate", 10000)
    seed(stores, "foyer", 10010, person=7)
    visits = stores[3]
    nodes = [n for v in visits.list(9990, 10100, config())["visits"] for n in v["sightings"]]
    visits.decide(nodes[0]["id"], nodes[1]["id"], nodes[0]["revision"], nodes[1]["revision"], "reject", 9990, 10100, config())
    with stores[1]._connect() as db:
        db.execute("update face_people set name='New name' where id=7")
    assert len(visits.list(9990, 10100, config(visit_auto_link_enabled=True))["visits"]) == 2
