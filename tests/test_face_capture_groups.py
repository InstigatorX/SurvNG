from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.face_capture.groups import FaceGroups, GroupNotFound
from survng.app.face_capture.pipeline import CapturedFace, CaptureResult
from survng.app.face_capture.routes import (
    FaceCaptureRouteDependencies,
    MergeRequest,
    NameRequest,
    create_face_capture_router,
)
from survng.app.face_capture.store import FaceCaptureStore
from survng.app.security import required_api_scope

RNG = np.random.default_rng(7)
ALICE, BOB = (RNG.normal(size=512) for _ in range(2))


def _vector(base, noise=0.3):
    vector = base / np.linalg.norm(base) + RNG.normal(size=512) * noise / np.sqrt(512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _face(base, track=0, offset=0.5, norm=17.0):
    return CapturedFace(track, offset, (0, 0, 60, 60), [], 0.9, norm, _vector(base), np.zeros((20, 20, 3), np.uint8))


def _capture(store, groups, event_id, *faces, camera="gate", epoch=None):
    store.record(event_id=event_id, camera_id=camera, event_epoch=epoch or 1000.0 + event_id,
                 result=CaptureResult(faces=list(faces)), embedding_model="m", seconds=1.0)
    groups.assign(event_id)


@pytest.fixture
def setup(tmp_path):
    store = FaceCaptureStore(tmp_path)
    return store, FaceGroups(store)


def test_the_same_person_joins_one_group_and_another_starts_a_new_one(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE), _face(ALICE, offset=1.0))
    _capture(store, groups, 2, _face(ALICE))
    _capture(store, groups, 3, _face(BOB))
    unnamed = groups.groups("unnamed")
    assert unnamed["total"] == 2
    sizes = sorted(group["faces"] for group in unnamed["groups"])
    assert sizes == [1, 3]
    alice = unnamed["groups"][0]
    assert alice["faces"] == 3 and alice["events"] == 2 and alice["cameras"] == ["gate"]
    assert len(alice["samples"]) == 3


def test_naming_two_groups_the_same_puts_them_under_one_person_and_new_faces_follow(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE))
    _capture(store, groups, 2, _face(BOB))
    first, second = (g["id"] for g in groups.groups("unnamed")["groups"])
    groups.name_group(first, "Person A")
    groups.name_group(second, "  person a ")  # same name, different spacing and case
    [person] = groups.people()
    assert person["name"] == "Person A" and person["groups"] == 2 and person["faces"] == 2
    _capture(store, groups, 3, _face(ALICE))
    assert groups.person(person["id"])["visits"][0]["event_id"] == 3  # named by joining its group
    assert groups.groups("unnamed")["total"] == 0


def test_ignore_merge_and_remove_a_face(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE), _face(BOB, track=1))
    alice, bob = sorted(groups.groups("unnamed")["groups"], key=lambda g: g["id"])
    groups.ignore_group(bob["id"], True)
    assert groups.groups("ignored")["total"] == 1 and groups.groups("unnamed")["total"] == 1
    _capture(store, groups, 2, _face(BOB))  # similar junk keeps landing in the ignored group
    assert groups.groups("ignored")["groups"][0]["faces"] == 2
    groups.ignore_group(bob["id"], False)
    merged = groups.merge_groups(bob["id"], alice["id"])
    assert merged["faces"] == 3
    with pytest.raises(GroupNotFound):
        groups.group(bob["id"])
    face_id = merged["faces_list"][0]["id"]
    moved = groups.remove_face(face_id)
    assert groups.group(moved["group_id"])["faces"] == 1
    assert groups.group(alice["id"])["faces"] == 2
    with pytest.raises(ValueError):
        groups.merge_groups(alice["id"], alice["id"])


def test_renaming_to_an_existing_name_joins_the_people_and_forgetting_unnames(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE))
    _capture(store, groups, 2, _face(BOB))
    first, second = (g["id"] for g in groups.groups("unnamed")["groups"])
    a = groups.name_group(first, "A")["person_id"]
    b = groups.name_group(second, "B")["person_id"]
    joined = groups.rename_person(b, "a")
    assert joined["id"] == a and len(joined["groups"]) == 2
    assert [p["name"] for p in groups.people()] == ["A"]
    groups.forget_person(a)
    assert groups.people() == [] and groups.groups("unnamed")["total"] == 2
    with pytest.raises(ValueError):
        groups.name_group(first, "   ")


def test_a_recapture_or_retention_keeps_group_counts_right(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE), _face(ALICE, offset=1.0))
    _capture(store, groups, 2, _face(ALICE))
    [group] = groups.groups("unnamed")["groups"]
    assert group["faces"] == 3
    _capture(store, groups, 1, _face(ALICE))  # event 1 recaptured with one face
    assert groups.group(group["id"])["faces"] == 2
    store.forget_events([1, 2])
    assert groups.groups("unnamed")["total"] == 0  # the emptied group is dropped


def test_routes_name_merge_and_serve_only_store_crops(setup, tmp_path):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE))
    _capture(store, groups, 2, _face(BOB))
    service = SimpleNamespace(store=store, groups=groups, enabled=lambda: True)
    manager = SimpleNamespace(face_capture=service, config=SimpleNamespace(cameras=[SimpleNamespace(id="gate", name="Gate")]))
    router = create_face_capture_router(FaceCaptureRouteDependencies(get_manager=lambda: manager))
    endpoint = {route.path + ":" + ",".join(sorted(route.methods)): route.endpoint for route in router.routes}
    listing = endpoint["/api/face-groups:GET"](status="unnamed", limit=60, offset=0)
    assert listing["total"] == 2 and listing["summary"]["unnamed_groups"] == 2
    first, second = (g["id"] for g in listing["groups"])
    named = endpoint["/api/face-groups/{group_id}/name:POST"](first, NameRequest(name="A"))
    assert named["person_name"] == "A"
    merged = endpoint["/api/face-groups/{group_id}/merge:POST"](second, MergeRequest(into=first))
    assert merged["faces"] == 2
    with pytest.raises(HTTPException) as error:
        endpoint["/api/face-groups/{group_id}:GET"](second)
    assert error.value.status_code == 404
    face_id = merged["faces_list"][0]["id"]
    assert Path(endpoint["/api/face-captures/{face_id}/image:GET"](face_id).path).is_file()
    with store._connect() as connection:  # a row pointing outside the crop folder is never served
        connection.execute("update face_captures set crop_path = ? where id = ?", (str(tmp_path / "survng.sqlite3"), face_id))
    with pytest.raises(HTTPException) as error:
        endpoint["/api/face-captures/{face_id}/image:GET"](face_id)
    assert error.value.status_code == 404
    people = endpoint["/api/face-people:GET"]()
    assert people["people"][0]["name"] == "A"


def test_people_routes_need_the_admin_scope():
    for method, path in (("GET", "/api/face-groups"), ("GET", "/api/face-people"),
                         ("GET", "/api/face-captures/1/image"), ("POST", "/api/face-groups/1/name")):
        assert required_api_scope(method, path) == "admin"


def test_an_unnamed_group_suggests_the_person_it_resembles(setup):
    store, groups = setup
    for event_id, base in ((1, ALICE), (2, ALICE), (3, BOB)):
        store.record(event_id=event_id, camera_id="gate", event_epoch=1000.0 + event_id,
                     result=CaptureResult(faces=[_face(base)]), embedding_model="m", seconds=1.0)
    groups.assign(None, 0.99)  # strict: every event starts its own group
    first, second, bob = sorted(g["id"] for g in groups.groups("unnamed")["groups"])
    groups.name_group(first, "A")
    rows = {g["id"]: g for g in groups.groups("unnamed")["groups"]}
    assert rows[second]["suggestion"]["name"] == "A"
    assert "suggestion" not in rows[bob]  # someone else: no suggestion


def test_a_recapture_keeps_the_name_and_renaming_a_group_drops_the_old_name(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE))
    [group] = groups.groups("unnamed")["groups"]
    groups.name_group(group["id"], "A")
    groups.name_group(group["id"], "B")
    assert [p["name"] for p in groups.people()] == ["B"]  # "A" named nothing any more
    _capture(store, groups, 1, _face(ALICE))  # the group's only event recaptured
    [person] = groups.people()
    assert person["name"] == "B" and person["faces"] == 1
    assert groups.groups("unnamed")["total"] == 0
    store.forget_events([1])
    assert groups.people() == [] and groups.summary()["named_groups"] == 0


def test_merging_a_named_group_into_an_ignored_one_does_not_name_it(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE), _face(BOB, track=1))
    alice, bob = sorted(groups.groups("unnamed")["groups"], key=lambda g: g["id"])
    groups.name_group(alice["id"], "A")
    groups.ignore_group(bob["id"], True)
    merged = groups.merge_groups(alice["id"], bob["id"])
    assert merged["ignored"] and merged["person_id"] is None
    assert groups.people() == []


def test_faces_by_event_list_who_was_seen_in_an_incident(setup):
    store, groups = setup
    _capture(store, groups, 1, _face(ALICE), _face(BOB, track=1))
    _capture(store, groups, 2, _face(ALICE))
    alice = next(g for g in groups.groups("unnamed")["groups"] if g["faces"] == 2)
    groups.name_group(alice["id"], "A")
    found = groups.for_events([1, 2, 99])
    assert set(found["events"]) == {"1", "2"}
    assert len(found["events"]["1"]) == 2 and found["events"]["2"] == [alice["id"]]
    names = {g["id"]: g.get("person_name") for g in found["groups"]}
    assert names[alice["id"]] == "A" and len(names) == 2
    assert all(g["incident_faces"] for g in found["groups"])
    assert groups.for_events([]) == {"events": {}, "groups": []}
    service = SimpleNamespace(store=store, groups=groups, enabled=lambda: True)
    manager = SimpleNamespace(face_capture=service, config=SimpleNamespace(cameras=[]))
    router = create_face_capture_router(FaceCaptureRouteDependencies(get_manager=lambda: manager))
    endpoint = next(r.endpoint for r in router.routes if r.path == "/api/face-captures/by-events")
    assert set(endpoint(ids="1,2")["events"]) == {"1", "2"}
    with pytest.raises(HTTPException) as error:
        endpoint(ids="1,x")
    assert error.value.status_code == 400
    assert required_api_scope("GET", "/api/face-captures/by-events") == "admin"
