import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from survng.app.face_routes import FaceRouteDependencies, create_face_router


def face_client(faces):
    app = FastAPI()
    bundle = create_face_router(FaceRouteDependencies(
        get_manager=lambda: SimpleNamespace(faces=faces),
        manager_lock=threading.RLock(),
        start_observation_sync=Mock(),
    ))
    app.include_router(bundle.router)
    return TestClient(app), bundle


def test_people_directory_revalidates_and_preserves_direct_handler() -> None:
    people = [{"id": 1, "name": "Ada"}]
    faces = Mock()
    faces.people.side_effect = lambda: deepcopy(people)
    client, bundle = face_client(faces)

    first = client.get("/api/faces/people")

    assert first.status_code == 200
    assert first.json() == people
    assert first.headers["cache-control"] == "private, no-cache"
    assert first.headers["etag"].startswith('"')
    assert bundle.handlers["face_people"]() == people

    unchanged = client.get(
        "/api/faces/people",
        headers={"If-None-Match": f'W/{first.headers["etag"]}, "other"'},
    )

    assert unchanged.status_code == 304
    assert unchanged.content == b""
    assert unchanged.headers["etag"] == first.headers["etag"]
    assert unchanged.headers["cache-control"] == "private, no-cache"


def test_people_mutation_changes_the_directory_validator() -> None:
    people = [{"id": 1, "name": "Ada"}]
    faces = Mock()
    faces.people.side_effect = lambda: deepcopy(people)

    def create_person(name, _observation_id, notes):
        created = {"id": 2, "name": name, "notes": notes}
        people.append(created)
        return created

    faces.create_person.side_effect = create_person
    client, _bundle = face_client(faces)
    first = client.get("/api/faces/people")

    created = client.post("/api/faces/people", json={"name": "Grace"})
    changed = client.get(
        "/api/faces/people",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert created.status_code == 200
    assert changed.status_code == 200
    assert changed.json() == people
    assert changed.headers["etag"] != first.headers["etag"]


def test_unknown_cluster_rebuild_changes_the_cluster_validator() -> None:
    clusters = [{"cluster_id": 1, "observation_count": 2}]
    faces = Mock()
    faces.unknown_clusters.side_effect = lambda: deepcopy(clusters)
    faces.unknown_cluster_health.return_value = {"cluster_count": 2}

    def rebuild():
        clusters.append({"cluster_id": 2, "observation_count": 3})

    faces.refresh_unknown_clusters.side_effect = rebuild
    client, bundle = face_client(faces)
    first = client.get("/api/faces/unknown-clusters")
    unchanged = client.get(
        "/api/faces/unknown-clusters",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert unchanged.status_code == 304
    assert bundle.handlers["face_unknown_clusters"]() == clusters

    rebuilt = client.post("/api/faces/unknown-clusters/rebuild")
    changed = client.get(
        "/api/faces/unknown-clusters",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert rebuilt.status_code == 200
    assert changed.status_code == 200
    assert changed.json() == clusters
    assert changed.headers["etag"] != first.headers["etag"]
