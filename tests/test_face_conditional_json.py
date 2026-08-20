import threading
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from survng.app.face_routes import FaceRouteDependencies, create_face_router
from survng.app.faces import FaceStore


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
    faces.people_directory_revision.return_value = "people:1"
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
    # One materialization served the HTTP 200 and one served the legacy direct
    # handler; the conditional hit did not query the directory again.
    assert faces.people.call_count == 2


def test_people_mutation_changes_the_directory_validator() -> None:
    people = [{"id": 1, "name": "Ada"}]
    revision = {"value": 1}
    faces = Mock()
    faces.people.side_effect = lambda: deepcopy(people)
    faces.people_directory_revision.side_effect = lambda: f"people:{revision['value']}"

    def create_person(name, _observation_id, notes):
        created = {"id": 2, "name": name, "notes": notes}
        people.append(created)
        revision["value"] += 1
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
    revision = {"value": 1}
    faces = Mock()
    faces.unknown_clusters.side_effect = lambda: deepcopy(clusters)
    faces.unknown_clusters_revision.side_effect = lambda: f"clusters:{revision['value']}"
    faces.unknown_cluster_health.return_value = {"cluster_count": 2}

    def rebuild():
        clusters.append({"cluster_id": 2, "observation_count": 3})
        revision["value"] += 1

    faces.refresh_unknown_clusters.side_effect = rebuild
    client, bundle = face_client(faces)
    first = client.get("/api/faces/unknown-clusters")
    unchanged = client.get(
        "/api/faces/unknown-clusters",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert unchanged.status_code == 304
    assert bundle.handlers["face_unknown_clusters"]() == clusters
    assert faces.unknown_clusters.call_count == 2

    rebuilt = client.post("/api/faces/unknown-clusters/rebuild")
    changed = client.get(
        "/api/faces/unknown-clusters",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert rebuilt.status_code == 200
    assert changed.status_code == 200
    assert changed.json() == clusters
    assert changed.headers["etag"] != first.headers["etag"]


def test_face_store_revision_tokens_follow_people_observations_and_rebuilds(tmp_path) -> None:
    recognizer = SimpleNamespace(
        enabled=False,
        config=SimpleNamespace(face_unknown_cluster_threshold=0.55),
        status=lambda: {"model_fingerprint": "model-v1"},
    )
    store = FaceStore(tmp_path, recognizer=recognizer, start_recognition=False)
    initial_people = store.people_directory_revision()

    store.create_person("Ada")

    assert store.people_directory_revision() != initial_people
    before_observation_people = store.people_directory_revision()
    before_observation_clusters = store.unknown_clusters_revision()
    now = datetime.now(timezone.utc).isoformat()
    with store._lock, store._connect() as connection:
        connection.execute(
            """
            insert into face_observations (
                event_id, object_index, camera_id, snapshot_path, box_json,
                confidence, observed_at, created_at, recognition_pending,
                recognition_outcome, embedding_model, embedding_blob, quality_score
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1, 0, "gate", "snapshot.jpg", "{}", 0.9, now, now,
                0, "embedded", "model-v1",
                np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
                0.9,
            ),
        )
        connection.execute(
            """
            insert into face_observations (
                event_id, object_index, camera_id, snapshot_path, box_json,
                confidence, observed_at, created_at, recognition_pending,
                recognition_outcome, embedding_model, embedding_blob, quality_score
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2, 0, "driveway", "snapshot-2.jpg", "{}", 0.9, now, now,
                0, "embedded", "model-v1",
                np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
                0.9,
            ),
        )

    assert store.people_directory_revision() != before_observation_people
    assert store.unknown_clusters_revision() != before_observation_clusters
    before_rebuild = store.unknown_clusters_revision()

    store.refresh_unknown_clusters()

    assert store.unknown_clusters_revision() != before_rebuild
