from queue import Queue
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from survng.app.faces import FaceStore
from survng.app.incident_queries import IncidentQueryService


@pytest.fixture
def clustered_faces(tmp_path):
    recognizer = SimpleNamespace(
        config=SimpleNamespace(face_unknown_cluster_threshold=0.55),
        status=lambda: {"model_fingerprint": "model-v1"},
    )
    store = FaceStore(tmp_path, recognizer=recognizer, start_recognition=False)
    with store._connect() as connection:
        for event_id in (41, 42):
            connection.execute(
                """
                insert into face_observations (
                    event_id, object_index, camera_id, snapshot_path, box_json,
                    confidence, observed_at, created_at, quality_score,
                    embedding_blob, embedding_model, recognition_pending,
                    recognition_outcome
                ) values (?, 0, 'gate', '', '{}', 0.9, ?, ?, 0.8,
                    ?, 'model-v1', 0, 'embedded')
                """,
                (
                    event_id,
                    f"2026-09-09T10:00:{event_id}+00:00",
                    f"2026-09-09T10:00:{event_id}+00:00",
                    np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
                ),
            )
    store.refresh_unknown_clusters()
    assert [row["observation_count"] for row in store.unknown_clusters()] == [2]
    assert {row["cluster_size"] for row in store.review_queue()} == {2}
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize(
    "state_change",
    [
        "recognition_pending = 1",
        "recognition_outcome = 'failed'",
        "embedding_model = 'old-model'",
        "embedding_blob = null",
        "person_id = (select id from face_people limit 1), review_status = 'confirmed'",
        "canonical = 0",
    ],
)
def test_stale_membership_is_hidden_consistently(clustered_faces, state_change):
    store = clustered_faces
    store.create_person("Alice")
    cluster_id = store.unknown_clusters()[0]["cluster_id"]
    with store._connect() as connection:
        # Model refresh, recognition and review can invalidate a saved membership
        # before the operator next rebuilds the unknown directory.
        connection.execute(
            f"update face_observations set {state_change} where event_id = 41"
        )

    clusters = store.unknown_clusters()
    assert clusters[0]["observation_count"] == 1
    assert [row["event_id"] for row in store.unknown_cluster_members(cluster_id)] == [42]
    event_rows = {row["event_id"]: row for row in store.for_event_ids([41, 42])}
    assert event_rows[42]["unknown_cluster_id"] == cluster_id
    if 41 in event_rows:
        assert event_rows[41]["unknown_cluster_id"] is None
    for item in store.review_queue():
        assert item["cluster_size"] == 1
        assert "recurring_unknown" not in item["review_reasons"]
        if item["event_id"] == 41:
            assert item["unknown_cluster_id"] is None

    incidents = [{"events": [{"id": 41}, {"id": 42}]}]
    IncidentQueryService.with_faces(SimpleNamespace(faces=store), incidents)
    first_event, second_event = incidents[0]["events"]
    assert all(face["unknown_cluster_id"] is None for face in first_event["faces"])
    assert second_event["faces"][0]["name"] == f"Unknown Person {cluster_id}"


def test_policy_change_does_not_reuse_old_cluster_or_review_size(clustered_faces):
    store = clustered_faces
    store.recognizer.config.face_unknown_cluster_threshold = 0.8

    assert store.unknown_clusters() == []
    assert all(row["unknown_cluster_id"] is None for row in store.for_event_ids([41, 42]))
    queue = store.review_queue()
    assert len(queue) == 2
    assert all(item["cluster_size"] == 1 for item in queue)


def test_close_joins_worker_even_when_stop_sentinel_queue_is_full(tmp_path):
    store = FaceStore(tmp_path, start_recognition=False)
    store._recognition_queue = Queue(maxsize=1)
    store._recognition_queue.put_nowait(1)
    worker = Thread(target=store._recognition_stop.wait)
    store._recognition_thread = worker
    worker.start()

    store.close()

    assert store._recognition_stop.is_set()
    assert not worker.is_alive()
    assert store._recognition_thread is None
