from pathlib import Path

from survng.app.faces import FaceStore


def _store(tmp_path: Path) -> FaceStore:
    return FaceStore(tmp_path, start_recognition=False)


def _insert(store: FaceStore, *, event_id: int, person_id=None, camera="gate"):
    with store._connect() as connection:
        now = "2026-08-18T20:00:00+00:00"
        cursor = connection.execute(
            """
            insert into face_observations (
                event_id, object_index, person_id, camera_id, snapshot_path,
                box_json, confidence, observed_at, match_confidence,
                review_status, created_at, recognition_pending,
                recognition_outcome, canonical
            ) values (?, 0, ?, ?, '', '{"x1":1,"y1":1,"x2":100,"y2":100}',
                0.9, ?, ?, ?, ?, 0, 'embedded', 1)
            """,
            (
                event_id,
                person_id,
                camera,
                now,
                1.0 if person_id is not None else None,
                "confirmed" if person_id is not None else "unknown",
                now,
            ),
        )
        return int(cursor.lastrowid)


def test_person_history_and_bulk_unassign(tmp_path):
    store = _store(tmp_path)
    try:
        person = store.create_person("Steve")
        observation_id = _insert(
            store,
            event_id=101,
            person_id=int(person["id"]),
            camera="gate",
        )
        history = store.person_history(int(person["id"]))
        assert history["summary"]["observations"] == 1
        assert history["cameras"][0]["camera_id"] == "gate"

        result = store.bulk_review(
            [observation_id],
            action="unassign",
        )
        assert result["changed"] == 1
        assert store.observation(observation_id)["person_id"] is None
    finally:
        store.close()


def test_manual_bulk_assignment_emits_identity_event(tmp_path):
    store = _store(tmp_path)
    emitted = []
    store.set_identity_event_publisher(emitted.append)
    try:
        person = store.create_person("Steve")
        observation_id = _insert(store, event_id=102)
        store.bulk_review(
            [observation_id],
            action="assign",
            person_id=int(person["id"]),
        )
        assert emitted
        assert emitted[0]["name"] == "Steve"
        assert emitted[0]["event_id"] == 102
        assert emitted[0]["source"] == "manual_bulk"
    finally:
        store.close()
