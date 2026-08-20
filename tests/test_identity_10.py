from pathlib import Path
import json

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
        assert emitted[0]["identity_status"] == "confirmed"
    finally:
        store.close()


def test_bulk_reassignment_clears_reference_and_stale_match_state(tmp_path):
    store = _store(tmp_path)
    try:
        alice = store.create_person("Alice")
        bob = store.create_person("Bob")
        observation_id = _insert(
            store,
            event_id=103,
            person_id=int(alice["id"]),
        )
        emitted = []
        store.set_identity_event_publisher(emitted.append)
        with store._connect() as connection:
            connection.execute(
                """update face_observations
                set reference_pinned = 1, reference_auto_pinned = 1,
                    match_details_json = ?
                where id = ?""",
                (json.dumps({"person_id": int(alice["id"]), "score": 0.91}), observation_id),
            )

        store.bulk_review(
            [observation_id],
            action="assign",
            person_id=int(bob["id"]),
        )

        observation = store.observation(observation_id)
        assert observation["person_id"] == int(bob["id"])
        assert observation["reference_pinned"] is False
        assert observation["reference_auto_pinned"] is False
        assert observation["match_details"] == {}
        with store._connect() as connection:
            rejected = connection.execute(
                """select 1 from face_rejections
                where observation_id = ? and person_id = ?""",
                (observation_id, int(alice["id"])),
            ).fetchone()
        assert rejected is not None
        assert emitted[0]["action"] == "corrected"
        assert emitted[0]["previous_person_id"] == int(alice["id"])
        assert emitted[0]["current_person_id"] == int(bob["id"])
    finally:
        store.close()


def test_bulk_unassign_rejects_previous_automatic_identity(tmp_path):
    store = _store(tmp_path)
    try:
        person = store.create_person("Alice")
        observation_id = _insert(
            store,
            event_id=104,
            person_id=int(person["id"]),
        )
        emitted = []
        store.set_identity_event_publisher(emitted.append)
        with store._connect() as connection:
            connection.execute(
                """update face_observations
                set review_status = 'auto_identified', auto_identified = 1
                where id = ?""",
                (observation_id,),
            )
        store._emit_identity_update(observation_id, source="auto_recognition")
        assert emitted[0]["identity_status"] == "automatic"

        store.bulk_review([observation_id], action="unassign")

        with store._connect() as connection:
            rejected = connection.execute(
                """select 1 from face_rejections
                where observation_id = ? and person_id = ?""",
                (observation_id, int(person["id"])),
            ).fetchone()
        assert rejected is not None
        observation = store.observation(observation_id)
        assert observation["review_status"] == "rejected"
        assert observation["rejected_person_id"] == int(person["id"])
        assert emitted[1]["action"] == "cleared"
        assert emitted[1]["active"] is False
        assert emitted[1]["previous_person_id"] == int(person["id"])
    finally:
        store.close()
