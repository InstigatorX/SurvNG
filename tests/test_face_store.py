from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.faces import FaceStore
from survng.app.inference import InferenceUnavailable


class FaceStoreTest(unittest.TestCase):
    def test_database_can_be_local_while_snapshots_remain_in_media_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as database:
            store = FaceStore(
                Path(storage),
                start_recognition=False,
                database_dir=Path(database),
            )

            self.assertEqual(store.storage_dir, Path(storage))
            self.assertEqual(store.db_path, Path(database) / "survng.sqlite3")

    @staticmethod
    def insert_observation(
        store: FaceStore,
        event_id: int,
        *,
        observed_at: str,
        person_id: int | None = None,
        candidate_person_id: int | None = None,
        embedding: np.ndarray | None = None,
        embedding_model: str = "",
        recognition_pending: int = 1,
    ) -> int:
        with store._connect() as connection:
            cursor = connection.execute(
                """
                insert into face_observations (
                    event_id, object_index, person_id, camera_id, snapshot_path,
                    box_json, confidence, observed_at, match_confidence,
                    review_status, created_at, candidate_person_id,
                    embedding_blob, embedding_model, recognition_pending
                ) values (?, 0, ?, 'gate', ?, ?, 0.8, ?, null, 'unknown', ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    person_id,
                    str(store.storage_dir / "snapshots" / f"{event_id}.jpg"),
                    json.dumps({"x1": 1, "y1": 2, "x2": 20, "y2": 24}),
                    observed_at,
                    observed_at,
                    candidate_person_id,
                    embedding.astype(np.float32).tobytes() if embedding is not None else None,
                    embedding_model,
                    recognition_pending,
                ),
            )
            return int(cursor.lastrowid)

    def test_legacy_database_migrates_pending_work_and_rejection_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "survng.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    create table face_people (
                        id integer primary key autoincrement,
                        name text not null,
                        notes text not null default '',
                        created_at text not null,
                        updated_at text not null
                    );
                    create table face_observations (
                        id integer primary key autoincrement,
                        event_id integer not null,
                        object_index integer not null,
                        person_id integer,
                        camera_id text not null,
                        snapshot_path text not null,
                        box_json text not null,
                        confidence real not null default 0,
                        observed_at text not null,
                        match_confidence real,
                        review_status text not null default 'unknown',
                        created_at text not null,
                        embedding_blob blob,
                        embedding_model text not null default '',
                        candidate_person_id integer,
                        candidate_confidence real,
                        rejected_person_id integer,
                        recognition_error text not null default '',
                        recognized_at text not null default '',
                        unique(event_id, object_index)
                    );
                    insert into face_people (id, name, created_at, updated_at)
                    values (1, 'Alice', '2026-07-26T12:00:00+00:00', '2026-07-26T12:00:00+00:00');
                    insert into face_observations (
                        event_id, object_index, camera_id, snapshot_path, box_json,
                        observed_at, created_at, rejected_person_id
                    ) values (
                        1, 0, 'gate', 'missing.jpg',
                        '{"x1":1,"y1":2,"x2":20,"y2":24}',
                        '2026-07-26T12:00:00+00:00', '2026-07-26T12:00:00+00:00', 1
                    );
                    """
                )

            store = FaceStore(Path(tmpdir), start_recognition=False)

            with store._connect() as connection:
                columns = {
                    str(row[1])
                    for row in connection.execute("pragma table_info(face_observations)")
                }
                observation = connection.execute(
                    "select recognition_pending from face_observations where id = 1"
                ).fetchone()
                rejection = connection.execute(
                    "select person_id from face_rejections where observation_id = 1"
                ).fetchone()
            self.assertIn("recognition_pending", columns)
            self.assertEqual(observation["recognition_pending"], 1)
            self.assertEqual(rejection["person_id"], 1)

    def test_recognition_is_queued_only_after_observation_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            snapshot = Path(tmpdir) / "snapshots" / "face.jpg"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"image")
            committed_ids: list[int] = []

            def assert_committed(observation_id: int) -> None:
                with sqlite3.connect(store.db_path) as connection:
                    row = connection.execute(
                        "select id from face_observations where id = ?",
                        (observation_id,),
                    ).fetchone()
                self.assertIsNotNone(row)
                committed_ids.append(observation_id)

            store._queue_recognition = assert_committed  # type: ignore[method-assign]
            inserted = store.ingest_events([
                {
                    "id": 42,
                    "camera_id": "gate",
                    "snapshot_path": str(snapshot),
                    "created_at": "2026-07-26T12:00:00+00:00",
                    "objects_json": json.dumps([
                        {
                            "label": "face",
                            "confidence": 0.8,
                            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                        }
                    ]),
                }
            ])

            self.assertEqual(inserted, 1)
            self.assertEqual(len(committed_ids), 1)

    def test_bulk_ingestion_queues_only_retained_recent_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), max_observations=100, start_recognition=False)
            snapshot = Path(tmpdir) / "snapshots" / "face.jpg"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"image")
            queued: list[int] = []
            store._queue_recognition = queued.append  # type: ignore[method-assign]
            events = [
                {
                    "id": event_id,
                    "camera_id": "gate",
                    "snapshot_path": str(snapshot),
                    "created_at": f"2026-07-26T12:{event_id // 60:02d}:{event_id % 60:02d}+00:00",
                    "objects_json": json.dumps([{
                        "label": "face",
                        "confidence": 0.8,
                        "box": {"x1": 1, "y1": 2, "x2": 20, "y2": 24},
                    }]),
                }
                for event_id in range(1, 106)
            ]

            self.assertEqual(store.ingest_events(events), 105)

            with store._connect() as connection:
                retained = {int(row[0]) for row in connection.execute("select id from face_observations")}
            self.assertEqual(len(retained), 100)
            self.assertEqual(set(queued), retained)

    def test_malformed_event_objects_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            snapshot = Path(tmpdir) / "snapshots" / "face.jpg"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"image")
            inserted = store.ingest_events([
                {
                    "id": 1,
                    "snapshot_path": "snapshot.jpg",
                    "objects_json": json.dumps(["not-an-object", None]),
                },
                {
                    "id": 2,
                    "snapshot_path": "snapshot.jpg",
                    "objects_json": json.dumps({"label": "face"}),
                },
                {
                    "id": 3,
                    "snapshot_path": "snapshot.jpg",
                    "objects_json": json.dumps([
                        {
                            "label": "face",
                            "confidence": "invalid",
                            "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                        }
                    ]),
                },
                {
                    "id": "invalid",
                    "snapshot_path": str(snapshot),
                    "objects_json": json.dumps([{
                        "label": "face",
                        "confidence": 0.9,
                        "box": {"x1": 1, "y1": 2, "x2": 20, "y2": 24},
                    }]),
                },
            ])

            self.assertEqual(inserted, 0)

    def test_ingestion_rejects_outside_snapshots_and_non_finite_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            outside_snapshot = Path(outside) / "face.jpg"
            outside_snapshot.write_bytes(b"image")
            inside_snapshot = Path(tmpdir) / "snapshots" / "face.jpg"
            inside_snapshot.parent.mkdir()
            inside_snapshot.write_bytes(b"image")
            events = [
                {
                    "id": 1,
                    "snapshot_path": str(outside_snapshot),
                    "objects_json": json.dumps([{
                        "label": "face",
                        "confidence": 0.9,
                        "box": {"x1": 1, "y1": 2, "x2": 20, "y2": 24},
                    }]),
                },
                {
                    "id": 2,
                    "snapshot_path": str(inside_snapshot),
                    "objects_json": json.dumps([{
                        "label": "face",
                        "confidence": 0.9,
                        "box": {"x1": 1, "y1": 2, "x2": float("nan"), "y2": 24},
                    }]),
                },
                {
                    "id": 3,
                    "snapshot_path": str(inside_snapshot),
                    "objects_json": json.dumps([{
                        "label": "face",
                        "confidence": float("inf"),
                        "box": {"x1": 1, "y1": 2, "x2": 20, "y2": 24},
                    }]),
                },
            ]

            self.assertEqual(store.ingest_events(events), 0)
            self.assertEqual(store.stats()["observations"], 0)

    def test_recognition_queue_deduplicates_and_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), recognizer=Mock(), start_recognition=False)

            for _ in range(20):
                store._queue_recognition(7)
            for observation_id in range(8, 200):
                store._queue_recognition(observation_id)

            self.assertEqual(list(store._recognition_queue.queue).count(7), 1)
            self.assertLessEqual(store._recognition_queue.qsize(), store.max_observations + 1)
            self.assertEqual(len(store._recognition_pending), store._recognition_queue.qsize())

    def test_concurrent_start_creates_only_one_recognition_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"model_fingerprint": "model-v1"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            callers = [threading.Thread(target=store.start) for _ in range(12)]
            for caller in callers:
                caller.start()
            for caller in callers:
                caller.join()
            self.addCleanup(store.close)

            recognition_threads = [
                thread
                for thread in threading.enumerate()
                if thread.name == "survng-face-recognition" and thread.is_alive()
            ]
            self.assertEqual(recognition_threads, [store._recognition_thread])

    def test_startup_queues_only_dirty_or_stale_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"model_fingerprint": "model-v2"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            current_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v2",
                recognition_pending=0,
            )
            stale_id = self.insert_observation(
                store,
                2,
                observed_at="2026-07-26T12:00:01+00:00",
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=0,
            )
            dirty_id = self.insert_observation(
                store,
                3,
                observed_at="2026-07-26T12:00:02+00:00",
                recognition_pending=1,
            )
            failed_id = self.insert_observation(
                store,
                4,
                observed_at="2026-07-26T12:00:03+00:00",
                recognition_pending=0,
            )

            store._queue_pending_recognition()

            queued = set(store._recognition_queue.queue)
            self.assertEqual(queued, {stale_id, dirty_id})
            self.assertNotIn(current_id, queued)
            self.assertNotIn(failed_id, queued)

    def test_successful_recognition_clears_durable_pending_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(
                    face_min_size=4,
                    face_max_references=5,
                    face_match_threshold=0.4,
                ),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "model-v1",
                },
                embed=lambda _face: np.asarray([1.0, 0.0], dtype=np.float32),
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )
            snapshot = store.storage_dir / "snapshots" / "1.jpg"
            snapshot.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((32, 32, 3), dtype=np.uint8)))

            store._recognize_observation(observation_id)

            with store._connect() as connection:
                row = connection.execute(
                    "select embedding_model, recognition_pending, recognition_error from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertEqual(row["embedding_model"], "model-v1")
            self.assertEqual(row["recognition_pending"], 0)
            self.assertEqual(row["recognition_error"], "")

    def test_permanent_recognition_error_is_not_retried_on_every_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(face_min_size=4),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "model-v1",
                },
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )

            store._recognize_observation(observation_id)
            store._queue_pending_recognition()

            with store._connect() as connection:
                row = connection.execute(
                    "select recognition_pending, recognition_error from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertEqual(row["recognition_pending"], 0)
            self.assertTrue(row["recognition_error"])
            self.assertEqual(store._recognition_queue.qsize(), 0)

    def test_invalid_embedding_is_rejected_without_poisoning_matching_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(face_min_size=4),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "model-v1",
                    "embedding_size": 2,
                },
                embed=lambda _face: np.asarray([float("nan"), 0.0], dtype=np.float32),
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )
            snapshot = store.storage_dir / "snapshots" / "1.jpg"
            snapshot.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((32, 32, 3), dtype=np.uint8)))

            store._recognize_observation(observation_id)

            with store._connect() as connection:
                row = connection.execute(
                    "select embedding_blob, recognition_pending, recognition_error from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertIsNone(row["embedding_blob"])
            self.assertEqual(row["recognition_pending"], 0)
            self.assertIn("invalid", row["recognition_error"].lower())

    def test_identity_changes_rematch_saved_embeddings_without_reinference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(face_max_references=5, face_match_threshold=0.4),
                status=lambda: {"model_fingerprint": "model-v1"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            person = store.create_person("Alice")
            self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=0,
            )
            unknown_id = self.insert_observation(
                store,
                2,
                observed_at="2026-07-26T12:00:01+00:00",
                embedding=np.asarray([2.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=0,
            )

            store._refresh_unknown_recognition()

            observation = store.observation(unknown_id)
            self.assertEqual(observation["candidate_person_id"], person["id"])
            self.assertEqual(observation["candidate_confidence"], 1.0)
            self.assertEqual(store._recognition_queue.qsize(), 0)

    def test_rematch_failure_does_not_undo_committed_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(enabled=True)
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )
            store._refresh_unknown_recognition = Mock(  # type: ignore[method-assign]
                side_effect=sqlite3.OperationalError("database busy")
            )

            with self.assertLogs("survng.app.faces", level="ERROR"):
                person = store.create_person("Alice", observation_id)

            self.assertEqual(store.observation(observation_id)["person_id"], person["id"])

    def test_close_drains_queue_so_same_store_can_restart_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"ready": False, "model_fingerprint": "model-v1"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            store.start()
            store._queue_recognition(99)
            store.close()

            self.assertEqual(store._recognition_queue.qsize(), 0)
            self.assertEqual(store._recognition_pending, set())
            store.start()
            self.addCleanup(store.close)
            self.assertTrue(store._recognition_thread and store._recognition_thread.is_alive())

    def test_unavailable_isolated_worker_defers_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {
                    "ready": False,
                    "isolation": {"worker_alive": False, "last_error": "worker restarting"},
                },
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)

            with self.assertRaisesRegex(InferenceUnavailable, "worker restarting"):
                store._recognize_observation(1)

    def test_create_person_validates_observation_assignment_and_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )

            with self.assertRaisesRegex(ValueError, "observation not found"):
                store.create_person("Missing", 999)
            alice = store.create_person(" Alice ", observation_id)
            with self.assertRaisesRegex(ValueError, "name already exists"):
                store.create_person("alice")
            with self.assertRaisesRegex(ValueError, "already assigned"):
                store.create_person("Bob", observation_id)

            self.assertEqual(alice["name"], "Alice")
            self.assertEqual(store.stats()["people"], 1)

    def test_rejected_identity_history_accumulates_until_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            alice = store.create_person("Alice")
            bob = store.create_person("Bob")
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                candidate_person_id=alice["id"],
            )

            store.assign(observation_id, None)
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set candidate_person_id = ? where id = ?",
                    (bob["id"], observation_id),
                )
            store.assign(observation_id, None)
            with store._connect() as connection:
                rejected = {
                    int(row[0])
                    for row in connection.execute(
                        "select person_id from face_rejections where observation_id = ?",
                        (observation_id,),
                    )
                }
            self.assertEqual(rejected, {alice["id"], bob["id"]})

            store.assign(observation_id, alice["id"])
            with store._connect() as connection:
                remaining = connection.execute(
                    "select count(*) from face_rejections where observation_id = ?",
                    (observation_id,),
                ).fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_matching_skips_corrupt_reference_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                config=SimpleNamespace(face_max_references=5, face_match_threshold=0.4),
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            person = store.create_person("Alice")
            self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
            )
            corrupt_id = self.insert_observation(
                store,
                2,
                observed_at="2026-07-26T12:00:01+00:00",
                person_id=person["id"],
                embedding_model="model-v1",
            )
            target_id = self.insert_observation(
                store,
                3,
                observed_at="2026-07-26T12:00:02+00:00",
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set embedding_blob = ? where id = ?",
                    (b"bad", corrupt_id),
                )
                candidate_id, confidence = store._best_match(
                    connection,
                    target_id,
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    "model-v1",
                )

            self.assertEqual(candidate_id, person["id"])
            self.assertEqual(confidence, 1.0)

    def test_delete_person_resets_confirmed_observation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
            )
            person = store.create_person("Alice", observation_id)
            store._try_refresh_unknown_recognition = Mock(return_value=True)  # type: ignore[method-assign]

            self.assertTrue(store.delete_person(person["id"]))

            observation = store.observation(observation_id)
            self.assertIsNone(observation["person_id"])
            self.assertEqual(observation["review_status"], "unknown")
            self.assertIsNone(observation["match_confidence"])
            store._try_refresh_unknown_recognition.assert_called_once_with()

    def test_pruning_removes_oldest_unprotected_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), max_observations=100, start_recognition=False)
            person = store.create_person("Alice")
            oldest_known = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T00:00:01+00:00",
                person_id=person["id"],
            )
            protected_known = self.insert_observation(
                store,
                2,
                observed_at="2026-07-26T00:00:02+00:00",
                person_id=person["id"],
            )
            first_unknown = 0
            for event_id in range(3, 102):
                observation_id = self.insert_observation(
                    store,
                    event_id,
                    observed_at=f"2026-07-26T00:{event_id // 60:02d}:{event_id % 60:02d}+00:00",
                )
                first_unknown = first_unknown or observation_id

            with store._lock, store._connect() as connection:
                self.assertEqual(store._prune_locked(connection), 1)

            self.assertIsNone(store.observation(oldest_known))
            self.assertIsNotNone(store.observation(protected_known))
            self.assertIsNotNone(store.observation(first_unknown))

    def test_close_reports_an_unreaped_recognition_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            thread = Mock()
            thread.is_alive.return_value = True
            store._recognition_thread = thread

            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                store.close()

            thread.join.assert_called_once_with(timeout=20.0)


if __name__ == "__main__":
    unittest.main()
