from __future__ import annotations

import json
from queue import Queue
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from survng.app.faces import FaceStore, _face_quality
from survng.app.inference import InferenceUnavailable


class FaceStoreTest(unittest.TestCase):
    def test_refined_candidates_remain_bounded_to_four_per_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            store = FaceStore(storage, start_recognition=False)
            batches = []
            paths = []
            for batch_index in range(2):
                batch = []
                for local_index in range(4):
                    offset = batch_index * 4 + local_index + 1
                    path = storage / f"candidate-{offset}.jpg"
                    path.write_bytes(b"candidate")
                    paths.append(path)
                    batch.append({
                        "snapshot_path": str(path),
                        "box": {"x1": 5, "y1": 5, "x2": 35, "y2": 35},
                        "confidence": 0.9,
                        "track_id": "face-1",
                        "rank": local_index + 1,
                        "offset_seconds": float(offset),
                        "quality_score": float(offset) / 10.0,
                    })
                batches.append(batch)

            store.ingest_candidates(103, "gate", "2026-08-08T12:00:00+00:00", batches[0])
            store.ingest_candidates(103, "gate", "2026-08-08T12:00:00+00:00", batches[1])

            with store._connect() as connection:
                retained = connection.execute(
                    "select candidate_offset_seconds from face_observations where event_id = 103"
                ).fetchall()
            self.assertEqual(
                {float(row["candidate_offset_seconds"]) for row in retained},
                {5.0, 6.0, 7.0, 8.0},
            )
            self.assertEqual(sum(path.exists() for path in paths), 4)
            self.assertEqual(store.observation_count(), 1)

    def test_pruning_removes_complete_candidate_track_and_its_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            store = FaceStore(storage, max_observations=100, start_recognition=False)
            first_paths: list[Path] = []
            with store._connect() as connection:
                for event_id in range(1, 102):
                    for rank in (1, 2):
                        path = storage / f"face-{event_id}-{rank}.jpg"
                        path.write_bytes(b"candidate")
                        if event_id == 1:
                            first_paths.append(path)
                        connection.execute(
                            """
                            insert into face_observations (
                                event_id, object_index, camera_id, snapshot_path,
                                box_json, observed_at, created_at,
                                candidate_track_id, candidate_rank,
                                candidate_offset_seconds, canonical
                            ) values (?, ?, 'gate', ?, ?, ?, ?, 'face-1', ?, ?, ?)
                            """,
                            (
                                event_id,
                                rank - 1,
                                path.name,
                                '{"x1":1,"y1":1,"x2":10,"y2":10}',
                                f"{event_id:03}",
                                f"{event_id:03}",
                                rank,
                                float(rank),
                                int(rank == 1),
                            ),
                        )
                removed = store._prune_locked(connection)

            self.assertEqual(removed, 2)
            self.assertEqual(store.observation_count(), 100)
            self.assertTrue(all(not path.exists() for path in first_paths))
            with store._connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "select count(*) from face_observations where event_id = 1"
                    ).fetchone()[0],
                    0,
                )

    def test_temporal_candidates_expose_only_one_canonical_face(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            candidates = []
            for rank, quality in enumerate((0.7, 0.9, 0.8), start=1):
                path = storage / f"candidate-{rank}.jpg"
                self.assertTrue(cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8)))
                candidates.append({
                    "snapshot_path": str(path),
                    "box": {"x1": 5, "y1": 5, "x2": 35, "y2": 35},
                    "confidence": 0.9,
                    "track_id": "face-1",
                    "rank": rank,
                    "offset_seconds": float(rank),
                    "quality_score": quality,
                })
            store = FaceStore(storage, start_recognition=False)

            self.assertEqual(
                store.ingest_candidates(
                    101,
                    "gate",
                    "2026-08-08T12:00:00+00:00",
                    candidates,
                ),
                3,
            )
            self.assertEqual(store.observation_count(), 1)
            observations = store.observations()
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0]["candidate_track_id"], "face-1")
            self.assertEqual(observations[0]["candidate_rank"], 2)

    def test_temporal_consensus_selects_best_supported_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            with store._connect() as connection:
                person_id = int(connection.execute(
                    "insert into face_people (name, notes, created_at, updated_at) values ('Steve', '', 'now', 'now')"
                ).lastrowid)
            candidates = []
            for rank, quality in enumerate((0.55, 0.92, 0.75), start=1):
                path = Path(tmpdir) / f"candidate-{rank}.jpg"
                self.assertTrue(cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8)))
                candidates.append({
                    "snapshot_path": str(path),
                    "box": {"x1": 5, "y1": 5, "x2": 35, "y2": 35},
                    "confidence": 0.9,
                    "track_id": "face-1",
                    "rank": rank,
                    "offset_seconds": float(rank),
                    "quality_score": quality,
                })
            store.ingest_candidates(102, "gate", "2026-08-08T12:00:00+00:00", candidates)
            with store._connect() as connection:
                rows = connection.execute(
                    "select id from face_observations where event_id = 102 order by candidate_rank"
                ).fetchall()
                for row, confidence in zip(rows, (0.82, 0.88, 0.86)):
                    connection.execute(
                        """update face_observations set recognition_pending = 0,
                            candidate_person_id = ?, candidate_confidence = ?, quality_score = ?
                            where id = ?""",
                        (person_id, confidence, confidence, int(row["id"])),
                    )
                store._reconcile_candidate_track(connection, 102, "face-1")

            observation = store.observations()[0]
            self.assertEqual(observation["candidate_rank"], 2)
            self.assertEqual(observation["candidate_person_id"], person_id)
            self.assertEqual(observation["consensus"]["agreement_count"], 3)

    def test_face_quality_preserves_sharpness_separation(self) -> None:
        coordinates = np.indices((128, 128)) // 8
        checker = (coordinates.sum(axis=0) % 2 * 255).astype(np.uint8)
        sharp = np.repeat(checker[..., None], 3, axis=2)
        blurred = cv2.GaussianBlur(sharp, (11, 11), 0)

        sharp_quality = _face_quality(sharp, 0.9)
        blurred_quality = _face_quality(blurred, 0.9)

        self.assertLess(sharp_quality.sharpness, 1.0)
        self.assertGreater(sharp_quality.sharpness, blurred_quality.sharpness)
        self.assertGreater(sharp_quality.edge_detail, blurred_quality.edge_detail)

    def test_observation_limit_reconfigures_and_prunes_without_restarting_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), max_observations=100)
            with patch.object(store, "_prune_locked", return_value=4) as prune:
                removed = store.reconfigure_max_observations(500)

            self.assertEqual(removed, 4)
            self.assertEqual(store.max_observations, 500)
            prune.assert_called_once()

    def test_failed_observation_limit_reconfiguration_restores_previous_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), max_observations=250)
            with (
                patch.object(store, "_prune_locked", side_effect=sqlite3.OperationalError("busy")),
                self.assertRaisesRegex(sqlite3.OperationalError, "busy"),
            ):
                store.reconfigure_max_observations(500)

            self.assertEqual(store.max_observations, 250)

    def test_database_can_be_local_while_snapshots_remain_in_media_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as database:
            store = FaceStore(
                Path(storage),
                start_recognition=False,
                database_dir=Path(database),
            )

            self.assertEqual(store.storage_dir, Path(storage))
            self.assertEqual(store.db_path, Path(database) / "survng.sqlite3")

    def test_face_stats_separate_too_small_from_processing_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            person = store.create_person("Alice")
            known_id = self.insert_observation(
                store, 1, observed_at="2026-08-13T12:00:00+00:00", person_id=person["id"]
            )
            unknown_id = self.insert_observation(
                store,
                2,
                observed_at="2026-08-13T12:01:00+00:00",
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=0,
            )
            too_small_id = self.insert_observation(
                store, 3, observed_at="2026-08-13T12:02:00+00:00", recognition_pending=0
            )
            failed_id = self.insert_observation(
                store, 4, observed_at="2026-08-13T12:03:00+00:00", recognition_pending=0
            )
            pending_id = self.insert_observation(
                store, 5, observed_at="2026-08-13T12:04:00+00:00"
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set recognition_error = 'Face is smaller than 48px.', recognition_outcome = 'too_small' where id = ?",
                    (too_small_id,),
                )
                connection.execute(
                    "update face_observations set recognition_error = 'Snapshot is unavailable.', recognition_outcome = 'failed' where id = ?",
                    (failed_id,),
                )

            stats = store.stats()

            self.assertEqual(stats["observations"], 5)
            self.assertEqual(stats["actionable_observations"], 2)
            self.assertEqual(stats["known"], 1)
            self.assertEqual(stats["unknown"], 1)
            self.assertEqual(stats["embedded_unknown"], 1)
            self.assertEqual(stats["too_small"], 1)
            self.assertEqual(stats["processing_failed"], 1)
            self.assertEqual(stats["pending"], 1)
            self.assertEqual(stats["identified_percent"], 50.0)
            self.assertEqual(stats["by_camera"][0]["camera_id"], "gate")
            self.assertEqual(store.observation_count(status="unknown"), 1)
            self.assertEqual(store.observation_count(status="pending"), 1)
            self.assertEqual(store.observation_count(status="unusable"), 2)
            self.assertEqual([item["id"] for item in store.observations(status="unknown")], [unknown_id])
            self.assertEqual(
                {item["id"] for item in store.observations(status="unusable")},
                {too_small_id, failed_id},
            )
            self.assertNotIn(pending_id, {item["id"] for item in store.observations(status="unknown")})
            self.assertEqual(
                [item["id"] for item in store.observations(status="pending")],
                [pending_id],
            )
            self.assertIsNotNone(store.observation(known_id))

    def test_recognition_status_classifies_small_faces_as_non_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                status=lambda: {"enabled": True, "ready": True, "model_fingerprint": "model-v1"}
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            too_small_id = self.insert_observation(
                store, 1, observed_at="2026-08-13T12:00:00+00:00", recognition_pending=0
            )
            failed_id = self.insert_observation(
                store, 2, observed_at="2026-08-13T12:01:00+00:00", recognition_pending=0
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set recognition_error = 'Face is smaller than 48px.', recognition_outcome = 'too_small' where id = ?",
                    (too_small_id,),
                )
                connection.execute(
                    "update face_observations set recognition_error = 'Face crop is invalid.', recognition_outcome = 'failed' where id = ?",
                    (failed_id,),
                )

            status = store.recognition_status()

            self.assertEqual(status["too_small"], 1)
            self.assertEqual(status["failed"], 1)

    def test_recognition_status_does_not_publish_stale_pending_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                status=lambda: {"enabled": True, "ready": True, "model_fingerprint": "model-v2"}
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            person = store.create_person("Alice")
            stale_id = self.insert_observation(
                store,
                1,
                observed_at="2026-08-13T12:00:00+00:00",
                candidate_person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=1,
            )
            current_id = self.insert_observation(
                store,
                2,
                observed_at="2026-08-13T12:01:00+00:00",
                candidate_person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v2",
                recognition_pending=0,
            )

            status = store.recognition_status()

            self.assertEqual(status["suggested"], 1)
            self.assertEqual(
                [item["id"] for item in store.observations(status="suggested")],
                [current_id],
            )
            self.assertNotEqual(stale_id, current_id)

    def test_people_count_only_references_for_the_active_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                status=lambda: {"model_fingerprint": "current-model"}
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            person = store.create_person("Alice")
            self.insert_observation(
                store,
                1,
                observed_at="2026-08-13T12:00:00+00:00",
                person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="current-model",
                recognition_pending=0,
            )
            self.insert_observation(
                store,
                2,
                observed_at="2026-08-13T12:01:00+00:00",
                person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="old-model",
                recognition_pending=0,
            )

            result = store.people()[0]

            self.assertEqual(result["reference_count"], 2)
            self.assertEqual(result["usable_reference_count"], 1)

    def test_model_upgrade_marks_stale_embeddings_pending_before_queueing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"model_fingerprint": "current-model"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-08-13T12:00:00+00:00",
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="old-model",
                recognition_pending=0,
            )

            store._queue_pending_recognition()

            with store._connect() as connection:
                row = connection.execute(
                    "select recognition_pending, recognition_outcome from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertEqual(row["recognition_pending"], 1)
            self.assertEqual(row["recognition_outcome"], "pending")
            self.assertEqual(store.stats()["unknown"], 0)
            self.assertEqual(store.stats()["pending"], 1)
            self.assertEqual(store._recognition_queue.get_nowait(), observation_id)

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
                    embedding_blob, embedding_model, recognition_pending,
                    recognition_outcome
                ) values (?, 0, ?, 'gate', ?, ?, 0.8, ?, null, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    person_id,
                    str(store.storage_dir / "snapshots" / f"{event_id}.jpg"),
                    json.dumps({"x1": 1, "y1": 2, "x2": 20, "y2": 24}),
                    observed_at,
                    "confirmed" if person_id is not None else "unknown",
                    observed_at,
                    candidate_person_id,
                    embedding.astype(np.float32).tobytes() if embedding is not None else None,
                    embedding_model,
                    recognition_pending,
                    "pending" if recognition_pending else "embedded" if embedding is not None else "failed",
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

    def test_provisional_event_snapshot_is_not_ingested_as_face_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            snapshot = Path(tmpdir) / "snapshot.jpg"
            snapshot.write_bytes(b"image")

            inserted = store.ingest_events([{
                "id": 42,
                "camera_id": "front-door",
                "snapshot_path": str(snapshot),
                "created_at": "2026-08-08T12:00:00+00:00",
                "objects_json": json.dumps([
                    {
                        "label": "face",
                        "confidence": 0.8,
                        "box": {"x1": 1, "y1": 2, "x2": 20, "y2": 24},
                    },
                    {"status": "face_evidence_pending"},
                ]),
            }])

            self.assertEqual(inserted, 0)
            self.assertEqual(store.observation_count(), 0)

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

    def test_full_recognition_queue_marks_pending_work_for_refill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"model_fingerprint": "model-v1"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            store._recognition_queue = Queue(maxsize=1)
            first_id = self.insert_observation(
                store, 1, observed_at="2026-07-26T12:00:02+00:00"
            )
            second_id = self.insert_observation(
                store, 2, observed_at="2026-07-26T12:00:01+00:00"
            )

            with self.assertLogs("survng.app.faces", level="WARNING"):
                store._queue_pending_recognition()

            self.assertTrue(store._recognition_refill_needed.is_set())
            self.assertEqual(store._recognition_queue.get_nowait(), first_id)
            with store._recognition_pending_lock:
                store._recognition_pending.discard(first_id)
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set recognition_pending = 0 where id = ?",
                    (first_id,),
                )
            store._recognition_refill_needed.clear()
            store._queue_pending_recognition()

            self.assertEqual(store._recognition_queue.get_nowait(), second_id)

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

    def test_match_refresh_is_deferred_to_running_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(enabled=True)
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            store._recognition_thread = Mock(is_alive=Mock(return_value=True))
            store._try_refresh_unknown_recognition = Mock(return_value=True)  # type: ignore[method-assign]

            store.request_match_refresh()

            self.assertTrue(store._match_refresh_needed.is_set())
            store._try_refresh_unknown_recognition.assert_not_called()

    def test_quality_algorithm_upgrade_requeues_successful_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            observation_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
                recognition_pending=0,
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set quality_version = 0 where id = ?",
                    (observation_id,),
                )

            store._init_db()

            with store._connect() as connection:
                row = connection.execute(
                    "select recognition_pending from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertEqual(row["recognition_pending"], 1)

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

    def test_enabled_recognizer_warmup_defers_instead_of_dropping_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                status=lambda: {"ready": False, "isolation": {"worker_alive": True}},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)

            with self.assertRaisesRegex(InferenceUnavailable, "still starting"):
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

    def test_matching_ignores_identity_with_only_corrupt_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                config=SimpleNamespace(face_max_references=5, face_match_threshold=0.4),
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            person = store.create_person("Alice")
            corrupt_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:01+00:00",
                person_id=person["id"],
                embedding_model="model-v1",
            )
            target_id = self.insert_observation(
                store, 2, observed_at="2026-07-26T12:00:02+00:00"
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set embedding_blob = ? where id = ?",
                    (b"bad", corrupt_id),
                )
                match = store._best_match(
                    connection,
                    target_id,
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    "model-v1",
                )

            self.assertEqual(match, (None, None))

    def test_reference_gallery_prefers_embedding_diversity_over_recent_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            person = store.create_person("Alice")
            vectors = (
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.999, 0.001], dtype=np.float32),
                np.asarray([0.998, 0.002], dtype=np.float32),
                np.asarray([0.6, 0.8], dtype=np.float32),
            )
            ids = [
                self.insert_observation(
                    store,
                    index + 1,
                    observed_at=f"2026-07-26T12:00:0{index}+00:00",
                    person_id=person["id"],
                    embedding=vector,
                    embedding_model="model-v1",
                )
                for index, vector in enumerate(vectors)
            ]
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set quality_score = 0.8 where person_id = ?",
                    (person["id"],),
                )
                rows = connection.execute(
                    """select id, person_id, camera_id, confidence, quality_score,
                        reference_pinned, observed_at, embedding_blob
                        from face_observations where person_id = ?""",
                    (person["id"],),
                ).fetchall()

            selected = store._select_reference_gallery(rows, 2, (2,))

            self.assertEqual(len(selected), 2)
            self.assertIn(ids[-1], {int(item["id"]) for item in selected})

    def test_pinned_reference_is_retained_when_gallery_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            person = store.create_person("Alice")
            pinned_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T12:00:00+00:00",
                person_id=person["id"],
                embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                embedding_model="model-v1",
            )
            self.insert_observation(
                store,
                2,
                observed_at="2026-07-26T12:00:01+00:00",
                person_id=person["id"],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                embedding_model="model-v1",
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set reference_pinned = 1 where id = ?",
                    (pinned_id,),
                )
                rows = connection.execute(
                    """select id, person_id, camera_id, confidence, quality_score,
                        reference_pinned, observed_at, embedding_blob
                        from face_observations where person_id = ?""",
                    (person["id"],),
                ).fetchall()

            selected = store._select_reference_gallery(rows, 1, (2,))

            self.assertEqual([int(item["id"]) for item in selected], [pinned_id])

    def test_strong_well_separated_match_can_be_auto_identified_without_becoming_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(
                    face_min_size=4,
                    face_max_references=5,
                    face_match_threshold=0.4,
                    face_auto_identify_enabled=True,
                    face_auto_identify_threshold=0.9,
                    face_auto_identify_margin=0.5,
                ),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "model-v1",
                    "embedding_size": 2,
                },
                embed=lambda _face: np.asarray([1.0, 0.0], dtype=np.float32),
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            alice = store.create_person("Alice")
            bob = store.create_person("Bob")
            for index in range(3):
                self.insert_observation(
                    store,
                    index + 1,
                    observed_at=f"2026-07-26T12:00:0{index}+00:00",
                    person_id=alice["id"],
                    embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                    embedding_model="model-v1",
                    recognition_pending=0,
                )
                self.insert_observation(
                    store,
                    index + 10,
                    observed_at=f"2026-07-26T12:01:0{index}+00:00",
                    person_id=bob["id"],
                    embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                    embedding_model="model-v1",
                    recognition_pending=0,
                )
            target_id = self.insert_observation(
                store,
                20,
                observed_at="2026-07-26T12:02:00+00:00",
            )
            checker = np.indices((32, 32)).sum(axis=0) % 2
            snapshot = store.storage_dir / "snapshots" / "20.jpg"
            snapshot.parent.mkdir(exist_ok=True)
            image = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
            self.assertTrue(cv2.imwrite(str(snapshot), image))

            store._recognize_observation(target_id)

            observation = store.observation(target_id)
            self.assertEqual(observation["person_id"], alice["id"])
            self.assertEqual(observation["review_status"], "auto_identified")
            self.assertTrue(observation["auto_identified"])
            self.assertFalse(observation["reference_pinned"])

    def test_calibration_uses_confirmed_identity_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = SimpleNamespace(
                config=SimpleNamespace(
                    face_match_threshold=0.4,
                    face_auto_identify_threshold=0.55,
                    face_auto_identify_margin=0.12,
                ),
                status=lambda: {"model_fingerprint": "model-v1"},
            )
            store = FaceStore(Path(tmpdir), recognizer=recognizer, start_recognition=False)
            alice = store.create_person("Alice")
            bob = store.create_person("Bob")
            for index in range(3):
                self.insert_observation(
                    store,
                    index + 1,
                    observed_at=f"2026-07-26T12:00:0{index}+00:00",
                    person_id=alice["id"],
                    embedding=np.asarray([1.0, 0.02 * index], dtype=np.float32),
                    embedding_model="model-v1",
                    recognition_pending=0,
                )
                self.insert_observation(
                    store,
                    index + 10,
                    observed_at=f"2026-07-26T12:01:0{index}+00:00",
                    person_id=bob["id"],
                    embedding=np.asarray([0.02 * index, 1.0], dtype=np.float32),
                    embedding_model="model-v1",
                    recognition_pending=0,
                )

            result = store.calibration()

            self.assertTrue(result["ready"])
            self.assertEqual(result["confirmed_samples"], 6)
            self.assertEqual(result["rank_one_accuracy"], 1.0)
            self.assertGreaterEqual(result["recommended"]["automatic_threshold"], 0.5)

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

    def test_pruning_never_removes_a_pinned_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), max_observations=100, start_recognition=False)
            person = store.create_person("Alice")
            pinned_id = self.insert_observation(
                store,
                1,
                observed_at="2026-07-26T00:00:01+00:00",
                person_id=person["id"],
            )
            with store._connect() as connection:
                connection.execute(
                    "update face_observations set reference_pinned = 1 where id = ?",
                    (pinned_id,),
                )
            for event_id in range(2, 102):
                self.insert_observation(
                    store,
                    event_id,
                    observed_at=f"2026-07-26T00:{event_id // 60:02d}:{event_id % 60:02d}+00:00",
                )

            with store._lock, store._connect() as connection:
                self.assertEqual(store._prune_locked(connection), 1)

            self.assertIsNotNone(store.observation(pinned_id))

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
