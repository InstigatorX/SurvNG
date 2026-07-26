from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from survng.app.faces import FaceStore


class FaceStoreTest(unittest.TestCase):
    def test_recognition_is_queued_only_after_observation_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
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
                    "snapshot_path": str(Path(tmpdir) / "snapshots" / "face.jpg"),
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

    def test_malformed_event_objects_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
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
            ])

            self.assertEqual(inserted, 0)

    def test_close_reports_an_unreaped_recognition_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FaceStore(Path(tmpdir), start_recognition=False)
            thread = Mock()
            thread.is_alive.return_value = True
            store._recognition_thread = thread

            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                store.close()

            thread.join.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
