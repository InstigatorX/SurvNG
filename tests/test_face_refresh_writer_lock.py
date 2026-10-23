from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.faces import FaceStore
from survng.app.face_store.quality import FaceMatch


class FaceRefreshWriterLockTest(unittest.TestCase):
    def test_unknown_refresh_match_phase_does_not_hold_writer_lock(self) -> None:
        with TemporaryDirectory() as tmpdir:
            write_lock = threading.RLock()
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(
                    face_min_size=4,
                    face_max_references=8,
                    face_match_threshold=0.4,
                    face_auto_identify_enabled=True,
                    face_auto_identify_threshold=0.8,
                    face_auto_identify_margin=0.2,
                ),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "test-v1",
                    "embedding_size": 2,
                },
                embed=lambda _face: np.asarray([1.0, 0.0], dtype=np.float32),
            )
            store = FaceStore(
                Path(tmpdir),
                recognizer=recognizer,
                start_recognition=False,
                database_write_lock=write_lock,
            )
            alice = store.create_person("Alice")["id"]
            path = store.storage_dir / "gate.jpg"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8)))
            store.ingest_candidates(
                1,
                "gate",
                "2026-08-08T12:00:00+00:00",
                [{
                    "snapshot_path": str(path),
                    "box": {"x1": 0, "y1": 0, "x2": 40, "y2": 40},
                    "confidence": 0.9,
                    "track_id": "face-1",
                    "rank": 1,
                    "offset_seconds": 0.0,
                    "quality_score": 0.8,
                }],
            )
            with store._connect() as connection:
                observation_id = int(
                    connection.execute(
                        "select id from face_observations where event_id = 1"
                    ).fetchone()[0]
                )
            with patch(
                "survng.app.face_store.recognition._face_quality",
                return_value=SimpleNamespace(
                    score=0.8,
                    sharpness=0.8,
                    exposure=0.8,
                    contrast=0.8,
                    size=0.8,
                    edge_detail=0.8,
                ),
            ), patch.object(
                store,
                "_match_result",
                return_value=FaceMatch(alice, 0.95, 0.1, 0.85, (11, 12, 13), (0.95,) * 3),
            ):
                store._recognize_observation(observation_id)
            with store._connect() as connection:
                row = connection.execute(
                    "select embedding_blob, recognition_pending from face_observations where id = ?",
                    (observation_id,),
                ).fetchone()
            self.assertIsNotNone(row["embedding_blob"])
            self.assertEqual(int(row["recognition_pending"]), 0)

            match_started = threading.Event()
            release_match = threading.Event()
            acquired_during_match: list[bool] = []

            def blocking_match(connection, row_id, embedding, model_fingerprint):
                match_started.set()
                self.assertTrue(release_match.wait(2.0))
                return FaceMatch(alice, 0.95, 0.1, 0.85, (11, 12, 13), (0.95,) * 3)

            def competing_writer() -> None:
                self.assertTrue(match_started.wait(2.0))
                acquired = write_lock.acquire(timeout=0.25)
                acquired_during_match.append(acquired)
                if acquired:
                    write_lock.release()
                release_match.set()

            with patch.object(store, "_match_result", side_effect=blocking_match):
                competitor = threading.Thread(target=competing_writer)
                competitor.start()
                started = time.monotonic()
                store._refresh_unknown_recognition()
                competitor.join(2.0)
                self.assertFalse(competitor.is_alive())
                self.assertLess(time.monotonic() - started, 3.0)

            self.assertEqual(acquired_during_match, [True])


if __name__ == "__main__":
    unittest.main()
