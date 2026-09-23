from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from survng.app.appearance_index import AppearanceIndex
from survng.app.events import EventStore


class AppearancePersonLookupTest(unittest.TestCase):
    def test_person_embedding_for_track_returns_normalized_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event("gate", "motion", created_at="2026-01-01T00:00:00+00:00")
            index = AppearanceIndex(store.db_path)
            vector = np.asarray([3.0, 4.0], dtype=np.float32)
            index.replace_event(
                int(event["id"]),
                "gate",
                [
                    {
                        "track_id": 5,
                        "label": "person",
                        "model_kind": "person",
                        "model_fingerprint": "reid-v1",
                        "embedding": vector,
                        "match_threshold": 0.7,
                        "quality": 0.9,
                        "observation_count": 4,
                        "source": "tracking_multiframe",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "first_seen": "2026-01-01T00:00:00+00:00",
                        "last_seen": "2026-01-01T00:00:01+00:00",
                    }
                ],
            )
            record = index.person_embedding_for_track(int(event["id"]), 5)
            self.assertIsNotNone(record)
            assert record is not None
            embedding = record["embedding"]
            self.assertEqual(record["model_fingerprint"], "reid-v1")
            np.testing.assert_allclose(
                embedding, np.asarray([0.6, 0.8], dtype=np.float32), atol=1e-5
            )
            embedding[0] = 0.0
            again = index.person_embedding_for_track(int(event["id"]), 5)
            assert again is not None
            np.testing.assert_allclose(
                again["embedding"], np.asarray([0.6, 0.8], dtype=np.float32), atol=1e-5
            )

    def test_person_embedding_for_track_ignores_vehicles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event("gate", "motion", created_at="2026-01-01T00:00:00+00:00")
            index = AppearanceIndex(store.db_path)
            index.replace_event(
                int(event["id"]),
                "gate",
                [
                    {
                        "track_id": 5,
                        "label": "car",
                        "model_kind": "vehicle",
                        "model_fingerprint": "veh-v1",
                        "embedding": [1.0, 0.0],
                        "match_threshold": 0.8,
                        "quality": 0.9,
                        "observation_count": 2,
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            )
            self.assertIsNone(index.person_embedding_for_track(int(event["id"]), 5))


if __name__ == "__main__":
    unittest.main()
