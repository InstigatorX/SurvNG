from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from survng.app.semantic_search import (
    SemanticEvidence,
    SemanticIndex,
    SemanticModelIdentity,
    SemanticSearchService,
    normalized_matrix,
)


class SemanticIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "events.db"
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("create table events (id integer primary key)")
            connection.executemany("insert into events(id) values (?)", [(1,), (2,)])
        self.index = SemanticIndex(self.database_path)
        self.identity = SemanticModelIdentity("test", "model-a", "prep-a", 3)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_normalization_rejects_invalid_embeddings(self) -> None:
        with self.assertRaises(ValueError):
            normalized_matrix([0.0, 0.0])
        with self.assertRaises(ValueError):
            normalized_matrix([1.0, float("nan")])

    def test_search_ranks_cosine_similarity_and_preserves_evidence(self) -> None:
        evidence = [
            SemanticEvidence(1, "gate", "2026-08-03T12:00:00+00:00", "full_frame", "frame", "one.webp"),
            SemanticEvidence(2, "driveway", "2026-08-03T12:01:00+00:00", "object_crop", "car:0", "two.webp", "car", (1, 2, 3, 4)),
        ]
        self.assertEqual(
            self.index.upsert(evidence, [[1, 0, 0], [0.8, 0.2, 0]], self.identity),
            2,
        )

        hits = self.index.search([1, 0, 0], self.identity, limit=2)

        self.assertEqual([hit.event_id for hit in hits], [1, 2])
        self.assertEqual(hits[1].bbox, (1, 2, 3, 4))
        self.assertGreater(hits[0].score, hits[1].score)

    def test_generations_are_isolated(self) -> None:
        evidence = [SemanticEvidence(1, "gate", "now", "full_frame", "frame", "one.webp")]
        self.index.upsert(evidence, [[1, 0, 0]], self.identity)
        other = SemanticModelIdentity("test", "model-b", "prep-a", 3)
        self.index.upsert(evidence, [[0, 1, 0]], other)

        self.assertEqual(self.index.coverage(self.identity)["event_count"], 1)
        self.assertEqual(self.index.coverage(other)["event_count"], 1)
        self.assertEqual(self.index.search([1, 0, 0], self.identity)[0].score, 1.0)

    def test_upsert_is_idempotent_per_generation_and_source(self) -> None:
        evidence = [SemanticEvidence(1, "gate", "now", "full_frame", "frame", "one.webp")]
        self.index.upsert(evidence, [[1, 0, 0]], self.identity)
        self.index.upsert(evidence, [[0, 1, 0]], self.identity)

        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 1, "event_count": 1})
        self.assertAlmostEqual(
            self.index.search([0, 1, 0], self.identity)[0].score,
            1.0,
            places=3,
        )

    def test_service_indexes_full_frame_and_object_crop(self) -> None:
        from survng.app.config import SemanticSearchConfig

        image_path = Path(self.temporary.name) / "snapshot.jpg"
        cv2.imwrite(str(image_path), np.full((100, 200, 3), 127, dtype=np.uint8))

        class FakeEncoder:
            identity = self.identity

            def encode_images(self, images):
                self.shapes = [image.shape for image in images]
                return np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            def encode_text(self, texts):
                return np.asarray([[1, 0, 0]], dtype=np.float32)

            def close(self):
                return

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        encoder = FakeEncoder()
        service.encoder = encoder
        service._storage_dir = Path(self.temporary.name)
        service._index_event({
            "id": 1, "camera_id": "gate", "created_at": "now",
            "snapshot_path": "snapshot.jpg",
            "objects_json": '[{"label":"car","bbox":[10,20,110,80]}]',
        })

        self.assertEqual(encoder.shapes, [(100, 200, 3), (60, 100, 3)])
        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 2, "event_count": 1})

    def test_live_events_have_priority_over_historical_backfill(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()
        service._queue.put_nowait((1, next(service._queue_sequence), {"id": 1}))

        self.assertTrue(service.queue_event({"id": 2, "snapshot_path": "live.webp"}))

        priority, _sequence, event = service._queue.get_nowait()
        self.assertEqual(priority, 0)
        self.assertEqual(event["id"], 2)

    def test_historical_backfill_reserves_capacity_for_live_events(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True, worker_queue_size=16),
            self.index,
            Path(self.temporary.name),
            {},
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()

        self.assertEqual(service._live_queue_reserve, 4)
        for event_id in range(12):
            service._queue.put_nowait(
                (1, next(service._queue_sequence), {"id": event_id + 1})
            )
        self.assertFalse(service._history_queue_has_capacity())
        self.assertTrue(service.queue_event({"id": 99, "snapshot_path": "live.webp"}))


if __name__ == "__main__":
    unittest.main()
