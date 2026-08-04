from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from survng.app.semantic_search import (
    SemanticEvidence,
    SemanticIndex,
    SemanticModelIdentity,
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


if __name__ == "__main__":
    unittest.main()
