from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.faces import FaceStore
from survng.app.face_store.quality import FaceMatch
from survng.app.inference import InferenceUnavailable


class FaceIdentityReconciliationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.recognizer = SimpleNamespace(
            enabled=True,
            config=SimpleNamespace(
                face_min_size=4,
                face_max_references=8,
                face_match_threshold=0.4,
                face_auto_identify_enabled=True,
                face_auto_identify_threshold=0.8,
                face_auto_identify_margin=0.2,
            ),
            status=lambda: {"ready": True, "model_fingerprint": "test-v1", "embedding_size": 2},
            embed=lambda _face: np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.store = FaceStore(Path(self.directory.name), recognizer=self.recognizer, start_recognition=False)
        self.alice = self.store.create_person("Alice")["id"]
        self.bob = self.store.create_person("Bob")["id"]
        self.matches: dict[int, FaceMatch] = {}
        self.match_patch = patch.object(
            self.store, "_match_result",
            side_effect=lambda _connection, observation_id, *_args: self.matches[observation_id],
        )
        self.match_patch.start()
        self.addCleanup(self.match_patch.stop)
        self.updates: list[dict] = []

        def publish(payload: dict) -> None:
            # A separate connection must see the committed identity, and the
            # notification must refer to the selected crop, not the worker job.
            with self.store._connect() as connection:
                row = connection.execute(
                    "select person_id from face_observations where id = ?",
                    (payload["observation_id"],),
                ).fetchone()
            self.assertEqual(row["person_id"], payload["current_person_id"])
            self.updates.append(payload)

        self.store.set_identity_event_publisher(publish)

    def candidates(self, people: list[int], scores: list[float] | None = None, *, event_id: int = 1, first_rank: int = 1) -> list[int]:
        items = []
        for rank in range(first_rank, first_rank + len(people)):
            path = self.store.storage_dir / f"{event_id}-{rank}.jpg"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((40, 40, 3), dtype=np.uint8)))
            items.append({
                "snapshot_path": str(path), "box": {"x1": 0, "y1": 0, "x2": 40, "y2": 40},
                "confidence": 0.9, "track_id": "face-1", "rank": rank,
                "offset_seconds": float(rank), "quality_score": 0.8,
            })
        self.store.ingest_candidates(event_id, "gate", "2026-08-08T12:00:00+00:00", items)
        with self.store._connect() as connection:
            ids = [int(row[0]) for row in connection.execute(
                "select id from face_observations where event_id = ? and candidate_rank >= ? order by candidate_rank",
                (event_id, first_rank),
            )]
        for observation_id, person, score in zip(ids, people, scores or [0.95] * len(people)):
            self.matches[observation_id] = FaceMatch(person, score, 0.1, score - 0.1, (11, 12, 13), (score,) * 3)
        return ids

    def recognize(self, observation_id: int, *, quality: float = 0.8) -> bool:
        with patch("survng.app.face_store.recognition._face_quality", return_value=SimpleNamespace(
            score=quality, sharpness=0.8, exposure=0.8, contrast=0.8, size=0.8, edge_detail=0.8,
        )):
            return self.store._recognize_observation(observation_id)

    def canonical(self, event_id: int = 1) -> dict:
        with self.store._connect() as connection:
            row = connection.execute(
                "select * from face_observations where event_id = ? and canonical = 1", (event_id,),
            ).fetchone()
        self.assertIsNotNone(row)
        return dict(row)

    def test_sequential_votes_survive_display_clearing_and_job_order(self) -> None:
        self.recognizer.config.face_auto_identify_enabled = False
        results = []
        for event_id, order in enumerate(((0, 1, 2), (2, 1, 0), (1, 0, 2)), start=1):
            ids = self.candidates([self.alice, self.bob, self.bob], [0.90, 0.85, 0.84], event_id=event_id)
            for index in order:
                self.recognize(ids[index])
            row = self.canonical(event_id)
            consensus = json.loads(row["consensus_json"])
            results.append((row["candidate_rank"], row["candidate_person_id"], consensus))
            with self.store._connect() as connection:
                evidence = [json.loads(item[0])["person_id"] for item in connection.execute(
                    "select match_details_json from face_observations where event_id = ? order by candidate_rank",
                    (event_id,),
                )]
            self.assertEqual(evidence, [self.alice, self.bob, self.bob])
            self.assertEqual(row["candidate_person_id"], self.bob)
            self.assertEqual(consensus["agreement_count"], 2)
        self.assertEqual(results, [results[0]] * len(results))

    def test_tied_final_evidence_never_publishes_a_partial_majority(self) -> None:
        for event_id, order in enumerate(((0, 1, 2, 3), (2, 3, 0, 1), (0, 2, 1, 3)), start=1):
            ids = self.candidates([self.alice, self.alice, self.bob, self.bob], event_id=event_id)
            for index in order:
                self.assertFalse(self.recognize(ids[index]))
                self.assertIsNone(self.canonical(event_id)["person_id"])
            self.assertEqual(json.loads(self.canonical(event_id)["consensus_json"])["agreement_count"], 2)
        self.assertEqual(self.updates, [])

    def test_terminal_majority_notifies_actual_canonical_once_after_commit(self) -> None:
        ids = self.candidates([self.alice, self.alice, self.bob])
        self.assertFalse(self.recognize(ids[0], quality=0.95))
        self.assertFalse(self.recognize(ids[1], quality=0.6))
        self.assertEqual(self.updates, [])
        self.assertTrue(self.recognize(ids[2]))
        self.assertEqual(self.canonical()["id"], ids[0])
        self.assertEqual(self.updates[0]["observation_id"], ids[0])
        self.assertEqual(self.updates[0]["person_id"], self.alice)
        self.assertEqual(self.updates[0]["source"], "auto_recognition")
        self.assertTrue(self.recognize(ids[2]))
        self.assertEqual(len(self.updates), 1)

    def test_pending_inference_blocks_assignment_and_terminal_failure_reconciles(self) -> None:
        ids = self.candidates([self.alice, self.alice, self.bob])
        self.recognize(ids[0], quality=0.95)
        self.recognize(ids[1], quality=0.6)
        with patch.object(self.recognizer, "embed", side_effect=InferenceUnavailable("retry")):
            with self.assertRaises(InferenceUnavailable):
                self.recognize(ids[2])
        self.assertIsNone(self.canonical()["person_id"])
        self.assertEqual(self.updates, [])
        with patch.object(self.recognizer, "embed", side_effect=ValueError("invalid embedding")):
            self.assertTrue(self.recognize(ids[2]))
        self.assertEqual(self.canonical()["person_id"], self.alice)
        self.assertEqual(self.updates[0]["observation_id"], ids[0])
        failed = self.store.observation(ids[2])
        self.assertEqual(failed["recognition_outcome"], "failed")

    def test_missing_runner_up_cannot_auto_identify_a_track(self) -> None:
        ids = self.candidates([self.alice, self.alice])
        for observation_id in ids:
            self.matches[observation_id] = FaceMatch(self.alice, 0.95, None, 0.95, (11, 12, 13), (0.95,) * 3)
            self.assertFalse(self.recognize(observation_id))
        self.assertIsNone(self.canonical()["person_id"])
        self.assertEqual(self.updates, [])

    def test_operator_confirmation_and_pin_remain_canonical(self) -> None:
        ids = self.candidates([self.bob, self.bob, self.bob])
        with patch.object(self.store, "bootstrap_person_references"):
            self.store.assign(ids[0], self.alice)
        self.store.set_reference_pinned(ids[0], True)
        self.updates.clear()
        for observation_id in ids[1:]:
            self.recognize(observation_id, quality=0.95)
            self.assertEqual(self.canonical()["id"], ids[0])
        self.recognize(ids[0], quality=0.1)
        row = self.canonical()
        self.assertEqual(row["id"], ids[0])
        self.assertEqual(row["person_id"], self.alice)
        self.assertEqual(row["review_status"], "confirmed")
        self.assertTrue(row["reference_pinned"])
        self.assertEqual(self.updates, [])

    def test_rejection_on_one_crop_blocks_sibling_evidence_and_stays_visible(self) -> None:
        ids = self.candidates([self.alice, self.alice, self.alice])
        self.recognize(ids[0], quality=0.5)
        self.store.assign(ids[0], None)
        for observation_id in ids[1:]:
            self.recognize(observation_id, quality=0.95)
        self.store._refresh_unknown_recognition()
        row = self.canonical()
        self.assertEqual(row["id"], ids[0])
        self.assertEqual(row["review_status"], "rejected")
        self.assertIsNone(row["person_id"])
        self.assertIsNone(row["candidate_person_id"])
        self.assertEqual(self.updates, [])
        # Exercise the real matcher too: a sibling must inherit this track's
        # rejection, even though its own rejection table has no entry.
        self.match_patch.stop()
        references = [
            {"id": 100 + person, "person_id": person, "_embedding": np.asarray(vector, dtype=np.float32)}
            for person, vector in ((self.alice, [1.0, 0.0]), (self.bob, [0.6, 0.8]))
        ]
        with patch.object(self.store, "_reference_gallery", return_value=references), self.store._connect() as connection:
            match = self.store._match_result(connection, ids[1], np.asarray([1.0, 0.0]), "test-v1")
        self.assertEqual(match.person_id, self.bob)

    def test_deleted_person_evidence_cannot_return_before_deferred_refresh(self) -> None:
        ids = self.candidates([self.alice, self.alice])
        for observation_id in ids:
            self.recognize(observation_id)
        with patch.object(self.store, "request_match_refresh"):
            self.assertTrue(self.store.delete_person(self.alice))
        # Refinement can run before the worker refreshes old match details.
        refined_ids = self.candidates([self.bob], first_rank=3)
        self.assertIsNone(self.canonical()["candidate_person_id"])
        self.assertIsNone(self.canonical()["person_id"])
        self.assertFalse(self.recognize(refined_ids[0]))
        row = self.canonical()
        self.assertEqual(row["candidate_person_id"], self.bob)
        self.assertIsNone(row["person_id"])
        self.assertEqual(json.loads(row["consensus_json"])["agreement_count"], 1)
        self.assertEqual([update["action"] for update in self.updates], ["assigned", "cleared"])

    def test_refinement_withdraws_automatic_identity_until_new_evidence_is_terminal(self) -> None:
        ids = self.candidates([self.alice, self.alice])
        for observation_id in ids:
            self.recognize(observation_id)
        self.assertEqual(self.canonical()["person_id"], self.alice)
        self.assertEqual(len(self.updates), 1)
        refined_ids = self.candidates([self.bob, self.bob], first_rank=3)
        self.assertIsNone(self.canonical()["person_id"])
        self.assertEqual(self.updates[-1]["action"], "cleared")
        self.assertEqual(self.updates[-1]["previous_person_id"], self.alice)
        self.assertEqual(self.updates[-1]["name"], "Alice")
        self.assertEqual(self.updates[-1]["observation_id"], ids[0])
        for observation_id in refined_ids:
            self.assertFalse(self.recognize(observation_id))
        consensus = json.loads(self.canonical()["consensus_json"])
        self.assertEqual(consensus["candidate_count"], 4)
        self.assertEqual(consensus["agreement_count"], 2)
        self.assertIsNone(self.canonical()["person_id"])
        self.assertEqual(len(self.updates), 2)

    def test_refresh_limit_counts_complete_tracks_instead_of_candidate_rows(self) -> None:
        ids = []
        for event_id in (1, 2):
            track_ids = self.candidates([self.alice] * 3, event_id=event_id)
            for observation_id in track_ids:
                self.recognize(observation_id)
            ids.extend(track_ids)
        self.store.max_observations = 4
        for observation_id in ids:
            self.matches[observation_id] = FaceMatch(self.bob, 0.95, 0.1, 0.85, (21, 22, 23), (0.95,) * 3)
        self.store._refresh_unknown_recognition()
        for event_id in (1, 2):
            row = self.canonical(event_id)
            self.assertEqual(row["person_id"], self.bob)
            self.assertEqual(json.loads(row["consensus_json"])["agreement_count"], 3)

    def test_model_mismatch_withdraws_all_automatic_tracks_beyond_queue_admission(self) -> None:
        for event_id in (1, 2):
            ids = self.candidates([self.alice] * 2, event_id=event_id)
            for observation_id in ids:
                self.recognize(observation_id)
        with self.store._connect() as connection:
            connection.execute("update face_observations set embedding_model = 'old-model'")
        self.store.max_observations = 1
        self.updates.clear()
        with patch.object(self.store, "_queue_recognition") as queue:
            self.store._refresh_unknown_recognition()
        self.assertEqual(queue.call_count, 1)
        for event_id in (1, 2):
            self.assertIsNone(self.canonical(event_id)["person_id"])
        self.assertEqual([update["action"] for update in self.updates], ["cleared", "cleared"])

    def test_refresh_corrects_automatic_identity_using_raw_quality_and_committed_updates(self) -> None:
        ids = self.candidates([self.alice, self.alice])
        for observation_id in ids:
            self.recognize(observation_id)
        self.assertEqual(self.canonical()["person_id"], self.alice)
        self.assertEqual(len(self.updates), 1)
        for observation_id in ids:
            self.matches[observation_id] = FaceMatch(self.bob, 0.95, 0.1, 0.85, (21, 22, 23), (0.95,) * 3)
        self.store._refresh_unknown_recognition()
        self.assertEqual(self.canonical()["person_id"], self.bob)
        self.assertEqual(self.updates[-1]["action"], "corrected")
        self.assertEqual(self.updates[-1]["previous_person_id"], self.alice)
        self.assertEqual(self.updates[-1]["current_person_id"], self.bob)
        self.store._refresh_unknown_recognition()
        self.assertEqual(len(self.updates), 2)


if __name__ == "__main__":
    unittest.main()
