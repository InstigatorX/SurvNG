from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from survng.app.appearance_index import AppearanceIndex
from survng.app.events import EventStore
from survng.app.faces import FaceStore


class PeopleIdentityFusionStoreTest(unittest.TestCase):
    def test_body_embedding_reinforces_face_suggestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            events = EventStore(root)
            event = events.add_event("gate", "motion", created_at="2026-01-01T00:00:00+00:00")
            appearance = AppearanceIndex(events.db_path)
            recognizer = SimpleNamespace(
                enabled=True,
                config=SimpleNamespace(
                    face_min_size=4,
                    face_max_references=8,
                    face_match_threshold=0.30,
                    face_auto_identify_enabled=False,
                    face_auto_identify_threshold=0.55,
                    face_auto_identify_margin=0.12,
                    people_body_match_threshold=0.70,
                    people_body_reinforce_threshold=0.75,
                    people_fused_match_threshold=0.45,
                    people_fused_auto_threshold=0.62,
                    people_fused_auto_margin=0.10,
                    people_modality_disagreement_gap=0.12,
                    people_face_fusion_weight=0.65,
                    people_body_fusion_weight=0.35,
                ),
                status=lambda: {
                    "ready": True,
                    "model_fingerprint": "face-v1",
                    "embedding_size": 2,
                },
                embed=lambda _face: np.asarray([1.0, 0.0], dtype=np.float32),
            )
            store = FaceStore(
                root,
                recognizer=recognizer,
                start_recognition=False,
                appearance_index=appearance,
            )
            person = store.create_person("Alex")["id"]
            # Seed confirmed face + body gallery references.
            for index in range(3):
                path = root / f"ref-face-{index}.jpg"
                self.assertTrue(cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8)))
                with store._lock, store._connect() as connection:
                    connection.execute(
                        """
                        insert into face_observations (
                            event_id, object_index, person_id, camera_id, snapshot_path,
                            box_json, confidence, observed_at, created_at, review_status,
                            embedding_blob, embedding_model, body_embedding_blob,
                            body_embedding_model, quality_score, canonical, recognition_pending,
                            recognition_outcome
                        ) values (?, ?, ?, 'gate', ?, ?, 0.9, ?, ?, 'confirmed', ?, 'face-v1', ?,
                            'reid-v1', 0.8, 1, 0, 'embedded')
                        """,
                        (
                            900 + index,
                            0,
                            person,
                            str(path),
                            '{"x1":0,"y1":0,"x2":40,"y2":40}',
                            f"2026-01-01T00:00:0{index}+00:00",
                            f"2026-01-01T00:00:0{index}+00:00",
                            np.asarray([1.0, 0.0], dtype=np.float32).tobytes(),
                            np.asarray([0.0, 1.0], dtype=np.float32).tobytes(),
                        ),
                    )
            store._invalidate_reference_gallery()
            appearance.replace_event(
                int(event["id"]),
                "gate",
                [
                    {
                        "track_id": 42,
                        "label": "person",
                        "model_kind": "person",
                        "model_fingerprint": "reid-v1",
                        "embedding": [0.0, 1.0],
                        "match_threshold": 0.7,
                        "quality": 0.9,
                        "observation_count": 3,
                        "created_at": "2026-01-01T00:01:00+00:00",
                    }
                ],
            )
            probe = root / "probe.jpg"
            self.assertTrue(cv2.imwrite(str(probe), np.zeros((64, 64, 3), dtype=np.uint8)))
            store.ingest_candidates(
                int(event["id"]),
                "gate",
                "2026-01-01T00:01:00+00:00",
                [
                    {
                        "snapshot_path": str(probe),
                        "box": {"x1": 0, "y1": 0, "x2": 40, "y2": 40},
                        "confidence": 0.9,
                        "track_id": "face-1",
                        "person_track_id": "42",
                        "rank": 1,
                        "offset_seconds": 0.0,
                        "quality_score": 0.8,
                    }
                ],
            )
            with store._connect() as connection:
                observation_id = int(
                    connection.execute(
                        "select id from face_observations where event_id = ?",
                        (int(event["id"]),),
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
            ):
                store._recognize_observation(observation_id)
            observation = store.observation(observation_id)
            self.assertIsNotNone(observation)
            assert observation is not None
            self.assertEqual(observation["candidate_person_id"], person)
            self.assertEqual(observation["modality"], "fused")
            self.assertGreaterEqual(float(observation["body_score"] or 0), 0.7)
            self.assertEqual(observation["match_details"]["fusion_reason"], "fused_agreement")


if __name__ == "__main__":
    unittest.main()
