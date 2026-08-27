"""Tests for ReID training review APIs and identity merge/split."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from survng.app.config import ImageStorageConfig
from survng.app.image_storage import DurableImageWriter
from survng.app.reid_training import ReidTrainingReviewService, ReidTrainingStore


def _write_crop(store: ReidTrainingStore, writer: DurableImageWriter, **fields) -> str:
    event_id = int(fields["event_id"])
    track_id = int(fields["track_id"])
    camera_id = str(fields["camera_id"])
    reason = str(fields.get("selection_reason") or "start")
    sample_id = str(fields.get("sample_id") or f"e{event_id}-t{track_id}-{reason}")
    frame = np.full((120, 64, 3), 90, dtype=np.uint8)
    frame[20:100, 10:54] = (40, 120, 200)
    path = writer.write(
        store.crops_root / camera_id / str(event_id),
        f"{sample_id}",
        frame,
    )
    assert path is not None
    person_id = fields.get("assigned_person_id")
    if person_id is None:
        person_id = store.create_identity()
    store.insert_sample({
        "sample_id": sample_id,
        "event_id": event_id,
        "camera_id": camera_id,
        "track_id": track_id,
        "captured_at": fields.get("captured_at") or "2026-08-01T12:00:00+00:00",
        "bounding_box": {"x1": 10, "y1": 20, "x2": 54, "y2": 100},
        "detection_confidence": 0.9,
        "crop_path": str(path.relative_to(store.storage_dir)),
        "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        "model_kind": "person",
        "model_fingerprint": "fp",
        "assigned_person_id": person_id,
        "assignment_source": "track",
        "assignment_confidence": 1.0,
        "review_status": fields.get("review_status") or "auto",
        "selection_reason": reason,
        "quality_score": 0.7,
    })
    return sample_id


class ReidTrainingReviewTests(unittest.TestCase):
    def test_confirm_same_merges_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            writer = DurableImageWriter(ImageStorageConfig(format="jpeg", quality=90))
            left_person = store.create_identity()
            right_person = store.create_identity()
            _write_crop(
                store,
                writer,
                event_id=1,
                track_id=2,
                camera_id="gate",
                assigned_person_id=left_person,
                sample_id="left-start",
            )
            _write_crop(
                store,
                writer,
                event_id=3,
                track_id=4,
                camera_id="foyer",
                assigned_person_id=right_person,
                sample_id="right-start",
            )

            def matches(_event_id, **_kwargs):
                return [{
                    "event_id": 3,
                    "anchor_track_id": 2,
                    "candidate_track_id": 4,
                    "anchor_label": "person",
                    "candidate_label": "person",
                    "similarity": 0.88,
                    "threshold": 0.7,
                    "visually_similar": True,
                }]

            service = ReidTrainingReviewService(store, matches)
            queue = service.review_queue(limit=5)
            self.assertEqual(len(queue["hard_pairs"]), 1)
            result = service.apply_review({
                "action": "confirm_same",
                "left_event_id": 1,
                "left_track_id": 2,
                "right_event_id": 3,
                "right_track_id": 4,
                "similarity": 0.88,
            })
            self.assertEqual(result["action"], "confirm_same")
            self.assertEqual(result["person_id"], left_person)
            left = store.samples_for_track(1, 2)
            right = store.samples_for_track(3, 4)
            self.assertEqual(left[0]["assigned_person_id"], left_person)
            self.assertEqual(right[0]["assigned_person_id"], left_person)
            self.assertEqual(left[0]["assignment_source"], "manual")
            self.assertTrue(store.pair_reviewed(1, 2, 3, 4))
            self.assertEqual(service.review_queue(limit=5)["hard_pairs"], [])

    def test_mark_different_splits_shared_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            writer = DurableImageWriter(ImageStorageConfig(format="jpeg", quality=90))
            shared = store.create_identity()
            _write_crop(
                store,
                writer,
                event_id=10,
                track_id=1,
                camera_id="gate",
                assigned_person_id=shared,
                sample_id="a",
            )
            _write_crop(
                store,
                writer,
                event_id=11,
                track_id=2,
                camera_id="drive",
                assigned_person_id=shared,
                sample_id="b",
            )
            service = ReidTrainingReviewService(store, lambda *_a, **_k: [])
            result = service.apply_review({
                "action": "mark_different",
                "left_event_id": 10,
                "left_track_id": 1,
                "right_event_id": 11,
                "right_track_id": 2,
            })
            self.assertEqual(result["action"], "mark_different")
            self.assertNotEqual(result["left_person_id"], result["right_person_id"])
            left = store.samples_for_track(10, 1)[0]
            right = store.samples_for_track(11, 2)[0]
            self.assertEqual(left["assigned_person_id"], result["left_person_id"])
            self.assertEqual(right["assigned_person_id"], result["right_person_id"])

    def test_crop_endpoint_serves_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            writer = DurableImageWriter(ImageStorageConfig(format="jpeg", quality=90))
            sample_id = _write_crop(
                store,
                writer,
                event_id=5,
                track_id=9,
                camera_id="gate",
                sample_id="crop-test",
            )
            path = store.resolve_crop_path(sample_id)
            self.assertTrue(path.is_file())
            image = cv2.imread(str(path))
            self.assertIsNotNone(image)
            listed = store.list_samples(limit=10)
            self.assertEqual(listed[0]["sample_id"], sample_id)
            self.assertTrue(listed[0]["crop_url"].endswith("/crop.jpg"))


if __name__ == "__main__":
    unittest.main()
