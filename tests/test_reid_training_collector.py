"""Tests for optional ReID domain-adaptation training crop collection."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from survng.app.config import CameraConfig, ImageStorageConfig, ObjectTrackingConfig
from survng.app.image_storage import DurableImageWriter
from survng.app.object_track.bytetrack import ByteTrackObjectTracker
from survng.app.object_track.session import ObjectTrackingSession
from survng.app.reid_training import (
    ReidTrainingBuffer,
    ReidTrainingCollector,
    ReidTrainingStore,
)


def _person_frame(color: tuple[int, int, int] = (40, 80, 120)) -> np.ndarray:
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    frame[40:200, 40:120] = color
    # Add texture so blur/quality heuristics accept the crop.
    noise = np.random.default_rng(0).integers(0, 40, size=frame.shape, dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class ReidTrainingStoreTests(unittest.TestCase):
    def test_creates_anonymous_identity_and_sample(self) -> None:
        root = Path(self.id().replace(".", "_"))
        # Use TemporaryDirectory via setUp would be nicer; keep local for isolation.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            person_id = store.create_identity()
            self.assertEqual(person_id, 1)
            sample_id = store.insert_sample({
                "sample_id": "e1-t1-start",
                "event_id": 1,
                "camera_id": "gate",
                "track_id": 1,
                "captured_at": "2026-01-01T00:00:00+00:00",
                "bounding_box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                "detection_confidence": 0.9,
                "crop_path": "reid_training/gate/1/crop.jpg",
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "model_kind": "person",
                "model_fingerprint": "abc",
                "assigned_person_id": person_id,
                "assignment_source": "track",
                "assignment_confidence": 1.0,
                "review_status": "auto",
                "selection_reason": "start",
                "quality_score": 0.5,
            })
            self.assertIsNotNone(sample_id)
            status = store.status()
            self.assertEqual(status["samples"], 1)
            self.assertEqual(status["identities"], 1)
            self.assertEqual(store.count_samples(event_id=1), 1)


class ReidTrainingCollectorTests(unittest.TestCase):
    def test_selects_representative_crops_and_assigns_track_identity(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            writer = DurableImageWriter(ImageStorageConfig(format="jpeg", quality=90))
            config = ObjectTrackingConfig(
                reid_training_collector_enabled=True,
                reid_training_min_samples_per_track=3,
                reid_training_max_samples_per_track=5,
                reid_training_min_crop_pixels=1000,
                reid_training_min_confidence=0.2,
                reid_training_min_quality=0.0,
            )
            collector = ReidTrainingCollector(store, writer, config)
            buffer = ReidTrainingBuffer()
            frame = _person_frame()
            detections = []
            tracked = []
            for index, epoch in enumerate((10.0, 10.5, 11.0, 11.5, 12.0)):
                box = {"x1": 40, "y1": 40, "x2": 120, "y2": 200}
                detected = {
                    "label": "person",
                    "confidence": 0.5 + index * 0.05,
                    "box": box,
                    "_tracking_embedding": np.asarray(
                        [1.0 if index % 2 == 0 else 0.0, 0.0 if index % 2 == 0 else 1.0],
                        dtype=np.float32,
                    ),
                }
                detections.append(detected)
                tracked.append({
                    "label": "person",
                    "confidence": detected["confidence"],
                    "box": box,
                    "track_id": 7,
                })
                collector.retain_from_frame(
                    buffer,
                    frame if index != 2 else _person_frame((10, 200, 30)),
                    tracked[-1:],
                    detections[-1:],
                    epoch,
                )
            self.assertEqual(len(buffer.by_track[7]), 5)
            stored = collector.flush(
                buffer,
                event_id=42,
                camera_id="gate",
                tracks=[{
                    "track_id": 7,
                    "label": "person",
                    "state": "confirmed",
                }],
                model_identity_for_label=lambda _label: {
                    "model_kind": "person",
                    "model_fingerprint": "fp-test",
                },
            )
            self.assertGreaterEqual(stored, 3)
            self.assertLessEqual(stored, 5)
            self.assertEqual(store.count_samples(event_id=42), stored)
            crop_dir = base / "storage" / "reid_training" / "gate" / "42"
            self.assertTrue(crop_dir.is_dir())
            self.assertGreaterEqual(len(list(crop_dir.glob("*"))), stored)


class ReidTrainingSessionHookTests(unittest.TestCase):
    def test_completed_session_flushes_training_crops(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = ReidTrainingStore(base / "db", base / "storage")
            writer = DurableImageWriter(ImageStorageConfig(format="jpeg", quality=90))
            config = ObjectTrackingConfig(
                reid_training_collector_enabled=True,
                reid_training_min_samples_per_track=1,
                reid_training_max_samples_per_track=3,
                reid_training_min_crop_pixels=1000,
                reid_training_min_confidence=0.2,
                reid_training_min_quality=0.0,
                min_confirmations=1,
            )
            collector = ReidTrainingCollector(store, writer, config)
            session = ObjectTrackingSession(
                camera=CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://example.invalid/main",
                ),
                config=config,
                detector=SimpleNamespace(
                    config=SimpleNamespace(confidence_threshold=0.7)
                ),
                frame_provider=lambda: None,
                update_event=lambda *_args: {},
                publisher=None,
                limiter=threading.BoundedSemaphore(1),
                training_crop_collector=collector,
            )
            tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
            frame = _person_frame()
            detected = {
                "label": "person",
                "confidence": 0.92,
                "box": {"x1": 40, "y1": 40, "x2": 120, "y2": 200},
                "_tracking_embedding": np.asarray([0.0, 1.0], dtype=np.float32),
            }
            tracked = tracker.update([detected], 10.0, confirm_new=True)
            session._retain_training_crops(frame, tracked, [detected], 10.0)
            session._persist(9, tracker, 11.0, None, 1, "complete")
            self.assertGreaterEqual(session.status()["reid_training_crops_stored"], 1)
            self.assertEqual(store.count_samples(event_id=9), session.status()["reid_training_crops_stored"])


if __name__ == "__main__":
    unittest.main()
