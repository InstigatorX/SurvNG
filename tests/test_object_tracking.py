from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.events import EventStore
from survng.app.object_tracking import ByteTrackObjectTracker, ObjectTrackingSession


def detection(
    label: str,
    confidence: float,
    box: tuple[float, float, float, float],
) -> dict:
    return {
        "label": label,
        "confidence": confidence,
        "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "incident_eligible": True,
    }


class ByteTrackObjectTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ObjectTrackingConfig(
            min_confirmations=2,
            low_confidence_threshold=0.25,
            match_iou_threshold=0.2,
            lost_timeout_seconds=1.0,
        )

    def test_preserves_unique_ids_for_multiple_moving_objects(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)

        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
            detection("person", 0.85, (100, 10, 130, 80)),
        ], 10.0, confirm_new=True)
        second = tracker.update([
            detection("person", 0.88, (15, 10, 45, 80)),
            detection("person", 0.82, (95, 10, 125, 80)),
        ], 10.5)

        self.assertEqual([item["track_id"] for item in first], [1, 2])
        self.assertEqual([item["track_id"] for item in second], [1, 2])
        summaries = tracker.summaries(10.5)
        self.assertEqual([item["observations"] for item in summaries], [2, 2])
        self.assertTrue(all(item["state"] == "confirmed" for item in summaries))

    def test_low_confidence_detection_recovers_track_but_does_not_start_one(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        initial = tracker.update(
            [detection("car", 0.9, (10, 10, 80, 60))],
            20.0,
            confirm_new=True,
        )
        recovered = tracker.update(
            [
                detection("car", 0.3, (12, 10, 82, 60)),
                detection("person", 0.3, (100, 10, 130, 80)),
            ],
            20.5,
        )

        self.assertEqual(initial[0]["track_id"], recovered[0]["track_id"])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(len(tracker.summaries(20.5)), 1)

    def test_expires_track_after_lost_timeout(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        tracker.update([detection("car", 0.9, (10, 10, 80, 60))], 30.0, confirm_new=True)

        tracker.update([], 31.1)

        self.assertFalse(tracker.has_live_tracks(31.1))
        self.assertEqual(tracker.summaries(31.1)[0]["state"], "lost")

    def test_initial_eligible_detection_below_global_threshold_is_seeded(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)

        tracked = tracker.update(
            [detection("person", 0.4, (10, 10, 40, 80))],
            40.0,
            confirm_new=True,
        )

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertEqual(tracker.summaries(40.0)[0]["state"], "confirmed")

    def test_ignored_zone_detection_cannot_start_new_track(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        ignored = detection("person", 0.9, (10, 10, 40, 80))
        ignored["incident_eligible"] = False

        tracked = tracker.update([ignored], 50.0)

        self.assertEqual(tracked, [])
        self.assertEqual(tracker.summaries(50.0), [])


class ObjectTrackingSessionTest(unittest.TestCase):
    def test_processes_live_frames_and_finishes_cleanly(self) -> None:
        update_ready = threading.Event()
        updates: list[dict] = []

        class Detector:
            config = SimpleNamespace(confidence_threshold=0.7)

            def detect(self, _frame, confidence_threshold=None):
                self.threshold = confidence_threshold
                return [detection("person", 0.8, (12, 10, 42, 80))]

        detector = Detector()

        def update_event(_event_id, tracking, _tracked_objects):
            updates.append(tracking)
            if int(tracking.get("frames_processed") or 0) >= 1:
                update_ready.set()

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=detector,
            frame_provider=lambda: np.zeros((100, 100, 3), dtype=np.uint8),
            update_event=update_event,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )

        started = session.start(
            42,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        )
        self.assertTrue(started)
        self.assertTrue(update_ready.wait(2.0))
        session.stop()

        self.assertEqual(detector.threshold, 0.25)
        self.assertEqual(updates[-1]["state"], "complete")
        self.assertGreaterEqual(updates[-1]["frames_processed"], 1)
        self.assertFalse(session.status()["active"])


class ObjectTrackingPersistenceTest(unittest.TestCase):
    def test_replaces_tracking_metadata_and_assigns_initial_track_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event(
                camera_id="gate",
                kind="motion",
                objects_json=json.dumps([
                    detection("person", 0.9, (10, 10, 40, 80)),
                    {"status": "motion_qualification", "motion_qualification": {}},
                ]),
            )
            tracked = [{
                **detection("person", 0.9, (10, 10, 40, 80)),
                "track_id": 7,
                "track_state": "confirmed",
                "track_observations": 1,
            }]

            store.update_object_tracking(
                int(event["id"]),
                {"state": "active", "tracks": [{"track_id": 7}]},
                tracked,
            )
            updated = store.update_object_tracking(
                int(event["id"]),
                {"state": "complete", "tracks": [{"track_id": 7}]},
            )

        self.assertIsNotNone(updated)
        objects = json.loads(str(updated["objects_json"]))
        self.assertEqual(objects[0]["track_id"], 7)
        tracking = [item for item in objects if item.get("status") == "object_tracking"]
        self.assertEqual(len(tracking), 1)
        self.assertEqual(tracking[0]["object_tracking"]["state"], "complete")


if __name__ == "__main__":
    unittest.main()
