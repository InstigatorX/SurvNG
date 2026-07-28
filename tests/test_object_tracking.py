from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from survng.app.config import CameraConfig, ObjectTrackingConfig
from survng.app.events import EventStore
from survng.app.object_tracking import (
    ByteTrackObjectTracker,
    ObjectTrackerRegistry,
    ObjectTrackingSession,
)


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

    def test_center_distance_fallback_preserves_id_when_box_shape_changes(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        tracker.update(
            [detection("person", 0.9, (700, 720, 950, 1040))],
            10.0,
            confirm_new=True,
        )

        tracked = tracker.update(
            [detection("person", 0.9, (680, 800, 820, 1280))],
            10.5,
        )

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertEqual(len(tracker.summaries(10.5)), 1)

    def test_extreme_containment_does_not_merge_unrelated_person_box(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        tracker.update(
            [detection("person", 0.9, (0, 0, 2000, 1200))],
            10.0,
            confirm_new=True,
        )

        tracked = tracker.update(
            [detection("person", 0.9, (900, 500, 950, 600))],
            10.5,
        )

        self.assertEqual(tracked[0]["track_id"], 2)

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

    def test_wall_clock_correction_cannot_move_track_backwards(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        tracker.update(
            [detection("person", 0.9, (10, 10, 40, 80))],
            30.0,
            confirm_new=True,
        )

        tracker.update([detection("person", 0.9, (12, 10, 42, 80))], 29.5)

        summary = tracker.summaries(30.0)[0]
        self.assertEqual(summary["duration_seconds"], 0.0)
        self.assertEqual(summary["last_seen"], datetime.fromtimestamp(30.0, timezone.utc).isoformat())
        self.assertEqual(tracker._tracks[1].velocity, (0.0, 0.0, 0.0, 0.0))

    def test_initial_eligible_detection_below_global_threshold_is_seeded(self) -> None:
        config = self.config.model_copy(update={"low_confidence_threshold": 0.8})
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)

        tracked = tracker.update(
            [detection("person", 0.4, (10, 10, 40, 80))],
            40.0,
            confirm_new=True,
        )

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertEqual(tracker.summaries(40.0)[0]["state"], "confirmed")

    def test_delayed_initial_detection_keeps_event_time_but_starts_fresh(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        initial = detection("person", 0.9, (10, 10, 40, 80))
        initial["_tracking_first_seen_at"] = 40.0

        tracked = tracker.update([initial], 48.0, confirm_new=True)

        summary = tracker.summaries(48.0)[0]
        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertNotIn("_tracking_first_seen_at", tracked[0])
        self.assertEqual(summary["first_seen"], datetime.fromtimestamp(40.0, timezone.utc).isoformat())
        self.assertEqual(summary["last_seen"], datetime.fromtimestamp(48.0, timezone.utc).isoformat())
        self.assertEqual(summary["trajectory"], [[48.0, 25.0, 45.0]])
        self.assertEqual(summary["box_history"], [[48.0, 10, 10, 40, 80]])
        self.assertTrue(tracker.has_live_tracks(48.0))

    def test_box_history_is_bounded_and_uses_monotonic_observation_times(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        tracker.update(
            [detection("person", 0.9, (10, 10, 40, 80))],
            100.0,
            confirm_new=True,
        )
        for offset in range(1, 66):
            tracker.update(
                [detection("person", 0.9, (10 + offset, 10, 40 + offset, 80))],
                100.0 + offset * 0.1,
            )

        summary = tracker.summaries(107.0)[0]
        self.assertEqual(len(summary["trajectory"]), 60)
        self.assertEqual(len(summary["box_history"]), 60)
        self.assertEqual(summary["box_history"][-1], [106.5, 75, 10, 105, 80])
        self.assertEqual(
            [sample[0] for sample in summary["box_history"]],
            sorted(sample[0] for sample in summary["box_history"]),
        )

    def test_ignored_zone_detection_cannot_start_new_track(self) -> None:
        tracker = ByteTrackObjectTracker(self.config, high_confidence_threshold=0.7)
        ignored = detection("person", 0.9, (10, 10, 40, 80))
        ignored["incident_eligible"] = False

        tracked = tracker.update([ignored], 50.0)

        self.assertEqual(tracked, [])
        self.assertEqual(tracker.summaries(50.0), [])

    def test_bounds_total_tracks_for_noisy_detector_output(self) -> None:
        config = self.config.model_copy(update={"max_tracks_per_session": 2})
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)

        tracked = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
            detection("person", 0.9, (50, 10, 80, 80)),
            detection("person", 0.9, (90, 10, 120, 80)),
        ], 60.0, confirm_new=True)

        self.assertEqual([item["track_id"] for item in tracked], [1, 2])
        self.assertEqual(len(tracker.summaries(60.0)), 2)

    def test_reid_revives_recently_lost_person_without_persisting_embedding(self) -> None:
        config = self.config.model_copy(update={
            "reid_enabled": True,
            "reid_match_threshold": 0.8,
            "reid_max_age_seconds": 30.0,
        })
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        first = detection("person", 0.9, (10, 10, 40, 80))
        first["_tracking_embedding"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        tracker.update([first], 10.0, confirm_new=True)
        tracker.update([], 12.0)
        reappeared = detection("person", 0.9, (500, 300, 600, 700))
        reappeared["_tracking_embedding"] = np.asarray([0.99, 0.01, 0.0], dtype=np.float32)

        tracked = tracker.update([reappeared], 15.0)

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertNotIn("_tracking_embedding", tracked[0])
        self.assertEqual(len(tracker.summaries(15.0)), 1)

    def test_reid_does_not_merge_dissimilar_people(self) -> None:
        config = self.config.model_copy(update={"reid_enabled": True})
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        first = detection("person", 0.9, (10, 10, 40, 80))
        first["_tracking_embedding"] = np.asarray([1.0, 0.0], dtype=np.float32)
        tracker.update([first], 10.0, confirm_new=True)
        tracker.update([], 12.0)
        different = detection("person", 0.9, (500, 300, 600, 700))
        different["_tracking_embedding"] = np.asarray([0.0, 1.0], dtype=np.float32)

        tracked = tracker.update([different], 15.0)

        self.assertEqual(tracked[0]["track_id"], 2)

    def test_malformed_reid_embedding_is_ignored(self) -> None:
        config = self.config.model_copy(update={"reid_enabled": True})
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        initial = detection("person", 0.9, (10, 10, 40, 80))
        initial["_tracking_embedding"] = {"invalid": "vector"}

        tracked = tracker.update([initial], 10.0, confirm_new=True)

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertNotIn("_tracking_embedding", tracked[0])

    def test_reid_is_never_applied_to_non_person_tracks(self) -> None:
        config = self.config.model_copy(update={"reid_enabled": True})
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        first = detection("car", 0.9, (10, 10, 80, 60))
        first["_tracking_embedding"] = np.asarray([1.0, 0.0], dtype=np.float32)
        tracker.update([first], 10.0, confirm_new=True)
        tracker.update([], 12.0)
        second = detection("car", 0.9, (500, 300, 700, 500))
        second["_tracking_embedding"] = np.asarray([1.0, 0.0], dtype=np.float32)

        tracked = tracker.update([second], 15.0)

        self.assertEqual(tracked[0]["track_id"], 2)

    def test_vehicle_reid_revives_a_recent_car_track(self) -> None:
        config = self.config.model_copy(update={
            "vehicle_reid_enabled": True,
            "vehicle_reid_model_path": "vehicle-reid.xml",
            "vehicle_reid_match_threshold": 0.7,
        })
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        first = detection("car", 0.9, (10, 10, 80, 60))
        first["_tracking_embedding"] = np.asarray([1.0, 0.0], dtype=np.float32)
        tracker.update([first], 10.0, confirm_new=True)
        tracker.update([], 14.0)
        reappeared = detection("car", 0.9, (500, 300, 700, 500))
        reappeared["_tracking_embedding"] = np.asarray([0.99, 0.01], dtype=np.float32)

        tracked = tracker.update([reappeared], 15.0)

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertEqual(tracker.summaries(15.0)[0]["reid_matches"], 1)

    def test_lazy_reid_skips_geometry_matches_and_refreshes_periodically(self) -> None:
        config = self.config.model_copy(update={
            "vehicle_reid_enabled": True,
            "vehicle_reid_model_path": "vehicle-reid.xml",
            "reid_refresh_interval_frames": 3,
        })
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        calls: list[int] = []
        reasons: list[str] = []

        def detected(x: int) -> dict:
            item = detection("car", 0.9, (x, 10, x + 70, 60))
            item["_tracking_embedding_provider"] = lambda: (
                calls.append(x)
                or reasons.append(str(item.get("_tracking_embedding_reason")))
                or np.asarray([1.0, 0.0], dtype=np.float32)
            )
            return item

        tracker.update([detected(10)], 10.0, confirm_new=True)
        tracker.update([detected(12)], 10.5)
        tracker.update([detected(14)], 11.0)
        tracker.update([detected(16)], 11.5)

        self.assertEqual(calls, [10, 16])
        self.assertEqual(reasons, ["track_seed", "periodic_refresh"])
        self.assertEqual(tracker.summaries(11.5)[0]["observations"], 4)
        self.assertEqual(tracker.diagnostics(), {
            "association_counts": {
                "new_track": 1,
                "geometry": 3,
                "appearance_recovery": 0,
            },
            "reid_avoided_geometry_matches": 2,
            "reid_avoided_by_label": {"car": 2},
        })

    def test_lazy_reid_is_resolved_for_geometry_failure_recovery(self) -> None:
        config = self.config.model_copy(update={
            "vehicle_reid_enabled": True,
            "vehicle_reid_model_path": "vehicle-reid.xml",
            "vehicle_reid_match_threshold": 0.7,
        })
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)
        initial = detection("car", 0.9, (10, 10, 80, 60))
        initial["_tracking_embedding_provider"] = lambda: np.asarray(
            [1.0, 0.0], dtype=np.float32
        )
        tracker.update([initial], 10.0, confirm_new=True)
        tracker.update([], 14.0)
        calls = 0

        def recover() -> np.ndarray:
            nonlocal calls
            calls += 1
            self.assertEqual(
                reappeared.get("_tracking_embedding_reason"),
                "geometry_recovery",
            )
            return np.asarray([0.99, 0.01], dtype=np.float32)

        reappeared = detection("car", 0.9, (500, 300, 700, 500))
        reappeared["_tracking_embedding_provider"] = recover

        tracked = tracker.update([reappeared], 15.0)

        self.assertEqual(calls, 1)
        self.assertEqual(tracked[0]["track_id"], 1)
        summary = tracker.summaries(15.0)[0]
        self.assertEqual(summary["reid_matches"], 1)
        self.assertEqual(summary["reid_recovery_history"], [{
            "captured_at": 15.0,
            "similarity": 0.9999,
            "resumed_completed_track": True,
            "box": [500, 300, 700, 500],
        }])

    def test_upper_garage_style_vehicle_sequence_reports_saved_reid_work(self) -> None:
        config = self.config.model_copy(update={
            "vehicle_reid_enabled": True,
            "vehicle_reid_model_path": "vehicle-reid.xml",
            "vehicle_reid_match_threshold": 0.8,
            "lost_timeout_seconds": 3.0,
            "reid_refresh_interval_frames": 8,
        })
        tracker = ByteTrackObjectTracker(config, high_confidence_threshold=0.7)

        def car(box, embedding=(1.0, 0.0)) -> dict:
            item = detection("car", 0.9, box)
            item["_tracking_embedding_provider"] = lambda: np.asarray(
                embedding, dtype=np.float32
            )
            return item

        tracker.update(
            [car((3196, 1161, 3925, 1708))],
            1785257053.547,
            confirm_new=True,
        )
        startup = tracker.update(
            [car((2652, 1095, 3282, 1471))],
            1785257054.987,
        )
        tracker.update([], 1785257058.1)
        recovered = tracker.update(
            [car((900, 900, 1600, 1400), embedding=(0.99, 0.01))],
            1785257058.2,
        )

        self.assertEqual(startup[0]["track_id"], 1)
        self.assertEqual(recovered[0]["track_id"], 1)
        self.assertEqual(len(tracker.summaries(1785257058.2)), 1)
        self.assertEqual(
            tracker.diagnostics()["association_counts"],
            {"new_track": 1, "geometry": 1, "appearance_recovery": 1},
        )
        self.assertEqual(
            tracker.diagnostics()["reid_avoided_geometry_matches"],
            1,
        )

    def test_confirmed_seed_tolerates_an_initial_sampling_gap(self) -> None:
        tracker = ByteTrackObjectTracker(
            self.config.model_copy(update={"lost_timeout_seconds": 3.0}),
            high_confidence_threshold=0.7,
        )
        tracker.update([
            detection("car", 0.9, (3196, 1161, 3925, 1708)),
        ], 1785257053.547, confirm_new=True)

        tracked = tracker.update([
            detection("car", 0.9, (2652, 1095, 3282, 1471)),
        ], 1785257054.987)

        self.assertEqual(tracked[0]["track_id"], 1)
        self.assertEqual(len(tracker.summaries(1785257054.987)), 1)

    def test_tracker_registry_rejects_unknown_or_duplicate_implementations(self) -> None:
        registry = ObjectTrackerRegistry()
        registry.register("byte", ByteTrackObjectTracker)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("BYTE", ByteTrackObjectTracker)
        with self.assertRaisesRegex(ValueError, "available: byte"):
            registry.require("kalman")


class ObjectTrackingSessionTest(unittest.TestCase):
    def test_reid_recovery_telemetry_accumulates_across_sessions(self) -> None:
        persisted: dict = {}

        def update_event(_event_id, tracking, _tracked):
            persisted.update(tracking)
            return {}

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=update_event,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )
        session._reid_recovery_base = 2
        session._reid_recovery_base_by_label = {"car": 2}
        session._reid_avoided_base = 4
        session._reid_avoided_base_by_label = {"car": 4}
        tracker = SimpleNamespace(
            summaries=lambda _captured_at: [
                {"label": "car", "reid_matches": 1},
                {"label": "person", "reid_matches": 2},
            ],
            diagnostics=lambda: {
                "association_counts": {"geometry": 6},
                "reid_avoided_geometry_matches": 3,
                "reid_avoided_by_label": {"car": 3},
            },
        )

        session._persist(7, tracker, 10.0, None, 3, "active")

        self.assertEqual(session.status()["reid_recoveries"], 5)
        self.assertEqual(
            session.status()["reid_recoveries_by_label"],
            {"car": 3, "person": 2},
        )
        self.assertEqual(session.status()["reid_avoided_geometry_matches"], 7)
        self.assertEqual(session.status()["reid_avoided_by_label"], {"car": 7})
        self.assertEqual(
            persisted["reid_diagnostics"]["association_counts"],
            {"geometry": 6},
        )

    def test_lazy_annotation_defers_inference_and_records_label_telemetry(self) -> None:
        class Encoder:
            enabled = True

            def __init__(self) -> None:
                self.calls = 0

            @staticmethod
            def supports_label(label):
                return label == "car"

            def embed_for_label(self, _label, _crop):
                self.calls += 1
                return np.asarray([1.0, 0.0], dtype=np.float32)

        encoder = Encoder()
        persisted: dict = {}
        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(
                vehicle_reid_enabled=True,
                vehicle_reid_model_path="vehicle.xml",
            ),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=lambda _event_id, tracking, _tracked: (
                persisted.update(tracking) or {}
            ),
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
            appearance_encoder=encoder,
        )
        objects = [detection("car", 0.9, (5, 5, 95, 80))]

        session._annotate_appearances(
            np.zeros((100, 100, 3), dtype=np.uint8),
            objects,
            lazy=True,
        )
        self.assertEqual(encoder.calls, 0)

        tracker = ByteTrackObjectTracker(session.config, high_confidence_threshold=0.7)
        tracker.update(objects, 10.0, confirm_new=True)
        session._persist(7, tracker, 10.0, None, 1, "active")

        status = session.status()
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(status["reid_attempts"], 1)
        self.assertEqual(status["reid_successes"], 1)
        self.assertEqual(status["reid_attempts_by_label"], {"car": 1})
        self.assertEqual(status["reid_attempts_by_reason"], {"track_seed": 1})
        self.assertEqual(
            persisted["reid_diagnostics"]["inference_attempts_by_reason"],
            {"track_seed": 1},
        )

    def test_label_aware_encoder_annotates_people_and_vehicles(self) -> None:
        class Encoder:
            enabled = True

            def __init__(self) -> None:
                self.labels = []

            @staticmethod
            def supports_label(label):
                return label in {"person", "car"}

            def embed_for_label(self, label, _crop):
                self.labels.append(label)
                return np.asarray([1.0, 0.0], dtype=np.float32)

        encoder = Encoder()
        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(
                reid_enabled=True,
                reid_model_path="person.xml",
                vehicle_reid_enabled=True,
                vehicle_reid_model_path="vehicle.xml",
            ),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=lambda _event_id, _tracking, _tracked: {},
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
            appearance_encoder=encoder,
        )
        objects = [
            detection("person", 0.9, (5, 5, 30, 90)),
            detection("car", 0.8, (35, 20, 95, 80)),
            detection("dog", 0.7, (5, 5, 30, 30)),
        ]

        session._annotate_appearances(
            np.zeros((100, 100, 3), dtype=np.uint8),
            objects,
        )

        self.assertEqual(encoder.labels, ["person", "car"])
        self.assertIn("_tracking_embedding", objects[0])
        self.assertIn("_tracking_embedding", objects[1])
        self.assertNotIn("_tracking_embedding", objects[2])

    def test_reid_work_is_bounded_and_failures_are_redacted_in_status(self) -> None:
        class Encoder:
            enabled = True

            def __init__(self) -> None:
                self.calls = 0

            def embed(self, _crop):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("https://camera:secret@example.invalid/reid")
                return np.asarray([1.0, 0.0], dtype=np.float32)

        encoder = Encoder()
        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(
                reid_enabled=True,
                reid_model_path="person-reid.xml",
                reid_max_embeddings_per_frame=3,
            ),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=lambda _event_id, _tracking, _tracked: {},
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
            appearance_encoder=encoder,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        objects = [
            detection("person", 0.9 - index * 0.01, (index * 10, 10, index * 10 + 8, 80))
            for index in range(6)
        ]

        session._annotate_appearances(frame, objects)

        self.assertEqual(encoder.calls, 2)
        self.assertEqual(session.status()["reid_failures"], 1)
        self.assertNotIn("secret", session.status()["last_reid_error"])
        self.assertEqual(
            sum("_tracking_embedding" in detected for detected in objects),
            1,
        )

    def test_malformed_confidence_and_seed_time_do_not_break_tracking(self) -> None:
        tracker = ByteTrackObjectTracker(ObjectTrackingConfig(), 0.7)
        invalid = detection("person", 0.9, (10, 10, 40, 80))
        invalid["confidence"] = "invalid"
        invalid["_tracking_first_seen_at"] = float("nan")

        self.assertEqual(tracker.update([invalid], 100.0), [])

        seeded = detection("person", 0.9, (10, 10, 40, 80))
        seeded["_tracking_first_seen_at"] = float("inf")
        tracked = tracker.update([seeded], 101.0, confirm_new=True)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["track_id"], 1)
    def test_rejects_new_sessions_after_admission_is_closed(self) -> None:
        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=lambda _event_id, _tracking, _tracked: {},
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )

        self.assertFalse(session.start(
            42,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        ))
        session.set_accepting(True)
        session.stop()
        self.assertFalse(session.start(
            43,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        ))

    def test_capacity_skip_is_terminal_and_releases_worker_ownership(self) -> None:
        terminal = threading.Event()
        updates: list[dict] = []
        limiter = threading.BoundedSemaphore(1)
        self.assertTrue(limiter.acquire(blocking=False))

        def update_event(_event_id, tracking, _tracked_objects):
            updates.append(tracking)
            terminal.set()
            return {}

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(),
            detector=SimpleNamespace(config=SimpleNamespace(confidence_threshold=0.7)),
            frame_provider=lambda: None,
            update_event=update_event,
            publisher=None,
            limiter=limiter,
        )
        session.set_accepting(True)

        self.assertTrue(session.start(
            42,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        ))
        self.assertTrue(terminal.wait(1.0))
        deadline = time.monotonic() + 1.0
        while session.running() and time.monotonic() < deadline:
            time.sleep(0.01)
        limiter.release()

        self.assertEqual(updates[-1]["state"], "skipped_capacity")
        self.assertFalse(session.running())

    def test_does_not_reprocess_same_captured_frame(self) -> None:
        duplicate_seen = threading.Event()
        provider_calls = 0

        class Detector:
            config = SimpleNamespace(confidence_threshold=0.7)

            def __init__(self) -> None:
                self.calls = 0

            def detect(self, _frame, confidence_threshold=None):
                self.calls += 1
                return [detection("person", 0.8, (12, 10, 42, 80))]

        detector = Detector()
        captured_at = time.time()
        frame_token = time.monotonic()

        def frame_provider():
            nonlocal provider_calls
            provider_calls += 1
            if provider_calls >= 2:
                duplicate_seen.set()
            return np.zeros((100, 100, 3), dtype=np.uint8), captured_at, frame_token

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=detector,
            frame_provider=frame_provider,
            update_event=lambda _event_id, _tracking, _tracked: {},
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )
        session.set_accepting(True)

        self.assertTrue(session.start(
            42,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        ))
        self.assertTrue(duplicate_seen.wait(2.0))
        session.stop()

        self.assertEqual(detector.calls, 1)

    def test_does_not_restart_when_admission_closes_during_session_transition(self) -> None:
        detector_entered = threading.Event()
        release_detector = threading.Event()

        class Detector:
            config = SimpleNamespace(confidence_threshold=0.7)

            def detect(self, _frame, confidence_threshold=None):
                detector_entered.set()
                release_detector.wait(2.0)
                return [detection("person", 0.8, (12, 10, 42, 80))]

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=Detector(),
            frame_provider=lambda: (
                np.zeros((100, 100, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            ),
            update_event=lambda _event_id, _tracking, _tracked: {},
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )
        session.set_accepting(True)
        self.assertTrue(session.start(
            42,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        ))
        self.assertTrue(detector_entered.wait(1.0))

        start_result: list[bool] = []
        replacement = threading.Thread(target=lambda: start_result.append(session.start(
            43,
            datetime.now(timezone.utc),
            [detection("person", 0.9, (10, 10, 40, 80))],
        )))
        replacement.start()
        deadline = time.monotonic() + 1.0
        while not session._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        closer = threading.Thread(target=lambda: session.set_accepting(False))
        closer.start()
        deadline = time.monotonic() + 1.0
        while session.status()["accepting"] and time.monotonic() < deadline:
            time.sleep(0.01)
        release_detector.set()
        replacement.join(2.0)
        closer.join(2.0)

        self.assertEqual(start_result, [False])
        self.assertFalse(session.running())

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
            return {}

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=detector,
            frame_provider=lambda: (
                np.zeros((100, 100, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            ),
            update_event=update_event,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )
        session.set_accepting(True)

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

    def test_detector_failures_persist_terminal_failure_state(self) -> None:
        terminal = threading.Event()
        updates: list[dict] = []

        class Detector:
            config = SimpleNamespace(confidence_threshold=0.7)

            def detect(self, _frame, confidence_threshold=None):
                return [{"status": "inference_error", "error": "GPU unavailable"}]

        def update_event(_event_id, tracking, _tracked_objects):
            updates.append(tracking)
            if tracking.get("state") == "failed":
                terminal.set()
            return {}

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=Detector(),
            frame_provider=lambda: (
                np.zeros((100, 100, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            ),
            update_event=update_event,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
            tracker_registry=(registry := ObjectTrackerRegistry()),
        )
        registry.register("survng_hybrid", ByteTrackObjectTracker)
        session.set_accepting(True)

        with self.assertLogs("survng.app.object_tracking", level="ERROR"):
            self.assertTrue(session.start(
                42,
                datetime.now(timezone.utc),
                [detection("person", 0.9, (10, 10, 40, 80))],
            ))
            self.assertTrue(terminal.wait(2.0))
            session.stop()

        self.assertEqual(updates[-1]["state"], "failed")
        self.assertIn("GPU unavailable", updates[-1]["error"])
        self.assertFalse(session.status()["active"])

    def test_failure_status_redacts_credentials(self) -> None:
        terminal = threading.Event()

        class Detector:
            config = SimpleNamespace(confidence_threshold=0.7)

            def detect(self, _frame, confidence_threshold=None):
                raise RuntimeError("rtsp://camera:secret@example.invalid/live")

        def update_event(_event_id, tracking, _tracked_objects):
            if tracking.get("state") == "failed":
                terminal.set()
            return {}

        session = ObjectTrackingSession(
            camera=CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
            config=ObjectTrackingConfig(sample_fps=5.0, max_session_seconds=3.0),
            detector=Detector(),
            frame_provider=lambda: (
                np.zeros((100, 100, 3), dtype=np.uint8),
                time.time(),
                time.monotonic(),
            ),
            update_event=update_event,
            publisher=None,
            limiter=threading.BoundedSemaphore(1),
        )
        session.set_accepting(True)

        with self.assertLogs("survng.app.object_tracking", level="ERROR"):
            self.assertTrue(session.start(
                43,
                datetime.now(timezone.utc),
                [detection("person", 0.9, (10, 10, 40, 80))],
            ))
            self.assertTrue(terminal.wait(2.0))
            session.stop()

        self.assertNotIn("secret", session.status()["last_error"])
        self.assertIn("camera:***@", session.status()["last_error"])


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
                {
                    "state": "complete",
                    "tracks": [{
                        "track_id": 7,
                        "reid_recovery_history": [{
                            "captured_at": 10.0,
                            "similarity": 0.91,
                            "box": [10, 10, 40, 80],
                        }],
                    }],
                    "reid_diagnostics": {
                        "inference_attempts": 2,
                        "reid_avoided_geometry_matches": 5,
                    },
                },
                [{
                    **detection("person", 0.9, (10, 10, 40, 80)),
                    "track_id": 99,
                }],
            )

        self.assertIsNotNone(updated)
        objects = json.loads(str(updated["objects_json"]))
        self.assertEqual(objects[0]["track_id"], 7)
        tracking = [item for item in objects if item.get("status") == "object_tracking"]
        self.assertEqual(len(tracking), 1)
        self.assertEqual(tracking[0]["object_tracking"]["state"], "complete")
        self.assertEqual(
            tracking[0]["object_tracking"]["reid_diagnostics"]["reid_avoided_geometry_matches"],
            5,
        )
        self.assertEqual(
            tracking[0]["object_tracking"]["tracks"][0]["reid_recovery_history"][0]["similarity"],
            0.91,
        )


if __name__ == "__main__":
    unittest.main()
