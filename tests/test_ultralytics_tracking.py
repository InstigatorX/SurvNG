from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from survng.app.config import ObjectTrackingConfig


ULTRALYTICS_AVAILABLE = (
    importlib.util.find_spec("ultralytics") is not None
    and importlib.util.find_spec("lap") is not None
)

if ULTRALYTICS_AVAILABLE:
    from survng.app.ultralytics_tracking import UltralyticsDeepOCSortObjectTracker


def detection(
    label: str,
    confidence: float,
    box: tuple[float, float, float, float],
    embedding: tuple[float, ...] | None = None,
) -> dict:
    item = {
        "label": label,
        "confidence": confidence,
        "box": dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
        "incident_eligible": True,
    }
    if embedding is not None:
        item["_tracking_embedding"] = np.asarray(embedding, dtype=np.float32)
    return item


@unittest.skipUnless(ULTRALYTICS_AVAILABLE, "optional Ultralytics tracker is not installed")
class UltralyticsDeepOCSortObjectTrackerTest(unittest.TestCase):
    def config(self, **updates) -> ObjectTrackingConfig:
        return ObjectTrackingConfig(
            lost_timeout_seconds=1.0,
            **updates,
        )

    def test_preserves_ids_and_never_associates_across_classes(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(), 0.7)
        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
            detection("car", 0.9, (100, 10, 180, 80)),
        ], 10.0, confirm_new=True)
        crossed = [
            detection("person", 0.9, (100, 10, 180, 80)),
            detection("car", 0.9, (10, 10, 40, 80)),
        ]
        second = tracker.update(crossed, 10.5)
        third = tracker.update(crossed, 11.0)

        first_ids = {item["label"]: item["track_id"] for item in first}
        third_ids = {item["label"]: item["track_id"] for item in third}
        self.assertEqual(second, [])
        self.assertNotEqual(third_ids["person"], first_ids["car"])
        self.assertNotEqual(third_ids["car"], first_ids["person"])

    def test_persists_sampled_boxes_for_video_review(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(), 0.7)
        tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
        ], 10.0, confirm_new=True)
        tracker.update([
            detection("person", 0.9, (12, 10, 42, 80)),
        ], 10.5)

        summary = tracker.summaries(10.5)[0]
        self.assertEqual(summary["box_history"][0], [10.0, 10, 10, 40, 80])
        self.assertEqual(summary["box_history"][-1], [10.5, 12, 10, 42, 80])

    def test_passes_survng_embeddings_to_deep_ocsort(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
        ), 0.7)

        tracker.update([
            detection("person", 0.9, (10, 10, 40, 80), (3.0, 4.0)),
        ], 10.0, confirm_new=True)

        native = tracker._tracker.tracked_stracks[0]
        np.testing.assert_allclose(native.curr_feat, np.asarray([0.6, 0.8], dtype=np.float32))

    def test_reid_can_recover_a_far_person_when_proximity_is_disabled(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
        ), 0.7)
        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80), (1.0, 0.0)),
        ], 10.0, confirm_new=True)
        tracker.update([], 10.5)
        recovered = tracker.update([
            detection("person", 0.9, (500, 300, 600, 700), (0.99, 0.01)),
        ], 11.0)

        self.assertEqual(recovered[0]["track_id"], first[0]["track_id"])
        self.assertNotIn("_tracking_embedding", recovered[0])

    def test_dissimilar_person_starts_a_new_track(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
        ), 0.7)
        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80), (1.0, 0.0)),
        ], 10.0, confirm_new=True)
        tracker.update([], 10.5)
        different = detection("person", 0.9, (500, 300, 600, 700), (0.0, 1.0))
        self.assertEqual(tracker.update([different], 11.0), [])
        second = tracker.update([different], 11.5)

        self.assertNotEqual(second[0]["track_id"], first[0]["track_id"])

    def test_reid_uses_survng_direct_cosine_similarity_threshold(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
            reid_match_threshold=0.82,
        ), 0.7)
        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80), (1.0, 0.0)),
        ], 10.0, confirm_new=True)
        tracker.update([], 10.5)
        borderline = detection(
            "person",
            0.9,
            (500, 300, 600, 700),
            (0.7, 0.71414284),
        )
        self.assertEqual(tracker.update([borderline], 11.0), [])
        second = tracker.update([borderline], 11.5)

        self.assertNotEqual(second[0]["track_id"], first[0]["track_id"])

    def test_reid_never_recovers_beyond_configured_wall_clock_age(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(
            reid_enabled=True,
            reid_model_path="person-reid.xml",
            reid_max_age_seconds=1.0,
        ), 0.7)
        first = tracker.update([
            detection("person", 0.9, (10, 10, 40, 80), (1.0, 0.0)),
        ], 10.0, confirm_new=True)
        tracker.update([], 10.5)
        reappeared = detection(
            "person",
            0.9,
            (500, 300, 600, 700),
            (1.0, 0.0),
        )
        self.assertEqual(tracker.update([reappeared], 11.1), [])
        second = tracker.update([reappeared], 11.6)

        self.assertNotEqual(second[0]["track_id"], first[0]["track_id"])

    def test_ignored_detection_does_not_create_a_track(self) -> None:
        tracker = UltralyticsDeepOCSortObjectTracker(self.config(), 0.7)
        ignored = detection("person", 0.9, (10, 10, 40, 80))
        ignored["incident_eligible"] = False

        self.assertEqual(tracker.update([ignored], 10.0, confirm_new=True), [])
        self.assertEqual(tracker.summaries(10.0), [])

    def test_concurrent_tracker_instances_have_independent_id_counters(self) -> None:
        first_tracker = UltralyticsDeepOCSortObjectTracker(self.config(), 0.7)
        first = first_tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
        ], 10.0, confirm_new=True)
        second_tracker = UltralyticsDeepOCSortObjectTracker(self.config(), 0.7)
        second = second_tracker.update([
            detection("person", 0.9, (10, 10, 40, 80)),
        ], 10.0, confirm_new=True)

        new_object = detection("car", 0.9, (200, 10, 280, 80))
        first_tracker.update([
            detection("person", 0.9, (12, 10, 42, 80)),
            new_object,
        ], 10.5)
        later = first_tracker.update([
            detection("person", 0.9, (14, 10, 44, 80)),
            new_object,
        ], 11.0)

        self.assertEqual(first[0]["track_id"], 1)
        self.assertEqual(second[0]["track_id"], 1)
        self.assertEqual(
            {item["label"]: item["track_id"] for item in later},
            {"person": 1, "car": 2},
        )


if __name__ == "__main__":
    unittest.main()
