from __future__ import annotations

import unittest

import numpy as np

from survng.app.config import ObjectTrackingConfig
from survng.app.object_track.registry import build_builtin_object_tracker_registry
from survng.app.object_track.sparse_identity import SparseIdentityObjectTracker
from survng.app.tracking_evaluation import identity_metrics, promotion_gate, replay_digest
from tests import test_object_tracking as existing_contract


def detection(box, confidence=0.9, *, embedding=None, label="person"):
    item = existing_contract.detection(label, confidence, box)
    if embedding is not None:
        item["_tracking_embedding"] = np.asarray(embedding, dtype=np.float32)
    return item


def unit(vector):
    array = np.asarray(vector, dtype=np.float32)
    return array / max(1e-6, float(np.linalg.norm(array)))


class SparseIdentityTrackerTest(unittest.TestCase):
    def tracker(self, **overrides) -> SparseIdentityObjectTracker:
        settings = {
            "min_confirmations": 1,
            "reid_enabled": True,
            "reid_model_path": "person-reid.xml",
            "reid_match_threshold": 0.7,
            "lost_timeout_seconds": 1.0,
            "reid_max_age_seconds": 30.0,
        }
        settings.update(overrides)
        return SparseIdentityObjectTracker(ObjectTrackingConfig(**settings), 0.7)

    def test_registry_exposes_offline_sparse_identity(self) -> None:
        tracker = build_builtin_object_tracker_registry().create(
            "survng_sparse_identity",
            ObjectTrackingConfig(reid_enabled=True, reid_model_path="person.xml"),
            0.7,
        )
        self.assertIsInstance(tracker, SparseIdentityObjectTracker)

    def test_production_config_can_select_sparse_identity(self) -> None:
        config = ObjectTrackingConfig(
            implementation="survng_sparse_identity",
            reid_enabled=True,
            reid_model_path="person.xml",
        )
        self.assertEqual(config.implementation, "survng_sparse_identity")
        self.assertEqual(ObjectTrackingConfig().implementation, "survng_sparse_identity")

    def test_short_occlusion_resumes_same_track(self) -> None:
        tracker = self.tracker(lost_timeout_seconds=0.5)
        emb = unit([1.0, 0.0, 0.0, 0.0])
        first = tracker.update([detection((10, 10, 40, 80), embedding=emb)], 10.0, confirm_new=True)
        self.assertEqual(first[0]["track_id"], 1)
        tracker.update([], 10.6)
        self.assertIn(1, tracker._completed)
        resumed = tracker.update([detection((12, 12, 42, 82), embedding=emb)], 11.2)
        self.assertEqual([item["track_id"] for item in resumed], [1])
        self.assertGreaterEqual(tracker._tracks[1].reid_matches, 1)

    def test_crossing_without_swap_when_appearance_disambiguates(self) -> None:
        left = unit([1.0, 0.0, 0.0, 0.0])
        right = unit([0.0, 1.0, 0.0, 0.0])
        tracker = self.tracker(
            match_iou_threshold=0.05,
            match_center_distance_ratio=0.4,
            reid_spatial_gate_ratio=2.5,
        )
        seeded = tracker.update(
            [
                detection((100, 100, 140, 180), embedding=left),
                detection((180, 100, 220, 180), embedding=right),
            ],
            10.0,
            confirm_new=True,
        )
        ids = {round(item["box"]["x1"]): item["track_id"] for item in seeded}
        crossed = tracker.update(
            [
                detection((175, 100, 215, 180), embedding=left),
                detection((105, 100, 145, 180), embedding=right),
            ],
            10.5,
        )
        by_x = {round(item["box"]["x1"]): item["track_id"] for item in crossed}
        self.assertEqual(by_x[175], ids[100])
        self.assertEqual(by_x[105], ids[180])

    def test_thin_top_two_margin_refuses_false_merge(self) -> None:
        shared = unit([1.0, 0.1, 0.0, 0.0])
        almost = unit([1.0, 0.12, 0.0, 0.0])
        tracker = self.tracker(
            reid_top_two_margin=0.08,
            reid_match_threshold=0.5,
            reid_spatial_gate_ratio=4.0,
            entity_relink_enabled=False,
            lost_timeout_seconds=0.5,
        )
        tracker.update(
            [
                detection((10, 10, 40, 80), embedding=shared),
                detection((200, 10, 230, 80), embedding=almost),
            ],
            10.0,
            confirm_new=True,
        )
        tracker.update([], 11.5)
        self.assertEqual(set(tracker._completed), {1, 2})
        result = tracker.update(
            [detection((400, 10, 430, 80), embedding=unit([1.0, 0.11, 0.0, 0.0]))],
            12.0,
        )
        self.assertEqual(len(result), 1)
        self.assertNotIn(result[0]["track_id"], {1, 2})

    def test_polluted_crop_does_not_update_gallery(self) -> None:
        clean = unit([1.0, 0.0, 0.0, 0.0])
        dirty = unit([0.0, 1.0, 0.0, 0.0])
        tracker = self.tracker()
        tracker.update([detection((10, 10, 40, 80), embedding=clean)], 10.0, confirm_new=True)
        before = tracker._tracks[1].appearance.copy()
        # Tiny low-confidence crop should fail quality gate.
        tracker.update(
            [detection((10, 10, 18, 18), confidence=0.3, embedding=dirty)],
            10.5,
        )
        self.assertTrue(np.allclose(tracker._tracks[1].appearance, before, atol=1e-5))

    def test_co_occlusion_freezes_gallery_updates(self) -> None:
        left = unit([1.0, 0.0, 0.0, 0.0])
        right = unit([0.0, 1.0, 0.0, 0.0])
        tracker = self.tracker()
        tracker.update(
            [
                detection((100, 100, 160, 200), embedding=left),
                detection((120, 110, 180, 210), embedding=right),
            ],
            10.0,
            confirm_new=True,
        )
        before = tracker._tracks[1].appearance.copy()
        tracker.update(
            [
                detection((100, 100, 160, 200), embedding=unit([0.2, 0.8, 0.0, 0.0])),
                detection((120, 110, 180, 210), embedding=unit([0.8, 0.2, 0.0, 0.0])),
            ],
            10.5,
        )
        self.assertTrue(np.allclose(tracker._tracks[1].appearance, before, atol=1e-4))

    def test_completed_bank_prunes_beyond_reid_age(self) -> None:
        emb = unit([1.0, 0.0, 0.0, 0.0])
        tracker = self.tracker(reid_max_age_seconds=2.0, lost_timeout_seconds=0.5)
        tracker.update([detection((10, 10, 40, 80), embedding=emb)], 10.0, confirm_new=True)
        tracker.update([], 11.0)
        self.assertIn(1, tracker._completed)
        tracker.update([], 13.5)
        self.assertNotIn(1, tracker._completed)

    def test_entity_relink_assigns_entity_id_to_new_tracklet(self) -> None:
        emb = unit([1.0, 0.0, 0.0, 0.0])
        tracker = self.tracker(
            entity_relink_enabled=True,
            lost_timeout_seconds=0.5,
            reid_spatial_gate_ratio=3.0,
        )
        tracker.update([detection((10, 10, 40, 80), embedding=emb)], 10.0, confirm_new=True)
        tracker.update([], 11.0)
        self.assertIn(1, tracker._completed)
        result = tracker.update(
            [detection((12, 12, 42, 82), embedding=emb)],
            12.5,
        )
        self.assertEqual(len(result), 1)
        self.assertNotEqual(result[0]["track_id"], 1)
        self.assertEqual(result[0]["entity_id"], 1)

    def test_geometry_keeps_ambiguous_match_when_reid_disabled(self) -> None:
        tracker = SparseIdentityObjectTracker(
            ObjectTrackingConfig(
                min_confirmations=1,
                reid_enabled=False,
                match_iou_threshold=0.05,
                match_center_distance_ratio=1.2,
            ),
            0.7,
        )
        seeded = tracker.update(
            [
                detection((100, 100, 140, 180)),
                detection((180, 100, 220, 180)),
            ],
            10.0,
            confirm_new=True,
        )
        self.assertEqual(len(seeded), 2)
        continued = tracker.update(
            [
                detection((110, 100, 150, 180)),
                detection((170, 100, 210, 180)),
            ],
            10.5,
        )
        self.assertEqual(len(continued), 2)
        self.assertEqual(
            {item["track_id"] for item in continued},
            {item["track_id"] for item in seeded},
        )
        self.assertEqual(tracker.diagnostics()["association_counts"]["geometry"], 2)

    def test_person_retention_profile_raises_floors(self) -> None:
        config = ObjectTrackingConfig(
            tracking_profile="person_retention",
            lost_timeout_seconds=3.0,
            reid_enabled=True,
            reid_model_path="person.xml",
        )
        self.assertGreaterEqual(config.lost_timeout_seconds, 5.0)
        self.assertTrue(config.entity_relink_enabled)


class PromotionGateTest(unittest.TestCase):
    def test_promotion_requires_improvement_without_false_merge_rise(self) -> None:
        baseline = {
            "idf1": 0.7,
            "id_switches": 4,
            "fragmentations": 6,
            "false_merges": 1,
        }
        better = {
            "idf1": 0.75,
            "id_switches": 3,
            "fragmentations": 4,
            "false_merges": 1,
        }
        worse_merges = {**better, "false_merges": 2}
        self.assertTrue(promotion_gate(baseline, better)["promote"])
        self.assertFalse(promotion_gate(baseline, worse_merges)["promote"])


class RecoveryMetricsTest(unittest.TestCase):
    def test_recovery_by_gap_reports_same_track_precision(self) -> None:
        replay = {
            "schema_version": 1,
            "camera_id": "gate",
            "tracking_config": ObjectTrackingConfig(min_confirmations=1).model_dump(mode="json"),
            "high_confidence_threshold": 0.7,
            "timestamp_source": "source_pts",
            "source_pts_frames": 4,
            "appearance_source": "shared_supplied_embeddings",
            "frames": [
                {
                    "frame_index": i,
                    "captured_at": 100 + i,
                    "width": 120,
                    "height": 100,
                    "detections": [detection((10, 10, 40, 80))],
                }
                for i in range(4)
            ],
        }
        replay["replay_id"] = replay_digest(replay)
        labels = {
            "schema_version": 1,
            "replay_id": replay["replay_id"],
            "frames": [
                {
                    "frame_index": i,
                    "objects": [{
                        "identity": "person-A",
                        "label": "person",
                        "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80},
                    }],
                }
                for i in range(4)
            ],
        }
        observations = [
            {"frame_index": 0, "captured_at": 100.0, "objects": [{"track_id": 1, "label": "person", "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}]},
            {"frame_index": 1, "captured_at": 101.0, "objects": []},
            {"frame_index": 2, "captured_at": 102.0, "objects": [{"track_id": 1, "label": "person", "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}]},
            {"frame_index": 3, "captured_at": 103.0, "objects": [{"track_id": 1, "label": "person", "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 80}}]},
        ]
        score = identity_metrics(observations, labels, replay)
        self.assertEqual(score["fragmentations"], 1)
        self.assertEqual(score["recovery_by_gap"]["0.5s"]["attempts"], 1)
        self.assertEqual(score["recovery_by_gap"]["0.5s"]["same_track_successes"], 1)
        self.assertEqual(score["recovery_by_gap"]["0.5s"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
