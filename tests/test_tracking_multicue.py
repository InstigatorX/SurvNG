from __future__ import annotations

import copy
import itertools
import json
import math
import random
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from survng.app.config import ObjectTrackingConfig
from survng.app.object_track.hybrid import HybridObjectTracker
from survng.app.object_track.multicue import (
    CUE_VERSION, TRACE_LIMIT, HybridMultiCueObjectTracker,
    _direction_distance, _height_modulated_iou,
)
from tests import test_tracking_candidate as established


def detection(x=100.0, *, y=100.0, width=80.0, height=180.0, confidence=0.9, label="person"):
    return {"label": label, "confidence": confidence, "incident_eligible": True,
            "box": {"x1": x, "y1": y, "x2": x + width, "y2": y + height}}


class HybridMultiCueTest(unittest.TestCase):
    def tracker(self, **weights):
        return HybridMultiCueObjectTracker(ObjectTrackingConfig(), 0.7, **weights)

    def test_hmiou_exact_horizontal_vertical_and_disjoint(self):
        box = (100, 100, 200, 200)
        self.assertEqual(_height_modulated_iou(box, box), (1.0, 1.0))
        horizontal = _height_modulated_iou(box, (120, 100, 220, 200))
        vertical = _height_modulated_iou(box, (100, 120, 200, 220))
        self.assertAlmostEqual(horizontal[0], vertical[0])
        self.assertAlmostEqual(horizontal[1], 2 / 3)
        self.assertAlmostEqual(vertical[1], 4 / 9)
        self.assertEqual(_height_modulated_iou(box, (300, 300, 400, 400)), (0.0, 0.0))

    def test_height_overlap_breaks_equal_iou_tie(self):
        for reverse in (False, True):
            tracker = self.tracker(confidence_weight=0, direction_weight=0)
            tracker.update([detection(width=100, height=100)], 10, confirm_new=True)
            objects = [detection(y=120, width=100, height=100), detection(x=120, width=100, height=100)]
            result = tracker.update(objects[::-1] if reverse else objects, 10.5)
            retained = next(item for item in result if item["track_id"] == 1)
            self.assertEqual((retained["box"]["x1"], retained["box"]["y1"]), (120, 100))
            self.assertEqual(len(result), 2)  # No birth suppression of the other detection.
            self.assertGreater(tracker.diagnostics()["association_cues"]["score_totals"]["overlap_penalty"], 0)

    def test_confidence_projection_uses_seconds_and_caps_extrapolation(self):
        tracker = self.tracker()
        tracker.update([detection(confidence=0.8)], 10, confirm_new=True)
        self.assertEqual(tracker._projected_confidence(1, 10.5), (None, "short_history"))
        tracker.update([detection(confidence=0.9)], 11)
        self.assertAlmostEqual(tracker._projected_confidence(1, 11.5)[0], 0.95)
        self.assertAlmostEqual(tracker._projected_confidence(1, 12.5)[0], 1.0)
        self.assertEqual(tracker._projected_confidence(1, 13.01)[0], None)
        self.assertEqual(tracker._projected_confidence(1, 11)[0], None)
        tracker._confidence_samples[1] = ((10.0, 0.2), (11.0, 0.05))
        self.assertEqual(tracker._projected_confidence(1, 12)[0], 0.0)

    def test_confidence_is_pair_specific_not_a_reward_for_the_highest_detector_score(self):
        for reverse in (False, True):
            tracker = self.tracker(overlap_weight=0, direction_weight=0)
            tracker.update([detection(confidence=0.9)], 10, confirm_new=True)
            tracker.update([detection(confidence=0.85)], 10.5)
            objects = [detection(110, confidence=0.95), detection(90, confidence=0.8)]
            result = tracker.update(objects[::-1] if reverse else objects, 11)
            retained = next(item for item in result if item["track_id"] == 1)
            self.assertEqual(retained["confidence"], 0.8)
            self.assertEqual(retained["box"]["x1"], 90)

    def test_direction_preserves_crossing_ids_at_two_sparse_cadences(self):
        # Geometry alone swaps these strongly overlapping objects at the crossing.
        # Their previous two movements point in opposite, stable directions.
        for interval, reverse, isolated in itertools.product((0.5, 1 / 0.75), (False, True), (False, True)):
            with self.subTest(interval=interval, reverse=reverse, isolated=isolated):
                tracker = self.tracker(overlap_weight=0, confidence_weight=0) if isolated else self.tracker()
                baseline = HybridObjectTracker(ObjectTrackingConfig(), 0.7)
                original = {}
                for index, (a, b) in enumerate(((90, 116), (95, 111), (100, 106), (111, 95))):
                    objects = [detection(a), detection(b)]
                    if reverse:
                        objects.reverse()
                    current = tracker.update(copy.deepcopy(objects), 10 + index * interval, confirm_new=index == 0)
                    old = baseline.update(copy.deepcopy(objects), 10 + index * interval, confirm_new=index == 0)
                    if index == 0:
                        original = {item["box"]["x1"]: item["track_id"] for item in current}
                self.assertEqual({item["box"]["x1"]: item["track_id"] for item in current},
                                 {111: original[90], 95: original[116]})
                self.assertEqual({item["box"]["x1"]: item["track_id"] for item in old},
                                 {95: original[90], 111: original[116]})
                self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 2)

    def test_direction_is_neutral_for_short_stationary_stale_and_turning_history(self):
        tracker = self.tracker()
        tracker.update([detection()], 10, confirm_new=True)
        track = tracker._tracks[1]
        box = (120, 100, 200, 280)
        self.assertEqual(_direction_distance(track, box, 10.5)[1], "short_history")
        for timestamp in (10.5, 11):
            tracker.update([detection()], timestamp)
        self.assertEqual(_direction_distance(track, box, 11.5)[1], "stationary_or_jitter")
        self.assertEqual(_direction_distance(track, box, 14)[1], "timestamp_gap")
        self.assertEqual(_direction_distance(track, box, 11)[1], "timestamp_gap")
        track.trajectory = [(10, 100, 100), (10.5, 110, 100), (11, 100, 100)]
        self.assertEqual(_direction_distance(track, box, 11.5)[1], "recent_turn")
        track.trajectory = [(7, 100, 100), (10.5, 110, 100), (11, 120, 100)]
        self.assertEqual(_direction_distance(track, box, 11.5)[1], "stale_history")

    def test_lone_object_turn_does_not_get_a_direction_penalty(self):
        tracker = self.tracker()
        baseline = HybridObjectTracker(ObjectTrackingConfig(), 0.7)
        for index, x in enumerate((100, 105, 110, 100, 90)):
            self.assertEqual(tracker.update([detection(x)], 10 + index * 0.5, confirm_new=index == 0),
                             baseline.update([detection(x)], 10 + index * 0.5, confirm_new=index == 0))
        self.assertEqual(tracker.diagnostics()["association_cues"]["counts"]["ambiguous_pairs"], 0)

    def test_equal_or_backwards_timestamp_resets_confidence_history(self):
        for timestamp in (10.5, 10.4):
            tracker = self.tracker()
            tracker.update([detection()], 10, confirm_new=True)
            tracker.update([detection()], 10.5)
            tracker.update([detection(confidence=0.8)], timestamp)
            self.assertEqual(tracker._projected_confidence(1, 11)[1], "short_history")

    def test_zero_weights_reproduce_production_on_seeded_random_sequences(self):
        rng = random.Random(247)
        tracker = self.tracker(overlap_weight=0, confidence_weight=0, direction_weight=0)
        baseline = HybridObjectTracker(ObjectTrackingConfig(), 0.7)
        captured_at = 10.0
        for frame in range(80):
            captured_at += rng.choice((0.5, 1.2, 2.5))
            objects = [detection(rng.randrange(200), y=rng.randrange(100),
                                 confidence=rng.choice((0.2, 0.3, 0.75, 0.9)),
                                 label=rng.choice(("person", "car", "face")))
                       for _ in range(rng.randrange(6))]
            self.assertEqual(tracker.update(copy.deepcopy(objects), captured_at, confirm_new=frame == 0),
                             baseline.update(copy.deepcopy(objects), captured_at, confirm_new=frame == 0))
            self.assertEqual(tracker.summaries(captured_at), baseline.summaries(captured_at))

    def test_invalid_class_scale_age_edges_are_not_revived(self):
        tracker = self.tracker()
        tracker.update([detection()], 10, confirm_new=True)
        tracker.update([detection()], 10.5)
        objects = [detection(label="car"), detection(x=1000, width=1000, height=2000)]
        rows = [(index, obj, tuple(obj["box"][key] for key in ("x1", "y1", "x2", "y2")))
                for index, obj in enumerate(objects)]
        self.assertEqual(tracker._geometry_scores([1], rows, 11), [[0.0, 0.0]])
        rows = [(0, detection(), (100, 100, 180, 280))]
        self.assertEqual(tracker._geometry_scores([1], rows, 15), [[0.0]])

    def test_new_track_creation_and_confirmation_policy_is_unchanged(self):
        tracker = self.tracker()
        result = tracker.update([detection(), detection()], 10)
        self.assertEqual([item["track_id"] for item in result], [1, 2])
        self.assertTrue(all(item["track_state"] == "tentative" for item in result))
        extra = detection(1000)
        extra["incident_eligible"] = False
        result = tracker.update([extra, detection(2000, confidence=0.3)], 10.5)
        self.assertEqual(result, [])
        self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 2)

    def test_diagnostics_are_bounded_decomposable_copied_and_json_safe(self):
        tracker = self.tracker()
        for frame in range(10):
            tracker.update([detection(x=100 + index, y=100 + index, confidence=0.8 + index * 0.02)
                            for index in range(5)], 10 + frame * 0.5, confirm_new=frame == 0)
        diagnostics = tracker.diagnostics()
        cues = diagnostics["association_cues"]
        self.assertEqual(cues["version"], CUE_VERSION)
        self.assertEqual(len(cues["pair_samples"]), TRACE_LIMIT)
        self.assertGreater(cues["pair_samples_truncated"], 0)
        self.assertEqual(cues["new_track_policy"], "unchanged")
        for row in cues["pair_samples"]:
            self.assertAlmostEqual(row["final_score"], max(0.0, row["base_score"] - sum(
                row[key] for key in ("overlap_penalty", "confidence_penalty", "direction_penalty"))))
        totals = cues["score_totals"]
        self.assertAlmostEqual(totals["final_score"], totals["base_score"] - sum(
            totals[key] for key in ("overlap_penalty", "confidence_penalty", "direction_penalty")))
        json.dumps(diagnostics, allow_nan=False)
        cues["pair_samples"][0]["base_score"] = -1
        cues["weights"]["overlap"] = 99
        self.assertGreaterEqual(tracker.diagnostics()["association_cues"]["pair_samples"][0]["base_score"], 0)
        self.assertEqual(tracker.diagnostics()["association_cues"]["weights"]["overlap"], 0.25)
        self.assertLessEqual(len(tracker._confidence_samples), tracker.config.max_tracks_per_session)
        self.assertTrue(all(len(samples) <= 2 for samples in tracker._confidence_samples.values()))

    def test_exact_continuation_regression_stays_fixed(self):
        tracker = HybridMultiCueObjectTracker(ObjectTrackingConfig(
            reid_enabled=True, reid_model_path="person-reid.xml",
        ), 0.7)
        def item(x, embedding):
            value = detection(x, width=40, height=80)
            value["_tracking_embedding"] = embedding
            return value
        seeds = [item(200, [1, 0, 0]), item(255, [0, 1, 0])]
        tracker.update(copy.deepcopy(seeds), 10, confirm_new=True)
        tracker.update(copy.deepcopy(seeds), 10.5)
        result = tracker.update([item(200, [1, 0, 0]), item(155, [0, 0, 1])], 12)
        self.assertEqual({obj["box"]["x1"]: obj["track_id"] for obj in result}, {200: 1, 155: 3})
        self.assertEqual(tracker._tracks[1].appearance.tolist(), [1, 0, 0])
        self.assertEqual(tracker._tracks[2].appearance.tolist(), [0, 1, 0])

    def test_high_reid_precedes_low_false_box_regression_stays_fixed(self):
        tracker = HybridMultiCueObjectTracker(ObjectTrackingConfig(
            reid_enabled=True, reid_model_path="person-reid.xml",
        ), 0.7)
        seed = detection(200, width=40, height=80)
        seed["_tracking_embedding"] = [1, 0]
        tracker.update(copy.deepcopy([seed]), 10, confirm_new=True)
        tracker.update(copy.deepcopy([seed]), 10.5)
        high = detection(400, width=40, height=80)
        calls = []
        def provide():
            calls.append(high.get("_tracking_embedding_reason"))
            return [1, 0]
        high["_tracking_embedding_provider"] = provide
        result = tracker.update([high, detection(200, width=40, height=80, confidence=0.3)], 12)
        self.assertEqual([(item["box"]["x1"], item["track_id"]) for item in result], [(400, 1)])
        self.assertEqual(calls, ["geometry_recovery"])

    def test_invalid_weights_rejected(self):
        for weights in ({"overlap_weight": -1}, {"confidence_weight": math.inf},
                        {"direction_weight": math.nan}, {"overlap_weight": 0.5}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.tracker(**weights)


class MultiCueEstablishedIdentityTest(established.HybridIdentityRegressionTest):
    """Keep all #180 positive, negative, per-label, and lazy-ReID regressions."""

    def setUp(self):
        tracker_patch = patch.object(established, "HybridObjectTracker", HybridMultiCueObjectTracker)
        tracker_patch.start()
        self.addCleanup(tracker_patch.stop)
        super().setUp()


class AssociationCuesReplayCliTest(unittest.TestCase):
    def test_opt_in_compares_two_distinct_engines_without_changing_live_registry(self):
        from survng.app import tracking_evaluation
        from survng.app.object_track.multicue import IMPLEMENTATION
        from survng.app.object_track.registry import build_builtin_object_tracker_registry
        from survng.app.tracking_comparison import TRACKING_COMPARISON_IMPLEMENTATIONS

        replay = {
            "schema_version": 1,
            "tracking_config": ObjectTrackingConfig().model_dump(mode="json"),
            "high_confidence_threshold": 0.7, "timestamp_source": "test_fixture",
            "appearance_source": "no_embeddings", "source_pts_frames": 0,
            "frames": [{"frame_index": i, "captured_at": 10 + i * .5,
                        "width": 640, "height": 360, "detections": [detection(x)]}
                       for i, x in enumerate((100, 105, 110))],
        }
        replay["replay_id"] = tracking_evaluation.replay_digest(replay)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "replay.json", Path(directory) / "result.json"
            source.write_text(json.dumps(replay))
            original = source.read_bytes()
            with patch.object(sys, "argv", ["replay", str(source), "--association-cues", "--output", str(output)]):
                tracking_evaluation.main()
            result = json.loads(output.read_text())
            self.assertEqual(set(result["engines"]), {"survng_hybrid", IMPLEMENTATION})
            self.assertEqual(result["replay_id"], replay["replay_id"])
            self.assertEqual(source.read_bytes(), original)
            for engine in result["engines"].values():
                self.assertNotIn("error", engine)
            self.assertEqual(result["engines"][IMPLEMENTATION]["reid_diagnostics"]["association_cues"]["version"], CUE_VERSION)
        with self.assertRaises(ValueError):
            build_builtin_object_tracker_registry().require(IMPLEMENTATION)
        self.assertEqual(TRACKING_COMPARISON_IMPLEMENTATIONS,
                         ("survng_hybrid", "ultralytics_tracktrack", "ultralytics_botsort"))

    def test_label_template_cannot_accidentally_run_cue_experiment(self):
        from survng.app import tracking_evaluation
        with patch.object(sys, "argv", ["replay", "missing.json", "--label-template",
                                        "--association-cues", "--output", "unused.json"]):
            with self.assertRaises(SystemExit) as raised:
                tracking_evaluation.main()
        self.assertEqual(raised.exception.code, 2)
