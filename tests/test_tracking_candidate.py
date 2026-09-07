from __future__ import annotations

import itertools
import math
import random
import unittest
from unittest.mock import patch

import numpy as np

from survng.app.config import ObjectTrackingConfig
from survng.app.object_track.assignment import maximum_weight_assignment
from survng.app.object_track.candidate import HybridCandidateObjectTracker
from survng.app.object_track.hybrid import HybridObjectTracker
from survng.app.object_track.registry import build_builtin_object_tracker_registry
from tests import test_object_tracking as existing_contract


def detection(box: tuple[float, float, float, float], confidence: float = 0.9) -> dict:
    return existing_contract.detection("person", confidence, box)


class MaximumWeightAssignmentTest(unittest.TestCase):
    def test_adversarial_matrix_outweighs_greedy_first_edge(self) -> None:
        self.assertEqual(maximum_weight_assignment([[9, 8], [7, 0]]), [(0, 1), (1, 0)])

    def test_weight_objective_does_not_implicitly_prefer_cardinality(self) -> None:
        self.assertEqual(maximum_weight_assignment([[10, 1], [1, 0]]), [(0, 0)])

    def test_rectangular_and_unmatched_rows(self) -> None:
        self.assertEqual(maximum_weight_assignment([[0, 4, 0], [0, 0, 3]]), [(0, 1), (1, 2)])
        self.assertEqual(maximum_weight_assignment([[0], [4], [2]]), [(1, 0)])
        self.assertEqual(maximum_weight_assignment([[0, 0], [0, 0]]), [])
        self.assertEqual(maximum_weight_assignment([]), [])
        self.assertEqual(maximum_weight_assignment([[], []]), [])

    def test_ties_are_deterministic(self) -> None:
        for _ in range(4):
            self.assertEqual(maximum_weight_assignment([[1, 1], [1, 1]]), [(0, 0), (1, 1)])

    def test_invalid_matrices_are_rejected(self) -> None:
        for weights in ([[1], []], [[-1]], [[float("nan")]], [[float("inf")]]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                maximum_weight_assignment(weights)

    def test_small_random_matrices_match_exhaustive_optimum(self) -> None:
        # Independent enumeration covers rectangular cases, zero edges and
        # alternating paths, rather than mirroring the assignment algorithm.
        rng = random.Random(417)
        for rows in range(1, 5):
            for columns in range(1, 5):
                for _ in range(12):
                    weights = [[rng.randrange(10) / 3 for _ in range(columns)] for _ in range(rows)]
                    optimum = 0.0
                    for choice in itertools.product(range(-1, columns), repeat=rows):
                        used = [column for column in choice if column >= 0]
                        if len(used) != len(set(used)):
                            continue
                        optimum = max(optimum, sum(
                            weights[row][column]
                            for row, column in enumerate(choice) if column >= 0
                        ))
                    actual = maximum_weight_assignment(weights)
                    self.assertEqual(len({row for row, _ in actual}), len(actual))
                    self.assertEqual(len({column for _, column in actual}), len(actual))
                    self.assertTrue(all(weights[row][column] > 0 for row, column in actual))
                    self.assertAlmostEqual(sum(weights[row][column] for row, column in actual), optimum)


class HybridProductionRegressionTest(unittest.TestCase):
    def tracker(self) -> HybridObjectTracker:
        return HybridObjectTracker(ObjectTrackingConfig(), 0.7)

    def test_registry_uses_promoted_hybrid(self) -> None:
        tracker = build_builtin_object_tracker_registry().create(
            "survng_hybrid",
            ObjectTrackingConfig(),
            0.7,
        )
        self.assertIsInstance(tracker, HybridObjectTracker)

    def test_candidate_name_remains_compatible(self) -> None:
        self.assertTrue(issubclass(HybridCandidateObjectTracker, HybridObjectTracker))

    def test_box_shrink_does_not_collapse_prediction_or_fragment_identity(self) -> None:
        for gap in (1 / 0.75, 2.0):
            with self.subTest(gap=gap):
                tracker = self.tracker()
                tracker.update([detection((0, 0, 200, 200))], 10.0, confirm_new=True)
                tracker.update([detection((50, 50, 150, 150))], 10.5)

                self.assertEqual(tracker._tracks[1].predicted_box(10.5 + gap), (50, 50, 150, 150))
                result = tracker.update([detection((50, 50, 150, 150))], 10.5 + gap)

                self.assertEqual([item["track_id"] for item in result], [1])
                self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 1)

    def test_prediction_retains_measured_size_while_advancing_center(self) -> None:
        tracker = self.tracker()
        tracker.update([detection((0, 0, 200, 200))], 10.0, confirm_new=True)
        tracker.update([detection((60, 50, 160, 150))], 10.5)

        # Measured center moved right 10px / .5s; existing EMA gives 7px/s.
        self.assertEqual(tracker._tracks[1].predicted_box(11.5), (67, 50, 167, 150))

    def test_global_matching_preserves_both_valid_continuations(self) -> None:
        previous = [(200, 100, 240, 180), (280, 100, 320, 180)]
        following = [(230, 100, 270, 180), (160, 100, 200, 180)]
        for seeds in itertools.permutations(previous):
            for next_boxes in itertools.permutations(following):
                with self.subTest(seeds=seeds, next_boxes=next_boxes):
                    tracker = self.tracker()
                    first = tracker.update([detection(box) for box in seeds], 10.0, confirm_new=True)
                    old_ids = {item["box"]["x1"]: item["track_id"] for item in first}

                    result = tracker.update([detection(box) for box in next_boxes], 10.5)

                    self.assertEqual(
                        {item["box"]["x1"]: item["track_id"] for item in result},
                        {230: old_ids[280], 160: old_ids[200]},
                    )
                    self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 2)

    def test_relaxed_high_match_cannot_steal_exact_low_continuation(self) -> None:
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                tracker = self.tracker()
                box = (200, 100, 240, 180)
                tracker.update([detection(box)], 10.0, confirm_new=True)
                tracker.update([detection(box)], 10.5)
                objects = [detection((300, 100, 340, 180)), detection(box, 0.3)]
                result = tracker.update(list(reversed(objects)) if reverse else objects, 11.0)

                self.assertEqual(
                    {item["box"]["x1"]: item["track_id"] for item in result}, {200: 1, 300: 2},
                )

    def test_normal_high_confidence_association_still_precedes_low_pass(self) -> None:
        tracker = self.tracker()
        box = (200, 100, 240, 180)
        tracker.update([detection(box)], 10.0, confirm_new=True)
        result = tracker.update([detection((220, 100, 260, 180)), detection(box, 0.3)], 10.5)

        self.assertEqual([(item["track_id"], item["box"]["x1"]) for item in result], [(1, 220)])


class HybridIdentityRegressionTest(unittest.TestCase):
    @staticmethod
    def object(x: float, *, label: str = "person", confidence: float = 0.9,
               embedding: list[float] | None = None) -> dict:
        item = detection((x, 100, x + 40, 180), confidence)
        item["label"] = label
        if embedding is not None:
            item["_tracking_embedding"] = np.asarray(embedding, dtype=np.float32)
        return item

    def test_exact_continuation_is_not_traded_for_two_weaker_matches(self) -> None:
        # A remains, B disappears, and C enters. Preserving two old IDs would
        # assign both visible people the wrong identities and pollute their ReID.
        for reid_enabled, seed_order, next_order in itertools.product(
            (False, True), ((200, 255), (255, 200)), ((200, 155), (155, 200)),
        ):
            with self.subTest(reid=reid_enabled, seeds=seed_order, following=next_order):
                tracker = HybridObjectTracker(ObjectTrackingConfig(
                    reid_enabled=reid_enabled, reid_model_path="person-reid.xml",
                ), 0.7)
                embeddings = {200: [1.0, 0.0, 0.0], 255: [0.0, 1.0, 0.0],
                              155: [0.0, 0.0, 1.0]}

                def observation(x):
                    return self.object(x, embedding=embeddings[x] if reid_enabled else None)

                first = tracker.update([observation(x) for x in seed_order], 10.0, confirm_new=True)
                old_ids = {item["box"]["x1"]: item["track_id"] for item in first}
                tracker.update([observation(x) for x in seed_order], 10.5)
                result = tracker.update([observation(x) for x in next_order], 12.0)

                self.assertEqual({item["box"]["x1"]: item["track_id"] for item in result},
                                 {200: old_ids[200], 155: 3})
                self.assertEqual({item["box"]["x1"]: item["track_state"] for item in result},
                                 {200: "confirmed", 155: "tentative"})
                self.assertEqual(tracker._tracks[old_ids[255]].last_seen, 10.5)
                self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 3)
                if reid_enabled:
                    for x in seed_order:
                        np.testing.assert_array_equal(tracker._tracks[old_ids[x]].appearance, embeddings[x])

    def test_high_confidence_reid_precedes_low_geometry_with_per_label_gates(self) -> None:
        # Lazy appearance work must happen before an exact but low-confidence
        # false box consumes the real person's or vehicle's track.
        cases = (("person", 1.0, True), ("person", 0.75, True),
                 ("car", 0.75, False), ("car", 0.9, True),
                 ("person", 0.0, False), ("person", None, False))
        for (label, similarity, recover), reverse in itertools.product(cases, (False, True)):
            with self.subTest(label=label, similarity=similarity, reverse=reverse):
                tracker = HybridObjectTracker(ObjectTrackingConfig(
                    reid_enabled=True, reid_model_path="person-reid.xml",
                    vehicle_reid_enabled=True, vehicle_reid_model_path="vehicle-reid.xml",
                    reid_match_threshold=0.7, vehicle_reid_match_threshold=0.8,
                ), 0.7)
                for timestamp in (10.0, 10.5):
                    tracker.update([self.object(200, label=label, embedding=[1.0, 0.0])],
                                   timestamp, confirm_new=timestamp == 10.0)
                high = self.object(400, label=label)
                calls = []

                def provide():
                    calls.append(high.get("_tracking_embedding_reason"))
                    if similarity is None:
                        return None
                    return np.asarray([similarity, math.sqrt(1.0 - similarity ** 2)], dtype=np.float32)

                high["_tracking_embedding_provider"] = provide
                low = self.object(200, label=label, confidence=0.3)
                result = tracker.update([low, high] if reverse else [high, low], 12.0)

                self.assertEqual({item["box"]["x1"]: item["track_id"] for item in result},
                                 {400: 1} if recover else {400: 2, 200: 1})
                self.assertEqual(calls, ["geometry_recovery"])
                self.assertEqual(tracker.diagnostics()["association_counts"]["appearance_recovery"],
                                 int(recover))
                self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"],
                                 1 if recover else 2)

    def test_unambiguous_high_geometry_still_avoids_lazy_reid(self) -> None:
        tracker = HybridObjectTracker(ObjectTrackingConfig(
            reid_enabled=True, reid_model_path="person-reid.xml",
        ), 0.7)
        tracker.update([self.object(200, embedding=[1.0, 0.0])], 10.0, confirm_new=True)
        high = self.object(202)

        def unexpected_embedding():
            self.fail("An ordinary geometry match must not add embedding work")

        high["_tracking_embedding_provider"] = unexpected_embedding
        result = tracker.update([high, self.object(200, confidence=0.3)], 10.5)

        self.assertEqual([(item["box"]["x1"], item["track_id"]) for item in result], [(202, 1)])
        self.assertEqual(tracker.diagnostics()["reid_avoided_geometry_matches"], 1)

    def test_completed_track_can_still_be_recovered_once(self) -> None:
        tracker = HybridObjectTracker(ObjectTrackingConfig(
            reid_enabled=True, reid_model_path="person-reid.xml",
        ), 0.7)
        tracker.update([self.object(200, embedding=[1.0, 0.0])], 10.0, confirm_new=True)
        tracker.update([], 14.0)
        self.assertFalse(tracker.has_live_tracks(14.0))
        result = tracker.update([self.object(400, embedding=[1.0, 0.0]),
                                 self.object(200, confidence=0.3)], 15.0)

        self.assertEqual([(item["box"]["x1"], item["track_id"]) for item in result], [(400, 1)])
        self.assertEqual(tracker._tracks[1].reid_matches, 1)
        self.assertTrue(tracker._tracks[1].reid_recovery_history[0]["resumed_completed_track"])
        self.assertEqual(tracker.diagnostics()["association_counts"]["new_track"], 1)


class HybridProductionExistingContractTest(existing_contract.ByteTrackObjectTrackerTest):
    """Run the established seed, high/low, retention and appearance contracts."""

    def setUp(self) -> None:
        tracker_patch = patch.object(existing_contract, "ByteTrackObjectTracker", HybridObjectTracker)
        tracker_patch.start()
        self.addCleanup(tracker_patch.stop)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
