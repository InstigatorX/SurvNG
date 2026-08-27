from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from survng.app.appearance_index import AppearanceIndex
from survng.app.events import EventStore


class AppearanceIndexTest(unittest.TestCase):
    def test_matches_only_compatible_model_vectors_without_exposing_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            anchor = store.add_event(
                "gate",
                "motion",
                created_at="2026-08-01T12:00:00+00:00",
            )
            similar = store.add_event(
                "upper-garage",
                "motion",
                created_at="2026-08-01T12:05:00+00:00",
            )
            different_model = store.add_event(
                "back-left",
                "motion",
                created_at="2026-08-01T12:06:00+00:00",
            )
            index = AppearanceIndex(store.db_path)
            common = {
                "track_id": 1,
                "label": "car",
                "model_kind": "vehicle",
                "model_fingerprint": "vehicle-model-v1",
                "match_threshold": 0.8,
                "observation_count": 5,
                "quality": 0.9,
                "first_seen": "2026-08-01T12:00:00+00:00",
                "last_seen": "2026-08-01T12:00:05+00:00",
            }
            index.replace_event(
                int(anchor["id"]),
                "gate",
                [{**common, "embedding": np.asarray([1.0, 0.0])}],
            )
            index.replace_event(
                int(similar["id"]),
                "upper-garage",
                [{
                    **common,
                    "label": "truck",
                    "embedding": np.asarray([0.98, 0.2]),
                    "last_seen": "2026-08-01T12:05:05+00:00",
                }],
            )
            index.replace_event(
                int(different_model["id"]),
                "back-left",
                [{
                    **common,
                    "model_fingerprint": "vehicle-model-v2",
                    "embedding": np.asarray([1.0, 0.0]),
                    "last_seen": "2026-08-01T12:06:05+00:00",
                }],
            )

            matches = index.matches(
                int(anchor["id"]),
                start_at="2026-08-01T11:00:00+00:00",
                end_at="2026-08-01T13:00:00+00:00",
            )

        self.assertEqual([item["event_id"] for item in matches], [similar["id"]])
        self.assertTrue(matches[0]["visually_similar"])
        self.assertGreater(matches[0]["similarity"], 0.97)
        self.assertEqual(matches[0]["anchor_label"], "car")
        self.assertEqual(matches[0]["candidate_label"], "truck")
        self.assertNotIn("embedding", matches[0])
        self.assertNotIn("model_fingerprint", matches[0])

    def test_cross_camera_filter_and_threshold_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            anchor = store.add_event("gate", "motion", created_at="2026-08-01T12:00:00+00:00")
            same_camera = store.add_event("gate", "motion", created_at="2026-08-01T12:01:00+00:00")
            other_camera = store.add_event("foyer", "motion", created_at="2026-08-01T12:02:00+00:00")
            index = AppearanceIndex(store.db_path)

            def save(event: dict, camera: str, vector: list[float]) -> None:
                index.replace_event(int(event["id"]), camera, [{
                    "track_id": 1,
                    "label": "person",
                    "model_kind": "person",
                    "model_fingerprint": "person-v1",
                    "match_threshold": 0.8,
                    "embedding": np.asarray(vector, dtype=np.float32),
                    "last_seen": event["created_at"],
                }])

            save(anchor, "gate", [1.0, 0.0])
            save(same_camera, "gate", [1.0, 0.0])
            save(other_camera, "foyer", [0.6, 0.8])

            cross_camera = index.matches(int(anchor["id"]))
            all_cameras = index.matches(int(anchor["id"]), cross_camera_only=False)
            status = index.status()

        self.assertEqual([item["event_id"] for item in cross_camera], [other_camera["id"]])
        self.assertFalse(cross_camera[0]["visually_similar"])
        self.assertEqual(all_cameras[0]["event_id"], same_camera["id"])
        self.assertEqual(status["vectors"], 3)
        self.assertEqual(status["events"], 3)

    def test_track_id_filters_to_selected_anchor_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            anchor = store.add_event("gate", "motion")
            first_match = store.add_event("foyer", "motion")
            second_match = store.add_event("driveway", "motion")
            index = AppearanceIndex(store.db_path)
            common = {
                "label": "person",
                "model_kind": "person",
                "model_fingerprint": "person-v1",
                "match_threshold": 0.8,
            }
            index.replace_event(int(anchor["id"]), "gate", [
                {
                    **common,
                    "track_id": 1,
                    "embedding": np.asarray([1.0, 0.0]),
                },
                {
                    **common,
                    "track_id": 2,
                    "embedding": np.asarray([0.0, 1.0]),
                },
            ])
            index.replace_event(int(first_match["id"]), "foyer", [{
                **common,
                "track_id": 10,
                "embedding": np.asarray([1.0, 0.0]),
            }])
            index.replace_event(int(second_match["id"]), "driveway", [{
                **common,
                "track_id": 20,
                "embedding": np.asarray([0.0, 1.0]),
            }])

            matches = index.matches(int(anchor["id"]), track_id=2)
            missing = index.matches(int(anchor["id"]), track_id=99)

        self.assertEqual(matches[0]["event_id"], second_match["id"])
        self.assertTrue(all(item["anchor_track_id"] == 2 for item in matches))
        self.assertEqual(missing, [])

    def test_invalid_replacement_does_not_erase_a_valid_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EventStore(Path(tmpdir))
            event = store.add_event("gate", "motion")
            index = AppearanceIndex(store.db_path)
            saved = index.replace_event(int(event["id"]), "gate", [{
                "track_id": 1,
                "label": "car",
                "model_kind": "vehicle",
                "model_fingerprint": "vehicle-v1",
                "match_threshold": 0.8,
                "embedding": np.asarray([1.0, 0.0]),
                "last_seen": event["created_at"],
            }])
            rejected = index.replace_event(int(event["id"]), "gate", [{
                "track_id": 1,
                "label": "car",
                "model_kind": "vehicle",
                "model_fingerprint": "vehicle-v1",
                "embedding": np.asarray([float("nan")]),
            }])
            status = index.status()

        self.assertEqual(saved, 1)
        self.assertEqual(rejected, 0)
        self.assertEqual(status["vectors"], 1)
