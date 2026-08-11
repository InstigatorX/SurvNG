from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from survng.app.config import AppConfig
from survng.app.events import EventStore
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.training_routes import (
    TrainingRouteDependencies,
    create_training_router,
)


class TrainingRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = EventStore(Path(self.temporary.name))
        self.config = AppConfig(base_path="/survng")
        self.manager = SimpleNamespace(
            events=self.store,
            storage_dir=Path(self.temporary.name),
        )
        router = create_training_router(TrainingRouteDependencies(
            get_config=lambda: self.config,
            get_manager=lambda: self.manager,
            manager_lock=threading.RLock(),
            manager_access=ManagerAccessCoordinator(),
        ))
        self.route = next(
            route
            for route in router.routes
            if route.path == "/api/training/samples"
        )
        self.endpoint = self.route.endpoint

    @staticmethod
    def detected_object(
        label: str,
        confidence: float,
        *,
        eligible: bool = True,
        offset: float = 0.0,
    ) -> dict:
        return {
            "label": label,
            "confidence": confidence,
            "box": {"x1": 10, "y1": 5, "x2": 50, "y2": 25},
            "detection_frame_width": 100,
            "detection_frame_height": 50,
            "incident_eligible": eligible,
            "temporal_consensus": True,
            "temporal_sample_offset_seconds": offset,
            "semantic_tier": "standard",
            "zones": ["driveway"],
        }

    def request(self, **overrides):
        arguments = {
            "start_at": "2026-08-10T11:00:00-04:00",
            "end_at": "2026-08-10T13:00:00-04:00",
            "camera_ids": "",
            "object_labels": "",
            "eligibility": "eligible",
            "minimum_confidence": 0.0,
            "include_empty": False,
            "limit": 100,
            "cursor": "",
        }
        arguments.update(overrides)
        return self.endpoint(**arguments)

    def add_event(
        self,
        *,
        camera_id: str,
        created_at: str,
        objects: list[dict],
        suffix: str = "webp",
    ) -> dict:
        snapshot = (
            Path(self.temporary.name)
            / "snapshots"
            / camera_id
            / f"{created_at[-8:]}.{suffix}"
        )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"training-image")
        return self.store.add_event(
            camera_id=camera_id,
            kind="motion",
            snapshot_path=str(snapshot),
            objects_json=json.dumps(objects),
            created_at=created_at,
        )

    def test_manifest_returns_original_image_and_training_coordinates(self) -> None:
        event = self.add_event(
            camera_id="gate",
            created_at="2026-08-10T16:00:00+00:00",
            objects=[self.detected_object("person", 0.94, offset=1.5)],
        )

        payload = self.request()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(self.route.response_model.__name__, "TrainingSamplesResponse")
        self.assertEqual(payload["count"], 1)
        sample = payload["samples"][0]
        self.assertEqual(sample["event_id"], event["id"])
        self.assertEqual(len(sample["revision"]), 20)
        self.assertEqual(
            sample["image"]["url"],
            f"/survng/api/events/{event['id']}/snapshot.jpg",
        )
        self.assertEqual(sample["image"]["media_type"], "image/webp")
        self.assertEqual(sample["image"]["width"], 100)
        self.assertEqual(sample["captured_at"], "2026-08-10T16:00:01.500000+00:00")
        annotation = sample["annotations"][0]
        self.assertEqual(annotation["bbox_xyxy"], [10.0, 5.0, 50.0, 25.0])
        self.assertEqual(annotation["bbox_xywh"], [10.0, 5.0, 40.0, 20.0])
        self.assertEqual(annotation["bbox_normalized_xyxy"], [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(annotation["bbox_normalized_cxcywh"], [0.3, 0.3, 0.4, 0.4])
        self.assertEqual(annotation["annotation_state"], "model_generated")

    def test_filters_annotations_without_exposing_other_coordinate_planes(self) -> None:
        mismatched = self.detected_object("person", 0.99)
        mismatched["detection_frame_width"] = 200
        self.add_event(
            camera_id="gate",
            created_at="2026-08-10T16:00:00+00:00",
            objects=[
                self.detected_object("car", 0.93),
                self.detected_object("person", 0.82),
                mismatched,
                self.detected_object("person", 0.95, eligible=False),
            ],
        )

        payload = self.request(
            camera_ids="gate",
            object_labels="person",
            minimum_confidence=0.8,
        )

        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["samples"][0]["annotations"]), 1)
        self.assertEqual(payload["samples"][0]["annotations"][0]["label"], "person")

    def test_cursor_does_not_drop_unconsumed_rows_from_short_database_page(self) -> None:
        older = self.add_event(
            camera_id="gate",
            created_at="2026-08-10T16:00:00+00:00",
            objects=[self.detected_object("person", 0.8)],
        )
        newer = self.add_event(
            camera_id="gate",
            created_at="2026-08-10T16:01:00+00:00",
            objects=[self.detected_object("person", 0.9)],
        )

        first = self.request(limit=1)
        second = self.request(limit=1, cursor=first["next_cursor"])

        self.assertEqual(first["samples"][0]["event_id"], newer["id"])
        self.assertTrue(first["next_cursor"])
        self.assertEqual(second["samples"][0]["event_id"], older["id"])
        self.assertEqual(second["next_cursor"], "")

    def test_rejects_naive_dates_and_oversized_ranges(self) -> None:
        with self.assertRaisesRegex(HTTPException, "timezone"):
            self.request(start_at="2026-08-10T12:00:00")
        with self.assertRaisesRegex(HTTPException, "366 days"):
            self.request(
                start_at="2025-01-01T00:00:00+00:00",
                end_at="2026-08-10T00:00:00+00:00",
            )
        with self.assertRaisesRegex(HTTPException, "cursor is invalid"):
            self.request(cursor="not-base64!!")


if __name__ == "__main__":
    unittest.main()
