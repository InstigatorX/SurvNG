from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
        self.preview = Mock(return_value=Path(self.temporary.name) / "main.jpg")
        self.manager.recorder = Mock()
        router = create_training_router(TrainingRouteDependencies(
            get_config=lambda: self.config,
            get_manager=lambda: self.manager,
            manager_lock=threading.RLock(),
            manager_access=ManagerAccessCoordinator(),
            recording_preview_path=self.preview,
            recording_preview_timestamp=lambda _: (1786377600.01, "source_pts"),
        ))
        self.route = next(
            route
            for route in router.routes
            if route.path == "/api/training/samples"
        )
        self.endpoint = self.route.endpoint
        self.main_image = next(route.endpoint for route in router.routes
                               if route.path.endswith("/main.jpg"))

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
            "sample_kinds": "",
            "sources": "",
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
        self.assertEqual(payload["scanned_events"], 1)
        sample = payload["samples"][0]
        self.assertEqual(sample["event_id"], event["id"])
        self.assertEqual(sample["source_id"], event["id"])
        self.assertEqual(sample["source"], "event")
        self.assertEqual(sample["sample_kind"], "annotated")
        self.assertFalse(sample["assumed_negative"])
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
        off_frame = self.detected_object("person", 0.98)
        off_frame["snapshot_visible"] = False
        self.add_event(
            camera_id="gate",
            created_at="2026-08-10T16:00:00+00:00",
            objects=[
                self.detected_object("car", 0.93),
                self.detected_object("person", 0.82),
                mismatched,
                off_frame,
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

    def test_motion_audits_are_unreviewed_negative_candidates(self) -> None:
        snapshot = Path(self.temporary.name) / "motion" / "gate.webp"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"clean-motion-image")
        audit = self.store.add_motion_audit(
            camera_id="gate",
            snapshot_path=str(snapshot),
            created_at="2026-08-10T16:00:00+00:00",
            mode="camera_rescue",
            sensitivity="balanced",
            score=0.77,
            threshold=0.65,
            reason="visual_backup_no_object",
            object_detected=False,
            trigger_count=1,
            features={},
            category="visual_backup",
        )

        payload = self.request(
            sample_kinds="negative_candidate",
            sources="motion_audit",
        )

        self.assertEqual(payload["count"], 1)
        sample = payload["samples"][0]
        self.assertEqual(sample["sample_id"], f"motion_audit-{audit['id']}")
        self.assertEqual(sample["source_id"], audit["id"])
        self.assertIsNone(sample["event_id"])
        self.assertEqual(sample["source"], "motion_audit")
        self.assertEqual(sample["sample_kind"], "negative_candidate")
        self.assertTrue(sample["assumed_negative"])
        self.assertEqual(sample["annotation_state"], "unreviewed")
        self.assertEqual(sample["annotations"], [])
        self.assertEqual(
            sample["image"]["url"],
            f"/survng/api/motion-audit/{audit['id']}/snapshot.jpg",
        )

    def test_motion_audit_negative_candidates_exclude_known_object_matches(self) -> None:
        for index, detected in enumerate((False, True)):
            snapshot = Path(self.temporary.name) / "motion" / f"gate-{index}.webp"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(b"clean-motion-image")
            self.store.add_motion_audit(
                camera_id="gate",
                snapshot_path=str(snapshot),
                created_at=f"2026-08-10T16:0{index}:00+00:00",
                mode="camera_rescue",
                sensitivity="balanced",
                score=0.77,
                threshold=0.65,
                reason="visual_backup_object" if detected else "visual_backup_no_object",
                object_detected=detected,
                trigger_count=1,
                features={},
                category="visual_backup",
            )

        payload = self.request(
            sample_kinds="negative_candidate",
            sources="motion_audit",
        )

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["samples"][0]["event_at"], "2026-08-10T16:00:00+00:00")

    def add_negative_audit(self, **overrides):
        payload = dict(camera_id="gate", snapshot_path="",
                       created_at="2026-08-10T16:00:00+00:00",
                       mode="camera_rescue", sensitivity="balanced", score=0.5,
                       threshold=0.6, reason="visual_backup_no_object",
                       object_detected=False, trigger_count=1, features={})
        payload.update(overrides)
        return self.store.add_motion_audit(**payload)

    def test_main_manifest_does_not_extract_or_require_stored_snapshot(self):
        audit = self.add_negative_audit()
        payload = self.request(sample_kinds="negative_candidate", image_source="main_recording")
        self.assertEqual(payload["count"], 1)
        sample = payload["samples"][0]
        self.assertEqual(sample["image"]["url"],
                         f"/survng/api/training/motion-audits/{audit['id']}/main.jpg")
        self.assertEqual(sample["image"]["media_type"], "image/jpeg")
        self.assertEqual(sample["annotations"], [])
        self.assertIsNone(sample["image"]["width"])
        self.preview.assert_not_called()
        self.manager.recorder.recording_rows_between.assert_not_called()
        self.assertEqual(self.request(sample_kinds="negative_candidate")["count"], 0)

    def test_main_images_reject_annotated_samples(self):
        with self.assertRaises(HTTPException) as error:
            self.request(image_source="main_recording")
        self.assertEqual(error.exception.status_code, 422)

    def test_main_image_uses_main_recording_at_audit_time(self):
        audit = self.add_negative_audit()
        epoch = 1786377600.0
        row = {"start_epoch": epoch - 5, "end_epoch": epoch + 5}
        self.manager.recorder.recording_rows_between.return_value = [row]
        response = self.main_image(audit["id"])
        self.manager.recorder.recording_rows_between.assert_called_once_with(
            "gate", epoch - 0.001, epoch + 0.001, "main", discover_missing=False)
        self.preview.assert_called_once_with(
            self.manager, row, epoch, exact=True, native_resolution=True)
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertEqual(response.headers["X-SurvNG-Actual-Timestamp"], "1786377600.010000")

    def test_main_image_missing_coverage_does_not_fall_back(self):
        audit = self.add_negative_audit()
        for rows in ([], [{"start_epoch": 1, "end_epoch": 2}]):
            with self.subTest(rows=rows):
                self.manager.recorder.recording_rows_between.return_value = rows
                with self.assertRaises(HTTPException) as error:
                    self.main_image(audit["id"])
                self.assertEqual(error.exception.status_code, 404)
        self.preview.assert_not_called()

    def test_main_image_rejects_objects_and_incident_activity(self):
        for payload in ({"object_detected": True}, {"reason": "event_state_active"},
                        {"reason": "event_state_cooldown"}):
            audit = self.add_negative_audit(**payload)
            with self.assertRaises(HTTPException) as error:
                self.main_image(audit["id"])
            self.assertEqual(error.exception.status_code, 422)
        self.preview.assert_not_called()

    def test_main_image_preserves_busy_response(self):
        audit = self.add_negative_audit()
        self.manager.recorder.recording_rows_between.return_value = [
            {"start_epoch": 1786377590, "end_epoch": 1786377610}]
        self.preview.side_effect = HTTPException(429, "busy", headers={"Retry-After": "1"})
        with self.assertRaises(HTTPException) as error:
            self.main_image(audit["id"])
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(error.exception.headers, {"Retry-After": "1"})


if __name__ == "__main__":
    unittest.main()
