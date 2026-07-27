from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from survng.app import main
from survng.app.config import AppConfig, CameraConfig
from survng.app.state_events import StateEventBroker
from fastapi import HTTPException


class EventApiSerializationTest(unittest.TestCase):
    def test_motion_audit_ai_context_explains_current_decision_outcome(self) -> None:
        config = AppConfig(
            cameras=[CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera.invalid/main",
                live_stream_url="rtsp://camera.invalid/sub",
                onvif={"enabled": True},
            )],
        )

        class Events:
            @staticmethod
            def get(_event_id: int):
                return None

            @staticmethod
            def motion_audits(**_kwargs):
                return [], 0

            @staticmethod
            def for_camera_range(_camera_id, _start_at, _end_at, limit=1000):
                return [{
                    "id": 99,
                    "created_at": "2026-07-27T12:17:07+00:00",
                    "objects_json": json.dumps([{
                        "label": "person",
                        "confidence": 0.9048,
                        "incident_eligible": True,
                    }]),
                }]

        manager = type("Manager", (), {"events": Events()})()
        context = main._audit_ai_context(
            {
                "id": 7,
                "camera_id": "gate",
                "features_json": "{}",
                "created_at": "2026-07-27T12:17:16+00:00",
                "reason": "event_state_active",
                "event_id": None,
                "object_detected": None,
            },
            config,
            manager,
        )

        self.assertEqual(context["motion_paradigm"]["paradigm"], "camera_triggered")
        self.assertEqual(context["motion_paradigm"]["adaptive_visual"]["role"], "validator")
        self.assertTrue(context["decision_outcome"]["filtered_before_object_detection"])
        self.assertFalse(context["decision_outcome"]["object_detection_ran"])
        self.assertIsNone(context["decision_outcome"]["object_detected"])
        self.assertEqual(
            context["decision_outcome"]["interpretation"]["category"],
            "duplicate_active_event",
        )
        self.assertEqual(context["related_prior_event"]["event_id"], 99)
        self.assertEqual(context["related_prior_event"]["seconds_before"], 9.0)
        self.assertEqual(context["related_prior_event"]["objects"][0]["label"], "person")

    def test_initial_event_stream_does_not_drop_change_racing_snapshot(self) -> None:
        class Request:
            headers: dict[str, str] = {}

            async def is_disconnected(self) -> bool:
                return False

        class Manager:
            def __init__(self) -> None:
                self.state_events = StateEventBroker()

            def statuses(self) -> list[dict]:
                self.state_events.publish("camera_state", {"id": "gate", "running": True})
                return [{"id": "gate", "running": False}]

        async def messages() -> list[str]:
            manager = Manager()
            with (
                patch.object(main, "manager", manager),
                patch.object(main, "system_status", return_value={}),
            ):
                response = await main.application_event_stream(Request())
                iterator = response.body_iterator
                return [await anext(iterator) for _ in range(5)]

        payloads = asyncio.run(messages())

        self.assertIn('"running":false', payloads[1])
        self.assertIn("event: camera_state", payloads[4])
        self.assertIn('"running":true', payloads[4])

    def test_face_crop_rejects_non_finite_padding_before_storage_access(self) -> None:
        with self.assertRaises(HTTPException) as invalid:
            main.face_crop(1, padding=float("nan"))

        self.assertEqual(invalid.exception.status_code, 422)

    def test_incident_search_rejects_unsafe_timezone_paths(self) -> None:
        with self.assertRaises(HTTPException) as invalid:
            main.incident_search(time_zone="../../etc/passwd")

        self.assertEqual(invalid.exception.status_code, 422)

    def test_event_row_tolerates_malformed_legacy_object_entries(self) -> None:
        row = main._event_row({
            "id": 1,
            "objects_json": json.dumps([
                "legacy",
                {"label": "person", "confidence": "invalid", "zones": "not-a-list"},
                {"label": "car", "confidence": 0.8, "zones": ["driveway"]},
            ]),
        })

        self.assertEqual(row["labels"], ["car"])
        self.assertEqual(row["zones"], ["driveway"])
        self.assertEqual(len(row["objects"]), 2)

        selected = main._best_incident_event([row])
        self.assertEqual(selected["id"], 1)

    def test_linked_motion_observation_extends_incident_without_duplicate_event(self) -> None:
        incident = main._incident_row("foyer", [{
            "id": 42,
            "camera_id": "foyer",
            "created_at": "2026-07-27T12:17:07+00:00",
            "has_objects": True,
            "labels": ["person"],
            "zones": [],
            "objects": [{"label": "person", "confidence": 0.9}],
            "motion_observations": [{
                "id": 3278,
                "created_at": "2026-07-27T12:17:16+00:00",
                "reason": "event_state_active",
            }],
        }])

        self.assertEqual(incident["event_count"], 1)
        self.assertEqual(incident["motion_observation_count"], 1)
        self.assertEqual(incident["duration_seconds"], 9.0)
        self.assertEqual(incident["end_at"], "2026-07-27T12:17:16+00:00")
        self.assertEqual(incident["motion_observations"][0]["id"], 3278)

    def test_motion_audit_snapshot_status_is_confined_to_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as outside:
            outside_image = Path(outside) / "private.jpg"
            outside_image.write_bytes(b"image")

            row = main._motion_audit_row(
                {
                    "id": 1,
                    "features_json": "[]",
                    "snapshot_path": str(outside_image),
                    "object_detected": 0,
                },
                Path(storage),
            )

        self.assertEqual(row["features"], {})
        self.assertFalse(row["has_snapshot"])
        self.assertFalse(row["object_detected"])


if __name__ == "__main__":
    unittest.main()
