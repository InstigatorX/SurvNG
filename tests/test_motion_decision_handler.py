from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

import numpy as np

from survng.app.motion_pipeline import MotionDecisionHandler


class RecordingEventStore:
    def __init__(self) -> None:
        self.payload: dict | None = None
        self.audit_payload: dict | None = None

    def add_event(self, **payload):
        self.payload = payload
        return {"id": 42, **payload}

    def add_motion_audit(self, **payload):
        self.audit_payload = payload
        return {"id": 7, **payload}


class MotionDecisionHandlerTest(unittest.TestCase):
    def test_handler_dispatches_detection_persists_event_and_publishes_outputs(self) -> None:
        events = RecordingEventStore()
        published = []
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        qualification = {"borderline_candidate": True, "would_suppress": True}
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{"label": "car", "confidence": 0.8, "incident_eligible": True}],
                "recording.mp4",
            ),
            snapshot_writer=lambda value: "snapshot.jpg" if value is frame else "",
            object_serializer=lambda objects: json.dumps(objects, separators=(",", ":")),
            event_callback=lambda event_type, payload: published.append((event_type, payload)),
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            qualification,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(outcome.snapshot_path, "snapshot.jpg")
        self.assertTrue(outcome.object_detected)
        self.assertTrue(qualification["rescued_by_object"])
        self.assertTrue(qualification["effective_accepted"])
        self.assertFalse(qualification["would_suppress"])
        self.assertEqual(events.payload["camera_id"], "gate")
        self.assertEqual(events.payload["recording_path"], "recording.mp4")
        stored_objects = json.loads(events.payload["objects_json"])
        self.assertEqual(stored_objects[0]["label"], "car")
        self.assertEqual(stored_objects[-1]["status"], "motion_qualification")
        self.assertEqual([event_type for event_type, _payload in published], ["incident", "object"])

    def test_handler_records_missing_frame_without_publishing_object(self) -> None:
        events = RecordingEventStore()
        published = []
        handler = MotionDecisionHandler(
            camera_id="foyer",
            events=events,
            detection_provider=lambda _event_at: (None, [{"label": "stale"}], ""),
            snapshot_writer=lambda _frame: self.fail("snapshot writer should not run"),
            object_serializer=json.dumps,
            event_callback=lambda event_type, payload: published.append((event_type, payload)),
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            {},
        )

        self.assertIsNone(outcome.object_detected)
        self.assertEqual(json.loads(events.payload["objects_json"])[0]["status"], "no_recorded_frame")
        self.assertEqual([event_type for event_type, _payload in published], ["incident"])

    def test_handler_reports_detector_failure_as_not_run(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{"status": "detector_unavailable", "error": "worker timed out"}],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            {},
        )

        self.assertIsNone(outcome.object_detected)
        self.assertEqual(json.loads(events.payload["objects_json"])[0]["status"], "detector_unavailable")

    def test_handler_owns_motion_audit_persistence_and_notification(self) -> None:
        events = RecordingEventStore()
        published = []
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (None, [], ""),
            snapshot_writer=lambda _frame: "",
            object_serializer=json.dumps,
            event_callback=lambda event_type, payload: published.append((event_type, payload)),
        )

        audit = handler.record_audit(
            related_event_id=42,
            event_at=datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            snapshot_path="snapshot.jpg",
            mode="audit",
            sensitivity="balanced",
            score=0.4,
            threshold=0.48,
            reason="edge_motion",
            object_detected=False,
            trigger_count=2,
            features={"persistence": 0.5},
        )

        self.assertEqual(audit["id"], 7)
        self.assertEqual(events.audit_payload["camera_id"], "gate")
        self.assertIsNone(events.audit_payload["event_id"])
        self.assertEqual(events.audit_payload["related_event_id"], 42)
        self.assertEqual(published, [("motion_audit", audit)])

    def test_post_commit_notification_failure_does_not_replay_incident(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (frame, [], ""),
            snapshot_writer=lambda _frame: "snapshot.jpg",
            object_serializer=json.dumps,
            event_callback=lambda _event_type, _payload: (_ for _ in ()).throw(
                RuntimeError("broker unavailable")
            ),
        )

        with self.assertLogs(
            "survng.app.motion_pipeline.decision_handler",
            level="ERROR",
        ):
            outcome = handler.handle(
                "onvif/motion",
                "motion",
                datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
                {},
            )

        self.assertEqual(outcome.event_id, 42)
        self.assertIsNotNone(events.payload)

    def test_post_commit_audit_notification_failure_is_nonfatal(self) -> None:
        events = RecordingEventStore()
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (None, [], ""),
            snapshot_writer=lambda _frame: "",
            object_serializer=json.dumps,
            event_callback=lambda _event_type, _payload: (_ for _ in ()).throw(
                RuntimeError("broker unavailable")
            ),
        )

        with self.assertLogs(
            "survng.app.motion_pipeline.decision_handler",
            level="ERROR",
        ):
            audit = handler.record_audit(
                event_at=datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
                snapshot_path="",
                mode="audit",
                sensitivity="balanced",
                score=0.4,
                threshold=0.48,
                reason="edge_motion",
                object_detected=False,
                trigger_count=1,
                features={},
            )

        self.assertEqual(audit["id"], 7)


if __name__ == "__main__":
    unittest.main()
