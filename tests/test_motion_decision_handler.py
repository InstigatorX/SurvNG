from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

import numpy as np

from survng.app.face_candidates import FaceCandidate
from survng.app.object_activity import ObjectActivityAttributor
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.object_detection import RecordedDetectionResult


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
    def test_handler_persists_cropped_temporal_face_candidates(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[20:70, 30:80] = 255
        result = RecordedDetectionResult(
            frame=frame,
            objects=[{"label": "person", "confidence": 0.9, "incident_eligible": True}],
            recording_path="recording.mp4",
            timings_ms={},
            face_candidates=(FaceCandidate(
                track_id="face-1",
                rank=1,
                offset_seconds=0.5,
                frame=frame,
                box={"x1": 30, "y1": 20, "x2": 80, "y2": 70},
                confidence=0.88,
                quality_score=0.75,
                sharpness_score=0.7,
                exposure_score=0.8,
                edge_clearance_ratio=0.1,
                detection_source="dedicated_face",
            ),),
        )
        writes = []
        ingested = []
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda value, at: (
                writes.append((value.shape, at)) or f"snapshot-{len(writes)}.webp"
            ),
            object_serializer=json.dumps,
            face_candidate_sink=lambda *args: ingested.append(args) or 1,
        )
        event_at = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

        outcome = handler.handle("onvif/motion", "motion", event_at, {})

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(writes[0][0], (100, 160, 3))
        self.assertEqual(writes[1][0], (70, 70, 3))
        self.assertEqual(writes[1][1], event_at.replace(microsecond=500000))
        self.assertEqual(ingested[0][0:3], (42, "front-door", event_at.isoformat()))
        self.assertEqual(ingested[0][3][0]["track_id"], "face-1")
        self.assertEqual(
            ingested[0][3][0]["box"],
            {"x1": 10.0, "y1": 10.0, "x2": 60.0, "y2": 60.0},
        )

    def test_disabled_object_activity_does_not_evaluate_or_learn(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected = {
            "label": "car",
            "confidence": 0.91,
            "incident_eligible": True,
            "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 40},
            "detection_frame_width": 100,
            "detection_frame_height": 100,
            "temporal_consensus": True,
            "temporal_track_observations": 4,
            "temporal_center_displacement_ratio": 0.001,
            "temporal_center_path_ratio": 0.003,
        }
        attributor = ObjectActivityAttributor("off")
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (frame, [detected], "recording.mp4"),
            snapshot_writer=lambda _frame, _event_at: "snapshot.webp",
            object_serializer=json.dumps,
            activity_attributor=attributor,
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 8, 8, 5, 0, tzinfo=timezone.utc),
            {},
        )

        self.assertIsNone(outcome.object_activity)
        self.assertEqual(attributor.status()["evaluated"], 0)
        self.assertEqual(attributor.status()["scene_context_memory_entries"], 0)
        stored = json.loads(events.payload["objects_json"])
        self.assertNotIn("activity_role", stored[0])

    def test_object_activity_context_is_persisted_but_not_incident_eligible(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected = {
            "label": "car",
            "confidence": 0.91,
            "incident_eligible": True,
            "box": {"x1": 10, "y1": 10, "x2": 40, "y2": 40},
            "detection_frame_width": 100,
            "detection_frame_height": 100,
            "temporal_consensus": True,
            "temporal_track_observations": 4,
            "temporal_pretrigger_observations": 2,
            "temporal_posttrigger_observations": 2,
            "temporal_center_displacement_ratio": 0.001,
            "temporal_center_path_ratio": 0.003,
        }
        attributor = ObjectActivityAttributor("enforce")
        event_at = datetime(2026, 8, 8, 5, 0, tzinfo=timezone.utc)
        for index in range(2):
            attributor.admit(
                [detected],
                {},
                event_key=f"prior-{index}",
                observed_at_epoch=event_at.timestamp() - 120 + index * 60,
            )
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (frame, [detected], "recording.mp4"),
            snapshot_writer=lambda _frame, _event_at: "snapshot.webp",
            object_serializer=json.dumps,
            activity_attributor=attributor,
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            event_at,
            {},
        )

        assert outcome.event_id == 42
        assert outcome.object_detected is False
        assert outcome.object_activity is not None
        assert outcome.object_activity["scene_context"] == 1
        stored = json.loads(events.payload["objects_json"])
        assert stored[0]["activity_role"] == "scene_context"
        assert stored[0]["incident_eligible"] is False
        assert detected["incident_eligible"] is True

    def test_handler_dispatches_detection_persists_event_and_publishes_outputs(self) -> None:
        events = RecordingEventStore()
        published = []
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        qualification = {"borderline_candidate": True, "would_suppress": True}
        snapshot_times = []
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{"label": "car", "confidence": 0.8, "incident_eligible": True}],
                "recording.mp4",
            ),
            snapshot_writer=lambda value, event_at: (
                snapshot_times.append(event_at) or ("snapshot.jpg" if value is frame else "")
            ),
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
        self.assertEqual(snapshot_times, [datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)])
        self.assertIn("processed_at", qualification)
        self.assertIn("object_detection_workflow_ms", qualification)
        self.assertNotIn("object_detection_duration_ms", qualification)
        self.assertIn("event_processing_delay_seconds", qualification)
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
            snapshot_writer=lambda _frame, _event_at: self.fail("snapshot writer should not run"),
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

    def test_verification_without_eligible_object_does_not_create_incident(self) -> None:
        events = RecordingEventStore()
        published = []
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        qualification = {
            "suppression_verification_candidate": True,
            "effective_accepted": True,
            "would_suppress": True,
        }
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (frame, [], "recording.mp4"),
            snapshot_writer=lambda _frame, _event_at: "verification.jpg",
            object_serializer=json.dumps,
            event_callback=lambda event_type, payload: published.append((event_type, payload)),
        )

        outcome = handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.snapshot_path, "verification.jpg")
        self.assertFalse(outcome.object_detected)
        self.assertFalse(qualification["suppression_verification_rescued"])
        self.assertFalse(qualification["effective_accepted"])
        self.assertTrue(qualification["would_suppress"])
        self.assertIsNone(events.payload)
        self.assertEqual(published, [])

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
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
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

    def test_ema_rescue_rejects_stationary_object_unrelated_to_motion(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "car",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "box": {"x1": 70, "y1": 70, "x2": 90, "y2": 90},
                    "temporal_center_displacement_ratio": 0.001,
                    "temporal_center_path_ratio": 0.002,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )
        qualification = {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}}

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")
        self.assertEqual(outcome.motion_correlation["eligible_object_count"], 1)
        self.assertEqual(outcome.motion_correlation["correlated_object_count"], 0)
        self.assertIsNone(events.payload)

    def test_ema_rescue_accepts_spatially_correlated_object(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "car",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "box": {"x1": 25, "y1": 20, "x2": 50, "y2": 50},
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )
        qualification = {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}}

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertTrue(outcome.object_detected)
        self.assertEqual(outcome.detected_objects[0]["motion_correlation"], "spatial")

    def test_ema_rescue_rejects_spatially_overlapping_stationary_object(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="lower-garage",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "car",
                    "confidence": 0.96,
                    "incident_eligible": True,
                    "box": {"x1": 60, "y1": 47, "x2": 92, "y2": 82},
                    "temporal_track_observations": 2,
                    "temporal_center_displacement_ratio": 0.0012,
                    "temporal_center_path_ratio": 0.0012,
                    "temporal_newly_appeared": False,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 1, 22, 10, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.60, 0.47, 0.78, 0.82]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")
        self.assertEqual(outcome.motion_correlation["stationary_spatial_rejection_count"], 1)

    def test_ema_rescue_accepts_spatial_object_with_substantial_reversal_path(self) -> None:
        """A person can turn around during a short sample window."""
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.91,
                    "incident_eligible": True,
                    "box": {"x1": 35, "y1": 30, "x2": 60, "y2": 90},
                    "temporal_track_observations": 3,
                    "temporal_center_displacement_ratio": 0.002,
                    "temporal_center_path_ratio": 0.08,
                    "temporal_newly_appeared": False,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 5, 20, 44, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.30, 0.25, 0.65, 0.95]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(outcome.detected_objects[0]["motion_correlation"], "spatial_path")
        self.assertEqual(outcome.motion_correlation["temporal_path_match_count"], 1)

    def test_ema_rescue_accepts_new_object_appearing_inside_motion_region(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "car",
                    "confidence": 0.91,
                    "incident_eligible": True,
                    "box": {"x1": 20, "y1": 20, "x2": 55, "y2": 65},
                    "temporal_track_observations": 2,
                    "temporal_center_displacement_ratio": 0.001,
                    "temporal_center_path_ratio": 0.001,
                    "temporal_newly_appeared": True,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 1, 22, 10, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.15, 0.15, 0.60, 0.70]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(outcome.detected_objects[0]["motion_correlation"], "appearance")
        self.assertEqual(outcome.motion_correlation["new_appearance_match_count"], 1)

    def test_ema_rescue_accepts_temporally_moving_object_outside_latest_region(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "box": {"x1": 70, "y1": 70, "x2": 90, "y2": 95},
                    "temporal_center_displacement_ratio": 0.02,
                    "temporal_center_path_ratio": 0.025,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )
        qualification = {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}}

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertTrue(outcome.object_detected)
        self.assertEqual(outcome.detected_objects[0]["motion_correlation"], "temporal")

    def test_ema_rescue_scales_temporal_movement_for_distant_object(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="back-right",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.86,
                    "incident_eligible": True,
                    "box": {"x1": 800, "y1": 150, "x2": 830, "y2": 210},
                    "temporal_track_observations": 2,
                    "temporal_center_displacement_ratio": 0.004,
                    "temporal_center_path_ratio": 0.004,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 1, 22, 10, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.1, 0.1, 0.2, 0.2]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(outcome.detected_objects[0]["motion_correlation"], "temporal")
        self.assertLess(outcome.detected_objects[0]["motion_correlation_threshold"], 0.004)

    def test_handler_owns_motion_audit_persistence_and_notification(self) -> None:
        events = RecordingEventStore()
        published = []
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (None, [], ""),
            snapshot_writer=lambda _frame, _event_at: "",
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
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
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
            snapshot_writer=lambda _frame, _event_at: "",
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
