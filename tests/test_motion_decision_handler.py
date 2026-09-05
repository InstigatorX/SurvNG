from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np

from survng.app.face_candidates import FaceCandidate
from survng.app.object_activity import ObjectActivityAttributor
from survng.app.motion_pipeline import MotionDecisionHandler
from survng.app.motion_pipeline.decision_handler import motion_correlated_objects
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
    def test_off_frame_evidence_uses_its_detection_frame_geometry(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected = {
            "label": "person",
            "incident_eligible": True,
            "snapshot_visible": False,
            "box": {"x1": 100, "y1": 20, "x2": 140, "y2": 60},
            "detection_frame_width": 200,
            "detection_frame_height": 100,
            "temporal_track_observations": 2,
            "temporal_robust_new_appearance": True,
        }
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [detected],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 1, 22, 10, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.45, 0.1, 0.75, 0.7]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(outcome.detected_objects, ())
        self.assertEqual(outcome.motion_correlation["new_appearance_match_count"], 1)
        stored = json.loads(events.payload["objects_json"])
        stored_person = next(item for item in stored if item.get("label") == "person")
        self.assertFalse(stored_person["snapshot_visible"])
        self.assertEqual(stored_person["detection_frame_width"], 200)
        self.assertEqual(stored_person["motion_correlation"], "appearance")

    def test_depth_attribution_shadow_is_decision_scoped_and_preserves_result(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected = {
            "label": "person",
            "incident_eligible": True,
            "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 40},
            "depth_stats": {"median_m": 4.0},
            "frame_offset_s": 0.5,
            "temporal_sample_count": 1,
            "temporal_track_observations": 2,
        }

        correlated, summary = motion_correlated_objects(
            frame,
            [detected],
            {"features": {"motion_regions": [[0.1, 0.1, 0.5, 0.5]]}},
            depth_attribution_mode="shadow",
        )

        self.assertEqual(correlated, [])
        self.assertFalse(detected["incident_eligible"])
        self.assertEqual(detected["depth_attribution"], {
            "mode": "shadow",
            "decision_scoped": True,
            "median_m": 4.0,
            "valid_depth": True,
            "near_depth": True,
            "maximum_m": 10.0,
            "alignment_reliable": True,
            "spatial_match": True,
            "stable_geometry": False,
            "normal_motion_correlated": False,
            "would_admit": True,
            "provenance": {"frame_offset_s": 0.5, "temporal_sample_count": 1},
        })
        self.assertEqual(summary["correlated_object_count"], 0)

    def test_depth_attribution_is_off_by_default(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected = {
            "label": "person",
            "incident_eligible": True,
            "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 40},
            "depth_stats": {"median_m": 4.0},
        }

        motion_correlated_objects(
            frame,
            [detected],
            {"features": {"motion_regions": [[0.1, 0.1, 0.5, 0.5]]}},
        )

        self.assertNotIn("depth_attribution", detected)

    def test_route_duplicate_does_not_publish_or_start_downstream_side_effects(self) -> None:
        events = Mock()
        events.add_event.return_value = {"id": 42, "created": False}
        published: list[tuple[str, dict]] = []
        face_sink = Mock()
        admitted = Mock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="upper-garage",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{"label": "car", "confidence": 0.9, "incident_eligible": True}],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _at: "alternate.webp",
            object_serializer=json.dumps,
            event_callback=lambda kind, payload: published.append((kind, payload)),
            face_candidate_sink=face_sink,
            route_admission_callback=admitted,
        )
        qualification = {"features": {"route_detection_watch": {
            "source_camera_id": "lower-garage",
            "source_event_id": 99,
            "origin_camera_id": "gate",
            "origin_event_id": 10,
        }}}

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 27, 21, 13, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "route_target_already_admitted")
        add_kwargs = events.add_event.call_args.kwargs
        self.assertEqual(add_kwargs["route_origin_camera_id"], "gate")
        self.assertEqual(add_kwargs["route_origin_event_id"], 10)
        admitted.assert_called_once_with("upper-garage", "gate", 10)
        face_sink.assert_not_called()
        self.assertEqual(published, [])

    def test_same_intent_route_replay_recovers_canonical_event_refinement(self) -> None:
        events = Mock()
        events.add_event.return_value = {
            "id": 42,
            "created": False,
            "canonical_detection_intent_id": "route:gate:upper-garage:10",
        }
        published: list[tuple[str, dict]] = []
        face_sink = Mock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = RecordedDetectionResult(
            frame=frame,
            objects=[{"label": "car", "confidence": 0.9, "incident_eligible": True}],
            recording_path="recording.mp4",
            timings_ms={},
            refinement_pending=True,
        )
        handler = MotionDecisionHandler(
            camera_id="upper-garage",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, _at: "replayed.webp",
            object_serializer=json.dumps,
            event_callback=lambda kind, payload: published.append((kind, payload)),
            face_candidate_sink=face_sink,
        )
        qualification = {
            "detection_intent_id": "route:gate:upper-garage:10",
            "features": {"route_detection_watch": {
                "origin_camera_id": "gate",
                "origin_event_id": 10,
            }},
        }

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 27, 21, 13, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.refinement_event_id, 42)
        self.assertTrue(outcome.refinement_pending)
        self.assertEqual(outcome.rejection_reason, "route_target_already_admitted")
        face_sink.assert_not_called()
        self.assertEqual(published, [])

    def test_competing_route_duplicate_cannot_refine_winning_event(self) -> None:
        events = Mock()
        events.add_event.return_value = {
            "id": 42,
            "created": False,
            "canonical_detection_intent_id": "route:gate:upper-garage:winner",
        }
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = RecordedDetectionResult(
            frame=frame,
            objects=[{"label": "car", "confidence": 0.9, "incident_eligible": True}],
            recording_path="recording.mp4",
            timings_ms={},
            refinement_pending=True,
        )
        handler = MotionDecisionHandler(
            camera_id="upper-garage",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, _at: "alternate.webp",
            object_serializer=json.dumps,
        )
        qualification = {
            "detection_intent_id": "route:gate:upper-garage:alternate",
            "features": {"route_detection_watch": {
                "origin_camera_id": "gate",
                "origin_event_id": 10,
            }},
        }

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 27, 21, 13, tzinfo=timezone.utc),
            qualification,
            require_eligible_object=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertIsNone(outcome.refinement_event_id)
        self.assertFalse(outcome.refinement_pending)

    def test_winning_route_event_can_still_receive_refined_evidence(self) -> None:
        events = Mock()
        published: list[tuple[str, dict]] = []
        admitted = Mock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="upper-garage",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{"label": "car", "confidence": 0.94, "incident_eligible": True}],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _at: "refined.webp",
            object_serializer=json.dumps,
            event_callback=lambda kind, payload: published.append((kind, payload)),
            route_admission_callback=admitted,
        )
        qualification = {"features": {"route_detection_watch": {
            "origin_camera_id": "gate",
            "origin_event_id": 10,
        }}}

        outcome = handler.refine(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 27, 21, 13, tzinfo=timezone.utc),
            qualification,
            existing_event_id=42,
            require_eligible_object=True,
        )

        self.assertTrue(outcome.object_detected)
        self.assertEqual(outcome.event_id, 42)
        events.add_event.assert_not_called()
        events.refine_event_evidence.assert_called_once()
        admitted.assert_called_once_with("upper-garage", "gate", 10)
        self.assertEqual([kind for kind, _payload in published], ["object"])

    def test_failed_refinement_preserves_existing_provisional_evidence(self) -> None:
        events = Mock()
        snapshot_writer = Mock()
        failed = RecordedDetectionResult(
            frame=None,
            objects=[{"status": "detector_unavailable", "error": "busy"}],
            recording_path="",
            timings_ms={"workflow_ms": 1.0},
        )
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: failed,
            snapshot_writer=snapshot_writer,
            object_serializer=json.dumps,
        )

        outcome = handler.refine(
            "onvif/motion",
            "motion",
            datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc),
            {},
            existing_event_id=42,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertIsNone(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "refinement_unavailable_preserved")
        snapshot_writer.assert_not_called()
        events.refine_event_evidence.assert_not_called()

    def test_empty_recorded_refinement_cannot_replace_existing_cover(self) -> None:
        events = Mock()
        promoter = Mock()
        result = RecordedDetectionResult(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
            objects=[],
            recording_path="main.mp4",
            timings_ms={},
            frame_captured_at_epoch=1004.0,
            frame_source="recorded_main",
            frame_timestamp_exact=True,
        )
        handler = MotionDecisionHandler(
            camera_id="lower-garage",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, _at: "empty-main.webp",
            object_serializer=json.dumps,
            refinement_cover_promoter=promoter,
        )

        outcome = handler.refine(
            "adaptive/visual_backup",
            "backup",
            datetime.fromtimestamp(1000.0, timezone.utc),
            {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
            existing_event_id=42,
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "no_eligible_object")
        self.assertFalse(outcome.cover_promoted)
        events.refine_event_evidence.assert_not_called()
        promoter.assert_not_called()

    def test_uncorrelated_refinement_can_promote_compatible_cover_without_changing_decision(self) -> None:
        events = Mock()
        promoter = Mock(return_value={"id": 42})
        published = []
        writes = []
        captured_at = datetime(2026, 8, 15, 19, 8, 42, tzinfo=timezone.utc)
        result = RecordedDetectionResult(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
            objects=[{
                "label": "person",
                "confidence": 0.91,
                "confidence_eligible": True,
                "zone_eligible": True,
                "incident_eligible": True,
                "temporal_consensus": True,
                "temporal_track_observations": 3,
                "temporal_center_displacement_ratio": 0.001,
                "temporal_center_path_ratio": 0.002,
                "box": {"x1": 1400, "y1": 200, "x2": 1700, "y2": 900},
            }],
            recording_path="main.mp4",
            timings_ms={},
            frame_captured_at_epoch=captured_at.timestamp(),
            frame_source="recorded_main",
            frame_timestamp_exact=True,
        )
        handler = MotionDecisionHandler(
            camera_id="back-right",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, at: writes.append(at) or "main.webp",
            object_serializer=json.dumps,
            event_callback=lambda kind, payload: published.append((kind, payload)),
            spatial_alignment={"mode": "auto", "reliable": False, "confidence": 0.0},
            refinement_cover_promoter=promoter,
        )

        outcome = handler.refine(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 15, 19, 8, 37, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.5, 0.4, 0.8, 0.8]]}},
            existing_event_id=42,
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")
        self.assertTrue(outcome.cover_promoted)
        self.assertEqual(writes, [captured_at])
        kwargs = promoter.call_args.kwargs
        self.assertEqual(kwargs["frame_width"], 1920)
        self.assertEqual(kwargs["captured_at"], captured_at.timestamp())
        self.assertFalse(kwargs["cover_objects"][0]["incident_eligible"])
        self.assertEqual(published[0][0], "incident_update")
        events.refine_event_evidence.assert_not_called()

    def test_cover_promotion_failure_does_not_retry_completed_security_decision(self) -> None:
        result = RecordedDetectionResult(
            frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
            objects=[{
                "label": "person",
                "confidence": 0.9,
                "incident_eligible": True,
                "temporal_consensus": True,
                "temporal_track_observations": 3,
                "temporal_center_displacement_ratio": 0.0,
                "temporal_center_path_ratio": 0.0,
                "box": {"x1": 200, "y1": 100, "x2": 500, "y2": 900},
            }],
            recording_path="main.mp4",
            timings_ms={},
            frame_captured_at_epoch=1004.0,
            frame_source="recorded_main",
        )
        handler = MotionDecisionHandler(
            camera_id="back-right",
            events=Mock(),
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, _at: "main.webp",
            object_serializer=json.dumps,
            spatial_alignment={"mode": "auto", "reliable": False},
            refinement_cover_promoter=Mock(side_effect=OSError("storage busy")),
        )

        with self.assertLogs("survng.app.motion_pipeline.decision_handler", level="ERROR"):
            outcome = handler.refine(
                "adaptive/visual_backup",
                "backup",
                datetime.fromtimestamp(1000.0, timezone.utc),
                {"features": {"motion_regions": []}},
                existing_event_id=42,
                require_eligible_object=True,
                require_motion_correlation=True,
            )

        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")
        self.assertEqual(
            outcome.cover_promotion_reason,
            "refinement_cover_promotion_failed",
        )

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
                frame=frame[10:80, 20:90].copy(),
                box={"x1": 10, "y1": 10, "x2": 60, "y2": 60},
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

    def test_handler_retains_candidates_while_marking_provisional_face_evidence(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        candidate = FaceCandidate(
            track_id="face-1",
            rank=1,
            offset_seconds=0.0,
            frame=frame,
            box={"x1": 5, "y1": 5, "x2": 35, "y2": 35},
            confidence=0.9,
            quality_score=0.8,
            sharpness_score=0.8,
            exposure_score=0.8,
            edge_clearance_ratio=0.1,
            detection_source="dedicated_face",
        )
        result = RecordedDetectionResult(
            frame=frame,
            objects=[{"label": "person", "confidence": 0.9, "incident_eligible": True}],
            recording_path="recording.mp4",
            timings_ms={},
            refinement_pending=True,
            face_candidates=(candidate,),
        )
        sink = Mock()
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: result,
            snapshot_writer=lambda _frame, _at: "snapshot.webp",
            object_serializer=json.dumps,
            face_candidate_sink=sink,
        )

        handler.handle(
            "onvif/motion",
            "motion",
            datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            {},
        )

        sink.assert_called_once()
        stored = json.loads(events.payload["objects_json"])
        self.assertIn("face_evidence_pending", {item.get("status") for item in stored})

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

    def test_untrusted_main_sub_alignment_fails_open_without_temporal_evidence(self) -> None:
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
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
            spatial_alignment={
                "mode": "auto",
                "reliable": False,
                "confidence": 0.0,
            },
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertEqual(outcome.event_id, 42)
        self.assertEqual(
            outcome.detected_objects[0]["motion_correlation"],
            "alignment_unverified",
        )
        self.assertFalse(outcome.motion_correlation["alignment_reliable"])

    def test_untrusted_alignment_still_rejects_temporally_stationary_object(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="gate",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "car",
                    "confidence": 0.96,
                    "incident_eligible": True,
                    "box": {"x1": 60, "y1": 50, "x2": 90, "y2": 85},
                    "temporal_track_observations": 3,
                    "temporal_center_displacement_ratio": 0.001,
                    "temporal_center_path_ratio": 0.002,
                    "temporal_newly_appeared": False,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
            spatial_alignment={"mode": "untrusted", "reliable": False},
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.6, 0.5, 0.9, 0.85]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")

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

    def test_ema_rescue_accepts_robust_appearance_inside_reliably_aligned_region(self) -> None:
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
                    "temporal_robust_new_appearance": True,
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

    def test_ema_rescue_rejects_new_appearance_without_robust_evidence(self) -> None:
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

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.rejection_reason, "object_not_motion_correlated")

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

    def test_low_confidence_causal_movement_cannot_override_threshold(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.55,
                    "incident_eligible": False,
                    "spatial_zone_eligible": True,
                    "semantic_tier": "rescue_candidate",
                    "temporal_rescue_candidate": True,
                    "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 70},
                    "temporal_track_observations": 3,
                    "temporal_center_displacement_ratio": 0.025,
                    "temporal_center_path_ratio": 0.03,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
            spatial_alignment={"reliable": False, "mode": "untrusted"},
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
            {"features": {"motion_regions": []}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.detected_objects, ())

    def test_semantic_rescue_cannot_override_missing_qualifying_confirmation(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.9,
                    "confidence_eligible": True,
                    "incident_eligible": False,
                    "spatial_zone_eligible": True,
                    "semantic_tier": "rescue_candidate",
                    "temporal_rescue_candidate": True,
                    "temporal_incident_observations": 1,
                    "temporal_required_observations": 2,
                    "box": {"x1": 20, "y1": 20, "x2": 40, "y2": 70},
                    "temporal_track_observations": 3,
                    "temporal_center_displacement_ratio": 0.025,
                    "temporal_center_path_ratio": 0.03,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
            spatial_alignment={"reliable": False, "mode": "untrusted"},
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.1, 0.1, 0.3, 0.3]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertFalse(outcome.object_detected)
        self.assertEqual(outcome.rejection_reason, "no_eligible_object")

    def test_low_confidence_stable_appearance_is_not_rescued_when_alignment_is_untrusted(self) -> None:
        events = RecordingEventStore()
        frame = np.zeros((2560, 1920, 3), dtype=np.uint8)
        handler = MotionDecisionHandler(
            camera_id="front-door",
            events=events,
            detection_provider=lambda _event_at: (
                frame,
                [{
                    "label": "person",
                    "confidence": 0.55,
                    "incident_eligible": False,
                    "spatial_zone_eligible": True,
                    "semantic_tier": "rescue_candidate",
                    "temporal_rescue_candidate": True,
                    "box": {"x1": 982, "y1": 1322, "x2": 1015, "y2": 1409},
                    "temporal_track_observations": 4,
                    "temporal_pretrigger_observations": 0,
                    "temporal_posttrigger_observations": 4,
                    "temporal_robust_new_appearance": True,
                    "temporal_center_displacement_ratio": 0.0013,
                    "temporal_center_path_ratio": 0.00266,
                }],
                "recording.mp4",
            ),
            snapshot_writer=lambda _frame, _event_at: "snapshot.jpg",
            object_serializer=json.dumps,
            spatial_alignment={"reliable": False, "mode": "untrusted"},
        )

        outcome = handler.handle(
            "adaptive/visual_backup",
            "backup",
            datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
            {"features": {"motion_regions": [[0.45, 0.45, 0.60, 0.65]]}},
            require_eligible_object=True,
            require_motion_correlation=True,
        )

        self.assertIsNone(outcome.event_id)
        self.assertEqual(outcome.rejection_reason, "no_eligible_object")

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
