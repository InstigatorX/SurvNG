from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from survng.app import main
from survng.app.config import AppConfig, CameraConfig
from survng.app.image_cache import LocalImageCache
from survng.app.state_events import StateEventBroker
from fastapi import HTTPException


class EventApiSerializationTest(unittest.TestCase):
    def test_cgroup_memory_separates_application_and_file_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group = root / "system.slice" / "survng.service"
            group.mkdir(parents=True)
            process_cgroup = root / "process.cgroup"
            process_cgroup.write_text("0::/system.slice/survng.service\n", encoding="utf-8")
            (group / "memory.current").write_text("9000\n", encoding="utf-8")
            (group / "memory.stat").write_text(
                "anon 1500\nfile 7000\nshmem 500\ninactive_file 6200\nkernel 500\n",
                encoding="utf-8",
            )

            status = main._cgroup_memory_status(root, process_cgroup)

        self.assertEqual(status["total_bytes"], 9000)
        self.assertEqual(status["application_bytes"], 2000)
        self.assertEqual(status["file_cache_bytes"], 6500)
        self.assertEqual(status["reclaimable_file_cache_bytes"], 5700)
        self.assertEqual(status["kernel_bytes"], 500)

    def test_telemetry_history_throttles_samples_and_replaces_latest(self) -> None:
        history = main.deque(maxlen=360)
        state = {"last_sample_at": 0.0}
        with patch.object(main, "TELEMETRY_HISTORY", history), patch.object(main, "TELEMETRY_HISTORY_STATE", state):
            first = main._record_telemetry_history({"sampled_at": "first"}, 10.0)
            replaced = main._record_telemetry_history({"sampled_at": "replacement"}, 12.0)
            appended = main._record_telemetry_history({"sampled_at": "second"}, 16.0)

        self.assertEqual(first, [{"sampled_at": "first"}])
        self.assertEqual(replaced, [{"sampled_at": "replacement"}])
        self.assertEqual(appended, [{"sampled_at": "replacement"}, {"sampled_at": "second"}])

    def test_gpu_status_calculates_drm_engine_utilization_between_samples(self) -> None:
        detector = {
            "workers": {
                "object": {"worker_pid": 42, "worker_alive": True},
                "face": {"worker_pid": 0, "worker_alive": False},
            },
        }
        counters = {
            "engines": {"render": 2_000_000_000},
            "allocated_bytes": 128 * 1024 * 1024,
            "resident_bytes": 96 * 1024 * 1024,
            "driver": "i915",
        }

        with (
            patch.dict(main.GPU_SAMPLE, {"at": 8.0, "pids": (42,), "engines": {"render": 1_000_000_000}}, clear=True),
            patch.object(main, "_drm_worker_counters", return_value=counters),
            patch.object(main, "_read_integer", return_value=None),
            patch.object(main.time, "monotonic", return_value=10.0),
        ):
            status = main._gpu_status(detector)

        self.assertEqual(status["utilization_percent"], 50.0)
        self.assertEqual(status["engine_utilization"], {"render": 50.0})
        self.assertEqual(status["resident_bytes"], 96 * 1024 * 1024)
        self.assertEqual(status["driver"], "i915")
        self.assertTrue(status["sample_ready"])

    def test_telemetry_combines_history_with_runtime_camera_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            activity = {
                "hours": 24,
                "last_hour": {"events": 1},
                "last_24h": {"events": 2},
                "hourly": [],
                "by_camera": {
                    "gate": {
                        "last_hour": {"events": 1},
                        "last_24h": {"events": 2},
                    },
                },
            }
            fake_manager = SimpleNamespace(
                storage_dir=root,
                database_dir=root,
                events=SimpleNamespace(telemetry_activity=Mock(return_value=activity)),
                detector_status=Mock(return_value={"runtime": {"total_inferences": 8}}),
                statuses=Mock(return_value=[{
                    "id": "gate",
                    "name": "Gate",
                    "connected": True,
                    "recording": True,
                    "detection_enabled": True,
                    "onvif_enabled": True,
                    "onvif_connected": True,
                    "onvif_notifications_received": 12,
                    "onvif_motion_events_received": 4,
                    "motion_qualification": {"passed": 3, "audit_rejected": 2},
                    "object_tracking": {
                        "reid_attempts": 7,
                        "reid_successes": 6,
                        "reid_failures": 1,
                        "reid_average_ms": 4.25,
                        "reid_attempts_by_label": {"car": 7},
                        "reid_attempts_by_reason": {
                            "track_seed": 2,
                            "geometry_recovery": 5,
                        },
                        "reid_recoveries": 2,
                        "reid_recoveries_by_label": {"car": 2},
                        "reid_avoided_geometry_matches": 19,
                        "reid_avoided_by_label": {"car": 19},
                    },
                }]),
            )

            with patch.object(main, "manager", fake_manager), patch.object(main.os, "getloadavg", return_value=(1.0, 0.5, 0.25)):
                payload = main.telemetry(hours=24)

            self.assertEqual(payload["activity"], activity)
            self.assertEqual(payload["cameras"][0]["activity"]["last_24h"]["events"], 2)
            self.assertEqual(payload["cameras"][0]["onvif"]["notifications"], 12)
            self.assertEqual(payload["cameras"][0]["motion"]["rejected"], 2)
            self.assertEqual(payload["cameras"][0]["tracking"]["reid_attempts"], 7)
            self.assertEqual(payload["cameras"][0]["tracking"]["reid_recoveries"], 2)
            self.assertEqual(
                payload["cameras"][0]["tracking"]["reid_avoided_geometry_matches"],
                19,
            )
            self.assertEqual(
                payload["cameras"][0]["tracking"]["reid_attempts_by_reason"]["geometry_recovery"],
                5,
            )
            fake_manager.events.telemetry_activity.assert_called_once_with(hours=24)

    def test_object_tracking_catalog_exposes_safe_default_and_optional_backend(self) -> None:
        catalog = main.object_tracking_catalog()
        implementations = {
            item["id"]: item for item in catalog["implementations"]
        }

        self.assertTrue(implementations["survng_hybrid"]["available"])
        self.assertIn("ultralytics_botsort", implementations)

    def test_tracking_comparison_uses_bounded_shared_incident_frames(self) -> None:
        event = {
            "id": 43,
            "camera_id": "gate",
            "created_at": "2026-07-27T12:17:07+00:00",
            "objects_json": "[]",
        }
        active_config = AppConfig(
            cameras=[CameraConfig(
                id="gate",
                name="Gate",
                stream_url="rtsp://camera.invalid/main",
            )],
        )
        event_store = SimpleNamespace(
            get=lambda _event_id: event,
            save_tracking_comparison=Mock(return_value={"id": 8, "verdict": ""}),
            tracking_comparison_summary=Mock(return_value={"total": 1, "reviewed": 0}),
        )
        active_manager = SimpleNamespace(
            events=event_store,
            detector=object(),
            person_reidentifier=object(),
        )
        limiter = SimpleNamespace(acquire=Mock(return_value=True), release=Mock())
        comparison = {
            "frames_processed": 4,
            "engines": {"survng_hybrid": {}, "ultralytics_botsort": {}},
        }
        runner = SimpleNamespace(run=Mock(return_value=comparison))

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            patch.object(main, "TRACKING_COMPARISON_LIMITER", limiter),
            patch.object(main, "ultralytics_botsort_dependency_status", return_value={"available": True, "reason": ""}),
            patch.object(main, "_ensure_event_clip", return_value=Path("comparison.mp4")) as ensure_clip,
            patch.object(main, "sampled_video_frames", return_value=[(1.0, np.zeros((2, 2, 3), dtype=np.uint8))]) as sampled_frames,
            patch.object(main, "TrackingComparisonRunner", return_value=runner),
        ):
            result = main.compare_event_tracking(43, duration_seconds=200.0)

        self.assertEqual(result["frames_processed"], 4)
        self.assertEqual(result["comparison_id"], 8)
        self.assertEqual(result["requested_duration_seconds"], 30.0)
        saved_result = event_store.save_tracking_comparison.call_args.kwargs["result"]
        self.assertNotIn("tracks", saved_result["engines"]["survng_hybrid"])
        ensure_clip.assert_called_once()
        self.assertEqual(sampled_frames.call_args.kwargs["ffmpeg_path"], active_config.ffmpeg_path)
        self.assertEqual(sampled_frames.call_args.kwargs["maximum_width"], 640)
        runner.run.assert_called_once()
        limiter.release.assert_called_once_with()

    def test_tracking_comparison_rejects_missing_optional_backend_without_work(self) -> None:
        with patch.object(
            main,
            "ultralytics_botsort_dependency_status",
            return_value={"available": False, "reason": "not installed"},
        ):
            with self.assertRaises(HTTPException) as unavailable:
                main.compare_event_tracking(43)

        self.assertEqual(unavailable.exception.status_code, 503)

    def test_tracking_comparison_history_and_verdict_api_use_event_store(self) -> None:
        events = SimpleNamespace(
            tracking_comparison_history=Mock(return_value=[{"id": 4}]),
            tracking_comparison_summary=Mock(return_value={"total": 1, "reviewed": 1}),
            set_tracking_comparison_verdict=Mock(return_value={"id": 4, "camera_id": "gate"}),
        )
        with patch.object(main, "manager", SimpleNamespace(events=events)):
            history = main.tracking_comparison_history(camera_id="gate", limit=7)
            verdict = main.update_tracking_comparison_verdict(
                4,
                main.TrackingComparisonVerdictRequest(verdict="inconclusive"),
            )

        self.assertEqual(history["items"], [{"id": 4}])
        events.tracking_comparison_history.assert_called_once_with(camera_id="gate", limit=7)
        events.set_tracking_comparison_verdict.assert_called_once_with(4, "inconclusive")
        self.assertEqual(verdict["comparison"]["camera_id"], "gate")

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

    def test_sampled_suppression_context_records_that_object_detection_ran(self) -> None:
        config = AppConfig(cameras=[CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://camera.invalid/main",
        )])
        events = SimpleNamespace(
            get=lambda _event_id: None,
            motion_audits=lambda **_kwargs: ([], 0),
            for_camera_range=lambda *_args, **_kwargs: [],
        )

        context = main._audit_ai_context(
            {
                "id": 8,
                "camera_id": "gate",
                "features_json": json.dumps({"suppression_verification": True}),
                "created_at": "2026-07-27T12:17:16+00:00",
                "reason": "stationary_foreground",
                "event_id": None,
                "object_detected": 0,
            },
            config,
            SimpleNamespace(events=events),
        )

        self.assertTrue(context["decision_outcome"]["object_detection_ran"])
        self.assertTrue(context["decision_outcome"]["object_detection_completed"])
        self.assertFalse(context["decision_outcome"]["filtered_before_object_detection"])

    def test_motion_audit_ai_context_preserves_temporal_object_evidence(self) -> None:
        config = AppConfig.model_validate({
            "detector": {
                "event_confirmation_frames": 2,
                "event_class_confirmation_frames": {"robot_lawnmower": 3},
            },
            "cameras": [{
                "id": "back-middle",
                "name": "Back Middle",
                "stream_url": "rtsp://camera.invalid/main",
            }],
        })
        event = {
            "objects_json": json.dumps([{
                "label": "robot_lawnmower",
                "confidence": 0.81,
                "incident_eligible": False,
                "temporal_consensus": False,
                "temporal_sample_offset_seconds": -0.5,
                "temporal_observations": 2,
                "temporal_track_observations": 3,
                "temporal_incident_observations": 1,
                "temporal_required_observations": 3,
                "temporal_samples": 3,
                "temporal_peak_confidence": 0.91,
                "temporal_label_votes": {"robot_lawnmower": 2, "car": 1},
            }]),
        }
        events = SimpleNamespace(
            get=lambda _event_id: event,
            motion_audits=lambda **_kwargs: ([], 0),
            for_camera_range=lambda *_args, **_kwargs: [],
        )

        context = main._audit_ai_context(
            {
                "id": 9,
                "camera_id": "back-middle",
                "features_json": "{}",
                "created_at": "2026-07-28T22:48:15+00:00",
                "reason": "qualified",
                "event_id": 33,
                "object_detected": 0,
            },
            config,
            SimpleNamespace(events=events),
        )

        detected = context["detected_objects"][0]
        self.assertEqual(detected["temporal_observations"], 2)
        self.assertEqual(detected["temporal_incident_observations"], 1)
        self.assertEqual(detected["temporal_required_observations"], 3)
        self.assertEqual(detected["temporal_label_votes"]["car"], 1)
        self.assertEqual(context["effective_settings"]["object_confirmation_frames"], 2)
        self.assertEqual(
            context["effective_settings"]["object_class_confirmation_frames"],
            {"robot_lawnmower": 3},
        )

    def test_manual_camera_review_starts_a_background_job(self) -> None:
        config = AppConfig.model_validate({
            "audit_ai": {"enabled": True, "api_key": "secret"},
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://camera.invalid/main",
            }],
        })
        events = SimpleNamespace(
            motion_audits=lambda **_kwargs: ([{"id": 1, "camera_id": "gate"}], 1),
            create_motion_ai_review=lambda camera_id, count: {
                "id": 17,
                "camera_id": camera_id,
                "status": "queued",
                "audits_considered": count,
            },
        )
        limiter = SimpleNamespace(acquire=Mock(return_value=True), release=Mock())
        thread = SimpleNamespace(start=Mock())
        manager = SimpleNamespace(events=events)

        with (
            patch.object(main, "config", config),
            patch.object(main, "manager", manager),
            patch.object(main, "AUDIT_AI_LIMITER", limiter),
            patch.object(main.threading, "Thread", return_value=thread) as thread_factory,
        ):
            review = main.start_motion_ai_review(main.MotionAiReviewRequest(camera_id="gate"))

        self.assertEqual(review["id"], 17)
        limiter.acquire.assert_called_once_with(blocking=False)
        thread.start.assert_called_once_with()
        self.assertEqual(thread_factory.call_args.kwargs["name"], "motion-ai-review-gate")

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

            async def inline_to_thread(function, *args, **kwargs):
                # This test owns a synthetic manager and verifies the snapshot
                # cursor boundary, not asyncio's executor. Running inline keeps
                # the race deterministic and avoids leaking an executor thread
                # when an isolated test runner is interrupted.
                return function(*args, **kwargs)

            with (
                patch.object(main, "manager", manager),
                patch.object(main, "system_status", return_value={}),
                patch.object(main.asyncio, "to_thread", new=inline_to_thread),
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

    def test_event_thumbnail_is_resized_and_cached_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "1.jpg"
            snapshot.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((600, 1200, 3), dtype=np.uint8)))
            fake_manager = SimpleNamespace(
                storage_dir=root,
                image_cache=LocalImageCache(root / "cache"),
                events=SimpleNamespace(get=lambda _event_id: {"id": 1, "snapshot_path": str(snapshot)}),
            )

            with patch.object(main, "manager", fake_manager):
                first = main.event_thumbnail(1, width=320, quality=80)
                with patch.object(main.cv2, "imread", side_effect=AssertionError("cache miss")):
                    second = main.event_thumbnail(1, width=320, quality=80)

            cached = cv2.imread(str(first.path))
            self.assertIsNotNone(cached)
            self.assertEqual(cached.shape[1], 320)
            self.assertEqual(first.path, second.path)
            self.assertTrue(str(first.path).startswith(str(root / "cache")))

    def test_face_crop_is_cached_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "1.jpg"
            snapshot.parent.mkdir(parents=True)
            frame = np.zeros((300, 400, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(snapshot), frame))
            fake_manager = SimpleNamespace(
                image_cache=LocalImageCache(root / "cache"),
                faces=SimpleNamespace(
                    snapshot_path=lambda _observation_id: (
                        snapshot,
                        {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
                    )
                ),
            )

            with patch.object(main, "manager", fake_manager):
                first = main.face_crop(7, padding=0.2)
                with patch.object(main.cv2, "imread", side_effect=AssertionError("cache miss")):
                    second = main.face_crop(7, padding=0.2)

            crop = cv2.imread(str(first.path))
            self.assertIsNotNone(crop)
            self.assertEqual(crop.shape[:2], (140, 140))
            self.assertEqual(first.path, second.path)

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
                {"label": "ghost", "confidence": float("inf")},
                {"label": "shadow", "confidence": float("nan")},
                {"label": "car", "confidence": 0.8, "zones": ["driveway"]},
            ]),
        })

        self.assertEqual(row["labels"], ["car"])
        self.assertEqual(row["zones"], ["driveway"])
        self.assertEqual(len(row["objects"]), 4)

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

    def test_object_tracking_extends_same_incident_and_exposes_track_ids(self) -> None:
        event = main._event_row({
            "id": 43,
            "camera_id": "gate",
            "created_at": "2026-07-27T12:17:07+00:00",
            "objects_json": json.dumps([
                {
                    "label": "person",
                    "confidence": 0.9,
                    "incident_eligible": True,
                    "track_id": 1,
                    "track_state": "confirmed",
                    "track_observations": 4,
                },
                {
                    "status": "object_tracking",
                    "object_tracking": {
                        "state": "complete",
                        "updated_at": "2026-07-27T12:17:15+00:00",
                        "tracks": [{
                            "track_id": 1,
                            "label": "person",
                            "state": "confirmed",
                            "observations": 4,
                        }, {
                            "track_id": 2,
                            "label": "car",
                            "state": "confirmed",
                            "observations": 2,
                            "zones": ["driveway"],
                        }],
                    },
                },
            ]),
        })

        incident = main._incident_row("gate", [event])

        self.assertEqual(incident["event_count"], 1)
        self.assertEqual(incident["duration_seconds"], 8.0)
        self.assertEqual(incident["end_at"], "2026-07-27T12:17:15+00:00")
        self.assertEqual(incident["object_tracking"]["tracks"][0]["track_id"], 1)
        self.assertEqual(incident["events"][0]["objects"][0]["track_id"], 1)
        self.assertEqual(incident["labels"], ["car", "person"])
        self.assertEqual(incident["zones"], ["driveway"])

    def test_capacity_skip_does_not_extend_incident_duration(self) -> None:
        event = main._event_row({
            "id": 43,
            "camera_id": "gate",
            "created_at": "2026-07-27T12:17:07+00:00",
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.9,
                "incident_eligible": True,
            }, {
                "status": "object_tracking",
                "object_tracking": {
                    "state": "skipped_capacity",
                    "updated_at": "2026-07-27T12:17:17+00:00",
                    "tracks": [],
                },
            }]),
        })

        incident = main._incident_row("gate", [event])

        self.assertEqual(incident["duration_seconds"], 0.0)
        self.assertEqual(incident["end_at"], "2026-07-27T12:17:07+00:00")

    def test_incident_tracking_matches_representative_event_not_latest_event(self) -> None:
        representative = {
            "id": 43,
            "camera_id": "foyer",
            "created_at": "2026-07-27T12:17:07+00:00",
            "has_objects": True,
            "labels": ["person"],
            "zones": [],
            "objects": [{"label": "person", "confidence": 0.95}],
            "object_tracking": {"state": "complete", "tracks": [{"track_id": 1}]},
        }
        later = {
            "id": 44,
            "camera_id": "foyer",
            "created_at": "2026-07-27T12:17:20+00:00",
            "has_objects": True,
            "labels": ["person"],
            "zones": [],
            "objects": [{"label": "person", "confidence": 0.80}],
            "object_tracking": {
                "state": "complete",
                "tracks": [{"track_id": index} for index in range(1, 5)],
            },
        }

        incident = main._incident_row("foyer", [representative, later])

        self.assertEqual(incident["representative_event_id"], 43)
        self.assertEqual(len(incident["object_tracking"]["tracks"]), 1)

    def test_incident_payload_keeps_annotation_fields_without_detector_diagnostics(self) -> None:
        payload = main._incident_event_payload({
            "id": 42,
            "topic": "private/topic",
            "message": "large raw payload",
            "objects": [{
                "label": "person",
                "confidence": 0.9,
                "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                "zones": ["yard"],
                "mask_polygon": [[1, 2], [3, 4]],
                "incident_eligible": True,
                "track_id": 3,
                "track_state": "confirmed",
                "track_observations": 5,
                "temporal_consensus": True,
                "temporal_sample_offset_seconds": 0.5,
                "temporal_observations": 3,
                "temporal_track_observations": 4,
                "temporal_incident_observations": 2,
                "temporal_required_observations": 2,
                "temporal_samples": 5,
                "temporal_peak_confidence": 0.95,
                "temporal_label_votes": {"person": 3, "dog": 1},
                "raw_detection_tensor": [1, 2, 3],
                "frame_source": "diagnostic-only",
            }],
        })

        self.assertNotIn("topic", payload)
        self.assertNotIn("message", payload)
        self.assertEqual(payload["objects"][0]["label"], "person")
        self.assertEqual(payload["objects"][0]["zones"], ["yard"])
        self.assertEqual(payload["objects"][0]["track_id"], 3)
        self.assertEqual(payload["objects"][0]["temporal_observations"], 3)
        self.assertEqual(payload["objects"][0]["temporal_incident_observations"], 2)
        self.assertEqual(payload["objects"][0]["temporal_sample_offset_seconds"], 0.5)
        self.assertEqual(payload["objects"][0]["temporal_required_observations"], 2)
        self.assertNotIn("raw_detection_tensor", payload["objects"][0])
        self.assertNotIn("frame_source", payload["objects"][0])

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
