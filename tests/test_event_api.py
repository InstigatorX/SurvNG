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
from survng.app.audit_ai import AuditAiChange
from survng.app.config import AppConfig, AuditAiConfig, CameraConfig
from survng.app.image_cache import LocalImageCache
from survng.app.state_events import StateEventBroker
from survng.app.semantic_search import SemanticSearchHit
from survng.app.system_telemetry import SystemTelemetryService
from survng.app.detection_routes import (
    TrackingComparisonVerdictRequest,
    _tracking_comparison_duration,
)
from survng.app.incident_presenter import (
    _best_incident_event,
    _event_row,
    _incident_event_payload,
    _incident_list_payload,
    _incident_row,
)
from survng.app.incident_queries import (
    _filter_incidents_by_person,
    _filter_incidents_by_event_type,
    _motion_audit_row,
)
from survng.app.intelligence_routes import AuditAiApplyRequest, MotionAiReviewRequest
from survng.app.semantic_routes import SemanticSearchRequest
from fastapi import HTTPException


class EventApiSerializationTest(unittest.TestCase):
    def test_person_filter_matches_confirmed_face_observation(self) -> None:
        incidents = [
            {"id": "incident-gate-7", "events": [{"id": 7}]},
            {"id": "incident-gate-8", "events": [{"id": 8}]},
        ]
        faces = SimpleNamespace(for_event_ids=Mock(return_value=[
            {"event_id": 7, "person_id": 42},
            {"event_id": 8, "person_id": None},
        ]))

        filtered = _filter_incidents_by_person(
            SimpleNamespace(faces=faces), incidents, 42
        )

        self.assertEqual(filtered, [incidents[0]])
        faces.for_event_ids.assert_called_once_with([7, 8])

    def test_semantic_search_deduplicates_evidence_by_event(self) -> None:
        hits = [
            SemanticSearchHit(7, "gate", "now", "object_crop", "car:0", "a.webp", "car", (1, 2, 3, 4), 0.91),
            SemanticSearchHit(7, "gate", "now", "full_frame", "frame", "a.webp", "", None, 0.85),
        ]
        events = SimpleNamespace(get_many=Mock(return_value=[{
            "id": 7, "camera_id": "gate", "kind": "object", "snapshot_path": "a.webp",
            "recording_path": "", "objects_json": "[]", "created_at": "now",
        }]))
        semantic = SimpleNamespace(search_text=Mock(return_value=hits))
        active = SimpleNamespace(
            semantic_search=semantic, events=events,
            config=AppConfig(base_path="/survng"),
        )
        with patch.object(main, "manager", active):
            response = main.semantic_search(SemanticSearchRequest(query="red car"))

        self.assertEqual(response["count"], 1)
        self.assertEqual(response["results"][0]["score"], 0.91)
        self.assertEqual(response["results"][0]["rank_score"], 0.91)
        self.assertEqual(
            response["results"][0]["match_strength"], "visual_similarity"
        )
        self.assertEqual(response["results"][0]["snapshot_url"], "/survng/api/events/7/snapshot.jpg")
        self.assertNotIn("objects_json", response["results"][0]["event"])
        self.assertEqual(
            response["results"][0]["event"],
            {"id": 7, "camera_id": "gate", "kind": "object", "created_at": "now"},
        )

    def test_motion_audit_apply_requires_bound_camera_recommendation(self) -> None:
        active_config = AppConfig(
            audit_ai=AuditAiConfig(allow_apply_recommendations=True),
            cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
        )
        camera = active_config.cameras[0]
        change = AuditAiChange(
            scope="camera",
            setting="sensitivity",
            value="high",
            reason="The reviewed audit supports this camera adjustment.",
        )
        fingerprint = main._intelligence_route_bundle.service._assistant_motion_config_fingerprint(active_config, camera)
        token = main._intelligence_route_bundle.service._issue_ai_recommendation_token(
            kind="motion_audit",
            record_id=7,
            camera_id="gate",
            configuration_fingerprint=fingerprint,
            changes=[change],
        )
        active_manager = SimpleNamespace(
            events=SimpleNamespace(get_motion_audit=lambda _audit_id: {"camera_id": "gate"}),
        )
        request = AuditAiApplyRequest(
            changes=[change],
            confirmed=True,
            configuration_fingerprint=fingerprint,
            recommendation_proof=token,
        )
        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            patch.object(
                main,
                "apply_config_update",
                return_value=(active_config, {
                    "camera_workers_restarted": True,
                    "apply_mode": "manager_reload",
                }),
            ) as apply_update,
        ):
            response = main._intelligence_route_bundle.service.motion_audit_ai_apply(7, request)

        self.assertTrue(response["ok"])
        apply_update.assert_called_once()

        with self.assertRaises(HTTPException) as raised:
            main._intelligence_route_bundle.service.motion_audit_ai_apply(
                7,
                request.model_copy(update={"confirmed": False}),
            )
        self.assertEqual(raised.exception.status_code, 400)

    def test_appearance_match_endpoint_bounds_window_and_keeps_vectors_private(self) -> None:
        events = SimpleNamespace(get=lambda event_id: {
            "id": event_id,
            "created_at": "2026-08-01T12:00:00+00:00",
        })
        appearance_index = SimpleNamespace(matches=Mock(return_value=[{
            "event_id": 8,
            "camera_id": "foyer",
            "similarity": 0.91,
            "visually_similar": True,
        }]))
        active_manager = SimpleNamespace(
            events=events,
            appearance_index=appearance_index,
        )
        with patch.object(main, "manager", active_manager):
            response = main.event_appearance_matches(
                7,
                hours=1000,
                limit=500,
            )

        self.assertEqual(response["hours"], 720.0)
        self.assertEqual(response["matches"][0]["similarity"], 0.91)
        self.assertNotIn("embedding", response["matches"][0])
        call = appearance_index.matches.call_args
        self.assertEqual(call.kwargs["limit"], 100)
        self.assertEqual(
            call.kwargs["start_at"],
            "2026-07-02T12:00:00+00:00",
        )

    def test_related_incidents_promotes_configured_directional_camera_route(self) -> None:
        car_objects = json.dumps([{"label": "car", "incident_eligible": True}])
        anchor = {
            "id": 7,
            "camera_id": "back-left",
            "created_at": "2026-08-05T21:52:05+00:00",
            "objects_json": car_objects,
        }
        candidate = {
            "id": 8,
            "camera_id": "gate",
            "created_at": "2026-08-05T21:52:09+00:00",
            "objects_json": car_objects,
        }
        app_config = AppConfig.model_validate({
            "cameras": [
                {"id": "back-left", "name": "Back Left", "stream_url": "rtsp://example.invalid/a"},
                {"id": "gate", "name": "Gate", "stream_url": "rtsp://example.invalid/b"},
            ],
            "detector": {"tracking": {
                "vehicle_reid_labels": ["car"],
                "camera_transition_routes": [{
                    "from_camera": "back-left",
                    "to_camera": "gate",
                    "min_seconds": 1,
                    "max_seconds": 10,
                    "name": "Back yard to gate",
                }],
            }},
        })
        events = SimpleNamespace(
            get=lambda event_id: anchor if event_id == 7 else None,
            between=Mock(return_value=[candidate, anchor]),
        )
        active_manager = SimpleNamespace(
            events=events,
            appearance_index=SimpleNamespace(matches=Mock(return_value=[])),
            config=app_config,
        )

        with (
            patch.object(main, "manager", active_manager),
            patch.object(
                main.INCIDENT_QUERIES,
                "resolve_event",
                return_value={"id": "incident-gate-8", "representative_event_id": 8},
            ),
        ):
            response = main.event_related_incidents(7)

        self.assertEqual(response["configured_routes"], 1)
        self.assertEqual(response["matches"][0]["relation_type"], "expected_route")
        self.assertEqual(response["matches"][0]["route_name"], "Back yard to gate")
        self.assertEqual(response["matches"][0]["sequence_delta_seconds"], 4.0)
        self.assertEqual(response["matches"][0]["incident_id"], "incident-gate-8")

    def test_related_incidents_collapses_duplicate_route_observations(self) -> None:
        person_objects = json.dumps([{"label": "person", "incident_eligible": True}])
        anchor = {
            "id": 7,
            "camera_id": "upper-garage",
            "created_at": "2026-08-05T21:52:05+00:00",
            "objects_json": person_objects,
        }
        candidates = [
            {
                "id": 8,
                "camera_id": "front-door",
                "created_at": "2026-08-05T21:52:08+00:00",
                "objects_json": person_objects,
                "recording_path": "/recordings/front-door/segment.mp4",
            },
            {
                "id": 9,
                "camera_id": "front-door",
                "created_at": "2026-08-05T21:52:08.500000+00:00",
                "objects_json": person_objects,
                "recording_path": "/recordings/front-door/segment.mp4",
            },
        ]
        app_config = AppConfig.model_validate({
            "cameras": [
                {"id": "upper-garage", "name": "Upper Garage", "stream_url": "rtsp://example.invalid/a"},
                {"id": "front-door", "name": "Front Door", "stream_url": "rtsp://example.invalid/b"},
            ],
            "detector": {"tracking": {
                "camera_transition_routes": [{
                    "from_camera": "upper-garage",
                    "to_camera": "front-door",
                    "min_seconds": 0,
                    "max_seconds": 30,
                }],
            }},
        })
        events = SimpleNamespace(
            get=lambda event_id: anchor if event_id == 7 else None,
            between=Mock(return_value=[*candidates, anchor]),
        )
        active_manager = SimpleNamespace(
            events=events,
            appearance_index=SimpleNamespace(matches=Mock(return_value=[])),
            config=app_config,
        )

        def resolve_event(_manager, event_id: int):
            return {
                "id": "incident-front-door-8",
                "representative_event_id": 8,
            } if event_id in {8, 9} else None

        with (
            patch.object(main, "manager", active_manager),
            patch.object(main.INCIDENT_QUERIES, "resolve_event", side_effect=resolve_event),
        ):
            response = main.event_related_incidents(7)

        self.assertEqual([match["event_id"] for match in response["matches"]], [8])
        self.assertEqual(response["matches"][0]["incident_id"], "incident-front-door-8")

    def test_incident_for_event_returns_complete_resolved_incident(self) -> None:
        expected = {
            "id": "incident-gate-7",
            "representative_event_id": 7,
            "events": [{"id": 7}, {"id": 8}],
        }
        active_manager = SimpleNamespace()
        with (
            patch.object(main, "manager", active_manager),
            patch.object(main.INCIDENT_QUERIES, "resolve_event", return_value=expected) as resolve,
        ):
            response = main.incident_for_event(7)

        self.assertEqual(response, expected)
        resolve.assert_called_once_with(active_manager, 7)

    def test_incident_for_event_returns_not_found_for_stale_match(self) -> None:
        with (
            patch.object(main, "manager", SimpleNamespace()),
            patch.object(main.INCIDENT_QUERIES, "resolve_event", return_value=None),
            self.assertRaises(HTTPException) as raised,
        ):
            main.incident_for_event(999)

        self.assertEqual(raised.exception.status_code, 404)

    def test_motion_audit_endpoint_filters_named_categories(self) -> None:
        events = SimpleNamespace(motion_audits=Mock(return_value=([], 0)))
        active_manager = SimpleNamespace(events=events, storage_dir=Path("/tmp/survng-test"), media_storage=None)
        with patch.object(main, "manager", active_manager):
            response = main._intelligence_route_bundle.service.motion_audit(category="visual_backup")

        self.assertEqual(response["total"], 0)
        events.motion_audits.assert_called_once_with(
            limit=24,
            offset=0,
            camera_id="",
            outcome="all",
            category="visual_backup",
        )
        events.motion_audits.reset_mock()
        with patch.object(main, "manager", active_manager):
            main._intelligence_route_bundle.service.motion_audit(
                category="active_followup"
            )
        events.motion_audits.assert_called_once_with(
            limit=24,
            offset=0,
            camera_id="",
            outcome="all",
            category="active_followup",
        )
        with self.assertRaises(HTTPException):
            main._intelligence_route_bundle.service.motion_audit(category="unexpected")

    def test_motion_audit_detail_supports_direct_deep_links(self) -> None:
        events = SimpleNamespace(get_motion_audit=Mock(return_value={
            "id": 41,
            "camera_id": "gate",
            "features_json": "{}",
            "snapshot_path": "",
            "object_detected": None,
        }))
        active_manager = SimpleNamespace(events=events, storage_dir=Path("/tmp/survng-test"), media_storage=None)
        with patch.object(main, "manager", active_manager):
            response = main._intelligence_route_bundle.service.motion_audit_detail(41)

        self.assertEqual(response["id"], 41)
        self.assertEqual(response["camera_id"], "gate")
        events.get_motion_audit.assert_called_once_with(41)

    def test_motion_audit_detail_returns_not_found(self) -> None:
        active_manager = SimpleNamespace(
            events=SimpleNamespace(get_motion_audit=Mock(return_value=None)),
            storage_dir=Path("/tmp/survng-test"),
            media_storage=None,
        )
        with (
            patch.object(main, "manager", active_manager),
            self.assertRaises(HTTPException) as raised,
        ):
            main._intelligence_route_bundle.service.motion_audit_detail(404)

        self.assertEqual(raised.exception.status_code, 404)

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

            status = SystemTelemetryService.cgroup_memory_status(root, process_cgroup)

        self.assertEqual(status["total_bytes"], 9000)
        self.assertEqual(status["working_set_bytes"], 3300)
        self.assertEqual(status["application_bytes"], 2000)
        self.assertEqual(status["file_cache_bytes"], 6500)
        self.assertEqual(status["reclaimable_file_cache_bytes"], 5700)
        self.assertEqual(status["kernel_bytes"], 500)

    def test_telemetry_history_throttles_samples_and_replaces_latest(self) -> None:
        service = SystemTelemetryService()
        first = service.record_history({"sampled_at": "first"}, 10.0)
        replaced = service.record_history({"sampled_at": "replacement"}, 12.0)
        appended = service.record_history({"sampled_at": "second"}, 16.0)

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

        service = SystemTelemetryService()
        service._gpu_sample = {
            "at": 8.0,
            "pids": (42,),
            "engines": {"render": 1_000_000_000},
        }
        with (
            patch.object(SystemTelemetryService, "drm_worker_counters", return_value=counters),
            patch.object(SystemTelemetryService, "read_integer", return_value=None),
            patch("survng.app.system_telemetry.time.monotonic", return_value=10.0),
        ):
            status = service.gpu_status(detector)

        self.assertEqual(status["utilization_percent"], 50.0)
        self.assertEqual(status["engine_utilization"], {"render": 50.0})
        self.assertEqual(status["resident_bytes"], 96 * 1024 * 1024)
        self.assertEqual(status["driver"], "i915")
        self.assertTrue(status["sample_ready"])

    def test_gpu_status_samples_every_object_worker_process(self) -> None:
        detector = {
            "workers": {
                "object": {
                    "worker_pid": 41,
                    "worker_pids": [41, 42],
                    "worker_alive": True,
                },
            },
        }
        service = SystemTelemetryService()
        with (
            patch.object(
                SystemTelemetryService,
                "drm_worker_counters",
                return_value={
                    "engines": {},
                    "allocated_bytes": 0,
                    "resident_bytes": 0,
                    "driver": "i915",
                },
            ) as counters,
            patch.object(SystemTelemetryService, "read_integer", return_value=None),
        ):
            status = service.gpu_status(detector)

        self.assertEqual(status["worker_pids"], [41, 42])
        self.assertEqual(
            {call.args[0] for call in counters.call_args_list},
            {41, 42},
        )

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
                runtime_monitor=SimpleNamespace(
                    diagnostic_status=Mock(return_value={"active": []})
                ),
                events=SimpleNamespace(
                    telemetry_activity=Mock(return_value=activity),
                    tracking_capacity_activity=Mock(return_value=[]),
                ),
                telemetry=SimpleNamespace(
                    operational_history=Mock(return_value=[]),
                    memory_history=Mock(return_value=[]),
                    sample_times=Mock(return_value=[]),
                    operational_event_history=Mock(return_value=[]),
                ),
                detector_status=Mock(return_value={"runtime": {"total_inferences": 8}}),
                semantic_search_status=Mock(return_value={
                    "enabled": True,
                    "state": "ready",
                    "device": "GPU",
                    "event_count": 120,
                    "evidence_count": 245,
                    "queue_depth": 2,
                }),
                faces=SimpleNamespace(
                    stats=Mock(return_value={
                        "observations": 30,
                        "candidate_frames": 72,
                        "temporal_tracks": 20,
                        "multi_frame_tracks": 18,
                        "consensus_tracks": 12,
                        "average_candidates_per_track": 3.6,
                    }),
                    recognition_status=Mock(return_value={
                        "queue_depth": 2,
                        "pending": 3,
                        "failed": 1,
                    }),
                ),
                statuses=Mock(return_value=[{
                    "id": "gate",
                    "name": "Gate",
                    "connected": True,
                    "frame_fresh": True,
                    "recording": True,
                    "recording_timestamp_health": {
                        "main": {
                            "discontinuities": 2,
                            "epoch_rollovers": 2,
                            "rollover_pending": False,
                        }
                    },
                    "detection_enabled": True,
                    "onvif_enabled": True,
                    "onvif_connected": True,
                    "onvif_notifications_received": 12,
                    "onvif_motion_events_received": 4,
                    "motion_qualification": {
                        "passed": 3,
                        "audit_rejected": 2,
                        "visual_backup_candidates": 11,
                        "visual_backup_triggers": 4,
                        "visual_backup_onvif_matches": 3,
                        "visual_backup_rate_limited": 2,
                    },
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
                        "prewarm_failures": 1,
                        "last_prewarm_failure": {
                            "timestamp": "2026-08-06T11:59:00+00:00",
                            "error": "main capture unavailable",
                        },
                        "handoff_failures": 2,
                        "last_handoff_failure": {
                            "event_id": 41,
                            "timestamp": "2026-08-06T12:00:00+00:00",
                            "error": "decoder unavailable",
                        },
                    },
                }]),
            )

            service = SystemTelemetryService()
            with patch("survng.app.system_telemetry.os.getloadavg", return_value=(1.0, 0.5, 0.25)):
                payload = service.telemetry(
                    fake_manager, main.config, hours=24, camera_id="gate"
                )

            self.assertEqual(payload["activity"], activity)
            self.assertEqual(payload["semantic_search"]["state"], "ready")
            self.assertEqual(payload["semantic_search"]["event_count"], 120)
            self.assertEqual(payload["face_recognition"]["candidate_frames"], 72)
            self.assertEqual(payload["face_recognition"]["consensus_tracks"], 12)
            self.assertEqual(payload["cameras"][0]["activity"]["last_24h"]["events"], 2)
            self.assertEqual(payload["cameras"][0]["onvif"]["notifications"], 12)
            self.assertEqual(payload["cameras"][0]["motion"]["rejected"], 2)
            self.assertEqual(payload["cameras"][0]["performance"]["status"], "warming_up")
            self.assertEqual(payload["cameras"][0]["motion"]["visual_backup_candidates"], 11)
            self.assertEqual(payload["cameras"][0]["motion"]["visual_backup_triggers"], 4)
            self.assertEqual(payload["cameras"][0]["motion"]["visual_backup_onvif_matches"], 3)
            self.assertEqual(payload["cameras"][0]["motion"]["visual_backup_rate_limited"], 2)
            self.assertEqual(
                payload["cameras"][0]["recording_timestamps"]["main"]["epoch_rollovers"],
                2,
            )
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
            self.assertEqual(payload["cameras"][0]["tracking"]["handoff_failures"], 2)
            self.assertEqual(payload["cameras"][0]["tracking"]["prewarm_failures"], 1)
            self.assertEqual(
                payload["cameras"][0]["tracking"]["last_handoff_failure"]["event_id"],
                41,
            )
            fake_manager.events.telemetry_activity.assert_called_once_with(hours=24)
            self.assertEqual(fake_manager.telemetry.operational_history.call_count, 2)
            self.assertEqual(fake_manager.events.tracking_capacity_activity.call_count, 2)
            for call in fake_manager.telemetry.operational_history.call_args_list:
                self.assertEqual(call.kwargs["camera_id"], "gate")
            for call in fake_manager.events.tracking_capacity_activity.call_args_list:
                self.assertEqual(call.kwargs["camera_id"], "gate")

    def test_telemetry_history_survives_interruption_annotation_failure(self) -> None:
        events = SimpleNamespace(
            tracking_capacity_activity=Mock(return_value=[]),
        )
        telemetry = SimpleNamespace(
            sample_times=Mock(side_effect=RuntimeError("database busy")),
            operational_history=Mock(return_value=[{"sampled_at": "2026-08-07T10:00:00+00:00"}]),
            memory_history=Mock(return_value=[]),
            lifecycle_events=Mock(return_value=[]),
        )
        fake_manager = SimpleNamespace(events=events, telemetry=telemetry)

        service = SystemTelemetryService()
        with patch("survng.app.system_telemetry.time.monotonic", return_value=1234.0):
            history = service.persisted_history(fake_manager, "")

        self.assertEqual(history["interruptions"], [])
        self.assertEqual(history["interruption_summary"]["total"], 0)
        self.assertEqual(len(history["runtime"]["short"]), 1)
        self.assertEqual(telemetry.operational_history.call_count, 2)

    def test_camera_telemetry_omits_system_interruption_annotations(self) -> None:
        events = SimpleNamespace(
            tracking_capacity_activity=Mock(return_value=[]),
        )
        telemetry = SimpleNamespace(
            sample_times=Mock(return_value=["2026-08-07T10:00:00+00:00"]),
            operational_history=Mock(return_value=[]),
            memory_history=Mock(return_value=[]),
            lifecycle_events=Mock(return_value=[]),
        )
        fake_manager = SimpleNamespace(events=events, telemetry=telemetry)

        service = SystemTelemetryService()
        with patch("survng.app.system_telemetry.time.monotonic", return_value=2345.0):
            history = service.persisted_history(fake_manager, "gate")

        self.assertEqual(history["interruptions"], [])
        self.assertEqual(history["interruption_summary"]["total"], 0)
        telemetry.sample_times.assert_not_called()
        telemetry.lifecycle_events.assert_not_called()

    def test_object_tracking_catalog_exposes_only_safe_production_backend(self) -> None:
        catalog = main.object_tracking_catalog()
        implementations = {
            item["id"]: item for item in catalog["implementations"]
        }

        self.assertTrue(implementations["survng_hybrid"]["available"])
        self.assertEqual(set(implementations), {"survng_hybrid"})

    def test_manual_detection_stays_on_one_manager_generation_during_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            snapshot = root / "snapshots" / "gate" / "event.jpg"
            snapshot.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((24, 32, 3), dtype=np.uint8)))
            event = {
                "id": 7,
                "camera_id": "gate",
                "snapshot_path": str(snapshot),
                "recording_path": "",
                "created_at": "2026-07-29T12:00:00+00:00",
                "objects_json": "[]",
            }
            events = SimpleNamespace(
                get=Mock(return_value=event),
                replace_detected_objects=Mock(return_value=event),
            )
            replacement = SimpleNamespace(
                detector=SimpleNamespace(detect=Mock(side_effect=AssertionError("new manager used"))),
                events=SimpleNamespace(
                    replace_detected_objects=Mock(side_effect=AssertionError("new manager used"))
                ),
            )

            def detect(_frame, confidence_threshold=None):
                main.manager = replacement
                return [{
                    "label": "person",
                    "confidence": 0.9,
                    "box": {"x1": 1, "y1": 1, "x2": 10, "y2": 20},
                }]

            active_manager = SimpleNamespace(
                storage_dir=root,
                detector=SimpleNamespace(detect=detect),
                events=events,
                publish_event=Mock(),
                detector_status=Mock(return_value={"enabled": True}),
            )
            active_config = AppConfig(
                storage_dir=str(root),
                cameras=[CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera.invalid/main",
                )],
            )

            with patch.object(main, "manager", active_manager), patch.object(main, "config", active_config):
                response = main.detect_event_snapshot(7)

            self.assertEqual(response["object_count"], 1)
            events.replace_detected_objects.assert_called_once()
            active_manager.publish_event.assert_called_once()
            replacement.events.replace_detected_objects.assert_not_called()

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
            "engines": {"survng_hybrid": {}, "ultralytics_fasttrack": {}},
        }
        runner = SimpleNamespace(run=Mock(return_value=comparison))

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            patch.object(main, "TRACKING_COMPARISON_LIMITER", limiter),
            patch.object(main, "ultralytics_fasttrack_dependency_status", return_value={"available": True, "reason": ""}),
            patch.object(
                main._recording_media_runtime,
                "_ensure_event_clip",
                return_value=Path("comparison.mp4"),
            ) as ensure_clip,
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
        self.assertIs(
            runner.run.call_args.args[1],
            sampled_frames.return_value,
        )
        limiter.release.assert_called_once_with()

    def test_tracking_comparison_duration_defaults_to_full_bounded_window(self) -> None:
        self.assertEqual(_tracking_comparison_duration(None), 30.0)
        self.assertEqual(_tracking_comparison_duration(6.0), 6.0)
        self.assertEqual(_tracking_comparison_duration(200.0), 30.0)
        self.assertEqual(_tracking_comparison_duration(1.0), 3.0)

    def test_tracking_comparison_keeps_the_original_manager_during_reload(self) -> None:
        event = {
            "id": 43,
            "camera_id": "gate",
            "created_at": "2026-07-27T12:17:07+00:00",
            "objects_json": "[]",
        }
        active_config = AppConfig(cameras=[CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://camera.invalid/main",
        )])
        active_events = SimpleNamespace(
            get=Mock(return_value=event),
            save_tracking_comparison=Mock(return_value={"id": 8, "verdict": ""}),
            tracking_comparison_summary=Mock(return_value={"total": 1}),
        )
        replacement_events = SimpleNamespace(
            save_tracking_comparison=Mock(side_effect=AssertionError("replacement used")),
        )
        replacement = SimpleNamespace(events=replacement_events)
        active_manager = SimpleNamespace(
            events=active_events,
            detector=SimpleNamespace(input_shape=[]),
            person_reidentifier=object(),
        )

        def run(_camera, _frames):
            main.manager = replacement
            return {
                "frames_processed": 1,
                "engines": {"survng_hybrid": {}, "ultralytics_fasttrack": {}},
            }

        with (
            patch.object(main, "config", active_config),
            patch.object(main, "manager", active_manager),
            patch.object(main, "ultralytics_fasttrack_dependency_status", return_value={"available": True, "reason": ""}),
            patch.object(
                main._recording_media_runtime,
                "_ensure_event_clip",
                return_value=Path("comparison.mp4"),
            ),
            patch.object(main, "sampled_video_frames", return_value=[]),
            patch.object(main, "TrackingComparisonRunner", return_value=SimpleNamespace(run=run)),
        ):
            result = main.compare_event_tracking(43)

        self.assertEqual(result["comparison_id"], 8)
        active_events.save_tracking_comparison.assert_called_once()
        replacement_events.save_tracking_comparison.assert_not_called()

    def test_tracking_comparison_rejects_missing_optional_backend_without_work(self) -> None:
        with patch.object(
            main,
            "ultralytics_fasttrack_dependency_status",
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
                TrackingComparisonVerdictRequest(verdict="inconclusive"),
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
        context = main._intelligence_route_bundle.service._audit_ai_context(
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

        self.assertEqual(
            context["motion_paradigm"]["paradigm"],
            "camera_triggered_with_visual_backup",
        )
        self.assertEqual(
            context["motion_paradigm"]["adaptive_visual"]["role"],
            "validator_and_backup_trigger",
        )
        self.assertTrue(context["decision_outcome"]["filtered_before_object_detection"])
        self.assertFalse(context["decision_outcome"]["object_detection_ran"])
        self.assertIsNone(context["decision_outcome"]["object_detected"])
        self.assertEqual(context["audit"]["category"], "qualification")
        self.assertFalse(context["decision_outcome"]["visual_backup"])
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

        context = main._intelligence_route_bundle.service._audit_ai_context(
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
                "require_incident_zone": False,
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
                "temporal_peak_confidence_offset_seconds": 8.0,
                "temporal_label_votes": {"robot_lawnmower": 2, "car": 1},
                "track_id": 7,
                "track_state": "confirmed",
                "track_observations": 5,
            }, {
                "status": "object_tracking",
                "object_tracking": {
                    "state": "complete",
                    "tracks": [{"track_id": 7, "label": "robot_lawnmower", "observations": 5}],
                },
            }]),
        }
        events = SimpleNamespace(
            get=lambda _event_id: event,
            motion_audits=lambda **_kwargs: ([], 0),
            for_camera_range=lambda *_args, **_kwargs: [],
        )

        context = main._intelligence_route_bundle.service._audit_ai_context(
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
        self.assertEqual(detected["temporal_peak_confidence_offset_seconds"], 8.0)
        self.assertEqual(detected["track_id"], 7)
        self.assertEqual(context["object_tracking"]["state"], "complete")
        self.assertEqual(context["effective_settings"]["object_confirmation_frames"], 2)
        self.assertEqual(
            context["effective_settings"]["object_class_confirmation_frames"],
            {"robot_lawnmower": 3},
        )
        self.assertEqual(
            context["effective_settings"]["incident_eligibility_policy"],
            "zones_plus_full_frame",
        )
        self.assertEqual(
            context["motion_paradigm"]["incident_eligibility"]["policy"],
            "zones_plus_full_frame",
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
            patch.object(main, "_begin_ai_operation"),
            patch.object(main.threading, "Thread", return_value=thread) as thread_factory,
        ):
            review = main._intelligence_route_bundle.service.start_motion_ai_review(MotionAiReviewRequest(camera_id="gate"))

        self.assertEqual(review["id"], 17)
        limiter.acquire.assert_called_once_with(blocking=False)
        thread.start.assert_called_once_with()
        self.assertEqual(thread_factory.call_args.kwargs["name"], "camera-intelligence-gate")
        self.assertIs(
            thread_factory.call_args.kwargs["target"].__func__,
            main._intelligence_route_bundle.service._run_camera_intelligence_review.__func__,
        )

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
                patch.object(SystemTelemetryService, "system_status", return_value={}),
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
                with patch.object(cv2, "imread", side_effect=AssertionError("cache miss")):
                    second = main.event_thumbnail(1, width=320, quality=80)

            cached = cv2.imread(str(first.path))
            self.assertIsNotNone(cached)
            self.assertEqual(cached.shape[1], 320)
            self.assertEqual(first.path, second.path)
            self.assertTrue(str(first.path).startswith(str(root / "cache")))

    def test_event_thumbnail_supports_large_desktop_pixel_density(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "large.jpg"
            snapshot.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((300, 3000, 3), dtype=np.uint8)))
            fake_manager = SimpleNamespace(
                storage_dir=root,
                image_cache=LocalImageCache(root / "cache"),
                events=SimpleNamespace(get=lambda _event_id: {"id": 1, "snapshot_path": str(snapshot)}),
            )

            with patch.object(main, "manager", fake_manager):
                response = main.event_thumbnail(1, width=9000, quality=100)

            cached = cv2.imread(str(response.path))
            self.assertIsNotNone(cached)
            self.assertEqual(cached.shape[1], 2560)

    def test_event_thumbnail_object_focus_crops_from_full_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "wide.jpg"
            snapshot.parent.mkdir(parents=True)
            frame = np.zeros((400, 2000, 3), dtype=np.uint8)
            frame[:, :] = (10, 10, 10)
            frame[150:250, 900:1100] = (0, 0, 255)
            self.assertTrue(cv2.imwrite(str(snapshot), frame))
            event = {
                "id": 1,
                "snapshot_path": str(snapshot),
                "objects_json": json.dumps([{
                    "label": "car",
                    "incident_eligible": True,
                    "detection_frame_width": 2000,
                    "detection_frame_height": 400,
                    "box": {"x1": 900, "y1": 150, "x2": 1100, "y2": 250},
                }]),
            }
            fake_manager = SimpleNamespace(
                storage_dir=root,
                image_cache=LocalImageCache(root / "cache"),
                events=SimpleNamespace(get=lambda _event_id: event),
            )

            with patch.object(main, "manager", fake_manager):
                full = main.event_thumbnail(1, width=640, quality=90)
                focused = main.event_thumbnail(1, width=640, quality=90, object_focus=True, zoom=1.0)

            full_image = cv2.imread(str(full.path))
            focused_image = cv2.imread(str(focused.path))
            self.assertIsNotNone(full_image)
            self.assertIsNotNone(focused_image)
            self.assertEqual(full_image.shape[1], 640)
            self.assertLess(focused_image.shape[1], full_image.shape[1])
            self.assertGreater(float(focused_image.mean()), float(full_image.mean()))
            self.assertNotEqual(full.path, focused.path)

    def test_object_focus_crop_rect_tightens_with_zoom(self) -> None:
        from survng.app.appearance_routes import object_focus_crop_rect

        boxes = [(900, 150, 1100, 250)]
        loose = object_focus_crop_rect(2000, 400, boxes, zoom=0.5)
        fitted = object_focus_crop_rect(2000, 400, boxes, zoom=1.0)
        tight = object_focus_crop_rect(2000, 400, boxes, zoom=2.0)
        self.assertIsNotNone(loose)
        self.assertIsNotNone(fitted)
        self.assertIsNotNone(tight)
        assert loose is not None and fitted is not None and tight is not None
        loose_area = (loose[2] - loose[0]) * (loose[3] - loose[1])
        fitted_area = (fitted[2] - fitted[0]) * (fitted[3] - fitted[1])
        tight_area = (tight[2] - tight[0]) * (tight[3] - tight[1])
        self.assertGreater(loose_area, fitted_area)
        self.assertGreater(fitted_area, tight_area)

    def test_webp_snapshot_uses_correct_media_type_and_download_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            snapshot = root / "snapshots" / "gate" / "event.webp"
            snapshot.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(snapshot), np.zeros((24, 32, 3), dtype=np.uint8)))
            fake_manager = SimpleNamespace(
                storage_dir=root,
                events=SimpleNamespace(
                    get=lambda _event_id: {"id": 1, "snapshot_path": str(snapshot)},
                ),
            )

            with patch.object(main, "manager", fake_manager):
                inline = main.event_snapshot(1)
                downloaded = main.event_snapshot(1, download=True)

            self.assertEqual(inline.media_type, "image/webp")
            self.assertNotIn("content-disposition", inline.headers)
            self.assertIn("event.webp", downloaded.headers["content-disposition"])

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
                with patch.object(cv2, "imread", side_effect=AssertionError("cache miss")):
                    second = main.face_crop(7, padding=0.2)

            crop = cv2.imread(str(first.path))
            self.assertIsNotNone(crop)
            self.assertEqual(crop.shape[:2], (140, 140))
            self.assertEqual(first.path, second.path)

    def test_incident_search_rejects_unsafe_timezone_paths(self) -> None:
        with self.assertRaises(HTTPException) as invalid:
            main.incident_search(time_zone="../../etc/passwd")

        self.assertEqual(invalid.exception.status_code, 422)

    def test_incident_event_type_filters_are_mutually_exclusive(self) -> None:
        motion = {"id": 1, "has_objects": False}
        object_incident = {"id": 2, "has_objects": True}
        incidents = [motion, object_incident]

        self.assertEqual(_filter_incidents_by_event_type(incidents, "motion"), [motion])
        self.assertEqual(_filter_incidents_by_event_type(incidents, "object"), [object_incident])
        self.assertEqual(_filter_incidents_by_event_type(incidents, "all"), incidents)

    def test_recent_incident_feed_filters_before_applying_page_limit(self) -> None:
        rows = [
            {
                "id": 4,
                "camera_id": "gate",
                "kind": "motion",
                "objects_json": "[]",
                "created_at": "2026-07-30T14:03:00+00:00",
            },
            {
                "id": 3,
                "camera_id": "gate",
                "kind": "motion",
                "objects_json": "[]",
                "created_at": "2026-07-30T14:02:00+00:00",
            },
            {
                "id": 2,
                "camera_id": "gate",
                "kind": "object",
                "objects_json": json.dumps([{"label": "car", "confidence": 0.9}]),
                "created_at": "2026-07-29T14:01:00+00:00",
            },
            {
                "id": 1,
                "camera_id": "foyer",
                "kind": "object",
                "objects_json": json.dumps([{"label": "person", "confidence": 0.9}]),
                "created_at": "2026-07-28T14:00:00+00:00",
            },
        ]
        fake_manager = SimpleNamespace(
            events=SimpleNamespace(recent_compact=lambda *_args, **_kwargs: rows)
        )

        with patch.object(main, "manager", fake_manager):
            first, first_has_more, _ = main.INCIDENT_QUERIES.recent_filtered_summaries(
                fake_manager,
                limit=1,
                offset=0,
                gap_seconds=45,
                event_type="object",
            )
            second, second_has_more, _ = main.INCIDENT_QUERIES.recent_filtered_summaries(
                fake_manager,
                limit=1,
                offset=1,
                gap_seconds=45,
                event_type="object",
            )

        self.assertEqual(first[0]["labels"], ["car"])
        self.assertTrue(first_has_more)
        self.assertEqual(second[0]["labels"], ["person"])
        self.assertFalse(second_has_more)

    def test_incident_list_payload_omits_investigation_only_data(self) -> None:
        event = _event_row({
            "id": 7,
            "camera_id": "gate",
            "kind": "object",
            "snapshot_path": "snapshots/gate/7.jpg",
            "recording_path": "recordings/gate/7.mp4",
            "objects_json": json.dumps([
                {
                    "label": "person",
                    "confidence": 0.91,
                    "box": [1, 2, 30, 40],
                    "temporal_samples": [{"offset": 0.1}] * 20,
                },
                {
                    "status": "object_tracking",
                    "object_tracking": {
                        "frame_width": 1920,
                        "frame_height": 1080,
                        "tracks": [{"id": 1}],
                        "samples": [{}] * 20,
                    },
                },
            ]),
            "created_at": "2026-07-30T14:00:00+00:00",
        })
        incident = _incident_row("gate", [event])
        incident["faces"] = [{"name": "Someone"}]
        incident["motion_observations"] = [{"id": 1}]

        payload = _incident_list_payload(incident)

        self.assertEqual(payload["object_tracking"], {
            "frame_width": 1920,
            "frame_height": 1080,
        })
        self.assertNotIn("faces", payload)
        self.assertNotIn("motion_observations", payload)
        self.assertEqual(payload["snapshot_path"], "available")
        self.assertEqual(payload["recording_path"], "available")
        self.assertEqual(payload["objects"], [{
            "label": "person",
            "confidence": 0.91,
            "box": [1, 2, 30, 40],
        }])
        self.assertEqual(payload["events"][0]["id"], 7)
        self.assertEqual(payload["events"][0]["object_tracking"], {
            "frame_width": 1920,
            "frame_height": 1080,
        })
        self.assertNotIn("temporal_samples", payload["events"][0]["objects"][0])

    def test_incident_exposes_source_that_opened_incident(self) -> None:
        ema_event = _event_row({
            "id": 7,
            "camera_id": "gate",
            "kind": "object",
            "objects_json": json.dumps([{
                "status": "motion_qualification",
                "motion_qualification": {"trigger_source": "visual_backup"},
            }, {"label": "car", "confidence": 0.8}]),
            "created_at": "2026-07-30T14:00:00+00:00",
        })
        camera_event = _event_row({
            "id": 8,
            "camera_id": "gate",
            "kind": "object",
            "objects_json": json.dumps([{
                "status": "motion_qualification",
                "motion_qualification": {"trigger_source": "camera"},
            }, {"label": "car", "confidence": 0.95}]),
            "created_at": "2026-07-30T14:00:10+00:00",
        })

        incident = _incident_row("gate", [camera_event, ema_event])
        payload = _incident_list_payload(incident)

        self.assertEqual(ema_event["trigger_source"], "ema")
        self.assertEqual(camera_event["trigger_source"], "camera")
        self.assertEqual(incident["representative_event_id"], 8)
        self.assertEqual(incident["trigger_source"], "ema")
        self.assertEqual(payload["trigger_source"], "ema")
        self.assertEqual(payload["events"][0]["trigger_source"], "camera")

    def test_incident_detail_hydrates_only_requested_incident(self) -> None:
        rows = [{
            "id": 7,
            "camera_id": "gate",
            "kind": "object",
            "topic": "camera/gate",
            "message": "motion",
            "snapshot_path": "snapshots/gate/7.jpg",
            "recording_path": "",
            "objects_json": json.dumps([{"label": "person", "confidence": 0.91}]),
            "created_at": "2026-07-30T14:00:00+00:00",
        }]
        fake_manager = SimpleNamespace(
            storage_dir=Path("/tmp"),
            events=SimpleNamespace(
                get_many=lambda event_ids: [row for row in rows if row["id"] in event_ids],
                motion_audits_for_related_events=lambda _event_ids: [],
            ),
            faces=SimpleNamespace(for_event_ids=lambda _event_ids: []),
        )

        with patch.object(main, "manager", fake_manager):
            detail = main.incident_detail("7")

        self.assertEqual(detail["camera_id"], "gate")
        self.assertEqual(detail["representative_event_id"], 7)
        self.assertEqual(detail["objects"][0]["label"], "person")
        self.assertIn("faces", detail)

    def test_incident_detail_rejects_events_from_multiple_incidents(self) -> None:
        rows = [
            {"id": 1, "camera_id": "gate", "kind": "motion", "objects_json": "[]", "created_at": "2026-07-30T14:00:00+00:00"},
            {"id": 2, "camera_id": "foyer", "kind": "motion", "objects_json": "[]", "created_at": "2026-07-30T14:00:01+00:00"},
        ]
        fake_manager = SimpleNamespace(events=SimpleNamespace(get_many=lambda _event_ids: rows))

        with patch.object(main, "manager", fake_manager), self.assertRaises(HTTPException) as invalid:
            main.incident_detail("1,2")

        self.assertEqual(invalid.exception.status_code, 422)

    def test_event_row_tolerates_malformed_legacy_object_entries(self) -> None:
        row = _event_row({
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

        selected = _best_incident_event([row])
        self.assertEqual(selected["id"], 1)

    def test_incident_representative_prefers_clearer_confirmed_snapshot(self) -> None:
        blurred_high_confidence = _event_row({
            "id": 10,
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.94,
                "incident_eligible": True,
                "snapshot_quality_score": 0.61,
            }]),
        })
        clear_lower_confidence = _event_row({
            "id": 11,
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.78,
                "incident_eligible": True,
                "snapshot_quality_score": 0.88,
            }]),
        })

        selected = _best_incident_event([
            blurred_high_confidence,
            clear_lower_confidence,
        ])

        self.assertEqual(selected["id"], 11)

    def test_incident_representative_prefers_fully_framed_primary_subject(self) -> None:
        crowded_clipped = _event_row({
            "id": 20,
            "objects_json": json.dumps([
                {
                    "label": "person",
                    "confidence": 0.93,
                    "incident_eligible": True,
                    "snapshot_primary_subject": True,
                    "snapshot_edge_clearance_ratio": 0.0,
                    "snapshot_subject_area_ratio": 0.01,
                    "snapshot_quality_score": 0.70,
                },
                {
                    "label": "car",
                    "confidence": 0.91,
                    "incident_eligible": True,
                    "snapshot_primary_subject": False,
                    "snapshot_quality_score": 0.70,
                },
            ]),
        })
        clear_subject = _event_row({
            "id": 21,
            "objects_json": json.dumps([{
                "label": "person",
                "confidence": 0.80,
                "incident_eligible": True,
                "snapshot_primary_subject": True,
                "snapshot_edge_clearance_ratio": 0.08,
                "snapshot_subject_area_ratio": 0.04,
                "snapshot_quality_score": 0.82,
            }]),
        })

        selected = _best_incident_event([crowded_clipped, clear_subject])

        self.assertEqual(selected["id"], 21)

    def test_incident_representative_prefers_larger_fully_framed_primary_subject(self) -> None:
        distant = _event_row({
            "id": 22,
            "objects_json": json.dumps([{
                "label": "car",
                "confidence": 0.90,
                "incident_eligible": True,
                "snapshot_primary_subject": True,
                "snapshot_edge_clearance_ratio": 0.40,
                "snapshot_subject_area_ratio": 0.001,
                "snapshot_quality_score": 0.91,
            }]),
        })
        close = _event_row({
            "id": 23,
            "objects_json": json.dumps([{
                "label": "car",
                "confidence": 0.88,
                "incident_eligible": True,
                "snapshot_primary_subject": True,
                "snapshot_edge_clearance_ratio": 0.08,
                "snapshot_subject_area_ratio": 0.08,
                "snapshot_quality_score": 0.87,
            }]),
        })

        selected = _best_incident_event([distant, close])

        self.assertEqual(selected["id"], 23)

    def test_linked_motion_observation_extends_incident_without_duplicate_event(self) -> None:
        incident = _incident_row("foyer", [{
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
        event = _event_row({
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

        incident = _incident_row("gate", [event])

        self.assertEqual(incident["event_count"], 1)
        self.assertEqual(incident["duration_seconds"], 8.0)
        self.assertEqual(incident["end_at"], "2026-07-27T12:17:15+00:00")
        self.assertEqual(incident["object_tracking"]["tracks"][0]["track_id"], 1)
        self.assertEqual(incident["events"][0]["objects"][0]["track_id"], 1)
        self.assertEqual(incident["labels"], ["car", "person"])
        self.assertEqual(incident["zones"], ["driveway"])

    def test_capacity_skip_does_not_extend_incident_duration(self) -> None:
        event = _event_row({
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

        incident = _incident_row("gate", [event])

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

        incident = _incident_row("foyer", [representative, later])

        self.assertEqual(incident["representative_event_id"], 43)
        self.assertEqual(len(incident["object_tracking"]["tracks"]), 1)

    def test_incident_payload_keeps_annotation_fields_without_detector_diagnostics(self) -> None:
        payload = _incident_event_payload({
            "id": 42,
            "topic": "private/topic",
            "message": "large raw payload",
            "objects": [{
                "label": "person",
                "confidence": 0.9,
                "box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                "detection_frame_width": 2560,
                "detection_frame_height": 1920,
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
        self.assertEqual(payload["objects"][0]["detection_frame_width"], 2560)
        self.assertEqual(payload["objects"][0]["detection_frame_height"], 1920)
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

            row = _motion_audit_row(
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
