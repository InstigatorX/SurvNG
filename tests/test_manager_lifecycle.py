from __future__ import annotations

import json
import threading
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from survng.app.config import AppConfig, CameraConfig
from survng.app.camera_fleet import CameraFleetLifecycle
from survng.app.camera_control import CameraControlService
from survng.app.camera_startup import CameraStartupCoordinator
from survng.app.manager import AppManager
from survng.app.detection_watch import RouteDetectionWatch
from survng.app.mqtt_lifecycle import MqttLifecycle
from survng.app.recording_lifecycle import RecordingLifecycle
from survng.app.runtime_monitor import ApplicationRuntimeMonitor


def manager_with_mocks() -> AppManager:
    manager = object.__new__(AppManager)
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
    manager.config = AppConfig(cameras=[camera])
    manager._lifecycle_lock = threading.RLock()
    manager._stopping = False
    manager._started = False
    manager._closed = False
    manager._startup_services_ready = False
    manager._startup_timings = {}
    camera_startup = CameraStartupCoordinator(
        readiness_timeout_seconds=0.01,
        recorder_settle_seconds=0.0,
        poll_interval_seconds=0.01,
    )
    manager.detector = Mock()
    manager.faces = Mock()
    manager.inference = Mock()
    manager.inference.detector = manager.detector
    manager.inference.faces = manager.faces
    manager.inference.face_recognizer = Mock()
    manager.inference.person_reidentifier = Mock()
    manager.inference.semantic_search = Mock()
    manager.inference.appearance_backfill = Mock()
    manager.inference.tracking_limiter = Mock()
    manager.inference.tracking_factory = Mock()
    manager.inference.status.return_value = {}
    manager._motion_analysis_limiter = Mock()
    manager.motion_object_detector_factory = Mock()
    manager.motion_object_detector_factory.decode_budget.status.return_value = {}
    manager.recorder = Mock()
    manager.recorder.ffmpeg_path = manager.config.ffmpeg_path
    manager.recorder.hardware_acceleration = manager.config.hardware_acceleration
    manager.recorder.segment_seconds = manager.config.recording_segment_seconds
    manager.recording = RecordingLifecycle(
        config=manager.config,
        storage_dir=Path("."),
        protected_recording_paths=set,
        recorder=manager.recorder,
    )
    manager.mqtt = Mock()
    manager.state_events = Mock()
    manager.detection_watch = Mock()
    manager.detection_watch.observe_incident.return_value = ()
    manager.detection_watch.status.return_value = {}
    manager.workers = {"gate": Mock()}
    manager.workers["gate"].live_capture_ready.return_value = True
    manager.workers["gate"].wait_stopped.return_value = True
    manager.workers["gate"].wait_onvif_stopped.return_value = True
    manager.workers["gate"].active_workers.return_value = []
    manager.runtime_monitor = Mock()
    manager.ema_route_candidates = Mock()
    manager.ema_route_candidates.status.return_value = {}
    manager.camera_fleet = CameraFleetLifecycle(
        cameras=[camera],
        workers=manager.workers,
        recorder=manager.recorder,
        startup=camera_startup,
        state_publisher=manager.mqtt,
    )
    manager.camera_controls = CameraControlService(
        cameras=[camera],
        workers=manager.workers,
        recording=manager.recording,
        fleet=manager.camera_fleet,
        mqtt=manager.mqtt,
        runtime_monitor=manager.runtime_monitor,
        state_path=Path("runtime_state.json"),
    )
    manager.camera_controls._persist_locked = Mock()
    return manager


class ManagerLifecycleTest(unittest.TestCase):
    def test_restart_reconstructs_unexpired_route_watch_from_incident_store(self) -> None:
        now = datetime.now(timezone.utc)
        manager = object.__new__(AppManager)
        manager.config = AppConfig(
            cameras=[
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate/main"),
                CameraConfig(
                    id="back-left",
                    name="Back Left",
                    stream_url="rtsp://back-left/main",
                ),
            ],
            detector={"tracking": {"camera_transition_routes": [{
                "from_camera": "gate",
                "to_camera": "back-left",
                "max_seconds": 30,
            }]}},
        )
        routes = manager.config.detector.tracking.camera_transition_routes
        manager.detection_watch = RouteDetectionWatch(routes)
        manager._restored_detection_watches = []
        manager.events = Mock()
        manager.events.route_watch_consumed.return_value = False
        manager.events.between.return_value = [{
            "id": 88,
            "camera_id": "gate",
            "created_at": (now - timedelta(seconds=5)).isoformat(),
            "objects_json": json.dumps([{
                "label": "car",
                "incident_eligible": True,
            }]),
        }]

        manager._restore_detection_watches()

        assert manager.detection_watch.match("back-left", now.timestamp()) is not None

    def test_restart_does_not_restore_consumed_route_watch(self) -> None:
        now = datetime.now(timezone.utc)
        manager = object.__new__(AppManager)
        manager.config = AppConfig(
            cameras=[
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate/main"),
                CameraConfig(
                    id="back-left",
                    name="Back Left",
                    stream_url="rtsp://back-left/main",
                ),
            ],
            detector={"tracking": {"camera_transition_routes": [{
                "from_camera": "gate",
                "to_camera": "back-left",
                "max_seconds": 30,
            }]}},
        )
        manager.detection_watch = RouteDetectionWatch(
            manager.config.detector.tracking.camera_transition_routes
        )
        manager._restored_detection_watches = []
        manager.events = Mock()
        manager.events.route_watch_consumed.return_value = True
        manager.events.between.return_value = [{
            "id": 88,
            "camera_id": "gate",
            "created_at": (now - timedelta(seconds=5)).isoformat(),
            "objects_json": json.dumps([{
                "label": "car",
                "incident_eligible": True,
            }]),
        }]

        manager._restore_detection_watches()

        assert manager.detection_watch.match("back-left", now.timestamp()) is None

    def test_restart_does_not_replay_durably_admitted_origin_target(self) -> None:
        now = datetime.now(timezone.utc)
        manager = object.__new__(AppManager)
        manager.config = AppConfig(
            cameras=[
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate/main"),
                CameraConfig(
                    id="back-left",
                    name="Back Left",
                    stream_url="rtsp://back-left/main",
                ),
            ],
            detector={"tracking": {"camera_transition_routes": [{
                "from_camera": "gate",
                "to_camera": "back-left",
                "max_seconds": 30,
            }]}},
        )
        manager.detection_watch = RouteDetectionWatch(
            manager.config.detector.tracking.camera_transition_routes
        )
        manager._restored_detection_watches = []
        manager.events = Mock()
        manager.events.route_target_admitted.return_value = True
        manager.events.route_watch_consumed.return_value = False
        manager.events.between.return_value = [{
            "id": 88,
            "camera_id": "gate",
            "created_at": (now - timedelta(seconds=5)).isoformat(),
            "objects_json": json.dumps([{
                "label": "car",
                "incident_eligible": True,
            }]),
        }]

        manager._restore_detection_watches()

        assert manager.detection_watch.match("back-left", now.timestamp()) is None
        manager.events.route_target_admitted.assert_called_once_with(
            "gate", 88, "back-left"
        )

    def test_restart_preserves_route_lineage_and_does_not_reopen_ancestor(self) -> None:
        now = datetime.now(timezone.utc)
        manager = object.__new__(AppManager)
        manager.config = AppConfig(
            cameras=[
                CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate/main"),
                CameraConfig(
                    id="lower-garage",
                    name="Lower Garage",
                    stream_url="rtsp://lower/main",
                ),
                CameraConfig(
                    id="upper-garage",
                    name="Upper Garage",
                    stream_url="rtsp://upper/main",
                ),
            ],
            detector={"tracking": {"camera_transition_routes": [
                {
                    "from_camera": "gate",
                    "to_camera": "lower-garage",
                    "bidirectional": True,
                },
                {
                    "from_camera": "lower-garage",
                    "to_camera": "upper-garage",
                    "bidirectional": True,
                },
                {
                    "from_camera": "gate",
                    "to_camera": "upper-garage",
                    "bidirectional": True,
                },
            ]}},
        )
        manager.detection_watch = RouteDetectionWatch(
            manager.config.detector.tracking.camera_transition_routes
        )
        manager._restored_detection_watches = []
        manager.events = Mock()
        manager.events.route_watch_consumed.return_value = False
        manager.events.between.return_value = [{
            "id": 90,
            "camera_id": "upper-garage",
            "created_at": (now - timedelta(seconds=5)).isoformat(),
            "objects_json": json.dumps([{
                "label": "car",
                "incident_eligible": True,
            }, {
                "status": "motion_qualification",
                "motion_qualification": {"features": {
                    "route_detection_watch": {
                        "route_path": ["gate", "lower-garage", "upper-garage"],
                    },
                }},
            }]),
        }]

        manager._restore_detection_watches()

        assert manager.detection_watch.status(now.timestamp())["active"] == 0

    def test_startup_replays_restored_watch_into_target_worker(self) -> None:
        manager = object.__new__(AppManager)
        watch = SimpleNamespace(
            target_camera_id="back-left",
            source_event_id=88,
        )
        target = Mock()
        target.consider_route_detection_watch.return_value = True
        manager._restored_detection_watches = [watch]
        manager.workers = {"back-left": target}

        manager._replay_restored_detection_watches()

        target.consider_route_detection_watch.assert_called_once_with(watch)
        assert manager._restored_detection_watches == []

    def test_startup_route_replay_retries_false_and_exception_until_success(self) -> None:
        manager = object.__new__(AppManager)
        watch = SimpleNamespace(
            target_camera_id="back-left",
            source_event_id=88,
            expires_at=time.time() + 30.0,
        )
        target = Mock()
        target.consider_route_detection_watch.side_effect = [
            False,
            RuntimeError("temporary candidate read failure"),
            True,
        ]
        manager._restored_detection_watches = [watch]
        manager._restored_watch_retry_lock = threading.Lock()
        manager._restored_watch_retry_timer = None
        manager._started = False
        manager._stopping = False
        manager._closed = False
        manager.workers = {"back-left": target}

        manager._replay_restored_detection_watches()
        assert manager._restored_detection_watches == [watch]
        manager._replay_restored_detection_watches()
        assert manager._restored_detection_watches == [watch]
        manager._replay_restored_detection_watches()

        assert target.consider_route_detection_watch.call_count == 3
        assert manager._restored_detection_watches == []

    def test_confirmed_object_event_opens_route_detection_watch(self) -> None:
        manager = manager_with_mocks()
        manager.events = Mock()
        manager.events.get.return_value = None
        payload = {
            "event_id": 42,
            "camera_id": "gate",
            "timestamp": "2026-08-16T14:44:13+00:00",
            "objects": [{"label": "car", "confidence": 0.96}],
            "incident_objects": [{"label": "car", "confidence": 0.96}],
        }

        manager.publish_event("object", payload)

        manager.detection_watch.observe_incident.assert_called_once_with(
            camera_id="gate",
            event_id=42,
            event_at=datetime.fromisoformat(payload["timestamp"]).timestamp(),
            objects=payload["incident_objects"],
        )

    def test_ineligible_object_event_does_not_open_route_detection_watch(self) -> None:
        manager = manager_with_mocks()
        manager.events = Mock()
        manager.events.get.return_value = None

        manager.publish_event("object", {
            "event_id": 43,
            "camera_id": "gate",
            "timestamp": "2026-08-16T14:44:13+00:00",
            "objects": [{
                "label": "car",
                "confidence": 0.15,
                "incident_eligible": False,
            }],
            "incident_objects": [],
        })

        manager.detection_watch.observe_incident.assert_not_called()

    def test_presentation_update_refreshes_clients_without_reopening_mqtt_incident(self) -> None:
        manager = manager_with_mocks()
        manager.events = Mock()
        event = {"id": 42, "camera_id": "gate", "created_at": "2026-08-15T12:00:00+00:00"}
        manager.events.get.return_value = event

        manager.publish_event("incident_update", {
            "event_id": 42,
            "camera_id": "gate",
            "updated": True,
            "reason": "cover_promoted",
        })

        manager.semantic_search.index.delete_event.assert_called_once_with(42)
        manager.semantic_search.queue_event.assert_called_once_with(event)
        manager.state_events.publish.assert_called_once_with(
            "incident",
            {
                "event_id": 42,
                "camera_id": "gate",
                "updated": True,
                "reason": "cover_promoted",
            },
        )
        manager.mqtt.track_incident.assert_not_called()
        manager.mqtt.publish.assert_not_called()

    def test_allocator_trim_ignores_ordinary_main_capture(self) -> None:
        statuses = [{
            "main_running": True,
            "object_tracking": {"active": False, "worker_running": False},
        }]

        self.assertTrue(ApplicationRuntimeMonitor.allocator_trim_safe(statuses, {}))

    def test_allocator_trim_waits_for_tracking_and_inference(self) -> None:
        tracking = [{"object_tracking": {"worker_running": True}}]

        self.assertFalse(ApplicationRuntimeMonitor.allocator_trim_safe(tracking, {}))
        self.assertFalse(ApplicationRuntimeMonitor.allocator_trim_safe([], {"queue_depth": 1}))
        self.assertFalse(ApplicationRuntimeMonitor.allocator_trim_safe([], {"pending_frames": 1}))
        self.assertFalse(ApplicationRuntimeMonitor.allocator_trim_safe([], {"active_inferences": 1}))

    def test_tracking_burst_guard_fails_closed_during_manager_construction(self) -> None:
        manager = object.__new__(AppManager)

        self.assertFalse(manager._tracking_burst_available())

    def test_detector_status_includes_inference_lifecycle_health(self) -> None:
        manager = manager_with_mocks()
        manager.detector.status.return_value = {"enabled": True}
        manager.inference.status.return_value = {"retired_cleanup_pending": 1}
        manager.motion_object_detector_factory.decode_budget.status.return_value = {
            "ffmpeg_attempts": {"hardware": 2, "cpu": 1},
        }

        status = manager.detector_status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["lifecycle"]["retired_cleanup_pending"], 1)
        self.assertEqual(
            status["recorded_decode"]["ffmpeg_attempts"],
            {"hardware": 2, "cpu": 1},
        )
        recommendation = status["object_worker_recommendation"]
        self.assertIn("recommended", recommendation)
        self.assertIn("current", recommendation)
        self.assertIn("reasons", recommendation)
        self.assertIn("signals", recommendation)
        self.assertGreaterEqual(recommendation["recommended"], 1)
        self.assertLessEqual(recommendation["recommended"], 4)

    def test_mqtt_server_health_accepts_loaded_isolated_detector(self) -> None:
        manager = manager_with_mocks()
        manager.config = AppConfig(
            cameras=manager.config.cameras,
            detector={"enabled": True, "device": "GPU"},
        )
        manager._started = True
        manager._process_started_monotonic = 0.0
        manager._process_started_at = "2026-07-31T12:00:00+00:00"
        manager.statuses = Mock(return_value=[{
            "id": "gate",
            "running": True,
            "recording": True,
            "sub_recording": False,
        }])
        manager.detector_status = Mock(return_value={
            "enabled": True,
            "loaded_backend": "openvino",
            "configured_device": "GPU",
            "openvino_loaded": True,
            "runtime": {"queue_depth": 0},
            "isolation": {"enabled": True, "worker_alive": True},
        })
        manager.recorder.retention_status.return_value = {"state": "idle"}

        server = manager._mqtt_server_status()

        self.assertEqual(server["state"]["health"], "ok")
        self.assertEqual(server["metrics"]["detector_state"], "ready")
        self.assertEqual(server["metrics"]["detector_device"], "GPU")

    def test_mqtt_server_health_faults_when_isolated_detector_worker_is_dead(self) -> None:
        status = {
            "enabled": True,
            "loaded_backend": "openvino",
            "openvino_loaded": True,
            "isolation": {"enabled": True, "worker_alive": False},
        }

        self.assertFalse(AppManager._detector_runtime_ready(status))

    def test_mqtt_server_health_degrades_when_part_of_detector_pool_is_offline(self) -> None:
        manager = manager_with_mocks()
        manager.config = AppConfig(
            cameras=manager.config.cameras,
            detector={"enabled": True, "device": "GPU", "object_worker_count": 2},
        )
        manager._started = True
        manager._process_started_monotonic = time.monotonic()
        manager._process_started_at = "2026-08-08T00:00:00+00:00"
        manager.statuses = Mock(return_value=[])
        manager.detector_status = Mock(return_value={
            "enabled": True,
            "loaded_backend": "openvino",
            "openvino_loaded": True,
            "runtime": {"queue_depth": 1},
            "isolation": {
                "enabled": True,
                "worker_alive": True,
                "configured_workers": 2,
                "alive_workers": 1,
            },
        })
        manager.recorder.retention_status.return_value = {"state": "idle"}

        server = manager._mqtt_server_status()

        self.assertEqual(server["state"]["health"], "degraded")
        self.assertEqual(server["metrics"]["detector_state"], "degraded")
        self.assertEqual(server["metrics"]["object_workers_configured"], 2)
        self.assertEqual(server["metrics"]["object_workers_alive"], 1)

    def test_mqtt_reconfiguration_does_not_touch_camera_workers_or_recorders(self) -> None:
        manager = manager_with_mocks()
        previous = manager.mqtt
        replacement = Mock()
        manager.mqtt = MqttLifecycle(
            manager.config.mqtt,
            lambda _config: replacement,
            service=previous,
        )
        manager.reconfigure_mqtt(manager.config.mqtt)

        previous.stop.assert_called_once_with(
            lifecycle="restarting",
            require_quiesced=True,
        )
        replacement.start.assert_called_once_with(raise_on_failure=True)
        manager.workers["gate"].stop.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

    def test_failed_mqtt_reconfiguration_restores_previous_runtime(self) -> None:
        manager = manager_with_mocks()
        previous = manager.mqtt
        replacement = Mock()
        replacement.start.side_effect = RuntimeError("mqtt start failed")
        manager.mqtt = MqttLifecycle(
            manager.config.mqtt,
            lambda _config: replacement,
            service=previous,
        )
        with self.assertRaisesRegex(RuntimeError, "mqtt start failed"):
            manager.reconfigure_mqtt(manager.config.mqtt)

        self.assertIs(manager.mqtt.service, previous)
        replacement.stop.assert_called_once_with(lifecycle="restarting")
        previous.start.assert_called_once_with(raise_on_failure=True)
        manager.workers["gate"].stop.assert_not_called()

    def test_recorder_reconfiguration_keeps_camera_capture_running(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        next_config = manager.config.model_copy(update={"recording_segment_seconds": 30})

        manager.reconfigure_recorders(next_config)

        self.assertEqual(manager.recorder.set_camera_enabled.call_args_list, [
            unittest.mock.call("gate", False),
            unittest.mock.call("gate", True),
        ])
        manager.recorder.reconfigure_runtime.assert_called_once_with(
            ffmpeg_path=next_config.ffmpeg_path,
            hardware_acceleration=next_config.hardware_acceleration,
            segment_seconds=30,
        )
        manager.recorder.start.assert_called_once_with(manager.config.cameras[0], "main")
        manager.workers["gate"].stop.assert_not_called()
        manager.workers["gate"].start.assert_not_called()

    def test_tracking_reconfiguration_keeps_camera_capture_and_recorders_running(self) -> None:
        manager = manager_with_mocks()
        worker = manager.workers["gate"]
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.tracking.sample_fps = 3.0

        manager.reconfigure_object_tracking(next_detector)

        manager.inference.reconfigure_tracking.assert_called_once_with(next_detector)
        worker.stop.assert_not_called()
        worker.start.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

    def test_inference_reconfiguration_keeps_camera_capture_running(self) -> None:
        manager = manager_with_mocks()
        manager.detector.config = manager.config.detector
        manager.face_recognizer = Mock()
        manager.person_reidentifier = Mock()
        manager._mqtt_connected = Mock()
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.device = "GPU"

        manager.reconfigure_inference(next_detector, {"object"})

        manager.inference.reconfigure_roles.assert_called_once_with(
            next_detector,
            {"object"},
            refresh_tracking=False,
        )
        manager.faces.close.assert_not_called()
        manager.workers["gate"].stop.assert_not_called()
        manager.workers["gate"].start.assert_not_called()
        manager.recorder.stop_all.assert_not_called()
        manager._mqtt_connected.assert_called_once_with()

    def test_output_changing_inference_update_refreshes_tracking_sessions(self) -> None:
        manager = manager_with_mocks()
        manager.detector.config = manager.config.detector
        manager.face_recognizer = Mock()
        manager.person_reidentifier = Mock()
        manager._mqtt_connected = Mock()
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.model_path = "/models/replacement.xml"

        manager.reconfigure_inference(
            next_detector,
            {"object"},
            refresh_tracking=True,
        )

        manager.inference.reconfigure_roles.assert_called_once_with(
            next_detector,
            {"object"},
            refresh_tracking=True,
        )
        manager.workers["gate"].stop.assert_not_called()

    def test_failed_tracking_refresh_restores_inference_before_resuming_tracking(self) -> None:
        manager = manager_with_mocks()
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.device = "GPU"
        manager.inference.reconfigure_roles.side_effect = RuntimeError(
            "tracking refresh failed"
        )

        with self.assertRaisesRegex(RuntimeError, "tracking refresh failed"):
            manager.reconfigure_inference(
                next_detector,
                {"object"},
                refresh_tracking=True,
            )

        manager.inference.reconfigure_roles.assert_called_once_with(
            next_detector,
            {"object"},
            refresh_tracking=True,
        )

    def test_face_queue_is_restored_when_stopping_it_fails(self) -> None:
        manager = manager_with_mocks()
        manager.inference.reconfigure_roles.side_effect = RuntimeError("face queue busy")
        next_detector = manager.config.detector.model_copy(deep=True)
        next_detector.face_recognition_enabled = True

        with self.assertRaisesRegex(RuntimeError, "face queue busy"):
            manager.reconfigure_inference(next_detector, {"face"})

        manager.inference.reconfigure_roles.assert_called_once_with(
            next_detector,
            {"face"},
            refresh_tracking=False,
        )
        manager.workers["gate"].stop.assert_not_called()

    def test_camera_state_fingerprint_includes_trigger_health_changes(self) -> None:
        status = {
            "id": "gate",
            "onvif_connected": True,
            "onvif_notifications_received": 10,
            "onvif_motion_events_received": 2,
            "onvif_renewals": 1,
            "motion_qualification": {},
        }
        original = ApplicationRuntimeMonitor.camera_state_fingerprint(status)

        self.assertNotEqual(
            original,
            ApplicationRuntimeMonitor.camera_state_fingerprint({
                **status,
                "onvif_motion_events_received": 3,
            }),
        )
        self.assertNotEqual(
            original,
            ApplicationRuntimeMonitor.camera_state_fingerprint({
                **status,
                "onvif_renewal_errors": 1,
            }),
        )
        self.assertNotEqual(
            original,
            ApplicationRuntimeMonitor.camera_state_fingerprint({
                **status,
                "stream_dimensions": {"live": {"width": 896, "height": 672}},
            }),
        )

    def test_real_empty_manager_starts_and_stops_all_background_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AppManager(AppConfig(storage_dir=tmpdir))
            manager.start_all()
            self.assertTrue(manager._started)
            self.assertTrue(manager.runtime_monitor.running)
            server = manager._mqtt_server_status()
            self.assertEqual(server["state"]["health"], "ok")
            self.assertEqual(server["metrics"]["cameras_total"], 0)
            self.assertEqual(server["metrics"]["detector_state"], "disabled")

            manager.stop_all()

        self.assertTrue(manager._closed)
        self.assertFalse(manager.runtime_monitor.running)
        self.assertIsNone(manager.recorder._index_thread)
        self.assertIsNone(manager.recorder._watchdog_thread)

    def test_constructor_failure_closes_services_created_before_workers(self) -> None:
        inference = Mock()
        inference.detector = Mock()
        inference.faces = Mock()
        inference.face_recognizer = Mock()
        inference.person_reidentifier = Mock()
        inference.tracking_factory = Mock()
        recorder = Mock()
        recording = Mock()
        recording.recorder = recorder
        state_events = Mock()
        mqtt = Mock()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("survng.app.manager.InferenceLifecycle", return_value=inference),
            patch("survng.app.manager.RecordingLifecycle", return_value=recording),
            patch("survng.app.manager.StateEventBroker", return_value=state_events),
            patch("survng.app.manager.MqttService", return_value=mqtt),
            patch.object(AppManager, "_create_camera_worker", side_effect=RuntimeError("worker failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                AppManager(AppConfig(
                    storage_dir=tmpdir,
                    cameras=[CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")],
                ))

        mqtt.stop.assert_called_once_with()
        inference.close.assert_called_once_with()
        recording.close.assert_called_once_with()
        state_events.close.assert_called_once_with()

    def test_inference_construction_failure_closes_recorder_and_state_broker(self) -> None:
        recorder = Mock()
        recording = Mock()
        recording.recorder = recorder
        state_events = Mock()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("survng.app.manager.RecordingLifecycle", return_value=recording),
            patch("survng.app.manager.StateEventBroker", return_value=state_events),
            patch(
                "survng.app.manager.InferenceLifecycle",
                side_effect=RuntimeError("inference construction failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "inference construction failed"),
        ):
            AppManager(AppConfig(storage_dir=tmpdir, cameras=[]))

        recording.close.assert_called_once_with()
        state_events.close.assert_called_once_with()

    def test_manager_keeps_databases_and_onvif_caches_outside_media_storage(self) -> None:
        with tempfile.TemporaryDirectory() as storage, tempfile.TemporaryDirectory() as database:
            camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
            manager = AppManager(AppConfig(
                storage_dir=storage,
                database_dir=database,
                cameras=[camera],
            ))

            self.assertEqual(manager.events.db_path, Path(database) / "survng.sqlite3")
            self.assertEqual(manager.faces.db_path, Path(database) / "survng.sqlite3")
            self.assertEqual(manager.database_dir, Path(database))
            self.assertEqual(manager.workers["gate"].onvif._cache_dir, Path(database) / "onvif")
            self.assertEqual(manager._capture_open_limiter.capacity, 2)
            self.assertEqual(manager.camera_fleet.startup.max_concurrency, 2)
            self.assertEqual(
                manager.camera_fleet.startup.readiness_timeout_seconds,
                5.0,
            )
            self.assertEqual(
                manager.camera_fleet.startup.recorder_settle_seconds,
                0.5,
            )

            manager.stop_all()

    def test_quiesced_controls_prevent_late_recorder_launch(self) -> None:
        manager = manager_with_mocks()
        manager.camera_controls.quiesce()

        accepted = manager.set_recording("gate", True)

        self.assertFalse(accepted)
        manager.recorder.start.assert_not_called()

    def test_camera_startup_failure_is_isolated_and_reported(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].start.side_effect = RuntimeError("capture failed")

        manager.start_all()
        self.assertTrue(manager.wait_for_camera_startup(timeout=1))

        startup = manager.camera_startup_status()
        self.assertEqual(startup["counts"], {"failed": 1})
        self.assertEqual(startup["cameras"]["gate"]["error"], "capture failed")
        self.assertTrue(manager._started)
        manager.recorder.start.assert_not_called()

        manager.stop_all()

    def test_manager_returns_while_admitted_camera_warms_with_recorder(self) -> None:
        manager = manager_with_mocks()
        frame_ready = threading.Event()
        manager.camera_fleet.startup = CameraStartupCoordinator(
            readiness_timeout_seconds=1.0,
            recorder_settle_seconds=0.0,
            poll_interval_seconds=0.01,
        )
        camera_started = threading.Event()
        manager.workers["gate"].start.side_effect = camera_started.set
        manager.workers["gate"].live_capture_ready.side_effect = frame_ready.is_set

        manager.start_all()

        self.assertTrue(manager._started)
        self.assertTrue(camera_started.wait(timeout=1))
        manager.workers["gate"].start.assert_called_once_with()
        deadline = time.monotonic() + 1
        while not manager.recorder.start.called and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        manager.recorder.start.assert_called_once_with(manager.config.cameras[0], "main")
        self.assertEqual(
            manager.camera_startup_status()["cameras"]["gate"]["phase"],
            "waiting_for_frame",
        )

        frame_ready.set()
        self.assertTrue(manager.wait_for_camera_startup(timeout=1))
        self.assertEqual(
            [call.args[0] for call in manager.mqtt.set_server_lifecycle.call_args_list],
            ["starting", "running"],
        )

        manager.stop_all()

    def test_shutdown_continues_after_one_cleanup_failure(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.inference.close.side_effect = RuntimeError("face recognition failed")

        with self.assertRaisesRegex(RuntimeError, "inference lifecycle"):
            manager.stop_all()

        manager.inference.close.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()
        manager.state_events.close.assert_called_once_with()
        self.assertTrue(manager._closed)

    def test_camera_close_runs_even_when_camera_stop_fails(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.workers["gate"].request_stop.side_effect = RuntimeError("stop failed")

        with self.assertRaisesRegex(RuntimeError, "camera gate"):
            manager.stop_all()

        manager.workers["gate"].close.assert_called_once_with()
        manager.inference.close.assert_called_once_with()
        manager.recorder.stop_all.assert_called_once_with()

    def test_shutdown_aggregate_does_not_chain_secret_bearing_camera_error(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.workers["gate"].request_stop.side_effect = RuntimeError(
            "rtsp://admin:supersecret@192.0.2.10/live"
        )

        with self.assertRaises(RuntimeError) as raised:
            manager.stop_all()

        self.assertNotIn("supersecret", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_manager_camera_shutdown_deadline_does_not_block_other_cleanup(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.workers["gate"].wait_stopped.return_value = False
        manager.workers["gate"].active_workers.return_value = ["capture: live"]

        with (
            patch("survng.app.camera_fleet.CAMERA_SHUTDOWN_TIMEOUT_SECONDS", 0.02),
            self.assertRaisesRegex(RuntimeError, "gate"),
        ):
            manager.stop_all()

        manager.workers["gate"].close.assert_not_called()
        manager.inference.close.assert_not_called()
        manager.recorder.stop_all.assert_not_called()
        self.assertFalse(manager._closed)

    def test_onvif_release_deadline_is_bounded(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].wait_onvif_stopped.return_value = False

        with (
            patch("survng.app.camera_fleet.ONVIF_RELEASE_TIMEOUT_SECONDS", 0.02),
            self.assertRaisesRegex(RuntimeError, "gate"),
        ):
            manager.release_onvif_subscriptions()

    def test_onvif_shutdown_residual_keeps_shared_dependencies_alive(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        manager.workers["gate"].wait_onvif_stopped.return_value = False

        with self.assertRaisesRegex(RuntimeError, "gate"):
            manager.stop_all()

        manager.workers["gate"].request_stop.assert_not_called()
        manager.inference.close.assert_not_called()
        manager.recorder.stop_all.assert_not_called()
        self.assertFalse(manager._closed)

    def test_stuck_startup_admission_prevents_camera_stop_and_shared_teardown(self) -> None:
        manager = manager_with_mocks()
        manager._started = True

        with (
            patch.object(manager.camera_fleet.startup, "cancel", return_value=False),
            self.assertRaisesRegex(RuntimeError, "gate"),
        ):
            manager.stop_all()

        manager.workers["gate"].request_onvif_stop.assert_not_called()
        manager.workers["gate"].request_stop.assert_not_called()
        manager.inference.close.assert_not_called()
        manager.recorder.stop_all.assert_not_called()
        self.assertEqual(
            manager.camera_fleet.status()["shutdown_residual_camera_ids"],
            ["gate"],
        )
        self.assertFalse(manager._closed)

    def test_shutdown_releases_onvif_before_other_camera_components(self) -> None:
        manager = manager_with_mocks()
        manager._started = True
        order: list[str] = []
        manager.workers["gate"].request_onvif_stop.side_effect = (
            lambda: order.append("onvif")
        )
        manager.workers["gate"].request_stop.side_effect = lambda: order.append("camera")
        manager.mqtt.stop.side_effect = lambda: order.append("mqtt")

        manager.stop_all()

        self.assertLess(order.index("onvif"), order.index("mqtt"))
        self.assertLess(order.index("onvif"), order.index("camera"))

    def test_early_onvif_release_keeps_video_worker_running(self) -> None:
        manager = manager_with_mocks()

        manager.release_onvif_subscriptions()

        manager.workers["gate"].request_onvif_stop.assert_called_once_with()
        manager.workers["gate"].request_stop.assert_not_called()
        manager.recorder.stop_all.assert_not_called()

    def test_failed_camera_stop_remains_truthfully_off(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].stop.side_effect = RuntimeError("stop failed")

        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            manager.stop_camera("gate")

        self.assertFalse(manager.camera_controls.camera_enabled("gate"))
        self.assertFalse(manager.camera_fleet.camera_enabled("gate"))
        self.assertEqual(
            manager.recorder.set_camera_enabled.call_args_list,
            [unittest.mock.call("gate", False), unittest.mock.call("gate", False)],
        )
        self.assertEqual(manager.camera_controls._persist_locked.call_count, 1)
        manager.mqtt.publish_camera_state.assert_called_once_with("gate", False)

    def test_camera_power_persistence_failure_does_not_touch_runtime(self) -> None:
        manager = manager_with_mocks()
        manager.camera_controls._persist_locked.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.stop_camera("gate")

        self.assertTrue(manager.camera_controls.camera_enabled("gate"))
        self.assertTrue(manager.camera_fleet.camera_enabled("gate"))
        manager.recorder.set_camera_enabled.assert_not_called()
        manager.workers["gate"].stop.assert_not_called()

    def test_start_is_idempotent_and_stop_is_terminal(self) -> None:
        manager = manager_with_mocks()

        manager.start_all()
        manager.start_all()
        self.assertTrue(manager.wait_for_camera_startup(timeout=1))
        manager.stop_all()
        manager.stop_all()

        manager.inference.start_core.assert_called_once_with()
        manager.inference.start_auxiliary.assert_called_once_with()
        manager.workers["gate"].start.assert_called_once_with()
        manager.workers["gate"].request_stop.assert_called_once_with()
        manager.inference.close.assert_called_once_with()

    def test_runtime_preferences_are_filtered_for_a_replacement_manager(self) -> None:
        manager = manager_with_mocks()

        manager.apply_runtime_preferences(
            {
                "recording_enabled": {"gate": False, "removed": False},
                "detection_enabled": {"gate": True, "removed": True},
            }
        )

        self.assertEqual(
            manager.runtime_preferences(),
            {
                "recording_enabled": {"gate": False},
                "detection_enabled": {"gate": True},
                "camera_enabled": {"gate": True},
            },
        )

    def test_persisted_runtime_preferences_roll_back_on_write_failure(self) -> None:
        manager = manager_with_mocks()
        previous = manager.runtime_preferences()
        manager.camera_controls._persist_locked.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.apply_runtime_preferences(
                {
                    "recording_enabled": {"gate": False},
                    "detection_enabled": {"gate": False},
                    "camera_enabled": {"gate": False},
                },
                persist=True,
            )

        self.assertEqual(manager.runtime_preferences(), previous)

    def test_fresh_process_restores_saved_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "runtime_state.json").write_text(
                json.dumps({
                    "recording_enabled": {"gate": False},
                    "detection_enabled": {"gate": False},
                    "camera_enabled": {"gate": False},
                }),
                encoding="utf-8",
            )
            manager = AppManager(AppConfig(
                storage_dir=tmpdir,
                cameras=[CameraConfig(
                    id="gate",
                    name="Gate",
                    stream_url="rtsp://camera/main",
                )],
            ))

            self.assertEqual(
                manager.runtime_preferences(),
                {
                    "recording_enabled": {"gate": False},
                    "detection_enabled": {"gate": False},
                    "camera_enabled": {"gate": False},
                },
            )
            self.assertFalse(manager.recording_enabled("gate"))
            self.assertFalse(manager.detection_enabled("gate"))

            manager.stop_all()

    def test_runtime_preference_write_failure_rolls_back_memory(self) -> None:
        manager = manager_with_mocks()
        manager.camera_controls.apply({"recording_enabled": {"gate": True}})
        manager.camera_controls._persist_locked.side_effect = OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            manager.set_recording("gate", False)

        self.assertEqual(
            manager.camera_controls.snapshot()["recording_enabled"],
            {"gate": True},
        )
        manager.recorder.set_camera_enabled.assert_not_called()

    def test_runtime_state_write_is_atomic_and_cleans_failed_temporary_file(self) -> None:
        manager = manager_with_mocks()
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.camera_controls._state_path = Path(tmpdir) / "runtime_state.json"
            manager.camera_controls._state_path.write_text('{"original": true}\n', encoding="utf-8")
            manager.camera_controls._persist_locked = (
                CameraControlService._persist_locked.__get__(manager.camera_controls)
            )
            with patch("survng.app.camera_control.json.dump", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    manager.camera_controls.persist()
            contents = manager.camera_controls._state_path.read_text(encoding="utf-8")
            temporary_files = list(Path(tmpdir).glob(".runtime_state.json.*.tmp"))

        self.assertEqual(contents, '{"original": true}\n')
        self.assertEqual(temporary_files, [])

    def test_disabled_camera_remains_stopped_during_manager_start(self) -> None:
        manager = manager_with_mocks()
        manager.camera_controls.apply({"camera_enabled": {"gate": False}})

        manager.start_all()
        self.assertTrue(manager.wait_for_camera_startup(timeout=1))

        self.assertEqual(manager.camera_controls._persist_locked.call_count, 1)
        manager.workers["gate"].start.assert_not_called()
        manager.recorder.set_camera_enabled.assert_called_with("gate", False)
        manager.mqtt.publish_camera_state.assert_called_with("gate", False)
        manager.stop_all()

    def test_failed_camera_start_restores_previous_recorder_state(self) -> None:
        manager = manager_with_mocks()
        manager.workers["gate"].start.side_effect = RuntimeError("capture failed")

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            manager.start_camera("gate")

        self.assertTrue(manager.camera_controls.camera_enabled("gate"))
        self.assertEqual(
            manager.recorder.set_camera_enabled.call_args_list[-1].args,
            ("gate", True),
        )

    def test_motion_reconfiguration_publish_failure_retires_replacement(self) -> None:
        manager = manager_with_mocks()
        camera = manager.config.cameras[0]
        previous = manager.workers["gate"]
        previous.camera = camera
        replacement = Mock()
        replacement.camera = camera
        replacement.close.side_effect = lambda: self.assertTrue(
            replacement.stop.called
        )
        previous_evidence = object()
        replacement_evidence = object()
        manager.motion_evidence = {"gate": previous_evidence}

        def create_replacement(_camera: CameraConfig):
            manager.motion_evidence["gate"] = replacement_evidence
            return replacement

        manager._create_camera_worker = Mock(side_effect=create_replacement)
        manager.camera_fleet.replace_worker = Mock(
            side_effect=[RuntimeError("fleet publication failed"), None]
        )
        manager.camera_controls.replace_worker = Mock()
        manager.inference.replace_worker = Mock()

        with self.assertRaisesRegex(RuntimeError, "fleet publication failed"):
            manager.reconfigure_motion(
                manager.config,
                restart_camera_ids={"gate"},
                hot_camera_ids=set(),
            )

        replacement.start.assert_called_once_with()
        replacement.stop.assert_called_once_with()
        replacement.close.assert_called_once_with()
        previous.start.assert_called_once_with()
        self.assertIs(manager.workers["gate"], previous)
        self.assertIs(manager.motion_evidence["gate"], previous_evidence)
        self.assertEqual(manager.camera_fleet.replace_worker.call_count, 2)
        manager.camera_controls.replace_worker.assert_called_once_with(
            camera,
            previous,
        )
        manager.inference.replace_worker.assert_called_once_with(
            "gate",
            previous,
        )


if __name__ == "__main__":
    unittest.main()
