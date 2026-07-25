from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from .camera import CameraWorker
from .config import AppConfig, DetectionZone
from .events import EventStore
from .faces import FaceStore
from .detector import objects_to_json
from .go2rtc import Go2RtcAdapter
from .inference import InferenceSupervisor, IsolatedFaceRecognizer
from .mqtt import MqttService
from .motion_pipeline import (
    LoggingMotionPipelineObserver,
    MotionDecisionHandlerFactory,
    MotionPipelineFactory,
    RecordedMotionObjectDetectorFactory,
    build_builtin_motion_registry,
    default_motion_stage_configs,
)
from .recorder import Recorder
from .state_events import StateEventBroker


LOGGER = logging.getLogger("uvicorn.error")


class AppManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventStore(self.storage_dir)
        self.detector = InferenceSupervisor(config.detector)
        self.face_recognizer = IsolatedFaceRecognizer(self.detector)
        self.faces = FaceStore(
            self.storage_dir,
            config.detector.face_max_observations,
            self.face_recognizer,
            start_recognition=False,
        )
        recording_index_dir = Path(config.recording_index_dir) if config.recording_index_dir else None
        self.recorder = Recorder(
            config.ffmpeg_path,
            self.storage_dir,
            config.recording_segment_seconds,
            config.hardware_acceleration,
            index_dir=recording_index_dir,
        )
        self.go2rtc = Go2RtcAdapter()
        self.state_events = StateEventBroker()
        self.motion_pipeline_factory = MotionPipelineFactory(
            registry=build_builtin_motion_registry(),
            observer=LoggingMotionPipelineObserver(),
        )
        self.motion_decision_handler_factory = MotionDecisionHandlerFactory(
            events=self.events,
            object_serializer=objects_to_json,
        )
        self.motion_object_detector_factory = RecordedMotionObjectDetectorFactory(
            detector=self.detector,
            recorder=self.recorder,
        )
        self.mqtt = MqttService(
            config.mqtt,
            self._mqtt_power_command,
            self.set_recording,
            self.set_detection,
            self._mqtt_connected,
        )
        self._lifecycle_lock = threading.Lock()
        self._runtime_state_lock = threading.Lock()
        self._runtime_state_path = self.storage_dir / "runtime_state.json"
        runtime_state = self._load_runtime_state()
        self._recording_enabled = self._boolean_preferences(runtime_state, "recording_enabled")
        self._detection_enabled = self._boolean_preferences(runtime_state, "detection_enabled")
        self._camera_enabled = {camera.id: True for camera in config.cameras}
        self._stopping = False
        self._state_monitor_stop = threading.Event()
        self._state_monitor_thread: threading.Thread | None = None
        self.workers = {
            camera.id: CameraWorker(
                camera,
                self.storage_dir,
                config.motion_qualification,
                self.publish_event,
                motion_pipeline=self.motion_pipeline_factory.create(
                    camera.id,
                    default_motion_stage_configs(),
                ),
                motion_decision_handler_factory=self.motion_decision_handler_factory,
                motion_object_detector_factory=self.motion_object_detector_factory,
            )
            for camera in config.cameras
        }

    def _unique_cameras(self):
        seen: set[str] = set()
        for camera in self.config.cameras:
            if camera.id in seen:
                continue
            seen.add(camera.id)
            yield camera

    def _load_runtime_state(self) -> dict:
        try:
            payload = json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _boolean_preferences(payload: dict, key: str) -> dict[str, bool]:
        values = payload.get(key, {})
        if not isinstance(values, dict):
            return {}
        return {str(camera_id): bool(enabled) for camera_id, enabled in values.items()}

    def _save_runtime_state(self) -> None:
        with self._runtime_state_lock:
            payload = {
                "recording_enabled": self._recording_enabled,
                "detection_enabled": self._detection_enabled,
            }
            temporary = self._runtime_state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self._runtime_state_path)

    def recording_enabled(self, camera_id: str) -> bool:
        return self._recording_enabled.get(camera_id, True)

    def detection_enabled(self, camera_id: str) -> bool:
        return self._detection_enabled.get(camera_id, True)

    def _recorder_should_run(self, camera_id: str) -> bool:
        return self._camera_enabled.get(camera_id, True) and self.recording_enabled(camera_id)

    def _start_configured_recorders(self, camera) -> None:
        if camera.record:
            self.recorder.start(camera, "main")
        if camera.record_sub and camera.live_stream_url:
            self.recorder.start(camera, "live")

    def set_recording(self, camera_id: str, enabled: bool) -> bool:
        camera = self.camera(camera_id)
        if camera is None:
            return False
        self._recording_enabled[camera_id] = bool(enabled)
        self._save_runtime_state()
        should_run = self._recorder_should_run(camera_id)
        self.recorder.set_camera_enabled(camera_id, should_run)
        if should_run:
            self._start_configured_recorders(camera)
        self.mqtt.publish_camera_feature_state(camera_id, "recording", bool(enabled))
        self._publish_camera_status(camera_id)
        return True

    def set_detection(self, camera_id: str, enabled: bool) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        self._detection_enabled[camera_id] = bool(enabled)
        self._save_runtime_state()
        worker.set_detection_enabled(enabled)
        self.mqtt.publish_camera_feature_state(camera_id, "detection", bool(enabled))
        self._publish_camera_status(camera_id)
        return True

    def start_all(self) -> None:
        with self._lifecycle_lock:
            self._stopping = False
        self.detector.start()
        self.faces.start()
        cameras = list(self._unique_cameras())
        recorder_keys = set()
        for camera in cameras:
            if camera.record:
                recorder_keys.add((camera.id, "main"))
            if camera.record_sub and camera.live_stream_url:
                recorder_keys.add((camera.id, "live"))
        self.recorder.cleanup_stale_recorders(recorder_keys)
        for camera in cameras:
            self._camera_enabled[camera.id] = True
            self.recorder.set_camera_enabled(camera.id, self.recording_enabled(camera.id))
            self.workers[camera.id].set_detection_enabled(self.detection_enabled(camera.id))
            self.workers[camera.id].start()
            if self.recording_enabled(camera.id):
                self._start_configured_recorders(camera)
            self.mqtt.publish_camera_state(camera.id, True)
        self.recorder.start_indexer(cameras)
        self.recorder.start_watchdog(cameras)
        # Publish discovery only after persisted recording/detection preferences
        # have been applied to every worker.
        self.mqtt.start()
        self._start_state_monitor()

    def stop_all(self) -> None:
        with self._lifecycle_lock:
            if self._stopping:
                return
            self._stopping = True
        self._stop_state_monitor()
        started = time.monotonic()
        LOGGER.info("SurvNG shutdown: stopping MQTT command intake")
        self.mqtt.stop()
        LOGGER.info("SurvNG shutdown: stopping camera and ONVIF workers")
        camera_shutdowns = [
            threading.Thread(
                target=worker.stop,
                name=f"stop-camera-{camera_id}",
                daemon=False,
            )
            for camera_id, worker in self.workers.items()
        ]
        for thread in camera_shutdowns:
            thread.start()
        for thread in camera_shutdowns:
            thread.join()

        LOGGER.info("SurvNG shutdown: stopping face recognition")
        self.faces.close()

        LOGGER.info("SurvNG shutdown: stopping isolated inference workers")
        self.detector.stop()

        LOGGER.info("SurvNG shutdown: stopping recorder processes")
        self.recorder.stop_all()
        LOGGER.info("SurvNG shutdown complete in %.2fs", time.monotonic() - started)
        self.state_events.close()


    def camera(self, camera_id: str):
        return next((camera for camera in self.config.cameras if camera.id == camera_id), None)

    def start_camera(self, camera_id: str) -> bool:
        with self._lifecycle_lock:
            if self._stopping:
                return False
        camera = self.camera(camera_id)
        worker = self.workers.get(camera_id)
        if camera is None or worker is None:
            return False
        self._camera_enabled[camera_id] = True
        self.recorder.set_camera_enabled(camera_id, self.recording_enabled(camera_id))
        worker.start()
        if self.recording_enabled(camera_id):
            self._start_configured_recorders(camera)
        self.mqtt.publish_camera_state(camera_id, True)
        self._publish_camera_status(camera_id)
        return True

    def stop_camera(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        self._camera_enabled[camera_id] = False
        self.recorder.set_camera_enabled(camera_id, False)
        worker.stop()
        self.mqtt.publish_camera_state(camera_id, False)
        self._publish_camera_status(camera_id)
        return True

    def update_camera_zones(
        self,
        camera_id: str,
        zones: list[DetectionZone],
        previous_zones: list[dict],
    ) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.update_zones(zones)
        self.mqtt.remove_zone_discovery(camera_id, previous_zones, self.detector.labels)
        self._mqtt_connected()
        return True

    def _mqtt_power_command(self, camera_id: str, turn_on: bool) -> bool:
        return self.start_camera(camera_id) if turn_on else self.stop_camera(camera_id)

    def _mqtt_connected(self) -> None:
        self.mqtt.publish_discovery([
            {
                "id": camera.id,
                "name": camera.name,
                "model_classes": self.detector.labels,
                "recording_configured": bool(camera.record or camera.record_sub),
                "zones": [
                    {
                        "name": zone.name,
                        "enabled": zone.enabled,
                        "object_classes": zone.object_classes or self.detector.labels,
                    }
                    for zone in camera.zones
                ],
            }
            for camera in self._unique_cameras()
        ])
        for status in self.statuses():
            camera_id = str(status.get("id") or "")
            self.mqtt.publish_camera_state(camera_id, bool(status.get("running")))
            self.mqtt.publish_camera_feature_state(camera_id, "recording", bool(status.get("recording_enabled")))
            self.mqtt.publish_camera_feature_state(camera_id, "detection", bool(status.get("detection_enabled")))

    def publish_event(self, event_type: str, payload: dict) -> None:
        camera_id = str(payload.get("camera_id") or "")
        if not camera_id:
            return
        if event_type == "incident" or (event_type == "object" and payload.get("source") == "manual_openvino"):
            event_id = int(payload.get("event_id") or 0)
            event = self.events.get(event_id) if event_id else None
            camera = self.camera(camera_id)
            if event is not None:
                self.mqtt.track_incident(
                    event,
                    camera.name if camera is not None else camera_id,
                    self.config.base_path,
                    allow_new=event_type == "incident",
                )
        if event_type == "object":
            objects = payload.get("objects") or []
            event_id = payload.get("event_id")
            if event_id:
                event = self.events.get(int(event_id))
                if event:
                    self.faces.ingest_events([event])
            payload = {
                **payload,
                "classes": sorted({str(item.get("label")) for item in objects if item.get("label")}),
                "zones": sorted({str(zone) for item in objects for zone in item.get("zones", []) if zone}),
            }
        self.mqtt.publish(f"camera/{camera_id}/{event_type}", payload)
        self.state_events.publish(event_type, payload)
        if event_type == "object":
            camera = self.camera(camera_id)
            if camera is not None:
                self.mqtt.publish_zone_objects(
                    camera_id,
                    [
                        {
                            "name": zone.name,
                            "enabled": zone.enabled,
                            "object_classes": zone.object_classes or self.detector.labels,
                        }
                        for zone in camera.zones
                    ],
                    payload,
                )

    @staticmethod
    def _camera_state_fingerprint(status: dict) -> tuple:
        keys = (
            "running", "connected", "capture_running", "frame_fresh", "main_running",
            "main_frame_fresh", "last_error", "main_last_error", "onvif_connected",
            "onvif_last_event_at", "last_motion_at", "detection_enabled", "recording",
            "sub_recording", "recording_enabled", "recording_configured",
        )
        motion = status.get("motion_qualification") or {}
        return tuple(status.get(key) for key in keys) + (
            motion.get("passed"),
            motion.get("audit_rejected"),
            motion.get("suppressed"),
            motion.get("last_decision_at"),
        )

    def _publish_camera_status(self, camera_id: str) -> None:
        status = next((item for item in self.statuses() if item.get("id") == camera_id), None)
        if status is not None:
            self.state_events.publish("camera_state", status)

    def _start_state_monitor(self) -> None:
        if self._state_monitor_thread is not None and self._state_monitor_thread.is_alive():
            return
        self._state_monitor_stop.clear()

        def monitor() -> None:
            previous: dict[str, tuple] = {}
            while not self._state_monitor_stop.is_set():
                try:
                    for status in self.statuses():
                        camera_id = str(status.get("id") or "")
                        fingerprint = self._camera_state_fingerprint(status)
                        if camera_id and previous.get(camera_id) != fingerprint:
                            previous[camera_id] = fingerprint
                            self.state_events.publish("camera_state", status)
                except Exception:
                    LOGGER.exception("camera state monitor failed")
                self._state_monitor_stop.wait(1.0)

        self._state_monitor_thread = threading.Thread(target=monitor, name="camera-state-monitor", daemon=False)
        self._state_monitor_thread.start()

    def _stop_state_monitor(self) -> None:
        self._state_monitor_stop.set()
        thread = self._state_monitor_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._state_monitor_thread = None

    def mqtt_status(self) -> dict:
        return self.mqtt.status()

    def statuses(self) -> list[dict]:
        recording_keys = set()
        camera_config = {camera.id: camera for camera in self._unique_cameras()}
        for camera in camera_config.values():
            if camera.record:
                recording_keys.add((camera.id, "main"))
            if camera.record_sub and camera.live_stream_url:
                recording_keys.add((camera.id, "live"))
        recordings = self.recorder.status(recording_keys)
        return [
            {
                **worker.status(),
                "recording": recordings.get((camera_id, "main"), False),
                "sub_recording": recordings.get((camera_id, "live"), False),
                "recording_enabled": self.recording_enabled(camera_id),
                "recording_configured": bool(
                    camera_config.get(camera_id)
                    and (camera_config[camera_id].record or camera_config[camera_id].record_sub)
                ),
                "record_sub_enabled": bool(camera_config.get(camera_id) and camera_config[camera_id].record_sub),
            }
            for camera_id, worker in self.workers.items()
        ]

    def detector_status(self) -> dict:
        return self.detector.status()

    def go2rtc_status(self) -> dict:
        return self.go2rtc.status(list(self._unique_cameras()))
