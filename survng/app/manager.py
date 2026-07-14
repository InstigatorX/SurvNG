from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .camera import CameraWorker
from .config import AppConfig
from .detector import OpenVinoDetector
from .events import EventStore
from .face_recognition import OpenVinoFaceRecognizer
from .faces import FaceStore
from .hls import HlsStreamer
from .mqtt import MqttService
from .recorder import Recorder


LOGGER = logging.getLogger("uvicorn.error")


class AppManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventStore(self.storage_dir)
        self.detector = OpenVinoDetector(config.detector)
        self.face_recognizer = OpenVinoFaceRecognizer(config.detector)
        self.faces = FaceStore(
            self.storage_dir,
            config.detector.face_max_observations,
            self.face_recognizer,
        )
        self.recorder = Recorder(config.ffmpeg_path, self.storage_dir, config.recording_segment_seconds, config.hardware_acceleration)
        self.hls = HlsStreamer(config.ffmpeg_path, self.storage_dir, config.hardware_acceleration)
        self.mqtt = MqttService(config.mqtt, self._mqtt_power_command, self._mqtt_connected)
        self._lifecycle_lock = threading.Lock()
        self._stopping = False
        self.workers = {
            camera.id: CameraWorker(camera, self.storage_dir, self.detector, self.events, self.recorder, self.publish_event)
            for camera in config.cameras
        }

    def _unique_cameras(self):
        seen: set[str] = set()
        for camera in self.config.cameras:
            if camera.id in seen:
                continue
            seen.add(camera.id)
            yield camera

    def start_all(self) -> None:
        with self._lifecycle_lock:
            self._stopping = False
        self.mqtt.start()
        cameras = list(self._unique_cameras())
        recorder_keys = set()
        for camera in cameras:
            if camera.record:
                recorder_keys.add((camera.id, "main"))
            if camera.record_sub and camera.live_stream_url:
                recorder_keys.add((camera.id, "live"))
        self.recorder.cleanup_stale_recorders(recorder_keys)
        for camera in cameras:
            self.workers[camera.id].start()
            if camera.record:
                self.recorder.start(camera, "main")
            if camera.record_sub and camera.live_stream_url:
                self.recorder.start(camera, "live")
            self.mqtt.publish_camera_state(camera.id, True)
        self.recorder.start_indexer(cameras)
        self.recorder.start_watchdog(cameras)

    def stop_all(self) -> None:
        with self._lifecycle_lock:
            if self._stopping:
                return
            self._stopping = True
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

        LOGGER.info("SurvNG shutdown: stopping HLS and recorder processes")
        media_shutdowns = [
            threading.Thread(target=self.hls.stop_all, name="stop-hls", daemon=False),
            threading.Thread(target=self.recorder.stop_all, name="stop-recorder", daemon=False),
        ]
        for thread in media_shutdowns:
            thread.start()
        for thread in media_shutdowns:
            thread.join()
        LOGGER.info("SurvNG shutdown complete in %.2fs", time.monotonic() - started)


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
        worker.start()
        if camera.record:
            self.recorder.start(camera, "main")
        if camera.record_sub and camera.live_stream_url:
            self.recorder.start(camera, "live")
        self.mqtt.publish_camera_state(camera_id, True)
        return True

    def stop_camera(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.stop()
        self.recorder.stop(camera_id)
        self.hls.stop_camera_sources(camera_id)
        self.mqtt.publish_camera_state(camera_id, False)
        return True

    def _mqtt_power_command(self, camera_id: str, turn_on: bool) -> bool:
        return self.start_camera(camera_id) if turn_on else self.stop_camera(camera_id)

    def _mqtt_connected(self) -> None:
        self.mqtt.publish_discovery([
            {
                "id": camera.id,
                "name": camera.name,
                "model_classes": self.detector.labels,
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
            self.mqtt.publish_camera_state(str(status.get("id") or ""), bool(status.get("running")))

    def publish_event(self, event_type: str, payload: dict) -> None:
        camera_id = str(payload.get("camera_id") or "")
        if not camera_id:
            return
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
                "record_sub_enabled": bool(camera_config.get(camera_id) and camera_config[camera_id].record_sub),
            }
            for camera_id, worker in self.workers.items()
        ]

    def detector_status(self) -> dict:
        return self.detector.status()
