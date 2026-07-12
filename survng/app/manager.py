from __future__ import annotations

import threading
from pathlib import Path

from .camera import CameraWorker
from .config import AppConfig
from .detector import OpenVinoDetector
from .events import EventStore
from .hls import HlsStreamer
from .recorder import Recorder


class AppManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventStore(self.storage_dir)
        self.detector = OpenVinoDetector(config.detector)
        self.recorder = Recorder(config.ffmpeg_path, self.storage_dir, config.recording_segment_seconds, config.hardware_acceleration)
        self.hls = HlsStreamer(config.ffmpeg_path, self.storage_dir, config.hardware_acceleration)
        self.workers = {
            camera.id: CameraWorker(camera, self.storage_dir, self.detector, self.events, self.recorder)
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
        self.recorder.start_indexer(cameras)
        self.recorder.start_watchdog(cameras)

    def stop_all(self) -> None:
        shutdowns = [
            threading.Thread(target=worker.stop, daemon=True)
            for worker in self.workers.values()
        ]
        shutdowns.extend(
            [
                threading.Thread(target=self.recorder.stop_all, daemon=True),
                threading.Thread(target=self.hls.stop_all, daemon=True),
            ]
        )
        for thread in shutdowns:
            thread.start()
        for thread in shutdowns:
            thread.join(timeout=15)


    def camera(self, camera_id: str):
        return next((camera for camera in self.config.cameras if camera.id == camera_id), None)

    def start_camera(self, camera_id: str) -> bool:
        camera = self.camera(camera_id)
        worker = self.workers.get(camera_id)
        if camera is None or worker is None:
            return False
        worker.start()
        if camera.record:
            self.recorder.start(camera, "main")
        if camera.record_sub and camera.live_stream_url:
            self.recorder.start(camera, "live")
        return True

    def stop_camera(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.stop()
        self.recorder.stop(camera_id)
        self.hls.stop_camera_sources(camera_id)
        return True

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
