from __future__ import annotations

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
        self.recorder = Recorder(config.ffmpeg_path, self.storage_dir)
        self.hls = HlsStreamer(config.ffmpeg_path, self.storage_dir)
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
        self.recorder.cleanup_stale_recorders({camera.id for camera in cameras if camera.record})
        for camera in cameras:
            self.workers[camera.id].start()
            if camera.record:
                self.recorder.start(camera)
        self.recorder.start_watchdog(cameras)

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        self.recorder.stop_all()
        self.hls.stop_all()


    def camera(self, camera_id: str):
        return next((camera for camera in self.config.cameras if camera.id == camera_id), None)

    def start_camera(self, camera_id: str) -> bool:
        camera = self.camera(camera_id)
        worker = self.workers.get(camera_id)
        if camera is None or worker is None:
            return False
        worker.start()
        if camera.record:
            self.recorder.start(camera)
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
        recording_ids = {camera.id for camera in self._unique_cameras() if camera.record}
        recordings = self.recorder.status(recording_ids)
        return [
            {
                **worker.status(),
                "recording": recordings.get(camera_id, False),
            }
            for camera_id, worker in self.workers.items()
        ]

    def detector_status(self) -> dict:
        return self.detector.status()
