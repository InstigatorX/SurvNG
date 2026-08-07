"""Lifecycle and reconfiguration ownership for continuous recording."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, CameraConfig, RecordingRetentionConfig
from .recorder import Recorder


@dataclass(frozen=True, slots=True)
class RecordingStartupTimings:
    cleanup_seconds: float
    services_seconds: float


class RecordingLifecycle:
    """Own the process-wide recorder generation and its transaction boundaries.

    Camera fleet admission still decides *when* an individual camera may start.
    This owner controls the shared indexer/watchdog lifetime and ensures an
    engine reconfiguration either completes or restores the previous runtime.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        storage_dir: Path,
        protected_recording_paths: Callable[[], set[str]],
        recorder: Recorder | None = None,
    ) -> None:
        recording_index_dir = (
            Path(config.recording_index_dir)
            if config.recording_index_dir
            else None
        )
        self.recorder = recorder or Recorder(
            config.ffmpeg_path,
            storage_dir,
            config.recording_segment_seconds,
            config.hardware_acceleration,
            index_dir=recording_index_dir,
            retention_config=config.retention,
            protected_recording_paths=protected_recording_paths,
        )
        self._lock = threading.RLock()
        self._closed = False

    def start_services(
        self,
        cameras: Sequence[CameraConfig],
        recorder_keys: set[tuple[str, str]],
    ) -> RecordingStartupTimings:
        """Clean stale processes and start shared recording maintenance."""
        with self._lock:
            if self._closed:
                raise RuntimeError("recording lifecycle is closed")
            camera_list = list(cameras)
            started = time.monotonic()
            self.recorder.cleanup_stale_recorders(recorder_keys)
            cleanup_seconds = time.monotonic() - started
            started = time.monotonic()
            try:
                self.recorder.start_indexer(camera_list)
                self.recorder.start_watchdog(camera_list)
            except BaseException:
                self.recorder.stop_all()
                raise
            return RecordingStartupTimings(
                cleanup_seconds=cleanup_seconds,
                services_seconds=time.monotonic() - started,
            )

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self.recorder.set_camera_enabled(camera_id, enabled)

    def start_camera(self, camera: CameraConfig) -> None:
        with self._lock:
            if self._closed:
                return
            if camera.record:
                self.recorder.start(camera, "main")
            if camera.record_sub and camera.live_stream_url:
                self.recorder.start(camera, "live")

    def reconfigure(
        self,
        next_config: AppConfig,
        cameras: Sequence[CameraConfig],
        desired_enabled: Mapping[str, bool],
        *,
        restart_recorders: bool,
    ) -> None:
        """Atomically replace recorder engine settings, with runtime rollback."""
        with self._lock:
            if self._closed:
                raise RuntimeError("recording lifecycle is closed")
            camera_list = list(cameras)
            previous = {
                "ffmpeg_path": self.recorder.ffmpeg_path,
                "hardware_acceleration": self.recorder.hardware_acceleration,
                "segment_seconds": self.recorder.segment_seconds,
            }
            for camera in camera_list:
                self.recorder.set_camera_enabled(camera.id, False)
            try:
                self.recorder.reconfigure_runtime(
                    ffmpeg_path=next_config.ffmpeg_path,
                    hardware_acceleration=next_config.hardware_acceleration,
                    segment_seconds=next_config.recording_segment_seconds,
                )
                for camera in camera_list:
                    self.recorder.set_camera_enabled(
                        camera.id,
                        bool(desired_enabled.get(camera.id, False)),
                    )
                if restart_recorders:
                    for camera in camera_list:
                        if desired_enabled.get(camera.id, False):
                            self.start_camera(camera)
            except BaseException:
                for camera in camera_list:
                    self.recorder.set_camera_enabled(camera.id, False)
                self.recorder.reconfigure_runtime(**previous)
                for camera in camera_list:
                    enabled = bool(desired_enabled.get(camera.id, False))
                    self.recorder.set_camera_enabled(camera.id, enabled)
                    if restart_recorders and enabled:
                        self.start_camera(camera)
                raise

    def reconfigure_retention(
        self,
        config: RecordingRetentionConfig,
        cameras: Sequence[CameraConfig],
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("recording lifecycle is closed")
            self.recorder.reconfigure_retention(config, list(cameras))

    def retention_status(self) -> dict[str, object]:
        return self.recorder.retention_status()

    def request_retention_run(self, *, apply: bool = False) -> dict[str, object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("recording lifecycle is closed")
            return self.recorder.request_retention_run(apply=apply)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.recorder.stop_all()
            self._closed = True
