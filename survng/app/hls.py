from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from .baichuan_native import BaichuanFfmpegPipe
from .baichuan_native import ffmpeg_input_args, start_ffmpeg_pipe
from .baichuan_native import is_native_baichuan
from .config import CameraConfig
from .ffmpeg_hw import encoder_device_args, hls_video_args


class HlsStreamer:
    def __init__(
        self,
        ffmpeg_path: str,
        storage_dir: Path,
        hardware_acceleration: str = "auto",
        idle_timeout: float = 20.0,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.hardware_acceleration = hardware_acceleration
        self.idle_timeout = max(10.0, float(idle_timeout))
        self.hls_dir = storage_dir / "hls"
        self.hls_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, tuple[subprocess.Popen, BaichuanFfmpegPipe | None]] = {}
        self.last_access: dict[str, float] = {}
        self._lock = threading.RLock()
        self._start_locks: dict[str, threading.Lock] = {}
        self._shutdown = threading.Event()
        self._reaper = threading.Thread(target=self._reap_idle, name="hls-idle-reaper", daemon=True)
        self._reaper.start()

    def start(self, camera: CameraConfig, source: str = "live") -> Path:
        key = self._key(camera.id, source)
        with self._lock:
            start_lock = self._start_locks.setdefault(key, threading.Lock())
        with start_lock:
            return self._start(camera, source)

    def _start(self, camera: CameraConfig, source: str) -> Path:
        key = self._key(camera.id, source)
        self.touch(camera.id, source)
        playlist = self.playlist_path(camera.id, source)
        with self._lock:
            item = self.processes.get(key)
        if item is not None and item[0].poll() is None:
            return playlist

        self.stop(key)
        camera_dir = self.hls_dir / camera.id / source
        camera_dir.mkdir(parents=True, exist_ok=True)
        for path in camera_dir.glob("*"):
            if path.is_file():
                path.unlink()

        keyframe_seconds = "1"
        hls_time = "1"
        hls_list_size = "5" if source != "main" else "6"
        segment_pattern = "segment_%05d.ts"
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            *encoder_device_args(self.hardware_acceleration),
            *ffmpeg_input_args(camera, source),
            "-an",
            "-sn",
            *hls_video_args(self.hardware_acceleration),
            "-force_key_frames",
            f"expr:gte(t,n_forced*{keyframe_seconds})",
            "-sc_threshold",
            "0",
            "-flush_packets",
            "1",
            "-f",
            "hls",
            "-hls_time",
            hls_time,
            "-hls_list_size",
            hls_list_size,
            "-hls_flags",
            "delete_segments+program_date_time+omit_endlist+independent_segments",
            "-hls_segment_filename",
            segment_pattern,
            playlist.name,
        ]
        process = subprocess.Popen(
            command,
            cwd=camera_dir,
            stdin=subprocess.PIPE if is_native_baichuan(camera) else None,
        )
        pipe = start_ffmpeg_pipe(camera, source, process)
        with self._lock:
            self.processes[key] = (process, pipe)
        return playlist

    def stop(self, key: str) -> None:
        with self._lock:
            item = self.processes.pop(key, None)
            self.last_access.pop(key, None)
        if item is None:
            return
        process, pipe = item
        if pipe is not None:
            pipe.stop()
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def stop_all(self) -> None:
        self._shutdown.set()
        self._reaper.join(timeout=2)
        with self._lock:
            keys = list(self.processes)
        shutdowns = [threading.Thread(target=self.stop, args=(key,), daemon=True) for key in keys]
        for thread in shutdowns:
            thread.start()
        for thread in shutdowns:
            thread.join(timeout=7)

    def stop_camera_sources(self, camera_id: str, except_key: str = "") -> None:
        prefix = f"{camera_id}:"
        with self._lock:
            keys = list(self.processes)
        for key in keys:
            if key.startswith(prefix) and key != except_key:
                self.stop(key)

    def touch(self, camera_id: str, source: str = "live") -> None:
        with self._lock:
            self.last_access[self._key(camera_id, source)] = time.monotonic()

    def _reap_idle(self) -> None:
        while not self._shutdown.wait(2.0):
            cutoff = time.monotonic() - self.idle_timeout
            with self._lock:
                idle_keys = [
                    key for key in self.processes
                    if self.last_access.get(key, 0.0) < cutoff
                ]
            for key in idle_keys:
                self.stop(key)

    def playlist_path(self, camera_id: str, source: str = "live") -> Path:
        return self.hls_dir / camera_id / source / "index.m3u8"

    def file_path(self, camera_id: str, source: str, filename: str) -> Path:
        return self.hls_dir / camera_id / source / Path(filename).name

    def wait_for_playlist(
        self,
        camera_id: str,
        source: str = "live",
        timeout: float = 12.0,
    ) -> Path | None:
        playlist = self.playlist_path(camera_id, source)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if playlist.exists() and playlist.stat().st_size > 0:
                return playlist
            time.sleep(0.2)
        return playlist if playlist.exists() else None

    def _key(self, camera_id: str, source: str) -> str:
        return f"{camera_id}:{source}"
