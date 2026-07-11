from __future__ import annotations

import os
import signal
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .baichuan_native import BaichuanFfmpegPipe, ffmpeg_input_args, is_native_baichuan, start_ffmpeg_pipe
from .config import CameraConfig


ProcessItem = tuple[subprocess.Popen, BaichuanFfmpegPipe | None, threading.Event, threading.Thread]


class Recorder:
    def __init__(self, ffmpeg_path: str, storage_dir: Path) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.recordings_dir = storage_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, ProcessItem] = {}
        self._starting: set[str] = set()
        self.owner_token = f"survng-{os.getpid()}"
        self._lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def start(self, camera: CameraConfig) -> None:
        with self._lock:
            existing = self.processes.get(camera.id)
            if existing is not None and existing[0].poll() is None:
                return
            if camera.id in self._starting:
                return
            if existing is not None:
                self.processes.pop(camera.id, None)
            self._starting.add(camera.id)

        camera_dir = self.recordings_dir / camera.id
        camera_dir.mkdir(parents=True, exist_ok=True)
        stop_event = threading.Event()
        keeper: threading.Thread | None = None
        process: subprocess.Popen | None = None
        pipe: BaichuanFfmpegPipe | None = None
        try:
            self._ensure_recording_dirs(camera_dir)
            keeper = threading.Thread(target=self._keep_recording_dirs, args=(camera_dir, stop_event), daemon=True)
            keeper.start()
            output = str(camera_dir / "%Y-%m-%d" / "%H" / "%Y%m%d-%H%M%S.mp4")
            command = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                *ffmpeg_input_args(camera, "main"),
                "-map",
                "0",
                "-metadata",
                f"survng_owner={self.owner_token}",
                "-metadata",
                f"survng_camera={camera.id}",
                "-c",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                "300",
                "-strftime",
                "1",
                "-reset_timestamps",
                "1",
                "-segment_format_options",
                "movflags=+frag_keyframe+empty_moov+default_base_moof",
                output,
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if is_native_baichuan(camera) else None,
                start_new_session=True,
            )
            pipe = start_ffmpeg_pipe(camera, "main", process)
            with self._lock:
                self.processes[camera.id] = (process, pipe, stop_event, keeper)
        except Exception:
            stop_event.set()
            if pipe is not None:
                pipe.stop()
            if process is not None and process.poll() is None:
                self._kill_pid(process.pid)
            if keeper is not None:
                keeper.join(timeout=1)
            raise
        finally:
            with self._lock:
                self._starting.discard(camera.id)
        self.cleanup_duplicate_recorders({camera.id})


    def _keep_recording_dirs(self, camera_dir: Path, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self._ensure_recording_dirs(camera_dir)
            stop_event.wait(60)

    def _ensure_recording_dirs(self, camera_dir: Path) -> None:
        now = datetime.now()
        for hours_ahead in range(3):
            target = now + timedelta(hours=hours_ahead)
            (camera_dir / target.strftime("%Y-%m-%d") / target.strftime("%H")).mkdir(parents=True, exist_ok=True)

    def stop(self, camera_id: str) -> None:
        with self._lock:
            item = self.processes.pop(camera_id, None)
            self._starting.discard(camera_id)
        if item is not None:
            self._stop_item(item)
        for pid in self._owned_ffmpeg_recorders({camera_id}).get(camera_id, []):
            self._kill_pid(pid)

    def _stop_item(self, item: ProcessItem) -> None:
        process, pipe, stop_event, keeper = item
        stop_event.set()
        if pipe is not None:
            pipe.stop()
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
        keeper.join(timeout=1)

    def status(self, camera_ids: set[str] | None = None) -> dict[str, bool]:
        with self._lock:
            stopped = [
                camera_id
                for camera_id, (process, _pipe, _stop_event, _keeper) in self.processes.items()
                if process.poll() is not None
            ]
            for camera_id in stopped:
                item = self.processes.pop(camera_id, None)
                if item is not None and item[1] is not None:
                    item[2].set()
                    item[1].stop()
                    item[3].join(timeout=1)
            tracked = {camera_id: True for camera_id in self.processes}

        if camera_ids:
            for camera_id, pids in self._owned_ffmpeg_recorders(camera_ids).items():
                if pids:
                    tracked[camera_id] = True
        return tracked

    def stop_all(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2)
            self._watchdog_thread = None
        with self._lock:
            items = list(self.processes.values())
            self.processes.clear()
        for item in items:
            self._stop_item(item)

    def start_watchdog(self, cameras: list[CameraConfig], interval: float = 15.0) -> None:
        self._watchdog_stop.clear()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        camera_map = {camera.id: camera for camera in cameras if camera.record}
        self._watchdog_thread = threading.Thread(target=self._watchdog, args=(camera_map, interval), daemon=True)
        self._watchdog_thread.start()

    def _watchdog(self, camera_map: dict[str, CameraConfig], interval: float) -> None:
        while not self._watchdog_stop.wait(interval):
            self.reconcile(camera_map)

    def reconcile(self, camera_map: dict[str, CameraConfig]) -> None:
        with self._lock:
            active_items = dict(self.processes)
        for camera_id, item in active_items.items():
            process = item[0]
            if camera_id not in camera_map:
                self.stop(camera_id)
            elif process.poll() is not None:
                self.stop(camera_id)
                self.start(camera_map[camera_id])
        with self._lock:
            active_ids = set(self.processes)
        for camera_id, camera in camera_map.items():
            if camera_id not in active_ids:
                self.start(camera)
        self.cleanup_duplicate_recorders(set(camera_map))

    def cleanup_duplicate_recorders(self, camera_ids: set[str]) -> None:
        owned = self._owned_ffmpeg_recorders(camera_ids)
        with self._lock:
            tracked_pids = {item[0].pid for item in self.processes.values()}
        for camera_id, pids in owned.items():
            extras = [pid for pid in pids if pid not in tracked_pids]
            if len([pid for pid in pids if pid in tracked_pids]) > 1:
                extras.extend(sorted(pid for pid in pids if pid in tracked_pids)[1:])
            for pid in sorted(set(extras)):
                self._kill_pid(pid)

    def cleanup_stale_recorders(self, camera_ids: set[str]) -> None:
        for pids in self._owned_ffmpeg_recorders(camera_ids).values():
            for pid in pids:
                self._kill_pid(pid)

    def _owned_ffmpeg_recorders(self, camera_ids: set[str]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {camera_id: [] for camera_id in camera_ids}
        try:
            output = subprocess.check_output(["ps", "-eo", "pid=,command="], text=True)
        except (OSError, subprocess.SubprocessError):
            return result
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            pid_text, _, command = stripped.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if "ffmpeg" not in command or "/recordings/" not in command or "%Y-%m-%d" not in command:
                continue
            for camera_id in camera_ids:
                if f"/recordings/{camera_id}/" in command:
                    result.setdefault(camera_id, []).append(pid)
                    break
        return result

    def _kill_pid(self, pid: int) -> None:
        if pid == os.getpid():
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        deadline = datetime.now() + timedelta(seconds=5)
        while datetime.now() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            threading.Event().wait(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def recent_files(self, camera_id: str, limit: int = 20) -> list[str]:
        camera_dir = self.recordings_dir / camera_id
        if not camera_dir.exists():
            return []
        files = sorted(
            camera_dir.glob("????-??-??/??/*.mp4"),
            key=lambda path: self.recording_start_epoch(path) or path.stat().st_mtime,
        )
        return [str(path) for path in files[-limit:]][::-1]

    def recording_rows(self, camera_id: str, limit: int = 1000) -> list[dict]:
        files = [Path(path) for path in self.recent_files(camera_id, limit=limit)]
        files.sort(key=lambda path: self.recording_start_epoch(path) or path.stat().st_mtime)
        rows: list[dict] = []
        for index, file_path in enumerate(files):
            start_epoch = self.recording_start_epoch(file_path)
            next_start_epoch = (
                self.recording_start_epoch(files[index + 1]) if index + 1 < len(files) else None
            )
            duration_seconds = 300.0
            if start_epoch is not None and next_start_epoch is not None:
                duration_seconds = max(1.0, min(300.0, next_start_epoch - start_epoch))
            rows.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": file_path.stat().st_mtime,
                    "start_epoch": start_epoch,
                    "start_at": (
                        datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
                        if start_epoch is not None
                        else ""
                    ),
                    "duration_seconds": duration_seconds,
                    "end_epoch": start_epoch + duration_seconds if start_epoch is not None else None,
                }
            )
        return rows

    def recording_at(self, camera_id: str, epoch: float, limit: int = 1000) -> dict | None:
        rows = self.recording_rows(camera_id, limit=limit)
        for row in rows:
            start_epoch = row.get("start_epoch")
            end_epoch = row.get("end_epoch")
            if start_epoch is None or end_epoch is None:
                continue
            if float(start_epoch) <= epoch <= float(end_epoch):
                return row
        return None

    def recording_start_epoch(self, path: Path) -> float | None:
        try:
            value = datetime.strptime(path.stem, "%Y%m%d-%H%M%S")
        except ValueError:
            return None
        return value.astimezone(timezone.utc).timestamp()
