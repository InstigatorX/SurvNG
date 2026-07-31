from __future__ import annotations

import os
import json
import logging
import math
import signal
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .baichuan_native import (
    BaichuanFfmpegPipe,
    ffmpeg_input_args,
    ffmpeg_timestamp_repair_args,
    is_native_baichuan,
    start_ffmpeg_pipe,
)
from .config import CameraConfig, RecordingRetentionConfig
from .go2rtc import Go2RtcAdapter, Go2RtcError
from .recording_media import mp4_stream_fingerprint
from .recording_retention import RecordingRetentionService


ProcessItem = tuple[subprocess.Popen, BaichuanFfmpegPipe | None, threading.Event, threading.Thread]
RecorderKey = tuple[str, str]
LOGGER = logging.getLogger(__name__)

RECORDING_FINALIZE_GRACE_SECONDS = 2.0
DTS_WARNING_WINDOW_SECONDS = 5.0
DTS_WARNING_RESTART_COUNT = 12


@dataclass(frozen=True)
class AudioStreamInfo:
    known: bool
    codec: str = ""
    sample_rate: int = 0


class Recorder:
    def __init__(
        self,
        ffmpeg_path: str,
        storage_dir: Path,
        segment_seconds: float = 10.0,
        hardware_acceleration: str = "auto",
        index_dir: Path | None = None,
        go2rtc: Go2RtcAdapter | None = None,
        retention_config: RecordingRetentionConfig | None = None,
        protected_recording_paths: Callable[[], set[str]] | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.hardware_acceleration = hardware_acceleration
        self.go2rtc = go2rtc or Go2RtcAdapter(timeout=1.5, cache_seconds=120.0)
        self.segment_seconds = max(2.0, min(300.0, float(segment_seconds or 10.0)))
        self.storage_dir = storage_dir
        self.recordings_dir = storage_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        resolved_index_dir = index_dir or storage_dir
        resolved_index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = resolved_index_dir / "recordings.sqlite3"
        self.processes: dict[RecorderKey, ProcessItem] = {}
        self._starting: set[RecorderKey] = set()
        self._retry_after: dict[RecorderKey, float] = {}
        self._audio_stream_cache: dict[tuple[str, str, str], AudioStreamInfo] = {}
        self._audio_probe_unavailable_hosts: set[str] = set()
        self._disabled_cameras: set[str] = set()
        self.owner_token = f"survng-{os.getpid()}"
        self._lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._index_stop = threading.Event()
        self._index_wake = threading.Event()
        self._index_thread: threading.Thread | None = None
        self._index_maintenance_thread: threading.Thread | None = None
        self._edge_refresh_lock = threading.Lock()
        self._edge_refresh_requests: dict[RecorderKey, float] = {}
        self._prune_cursor = 0
        self._fingerprint_pending: deque[str] = deque()
        self._fingerprint_pending_set: set[str] = set()
        self._fingerprint_lock = threading.Lock()
        self._validation_pending: deque[str] = deque()
        self._validation_pending_set: set[str] = set()
        self._validation_lock = threading.Lock()
        self._external_protected_recording_paths = protected_recording_paths or set
        self._playback_lease_lock = threading.Lock()
        self._playback_leases: dict[str, float] = {}
        self._init_recording_index()
        self.retention = RecordingRetentionService(
            self.storage_dir,
            self.recordings_dir,
            self._index_connection,
            retention_config or RecordingRetentionConfig(),
            self._protected_recording_paths,
        )

    def _index_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.index_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_recording_index(self) -> None:
        with self._index_connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS recordings (
                    path TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_at REAL NOT NULL,
                    start_epoch REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    end_epoch REAL NOT NULL,
                    playable INTEGER NOT NULL DEFAULT 1,
                    health_error TEXT NOT NULL DEFAULT '',
                    validated INTEGER NOT NULL DEFAULT 0,
                    stream_fingerprint TEXT NOT NULL DEFAULT '',
                    fingerprint_checked INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS recordings_range ON recordings(camera_id, source, start_epoch, end_epoch)"
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(recordings)")}
            if "playable" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN playable INTEGER NOT NULL DEFAULT 1")
            if "health_error" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN health_error TEXT NOT NULL DEFAULT ''")
            if "validated" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN validated INTEGER NOT NULL DEFAULT 0")
            if "stream_fingerprint" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN stream_fingerprint TEXT NOT NULL DEFAULT ''")
            if "fingerprint_checked" not in columns:
                connection.execute("ALTER TABLE recordings ADD COLUMN fingerprint_checked INTEGER NOT NULL DEFAULT 0")
                connection.execute(
                    """
                    UPDATE recordings
                    SET fingerprint_checked = 1
                    WHERE stream_fingerprint != ''
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS recordings_fingerprint_pending
                ON recordings(start_epoch DESC)
                WHERE fingerprint_checked = 0
                """
            )

    def _rebase_recording_index_paths(self) -> None:
        """Rebase absolute media paths when the same index is used under a new mount."""
        recordings_root = self.recordings_dir.resolve()
        with self._index_connection() as connection:
            indexed_paths = {
                str(row["path"])
                for row in connection.execute("SELECT path FROM recordings")
            }
            updates: list[tuple[str, str]] = []
            duplicates: list[tuple[str]] = []
            for old_path in tuple(indexed_paths):
                path = Path(old_path)
                try:
                    marker_index = path.parts.index("recordings")
                except ValueError:
                    continue
                relative_parts = path.parts[marker_index + 1:]
                if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
                    continue
                candidate = recordings_root.joinpath(*relative_parts)
                new_path = str(candidate)
                if new_path == old_path:
                    continue
                if new_path in indexed_paths:
                    duplicates.append((old_path,))
                else:
                    updates.append((new_path, old_path))
                    indexed_paths.add(new_path)
            if duplicates:
                connection.executemany(
                    "DELETE FROM recordings WHERE path = ?",
                    duplicates,
                )
            if updates:
                connection.executemany(
                    "UPDATE recordings SET path = ? WHERE path = ?",
                    updates,
                )
        changed = len(updates) + len(duplicates)
        if changed:
            LOGGER.info(
                "Rebased %d recording index paths for storage root %s",
                changed,
                self.storage_dir,
            )

    def start(self, camera: CameraConfig, source: str = "main") -> None:
        source = camera.normalized_source(source, default="main")
        key = (camera.id, source)
        with self._lock:
            if camera.id in self._disabled_cameras:
                return
            if time.monotonic() < self._retry_after.get(key, 0):
                return
            existing = self.processes.get(key)
            if existing is not None and existing[0].poll() is None:
                return
            if key in self._starting:
                return
            if existing is not None:
                self.processes.pop(key, None)
            self._starting.add(key)

        camera_dir = self._camera_dir(camera.id, source)
        camera_dir.mkdir(parents=True, exist_ok=True)
        stop_event = threading.Event()
        keeper: threading.Thread | None = None
        process: subprocess.Popen | None = None
        pipe: BaichuanFfmpegPipe | None = None
        try:
            self._ensure_recording_dirs(camera_dir)
            keeper = threading.Thread(target=self._keep_recording_dirs, args=(camera_dir, stop_event), daemon=True)
            keeper.start()
            output = str(camera_dir / "%Y-%m-%d" / "%H" / "%Y%m%d-%H%M%S%z.mp4")
            audio_args = self._audio_output_args(camera, source)
            command = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                *ffmpeg_input_args(camera, source),
                "-map",
                "0:v:0",
                *audio_args,
                "-metadata",
                f"survng_owner={self.owner_token}",
                "-metadata",
                f"survng_camera={camera.id}",
                "-metadata",
                f"survng_source={source}",
                "-c:v",
                "copy",
                *ffmpeg_timestamp_repair_args(camera),
                "-f",
                "segment",
                "-segment_time",
                f"{self.segment_seconds:g}",
                "-strftime",
                "1",
                "-reset_timestamps",
                "1",
                "-segment_format",
                "mp4",
                output,
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if is_native_baichuan(camera) else None,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            threading.Thread(
                target=self._monitor_ffmpeg_stderr,
                args=(key, process),
                name=f"recorder-stderr-{camera.id}-{source}",
                daemon=True,
            ).start()
            pipe = start_ffmpeg_pipe(camera, source, process)
            with self._lock:
                keep_process = key in self._starting and camera.id not in self._disabled_cameras
                if keep_process:
                    self.processes[key] = (process, pipe, stop_event, keeper)
                    self._retry_after.pop(key, None)
            if not keep_process:
                stop_event.set()
                self._stop_pipe(pipe)
                if process.poll() is None:
                    self._kill_pid(process.pid)
                keeper.join(timeout=1)
                return
        except Exception as error:
            stop_event.set()
            self._stop_pipe(pipe)
            if process is not None and process.poll() is None:
                self._kill_pid(process.pid)
            if keeper is not None:
                keeper.join(timeout=1)
            with self._lock:
                self._retry_after[key] = time.monotonic() + 60
            LOGGER.error("Recorder start failed for %s/%s: %s", camera.id, source, error)
            return
        finally:
            with self._lock:
                self._starting.discard(key)
        self.cleanup_duplicate_recorders({key})

    def _audio_output_args(self, camera: CameraConfig, source: str) -> list[str]:
        source_url = camera.source_url(source)
        cache_key = (camera.id, source, source_url)
        with self._lock:
            info = self._audio_stream_cache.get(cache_key)
        if info is None:
            info = self._probe_audio_stream(camera, source)
            if info.known:
                with self._lock:
                    self._audio_stream_cache[cache_key] = info
        if info.known and not info.codec:
            return []
        if (info.known and info.codec == "aac") or (
            not info.known and not is_native_baichuan(camera)
        ):
            return ["-map", "0:a:0?", "-c:a", "copy"]
        bitrate_kbps = 64
        if info.sample_rate > 0:
            bitrate_kbps = max(24, min(64, info.sample_rate * 6 // 1000))
        return ["-map", "0:a:0?", "-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]

    def _probe_audio_stream(self, camera: CameraConfig, source: str) -> AudioStreamInfo:
        if is_native_baichuan(camera):
            return AudioStreamInfo(known=False)
        probe_host = ""
        try:
            stream_ref = self.go2rtc.stream(camera, source)
            probe_host = stream_ref.host
            with self._lock:
                if probe_host in self._audio_probe_unavailable_hosts:
                    return AudioStreamInfo(known=False)
            stream: dict = {}
            for attempt in range(2):
                try:
                    stream = self.go2rtc.audio_stream_info(camera, source)
                    if stream.get("available"):
                        break
                except Go2RtcError:
                    if attempt:
                        raise
                time.sleep(0.1)
            if not stream.get("available"):
                return AudioStreamInfo(known=False)
            return AudioStreamInfo(
                known=True,
                codec=str(stream.get("codec") or "").strip().lower(),
                sample_rate=int(stream.get("sample_rate") or 0),
            )
        except Go2RtcError as error:
            if probe_host:
                with self._lock:
                    self._audio_probe_unavailable_hosts.add(probe_host)
                LOGGER.warning(
                    "Recorder audio metadata unavailable from go2rtc host %s: %s",
                    probe_host,
                    error,
                )
            return AudioStreamInfo(known=False)
        except (ValueError, TypeError):
            return AudioStreamInfo(known=False)

    def _monitor_ffmpeg_stderr(self, key: RecorderKey, process: subprocess.Popen) -> None:
        if process.stderr is None:
            return
        dts_warnings: deque[float] = deque()
        for raw_line in process.stderr:
            line = (
                raw_line.decode("utf-8", errors="replace").strip()
                if isinstance(raw_line, bytes)
                else raw_line.strip()
            )
            if not line:
                continue
            if "Non-monotonic DTS" in line:
                now = time.monotonic()
                dts_warnings.append(now)
                while dts_warnings and now - dts_warnings[0] > DTS_WARNING_WINDOW_SECONDS:
                    dts_warnings.popleft()
                if len(dts_warnings) >= DTS_WARNING_RESTART_COUNT:
                    LOGGER.warning(
                        "Recorder %s/%s detected persistent non-monotonic DTS; restarting FFmpeg",
                        key[0],
                        key[1],
                    )
                    self._kill_pid(process.pid)
                    return
                continue
            LOGGER.warning("Recorder FFmpeg %s/%s: %s", key[0], key[1], line)

    def _camera_dir(self, camera_id: str, source: str = "main") -> Path:
        source = "main" if source == "main" else "live"
        return self.recordings_dir / camera_id / source

    def _recording_search_dirs(self, camera_id: str, source: str) -> list[Path]:
        normalized = "main" if source == "main" else "live"
        directories = [self._camera_dir(camera_id, normalized)]
        if normalized == "main":
            directories.append(self.recordings_dir / camera_id)
        return directories

    def _keep_recording_dirs(self, camera_dir: Path, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._ensure_recording_dirs(camera_dir)
            except OSError as error:
                LOGGER.warning("Recording directory maintenance failed for %s: %s", camera_dir, error)
            stop_event.wait(60)

    def _ensure_recording_dirs(self, camera_dir: Path) -> None:
        now = datetime.now()
        for hours_ahead in range(3):
            target = now + timedelta(hours=hours_ahead)
            (camera_dir / target.strftime("%Y-%m-%d") / target.strftime("%H")).mkdir(parents=True, exist_ok=True)

    def stop(self, camera_id: str, source: str | None = None) -> None:
        sources = ("main", "live") if source is None else (source,)
        stopped_keys: set[RecorderKey] = set()
        for raw_source in sources:
            normalized = "main" if raw_source == "main" else "live"
            key = (camera_id, normalized)
            stopped_keys.add(key)
            with self._lock:
                item = self.processes.pop(key, None)
                self._starting.discard(key)
            if item is not None:
                self._stop_item(item)
        for pids in self._owned_ffmpeg_recorders(stopped_keys).values():
            for pid in pids:
                self._kill_pid(pid)

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        with self._lock:
            if enabled:
                self._disabled_cameras.discard(camera_id)
            else:
                self._disabled_cameras.add(camera_id)
        if not enabled:
            self.stop(camera_id)

    def camera_enabled(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id not in self._disabled_cameras

    def _stop_item(self, item: ProcessItem) -> None:
        process, pipe, stop_event, keeper = item
        stop_event.set()
        self._stop_pipe(pipe)
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        keeper.join(timeout=1)

    @staticmethod
    def _stop_pipe(pipe: BaichuanFfmpegPipe | None) -> None:
        if pipe is None:
            return
        try:
            pipe.stop()
        except Exception:
            # Recorder teardown must still terminate and reap FFmpeg even if a
            # native camera feeder reports its own shutdown failure.
            LOGGER.exception("Recorder input pipe shutdown failed")

    def status(self, keys: set[RecorderKey] | None = None) -> dict[RecorderKey, bool]:
        stopped_items: list[ProcessItem] = []
        with self._lock:
            stopped = [
                key
                for key, (process, _pipe, _stop_event, _keeper) in self.processes.items()
                if process.poll() is not None
            ]
            for key in stopped:
                item = self.processes.pop(key, None)
                if item is not None:
                    stopped_items.append(item)
            tracked = {key: True for key in self.processes}

        # Pipe shutdown and keeper joins may block. Never hold the recorder
        # state lock while cleaning up a process that has already exited.
        for _process, pipe, stop_event, keeper in stopped_items:
            stop_event.set()
            self._stop_pipe(pipe)
            keeper.join(timeout=1)

        if keys:
            for key, pids in self._owned_ffmpeg_recorders(keys).items():
                if pids:
                    tracked[key] = True
        return tracked

    def stop_all(self) -> None:
        self.retention.stop()
        self._index_stop.set()
        self._index_wake.set()
        for thread_name in ("_index_thread", "_index_maintenance_thread"):
            thread = getattr(self, thread_name)
            if thread is not None:
                thread.join(timeout=10)
                if thread.is_alive():
                    LOGGER.error("Recorder worker %s did not stop", thread.name)
                else:
                    setattr(self, thread_name, None)
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2)
            if self._watchdog_thread.is_alive():
                LOGGER.error("Recorder watchdog did not stop")
            else:
                self._watchdog_thread = None
        with self._lock:
            stopped_keys = set(self.processes)
            items = list(self.processes.values())
            self.processes.clear()
            self._starting.clear()
        shutdowns = [
            threading.Thread(
                target=self._stop_item,
                args=(item,),
                name=f"stop-recorder-{item[0].pid}",
                daemon=False,
            )
            for item in items
        ]
        for thread in shutdowns:
            thread.start()
        for thread in shutdowns:
            thread.join()
        for pids in self._owned_ffmpeg_recorders(stopped_keys).values():
            for pid in pids:
                self._kill_pid(pid)

    def start_watchdog(self, cameras: list[CameraConfig], interval: float = 15.0) -> None:
        self._watchdog_stop.clear()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        camera_map = {camera.id: camera for camera in cameras if camera.record or camera.record_sub}
        self._watchdog_thread = threading.Thread(target=self._watchdog, args=(camera_map, interval), daemon=True)
        self._watchdog_thread.start()

    def _wanted_keys(self, camera_map: dict[str, CameraConfig]) -> dict[RecorderKey, CameraConfig]:
        wanted: dict[RecorderKey, CameraConfig] = {}
        with self._lock:
            disabled_cameras = set(self._disabled_cameras)
        for camera_id, camera in camera_map.items():
            if camera_id in disabled_cameras:
                continue
            if camera.record:
                wanted[(camera_id, "main")] = camera
            if camera.record_sub and camera.live_stream_url:
                wanted[(camera_id, "live")] = camera
        return wanted

    def _watchdog(self, camera_map: dict[str, CameraConfig], interval: float) -> None:
        while not self._watchdog_stop.wait(interval):
            try:
                self.reconcile(camera_map)
            except Exception:
                LOGGER.exception("Recorder watchdog reconciliation failed")

    def reconcile(self, camera_map: dict[str, CameraConfig]) -> None:
        wanted = self._wanted_keys(camera_map)
        with self._lock:
            active_items = dict(self.processes)
        for key, item in active_items.items():
            process = item[0]
            if key not in wanted:
                self.stop(key[0], key[1])
            elif process.poll() is not None:
                self.stop(key[0], key[1])
                self.start(wanted[key], key[1])
        with self._lock:
            active_keys = set(self.processes)
        for key, camera in wanted.items():
            if key not in active_keys:
                self.start(camera, key[1])
        self.cleanup_duplicate_recorders(set(wanted))

    def cleanup_duplicate_recorders(self, keys: set[RecorderKey]) -> None:
        owned = self._owned_ffmpeg_recorders(keys)
        with self._lock:
            tracked_pids = {item[0].pid for item in self.processes.values()}
        for key, pids in owned.items():
            extras = [pid for pid in pids if pid not in tracked_pids]
            if len([pid for pid in pids if pid in tracked_pids]) > 1:
                extras.extend(sorted(pid for pid in pids if pid in tracked_pids)[1:])
            for pid in sorted(set(extras)):
                self._kill_pid(pid)

    def cleanup_stale_recorders(self, keys: set[RecorderKey]) -> None:
        for pids in self._owned_ffmpeg_recorders(keys).values():
            for pid in pids:
                self._kill_pid(pid)

    def _owned_ffmpeg_recorders(self, keys: set[RecorderKey]) -> dict[RecorderKey, list[int]]:
        result: dict[RecorderKey, list[int]] = {key: [] for key in keys}
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
            for camera_id, source in keys:
                source_dir = f"/recordings/{camera_id}/{source}/"
                legacy_main_dir = f"/recordings/{camera_id}/" if source == "main" else ""
                if source_dir in command or (legacy_main_dir and legacy_main_dir in command and "/main/" not in command and "/live/" not in command):
                    result.setdefault((camera_id, source), []).append(pid)
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
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
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

    def recent_files(self, camera_id: str, limit: int = 20, source: str = "main") -> list[str]:
        source = "main" if source == "main" else "live"
        search_dirs = self._recording_search_dirs(camera_id, source)
        files = []
        for camera_dir in search_dirs:
            if camera_dir.exists():
                files.extend(camera_dir.glob("????-??-??/??/*.mp4"))
        files = sorted(
            set(files),
            key=lambda path: self.recording_start_epoch(path) or path.stat().st_mtime,
        )
        return [str(path) for path in files[-limit:]][::-1]

    def recording_rows(self, camera_id: str, limit: int = 1000, source: str = "main") -> list[dict]:
        """Return recent playable recordings from the local index only."""
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            rows = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds,
                       end_epoch, source, playable, health_error, validated,
                       stream_fingerprint, fingerprint_checked
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                ORDER BY start_epoch DESC LIMIT ?
                """,
                (camera_id, source, max(1, int(limit))),
            ).fetchall()
        payloads = [dict(row) for row in reversed(rows)]
        for row in payloads:
            row["start_at"] = datetime.fromtimestamp(
                float(row["start_epoch"]), timezone.utc
            ).isoformat()
        return payloads

    def recording_rows_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        *,
        discover_missing: bool = True,
    ) -> list[dict]:
        source = "main" if source == "main" else "live"
        def indexed_rows() -> list[dict]:
            with self._index_connection() as connection:
                indexed = connection.execute(
                    """
                    SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds, end_epoch, source,
                           playable, health_error, stream_fingerprint, fingerprint_checked
                    FROM recordings
                    WHERE camera_id = ? AND source = ? AND playable = 1 AND end_epoch > ? AND start_epoch < ?
                    ORDER BY start_epoch
                    """,
                    (camera_id, source, start_epoch, end_epoch),
                ).fetchall()
            return [dict(row) for row in indexed]

        rows = indexed_rows()
        if not discover_missing:
            for row in rows:
                row["start_at"] = datetime.fromtimestamp(float(row["start_epoch"]), timezone.utc).isoformat()
            return rows

        stale_paths = {str(row["path"]) for row in rows if not Path(str(row["path"])).is_file()}
        if stale_paths:
            self._delete_index_paths(list(stale_paths))
            rows = [row for row in rows if str(row["path"]) not in stale_paths]

        start_date = datetime.fromtimestamp(start_epoch).date() - timedelta(days=1)
        end_date = datetime.fromtimestamp(end_epoch).date() + timedelta(days=1)
        files: list[Path] = []
        current_date = start_date
        while current_date <= end_date:
            for camera_dir in self._recording_search_dirs(camera_id, source):
                day_dir = camera_dir / current_date.isoformat()
                if day_dir.exists():
                    files.extend(day_dir.glob("??/*.mp4"))
            current_date += timedelta(days=1)
        with self._index_connection() as connection:
            indexed_paths = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT path FROM recordings
                    WHERE camera_id = ? AND source = ? AND end_epoch > ? AND start_epoch < ?
                    """,
                    (camera_id, source, start_epoch, end_epoch),
                )
            }
        relevant_files = [
            path
            for path in files
            if (file_start := self.recording_start_epoch(path)) is not None
            and file_start < end_epoch
            and file_start + self.segment_seconds > start_epoch
        ]
        with self._lock:
            active_item = self.processes.get((camera_id, source))
            recorder_active = active_item is not None and active_item[0].poll() is None
        if recorder_active and relevant_files:
            now = datetime.now()
            current_hour = self._camera_dir(camera_id, source) / now.strftime("%Y-%m-%d") / now.strftime("%H")
            active_tail = max(relevant_files, key=lambda path: self.recording_start_epoch(path) or 0.0)
            if active_tail.parent == current_hour:
                relevant_files.remove(active_tail)
        if any(str(path) not in indexed_paths for path in relevant_files):
            discovered = [
                row for row in self._recording_rows_for_files(camera_id, source, files)
                if row.get("start_epoch") is not None
                and row.get("end_epoch") is not None
                and float(row["end_epoch"]) > start_epoch
                and float(row["start_epoch"]) < end_epoch
            ]
            self._store_recording_rows(camera_id, source, discovered)
            self.queue_recording_validation(discovered)
            rows = indexed_rows()
        for row in rows:
            row["start_at"] = datetime.fromtimestamp(float(row["start_epoch"]), timezone.utc).isoformat()
        return rows

    def lease_recordings_for_playback(
        self,
        rows: list[dict],
        *,
        ttl_seconds: float = 90.0,
    ) -> None:
        """Keep manifest segments out of retention while clients fetch them."""
        expires_at = time.monotonic() + max(10.0, float(ttl_seconds))
        recordings_root = self.recordings_dir.resolve()
        leased: list[str] = []
        for row in rows:
            raw_path = str(row.get("path") or "")
            if not raw_path:
                continue
            resolved = Path(raw_path).resolve(strict=False)
            try:
                resolved.relative_to(recordings_root)
            except ValueError:
                continue
            leased.append(str(resolved))
        if not leased:
            return
        with self._playback_lease_lock:
            self._discard_expired_playback_leases_locked()
            for path in leased:
                self._playback_leases[path] = max(
                    expires_at,
                    self._playback_leases.get(path, 0.0),
                )

    def discard_missing_recording_rows(self, rows: list[dict]) -> list[dict]:
        """Remove missing indexed files before a playback manifest advertises them."""
        existing: list[dict] = []
        stale_paths: list[str] = []
        for row in rows:
            path = str(row.get("path") or "")
            if path and Path(path).is_file():
                existing.append(row)
            elif path:
                stale_paths.append(path)
        self._delete_index_paths(stale_paths)
        return existing

    def _protected_recording_paths(self) -> set[str]:
        protected = set(self._external_protected_recording_paths())
        with self._playback_lease_lock:
            self._discard_expired_playback_leases_locked()
            protected.update(self._playback_leases)
        return protected

    def _discard_expired_playback_leases_locked(self) -> None:
        now = time.monotonic()
        expired = [
            path for path, expires_at in self._playback_leases.items()
            if expires_at <= now
        ]
        for path in expired:
            self._playback_leases.pop(path, None)

    def recording_availability_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        *,
        discover_missing: bool = True,
    ) -> dict:
        source = "main" if source == "main" else "live"
        rows = [
            {"start_epoch": row["start_epoch"], "end_epoch": row["end_epoch"]}
            for row in self.recording_rows_between(
                camera_id,
                start_epoch,
                end_epoch,
                source,
                discover_missing=discover_missing,
            )
            if int(row.get("size_bytes") or 0) > 1024
        ]
        return {
            "ranges": self._merge_availability_rows(
                rows,
                start_epoch,
                end_epoch,
                gap_tolerance=min(5.0, max(0.25, self.segment_seconds / 2)),
            ),
            "segment_count": len(rows),
        }

    def refresh_recording_edge(
        self,
        camera_id: str,
        source: str,
        after_epoch: float,
    ) -> int:
        """Index completed segments near a live playback edge without probing them."""
        source = "main" if source == "main" else "live"
        cutoff = after_epoch - max(5.0, self.segment_seconds * 2)
        camera_dir = self._camera_dir(camera_id, source)
        files: list[Path] = []
        now = datetime.now()
        for hours_back in (1, 0):
            target = now - timedelta(hours=hours_back)
            hour_dir = camera_dir / target.strftime("%Y-%m-%d") / target.strftime("%H")
            if hour_dir.exists():
                files.extend(
                    path for path in hour_dir.glob("*.mp4")
                    if (self.recording_start_epoch(path) or 0.0) >= cutoff
                )
        rows = self._recording_rows_for_files(camera_id, source, files)
        self._store_recording_rows(camera_id, source, rows)
        return len(rows)

    @staticmethod
    def _merge_availability_rows(
        rows: list[dict],
        start_epoch: float,
        end_epoch: float,
        gap_tolerance: float = 0.25,
    ) -> list[dict]:
        ranges: list[dict] = []
        for row in rows:
            try:
                row_start = max(start_epoch, float(row["start_epoch"]))
                row_end = min(end_epoch, float(row["end_epoch"]))
            except (KeyError, TypeError, ValueError):
                continue
            if row_end <= row_start:
                continue
            if ranges and row_start <= float(ranges[-1]["end_epoch"]) + gap_tolerance:
                current = ranges[-1]
                current["end_epoch"] = max(float(current["end_epoch"]), row_end)
                current["duration_seconds"] = float(current["end_epoch"]) - float(current["start_epoch"])
                current["segment_count"] = int(current["segment_count"]) + 1
                continue
            ranges.append({
                "start_epoch": row_start,
                "end_epoch": row_end,
                "duration_seconds": row_end - row_start,
                "segment_count": 1,
            })
        return ranges

    def _recording_rows_for_files(self, camera_id: str, source: str, files: list[Path]) -> list[dict]:
        files = list({path for path in files if path.is_file()})
        def file_order(path: Path) -> float:
            start_epoch = self.recording_start_epoch(path)
            if start_epoch is not None:
                return start_epoch
            try:
                return path.stat().st_mtime
            except OSError:
                return float("inf")

        files.sort(key=file_order)
        all_files = files
        with self._lock:
            item = self.processes.get((camera_id, source))
            recorder_active = item is not None and item[0].poll() is None
        if recorder_active and files and self._recording_file_may_be_active(files[-1]):
            files = files[:-1]
        rows: list[dict] = []
        for index, file_path in enumerate(files):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            start_epoch = self.recording_start_epoch(file_path)
            if start_epoch is None:
                continue
            next_start_epoch = self.recording_start_epoch(all_files[index + 1]) if index + 1 < len(all_files) else None
            duration_seconds = self.segment_seconds
            if start_epoch is not None and next_start_epoch is not None:
                duration_seconds = max(1.0, min(self.segment_seconds, next_start_epoch - start_epoch))
            rows.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "start_epoch": start_epoch,
                    "start_at": (
                        datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
                        if start_epoch is not None
                        else ""
                    ),
                    "duration_seconds": duration_seconds,
                    "end_epoch": start_epoch + duration_seconds,
                    "source": source,
                }
            )
        return rows

    def _recording_file_may_be_active(self, path: Path, now_epoch: float | None = None) -> bool:
        now_epoch = time.time() if now_epoch is None else now_epoch
        start_epoch = self.recording_start_epoch(path)
        if start_epoch is not None:
            return now_epoch < start_epoch + self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        try:
            return now_epoch - path.stat().st_mtime < self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        except OSError:
            return False

    def _recording_file_is_stable(self, path: Path, now_epoch: float | None = None) -> bool:
        """Return whether FFmpeg has had enough time to finalize a segment."""
        now_epoch = time.time() if now_epoch is None else now_epoch
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            return False
        start_epoch = self.recording_start_epoch(path)
        if (
            start_epoch is not None
            and now_epoch < start_epoch + self.segment_seconds + RECORDING_FINALIZE_GRACE_SECONDS
        ):
            return False
        return now_epoch - modified_at >= RECORDING_FINALIZE_GRACE_SECONDS

    def _store_recording_rows(self, camera_id: str, source: str, rows: list[dict], validate_new: bool = False) -> None:
        if not rows:
            return
        if validate_new:
            with self._index_connection() as connection:
                existing = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT path FROM recordings WHERE path IN ({})".format(",".join("?" for _ in rows)),
                        [row["path"] for row in rows],
                    )
                }
            validated_rows: list[dict] = []
            for row in rows:
                if row["path"] not in existing:
                    duration, error = self._probe_recording(Path(row["path"]))
                    if error.startswith("probe failed") or error.startswith("probe metadata"):
                        continue
                    row["playable"] = not error
                    row["health_error"] = error
                    row["validated"] = True
                    row["stream_fingerprint"] = mp4_stream_fingerprint(Path(row["path"]))
                    row["fingerprint_checked"] = True
                    if duration is not None and row.get("start_epoch") is not None:
                        row["duration_seconds"] = duration
                        row["end_epoch"] = float(row["start_epoch"]) + duration
                validated_rows.append(row)
            rows = validated_rows
            if not rows:
                return
        values = [
            (
                row["path"], camera_id, source, row["name"], row["size_bytes"], row["modified_at"],
                row["start_epoch"], row["duration_seconds"], row["end_epoch"],
                1 if row.get("playable", True) else 0, str(row.get("health_error") or ""),
                1 if row.get("validated", False) else 0,
                str(row.get("stream_fingerprint") or ""),
                1 if row.get("fingerprint_checked", False) else 0,
            )
            for row in rows
        ]
        with self._index_connection() as connection:
            connection.executemany(
                """
                INSERT INTO recordings(path, camera_id, source, name, size_bytes, modified_at, start_epoch,
                                       duration_seconds, end_epoch, playable, health_error, validated,
                                       stream_fingerprint, fingerprint_checked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    modified_at=excluded.modified_at,
                    duration_seconds=excluded.duration_seconds,
                    end_epoch=excluded.end_epoch,
                    stream_fingerprint=CASE
                        WHEN excluded.fingerprint_checked = 1 THEN excluded.stream_fingerprint
                        ELSE recordings.stream_fingerprint
                    END,
                    fingerprint_checked=MAX(recordings.fingerprint_checked, excluded.fingerprint_checked)
                """,
                values,
            )

    def start_indexer(self, cameras: list[CameraConfig]) -> None:
        self._rebase_recording_index_paths()
        camera_map = {camera.id: camera for camera in cameras if camera.record or camera.record_sub}
        self._index_stop.clear()
        self._index_wake.clear()
        self.retention.start(cameras)
        if self._index_thread is None or not self._index_thread.is_alive():
            self._index_thread = threading.Thread(
                target=self._recording_index_loop,
                args=(camera_map,),
                name="recording-indexer",
                daemon=True,
            )
            self._index_thread.start()
        if (
            self._index_maintenance_thread is None
            or not self._index_maintenance_thread.is_alive()
        ):
            self._index_maintenance_thread = threading.Thread(
                target=self._recording_index_maintenance_loop,
                args=(camera_map,),
                name="recording-index-maintenance",
                daemon=True,
            )
            self._index_maintenance_thread.start()

    def retention_status(self) -> dict[str, object]:
        return self.retention.status()

    def request_retention_run(self, *, apply: bool = False) -> dict[str, object]:
        return self.retention.request_run(apply=apply)

    def reconfigure_retention(
        self,
        config: RecordingRetentionConfig,
        cameras: list[CameraConfig],
    ) -> None:
        self.retention.reconfigure(config, cameras)

    def reconfigure_runtime(
        self,
        *,
        ffmpeg_path: str,
        hardware_acceleration: str,
        segment_seconds: float,
    ) -> None:
        """Update recorder process settings after active recorders are stopped."""
        with self._lock:
            if self.processes or self._starting:
                raise RuntimeError("active recorders must be stopped before reconfiguration")
            self.ffmpeg_path = ffmpeg_path
            self.hardware_acceleration = hardware_acceleration
            self.segment_seconds = max(2.0, min(300.0, float(segment_seconds or 10.0)))
            self._retry_after.clear()
            self._audio_stream_cache.clear()
            self._audio_probe_unavailable_hosts.clear()

    def _recording_index_loop(self, camera_map: dict[str, CameraConfig]) -> None:
        try:
            self.refresh_recording_index(camera_map, full=False, run_maintenance=False)
        except Exception:
            LOGGER.exception("Initial recording index discovery failed")
        next_discovery = time.monotonic() + 10.0
        while not self._index_stop.is_set():
            self._index_wake.wait(max(0.0, next_discovery - time.monotonic()))
            self._index_wake.clear()
            if self._index_stop.is_set():
                return
            for (camera_id, source), after_epoch in self._take_recording_edge_refreshes().items():
                try:
                    self.refresh_recording_edge(camera_id, source, after_epoch)
                except Exception:
                    LOGGER.exception(
                        "Near-live recording index discovery failed for %s/%s",
                        camera_id,
                        source,
                    )
            if time.monotonic() < next_discovery:
                continue
            try:
                self.refresh_recording_index(camera_map, full=False, run_maintenance=False)
            except Exception:
                LOGGER.exception("Recording index discovery failed")
            finally:
                next_discovery = time.monotonic() + 10.0

    def _recording_index_maintenance_loop(self, camera_map: dict[str, CameraConfig]) -> None:
        if self._index_stop.wait(30):
            return
        while not self._index_stop.is_set():
            try:
                self._validate_index_batch(limit=20)
                self._backfill_stream_fingerprints(limit=20)
            except Exception:
                LOGGER.exception("Recording index maintenance failed")
            if self._index_stop.wait(5):
                return

    def _reconcile_recording_source(self, camera_id: str, source: str) -> None:
        files = [
            path
            for camera_dir in self._recording_search_dirs(camera_id, source)
            if camera_dir.exists()
            for path in camera_dir.glob("????-??-??/??/*.mp4")
        ]
        rows = self._recording_rows_for_files(camera_id, source, files)
        self._store_recording_rows(camera_id, source, rows)
        self._prune_recording_index(camera_id, source, files)

    def storage_index_health(
        self,
        *,
        full: bool = False,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
        quick_index_limit: int = 1000,
        quick_hours: int = 6,
    ) -> dict[str, object]:
        """Compare recording files with the index using bounded or exhaustive discovery."""
        if full:
            # Freeze the index side before walking remote storage. A full NFS walk
            # can take hours while the live index keeps receiving new segments.
            # Reading it afterward would misclassify those new rows as files that
            # disappeared merely because their directory was visited earlier.
            with self._index_connection() as connection:
                indexed_total = int(connection.execute("SELECT count(*) FROM recordings").fetchone()[0])
                indexed_rows = connection.execute(
                    "SELECT path, camera_id, source FROM recordings"
                ).fetchall()
            files_by_source = self._recording_files_by_source(cancel_event=cancel_event, progress=progress)
        else:
            files_by_source = self._recent_recording_files_by_source(
                hours=quick_hours,
                cancel_event=cancel_event,
                progress=progress,
            )
            with self._index_connection() as connection:
                indexed_total = int(connection.execute("SELECT count(*) FROM recordings").fetchone()[0])
                indexed_rows = connection.execute(
                    "SELECT path, camera_id, source FROM recordings ORDER BY start_epoch DESC LIMIT ?",
                    (max(1, quick_index_limit),),
                ).fetchall()
        all_disk_paths = {
            str(path)
            for files in files_by_source.values()
            for path in files
        }
        stable_disk_paths = {
            str(path)
            for files in files_by_source.values()
            for path in files
            if not self._recording_file_may_be_active(path)
        }
        indexed_sample = {str(row["path"]) for row in indexed_rows}
        if full:
            stale = sorted(indexed_sample - all_disk_paths)
            discovery_candidates = stable_disk_paths - indexed_sample
            indexed_for_discovery = indexed_sample | self.indexed_path_subset(discovery_candidates)
        else:
            stale = []
            for index, path in enumerate(indexed_sample, start=1):
                self._raise_if_cancelled(cancel_event)
                if not Path(path).is_file():
                    stale.append(path)
                if progress and index % 100 == 0:
                    progress("Checking indexed recordings", index, len(indexed_sample))
            indexed_for_discovery = self.indexed_path_subset(stable_disk_paths)
        unindexed = sorted(stable_disk_paths - indexed_for_discovery)
        return {
            "recording_files": len(all_disk_paths),
            "recent_recording_files": len(all_disk_paths - stable_disk_paths),
            "indexed_recordings": indexed_total,
            "index_rows_scanned": len(indexed_sample),
            "recording_hours_scanned": None if full else max(1, quick_hours),
            "scan_complete": full,
            "missing_index_files": stale,
            "unindexed_files": unindexed,
            "files_by_source": files_by_source,
        }

    def reconcile_storage_index(
        self,
        *,
        full: bool = False,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
        health: dict[str, object] | None = None,
    ) -> dict[str, int]:
        """Synchronize the local index with a recording-storage snapshot.

        Callers that already collected health may supply it so a full repair does
        not walk a large remote recording library a second time.
        """
        if health is None:
            health = self.storage_index_health(full=full, cancel_event=cancel_event, progress=progress)
        files_by_source = health.get("files_by_source", {})
        if not isinstance(files_by_source, dict):
            raise ValueError("recording health is missing its file snapshot")
        added = 0
        unindexed = set(health.get("unindexed_files", []))
        for (camera_id, source), files in files_by_source.items():
            rows = self._recording_rows_for_files(camera_id, source, files)
            if rows:
                self._store_recording_rows(camera_id, source, rows)
                added += sum(1 for row in rows if str(row["path"]) in unindexed)
        stale = list(health.get("missing_index_files", []))
        self._delete_index_paths(stale)
        return {"recordings_reindexed": added, "stale_index_rows_removed": len(stale)}

    def _recording_files_by_source(
        self,
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> dict[RecorderKey, list[Path]]:
        """Discover current and legacy recording layouts without relying on config state."""
        grouped: dict[RecorderKey, list[Path]] = {}
        if not self.recordings_dir.exists():
            return grouped
        for index, path in enumerate(self.recordings_dir.glob("*/**/*.mp4"), start=1):
            self._raise_if_cancelled(cancel_event)
            try:
                parts = path.relative_to(self.recordings_dir).parts
            except ValueError:
                continue
            if len(parts) == 5 and parts[1] in {"main", "live"}:
                camera_id, source = parts[0], parts[1]
            elif len(parts) == 4:
                camera_id, source = parts[0], "main"
            else:
                continue
            grouped.setdefault((camera_id, source), []).append(path)
            if progress and index % 1000 == 0:
                progress("Scanning all recording files", index, None)
        return grouped

    def _recent_recording_files_by_source(
        self,
        *,
        hours: int,
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> dict[RecorderKey, list[Path]]:
        grouped: dict[RecorderKey, list[Path]] = {}
        if not self.recordings_dir.exists():
            return grouped
        camera_dirs = [path for path in self.recordings_dir.iterdir() if path.is_dir()]
        targets = [datetime.now() - timedelta(hours=offset) for offset in range(max(1, hours) + 1)]
        total = max(1, len(camera_dirs) * len(targets) * 2)
        checked = 0
        for camera_dir in camera_dirs:
            self._raise_if_cancelled(cancel_event)
            for source in ("main", "live"):
                search_roots = [camera_dir / source]
                if source == "main":
                    search_roots.append(camera_dir)
                files: list[Path] = []
                for target in targets:
                    for root in search_roots:
                        self._raise_if_cancelled(cancel_event)
                        hour_dir = root / target.strftime("%Y-%m-%d") / target.strftime("%H")
                        if hour_dir.exists():
                            files.extend(hour_dir.glob("*.mp4"))
                    checked += 1
                    if progress and checked % 10 == 0:
                        progress("Scanning recent recording folders", min(checked, total), total)
                if files:
                    grouped[(camera_dir.name, source)] = list(set(files))
        return grouped

    def indexed_path_subset(self, paths: set[str]) -> set[str]:
        """Return paths present in the local index without touching media storage."""
        if not paths:
            return set()
        found: set[str] = set()
        values = list(paths)
        with self._index_connection() as connection:
            for offset in range(0, len(values), 500):
                batch = values[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                found.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT path FROM recordings WHERE path IN ({placeholders})",
                        batch,
                    )
                )
        return found

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("storage maintenance was cancelled")

    def refresh_recording_index(
        self,
        camera_map: dict[str, CameraConfig],
        full: bool = False,
        run_maintenance: bool = True,
    ) -> None:
        now = datetime.now()
        wanted = self._wanted_keys(camera_map)
        for camera_id, source in wanted:
            camera_dir = self._camera_dir(camera_id, source)
            if full:
                files = [
                    path
                    for search_dir in self._recording_search_dirs(camera_id, source)
                    if search_dir.exists()
                    for path in search_dir.glob("????-??-??/??/*.mp4")
                ]
            else:
                files = []
                for hours_back in (1, 0):
                    target = now - timedelta(hours=hours_back)
                    hour_dir = camera_dir / target.strftime("%Y-%m-%d") / target.strftime("%H")
                    if hour_dir.exists():
                        files.extend(hour_dir.glob("*.mp4"))
            rows = self._recording_rows_for_files(camera_id, source, files)
            self._store_recording_rows(camera_id, source, rows)
            if full:
                self._prune_recording_index(camera_id, source, files)
        if run_maintenance:
            self._prune_missing_index_rows()
            self._validate_index_batch()
            self._backfill_stream_fingerprints()

    def queue_stream_fingerprints(self, rows: list[dict]) -> None:
        with self._fingerprint_lock:
            for row in rows:
                if int(row.get("fingerprint_checked") or 0):
                    continue
                path = str(row.get("path") or "")
                if path and path not in self._fingerprint_pending_set:
                    self._fingerprint_pending.append(path)
                    self._fingerprint_pending_set.add(path)

    def queue_recording_validation(self, rows: list[dict]) -> None:
        candidates = [str(row.get("path") or "") for row in rows if not int(row.get("validated") or 0)]
        candidates = [path for path in candidates if path]
        if not candidates:
            return
        eligible: set[str] = set()
        with self._index_connection() as connection:
            for offset in range(0, len(candidates), 500):
                batch = candidates[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                eligible.update(
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT path FROM recordings WHERE validated = 0 AND path IN ({placeholders})",
                        batch,
                    )
                )
        with self._validation_lock:
            for path in candidates:
                if path in eligible and path not in self._validation_pending_set:
                    self._validation_pending.append(path)
                    self._validation_pending_set.add(path)

    def _backfill_stream_fingerprints(
        self,
        limit: int = 20,
        *,
        discover_unqueued: bool = False,
    ) -> int:
        batch_limit = max(1, limit)
        paths: list[str] = []
        with self._fingerprint_lock:
            while self._fingerprint_pending and len(paths) < batch_limit:
                path = self._fingerprint_pending.popleft()
                self._fingerprint_pending_set.discard(path)
                paths.append(path)
        if discover_unqueued and len(paths) < batch_limit:
            with self._index_connection() as connection:
                paths.extend(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT path FROM recordings
                        WHERE fingerprint_checked = 0
                        ORDER BY start_epoch DESC LIMIT ?
                        """,
                        (batch_limit - len(paths),),
                    )
                    if str(row[0]) not in paths
                )
        updated = 0
        for path_value in paths:
            if self._index_stop.is_set():
                break
            path = Path(path_value)
            if not path.is_file():
                self._delete_index_paths([path_value])
                continue
            fingerprint = mp4_stream_fingerprint(path)
            with self._index_connection() as connection:
                connection.execute(
                    """
                    UPDATE recordings
                    SET stream_fingerprint = ?, fingerprint_checked = 1
                    WHERE path = ?
                    """,
                    (fingerprint, path_value),
                )
            updated += 1
        return updated

    def _prune_recording_index(self, camera_id: str, source: str, files: list[Path]) -> None:
        existing_paths = {str(path) for path in files}
        with self._index_connection() as connection:
            indexed_paths = [
                str(row[0])
                for row in connection.execute(
                    "SELECT path FROM recordings WHERE camera_id = ? AND source = ?",
                    (camera_id, source),
                )
            ]
            stale = [(path,) for path in indexed_paths if path not in existing_paths]
            if stale:
                connection.executemany("DELETE FROM recordings WHERE path = ?", stale)

    def _delete_index_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        with self._index_connection() as connection:
            connection.executemany("DELETE FROM recordings WHERE path = ?", [(path,) for path in paths])

    def _prune_missing_index_rows(self, limit: int = 500) -> int:
        with self._index_connection() as connection:
            rows = connection.execute(
                "SELECT rowid, path FROM recordings WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (self._prune_cursor, max(1, limit)),
            ).fetchall()
            if not rows and self._prune_cursor:
                self._prune_cursor = 0
                rows = connection.execute(
                    "SELECT rowid, path FROM recordings ORDER BY rowid LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
        if not rows:
            return 0
        self._prune_cursor = int(rows[-1]["rowid"])
        stale_paths = [str(row["path"]) for row in rows if not Path(str(row["path"])).is_file()]
        self._delete_index_paths(stale_paths)
        return len(stale_paths)

    def _validate_index_batch(
        self,
        limit: int = 20,
        *,
        discover_unqueued: bool = False,
    ) -> int:
        batch_limit = max(1, limit)
        paths: list[str] = []
        with self._validation_lock:
            while self._validation_pending and len(paths) < batch_limit:
                path = self._validation_pending.popleft()
                self._validation_pending_set.discard(path)
                paths.append(path)
        if discover_unqueued and len(paths) < batch_limit:
            with self._index_connection() as connection:
                paths.extend(
                    str(row[0])
                    for row in connection.execute(
                        """
                        SELECT path FROM recordings
                        WHERE validated = 0
                        ORDER BY start_epoch DESC LIMIT ?
                        """,
                        (batch_limit - len(paths),),
                    )
                    if str(row[0]) not in paths
                )
        validated = 0
        for path_value in paths:
            if self._index_stop.is_set():
                break
            path = Path(path_value)
            if not path.is_file():
                self._delete_index_paths([str(path)])
                continue
            if not self._recording_file_is_stable(path):
                # Keep the row unvalidated. The maintenance loop will retry it
                # after FFmpeg has closed the file and written the MP4 trailer.
                with self._validation_lock:
                    if path_value not in self._validation_pending_set:
                        self._validation_pending.append(path_value)
                        self._validation_pending_set.add(path_value)
                continue
            duration, error = self._probe_recording(path)
            with self._index_connection() as connection:
                if duration is not None:
                    connection.execute(
                        """
                        UPDATE recordings
                        SET duration_seconds = ?, end_epoch = start_epoch + ?, playable = ?, health_error = ?,
                            validated = 1, stream_fingerprint = ?, fingerprint_checked = 1
                        WHERE path = ?
                        """,
                        (
                            duration, duration, 0 if error else 1, error,
                            mp4_stream_fingerprint(path), str(path),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE recordings SET playable = 0, health_error = ?, validated = 1 WHERE path = ?",
                        (error or "recording validation failed", str(path)),
                    )
            validated += 1
        return validated

    def maintain_historical_metadata(self, limit: int = 20) -> dict[str, int]:
        """Advance legacy recording metadata only during an explicit repair.

        Normal background work drains queues populated by recent indexing,
        playback, and playback-failure recovery. This bounded opt-in path keeps
        old validation and codec fingerprinting available without continuously
        reading the historical recording library.
        """
        safe_limit = max(1, min(100, int(limit)))
        validated = self._validate_index_batch(
            limit=safe_limit,
            discover_unqueued=True,
        )
        fingerprints = self._backfill_stream_fingerprints(
            limit=safe_limit,
            discover_unqueued=True,
        )
        return {
            "recordings_validated": validated,
            "recording_fingerprints_added": fingerprints,
        }

    def _probe_recording(self, path: Path) -> tuple[float | None, str]:
        ffprobe = str(Path(self.ffmpeg_path).with_name("ffprobe")) if Path(self.ffmpeg_path).name == "ffmpeg" else "ffprobe"
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                    "-show_entries", "format=duration:packet=flags", "-of", "json", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"probe failed: {exc}"
        if result.returncode != 0:
            return None, f"probe failed: {(result.stderr or 'invalid media').strip()[-240:]}"
        try:
            payload = json.loads(result.stdout or "{}")
            duration = float((payload.get("format") or {}).get("duration") or 0)
            flags = str(((payload.get("packets") or [{}])[0]).get("flags") or "")
        except (ValueError, TypeError, IndexError) as exc:
            return None, f"probe metadata invalid: {exc}"
        if not math.isfinite(duration) or duration <= 0 or duration > max(60.0, self.segment_seconds * 3):
            return None, "recording duration is invalid"
        if "K" not in flags:
            return duration, "recording does not start on a video keyframe"
        return duration, ""

    def schedule_revalidation(self, path: Path, error: str) -> None:
        path_value = str(path)
        with self._index_connection() as connection:
            connection.execute(
                "UPDATE recordings SET playable = 0, health_error = ?, validated = 0 WHERE path = ?",
                (str(error)[-500:], path_value),
            )
        with self._validation_lock:
            if path_value not in self._validation_pending_set:
                self._validation_pending.append(path_value)
                self._validation_pending_set.add(path_value)

    def latest_indexed_row(self, camera_id: str, source: str) -> dict | None:
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            row = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds, end_epoch, source
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                ORDER BY start_epoch DESC LIMIT 1
                """,
                (camera_id, source),
            ).fetchone()
        return dict(row) if row is not None else None

    def recording_at(self, camera_id: str, epoch: float, limit: int = 1000, source: str = "main") -> dict | None:
        """Return an indexed recording without scanning media storage on the request path."""
        del limit  # Retained for API compatibility with older callers.
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            row = connection.execute(
                """
                SELECT path, name, size_bytes, modified_at, start_epoch, duration_seconds,
                       end_epoch, source, playable, health_error, validated,
                       stream_fingerprint, fingerprint_checked
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                  AND start_epoch <= ? AND end_epoch >= ?
                ORDER BY start_epoch DESC LIMIT 1
                """,
                (camera_id, source, epoch, epoch),
            ).fetchone()
        if row is not None:
            payload = dict(row)
            if Path(str(payload["path"])).is_file():
                return payload
            self._delete_index_paths([str(payload["path"])])
        if abs(time.time() - epoch) <= max(60.0, self.segment_seconds * 6):
            self.request_recording_edge_refresh(camera_id, source, epoch)
        return None

    def request_recording_edge_refresh(self, camera_id: str, source: str, after_epoch: float) -> None:
        """Ask the background indexer to discover a newly finalized near-live segment."""
        key = (camera_id, "main" if source == "main" else "live")
        with self._edge_refresh_lock:
            previous = self._edge_refresh_requests.get(key)
            self._edge_refresh_requests[key] = (
                min(previous, after_epoch) if previous is not None else after_epoch
            )
        self._index_wake.set()

    def _take_recording_edge_refreshes(self) -> dict[RecorderKey, float]:
        with self._edge_refresh_lock:
            requests = self._edge_refresh_requests
            self._edge_refresh_requests = {}
        return requests

    def recording_start_epoch(self, path: Path) -> float | None:
        try:
            value = datetime.strptime(path.stem, "%Y%m%d-%H%M%S%z")
            return value.timestamp()
        except ValueError:
            try:
                value = datetime.strptime(path.stem, "%Y%m%d-%H%M%S")
            except ValueError:
                return None
        return value.astimezone(timezone.utc).timestamp()
