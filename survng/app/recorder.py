from __future__ import annotations

import os
import json
import signal
import sqlite3
import subprocess
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .baichuan_native import BaichuanFfmpegPipe, ffmpeg_input_args, is_native_baichuan, start_ffmpeg_pipe
from .config import CameraConfig
from .recording_media import mp4_stream_fingerprint


ProcessItem = tuple[subprocess.Popen, BaichuanFfmpegPipe | None, threading.Event, threading.Thread]
RecorderKey = tuple[str, str]


class Recorder:
    def __init__(self, ffmpeg_path: str, storage_dir: Path, segment_seconds: float = 10.0, hardware_acceleration: str = "auto") -> None:
        self.ffmpeg_path = ffmpeg_path
        self.hardware_acceleration = hardware_acceleration
        self.segment_seconds = max(2.0, min(300.0, float(segment_seconds or 10.0)))
        self.recordings_dir = storage_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = storage_dir / "recordings.sqlite3"
        self.processes: dict[RecorderKey, ProcessItem] = {}
        self._starting: set[RecorderKey] = set()
        self._disabled_cameras: set[str] = set()
        self.owner_token = f"survng-{os.getpid()}"
        self._lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._index_stop = threading.Event()
        self._index_thread: threading.Thread | None = None
        self._prune_cursor = 0
        self._fingerprint_pending: deque[str] = deque()
        self._fingerprint_pending_set: set[str] = set()
        self._fingerprint_lock = threading.Lock()
        self._init_recording_index()

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

    def start(self, camera: CameraConfig, source: str = "main") -> None:
        source = camera.normalized_source(source, default="main")
        key = (camera.id, source)
        with self._lock:
            if camera.id in self._disabled_cameras:
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
            output = str(camera_dir / "%Y-%m-%d" / "%H" / "%Y%m%d-%H%M%S.mp4")
            command = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                *ffmpeg_input_args(camera, source),
                "-map",
                "0",
                "-metadata",
                f"survng_owner={self.owner_token}",
                "-metadata",
                f"survng_camera={camera.id}",
                "-metadata",
                f"survng_source={source}",
                "-c",
                "copy",
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
                start_new_session=True,
            )
            pipe = start_ffmpeg_pipe(camera, source, process)
            with self._lock:
                self.processes[key] = (process, pipe, stop_event, keeper)
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
                self._starting.discard(key)
        self.cleanup_duplicate_recorders({key})

    def _camera_dir(self, camera_id: str, source: str = "main") -> Path:
        source = "main" if source == "main" else "live"
        return self.recordings_dir / camera_id / source

    def _keep_recording_dirs(self, camera_dir: Path, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self._ensure_recording_dirs(camera_dir)
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

    def status(self, keys: set[RecorderKey] | None = None) -> dict[RecorderKey, bool]:
        with self._lock:
            stopped = [
                key
                for key, (process, _pipe, _stop_event, _keeper) in self.processes.items()
                if process.poll() is not None
            ]
            for key in stopped:
                item = self.processes.pop(key, None)
                if item is not None and item[1] is not None:
                    item[2].set()
                    item[1].stop()
                    item[3].join(timeout=1)
            tracked = {key: True for key in self.processes}

        if keys:
            for key, pids in self._owned_ffmpeg_recorders(keys).items():
                if pids:
                    tracked[key] = True
        return tracked

    def stop_all(self) -> None:
        self._index_stop.set()
        if self._index_thread is not None:
            self._index_thread.join(timeout=10)
            self._index_thread = None
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2)
            self._watchdog_thread = None
        with self._lock:
            items = list(self.processes.values())
            self.processes.clear()
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
            self.reconcile(camera_map)

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

    def recent_files(self, camera_id: str, limit: int = 20, source: str = "main") -> list[str]:
        source = "main" if source == "main" else "live"
        search_dirs = [self._camera_dir(camera_id, source)]
        if source == "main":
            legacy_dir = self.recordings_dir / camera_id
            if legacy_dir != search_dirs[0]:
                search_dirs.append(legacy_dir)
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
        source = "main" if source == "main" else "live"
        files = [Path(path) for path in self.recent_files(camera_id, limit=limit, source=source)]
        return self._recording_rows_for_files(camera_id, source, files)

    def recording_rows_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
    ) -> list[dict]:
        source = "main" if source == "main" else "live"
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
        rows = [dict(row) for row in indexed]
        stale_paths = {str(row["path"]) for row in rows if not Path(str(row["path"])).is_file()}
        if stale_paths:
            self._delete_index_paths(list(stale_paths))
            rows = [row for row in rows if str(row["path"]) not in stale_paths]
        if rows:
            for row in rows:
                row["start_at"] = datetime.fromtimestamp(float(row["start_epoch"]), timezone.utc).isoformat()
            return rows

        start_date = datetime.fromtimestamp(start_epoch).date() - timedelta(days=1)
        end_date = datetime.fromtimestamp(end_epoch).date() + timedelta(days=1)
        files: list[Path] = []
        camera_dir = self._camera_dir(camera_id, source)
        current_date = start_date
        while current_date <= end_date:
            day_dir = camera_dir / current_date.isoformat()
            if day_dir.exists():
                files.extend(day_dir.glob("??/*.mp4"))
            current_date += timedelta(days=1)
        rows = [
            row for row in self._recording_rows_for_files(camera_id, source, files)
            if row.get("start_epoch") is not None
            and row.get("end_epoch") is not None
            and float(row["end_epoch"]) > start_epoch
            and float(row["start_epoch"]) < end_epoch
        ]
        self._store_recording_rows(camera_id, source, rows)
        return rows

    def recording_availability_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
    ) -> dict:
        source = "main" if source == "main" else "live"
        with self._index_connection() as connection:
            indexed = connection.execute(
                """
                SELECT start_epoch, end_epoch
                FROM recordings
                WHERE camera_id = ? AND source = ? AND playable = 1
                  AND size_bytes > 1024 AND end_epoch > ? AND start_epoch < ?
                ORDER BY start_epoch
                """,
                (camera_id, source, start_epoch, end_epoch),
            ).fetchall()
        rows = [dict(row) for row in indexed]
        if not rows:
            rows = [
                row for row in self.recording_rows_between(
                    camera_id,
                    start_epoch,
                    end_epoch,
                    source,
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
        files = list(set(files))
        files.sort(key=lambda path: self.recording_start_epoch(path) or path.stat().st_mtime)
        all_files = files
        with self._lock:
            item = self.processes.get((camera_id, source))
            recorder_active = item is not None and item[0].poll() is None
        if recorder_active and files:
            files = files[:-1]
        rows: list[dict] = []
        for index, file_path in enumerate(files):
            start_epoch = self.recording_start_epoch(file_path)
            next_start_epoch = self.recording_start_epoch(all_files[index + 1]) if index + 1 < len(all_files) else None
            duration_seconds = self.segment_seconds
            if start_epoch is not None and next_start_epoch is not None:
                duration_seconds = max(1.0, min(self.segment_seconds, next_start_epoch - start_epoch))
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
                    "source": source,
                }
            )
        return rows

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
        camera_map = {camera.id: camera for camera in cameras if camera.record or camera.record_sub}
        self._index_stop.clear()
        if self._index_thread is not None and self._index_thread.is_alive():
            return
        self._index_thread = threading.Thread(
            target=self._recording_index_loop,
            args=(camera_map,),
            name="recording-indexer",
            daemon=True,
        )
        self._index_thread.start()

    def _recording_index_loop(self, camera_map: dict[str, CameraConfig]) -> None:
        self.refresh_recording_index(camera_map, full=True)
        while not self._index_stop.wait(10):
            self.refresh_recording_index(camera_map, full=False)

    def refresh_recording_index(self, camera_map: dict[str, CameraConfig], full: bool = False) -> None:
        now = datetime.now()
        wanted = self._wanted_keys(camera_map)
        for camera_id, source in wanted:
            camera_dir = self._camera_dir(camera_id, source)
            if full:
                files = list(camera_dir.glob("????-??-??/??/*.mp4")) if camera_dir.exists() else []
            else:
                files = []
                for hours_back in (1, 0):
                    target = now - timedelta(hours=hours_back)
                    hour_dir = camera_dir / target.strftime("%Y-%m-%d") / target.strftime("%H")
                    if hour_dir.exists():
                        files.extend(hour_dir.glob("*.mp4"))
            rows = self._recording_rows_for_files(camera_id, source, files)
            self._store_recording_rows(camera_id, source, rows, validate_new=not full)
            if full:
                self._prune_recording_index(camera_id, source, files)
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

    def _backfill_stream_fingerprints(self, limit: int = 20) -> int:
        paths: list[str] = []
        with self._fingerprint_lock:
            while self._fingerprint_pending and len(paths) < max(1, limit):
                path = self._fingerprint_pending.popleft()
                self._fingerprint_pending_set.discard(path)
                paths.append(path)
        if len(paths) < max(1, limit):
            with self._index_connection() as connection:
                pending = connection.execute(
                    """
                    SELECT path
                    FROM recordings
                    WHERE fingerprint_checked = 0
                    ORDER BY start_epoch DESC
                    LIMIT ?
                    """,
                    (max(1, limit) - len(paths),),
                ).fetchall()
            for row in pending:
                path = str(row["path"])
                if path not in paths:
                    paths.append(path)

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

    def _validate_index_batch(self, limit: int = 20) -> int:
        with self._index_connection() as connection:
            rows = connection.execute(
                "SELECT path, start_epoch FROM recordings WHERE validated = 0 ORDER BY start_epoch DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        validated = 0
        for row in rows:
            if self._index_stop.is_set():
                break
            path = Path(str(row["path"]))
            if not path.is_file():
                self._delete_index_paths([str(path)])
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
        if duration <= 0:
            return None, "recording duration is invalid"
        if "K" not in flags:
            return duration, "recording does not start on a video keyframe"
        return duration, ""

    def mark_unplayable(self, path: Path, error: str) -> None:
        with self._index_connection() as connection:
            connection.execute(
                "UPDATE recordings SET playable = 0, health_error = ?, validated = 1 WHERE path = ?",
                (str(error)[-500:], str(path)),
            )

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
        rows = self.recording_rows(camera_id, limit=limit, source=source)
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
