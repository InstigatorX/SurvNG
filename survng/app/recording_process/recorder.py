from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import CameraConfig, RecordingRetentionConfig
from ..ffmpeg_input import ffmpeg_input_args, ffmpeg_timestamp_repair_args
from ..go2rtc import Go2RtcAdapter, Go2RtcError
from ..media_storage import MediaStorageRegistry
from ..recording_retention import RecordingRetentionService
from .index import RecordingIndexMixin


ProcessItem = tuple[subprocess.Popen, threading.Event, threading.Thread]
RecorderKey = tuple[str, str]
LOGGER = logging.getLogger("survng.app.recorder")

DTS_WARNING_WINDOW_SECONDS = 5.0
DTS_WARNING_RESTART_COUNT = 12
TIMESTAMP_ROLLOVER_LIMIT = 3
TIMESTAMP_ROLLOVER_LIMIT_WINDOW_SECONDS = 300.0


@dataclass(frozen=True)
class AudioStreamInfo:
    known: bool
    codec: str = ""
    sample_rate: int = 0


class Recorder(RecordingIndexMixin):
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
        snapshot_retention_plan: Callable[[float], Mapping[str, Any]] | None = None,
        apply_snapshot_retention: Callable[[float, int], Mapping[str, Any]] | None = None,
        migrate_snapshot_sizes: Callable[..., int] | None = None,
        media_storage: MediaStorageRegistry | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.hardware_acceleration = hardware_acceleration
        self.go2rtc = go2rtc or Go2RtcAdapter(timeout=1.5, cache_seconds=120.0)
        self.segment_seconds = max(2.0, min(300.0, float(segment_seconds or 10.0)))
        self.storage_dir = storage_dir
        self.media_storage = media_storage
        self.recording_roots = (
            media_storage.roots_for("recordings")
            if media_storage is not None
            else [storage_dir / "recordings"]
        )
        if not self.recording_roots:
            raise ValueError("at least one media location must support recordings")
        self.recordings_dir = self.recording_roots[0]
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
        self._watchdog_wake = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._timestamp_rollover_pending: dict[RecorderKey, tuple[int, str]] = {}
        self._timestamp_rollover_history: dict[RecorderKey, deque[float]] = {}
        self._timestamp_health: dict[RecorderKey, dict[str, object]] = {}
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
        self._location_backfill_lock = threading.Lock()
        self._external_protected_recording_paths = protected_recording_paths or set
        self._migrate_snapshot_sizes = migrate_snapshot_sizes
        self._playback_lease_lock = threading.Lock()
        self._playback_leases: dict[str, float] = {}
        self._retention_deletions: set[str] = set()
        self._init_recording_index()
        self.retention = RecordingRetentionService(
            self.storage_dir,
            self.recordings_dir,
            self._index_connection,
            retention_config or RecordingRetentionConfig(),
            self._protected_recording_paths,
            snapshot_plan_provider=snapshot_retention_plan,
            snapshot_cleanup_provider=apply_snapshot_retention,
            media_storage=self.media_storage,
            delete_recording_provider=self._delete_recording_for_retention,
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
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            threading.Thread(
                target=self._monitor_ffmpeg_stderr,
                args=(key, process),
                name=f"recorder-stderr-{camera.id}-{source}",
                daemon=True,
            ).start()
            with self._lock:
                keep_process = key in self._starting and camera.id not in self._disabled_cameras
                if keep_process:
                    self.processes[key] = (process, stop_event, keeper)
                    self._retry_after.pop(key, None)
            if not keep_process:
                stop_event.set()
                if process.poll() is None:
                    self._kill_pid(process.pid)
                keeper.join(timeout=1)
                return
        except Exception as error:
            stop_event.set()
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
        if (info.known and info.codec == "aac") or not info.known:
            return ["-map", "0:a:0?", "-c:a", "copy"]
        bitrate_kbps = 64
        if info.sample_rate > 0:
            bitrate_kbps = max(24, min(64, info.sample_rate * 6 // 1000))
        return ["-map", "0:a:0?", "-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]

    def _probe_audio_stream(self, camera: CameraConfig, source: str) -> AudioStreamInfo:
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
        rollover_requested = False
        last_invalid_dts_at: float | None = None
        for raw_line in process.stderr:
            line = (
                raw_line.decode("utf-8", errors="replace").strip()
                if isinstance(raw_line, bytes)
                else raw_line.strip()
            )
            if not line:
                continue
            warning_kind = self._timestamp_warning_kind(line)
            if warning_kind is not None:
                # FFmpeg 8 normally reports one DTS line and one paired PTS line.
                # Count the DTS as the packet signal and suppress its immediate
                # PTS twin. A PTS-only stream still remains eligible for recovery.
                now = time.monotonic()
                if (
                    warning_kind == "invalid_pts"
                    and last_invalid_dts_at is not None
                    and now - last_invalid_dts_at < 0.25
                ):
                    continue
                if warning_kind == "invalid_dts":
                    last_invalid_dts_at = now
                if rollover_requested:
                    continue
                dts_warnings.append(now)
                while dts_warnings and now - dts_warnings[0] > DTS_WARNING_WINDOW_SECONDS:
                    dts_warnings.popleft()
                if len(dts_warnings) >= DTS_WARNING_RESTART_COUNT:
                    rollover_requested = self._request_timestamp_rollover(
                        key,
                        process.pid,
                        warning_kind,
                    )
                continue
            LOGGER.warning("Recorder FFmpeg %s/%s: %s", key[0], key[1], line)

    @staticmethod
    def _timestamp_warning_kind(line: str) -> str | None:
        if "Non-monotonic DTS" in line:
            return "non_monotonic_dts"
        if " invalid dropping" not in line:
            return None
        if "] DTS " in line:
            return "invalid_dts"
        if "] PTS " in line:
            return "invalid_pts"
        return None

    def _request_timestamp_rollover(
        self,
        key: RecorderKey,
        process_pid: int,
        reason: str,
    ) -> bool:
        now_monotonic = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()
        limited = False
        with self._lock:
            existing = self._timestamp_rollover_pending.get(key)
            if existing is not None and existing[0] == process_pid:
                return True
            history = self._timestamp_rollover_history.setdefault(key, deque())
            while history and now_monotonic - history[0] > TIMESTAMP_ROLLOVER_LIMIT_WINDOW_SECONDS:
                history.popleft()
            health = self._timestamp_health.setdefault(
                key,
                {
                    "discontinuities": 0,
                    "epoch_rollovers": 0,
                    "rollover_failures": 0,
                    "rate_limited": 0,
                    "last_discontinuity_at": None,
                    "last_rollover_at": None,
                    "last_reason": "",
                },
            )
            health["discontinuities"] = int(health["discontinuities"]) + 1
            health["last_discontinuity_at"] = now_iso
            health["last_reason"] = reason
            if len(history) >= TIMESTAMP_ROLLOVER_LIMIT:
                health["rate_limited"] = int(health["rate_limited"]) + 1
                limited = True
            else:
                self._timestamp_rollover_pending[key] = (process_pid, reason)
        if limited:
            LOGGER.error(
                "Recorder %s/%s timestamp recovery rate-limited after %d rollovers in %.0f seconds",
                key[0],
                key[1],
                TIMESTAMP_ROLLOVER_LIMIT,
                TIMESTAMP_ROLLOVER_LIMIT_WINDOW_SECONDS,
            )
            return True
        LOGGER.warning(
            "Recorder %s/%s detected persistent timestamp discontinuity (%s); scheduling an epoch rollover",
            key[0],
            key[1],
            reason,
        )
        self._watchdog_wake.set()
        return True

    def timestamp_health(self) -> dict[RecorderKey, dict[str, object]]:
        with self._lock:
            pending = set(self._timestamp_rollover_pending)
            return {
                key: {**values, "rollover_pending": key in pending}
                for key, values in self._timestamp_health.items()
            }

    def _camera_dir(self, camera_id: str, source: str = "main") -> Path:
        source = "main" if source == "main" else "live"
        if self.media_storage is None:
            root = self.recordings_dir
        else:
            assignment = f"{camera_id}:{source}"
            root = self.media_storage.directory("recordings", assignment)
        return root / camera_id / source

    def _recording_search_dirs(self, camera_id: str, source: str) -> list[Path]:
        normalized = "main" if source == "main" else "live"
        directories: list[Path] = []
        for root in self.recording_roots:
            directories.append(root / camera_id / normalized)
            if normalized == "main":
                directories.append(root / camera_id)
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
        process, stop_event, keeper = item
        stop_event.set()
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

    def status(self, keys: set[RecorderKey] | None = None) -> dict[RecorderKey, bool]:
        stopped_items: list[ProcessItem] = []
        with self._lock:
            stopped = [
                key
                for key, (process, _stop_event, _keeper) in self.processes.items()
                if process.poll() is not None
            ]
            for key in stopped:
                item = self.processes.pop(key, None)
                if item is not None:
                    stopped_items.append(item)
            tracked = {key: True for key in self.processes}

        # Keeper joins may block. Never hold the recorder state lock while
        # cleaning up a process that has already exited.
        for _process, stop_event, keeper in stopped_items:
            stop_event.set()
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
        self._watchdog_wake.set()
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
        self._watchdog_wake.clear()
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
        while not self._watchdog_stop.is_set():
            self._watchdog_wake.wait(interval)
            self._watchdog_wake.clear()
            if self._watchdog_stop.is_set():
                break
            try:
                self.reconcile(camera_map)
            except Exception:
                LOGGER.exception("Recorder watchdog reconciliation failed")

    def reconcile(self, camera_map: dict[str, CameraConfig]) -> None:
        wanted = self._wanted_keys(camera_map)
        self._apply_timestamp_rollovers(wanted)
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

    def _apply_timestamp_rollovers(self, wanted: dict[RecorderKey, CameraConfig]) -> None:
        with self._lock:
            pending = list(self._timestamp_rollover_pending.items())
        for key, (expected_pid, reason) in pending:
            camera = wanted.get(key)
            with self._lock:
                item = self.processes.get(key)
                current_pid = item[0].pid if item is not None else None
                if camera is None or current_pid != expected_pid:
                    self._timestamp_rollover_pending.pop(key, None)
                    continue
                self._timestamp_rollover_pending.pop(key, None)
            LOGGER.info(
                "Recorder %s/%s beginning controlled timestamp epoch rollover (%s)",
                key[0],
                key[1],
                reason,
            )
            self.stop(key[0], key[1])
            self.start(camera, key[1])
            now_monotonic = time.monotonic()
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._lock:
                history = self._timestamp_rollover_history.setdefault(key, deque())
                history.append(now_monotonic)
                health = self._timestamp_health[key]
                replacement = self.processes.get(key)
                recovered = replacement is not None and replacement[0].poll() is None
                if recovered:
                    health["epoch_rollovers"] = int(health["epoch_rollovers"]) + 1
                    health["last_rollover_at"] = now_iso
                else:
                    health["rollover_failures"] = int(health["rollover_failures"]) + 1
            if recovered:
                LOGGER.info(
                    "Recorder %s/%s completed timestamp epoch rollover",
                    key[0],
                    key[1],
                )
            else:
                LOGGER.error(
                    "Recorder %s/%s could not start its replacement after timestamp epoch rollover",
                    key[0],
                    key[1],
                )

    def cleanup_duplicate_recorders(self, keys: set[RecorderKey]) -> None:
        owned = self._owned_ffmpeg_recorders(keys)
        with self._lock:
            tracked_pids = {item[0].pid for item in self.processes.values()}
        extras: set[int] = set()
        for key, pids in owned.items():
            key_extras = [pid for pid in pids if pid not in tracked_pids]
            if len([pid for pid in pids if pid in tracked_pids]) > 1:
                key_extras.extend(
                    sorted(pid for pid in pids if pid in tracked_pids)[1:]
                )
            extras.update(key_extras)
        self._kill_pids(extras)

    def cleanup_stale_recorders(self, keys: set[RecorderKey]) -> None:
        stale_pids = {
            pid
            for pids in self._owned_ffmpeg_recorders(keys).values()
            for pid in pids
        }
        self._kill_pids(stale_pids)

    def _kill_pids(self, pids: set[int]) -> None:
        """Terminate independent recorder processes within one shared wait window."""
        failures: list[tuple[int, Exception]] = []
        failures_lock = threading.Lock()

        def kill(pid: int) -> None:
            try:
                self._kill_pid(pid)
            except Exception as exc:
                with failures_lock:
                    failures.append((pid, exc))

        threads = [
            threading.Thread(
                target=kill,
                args=(pid,),
                name=f"cleanup-recorder-{pid}",
                daemon=False,
            )
            for pid in sorted(pids)
            if pid != os.getpid()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            failed_pids = ", ".join(str(pid) for pid, _exc in failures)
            raise RuntimeError(
                f"failed to terminate recorder processes: {failed_pids}"
            ) from failures[0][1]

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

