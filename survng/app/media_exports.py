from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .recording_media import concatenated_clip_timing
from .recorder import Recorder


LOGGER = logging.getLogger(__name__)
ACTIVE_STATUSES = {"queued", "running", "cancelling"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return safe.strip("-") or "camera"


class MediaExportStore:
    def __init__(self, database_dir: Path) -> None:
        database_dir.mkdir(parents=True, exist_ok=True)
        self.path = database_dir / "exports.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS media_exports (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    start_epoch REAL NOT NULL,
                    end_epoch REAL NOT NULL,
                    options_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT '',
                    progress REAL NOT NULL DEFAULT 0,
                    output_path TEXT NOT NULL DEFAULT '',
                    output_name TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "UPDATE media_exports SET status = 'failed', phase = 'Interrupted by restart', "
                "error = 'export interrupted by server restart', finished_at = ? "
                "WHERE status IN ('running', 'cancelling')",
                (_utc_now(),),
            )

    def create(self, payload: dict[str, object]) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO media_exports (
                    id, kind, camera_id, source, start_epoch, end_epoch,
                    options_json, status, phase, progress, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'Queued', 0, ?)
                """,
                (
                    job_id,
                    payload["kind"],
                    payload["camera_id"],
                    payload["source"],
                    payload["start_epoch"],
                    payload["end_epoch"],
                    json.dumps(payload.get("options") or {}, separators=(",", ":")),
                    created_at,
                ),
            )
        return self.get(job_id) or {}

    def get(self, job_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM media_exports WHERE id = ?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM media_exports ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [self._row(row) for row in rows]

    def queued_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM media_exports WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def update(self, job_id: str, **values: object) -> None:
        allowed = {
            "status", "phase", "progress", "output_path", "output_name", "size_bytes",
            "error", "started_at", "finished_at", "expires_at", "cancel_requested",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE media_exports SET {assignments} WHERE id = ?",
                (*updates.values(), job_id),
            )

    def delete(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM media_exports WHERE id = ?", (job_id,))

    def expired(self, now_iso: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM media_exports WHERE status IN ('completed', 'failed', 'cancelled') "
                "AND expires_at != '' AND expires_at < ?",
                (now_iso,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def completed_oldest_first(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM media_exports WHERE status = 'completed' ORDER BY created_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    def terminal_without_expiry(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM media_exports WHERE status IN ('completed', 'failed', 'cancelled') "
                "AND expires_at = ''"
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, object]:
        payload = dict(row)
        try:
            payload["options"] = json.loads(str(payload.pop("options_json") or "{}"))
        except (TypeError, json.JSONDecodeError):
            payload["options"] = {}
        payload["cancel_requested"] = bool(payload.get("cancel_requested"))
        return payload


class MediaExportManager:
    """Persistent, bounded recording export and timelapse worker."""

    def __init__(
        self,
        storage_dir: Path,
        database_dir: Path,
        recorder: Callable[[], Recorder],
        ffmpeg_path: Callable[[], str],
        hardware_backend: Callable[[], str],
        *,
        hardware_device: Callable[[str], str] | None = None,
        retention_hours: int = 24,
        max_storage_bytes: int = 20 * 1024 * 1024 * 1024,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.exports_dir = self.storage_dir / "exports"
        self.recording_dir = self.exports_dir / "recording"
        self.timelapse_dir = self.exports_dir / "timelapse"
        self.manifest_dir = self.exports_dir / "manifests"
        self.work_dir = database_dir.resolve() / "export-work"
        for directory in (self.recording_dir, self.timelapse_dir, self.manifest_dir, self.work_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._recorder = recorder
        self._ffmpeg_path = ffmpeg_path
        self._hardware_backend = hardware_backend
        self._hardware_device = hardware_device or (lambda _backend: "")
        self.retention_hours = max(1, min(int(retention_hours), 720))
        self.max_storage_bytes = max(1024 * 1024, int(max_storage_bytes))
        self.store = MediaExportStore(database_dir.resolve())
        default_expiry = (datetime.now(timezone.utc) + timedelta(hours=self.retention_hours)).isoformat()
        for job in self.store.terminal_without_expiry():
            self.store.update(str(job["id"]), expires_at=default_expiry)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_lock = threading.Lock()
        self._active_job_id = ""
        self._active_cancel: threading.Event | None = None
        self._active_process: subprocess.Popen | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="media-export-worker", daemon=False)
        self._thread.start()
        for job_id in self.store.queued_ids():
            self._enqueue(job_id)
        self.cleanup()

    def stop(self, timeout: float = 10.0) -> bool:
        self._stop.set()
        with self._active_lock:
            cancel = self._active_cancel
            process = self._active_process
        if cancel is not None:
            cancel.set()
        if process is not None and process.poll() is None:
            self._terminate_process(process)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, timeout))
        return not thread.is_alive()

    def create(self, payload: dict[str, object]) -> dict[str, object]:
        job = self.store.create(payload)
        self._enqueue(str(job["id"]))
        return self.public_job(job)

    def list(self, limit: int = 50) -> list[dict[str, object]]:
        return [self.public_job(job) for job in self.store.list(limit)]

    def active_jobs(self) -> list[dict[str, object]]:
        return [job for job in self.list(100) if job.get("status") in ACTIVE_STATUSES]

    def get(self, job_id: str) -> dict[str, object] | None:
        job = self.store.get(job_id)
        return self.public_job(job) if job else None

    def cancel_or_delete(self, job_id: str) -> dict[str, object]:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        status = str(job["status"])
        if status in ACTIVE_STATUSES:
            self.store.update(job_id, cancel_requested=1, status="cancelling", phase="Cancelling")
            with self._active_lock:
                if self._active_job_id == job_id and self._active_cancel is not None:
                    self._active_cancel.set()
                    if self._active_process is not None:
                        self._terminate_process(self._active_process)
            return self.get(job_id) or {}
        self._delete_job_files(job)
        self.store.delete(job_id)
        return {"id": job_id, "deleted": True}

    def output_path(self, job_id: str) -> tuple[Path, str]:
        job = self.store.get(job_id)
        if job is None or job.get("status") != "completed":
            raise FileNotFoundError(job_id)
        raw = str(job.get("output_path") or "")
        path = Path(raw).resolve(strict=True)
        path.relative_to(self.exports_dir)
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return path, str(job.get("output_name") or path.name)

    def public_job(self, job: dict[str, object]) -> dict[str, object]:
        payload = {key: value for key, value in job.items() if key != "output_path"}
        payload["download_url"] = f"/api/exports/{job['id']}/download" if job.get("status") == "completed" else ""
        return payload

    def cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        for job in self.store.expired(now.isoformat()):
            self._delete_job_files(job)
            self.store.delete(str(job["id"]))
        completed = self.store.completed_oldest_first()
        total = sum(int(job.get("size_bytes") or 0) for job in completed)
        for job in completed:
            if total <= self.max_storage_bytes:
                break
            total -= int(job.get("size_bytes") or 0)
            self._delete_job_files(job)
            self.store.delete(str(job["id"]))
        cutoff = time.time() - 3600
        for path in self.exports_dir.rglob("*.partial"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        for directory in self.work_dir.iterdir():
            try:
                if directory.is_dir() and directory.stat().st_mtime < cutoff:
                    shutil.rmtree(directory, ignore_errors=True)
            except OSError:
                continue

    def _enqueue(self, job_id: str) -> None:
        try:
            self._queue.put_nowait(job_id)
        except queue.Full as exc:
            self.store.update(job_id, status="failed", phase="Queue full", error="export queue is full", finished_at=_utc_now())
            raise RuntimeError("export queue is full") from exc

    def _run(self) -> None:
        next_cleanup = time.monotonic() + 60.0
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() >= next_cleanup:
                    self.cleanup()
                    next_cleanup = time.monotonic() + 60.0
                continue
            if job_id is None:
                return
            job = self.store.get(job_id)
            if job is None or job.get("status") not in {"queued", "cancelling"}:
                continue
            if job.get("cancel_requested"):
                self._finish_cancelled(job_id)
                continue
            cancel = threading.Event()
            with self._active_lock:
                self._active_job_id = job_id
                self._active_cancel = cancel
            try:
                self._execute(job, cancel)
            except InterruptedError:
                self._finish_cancelled(job_id)
            except BaseException as exc:
                LOGGER.exception("media export %s failed", job_id)
                self.store.update(
                    job_id,
                    status="failed",
                    phase="Failed",
                    error=str(exc)[:500],
                    finished_at=_utc_now(),
                    expires_at=(datetime.now(timezone.utc) + timedelta(hours=self.retention_hours)).isoformat(),
                )
            finally:
                with self._active_lock:
                    self._active_job_id = ""
                    self._active_cancel = None
                    self._active_process = None
                self.cleanup()
                next_cleanup = time.monotonic() + 60.0

    def _execute(self, job: dict[str, object], cancel: threading.Event) -> None:
        job_id = str(job["id"])
        self.store.update(job_id, status="running", phase="Reading recording index", progress=3, started_at=_utc_now())
        recorder = self._recorder()
        rows = recorder.recording_rows_between(
            str(job["camera_id"]),
            float(job["start_epoch"]),
            float(job["end_epoch"]),
            str(job["source"]),
            discover_missing=False,
        )
        rows = [row for row in rows if Path(str(row.get("path") or "")).is_file()]
        if not rows:
            raise RuntimeError("no indexed recordings exist in the selected range")
        lease_seconds = max(600.0, min(21600.0, float(job["end_epoch"]) - float(job["start_epoch"]) + 600.0))
        recorder.lease_recordings_for_playback(rows, ttl_seconds=lease_seconds)
        if cancel.is_set() or self._stop.is_set():
            raise InterruptedError
        work = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=self.work_dir))
        try:
            if job["kind"] == "recording":
                output, gaps = self._build_recording(job, rows, work, cancel)
            else:
                output, gaps = self._build_timelapse(job, rows, work, cancel)
            self.store.update(job_id, phase="Finalizing", progress=92)
            final_path, output_name = self._publish(job, output)
            manifest = {
                "job_id": job_id,
                "kind": job["kind"],
                "camera_id": job["camera_id"],
                "source": job["source"],
                "start_epoch": job["start_epoch"],
                "end_epoch": job["end_epoch"],
                "options": job.get("options") or {},
                "gaps": gaps,
                "generated_at": _utc_now(),
                "output_name": output_name,
            }
            self._write_manifest(job_id, manifest)
            expires = datetime.now(timezone.utc) + timedelta(hours=self.retention_hours)
            self.store.update(
                job_id,
                status="completed",
                phase="Ready",
                progress=100,
                output_path=str(final_path),
                output_name=output_name,
                size_bytes=final_path.stat().st_size,
                finished_at=_utc_now(),
                expires_at=expires.isoformat(),
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _build_recording(
        self,
        job: dict[str, object],
        rows: list[dict],
        work: Path,
        cancel: threading.Event,
    ) -> tuple[Path, list[dict[str, float]]]:
        groups, gaps = self._continuous_groups(rows, float(job["start_epoch"]), float(job["end_epoch"]))
        if not groups:
            raise RuntimeError("no continuous recording spans exist in the selected range")
        parts: list[Path] = []
        for index, group in enumerate(groups, start=1):
            if cancel.is_set() or self._stop.is_set():
                raise InterruptedError
            self._recorder().lease_recordings_for_playback(group, ttl_seconds=900)
            local_start, duration = concatenated_clip_timing(
                group,
                max(float(job["start_epoch"]), float(group[0]["start_epoch"])),
                min(float(job["end_epoch"]), float(group[-1]["end_epoch"])),
            )
            concat = self._write_concat(group, work / f"part-{index}.ffconcat")
            output = work / f"part-{index:02d}.mp4"
            command = [
                self._ffmpeg_path(), "-hide_banner", "-loglevel", "warning",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-ss", f"{local_start:.3f}", "-t", f"{duration:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                "-movflags", "+faststart", "-y", str(output),
            ]
            self._run_process(command, cancel, timeout=max(120.0, min(1800.0, duration * 0.5)))
            if not self._valid_media_container(output):
                raise RuntimeError("recording export produced no playable video")
            parts.append(output)
            self.store.update(str(job["id"]), phase=f"Exporting part {index} of {len(groups)}", progress=10 + (75 * index / len(groups)))
        if len(parts) == 1:
            return parts[0], gaps
        archive = work / "recording-parts.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for part in parts:
                bundle.write(part, part.name)
        return archive, gaps

    def _build_timelapse(
        self,
        job: dict[str, object],
        rows: list[dict],
        work: Path,
        cancel: threading.Event,
    ) -> tuple[Path, list[dict[str, float]]]:
        groups, gaps = self._continuous_groups(rows, float(job["start_epoch"]), float(job["end_epoch"]))
        if not groups:
            raise RuntimeError("no continuous recording spans exist in the selected range")
        options = dict(job.get("options") or {})
        interval = max(1.0, min(float(options.get("sample_interval_seconds") or 30.0), 3600.0))
        output_fps = max(1, min(int(options.get("output_fps") or 30), 60))
        width = max(320, min(int(options.get("width") or 1280), 3840))
        # A concat demuxer collapses recording gaps. That is desirable for a
        # timelapse: missing source time does not become a frozen video span.
        local_start, available_duration = concatenated_clip_timing(
            rows, float(job["start_epoch"]), float(job["end_epoch"])
        )
        estimated_frames = max(1, math.ceil(available_duration / interval))
        if estimated_frames > 20000:
            raise RuntimeError("timelapse would exceed 20,000 frames; increase the sample interval")
        concat = self._write_concat(rows, work / "timelapse.ffconcat")
        output = work / "timelapse.mp4"
        filters = (
            f"trim=start={local_start:.6f}:duration={available_duration:.6f},"
            "setpts=PTS-STARTPTS,"
            f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval:.6f})',"
            f"scale='min({width},iw)':-2,settb=AVTB,setpts=N/({output_fps}*TB)"
        )
        backend = self._hardware_backend()
        commands = self._timelapse_commands(backend, concat, output, filters, output_fps)
        last_error = "timelapse generation failed"
        for backend_name, command in commands:
            output.unlink(missing_ok=True)
            try:
                self.store.update(str(job["id"]), phase=f"Encoding timelapse ({backend_name})", progress=12)
                self._run_process(command, cancel, timeout=max(300.0, available_duration * 0.4))
            except RuntimeError as exc:
                last_error = str(exc)
                if backend_name != "cpu":
                    LOGGER.warning("%s timelapse export failed; trying fallback: %s", backend_name, last_error)
                    continue
                raise
            if self._valid_video_output(output):
                return output, gaps
            last_error = f"{backend_name} timelapse produced no playable video frames"
            if backend_name != "cpu":
                LOGGER.warning("%s; trying fallback", last_error)
                continue
            raise RuntimeError(last_error)
        raise RuntimeError(last_error)

    def _timelapse_commands(
        self,
        backend: str,
        concat: Path,
        output: Path,
        filters: str,
        output_fps: int,
    ) -> list[tuple[str, list[str]]]:
        input_args = [
            self._ffmpeg_path(), "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-map", "0:v:0", "-an",
        ]
        suffix = ["-movflags", "+faststart", "-y", str(output)]
        commands: list[tuple[str, list[str]]] = []
        if backend == "qsv":
            device = self._hardware_device("qsv")
            device_args = ["-qsv_device", device] if device else []
            commands.append(("qsv", [
                self._ffmpeg_path(), "-hide_banner", "-loglevel", "warning", *device_args,
                "-f", "concat", "-safe", "0", "-i", str(concat), "-map", "0:v:0", "-an",
                "-vf", filters, "-r", str(output_fps), "-fps_mode", "cfr",
                "-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "23",
                "-pix_fmt", "nv12", *suffix,
            ]))
        elif backend == "vaapi":
            device = self._hardware_device("vaapi") or "/dev/dri/renderD128"
            commands.append(("vaapi", [
                self._ffmpeg_path(), "-hide_banner", "-loglevel", "warning", "-vaapi_device", device,
                "-f", "concat", "-safe", "0", "-i", str(concat), "-map", "0:v:0", "-an",
                "-vf", f"{filters},format=nv12,hwupload", "-r", str(output_fps), "-fps_mode", "cfr",
                "-c:v", "h264_vaapi", "-qp", "23", *suffix,
            ]))
        commands.append(("cpu", [
            *input_args, "-vf", filters, "-r", str(output_fps), "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", *suffix,
        ]))
        return commands

    def _valid_video_output(self, path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < 1024:
                return False
            ffmpeg = Path(self._ffmpeg_path())
            sibling = ffmpeg.with_name("ffprobe")
            ffprobe = str(sibling) if ffmpeg.name == "ffmpeg" and sibling.is_file() else "ffprobe"
            probe = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames,duration",
                    "-of", "json", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if probe.returncode != 0:
                return False
            payload = json.loads(probe.stdout or "{}")
            stream = next(iter(payload.get("streams") or []), {})
            return int(stream.get("nb_read_frames") or 0) > 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return False

    def _valid_media_container(self, path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < 1024:
                return False
            ffmpeg = Path(self._ffmpeg_path())
            sibling = ffmpeg.with_name("ffprobe")
            ffprobe = str(sibling) if ffmpeg.name == "ffmpeg" and sibling.is_file() else "ffprobe"
            probe = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            payload = json.loads(probe.stdout or "{}") if probe.returncode == 0 else {}
            streams = payload.get("streams") or []
            duration = float((payload.get("format") or {}).get("duration") or 0)
            return bool(streams and duration > 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return False

    def _run_process(self, command: list[str], cancel: threading.Event, timeout: float) -> None:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
            with self._active_lock:
                self._active_process = process
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancel.is_set() or self._stop.is_set():
                        self._terminate_process(process)
                        raise InterruptedError
                    if time.monotonic() >= deadline:
                        self._terminate_process(process)
                        raise RuntimeError("export encoding timed out")
                    time.sleep(0.2)
                stderr_file.seek(0)
                stderr = stderr_file.read()
                if process.returncode != 0:
                    raise RuntimeError((stderr or "FFmpeg export failed").strip()[-500:])
            finally:
                with self._active_lock:
                    if self._active_process is process:
                        self._active_process = None

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def _write_concat(rows: list[dict], path: Path) -> Path:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                value = str(Path(str(row["path"])).resolve())
                if "\n" in value or "\r" in value:
                    raise RuntimeError("recording path is invalid")
                escaped = value.replace("\\", "\\\\").replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        return path

    @staticmethod
    def _continuous_groups(
        rows: list[dict], start_epoch: float, end_epoch: float
    ) -> tuple[list[list[dict]], list[dict[str, float]]]:
        ordered = sorted(rows, key=lambda row: float(row["start_epoch"]))
        groups: list[list[dict]] = []
        gaps: list[dict[str, float]] = []
        for row in ordered:
            row_start = float(row["start_epoch"])
            row_end = float(row["end_epoch"])
            if row_end <= start_epoch or row_start >= end_epoch:
                continue
            previous = groups[-1][-1] if groups else None
            has_gap = previous is not None and row_start > float(previous["end_epoch"]) + 0.5
            previous_fingerprint = str(previous.get("stream_fingerprint") or "") if previous else ""
            row_fingerprint = str(row.get("stream_fingerprint") or "")
            stream_changed = bool(
                previous_fingerprint and row_fingerprint and previous_fingerprint != row_fingerprint
            )
            if not groups or has_gap or stream_changed:
                if groups:
                    gap_start = float(groups[-1][-1]["end_epoch"])
                    if row_start > gap_start:
                        gaps.append({"start_epoch": gap_start, "end_epoch": row_start, "duration_seconds": row_start - gap_start})
                groups.append([row])
            else:
                groups[-1].append(row)
        return groups, gaps

    def _publish(self, job: dict[str, object], source: Path) -> tuple[Path, str]:
        timestamp = datetime.fromtimestamp(float(job["start_epoch"]), timezone.utc).strftime("%Y%m%d-%H%M%S")
        camera = _safe_component(str(job["camera_id"]))
        suffix = source.suffix.lower() if source.suffix else ".mp4"
        name = f"{camera}-{timestamp}-{str(job['kind'])}{suffix}"
        destination_dir = self.recording_dir if job["kind"] == "recording" else self.timelapse_dir
        final = destination_dir / f"{job['id']}-{name}"
        partial = final.with_suffix(final.suffix + ".partial")
        shutil.copyfile(source, partial)
        os.replace(partial, final)
        return final, name

    def _write_manifest(self, job_id: str, payload: dict[str, object]) -> None:
        final = self.manifest_dir / f"{job_id}.json"
        temporary = final.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, final)

    def _delete_job_files(self, job: dict[str, object]) -> None:
        raw = str(job.get("output_path") or "")
        if raw:
            try:
                path = Path(raw).resolve(strict=False)
                path.relative_to(self.exports_dir)
                path.unlink(missing_ok=True)
            except (OSError, ValueError):
                LOGGER.warning("refused to delete export path outside export storage: %s", raw)
        (self.manifest_dir / f"{job['id']}.json").unlink(missing_ok=True)

    def _finish_cancelled(self, job_id: str) -> None:
        self.store.update(
            job_id,
            status="cancelled",
            phase="Cancelled",
            progress=0,
            finished_at=_utc_now(),
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=self.retention_hours)).isoformat(),
        )
