"""Live capture through a shared DL Streamer / GStreamer supervisor process."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import select
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import numpy as np

from .camera_capture import (
    CAPTURE_OPEN_LOCK_POLL_SECONDS,
    CAPTURE_OPEN_TIMEOUT_MS,
    CAPTURE_PIPE_READ_CHUNK_BYTES,
    CAPTURE_READ_TIMEOUT_MS,
    CaptureHandle,
    CaptureOpenLimiter,
)
from .dlstreamer_protocol import (
    TYPE_DETECTIONS,
    TYPE_FATAL,
    TYPE_FRAME,
    TYPE_JPEG,
    TYPE_STATUS,
    MessageReader,
    decode_frame_payload,
    decode_jpeg_payload,
    decode_json_payload,
    decode_stream_payload,
)
from survng.dlstreamer_live import model_instance_id
from .live_detections import DetectionSnapshot
from .redact import redact_secret_text

LOGGER = logging.getLogger(__name__)
# Native GPU model compilation is part of opening a live inference pipeline,
# not a stalled RTSP read. Keep parent and child startup budgets in agreement.
DLSTREAMER_INFERENCE_STARTUP_TIMEOUT_MS = 30000
DLSTREAMER_INFERENCE_STALL_SECONDS = 5.0


def adjacent_model_proc(model_path: str) -> str:
    path = Path(model_path)
    if not path.name:
        return ""
    for candidate in (
        path.with_name(f"{path.stem}.json"),
        path.with_name(f"{path.stem}_proc.json"),
        path.parent / "model-proc.json",
    ):
        if candidate.is_file():
            return str(candidate)
    return ""


@dataclass(frozen=True, slots=True)
class DlStreamerCaptureOptions:
    python_executable: str = ""
    decoder: str = "va"
    open_timeout_ms: int = CAPTURE_OPEN_TIMEOUT_MS
    read_timeout_ms: int = CAPTURE_READ_TIMEOUT_MS
    admission_poll_seconds: float = CAPTURE_OPEN_LOCK_POLL_SECONDS
    rtsp_transport: str = "tcp"
    frame_rate: Callable[[], float] | None = None
    detection_frame_rate: Callable[[], float] | None = None
    main_frame_rate: Callable[[], float] | None = None
    model_path: str = ""
    labels_path: str = ""
    labels: tuple[str, ...] = ()
    model_proc_path: str = ""
    inference_device: str = "GPU"
    detect_enabled: bool = False
    frame_width: int = 320
    jpeg_fps: float = 1.0
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.45


def live_python_executable(preferred: str = "") -> str:
    """Prefer system Python so GStreamer GI plugins resolve.

    Isolated live capture runs under ``/usr/bin/python3``, where ``python3-gi``
    is installed. The SurvNG venv has pydantic and the rest of the app stack;
    the child must not import that.
    """
    candidates = [
        preferred,
        os.environ.get("SURVNG_DLSTREAMER_PYTHON", ""),
        "/usr/bin/python3",
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return sys.executable


class _StreamInbox:
    """Per-camera messages demuxed from the shared live supervisor."""

    def __init__(self) -> None:
        self.alive = True
        self.error = ""
        self.status: dict[str, object] = {}
        self._frames: queue.Queue[tuple[np.ndarray, int, float, str]] = queue.Queue(maxsize=2)
        self._detections: list[dict[str, object]] = []
        self._detection_snapshots: deque[DetectionSnapshot] = deque(maxlen=32)
        self.session = uuid.uuid4().hex
        self._last_pts: dict[str, float] = {}
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()

        # Video progress is not inference progress. An empty result counts,
        # but repeated metadata with the same PTS does not refresh liveness.
        self.inference_started_at: float | None = None
        self.last_inference_at: float | None = None
        self.last_frame_at: float | None = None
        self.video_resumed_at: float | None = None

    def put_frame(self, frame: np.ndarray, sequence: int, pts: float) -> None:
        session = self.qualify_pts("frame", pts)
        now = time.monotonic()
        if self.last_frame_at is None or now - self.last_frame_at > 1.0:
            self.video_resumed_at = now
        self.last_frame_at = now
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frames.put_nowait((frame, sequence, pts, session))
        except queue.Full:
            pass

    def get_frame(
        self,
        timeout_seconds: float,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, int, float, str] | None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancelled is not None and cancelled():
                return None
            if self.error:
                raise RuntimeError(self.error)
            if not self.alive and self._frames.empty():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return self._frames.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue

    def pop_detections(self) -> list[dict[str, object]]:
        with self._lock:
            detections = list(self._detections)
            self._detections = []
            return detections

    def pop_jpeg(self) -> bytes | None:
        with self._lock:
            jpeg, self._jpeg = self._jpeg, None
            return jpeg

    def set_detections(self, objects: list[dict[str, object]]) -> None:
        with self._lock:
            self._detections = objects

    def qualify_pts(self, kind: str, pts: float) -> str:
        with self._lock:
            previous = self._last_pts.get(kind)
            if previous is not None and math.isfinite(pts) and pts < previous:
                self.session = uuid.uuid4().hex
                self._last_pts.clear()
                self._detection_snapshots.clear()
                self._detections = []
                self._jpeg = None
                while not self._frames.empty():
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        break
            if math.isfinite(pts):
                self._last_pts[kind] = pts
            return self.session

    def add_detection_snapshot(self, payload: dict[str, object]) -> None:
        # Parse before altering the session; corrupt metadata must not reset it.
        snapshot = DetectionSnapshot.parse(payload)
        previous_pts = self._last_pts.get("detection")
        session = self.qualify_pts("detection", snapshot.source_pts)
        snapshot = DetectionSnapshot.parse(payload, session=session)
        with self._lock:
            self._detection_snapshots.append(snapshot)
            if previous_pts != snapshot.source_pts:
                self.last_inference_at = time.monotonic()

    def pop_detection_snapshots(self) -> list[DetectionSnapshot]:
        with self._lock:
            snapshots = list(self._detection_snapshots)
            self._detection_snapshots.clear()
            return snapshots

    def set_jpeg(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg

    def fail(self, error: str) -> None:
        self.error = error
        self.alive = False
        with self._lock:
            self._detection_snapshots.clear()
            self._detections = []


class _SharedLiveProcess:
    """One survng-dls supervisor hosting every live camera pipeline."""

    def __init__(self, command: list[str], *, read_timeout_ms: int) -> None:
        del read_timeout_ms
        self._command = command
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader = MessageReader()
        self._inboxes: dict[str, _StreamInbox] = {}
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._failed = False

    def is_running(self) -> bool:
        return (
            not self._failed and self._process is not None and self._process.poll() is None
            and (self._reader_thread is None or self._reader_thread.is_alive())
        )

    def stderr_text(self) -> str:
        return self._stderr.decode("utf-8", errors="replace").strip()[-400:]

    def start(self) -> None:
        if self.is_running():
            return
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_root, existing) if part
        )
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
            cwd=repo_root,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="dlstreamer-supervisor-stderr",
            daemon=True,
        )
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name="dlstreamer-supervisor-stdout",
            daemon=True,
        )
        self._stderr_thread.start()
        self._reader_thread.start()

    def add_stream(self, stream_id: str, source_url: str, *, source_role: str = "live", frame_width: int | None = None) -> _StreamInbox:
        inbox = _StreamInbox()
        with self._lock:
            self._inboxes[stream_id] = inbox
        command = {"op": "add", "stream_id": stream_id, "url": source_url, "source_role": source_role}
        if frame_width is not None:
            command["frame_width"] = frame_width
        self._send(command)
        return inbox

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            inbox = self._inboxes.pop(stream_id, None)
        if inbox is not None:
            inbox.alive = False
        if self.is_running():
            try:
                self._send({"op": "remove", "stream_id": stream_id})
            except Exception:
                pass

    def close(self) -> None:
        process, self._process = self._process, None
        with self._lock:
            inboxes = list(self._inboxes.values())
            self._inboxes.clear()
        for inbox in inboxes:
            inbox.fail("DL Streamer supervisor closed")
        if process is None:
            return
        if process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        for thread in (self._stderr_thread, self._reader_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        self._stderr_thread = None
        self._reader_thread = None
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _send(self, command: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("DL Streamer supervisor is not running")
        process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
        process.stdin.flush()

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            chunk = process.stderr.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            self._stderr.extend(chunk)
            if len(self._stderr) > 8192:
                del self._stderr[:-8192]

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        failure = "DL Streamer supervisor output ended"
        try:
            while True:
                self._check_inference_progress(time.monotonic())
                if not select.select([process.stdout], [], [], 0.5)[0]:
                    continue
                chunk = process.stdout.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
                if not chunk:
                    break
                self._reader.feed(chunk)
                while True:
                    popped = self._reader.pop()
                    if popped is None:
                        break
                    self._dispatch(*popped)
        except Exception as error:
            detail = redact_secret_text(str(error))[:400]
            failure = f"DL Streamer supervisor failed ({type(error).__name__}): {detail}"
            # One bounded diagnostic per supervisor failure. Keep the native
            # cause beyond the source-file prefix; camera retries stay concise.
            LOGGER.warning(
                "DL Streamer supervisor failed (%s): %s; native stderr: %s",
                type(error).__name__, redact_secret_text(str(error))[-4000:],
                redact_secret_text(self._stderr.decode("utf-8", errors="replace"))[-4000:],
            )
        finally:
            self._failed = True
            with self._lock:
                inboxes = list(self._inboxes.values())
            for inbox in inboxes:
                inbox.fail(failure)

    def _check_inference_progress(self, now: float) -> None:
        with self._lock:
            inboxes = list(self._inboxes.values())
        active = False
        for inbox in inboxes:
            if not inbox.alive or inbox.inference_started_at is None:
                continue
            # An RTSP outage belongs to this camera's read/reconnect lifecycle,
            # not to the shared model. Diagnose a silent inference stall only
            # while the same stream is still delivering video.
            if inbox.last_frame_at is None or now - inbox.last_frame_at > 1.0:
                continue
            active = True
            progress = inbox.last_inference_at
            if progress is None:
                progress = inbox.inference_started_at
            if inbox.video_resumed_at is not None:
                # A camera pause is not time spent awaiting inference on
                # continuous video. Give its first resumed frame the normal
                # bounded inference budget without inventing result progress.
                progress = max(progress, inbox.video_resumed_at)
            if now - progress <= DLSTREAMER_INFERENCE_STALL_SECONDS:
                # One delayed camera is not proof that the shared pool is
                # wedged: other cameras can still complete inference under
                # load. Their progress must prevent a global capture reset.
                # Per-camera snapshot freshness still rejects stale results.
                return
        if active:
            raise RuntimeError("shared live inference stalled: no new result for 5 seconds")

    def _dispatch(self, message_type: int, payload: bytes) -> None:
        if message_type == TYPE_FATAL:
            decoded = decode_json_payload(payload)
            raise RuntimeError(str(decoded.get("error") or "DL Streamer startup failed"))
        stream_id, inner = decode_stream_payload(payload)
        with self._lock:
            inbox = self._inboxes.get(stream_id)
        if inbox is None:
            return
        if message_type == TYPE_FRAME:
            width, height, sequence, pts, pixels = decode_frame_payload(inner)
            pixel_count = width * height
            if len(pixels) == pixel_count:
                frame = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width).copy()
            else:
                frame = (
                    np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
                )
            inbox.put_frame(frame, sequence, pts)
            return
        if message_type == TYPE_JPEG:
            _width, _height, _sequence, _pts, jpeg = decode_jpeg_payload(inner)
            inbox.set_jpeg(jpeg)
            return
        if message_type == TYPE_DETECTIONS:
            decoded = decode_json_payload(inner)
            objects = decoded.get("objects")
            if isinstance(objects, list):
                normalized = [item for item in objects if isinstance(item, dict)]
                inbox.set_detections(normalized)
                if all(key in decoded for key in ("source_pts", "inference_sequence", "width", "height")):
                    try:
                        inbox.add_detection_snapshot(decoded)
                    except ValueError:
                        inbox.status["invalid_detection_snapshots"] = int(inbox.status.get("invalid_detection_snapshots", 0)) + 1
            return
        if message_type == TYPE_STATUS:
            decoded = decode_json_payload(inner)
            inbox.status = decoded
            if decoded.get("ok") is True and decoded.get("detect") is True:
                if inbox.inference_started_at is None:
                    inbox.inference_started_at = time.monotonic()
            error = str(decoded.get("error") or "").strip()
            if decoded.get("ok") is False:
                if decoded.get("failure_scope") == "inference":
                    raise RuntimeError(f"shared live inference failed: {error or 'native stream error'}")
                inbox.fail(error or "DL Streamer stream failed")
            return
        raise RuntimeError(f"unsupported live-capture message type {message_type}")


class DlStreamerCaptureHandle:
    """One isolated live pipeline whose stdout carries gray frames and JPEG."""

    def __init__(self, *, read_timeout_ms: int) -> None:
        self._read_timeout_seconds = max(0.001, read_timeout_ms / 1000.0)
        self._process: subprocess.Popen[bytes] | None = None
        self._reader = MessageReader()
        self._prefetched: np.ndarray | None = None
        self._prefetched_identity: tuple[int, float, str] | None = None
        self._last_frame_identity: tuple[int, float, str] | None = None
        self._parsed_frame_identity: tuple[int, float, str] | None = None
        self._local_inbox = _StreamInbox()
        self.source_role = "live"
        self.frame_width: int | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._status: dict[str, object] = {}
        self._detections: list[dict[str, object]] = []
        self._detections_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._shared: _SharedLiveProcess | None = None
        self._stream_id = ""
        self._inbox: _StreamInbox | None = None

    def is_opened(self) -> bool:
        if self._inbox is not None:
            shared = self._shared
            return (
                self._inbox.alive
                and shared is not None
                and shared.is_running()
            )
        return self._process is not None and self._process.poll() is None

    def set_buffer_size(self, size: int) -> None:
        del size

    def set_source_role(self, source: str) -> None:
        if source not in {"live", "main"}:
            raise ValueError("invalid capture source role")
        self.source_role = source

    def set_frame_width(self, width: int) -> None:
        if not 240 <= width <= 960:
            raise ValueError("capture frame width must be between 240 and 960")
        self.frame_width = int(width)

    def start(self, command: list[str], source_url: str) -> None:
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_root, existing) if part
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
            cwd=repo_root,
        )
        assert self._process.stdin is not None
        self._process.stdin.write(f"{source_url}\n".encode("utf-8"))
        self._process.stdin.close()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="dlstreamer-live-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def attach(
        self,
        shared: _SharedLiveProcess,
        stream_id: str,
        inbox: _StreamInbox,
    ) -> None:
        self._shared = shared
        self._stream_id = stream_id
        self._inbox = inbox

    def prefetch(self, timeout_ms: int, cancelled: Callable[[], bool]) -> bool:
        frame = self._next_frame(
            max(0.001, timeout_ms / 1000.0),
            cancelled=cancelled,
        )
        if frame is None:
            return False
        self._prefetched = frame[0]
        self._prefetched_identity = (frame[1], frame[2], frame[3])
        self._harvest_available_messages()
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._prefetched is not None:
            frame, self._prefetched = self._prefetched, None
            self._last_frame_identity, self._prefetched_identity = self._prefetched_identity, None
            self._harvest_available_messages()
            return True, frame
        received = self._next_frame(self._read_timeout_seconds)
        frame = None if received is None else received[0]
        if received is not None:
            self._harvest_available_messages()
            self._last_frame_identity = (received[1], received[2], received[3])
        return (frame is not None), frame

    def pipeline_status(self) -> dict[str, object]:
        inbox = self._inbox
        if inbox is not None:
            return dict(inbox.status)
        return dict(self._status)

    def pop_detections(self) -> list[dict[str, object]]:
        inbox = self._inbox
        if inbox is not None:
            return inbox.pop_detections()
        with self._detections_lock:
            detections = list(self._detections)
            self._detections = []
        return detections

    def pop_detection_snapshots(self) -> list[DetectionSnapshot]:
        return (self._inbox or self._local_inbox).pop_detection_snapshots()

    def pop_frame_identity(self) -> tuple[int, float, str] | None:
        identity, self._last_frame_identity = self._last_frame_identity, None
        return identity

    def pop_jpeg(self) -> bytes | None:
        inbox = self._inbox
        if inbox is not None:
            return inbox.pop_jpeg()
        with self._detections_lock:
            jpeg, self._jpeg = self._jpeg, None
            return jpeg

    def close(self) -> None:
        shared, stream_id, inbox = self._shared, self._stream_id, self._inbox
        self._shared = None
        self._stream_id = ""
        self._inbox = None
        if shared is not None:
            if inbox is not None:
                self._status = dict(inbox.status)
                if inbox.error:
                    self._status["error"] = str(self._status.get("error") or inbox.error)
                self._stderr = bytearray(shared.stderr_text().encode("utf-8"))
            shared.remove_stream(stream_id)
            return
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        stderr_thread, self._stderr_thread = self._stderr_thread, None
        if stderr_thread is not None:
            stderr_thread.join()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def error_detail(self) -> str:
        inbox = self._inbox
        if inbox is not None:
            shared = self._shared
            status_error = str(inbox.status.get("error") or inbox.error or "").strip()
            detail = shared.stderr_text() if shared is not None else ""
            process = shared._process if shared is not None else None
            return_code = process.poll() if process is not None else None
        else:
            process = self._process
            return_code = process.poll() if process is not None else None
            status_error = str(self._status.get("error") or "").strip()
            detail = self._stderr.decode("utf-8", errors="replace").strip()[-400:]
        parts = [part for part in (status_error, detail) if part]
        combined = ": ".join(parts)
        if return_code is None:
            return combined
        outcome = (
            f"DL Streamer exited from signal {-return_code}"
            if return_code < 0
            else f"DL Streamer exited with status {return_code}"
        )
        return f"{outcome}: {combined}" if combined else outcome

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            chunk = process.stderr.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            self._stderr.extend(chunk)
            if len(self._stderr) > 8192:
                del self._stderr[:-8192]

    def _next_frame(
        self,
        timeout_seconds: float,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, int, float, str] | None:
        inbox = self._inbox
        if inbox is not None:
            return inbox.get_frame(timeout_seconds, cancelled=cancelled)
        process = self._process
        if process is None or process.stdout is None:
            return None
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancelled is not None and cancelled():
                return None
            popped = self._reader.pop()
            if popped is not None:
                frame = self._apply_message(*popped)
                if frame is not None:
                    sequence, pts, session = self._parsed_frame_identity or (0, float("nan"), "")
                    return frame, sequence, pts, session
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select(
                [process.stdout],
                [],
                [],
                min(remaining, CAPTURE_OPEN_LOCK_POLL_SECONDS),
            )
            if not readable:
                continue
            chunk = process.stdout.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return None
            self._reader.feed(chunk)

    def _harvest_available_messages(self) -> None:
        if self._inbox is not None:
            return
        process = self._process
        if process is None or process.stdout is None:
            return
        while True:
            popped = self._reader.pop()
            if popped is not None:
                self._apply_message(*popped)
                continue
            readable, _, _ = select.select([process.stdout], [], [], 0)
            if not readable:
                return
            chunk = process.stdout.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            self._reader.feed(chunk)

    def _apply_message(self, message_type: int, payload: bytes) -> np.ndarray | None:
        if message_type == TYPE_FRAME:
            width, height, sequence, pts, pixels = decode_frame_payload(payload)
            session = self._local_inbox.qualify_pts("frame", pts)
            self._parsed_frame_identity = (sequence, pts, session)
            pixel_count = width * height
            if len(pixels) == pixel_count:
                return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width).copy()
            return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
        if message_type == TYPE_JPEG:
            _width, _height, _sequence, _pts, jpeg = decode_jpeg_payload(payload)
            with self._detections_lock:
                self._jpeg = jpeg
            return None
        if message_type == TYPE_DETECTIONS:
            decoded = decode_json_payload(payload)
            objects = decoded.get("objects")
            if isinstance(objects, list):
                with self._detections_lock:
                    normalized = [item for item in objects if isinstance(item, dict)]
                    self._detections = normalized
                    if all(key in decoded for key in ("source_pts", "inference_sequence", "width", "height")):
                        try:
                            self._local_inbox.add_detection_snapshot(decoded)
                        except ValueError:
                            self._status["invalid_detection_snapshots"] = int(self._status.get("invalid_detection_snapshots", 0)) + 1
            return None
        if message_type == TYPE_STATUS:
            decoded = decode_json_payload(payload)
            self._status = decoded
            error = decoded.get("error")
            if decoded.get("ok") is False and error:
                raise RuntimeError(str(error))
            return None
        raise RuntimeError(f"unsupported live-capture message type {message_type}")


class DlStreamerCaptureBackend:
    """Shared GStreamer live capture with optional in-pipeline gvadetect."""

    def __init__(
        self,
        limiter: CaptureOpenLimiter,
        options: DlStreamerCaptureOptions | None = None,
    ) -> None:
        self.limiter = limiter
        self.options = options or DlStreamerCaptureOptions()
        if self.options.rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be tcp or udp")
        if self.options.decoder not in {"auto", "va"}:
            raise ValueError("decoder must be auto or va")
        self._credential_warning_lock = threading.Lock()
        self._credential_warning_hosts: set[str] = set()
        self._shared: _SharedLiveProcess | None = None
        self._shared_lock = threading.Lock()
        atexit.register(self.close)

    def create_handle(self) -> CaptureHandle:
        return DlStreamerCaptureHandle(read_timeout_ms=self.options.read_timeout_ms)

    @property
    def startup_timeout_ms(self) -> int:
        if self.options.detect_enabled:
            return max(self.options.open_timeout_ms, DLSTREAMER_INFERENCE_STARTUP_TIMEOUT_MS)
        return self.options.open_timeout_ms

    def close(self) -> None:
        with self._shared_lock:
            shared, self._shared = self._shared, None
        if shared is not None:
            shared.close()

    def open(
        self,
        handle: CaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        open_timeout_ms: int | None = None,
    ) -> bool:
        if not isinstance(handle, DlStreamerCaptureHandle):
            raise TypeError("DlStreamerCaptureBackend requires DlStreamerCaptureHandle")
        while not cancelled():
            if not self.limiter.acquire(self.options.admission_poll_seconds):
                continue
            try:
                if cancelled():
                    return False
                timeout_ms = max(
                    1,
                    int(
                        self.startup_timeout_ms
                        if open_timeout_ms is None
                        else open_timeout_ms
                    ),
                )
                self.warn_credentialed_url(source_url)
                command = self.command()
                if "--supervisor" in command:
                    opened = self._open_shared(
                        handle,
                        source_url,
                        cancelled,
                        timeout_ms=timeout_ms,
                    )
                else:
                    handle.start(command, source_url)
                    if cancelled():
                        handle.close()
                        return False
                    opened = handle.prefetch(timeout_ms, cancelled)
                    if not opened:
                        handle.close()
                return opened
            finally:
                self.limiter.release()
        return False

    def _open_shared(
        self,
        handle: DlStreamerCaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        timeout_ms: int,
    ) -> bool:
        with self._shared_lock:
            if self._shared is None or not self._shared.is_running():
                if self._shared is not None:
                    self._shared.close()
                self._shared = _SharedLiveProcess(
                    self.command(),
                    read_timeout_ms=self.options.read_timeout_ms,
                )
                self._shared.start()
            shared = self._shared
        stream_id = uuid.uuid4().hex
        handle.attach(shared, stream_id, shared.add_stream(
            stream_id, source_url, source_role=handle.source_role, frame_width=handle.frame_width,
        ))
        if cancelled():
            handle.close()
            return False
        if not handle.prefetch(timeout_ms, cancelled):
            handle.close()
            return False
        return True

    def command(self) -> list[str]:
        requested_rate = (
            self.options.frame_rate()
            if self.options.frame_rate is not None
            else 5.0
        )
        frame_rate = min(10.0, max(0.5, float(requested_rate)))
        requested_detection_rate = (
            self.options.detection_frame_rate()
            if self.options.detection_frame_rate is not None
            else frame_rate
        )
        detection_rate = min(10.0, max(0.5, float(requested_detection_rate)))
        main_rate = self.options.main_frame_rate() if self.options.main_frame_rate else frame_rate
        open_timeout = max(0.001, self.startup_timeout_ms / 1000.0)
        command = [
            live_python_executable(self.options.python_executable),
            "-m",
            "survng.dlstreamer_live",
            "--fps",
            f"{frame_rate:.6f}",
            "--detect-fps",
            f"{detection_rate:.6f}",
            "--main-fps",
            f"{min(10.0, max(0.5, float(main_rate))):.6f}",
            "--threshold",
            str(self.options.confidence_threshold),
            "--nms-threshold",
            str(self.options.nms_threshold),
            "--open-timeout",
            f"{open_timeout:.3f}",
            "--rtsp-transport",
            self.options.rtsp_transport,
            "--decoder",
            self.options.decoder,
            "--device",
            self.options.inference_device or "GPU",
            "--frame-width",
            str(max(240, min(960, int(self.options.frame_width or 320)))),
            "--jpeg-fps",
            f"{max(0.0, min(5.0, float(self.options.jpeg_fps))):.6f}",
            "--supervisor",
        ]
        model_path = self.options.model_path.strip()
        if self.options.detect_enabled and model_path:
            command.extend(["--model", model_path])
            command.extend(
                [
                    "--model-instance-id",
                    model_instance_id(
                        model_path,
                        self.options.inference_device or "GPU",
                    ),
                ]
            )
            labels_path = self.options.labels_path.strip()
            if labels_path:
                command.extend(["--labels", labels_path])
            elif self.options.labels:
                command.extend(["--labels-list", ",".join(self.options.labels)])
            model_proc = (
                self.options.model_proc_path.strip()
                or adjacent_model_proc(model_path)
            )
            if model_proc:
                command.extend(["--model-proc", model_proc])
        else:
            command.append("--no-detect")
        return command

    def warn_credentialed_url(self, source_url: str) -> None:
        try:
            parsed = urlsplit(source_url)
        except ValueError:
            return
        if parsed.username is None and parsed.password is None:
            return
        host = parsed.hostname or "unknown"
        with self._credential_warning_lock:
            if host in self._credential_warning_hosts:
                return
            self._credential_warning_hosts.add(host)
        LOGGER.warning(
            "camera capture URL for host %s contains credentials; route the "
            "camera through a credential-free go2rtc restream so the isolated "
            "live pipeline does not ingest a passworded URL",
            host,
        )
