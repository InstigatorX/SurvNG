"""Shared DL Streamer child-process transport and per-stream inbox state."""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import select
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np

from .camera_capture import CAPTURE_PIPE_READ_CHUNK_BYTES
from .dlstreamer_protocol import (
    PROTOCOL_FD_ENV,
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
from .live_detections import DetectionSnapshot
from .redact import redact_secret_text
from survng.dlstreamer_live import STREAM_STOP_TIMEOUT_SECONDS

LOGGER = logging.getLogger(__name__)
DLSTREAMER_INFERENCE_STALL_SECONDS = 5.0


def _diagnostic_tail(stderr: bytearray, stdout: bytearray, limit: int = 400) -> str:
    # Sanitize channels independently: native writes can split credentials
    # across chunks and must never be spliced together with another channel.
    return "\n".join(part for part in (
        _safe_stderr_tail(stderr, limit), _safe_stderr_tail(stdout, limit),
    ) if part)


def _start_child(
    command: list[str], env: dict[str, str], cwd: str,
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    """Own the data pipe separately from both native diagnostic streams."""
    read_fd, write_fd = os.pipe()
    try:
        child_env = dict(env)
        child_env[PROTOCOL_FD_ENV] = str(write_fd)
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=child_env, pass_fds=(write_fd,),
            start_new_session=True, cwd=cwd,
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    return process, os.fdopen(read_fd, "rb")


def _safe_stderr_tail(buffer: bytearray, limit: int = 400) -> str:
    """Redact complete lines before truncation; partial URLs cannot be scrubbed."""
    raw = bytes(buffer)
    if len(raw) >= 8192:
        # The rolling buffer may begin halfway through a credential.
        raw = raw.partition(b"\n")[2]
    # A pipe read may also end halfway through a credential-bearing URL.
    raw = raw.rpartition(b"\n")[0]
    if not raw:
        return "[partial native stderr omitted]" if buffer else ""
    return redact_secret_text(raw.decode("utf-8", errors="replace")).strip()[-limit:]


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
            # A new video frame reusing PTS cannot inherit the previous frame's
            # detection. Repeated metadata alone still does not reset liveness.
            if previous is not None and math.isfinite(pts) and (
                pts < previous or (kind == "frame" and pts == previous)
            ):
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

    def __init__(
        self,
        command: list[str],
        *,
        read_timeout_ms: int,
        inference_stall_seconds: float = DLSTREAMER_INFERENCE_STALL_SECONDS,
    ) -> None:
        del read_timeout_ms
        self._command = command
        self._inference_stall_seconds = max(
            0.001,
            float(inference_stall_seconds),
        )
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._output: BinaryIO | None = None
        self._stdout_thread: threading.Thread | None = None
        self._reader = MessageReader()
        self._inboxes: dict[str, _StreamInbox] = {}
        self._stderr = bytearray()
        self._native_stdout = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._failed = False
        self._generation = uuid.uuid4().hex

    def is_running(self) -> bool:
        return (
            not self._failed and self._process is not None and self._process.poll() is None
            and (self._reader_thread is None or self._reader_thread.is_alive())
        )

    def stderr_text(self) -> str:
        return _diagnostic_tail(self._stderr, self._native_stdout)

    def start(self) -> None:
        if self.is_running():
            return
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (repo_root, existing) if part
        )
        self._process, self._output = _start_child(self._command, env, repo_root)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="dlstreamer-supervisor-stderr",
            daemon=True,
        )
        self._reader_thread = threading.Thread(
            target=self._read_protocol,
            name="dlstreamer-supervisor-protocol",
            daemon=True,
        )
        self._stdout_thread = threading.Thread(
            target=self._drain_stderr, args=(self._process.stdout, self._native_stdout),
            name="dlstreamer-native-stdout", daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._reader_thread.start()

    def add_stream(self, stream_id: str, source_url: str, *, source_role: str = "live", frame_width: int | None = None, detection_enabled: bool = True, h264_decoder_compliance: str = "auto", spatial_plan: dict | None = None) -> _StreamInbox:
        inbox = _StreamInbox()
        with self._lock:
            self._inboxes[stream_id] = inbox
        command = {"op": "add", "stream_id": stream_id, "url": source_url, "source_role": source_role, "detection_enabled": detection_enabled}
        command["h264_decoder_compliance"] = h264_decoder_compliance
        if spatial_plan is not None:
            command["spatial_plan"] = spatial_plan
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
            # EOF lets the supervisor leave its blocking command read. Avoid
            # waiting on a writer if a wedged child has filled the input pipe.
            if self._command_lock.acquire(blocking=False):
                try:
                    if process.stdin is not None:
                        try:
                            process.stdin.close()
                        except (OSError, ValueError):
                            pass
                finally:
                    self._command_lock.release()
            process.terminate()
            try:
                process.wait(timeout=STREAM_STOP_TIMEOUT_SECONDS + 1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        # Reap before closing buffered stdin: an in-flight command write can
        # own its lock while the native child has stopped reading commands.
        if process.stdin is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        for thread in (self._stderr_thread, self._stdout_thread, self._reader_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        self._stderr_thread = None
        self._stdout_thread = None
        self._reader_thread = None
        for stream in (self._output, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._output = None

    def _send(self, command: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("DL Streamer supervisor is not running")
        with self._command_lock:
            process.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
            process.stdin.flush()

    def _drain_stderr(self, stream=None, buffer=None) -> None:
        process = self._process
        if stream is None:
            stream = process.stderr if process is not None else None
        if stream is None:
            return
        if buffer is None:
            buffer = self._stderr
        while True:
            chunk = stream.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
            if not chunk:
                return
            buffer.extend(chunk)
            if len(buffer) > 8192:
                del buffer[:-8192]

    def _read_protocol(self) -> None:
        process = self._process
        if process is None or self._output is None:
            return
        failure = "DL Streamer supervisor output ended"
        try:
            while True:
                self._check_inference_progress(time.monotonic())
                if not select.select([self._output], [], [], 0.5)[0]:
                    continue
                chunk = self._output.read1(CAPTURE_PIPE_READ_CHUNK_BYTES)
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
                "DL Streamer supervisor failed (%s): %s; native diagnostics: %s; "
                "generation=%s affected_streams=%d",
                type(error).__name__, redact_secret_text(str(error))[-4000:],
                _diagnostic_tail(self._stderr, self._native_stdout, 2000),
                self._generation, len(self._inboxes),
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
        stalled = False
        for inbox in inboxes:
            if not inbox.alive:
                continue
            # A real completion proves the pool is working even if that
            # camera's video has paused or its first status has not arrived.
            # Startup/resume grace alone is not evidence of pool progress.
            if (
                inbox.last_inference_at is not None
                and now - inbox.last_inference_at <= self._inference_stall_seconds
            ):
                return
            if inbox.inference_started_at is None:
                continue
            # An RTSP outage belongs to this camera's read/reconnect lifecycle,
            # not to the shared model. Diagnose a silent inference stall only
            # while the same stream is still delivering video.
            if inbox.last_frame_at is None or now - inbox.last_frame_at > 1.0:
                continue
            progress = inbox.last_inference_at
            if progress is None:
                progress = inbox.inference_started_at
            if inbox.video_resumed_at is not None:
                # A camera pause is not time spent awaiting inference on
                # continuous video. Give its first resumed frame the normal
                # bounded inference budget without inventing result progress.
                progress = max(progress, inbox.video_resumed_at)
            if now - progress > self._inference_stall_seconds:
                stalled = True
        # Grace for a newly added/resumed stream must not keep an already
        # stalled, continuously active pool alive indefinitely through churn.
        if stalled:
            raise RuntimeError("shared live inference stalled: no new result for 5 seconds")

    def _dispatch(self, message_type: int, payload: bytes) -> None:
        if message_type == TYPE_FATAL:
            decoded = decode_json_payload(payload)
            raise RuntimeError(str(decoded.get("error") or "DL Streamer startup failed"))
        # Keep the stream/frame header slices as views until the final owned
        # NumPy allocation. Native-resolution BGR payloads can be many MiB.
        stream_id, inner = decode_stream_payload(memoryview(payload) if message_type == TYPE_FRAME else payload)
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
            if "invalid_detection_snapshots" in inbox.status:
                decoded["invalid_detection_snapshots"] = inbox.status["invalid_detection_snapshots"]
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


