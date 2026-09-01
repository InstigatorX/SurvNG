"""Live capture through an isolated DL Streamer / GStreamer child process."""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import threading
import time
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
    TYPE_FRAME,
    TYPE_STATUS,
    MessageReader,
    decode_frame_payload,
    decode_json_payload,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DlStreamerCaptureOptions:
    python_executable: str = ""
    decoder: str = "va"
    open_timeout_ms: int = CAPTURE_OPEN_TIMEOUT_MS
    read_timeout_ms: int = CAPTURE_READ_TIMEOUT_MS
    admission_poll_seconds: float = CAPTURE_OPEN_LOCK_POLL_SECONDS
    rtsp_transport: str = "tcp"
    frame_rate: Callable[[], float] | None = None
    model_path: str = ""
    inference_device: str = "GPU"
    detect_enabled: bool = False


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


class DlStreamerCaptureHandle:
    """One isolated live pipeline whose stdout carries framed BGR messages."""

    def __init__(self, *, read_timeout_ms: int) -> None:
        self._read_timeout_seconds = max(0.001, read_timeout_ms / 1000.0)
        self._process: subprocess.Popen[bytes] | None = None
        self._reader = MessageReader()
        self._prefetched: np.ndarray | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._status: dict[str, object] = {}
        self._detections: list[dict[str, object]] = []
        self._detections_lock = threading.Lock()

    def is_opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def set_buffer_size(self, size: int) -> None:
        del size

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

    def prefetch(self, timeout_ms: int, cancelled: Callable[[], bool]) -> bool:
        frame = self._next_frame(
            max(0.001, timeout_ms / 1000.0),
            cancelled=cancelled,
        )
        if frame is None:
            return False
        self._prefetched = frame
        self._harvest_available_messages()
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._prefetched is not None:
            frame, self._prefetched = self._prefetched, None
            self._harvest_available_messages()
            return True, frame
        frame = self._next_frame(self._read_timeout_seconds)
        if frame is not None:
            self._harvest_available_messages()
        return (frame is not None), frame

    def pop_detections(self) -> list[dict[str, object]]:
        with self._detections_lock:
            detections = list(self._detections)
            self._detections = []
        return detections

    def close(self) -> None:
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
            chunk = process.stderr.read(CAPTURE_PIPE_READ_CHUNK_BYTES)
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
    ) -> np.ndarray | None:
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
                    return frame
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
            width, height, _sequence, _pts, pixels = decode_frame_payload(payload)
            return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()
        if message_type == TYPE_DETECTIONS:
            decoded = decode_json_payload(payload)
            objects = decoded.get("objects")
            if isinstance(objects, list):
                with self._detections_lock:
                    self._detections = [item for item in objects if isinstance(item, dict)]
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
    """Isolated GStreamer live capture with optional in-pipeline gvadetect."""

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

    def create_handle(self) -> CaptureHandle:
        return DlStreamerCaptureHandle(read_timeout_ms=self.options.read_timeout_ms)

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
                        self.options.open_timeout_ms
                        if open_timeout_ms is None
                        else open_timeout_ms
                    ),
                )
                self.warn_credentialed_url(source_url)
                handle.start(self.command(), source_url)
                if cancelled():
                    handle.close()
                    return False
                if not handle.prefetch(timeout_ms, cancelled):
                    handle.close()
                    return False
                return True
            finally:
                self.limiter.release()
        return False

    def command(self) -> list[str]:
        requested_rate = (
            self.options.frame_rate()
            if self.options.frame_rate is not None
            else 5.0
        )
        frame_rate = min(10.0, max(0.5, float(requested_rate)))
        open_timeout = max(0.001, self.options.open_timeout_ms / 1000.0)
        command = [
            live_python_executable(self.options.python_executable),
            "-m",
            "survng.dlstreamer_live",
            "--fps",
            f"{frame_rate:.6f}",
            "--open-timeout",
            f"{open_timeout:.3f}",
            "--rtsp-transport",
            self.options.rtsp_transport,
            "--decoder",
            self.options.decoder,
            "--device",
            self.options.inference_device or "GPU",
        ]
        model_path = self.options.model_path.strip()
        if self.options.detect_enabled and model_path:
            command.extend(["--model", model_path])
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
