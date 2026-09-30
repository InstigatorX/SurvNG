"""Experimental GStreamer capture backend.

FFmpeg remains the production capture path. This backend is selected only
when capture_backend is gstreamer. It decodes in software. A persistent live
hardware decoder would hold the same render node recorded evidence frames use.
"""

from __future__ import annotations

import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .camera_capture import CAPTURE_FRAME_MAX_BYTES, CaptureOpenLimiter
from .gstreamer_capture_worker import FRAME_MAGIC
from .security import redact_secret_text


_HEADER_BYTES = len(FRAME_MAGIC) + 8
_STDERR_TAIL_BYTES = 4096


@dataclass(frozen=True, slots=True)
class GStreamerCaptureOptions:
    python_path: str = "/usr/bin/python3"
    open_timeout_ms: int = 3000
    read_timeout_ms: int = 5000
    admission_poll_seconds: float = 0.1
    rtsp_transport: str = "tcp"
    frame_rate: Callable[[], float] | None = None
    # Live hardware decode stays off. See the module docstring.
    hardware_acceleration: str = "off"

    def __post_init__(self) -> None:
        if self.rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be tcp or udp")
        if self.hardware_acceleration != "off":
            raise ValueError(
                "gstreamer live capture stays on software decode"
            )


class GStreamerCaptureBackend:
    """One GStreamer process per source, delivering caller-owned bgr24."""

    def __init__(
        self,
        limiter: CaptureOpenLimiter,
        options: GStreamerCaptureOptions | None = None,
    ) -> None:
        self.limiter = limiter
        self.options = options or GStreamerCaptureOptions()

    def create_handle(self) -> GStreamerCaptureHandle:
        return GStreamerCaptureHandle(
            read_timeout_ms=self.options.read_timeout_ms,
        )

    def open(
        self,
        handle: object,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        open_timeout_ms: int | None = None,
    ) -> bool:
        if not isinstance(handle, GStreamerCaptureHandle):
            raise TypeError("GStreamerCaptureBackend requires GStreamerCaptureHandle")
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
                handle.close()
                handle.start(self._command(source_url))
                if cancelled():
                    handle.close()
                    return False
                return handle.prefetch(timeout_ms, cancelled)
            finally:
                self.limiter.release()
        return False

    def _command(self, source_url: str) -> list[str]:
        requested = (
            self.options.frame_rate()
            if self.options.frame_rate is not None
            else 5.0
        )
        # videorate max-rate is an integer. Motion sample rates are at least 2.
        frame_rate = max(1, min(10, int(round(min(10.0, max(0.5, float(requested)))))))
        worker = Path(__file__).with_name("gstreamer_capture_worker.py")
        return [
            self.options.python_path,
            str(worker),
            source_url,
            str(frame_rate),
            self.options.rtsp_transport,
        ]


class GStreamerCaptureHandle:
    """Read packed BGR frames from one GStreamer worker process."""

    def __init__(self, *, read_timeout_ms: int) -> None:
        self._read_timeout_seconds = max(0.001, read_timeout_ms / 1000.0)
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._prefetched: np.ndarray | None = None
        self._failure = ""
        self.decode_plan = ""
        self._lock = threading.Lock()

    def is_opened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def set_buffer_size(self, size: int) -> None:
        # appsink keeps one buffer and drops the older one. The worker queue
        # does the same while stdout is blocked.
        del size

    def start(self, command: list[str]) -> None:
        self.close()
        self._failure = ""
        self._stderr = bytearray()
        self.decode_plan = "gstreamer-cpu"
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            self._failure = redact_secret_text(str(error))
            self._process = None
            return
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            name="gstreamer-capture-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def prefetch(self, timeout_ms: int, cancelled: Callable[[], bool]) -> bool:
        frame = self._read_frame(timeout_ms / 1000.0, cancelled)
        if frame is None:
            if self._failure in {"", "timed out reading a frame"}:
                self._failure = "timed out connecting"
            self.close()
            return False
        self._prefetched = frame
        return self.is_opened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._prefetched is not None:
            frame = self._prefetched
            self._prefetched = None
            return True, frame
        frame = self._read_frame(self._read_timeout_seconds, lambda: False)
        if frame is None:
            return False, None
        return True, frame

    def close(self) -> None:
        process = self._process
        self._process = None
        self._prefetched = None
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, 15)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()
        thread = self._stderr_thread
        if thread is not None:
            thread.join(timeout=1.0)
        detail = self.error_detail()
        if process.returncode not in {0, None, -15} and not self._failure:
            self._failure = detail or f"gstreamer exited {process.returncode}"

    def error_detail(self) -> str:
        with self._lock:
            tail = redact_secret_text(bytes(self._stderr).decode("utf-8", "replace")).strip()
        if self._failure:
            return self._failure
        if tail:
            return tail[-400:]
        return ""

    def _drain_stderr(self, pipe) -> None:
        try:
            while True:
                chunk = pipe.read(1024)
                if not chunk:
                    break
                with self._lock:
                    self._stderr.extend(chunk)
                    if len(self._stderr) > _STDERR_TAIL_BYTES:
                        del self._stderr[:-_STDERR_TAIL_BYTES]
        finally:
            pipe.close()

    def _read_frame(
        self,
        timeout_seconds: float,
        cancelled: Callable[[], bool],
    ) -> np.ndarray | None:
        process = self._process
        if process is None or process.stdout is None:
            self._failure = self._failure or "gstreamer capture is not open"
            return None
        fd = process.stdout.fileno()
        deadline = time.monotonic() + timeout_seconds
        header = self._read_exact(fd, _HEADER_BYTES, deadline, cancelled)
        if header is None:
            return None
        if header[: len(FRAME_MAGIC)] != FRAME_MAGIC:
            self._failure = "gstreamer frame header did not match"
            return None
        width = int.from_bytes(header[4:8], "little")
        height = int.from_bytes(header[8:12], "little")
        payload_bytes = width * height * 3
        if width < 1 or height < 1 or payload_bytes > CAPTURE_FRAME_MAX_BYTES:
            self._failure = "gstreamer frame geometry is invalid"
            return None
        payload = self._read_exact(fd, payload_bytes, deadline, cancelled)
        if payload is None:
            return None
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(frame)

    def _read_exact(
        self,
        fd: int,
        size: int,
        deadline: float,
        cancelled: Callable[[], bool],
    ) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            if cancelled():
                self._failure = self._failure or "capture cancelled"
                return None
            wait = deadline - time.monotonic()
            if wait <= 0:
                self._failure = self._failure or "timed out reading a frame"
                return None
            readable, _, _ = select.select([fd], [], [], min(0.2, wait))
            if not readable:
                continue
            try:
                chunk = os.read(fd, remaining)
            except OSError as error:
                self._failure = redact_secret_text(str(error))
                return None
            if not chunk:
                self._failure = self._failure or "gstreamer capture ended"
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
