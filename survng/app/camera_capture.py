"""Replaceable camera capture backend and shared open admission control."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np


CAPTURE_OPEN_TIMEOUT_MS = 3000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_DECODER_THREADS = 1
CAPTURE_OPEN_LOCK_POLL_SECONDS = 0.1
CAPTURE_OPEN_CONCURRENCY = 2


class CaptureHandle(Protocol):
    def is_opened(self) -> bool: ...

    def set_buffer_size(self, size: int) -> None: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def close(self) -> None: ...


class CaptureBackend(Protocol):
    def create_handle(self) -> CaptureHandle: ...

    def open(
        self,
        handle: CaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
    ) -> bool: ...


class CaptureOpenLimiter:
    """Bound simultaneous native stream opens across all camera workers."""

    def __init__(self, capacity: int = CAPTURE_OPEN_CONCURRENCY) -> None:
        if capacity < 1:
            raise ValueError("capture open capacity must be positive")
        self.capacity = capacity
        self._semaphore = threading.BoundedSemaphore(capacity)

    def acquire(self, timeout: float) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()


class OpenCvCaptureHandle:
    """Small ownership wrapper around one OpenCV native capture handle."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture

    def is_opened(self) -> bool:
        return bool(self._capture.isOpened())

    def set_buffer_size(self, size: int) -> None:
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, size)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        return bool(ok), frame

    def close(self) -> None:
        self._capture.release()

    def open(self, source_url: str, options: list[int]) -> bool:
        return bool(self._capture.open(source_url, cv2.CAP_FFMPEG, options))


@dataclass(frozen=True, slots=True)
class OpenCvCaptureOptions:
    decoder_threads: int = CAPTURE_DECODER_THREADS
    open_timeout_ms: int = CAPTURE_OPEN_TIMEOUT_MS
    read_timeout_ms: int = CAPTURE_READ_TIMEOUT_MS
    admission_poll_seconds: float = CAPTURE_OPEN_LOCK_POLL_SECONDS


class OpenCvFfmpegCaptureBackend:
    """OpenCV capture backend using FFmpeg with bounded native I/O deadlines."""

    def __init__(
        self,
        limiter: CaptureOpenLimiter,
        options: OpenCvCaptureOptions | None = None,
    ) -> None:
        self.limiter = limiter
        self.options = options or OpenCvCaptureOptions()

    def create_handle(self) -> CaptureHandle:
        return OpenCvCaptureHandle(cv2.VideoCapture())

    def open(
        self,
        handle: CaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
    ) -> bool:
        if not isinstance(handle, OpenCvCaptureHandle):
            raise TypeError("OpenCvFfmpegCaptureBackend requires OpenCvCaptureHandle")
        while not cancelled():
            if not self.limiter.acquire(self.options.admission_poll_seconds):
                continue
            try:
                if cancelled():
                    return False
                return handle.open(
                    source_url,
                    [
                        cv2.CAP_PROP_N_THREADS,
                        self.options.decoder_threads,
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        self.options.open_timeout_ms,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        self.options.read_timeout_ms,
                    ],
                )
            finally:
                self.limiter.release()
        return False
