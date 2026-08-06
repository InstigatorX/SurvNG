"""Replaceable camera capture backend and shared open admission control."""

from __future__ import annotations

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

import cv2
import numpy as np

from .security import redact_secret_text


CAPTURE_OPEN_TIMEOUT_MS = 3000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_DECODER_THREADS = 1
CAPTURE_OPEN_LOCK_POLL_SECONDS = 0.1
CAPTURE_OPEN_CONCURRENCY = 2
FRAME_STALE_SECONDS = 10.0
MAIN_SOURCE_IDLE_SECONDS = 20.0
CAPTURE_RETRY_INITIAL_SECONDS = 1.0
CAPTURE_RETRY_MAX_SECONDS = 30.0

LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    source: str
    image: np.ndarray
    captured_at_epoch: float
    captured_at_monotonic: float
    captured_at_iso: str
    width: int
    height: int
    sequence: int


@dataclass(frozen=True, slots=True)
class SourceStartResult:
    running: bool
    started: bool


class CameraCaptureService:
    """Own both capture sources, their latest frames, and native lifecycle."""

    def __init__(
        self,
        *,
        camera_id: str,
        source_url: Callable[[str], str],
        backend: CaptureBackend,
        frame_observer: Callable[[CapturedFrame], None] | None = None,
        source_started_observer: Callable[[str], None] | None = None,
        source_stopped_observer: Callable[[str], None] | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        stale_seconds: float = FRAME_STALE_SECONDS,
        main_idle_seconds: float = MAIN_SOURCE_IDLE_SECONDS,
        retry_initial_seconds: float = CAPTURE_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = CAPTURE_RETRY_MAX_SECONDS,
    ) -> None:
        self.camera_id = camera_id
        self._source_url = source_url
        self.backend = backend
        self._frame_observer = frame_observer
        self._source_started_observer = source_started_observer
        self._source_stopped_observer = source_stopped_observer
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self.stale_seconds = stale_seconds
        self.main_idle_seconds = main_idle_seconds
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stop.set()
        self._threads: dict[str, threading.Thread] = {}
        self._source_stops: dict[str, threading.Event] = {}
        self._all_threads: dict[
            threading.Thread,
            tuple[str, threading.Event],
        ] = {}
        self._frames: dict[str, CapturedFrame] = {}
        self._last_access: dict[str, float] = {}
        self._errors: dict[str, str] = {}
        self._last_live_error = ""
        self._dimensions: dict[str, dict[str, int]] = {}
        self._frame_times: dict[str, deque[float]] = {
            "live": deque(maxlen=600),
            "main": deque(maxlen=600),
        }
        self._stats: dict[str, dict[str, int]] = {
            source: {
                "frames_received": 0,
                "read_failures": 0,
                "open_failures": 0,
                "reconnects": 0,
                "starts": 0,
                "observer_errors": 0,
                "close_failures": 0,
            }
            for source in ("live", "main")
        }
        self._sequence = 0

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source).strip().lower()
        if normalized not in {"live", "main"}:
            raise ValueError(f"unsupported camera capture source: {source!r}")
        return normalized

    def start(self) -> bool:
        with self._lock:
            if self._stop.is_set():
                lingering = [
                    source
                    for thread, (source, _stop_event) in self._all_threads.items()
                    if thread.is_alive()
                ]
                if lingering:
                    raise RuntimeError(
                        f"cannot restart capture for {self.camera_id} while "
                        f"sources are stopping: {', '.join(sorted(lingering))}"
                    )
            self._stop.clear()
        return self.ensure_source("live").running

    def ensure_source(self, source: str) -> SourceStartResult:
        source = self._normalize_source(source)
        with self._lock:
            if self._stop.is_set():
                return SourceStartResult(False, False)
            thread = self._threads.get(source)
            source_stop = self._source_stops.get(source)
            if (
                thread is not None
                and thread.is_alive()
                and source_stop is not None
                and not source_stop.is_set()
            ):
                return SourceStartResult(True, False)
            stop_event = threading.Event()
            start_gate = threading.Event()
            thread = threading.Thread(
                target=self._run_source_when_released,
                args=(source, stop_event, start_gate),
                name=f"camera-{self.camera_id}-{source}",
                daemon=False,
            )
            self._source_stops[source] = stop_event
            self._threads[source] = thread
            self._all_threads[thread] = (source, stop_event)
            try:
                thread.start()
            except BaseException:
                if self._threads.get(source) is thread:
                    self._threads.pop(source, None)
                    self._source_stops.pop(source, None)
                self._all_threads.pop(thread, None)
                raise
        try:
            self._notify_source(self._source_started_observer, source, "started")
        finally:
            start_gate.set()
        return SourceStartResult(True, True)

    def request_frame(self, source: str) -> CapturedFrame | None:
        source = self._normalize_source(source)
        if self._stop.is_set():
            return None
        with self._lock:
            self._last_access[source] = self._monotonic_clock()
        if not self.ensure_source(source).running:
            return None
        return self.latest(source)

    def latest(self, source: str) -> CapturedFrame | None:
        source = self._normalize_source(source)
        now = self._monotonic_clock()
        with self._lock:
            frame = self._frames.get(source)
            if frame is None or now - frame.captured_at_monotonic > self.stale_seconds:
                return None
            return CapturedFrame(
                source=frame.source,
                image=frame.image.copy(),
                captured_at_epoch=frame.captured_at_epoch,
                captured_at_monotonic=frame.captured_at_monotonic,
                captured_at_iso=frame.captured_at_iso,
                width=frame.width,
                height=frame.height,
                sequence=frame.sequence,
            )

    def request_stop(self) -> None:
        self._stop.set()
        with self._lock:
            stops = tuple(
                stop_event
                for _source, stop_event in self._all_threads.values()
            )
        for stop_event in stops:
            stop_event.set()

    def wait_stopped(self, timeout: float) -> dict[str, threading.Thread]:
        deadline = self._monotonic_clock() + max(0.0, timeout)
        with self._lock:
            threads = tuple(
                (source, thread)
                for thread, (source, _stop_event) in self._all_threads.items()
            )
        for _source, thread in threads:
            thread.join(timeout=max(0.0, deadline - self._monotonic_clock()))
        alive: dict[str, threading.Thread] = {}
        for source, thread in threads:
            if not thread.is_alive():
                continue
            label = source
            suffix = 2
            while label in alive:
                label = f"{source}#{suffix}"
                suffix += 1
            alive[label] = thread
        with self._lock:
            for thread, (source, stop_event) in tuple(self._all_threads.items()):
                if thread.is_alive():
                    continue
                self._all_threads.pop(thread, None)
                if (
                    self._threads.get(source) is thread
                    and self._source_stops.get(source) is stop_event
                ):
                    self._threads.pop(source, None)
                    self._source_stops.pop(source, None)
            if not alive:
                self._frames.clear()
                self._last_access.clear()
                self._errors.clear()
                self._last_live_error = ""
                for times in self._frame_times.values():
                    times.clear()
        return alive

    def close(self) -> None:
        with self._lock:
            alive = {
                source: thread
                for thread, (source, _stop_event) in self._all_threads.items()
                if thread.is_alive()
            }
        if alive:
            raise RuntimeError(
                f"capture sources still running for {self.camera_id}: "
                f"{', '.join(sorted(alive))}"
            )

    def status(self) -> dict[str, object]:
        now = self._monotonic_clock()
        with self._lock:
            capture_stats: dict[str, dict[str, int | float]] = {}
            for source in ("live", "main"):
                times = self._frame_times[source]
                while times and now - times[0] > 10.0:
                    times.popleft()
                fps = (
                    (len(times) - 1) / max(0.001, times[-1] - times[0])
                    if len(times) >= 2 and now - times[-1] <= self.stale_seconds
                    else 0.0
                )
                capture_stats[source] = {**self._stats[source], "fps": round(fps, 2)}
            live = self._frames.get("live")
            main = self._frames.get("main")
            return {
                "live_running": self._thread_running_locked("live"),
                "main_running": self._thread_running_locked("main"),
                "live_frame_at": live.captured_at_iso if live is not None else "",
                "main_frame_at": main.captured_at_iso if main is not None else "",
                "live_frame_monotonic": (
                    live.captured_at_monotonic if live is not None else None
                ),
                "main_frame_monotonic": (
                    main.captured_at_monotonic if main is not None else None
                ),
                "last_error": self._last_live_error,
                "main_error": self._errors.get("main", ""),
                "stream_dimensions": {
                    source: dict(dimensions)
                    for source, dimensions in self._dimensions.items()
                },
                "capture_stats": capture_stats,
            }

    def source_is_idle(self, source: str) -> bool:
        source = self._normalize_source(source)
        if source == "live":
            return False
        with self._lock:
            return self._source_is_idle_locked(source, self._monotonic_clock())

    def threads(self) -> dict[str, threading.Thread]:
        with self._lock:
            return dict(self._threads)

    def _thread_running_locked(self, source: str) -> bool:
        thread = self._threads.get(source)
        return bool(thread is not None and thread.is_alive())

    def _source_is_idle_locked(self, source: str, now: float) -> bool:
        last_access = self._last_access.get(source)
        return last_access is None or now - last_access >= self.main_idle_seconds

    def _should_exit_for_idle(
        self,
        source: str,
        stop_event: threading.Event,
    ) -> bool:
        if source == "live":
            return False
        with self._lock:
            idle = self._source_is_idle_locked(source, self._monotonic_clock())
            if idle and self._source_stops.get(source) is stop_event:
                # A concurrent request sees this source as stopping and creates
                # a replacement instead of returning a thread about to exit.
                stop_event.set()
            return idle

    def _run_source(self, source: str, stop_event: threading.Event) -> None:
        retry_delay = self.retry_initial_seconds
        try:
            while not self._cancelled(stop_event):
                if self._should_exit_for_idle(source, stop_event):
                    return
                failure_reason = ""
                handle: CaptureHandle | None = None
                try:
                    handle = self.backend.create_handle()
                    opened = self.backend.open(
                        handle,
                        self._source_url(source),
                        lambda: self._cancelled(stop_event),
                    )
                    if self._cancelled(stop_event):
                        return
                    if not opened or not handle.is_opened():
                        failure_reason = "failed to open stream"
                        self._increment(source, "open_failures")
                        self._set_error(source, failure_reason)
                    else:
                        handle.set_buffer_size(1)
                        with self._lock:
                            stats = self._stats[source]
                            if stats["frames_received"] > 0:
                                stats["reconnects"] += 1
                            stats["starts"] += 1
                        self._set_error(source, "")
                        while not self._cancelled(stop_event):
                            if self._should_exit_for_idle(source, stop_event):
                                return
                            ok, image = handle.read()
                            if not ok or image is None:
                                if self._cancelled(stop_event):
                                    break
                                failure_reason = "stream read failed"
                                self._increment(source, "read_failures")
                                self._set_error(source, failure_reason)
                                break
                            if self._cancelled(stop_event):
                                break
                            retry_delay = self.retry_initial_seconds
                            self._publish_frame(source, image, stop_event)
                except Exception as exc:
                    failure_reason = f"stream error: {redact_secret_text(exc)[:160]}"
                    self._set_error(source, failure_reason)
                    LOGGER.warning(
                        "camera stream failed for %s/%s: %s",
                        self.camera_id,
                        source,
                        failure_reason,
                    )
                finally:
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception as exc:
                            self._increment(source, "close_failures")
                            LOGGER.warning(
                                "camera stream close failed for %s/%s: %s",
                                self.camera_id,
                                source,
                                redact_secret_text(exc)[:160],
                            )
                if self._cancelled(stop_event) or self._should_exit_for_idle(
                    source, stop_event
                ):
                    break
                LOGGER.info(
                    "camera=%s source=%s retry_delay=%s failure_reason=%s",
                    self.camera_id,
                    source,
                    retry_delay,
                    failure_reason,
                )
                wait_delay = retry_delay
                if source != "live":
                    with self._lock:
                        last_access = self._last_access.get(source)
                    if last_access is not None:
                        idle_in = max(
                            0.0,
                            self.main_idle_seconds
                            - (self._monotonic_clock() - last_access),
                        )
                        wait_delay = min(wait_delay, idle_in)
                if stop_event.wait(wait_delay):
                    break
                retry_delay = min(retry_delay * 2.0, self.retry_max_seconds)
        finally:
            self._source_finished(source, stop_event)

    def _run_source_when_released(
        self,
        source: str,
        stop_event: threading.Event,
        start_gate: threading.Event,
    ) -> None:
        start_gate.wait()
        self._run_source(source, stop_event)

    def _cancelled(self, stop_event: threading.Event) -> bool:
        return self._stop.is_set() or stop_event.is_set()

    def _publish_frame(
        self,
        source: str,
        image: np.ndarray,
        stop_event: threading.Event | None = None,
    ) -> bool:
        captured_at_epoch = self._wall_clock()
        captured_at_monotonic = self._monotonic_clock()
        captured_at_iso = datetime.fromtimestamp(
            captured_at_epoch, timezone.utc
        ).isoformat()
        with self._lock:
            if self._stop.is_set() or (
                stop_event is not None and stop_event.is_set()
            ):
                return False
            self._sequence += 1
            frame = CapturedFrame(
                source=source,
                image=image.copy(),
                captured_at_epoch=captured_at_epoch,
                captured_at_monotonic=captured_at_monotonic,
                captured_at_iso=captured_at_iso,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                sequence=self._sequence,
            )
            self._frames[source] = frame
            self._dimensions[source] = {
                "width": frame.width,
                "height": frame.height,
            }
            self._frame_times[source].append(captured_at_monotonic)
            self._stats[source]["frames_received"] += 1
        if self._frame_observer is not None:
            try:
                self._frame_observer(CapturedFrame(
                    source=frame.source,
                    image=image,
                    captured_at_epoch=frame.captured_at_epoch,
                    captured_at_monotonic=frame.captured_at_monotonic,
                    captured_at_iso=frame.captured_at_iso,
                    width=frame.width,
                    height=frame.height,
                    sequence=frame.sequence,
                ))
            except Exception:
                failures = self._increment(source, "observer_errors")
                if failures == 1 or failures % 100 == 0:
                    LOGGER.exception(
                        "capture frame observer failed for %s/%s "
                        "(failures=%s)",
                    self.camera_id,
                    source,
                    failures,
                )
        return True

    def _source_finished(self, source: str, stop_event: threading.Event) -> None:
        current = threading.current_thread()
        owns_source = False
        with self._lock:
            self._all_threads.pop(current, None)
            if (
                self._threads.get(source) is current
                and self._source_stops.get(source) is stop_event
            ):
                owns_source = True
                self._threads.pop(source, None)
                self._source_stops.pop(source, None)
                self._last_access.pop(source, None)
                if source != "live":
                    self._frames.pop(source, None)
                    self._errors.pop(source, None)
        if owns_source:
            self._notify_source(self._source_stopped_observer, source, "stopped")

    def _set_error(self, source: str, message: str) -> None:
        message = redact_secret_text(message)
        with self._lock:
            self._errors[source] = message
            if source == "live":
                self._last_live_error = message

    def _increment(self, source: str, key: str) -> int:
        with self._lock:
            self._stats[source][key] += 1
            return self._stats[source][key]

    def _notify_source(
        self,
        observer: Callable[[str], None] | None,
        source: str,
        action: str,
    ) -> None:
        if observer is None:
            return
        try:
            observer(source)
        except Exception:
            LOGGER.exception(
                "capture source %s observer failed for %s/%s",
                action,
                self.camera_id,
                source,
            )
