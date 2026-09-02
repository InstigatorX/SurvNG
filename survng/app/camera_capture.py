"""Replaceable camera capture backend and shared open admission control."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

import numpy as np
import cv2

from .security import redact_secret_text


CAPTURE_OPEN_TIMEOUT_MS = 3000
CAPTURE_RECONNECT_OPEN_TIMEOUT_MS = 10000
CAPTURE_READ_TIMEOUT_MS = 5000
CAPTURE_OPEN_LOCK_POLL_SECONDS = 0.1
CAPTURE_OPEN_CONCURRENCY = 2
FRAME_STALE_SECONDS = 10.0
MAIN_SOURCE_IDLE_SECONDS = 20.0
CAPTURE_RETRY_INITIAL_SECONDS = 1.0
CAPTURE_RETRY_MAX_SECONDS = 30.0
CAPTURE_FRAME_MAX_BYTES = 256 * 1024 * 1024
CAPTURE_PIPE_READ_CHUNK_BYTES = 64 * 1024

LOGGER = logging.getLogger(__name__)


class CaptureHandle(Protocol):
    """Native decoder handle whose successful reads transfer frame ownership."""

    def is_opened(self) -> bool: ...

    def set_buffer_size(self, size: int) -> None: ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return a frame the caller exclusively owns and may retain."""
        ...

    def close(self) -> None: ...


class CaptureBackend(Protocol):
    def create_handle(self) -> CaptureHandle: ...

    def open(
        self,
        handle: CaptureHandle,
        source_url: str,
        cancelled: Callable[[], bool],
        *,
        open_timeout_ms: int | None = None,
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


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """Timestamped frame; shared instances expose a read-only image array."""

    source: str
    image: np.ndarray
    captured_at_epoch: float
    captured_at_monotonic: float
    captured_at_iso: str
    width: int
    height: int
    sequence: int
    generation: int = 0


@dataclass(frozen=True, slots=True)
class SourceStartResult:
    running: bool
    started: bool


class LatestFrameObserver:
    """Run one potentially slow observer behind a latest-only mailbox."""

    def __init__(
        self,
        *,
        camera_id: str,
        observer: Callable[[CapturedFrame], None],
        on_result: Callable[[str, float, float, BaseException | None], None],
        monotonic_clock: Callable[[], float],
    ) -> None:
        self.camera_id = camera_id
        self._observer = observer
        self._on_result = on_result
        self._monotonic_clock = monotonic_clock
        self._condition = threading.Condition()
        self._stop = False
        # Keep one latest frame per source. A single shared slot allowed a busy
        # main stream to replace the live frame needed by motion analysis (and
        # vice versa) before the observer could route either frame.
        self._pending: dict[str, CapturedFrame] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                if self._stop:
                    raise RuntimeError(
                        f"capture observer is still stopping for {self.camera_id}"
                    )
                return
            self._stop = False
            self._pending.clear()
            thread = threading.Thread(
                target=self._run,
                name=f"camera-{self.camera_id}-observer",
                daemon=False,
            )
            self._thread = thread
            thread.start()

    def submit(self, frame: CapturedFrame) -> tuple[bool, bool]:
        with self._condition:
            if self._stop or self._thread is None or not self._thread.is_alive():
                return False, False
            replaced = frame.source in self._pending
            self._pending[frame.source] = frame
            self._condition.notify()
            return True, replaced

    def request_stop(self) -> None:
        with self._condition:
            self._stop = True
            self._pending.clear()
            self._condition.notify_all()

    def wait_stopped(self, timeout: float) -> bool:
        with self._condition:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout=max(0.0, timeout))
        stopped = not thread.is_alive()
        if stopped:
            with self._condition:
                if self._thread is thread:
                    self._thread = None
        return stopped

    def running(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def pending(self) -> bool:
        with self._condition:
            return bool(self._pending)

    def thread(self) -> threading.Thread | None:
        with self._condition:
            return self._thread

    def close(self) -> None:
        if self.running():
            raise RuntimeError(
                f"capture observer is still running for {self.camera_id}"
            )

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        return
                    # Dict insertion order provides simple fairness: replacing
                    # one source keeps its position, while a newly pending
                    # source is queued behind sources already waiting.
                    source = next(iter(self._pending))
                    frame = self._pending.pop(source)
                started = self._monotonic_clock()
                wait_ms = max(
                    0.0,
                    (started - frame.captured_at_monotonic) * 1000.0,
                )
                error: BaseException | None = None
                try:
                    self._observer(frame)
                except BaseException as exc:
                    error = exc
                duration_ms = max(
                    0.0,
                    (self._monotonic_clock() - started) * 1000.0,
                )
                self._on_result(frame.source, wait_ms, duration_ms, error)
        finally:
            with self._condition:
                # Preserve the mailbox container across generations. start()
                # clears it before launching the replacement thread.
                self._pending.clear()


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
        initial_open_timeout_ms: int = CAPTURE_OPEN_TIMEOUT_MS,
        reconnect_open_timeout_ms: int = CAPTURE_RECONNECT_OPEN_TIMEOUT_MS,
    ) -> None:
        self.camera_id = camera_id
        self._source_url = source_url
        self.backend = backend
        self._source_started_observer = source_started_observer
        self._source_stopped_observer = source_stopped_observer
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self.stale_seconds = stale_seconds
        self.main_idle_seconds = main_idle_seconds
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds
        self.initial_open_timeout_ms = max(1, int(initial_open_timeout_ms))
        self.reconnect_open_timeout_ms = max(
            self.initial_open_timeout_ms,
            int(reconnect_open_timeout_ms),
        )
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
        self._detections: dict[str, list[dict[str, object]]] = {}
        self._pipeline_status: dict[str, dict[str, object]] = {}
        self._jpegs: dict[str, bytes] = {}
        self._preview: dict[str, np.ndarray] = {}
        self._last_access: dict[str, float] = {}
        self._errors: dict[str, str] = {}
        self._last_live_error = ""
        self._dimensions: dict[str, dict[str, int]] = {}
        self._frame_times: dict[str, deque[float]] = {
            "live": deque(maxlen=600),
            "main": deque(maxlen=600),
        }
        self._observer_durations_ms: dict[str, deque[float]] = {
            "live": deque(maxlen=600),
            "main": deque(maxlen=600),
        }
        self._observer_waits_ms: dict[str, deque[float]] = {
            "live": deque(maxlen=600),
            "main": deque(maxlen=600),
        }
        self._observer_dispatch = (
            LatestFrameObserver(
                camera_id=camera_id,
                observer=frame_observer,
                on_result=self._record_observer_result,
                monotonic_clock=monotonic_clock,
            )
            if frame_observer is not None
            else None
        )
        self._stats: dict[str, dict[str, int | float]] = {
            source: {
                "frames_received": 0,
                "read_failures": 0,
                "open_failures": 0,
                "open_timeout_escalations": 0,
                "last_open_timeout_ms": 0,
                "reconnects": 0,
                "starts": 0,
                "observer_errors": 0,
                "close_failures": 0,
                "frame_copy_count": 0,
                "frame_copy_bytes": 0,
                "frame_transfer_count": 0,
                "frame_transfer_bytes": 0,
                "observer_calls": 0,
                "observer_submissions": 0,
                "observer_frames_replaced": 0,
                "observer_wait_total_ms": 0.0,
                "observer_wait_last_ms": 0.0,
                "observer_wait_max_ms": 0.0,
                "observer_total_ms": 0.0,
                "observer_last_ms": 0.0,
                "observer_max_ms": 0.0,
            }
            for source in ("live", "main")
        }
        self._sequence = 0
        self._generation = 0

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source).strip().lower()
        if normalized not in {"live", "main"}:
            raise ValueError(f"unsupported camera capture source: {source!r}")
        return normalized

    def start(self) -> bool:
        with self._lock:
            starting_generation = self._stop.is_set()
            if starting_generation:
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
                if (
                    self._observer_dispatch is not None
                    and self._observer_dispatch.running()
                ):
                    raise RuntimeError(
                        f"cannot restart capture for {self.camera_id} while "
                        "observer is stopping"
                    )
            self._stop.clear()
            if starting_generation:
                self._generation += 1
        if self._observer_dispatch is not None:
            try:
                self._observer_dispatch.start()
            except BaseException:
                self._stop.set()
                raise
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
        # Published frames are immutable, so retaining the frame reference is
        # safe while the potentially large writable copy happens without
        # blocking capture publication or lifecycle/status access.
        copied_image = frame.image.copy()
        with self._lock:
            self._stats[source]["frame_copy_count"] += 1
            self._stats[source]["frame_copy_bytes"] += int(copied_image.nbytes)
        return CapturedFrame(
            source=frame.source,
            image=copied_image,
            captured_at_epoch=frame.captured_at_epoch,
            captured_at_monotonic=frame.captured_at_monotonic,
            captured_at_iso=frame.captured_at_iso,
            width=frame.width,
            height=frame.height,
            sequence=frame.sequence,
            generation=frame.generation,
        )

    def request_stop(self) -> None:
        self._stop.set()
        if self._observer_dispatch is not None:
            self._observer_dispatch.request_stop()
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
        if (
            self._observer_dispatch is not None
            and not self._observer_dispatch.wait_stopped(
                max(0.0, deadline - self._monotonic_clock())
            )
        ):
            observer_thread = self._observer_dispatch.thread()
            if observer_thread is not None:
                alive["observer"] = observer_thread
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
                self._jpegs.clear()
                self._preview.clear()
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
        if self._observer_dispatch is not None:
            self._observer_dispatch.close()

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
                observer_calls = int(capture_stats[source]["observer_calls"])
                capture_stats[source]["observer_average_ms"] = round(
                    float(capture_stats[source]["observer_total_ms"])
                    / max(1, observer_calls),
                    3,
                )
                durations = sorted(self._observer_durations_ms[source])
                capture_stats[source]["observer_p95_ms"] = self._percentile(
                    durations,
                    0.95,
                )
                capture_stats[source]["observer_p99_ms"] = self._percentile(
                    durations,
                    0.99,
                )
                observer_waits = sorted(self._observer_waits_ms[source])
                capture_stats[source]["observer_wait_average_ms"] = round(
                    float(capture_stats[source]["observer_wait_total_ms"])
                    / max(1, observer_calls),
                    3,
                )
                capture_stats[source]["observer_wait_p95_ms"] = self._percentile(
                    observer_waits,
                    0.95,
                )
                capture_stats[source]["observer_wait_p99_ms"] = self._percentile(
                    observer_waits,
                    0.99,
                )
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
                "observer_running": (
                    self._observer_dispatch.running()
                    if self._observer_dispatch is not None
                    else False
                ),
                "observer_pending": (
                    self._observer_dispatch.pending()
                    if self._observer_dispatch is not None
                    else False
                ),
                "stream_dimensions": {
                    source: dict(dimensions)
                    for source, dimensions in self._dimensions.items()
                },
                "live_detections": [
                    dict(item) for item in self._detections.get("live", ())
                ],
                "live_pipeline": dict(self._pipeline_status.get("live") or {}),
                "capture_stats": capture_stats,
            }

    def frame_ready(self, source: str = "live") -> bool:
        """Return freshness without copying the captured image."""
        source = self._normalize_source(source)
        now = self._monotonic_clock()
        with self._lock:
            frame = self._frames.get(source)
            return bool(
                frame is not None
                and now - frame.captured_at_monotonic <= self.stale_seconds
            )

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
        consecutive_open_failures = 0
        try:
            while not self._cancelled(stop_event):
                if self._should_exit_for_idle(source, stop_event):
                    return
                failure_reason = ""
                handle: CaptureHandle | None = None
                session_received_frame = False
                try:
                    handle = self.backend.create_handle()
                    open_timeout_ms = (
                        self.reconnect_open_timeout_ms
                        if source == "live" and consecutive_open_failures > 0
                        else self.initial_open_timeout_ms
                    )
                    with self._lock:
                        self._stats[source]["last_open_timeout_ms"] = open_timeout_ms
                        if open_timeout_ms > self.initial_open_timeout_ms:
                            self._stats[source]["open_timeout_escalations"] += 1
                    opened = self.backend.open(
                        handle,
                        self._source_url(source),
                        lambda: self._cancelled(stop_event),
                        open_timeout_ms=open_timeout_ms,
                    )
                    if self._cancelled(stop_event):
                        return
                    if not opened or not handle.is_opened():
                        consecutive_open_failures += 1
                        failure_reason = self._capture_failure_reason(
                            "failed to open stream",
                            handle,
                        )
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
                                if not session_received_frame:
                                    consecutive_open_failures += 1
                                failure_reason = self._capture_failure_reason(
                                    "stream read failed",
                                    handle,
                                )
                                self._increment(source, "read_failures")
                                self._set_error(source, failure_reason)
                                break
                            if self._cancelled(stop_event):
                                break
                            session_received_frame = True
                            consecutive_open_failures = 0
                            retry_delay = self.retry_initial_seconds
                            self._store_sidecar_state(source, handle)
                            self._store_preview(source, handle)
                            self._publish_frame(source, image, stop_event)
                except Exception as exc:
                    if not session_received_frame:
                        consecutive_open_failures += 1
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

    @staticmethod
    def _capture_failure_reason(summary: str, handle: CaptureHandle) -> str:
        error_detail = getattr(handle, "error_detail", None)
        if not callable(error_detail):
            return summary
        try:
            raw_detail = error_detail()
        except Exception:
            return summary
        if not isinstance(raw_detail, str):
            return summary
        detail = redact_secret_text(raw_detail).strip()
        return f"{summary}: {detail[:400]}" if detail else summary

    def latest_detections(self, source: str = "live") -> list[dict[str, object]]:
        source = self._normalize_source(source)
        with self._lock:
            return [dict(item) for item in self._detections.get(source, ())]

    def latest_jpeg(self, source: str = "live") -> bytes | None:
        source = self._normalize_source(source)
        now = self._monotonic_clock()
        with self._lock:
            frame = self._frames.get(source)
            payload = self._jpegs.get(source)
            if (
                frame is None
                or not payload
                or now - frame.captured_at_monotonic > self.stale_seconds
            ):
                return None
            return bytes(payload)

    def latest_preview_image(self, source: str = "live") -> np.ndarray | None:
        source = self._normalize_source(source)
        now = self._monotonic_clock()
        with self._lock:
            frame = self._frames.get(source)
            preview = self._preview.get(source)
            if (
                frame is None
                or now - frame.captured_at_monotonic > self.stale_seconds
            ):
                return None
            return preview

    def _store_preview(self, source: str, handle: CaptureHandle) -> None:
        pop = getattr(handle, "pop_jpeg", None)
        if not callable(pop):
            return
        try:
            jpeg = pop()
        except Exception:
            return
        if not isinstance(jpeg, (bytes, bytearray)) or not jpeg:
            return
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return
        decoded.setflags(write=False)
        with self._lock:
            self._jpegs[source] = bytes(jpeg)
            self._preview[source] = decoded

    def _store_sidecar_state(self, source: str, handle: CaptureHandle) -> None:
        self._store_detections(source, handle)
        status = getattr(handle, "pipeline_status", None)
        if not callable(status):
            return
        try:
            payload = status()
        except Exception:
            return
        if not isinstance(payload, dict) or not payload:
            return
        factory = str(payload.get("source_element") or "").strip()
        with self._lock:
            previous = str(
                (self._pipeline_status.get(source) or {}).get("source_element") or ""
            )
            self._pipeline_status[source] = dict(payload)
        if factory and factory != previous:
            LOGGER.info(
                "live capture for %s/%s uses %s",
                self.camera_id,
                source,
                factory,
            )

    def _store_detections(self, source: str, handle: CaptureHandle) -> None:
        pop = getattr(handle, "pop_detections", None)
        if not callable(pop):
            return
        try:
            detections = pop()
        except Exception:
            return
        if not isinstance(detections, list):
            return
        stored = [dict(item) for item in detections if isinstance(item, dict)]
        if not stored:
            return
        with self._lock:
            self._detections[source] = stored

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
            # CaptureHandle.read() transfers ownership. Store that allocation
            # directly and make shared access read-only; latest() remains the
            # explicit writable-copy boundary for independent consumers.
            stored_image = image
            stored_image.setflags(write=False)
            frame = CapturedFrame(
                source=source,
                image=stored_image,
                captured_at_epoch=captured_at_epoch,
                captured_at_monotonic=captured_at_monotonic,
                captured_at_iso=captured_at_iso,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                sequence=self._sequence,
                generation=self._generation,
            )
            self._frames[source] = frame
            self._dimensions[source] = {
                "width": frame.width,
                "height": frame.height,
            }
            self._frame_times[source].append(captured_at_monotonic)
            self._stats[source]["frames_received"] += 1
            self._stats[source]["frame_transfer_count"] += 1
            self._stats[source]["frame_transfer_bytes"] += int(stored_image.nbytes)
        if self._observer_dispatch is not None:
            accepted, replaced = self._observer_dispatch.submit(frame)
            if accepted:
                with self._lock:
                    self._stats[source]["observer_submissions"] += 1
                    if replaced:
                        self._stats[source]["observer_frames_replaced"] += 1
        return True

    def _record_observer_result(
        self,
        source: str,
        wait_ms: float,
        observer_ms: float,
        error: BaseException | None,
    ) -> None:
        failures = 0
        with self._lock:
            stats = self._stats[source]
            stats["observer_calls"] += 1
            stats["observer_wait_total_ms"] += wait_ms
            stats["observer_wait_last_ms"] = wait_ms
            stats["observer_wait_max_ms"] = max(
                float(stats["observer_wait_max_ms"]),
                wait_ms,
            )
            stats["observer_total_ms"] += observer_ms
            stats["observer_last_ms"] = observer_ms
            stats["observer_max_ms"] = max(
                float(stats["observer_max_ms"]),
                observer_ms,
            )
            self._observer_durations_ms[source].append(observer_ms)
            self._observer_waits_ms[source].append(wait_ms)
            if error is not None:
                stats["observer_errors"] += 1
                failures = int(stats["observer_errors"])
        if error is not None and (failures == 1 or failures % 100 == 0):
            LOGGER.error(
                "capture frame observer failed for %s/%s (failures=%s): %s",
                self.camera_id,
                source,
                failures,
                redact_secret_text(error),
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(math.ceil(len(values) * percentile) - 1)))
        return round(float(values[index]), 3)

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
            self._stats[source][key] = int(self._stats[source][key]) + 1
            return int(self._stats[source][key])

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
