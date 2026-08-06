from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import cv2
import numpy as np

from .motion import MotionQualificationResult, preprocess_motion_frame
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_coordinator import (
    VisualBackupAction,
    VisualBackupCoordinator,
    VisualBackupPolicy,
)
from .motion_decisions import (
    MotionAuditRecorder,
    audit_features,
    should_verify_suppression,
)
from .motion_events import MotionEventCoordinator, MotionTrigger
from .motion_pipeline import MotionDebugSnapshotStore

LOGGER = logging.getLogger(__name__)
CACHED_PREPROCESSOR_IMPLEMENTATION = "gray_blur"


@dataclass(frozen=True, slots=True)
class MotionAnalysisHooks:
    frame_analysis_required: Callable[[], bool]
    sample_fps: Callable[[], float]
    frame_width: Callable[[], int]
    preprocessor_implementation: Callable[[], str]
    observe_frame: Callable[[np.ndarray, float], None]
    motion_settings: Callable[[], tuple[str, str, int]]
    continuous_primary_required: Callable[[], bool]
    continuous_primary_due: Callable[[float, float], bool]
    execute_continuous: Callable[[float], None]
    execute_debug_capture: Callable[[float], None]
    adaptive_rearm_seconds: Callable[[], float]
    priority_dedup_seconds: Callable[[], float]
    run_pipeline: Callable[..., MotionQualificationResult]
    illumination_filter_enabled: Callable[[], bool]
    trigger_mode: Callable[[], str]
    detection_enabled: Callable[[], bool]
    with_source_evidence: Callable[..., MotionQualificationResult]
    visual_backup_settings: Callable[[], dict[str, float | int]]
    visual_backup_policy: Callable[[], VisualBackupPolicy]
    suppression_verification_rate: Callable[[], float]
    visual_backup_warmup_seconds: Callable[[], float]
    sample_rejected_motion: Callable[[datetime, MotionQualificationResult], str]
    publish_event: Callable[[str, dict[str, Any]], None]
    set_last_motion_at: Callable[[str], None]
    increment_stat: Callable[[str, int], None]
    record_analysis_wait: Callable[[float], None]
    reset_temporal_runtime: Callable[[], None]


@dataclass(frozen=True, slots=True)
class MotionFrameSubmission:
    """A stable captured frame handed to the latest-only analysis mailbox."""

    image: np.ndarray
    captured_at_epoch: float
    captured_at_monotonic: float
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class _AnalysisSlotWakeup:
    """Mailbox signal emitted when a fair qualification slot becomes available."""


ANALYSIS_SLOT_WAKEUP = _AnalysisSlotWakeup()


class MotionAnalysisService:
    """Own continuous frame-driven motion analysis for one camera."""

    def __init__(
        self,
        *,
        camera_id: str,
        frame_lock: threading.Lock,
        analysis_lock: threading.Lock,
        ring_size: int,
        queue_size: int,
        limiter: FairMotionAnalysisLimiter,
        events: MotionEventCoordinator,
        visual_backup: VisualBackupCoordinator,
        audit_recorder: MotionAuditRecorder,
        debug_store: MotionDebugSnapshotStore,
        hooks: MotionAnalysisHooks,
    ) -> None:
        self.camera_id = camera_id
        self.frame_lock = frame_lock
        self.analysis_lock = analysis_lock
        self.limiter = limiter
        self.events = events
        self.visual_backup = visual_backup
        self.audit_recorder = audit_recorder
        self.debug_store = debug_store
        self.hooks = hooks
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_size)
        self.color_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
        self.processed_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
        self.queue: queue.Queue[
            float | MotionFrameSubmission | _AnalysisSlotWakeup | None
        ] = queue.Queue(
            maxsize=queue_size
        )
        self.thread: threading.Thread | None = None
        self.last_sample_clock = 0.0
        self.last_processed_at = 0.0
        self.last_processed_sequence = 0
        self._submission_sequence = 0
        self._last_frame_epoch = 0.0
        self.primary_last_processed_at = 0.0
        self._pending_analysis_at = 0.0
        self._analysis_request_deferred = False
        self.last_continuous_result: MotionQualificationResult | None = None
        self.debug_last_run_clock = 0.0
        self._visual_lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._stop_requested = threading.Event()
        self._admission_lock = threading.Lock()
        self._accepting_frames = True
        self._telemetry_lock = threading.Lock()
        self._telemetry: dict[str, Any] = {
            "mailbox_high_water": 0,
            "mailbox_replacements": 0,
            "analysis_slot_deferrals": 0,
            "clock_discontinuity_resets": 0,
            "raw_frames_submitted": 0,
            "frames_sampled": 0,
            "preprocess_count": 0,
            "preprocess_total_ms": 0.0,
            "preprocess_last_ms": 0.0,
            "preprocess_max_ms": 0.0,
            "derived_frame_count": 0,
            "derived_frame_bytes": 0,
            "capture_to_analysis_count": 0,
            "capture_to_analysis_total_ms": 0.0,
            "capture_to_analysis_last_ms": 0.0,
            "capture_to_analysis_max_ms": 0.0,
            "analysis_cycle_count": 0,
            "analysis_cycle_total_ms": 0.0,
            "analysis_cycle_last_ms": 0.0,
            "analysis_cycle_max_ms": 0.0,
            "qualification_count": 0,
            "qualification_total_ms": 0.0,
            "qualification_last_ms": 0.0,
            "qualification_max_ms": 0.0,
            "copy_count": 0,
            "copy_bytes": 0,
            "copies_by_reason": {},
            "shared_read_count": 0,
            "shared_read_bytes": 0,
            "shared_reads_by_reason": {},
            "cached_derivative_reuse_count": 0,
            "cached_derivative_reuse_bytes": 0,
        }
        self._timing_samples_ms: dict[str, deque[float]] = {
            "preprocess": deque(maxlen=600),
            "capture_to_analysis": deque(maxlen=600),
            "analysis_cycle": deque(maxlen=600),
            "qualification": deque(maxlen=600),
        }

    def start(self, stop_event: threading.Event) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.clear_queue()
        self._stop_event = stop_event
        self._stop_requested.clear()
        with self._admission_lock:
            self._accepting_frames = True
        thread = threading.Thread(
            target=self.run,
            args=(stop_event,),
            name=f"motion-analysis-{self.camera_id}",
            daemon=False,
        )
        self.thread = thread
        try:
            thread.start()
        except BaseException:
            self.thread = None
            with self._admission_lock:
                self._accepting_frames = False
            self._stop_requested.set()
            self._stop_event = None
            raise

    def request_stop(self) -> None:
        with self._admission_lock:
            self._accepting_frames = False
            self._stop_requested.set()
            self.limiter.cancel(self.camera_id)
            self.clear_queue()
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

    def wait_stopped(self, timeout: float) -> bool:
        thread = self.thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        self.thread = None
        return True

    def clear_queue(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def reset(self) -> None:
        self.limiter.cancel(self.camera_id)
        with self.frame_lock:
            self.frames.clear()
            self.color_frames.clear()
            self.processed_frames.clear()
            self.last_sample_clock = 0.0
            self.last_continuous_result = None
            self.last_processed_at = 0.0
            self.last_processed_sequence = 0
            self._last_frame_epoch = 0.0
            self.primary_last_processed_at = 0.0
            self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False
        self.clear_queue()
        with self._visual_lock:
            self.visual_backup.reset()

    def schedule(self, captured_at: float, stop_event: threading.Event) -> None:
        self._enqueue_latest(captured_at, stop_event)

    def submit_frame(
        self,
        frame: np.ndarray,
        frame_clock: float,
        stop_event: threading.Event,
        captured_at: float | None = None,
    ) -> None:
        """Hand a stable raw frame to the analysis worker without preprocessing."""
        if not self._admit_frame(frame_clock, stop_event):
            return
        safe_frame = frame
        if frame.flags.writeable or not frame.flags.owndata:
            safe_frame = frame.copy()
            safe_frame.setflags(write=False)
            self._record_copy(safe_frame, "submission_safety")
        else:
            safe_frame.setflags(write=False)
        frame_epoch = captured_at if captured_at is not None else time.time()
        with self.frame_lock:
            self._submission_sequence += 1
            sequence = self._submission_sequence
        queued = self._enqueue_latest(
            MotionFrameSubmission(
                image=safe_frame,
                captured_at_epoch=frame_epoch,
                captured_at_monotonic=frame_clock,
                sequence=sequence,
            ),
            stop_event,
        )
        if queued:
            with self._telemetry_lock:
                self._telemetry["raw_frames_submitted"] += 1

    def _enqueue_latest(
        self,
        work: float | MotionFrameSubmission,
        stop_event: threading.Event,
    ) -> bool:
        with self._admission_lock:
            if stop_event.is_set() or not self._accepting_frames:
                return False
            try:
                self.queue.put_nowait(work)
                self._record_mailbox_depth()
                return True
            except queue.Full:
                dropped = 0
                try:
                    evicted = self.queue.get_nowait()
                    if evicted is not ANALYSIS_SLOT_WAKEUP:
                        dropped += 1
                except queue.Empty:
                    pass
            try:
                self.queue.put_nowait(work)
                self._record_mailbox_depth()
                queued = True
            except queue.Full:
                dropped += 1
                queued = False
        if dropped:
            self.hooks.increment_stat("analysis_frames_dropped", dropped)
            with self._telemetry_lock:
                self._telemetry["mailbox_replacements"] += dropped
        return queued

    def remember_frame(
        self,
        frame: np.ndarray,
        frame_clock: float,
        stop_event: threading.Event,
        captured_at: float | None = None,
    ) -> None:
        if not self._admit_frame(frame_clock, stop_event):
            return
        frame_epoch = captured_at if captured_at is not None else time.time()
        prepared = self._preprocess_frame(frame, frame_epoch)
        if prepared is None:
            return
        self.schedule(frame_epoch, stop_event)

    def _admit_frame(
        self,
        frame_clock: float,
        stop_event: threading.Event,
    ) -> bool:
        if (
            stop_event.is_set()
            or not self._accepting_frames
            or not self.hooks.frame_analysis_required()
        ):
            return False
        interval = 1.0 / max(1.0, self.hooks.sample_fps())
        with self.frame_lock:
            if frame_clock - self.last_sample_clock < interval * 0.85:
                return False
            self.last_sample_clock = frame_clock
        return True

    def _preprocess_frame(
        self,
        frame: np.ndarray,
        frame_epoch: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        self._reset_for_clock_discontinuity(frame_epoch)
        preprocess_started = time.monotonic()
        try:
            height, width = frame.shape[:2]
            frame_width = self.hooks.frame_width()
            target_height = max(90, round(height * frame_width / max(1, width)))
            resized = cv2.resize(
                frame,
                (frame_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            processed = (
                preprocess_motion_frame(gray)
                if self.hooks.preprocessor_implementation()
                == CACHED_PREPROCESSOR_IMPLEMENTATION
                else None
            )
        except (cv2.error, ValueError):
            return None
        resized.setflags(write=False)
        gray.setflags(write=False)
        if processed is not None:
            processed.setflags(write=False)
        preprocess_ms = max(0.0, (time.monotonic() - preprocess_started) * 1000.0)
        with self._telemetry_lock:
            self._record_timing_locked("preprocess", preprocess_ms)
            self._telemetry["frames_sampled"] += 1
            self._telemetry["derived_frame_count"] += 2 + int(processed is not None)
            self._telemetry["derived_frame_bytes"] += int(
                resized.nbytes
                + gray.nbytes
                + (processed.nbytes if processed is not None else 0)
            )
        with self.frame_lock:
            self.frames.append((frame_epoch, gray))
            self.color_frames.append((frame_epoch, resized))
            if processed is not None:
                self.processed_frames.append((frame_epoch, processed))
            else:
                self.processed_frames.clear()
        return gray, resized, processed

    def _reset_for_clock_discontinuity(self, frame_epoch: float) -> None:
        with self.frame_lock:
            previous_epoch = self._last_frame_epoch
            self._last_frame_epoch = frame_epoch
            if previous_epoch <= 0.0 or frame_epoch > previous_epoch:
                return
            self.frames.clear()
            self.color_frames.clear()
            self.processed_frames.clear()
            self.last_processed_at = 0.0
            self.primary_last_processed_at = 0.0
            self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False
        self.limiter.cancel(self.camera_id)
        with self._visual_lock:
            self.visual_backup.reset()
        self.events.reset_timebase()
        with self._telemetry_lock:
            self._telemetry["clock_discontinuity_resets"] += 1
        self.hooks.reset_temporal_runtime()

    def samples(self) -> list[tuple[float, np.ndarray]]:
        with self.frame_lock:
            return [
                (timestamp, self._share_frame(frame, "samples"))
                for timestamp, frame in self.frames
            ]

    def samples_since(self, captured_at: float) -> list[tuple[float, np.ndarray]]:
        with self.frame_lock:
            return [
                (timestamp, self._share_frame(frame, "samples_since"))
                for timestamp, frame in self.frames
                if timestamp >= captured_at
            ]

    def status(self) -> dict[str, Any]:
        with self.frame_lock:
            buffered_frames = len(self.frames)
            frame_shape = list(self.frames[-1][1].shape[:2]) if self.frames else None
            color_buffered_frames = len(self.color_frames)
            color_frame_shape = (
                list(self.color_frames[-1][1].shape) if self.color_frames else None
            )
            processed_buffered_frames = len(self.processed_frames)
            processed_frame_shape = (
                list(self.processed_frames[-1][1].shape)
                if self.processed_frames
                else None
            )
            last_result = (
                self.last_continuous_result.as_dict()
                if self.last_continuous_result is not None
                else None
            )
        return {
            "queue_depth": self.queue.qsize(),
            "worker_running": bool(self.thread is not None and self.thread.is_alive()),
            "buffered_frames": buffered_frames,
            "frame_shape": frame_shape,
            "color_buffered_frames": color_buffered_frames,
            "color_frame_shape": color_frame_shape,
            "processed_buffered_frames": processed_buffered_frames,
            "processed_frame_shape": processed_frame_shape,
            "continuous_last_result": last_result,
            "telemetry": self.telemetry_snapshot(),
        }

    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._telemetry_lock:
            result = {
                key: (
                    {
                        nested_key: (
                            dict(nested_value)
                            if isinstance(nested_value, dict)
                            else nested_value
                        )
                        for nested_key, nested_value in value.items()
                    }
                    if isinstance(value, dict)
                    else value
                )
                for key, value in self._telemetry.items()
            }
            timing_samples = {
                prefix: sorted(samples)
                for prefix, samples in self._timing_samples_ms.items()
            }
        for prefix in (
            "preprocess",
            "capture_to_analysis",
            "analysis_cycle",
            "qualification",
        ):
            count = int(result.get(f"{prefix}_count") or 0)
            total = float(result.get(f"{prefix}_total_ms") or 0.0)
            result[f"{prefix}_average_ms"] = round(total / count, 3) if count else 0.0
            for suffix in ("total_ms", "last_ms", "max_ms"):
                result[f"{prefix}_{suffix}"] = round(
                    float(result.get(f"{prefix}_{suffix}") or 0.0),
                    3,
                )
            samples = timing_samples[prefix]
            result[f"{prefix}_p95_ms"] = self._percentile(samples, 0.95)
            result[f"{prefix}_p99_ms"] = self._percentile(samples, 0.99)
        return result

    def _record_mailbox_depth(self) -> None:
        depth = self.queue.qsize()
        with self._telemetry_lock:
            self._telemetry["mailbox_high_water"] = max(
                int(self._telemetry["mailbox_high_water"]),
                depth,
            )

    def _record_timing(self, prefix: str, duration_ms: float) -> None:
        with self._telemetry_lock:
            self._record_timing_locked(prefix, duration_ms)

    def _record_timing_locked(self, prefix: str, duration_ms: float) -> None:
        self._telemetry[f"{prefix}_count"] += 1
        self._telemetry[f"{prefix}_total_ms"] += duration_ms
        self._telemetry[f"{prefix}_last_ms"] = duration_ms
        self._telemetry[f"{prefix}_max_ms"] = max(
            float(self._telemetry[f"{prefix}_max_ms"]),
            duration_ms,
        )
        self._timing_samples_ms[prefix].append(duration_ms)

    def _share_frame(self, frame: np.ndarray, reason: str) -> np.ndarray:
        frame.setflags(write=False)
        with self._telemetry_lock:
            self._telemetry["shared_read_count"] += 1
            self._telemetry["shared_read_bytes"] += int(frame.nbytes)
            reads_by_reason = self._telemetry["shared_reads_by_reason"]
            reason_stats = reads_by_reason.setdefault(
                reason,
                {"count": 0, "bytes": 0},
            )
            reason_stats["count"] += 1
            reason_stats["bytes"] += int(frame.nbytes)
        return frame

    def _record_copy(self, frame: np.ndarray, reason: str) -> None:
        with self._telemetry_lock:
            self._telemetry["copy_count"] += 1
            self._telemetry["copy_bytes"] += int(frame.nbytes)
            reason_stats = self._telemetry["copies_by_reason"].setdefault(
                reason,
                {"count": 0, "bytes": 0},
            )
            reason_stats["count"] += 1
            reason_stats["bytes"] += int(frame.nbytes)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = min(
            len(values) - 1,
            max(0, int(math.ceil(len(values) * percentile) - 1)),
        )
        return round(float(values[index]), 3)

    def visual_backup_snapshot(self) -> dict[str, Any]:
        with self._visual_lock:
            return self.visual_backup.snapshot()

    def record_visual_camera_match(self, observed_at: float) -> bool:
        with self._visual_lock:
            return self.visual_backup.record_camera_match(observed_at)

    def visual_backup_readiness(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> bool:
        with self._visual_lock:
            return self.visual_backup.readiness(
                result,
                captured_at,
                self.hooks.visual_backup_policy(),
            )

    @property
    def visual_backup_scene_ready(self) -> bool:
        with self._visual_lock:
            return self.visual_backup.scene_ready

    @property
    def visual_backup_stable_samples(self) -> int:
        with self._visual_lock:
            return self.visual_backup.stable_samples

    def reset_visual_backup_candidate(self) -> None:
        with self._visual_lock:
            self.visual_backup.reset_candidate()

    def run(self, stop_event: threading.Event) -> None:
        try:
            self._run(stop_event)
        finally:
            self.limiter.cancel(self.camera_id)
            self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False

    def _run(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        while not self._stopping():
            try:
                work = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self._pending_analysis_at <= 0.0:
                    continue
                try:
                    self._try_execute_pending_analysis()
                except Exception:
                    self.hooks.increment_stat("analysis_worker_errors", 1)
                    LOGGER.exception(
                        "deferred motion analysis cycle failed for %s",
                        self.camera_id,
                    )
                continue
            if work is None or stop_event.is_set():
                return
            if work is ANALYSIS_SLOT_WAKEUP:
                try:
                    self._try_execute_pending_analysis()
                except Exception:
                    self.hooks.increment_stat("analysis_worker_errors", 1)
                    LOGGER.exception(
                        "woken motion analysis cycle failed for %s",
                        self.camera_id,
                    )
                continue
            cycle_started = time.monotonic()
            try:
                work = self._take_latest_pending(work)
                if work is None or self._stopping():
                    return
                handoff_ms = (
                    (time.monotonic() - work.captured_at_monotonic) * 1000.0
                    if isinstance(work, MotionFrameSubmission)
                    else (time.time() - work) * 1000.0
                )
                self._record_timing(
                    "capture_to_analysis",
                    max(0.0, handoff_ms),
                )
                sequence = 0
                if isinstance(work, MotionFrameSubmission):
                    prepared = self._preprocess_frame(
                        work.image,
                        work.captured_at_epoch,
                    )
                    if prepared is None:
                        continue
                    captured_at = work.captured_at_epoch
                    sequence = work.sequence
                    frame = self._share_frame(
                        prepared[0],
                        "analysis_latest",
                    )
                else:
                    with self.frame_lock:
                        if not self.frames:
                            continue
                        captured_at, frame = self.frames[-1]
                        frame = self._share_frame(frame, "analysis_latest")
                if sequence > 0:
                    if sequence <= self.last_processed_sequence:
                        continue
                    self.last_processed_sequence = sequence
                elif captured_at <= self.last_processed_at:
                    continue
                self.last_processed_at = captured_at
                self.hooks.observe_frame(frame, captured_at)
                if self.hooks.continuous_primary_required() and (
                    self.hooks.continuous_primary_due(
                        captured_at,
                        self.primary_last_processed_at,
                    )
                ):
                    self._pending_analysis_at = captured_at
                    self._try_execute_pending_analysis()
                elif (
                    self.debug_store.enabled()
                    and time.monotonic() - self.debug_last_run_clock >= 1.0
                ):
                    self.debug_last_run_clock = time.monotonic()
                    self.hooks.execute_debug_capture(captured_at)
            except Exception:
                self.hooks.increment_stat("analysis_worker_errors", 1)
                LOGGER.exception("motion analysis cycle failed for %s", self.camera_id)
            finally:
                self._record_timing(
                    "analysis_cycle",
                    max(0.0, (time.monotonic() - cycle_started) * 1000.0),
                )

    def _try_execute_pending_analysis(self) -> bool:
        captured_at = self._pending_analysis_at
        if captured_at <= 0.0 or self._stopping():
            return False
        if not self.hooks.continuous_primary_required():
            self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False
            self.limiter.cancel(self.camera_id)
            return False
        with self.limiter.try_acquire(
            self.camera_id,
            on_available=self._wake_for_analysis_slot,
        ) as wait_seconds:
            if wait_seconds is None:
                if not self._analysis_request_deferred:
                    with self._telemetry_lock:
                        self._telemetry["analysis_slot_deferrals"] += 1
                    self._analysis_request_deferred = True
                return False
            self._analysis_request_deferred = False
            self.hooks.record_analysis_wait(max(0.0, wait_seconds * 1000.0))
            qualification_started = time.monotonic()
            try:
                self.hooks.execute_continuous(captured_at)
            except BaseException:
                if self._pending_analysis_at == captured_at:
                    self._pending_analysis_at = 0.0
                self._analysis_request_deferred = False
                raise
            finally:
                self._record_timing(
                    "qualification",
                    max(
                        0.0,
                        (time.monotonic() - qualification_started) * 1000.0,
                    ),
                )
            if self._pending_analysis_at == captured_at:
                self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False
            return True

    def _wake_for_analysis_slot(self) -> None:
        if self._stopping():
            return
        try:
            self.queue.put_nowait(ANALYSIS_SLOT_WAKEUP)
        except queue.Full:
            # A raw frame already in the latest-only mailbox will wake the
            # worker and retry the same pending qualification request.
            return

    def _take_latest_pending(
        self,
        work: float | MotionFrameSubmission,
    ) -> float | MotionFrameSubmission | None:
        """Replace queued work with the newest pending capture."""
        superseded = 0
        while True:
            try:
                candidate = self.queue.get_nowait()
            except queue.Empty:
                break
            if candidate is None:
                return None
            if candidate is ANALYSIS_SLOT_WAKEUP:
                continue
            work = candidate
            superseded += 1
        if superseded:
            self.hooks.increment_stat("analysis_frames_dropped", superseded)
            with self._telemetry_lock:
                self._telemetry["mailbox_replacements"] += superseded
        return work

    def analyze_continuous(self, captured_at: float) -> None:
        _mode, sensitivity, _frame_width = self.hooks.motion_settings()
        with self.frame_lock:
            source_samples = (
                self.color_frames
                if len(self.color_frames) >= 2
                else list(self.frames)[-3:]
            )
            samples = [
                (timestamp, self._share_frame(frame, "continuous_samples"))
                for timestamp, frame in source_samples
            ]
            processed_by_timestamp = dict(self.processed_frames)
            cached_processed = [
                processed_by_timestamp[timestamp]
                for timestamp, _frame in samples
                if timestamp in processed_by_timestamp
            ]
            if len(cached_processed) != len(samples):
                cached_processed = []
            else:
                for frame in cached_processed:
                    frame.setflags(write=False)
        if len(samples) < 2:
            return
        if cached_processed:
            with self._telemetry_lock:
                self._telemetry["cached_derivative_reuse_count"] += len(
                    cached_processed
                )
                self._telemetry["cached_derivative_reuse_bytes"] += sum(
                    int(frame.nbytes) for frame in cached_processed
                )
        try:
            with self.analysis_lock:
                result = self.hooks.run_pipeline(
                    [frame for _timestamp, frame in samples],
                    sensitivity,
                    captured_at,
                    [timestamp for timestamp, _frame in samples],
                    isolated=False,
                    capture_debug=self.debug_store.enabled(),
                    include_telemetry=False,
                    processed_frames=cached_processed or None,
                    processed_frame_implementation=(
                        CACHED_PREPROCESSOR_IMPLEMENTATION
                        if cached_processed
                        else ""
                    ),
                )
        except Exception as error:
            LOGGER.warning(
                "continuous motion analysis failed for %s: %s",
                self.camera_id,
                error,
            )
            return
        self.last_continuous_result = result
        self.primary_last_processed_at = captured_at
        self._record_continuous_stats(result)
        if self._stopping():
            return
        trigger_mode = self.hooks.trigger_mode()
        if trigger_mode == "camera_rescue":
            self.consider_visual_backup(result, samples, captured_at)
            return
        if trigger_mode != "adaptive" or not self.hooks.detection_enabled():
            return
        fused = self.hooks.with_source_evidence(
            result,
            samples[0][0],
            captured_at,
            include_telemetry=False,
            require_primary_trigger=True,
        )
        self.last_continuous_result = fused
        if (
            self._stopping()
            or not fused.accepted
            or not self.reserve_adaptive_trigger(captured_at)
        ):
            return
        if self._stopping():
            self.events.defer_adaptive(captured_at)
            return
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        try:
            queued = self._enqueue_trigger(
                MotionTrigger(
                    topic="adaptive/motion",
                    message="adaptive motion transition",
                    event_at=event_at,
                    received_at=captured_at,
                    prequalified=fused,
                )
            )
        except Exception:
            self.events.defer_adaptive(captured_at)
            raise
        if not queued:
            self.events.defer_adaptive(captured_at)
            return
        self._publish_motion(event_at, "adaptive")

    def consider_visual_backup(
        self,
        result: MotionQualificationResult,
        samples: list[tuple[float, np.ndarray]],
        captured_at: float,
    ) -> None:
        if self._stopping():
            return
        settings = self.hooks.visual_backup_settings()
        illumination_probe_allowed = bool(
            self.hooks.detection_enabled()
            and result.reason == "illumination_change"
            and result.features.get("illumination_would_reject")
            and should_verify_suppression(
                (
                    f"illumination:{self.camera_id}:"
                    f"{int(captured_at // max(5.0, float(settings['cooldown_seconds'])))}"
                ),
                self.hooks.suppression_verification_rate(),
            )
        )
        with self._visual_lock:
            decision = self.visual_backup.evaluate(
                result,
                captured_at,
                self.hooks.visual_backup_policy(),
                detection_enabled=self.hooks.detection_enabled(),
                camera_motion_times=self.events.camera_motion_snapshot(),
                illumination_probe_allowed=illumination_probe_allowed,
            )
        result = decision.result
        if decision.action in {VisualBackupAction.DISABLED, VisualBackupAction.IGNORED}:
            if decision.count_nonpromotion:
                self.hooks.increment_stat("visual_backup_not_promoted", 1)
            return
        if decision.action == VisualBackupAction.NOT_READY:
            self.hooks.increment_stat("visual_backup_not_ready", 1)
            if decision.readiness_audit_needed:
                self.record_visual_backup_readiness_audit(result, captured_at)
            return
        if decision.action in {
            VisualBackupAction.ACCUMULATING,
            VisualBackupAction.CAMERA_NOTICE,
            VisualBackupAction.READY,
        }:
            self.hooks.increment_stat("visual_backup_candidates", 1)
        if decision.action == VisualBackupAction.ACCUMULATING:
            return
        if decision.action == VisualBackupAction.CAMERA_NOTICE:
            if decision.new_camera_match:
                self.hooks.increment_stat("visual_backup_onvif_matches", 1)
            return
        if decision.action != VisualBackupAction.READY:
            return
        if not self._reserve_visual_backup_trigger(captured_at):
            self.hooks.increment_stat("visual_backup_not_promoted", 1)
            self.reset_visual_backup_candidate()
            return

        trigger_enqueued = False
        try:
            if self._stopping():
                return
            fused = self.hooks.with_source_evidence(
                result,
                samples[0][0],
                captured_at,
                include_telemetry=False,
                require_primary_trigger=True,
            )
            self.last_continuous_result = fused
            if self._stopping() or not fused.accepted:
                return
            fused = MotionQualificationResult(
                accepted=fused.accepted,
                score=fused.score,
                threshold=fused.threshold,
                reason=fused.reason,
                frame_count=fused.frame_count,
                features={
                    **fused.features,
                    "visual_backup": True,
                    "visual_backup_required_score": round(
                        decision.required_score, 4
                    ),
                    "visual_backup_consecutive": decision.consecutive,
                    "visual_backup_grace_seconds": settings["grace_seconds"],
                },
                telemetry=dict(fused.telemetry),
            )
            event_at = datetime.fromtimestamp(captured_at, timezone.utc)
            trigger_enqueued = self._enqueue_trigger(
                MotionTrigger(
                    topic="adaptive/visual_backup",
                    message="adaptive visual backup after missing camera notice",
                    event_at=event_at,
                    received_at=captured_at,
                    prequalified=fused,
                )
            )
            if not trigger_enqueued:
                return
            self.hooks.increment_stat("visual_backup_triggers", 1)
            self.hooks.increment_stat(
                "illumination_verification_probes",
                int(decision.illumination_probe),
            )
            with self._visual_lock:
                self.visual_backup.record_trigger(captured_at)
            self._publish_motion(event_at, "visual_backup")
        finally:
            if not trigger_enqueued:
                self.events.defer_adaptive(captured_at)
            self.reset_visual_backup_candidate()

    def capture_debug(self, captured_at: float) -> None:
        samples = self.samples()
        frames = [frame for _timestamp, frame in samples]
        if len(frames) < 2:
            return
        _mode, sensitivity, _frame_width = self.hooks.motion_settings()
        try:
            self.hooks.run_pipeline(
                frames,
                sensitivity,
                captured_at,
                [timestamp for timestamp, _frame in samples],
                isolated=True,
                capture_debug=True,
            )
        except Exception as error:
            LOGGER.debug(
                "motion debug capture failed for %s: %s",
                self.camera_id,
                error,
            )

    def _record_continuous_stats(self, result: MotionQualificationResult) -> None:
        self.hooks.increment_stat("continuous_frames", 1)
        self.hooks.increment_stat("continuous_candidates", int(result.accepted))
        illumination_available = bool(
            result.features.get("illumination_evidence_available")
        )
        illumination_candidate = bool(result.features.get("illumination_would_reject"))
        self.hooks.increment_stat("illumination_evaluations", int(illumination_available))
        self.hooks.increment_stat("illumination_candidates", int(illumination_candidate))
        self.hooks.increment_stat(
            "illumination_filtered",
            int(
                illumination_candidate
                and self.hooks.illumination_filter_enabled()
                and not result.accepted
            ),
        )

    def _enqueue_trigger(self, trigger: MotionTrigger) -> bool:
        return self.events.enqueue(
            trigger,
            evict_oldest=False,
            on_trigger=lambda name: self.hooks.increment_stat(name, 1),
            on_drop=lambda name: self.hooks.increment_stat(name, 1),
        )

    def reserve_adaptive_trigger(self, captured_at: float) -> bool:
        allowed = self.events.reserve_adaptive(
            captured_at,
            rearm_seconds=self.hooks.adaptive_rearm_seconds(),
            priority_tolerance_seconds=self.hooks.priority_dedup_seconds(),
        )
        if not allowed:
            self.hooks.increment_stat("adaptive_triggers_deferred", 1)
        return allowed

    def _reserve_visual_backup_trigger(self, captured_at: float) -> bool:
        with self._visual_lock:
            allowed = self.events.reserve_with(
                lambda pending, last_completed_at: self.visual_backup.reserve_trigger(
                    captured_at,
                    self.hooks.visual_backup_policy(),
                    trigger_pending=pending,
                    last_completed_at=last_completed_at,
                )
            )
        if not allowed:
            self.hooks.increment_stat("visual_backup_rate_limited", 1)
        return allowed

    def record_visual_backup_readiness_audit(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> None:
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        try:
            self.audit_recorder.record_audit(
                snapshot_path=self.hooks.sample_rejected_motion(event_at, result),
                event_at=event_at,
                mode="camera_rescue",
                sensitivity=self.hooks.motion_settings()[1],
                score=result.score,
                threshold=result.threshold,
                reason="startup_not_ready",
                object_detected=None,
                trigger_count=0,
                features={
                    **audit_features(result),
                    "visual_backup_scene_ready": False,
                    "visual_backup_warmup_seconds": (
                        self.hooks.visual_backup_warmup_seconds()
                    ),
                },
                category="visual_backup",
            )
        except Exception:
            LOGGER.exception(
                "failed to record visual backup readiness audit for %s",
                self.camera_id,
            )

    def _publish_motion(self, event_at: datetime, source: str) -> None:
        timestamp = event_at.isoformat()
        self.hooks.set_last_motion_at(timestamp)
        self.hooks.publish_event(
            "motion",
            {
                "camera_id": self.camera_id,
                "timestamp": timestamp,
                "source": source,
            },
        )

    def _stopping(self) -> bool:
        return bool(
            self._stop_requested.is_set()
            or (self._stop_event is not None and self._stop_event.is_set())
        )
