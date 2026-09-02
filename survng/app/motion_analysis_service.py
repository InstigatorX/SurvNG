from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from .motion import MotionQualificationResult, preprocess_motion_frame
from .config import MotionQualificationConfig
from .domain_events import MotionObserved
from .ema_v2 import (
    DetectionIntent,
    EmaPolicy,
    EmaQualified,
    EmaSignalDecision,
    EmaSignalAction,
    EmaSignalConditioner,
    EpisodeDecisionReason,
    VISUAL_BACKUP_EXCLUDED_REASONS,
    MotionSource,
)
from .ema_route_cache import EmaCandidateSubmitResult
from .motion_analysis import FairMotionAnalysisLimiter
from .motion_decisions import (
    MotionAuditRecorder,
    audit_features,
    should_verify_suppression,
)
from .motion_events import MotionEventCoordinator, MotionTrigger
from .motion_pipeline import MotionDebugSnapshotStore, MotionEvidenceRepository

LOGGER = logging.getLogger(__name__)
CACHED_PREPROCESSOR_IMPLEMENTATION = "gray_blur"


class MotionAnalysisQualification(Protocol):
    def frame_analysis_required(self) -> bool: ...
    def settings(self) -> tuple[str, str, int]: ...
    def preprocessor_implementation(self) -> str: ...
    def observe_frame(self, frame: np.ndarray, captured_at: float) -> None: ...
    def continuous_primary_required(self) -> bool: ...
    def continuous_primary_due(self, captured_at: float, previous: float) -> bool: ...
    def run_pipeline(self, *args: Any, **kwargs: Any) -> MotionQualificationResult: ...
    def illumination_filter_enabled(self) -> bool: ...
    def trigger_mode(self) -> str: ...
    def with_source_evidence(
        self, result: MotionQualificationResult, *args: Any, **kwargs: Any
    ) -> MotionQualificationResult: ...
    def visual_backup_settings(self) -> dict[str, float | int]: ...
    def visual_backup_policy(self) -> EmaPolicy: ...
    def suppression_verification_rate(self) -> float: ...
    def reset_runtime(self, **kwargs: Any) -> None: ...


class MotionAnalysisState(Protocol):
    def detection_enabled(self) -> bool: ...
    def publish_event(self, event_type: str, payload: dict[str, Any]) -> None: ...
    def set_last_motion_at(self, value: str) -> None: ...
    def increment_stat(self, name: str, amount: int = 1) -> None: ...
    def record_analysis_wait(self, wait_ms: float) -> None: ...
    def lifecycle_generation(self) -> int: ...


class MotionAnalysisMedia(Protocol):
    def sample_rejected_motion(
        self, event_at: datetime, result: MotionQualificationResult
    ) -> str: ...

    def sample_rejected_motion_frame(
        self,
        event_at: datetime,
        result: MotionQualificationResult,
        frame: np.ndarray,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class MotionFrameSubmission:
    """A stable captured frame handed to the latest-only analysis mailbox."""

    image: np.ndarray
    captured_at_epoch: float
    captured_at_monotonic: float
    sequence: int = 0
    capture_sequence: int = 0
    capture_generation: int = 0
    lifecycle_generation: int = 0


@dataclass(frozen=True, slots=True)
class MotionEvidenceFrame:
    """Bounded, generation-qualified color evidence retained for fast detection."""

    image: np.ndarray
    captured_at_epoch: float
    captured_at_monotonic: float
    sequence: int
    capture_generation: int
    lifecycle_generation: int


@dataclass(frozen=True, slots=True)
class _AnalysisSlotWakeup:
    """Mailbox signal emitted when a fair qualification slot becomes available."""


ANALYSIS_SLOT_WAKEUP = _AnalysisSlotWakeup()


@dataclass(slots=True)
class _VisualBackupNonpromotionEpisode:
    """One bounded episode of credible EMA motion below its rescue gate."""

    started_at: float
    last_seen_at: float
    observation_count: int
    peak_at: float
    peak_result: MotionQualificationResult
    peak_required_score: float
    peak_frame: np.ndarray


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
        evidence: MotionEvidenceRepository,
        audit_recorder: MotionAuditRecorder,
        debug_store: MotionDebugSnapshotStore,
        config: MotionQualificationConfig,
        qualification: MotionAnalysisQualification,
        media: MotionAnalysisMedia,
        state: MotionAnalysisState,
    ) -> None:
        self.camera_id = camera_id
        self.frame_lock = frame_lock
        self.analysis_lock = analysis_lock
        self.limiter = limiter
        self.events = events
        self.evidence = evidence
        self.ema_v2 = EmaSignalConditioner(camera_id)
        self.ema_verification = EmaSignalConditioner(camera_id)
        self.audit_recorder = audit_recorder
        self.debug_store = debug_store
        self.config = config
        self.qualification = qualification
        self.media = media
        self.state = state
        # Temporal filtering threshold: skip analysis if < this ratio of pixels changed
        # Configurable in config.json as motion_qualification.temporal_filter_threshold
        # Default: 0.005 (0.5% pixel change)
        # Lower values = more aggressive skipping; Higher values = more analysis
        self.temporal_filter_threshold = float(config.temporal_filter_threshold)
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_size)
        self.color_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
        self.evidence_frames: deque[MotionEvidenceFrame] = deque(
            # Enough history to cover the EMA decision/dispatch delay without
            # retaining the full qualification ring as color images.
            maxlen=max(6, min(16, ring_size))
        )
        self.processed_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
        self.qualification_results: deque[
            tuple[float, MotionQualificationResult]
        ] = deque(maxlen=ring_size)
        # Route confirmation may arrive after recorded main-stream refinement.
        # The maximum route horizon is 300s at up to 10 EMA samples per second;
        # retain the full supported window plus scheduling margin.
        self.recent_accepted_results: deque[
            tuple[float, MotionQualificationResult]
        ] = deque(maxlen=4096)
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
        self._visual_nonpromotion_episode: _VisualBackupNonpromotionEpisode | None = None
        self._onvif_effectiveness_observer: (
            Callable[[bool, float], None] | None
        ) = None
        self._onvif_effectiveness_provider: Callable[[], dict[str, Any]] | None = None
        self._route_watch_provider: Callable[[str, float], Any | None] | None = None
        self._route_watch_consumer: Callable[[str, int], bool] | None = None
        self._ema_candidate_sink: (
            Callable[[str, float, dict[str, Any]], object] | None
        ) = None
        self._ema_candidate_source: (
            Callable[[str, float, float], list[tuple[float, dict[str, Any]]]] | None
        ) = None
        self._last_submitted_ema_candidate_at = 0.0
        self._last_ema_candidate_failure_log_monotonic = 0.0
        self._stop_event: threading.Event | None = None
        self._stop_requested = threading.Event()
        self._admission_lock = threading.Lock()
        self._accepting_frames = True
        self._telemetry_lock = threading.Lock()
        self._telemetry_started_monotonic = time.monotonic()
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
            "temporal_filter_skips": 0,
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

    def reset_ema_policy_state(self) -> None:
        with self._visual_lock:
            self.ema_v2.reset()
            self.ema_verification.reset()

    def set_onvif_effectiveness_observer(
        self,
        observer: Callable[[bool, float], None] | None,
    ) -> None:
        """Attach diagnostic-only EMA/ONVIF correlation instrumentation."""
        self._onvif_effectiveness_observer = observer

    def set_security_verification_context(
        self,
        *,
        onvif_effectiveness: Callable[[], dict[str, Any]] | None = None,
        route_watch: Callable[[str, float], Any | None] | None = None,
        consume_route_watch: Callable[[str, int], bool] | None = None,
        record_ema_candidate: (
            Callable[[str, float, dict[str, Any]], object] | None
        ) = None,
        load_ema_candidates: (
            Callable[[str, float, float], list[tuple[float, dict[str, Any]]]] | None
        ) = None,
    ) -> None:
        """Attach advisory signals that may accelerate persistent EMA checks."""
        self._onvif_effectiveness_provider = onvif_effectiveness
        self._route_watch_provider = route_watch
        self._route_watch_consumer = consume_route_watch
        self._ema_candidate_sink = record_ema_candidate
        self._ema_candidate_source = load_ema_candidates

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

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

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
            self.evidence_frames.clear()
            self.processed_frames.clear()
            self.qualification_results.clear()
            self.recent_accepted_results.clear()
            self._last_submitted_ema_candidate_at = 0.0
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
            self.ema_v2.reset()
            self.ema_verification.reset()
            self._visual_nonpromotion_episode = None

    def schedule(self, captured_at: float, stop_event: threading.Event) -> None:
        self._enqueue_latest(captured_at, stop_event)

    def submit_frame(
        self,
        frame: np.ndarray,
        frame_clock: float,
        stop_event: threading.Event,
        captured_at: float | None = None,
        *,
        capture_sequence: int = 0,
        capture_generation: int = 0,
        lifecycle_generation: int = 0,
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
                capture_sequence=capture_sequence,
                capture_generation=capture_generation,
                lifecycle_generation=lifecycle_generation,
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
            self.state.increment_stat("analysis_frames_dropped", dropped)
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
            or not self.qualification.frame_analysis_required()
        ):
            return False
        interval = 1.0 / max(1.0, self.config.sample_fps)
        with self.frame_lock:
            if frame_clock - self.last_sample_clock < interval * 0.85:
                return False
            self.last_sample_clock = frame_clock
        return True

    def _preprocess_frame(
        self,
        frame: np.ndarray,
        frame_epoch: float,
        *,
        captured_at_monotonic: float = 0.0,
        capture_sequence: int = 0,
        capture_generation: int = 0,
        lifecycle_generation: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        self._reset_for_clock_discontinuity(frame_epoch)
        preprocess_started = time.monotonic()
        try:
            if frame.ndim == 2:
                source_image = frame
                is_gray = True
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                source_image = frame[:, :, 0]
                is_gray = True
            else:
                source_image = frame
                is_gray = False
            height, width = source_image.shape[:2]
            frame_width = self.qualification.settings()[2]
            scale = min(1.0, frame_width / max(1, width, height))
            target_width = max(1, round(width * scale))
            target_height = max(1, round(height * scale))
            if target_width == width and target_height == height:
                resized = source_image
            else:
                resized = cv2.resize(
                    source_image,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            if is_gray:
                gray = resized
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                color = resized
            processed = (
                preprocess_motion_frame(gray)
                if self.qualification.preprocessor_implementation()
                == CACHED_PREPROCESSOR_IMPLEMENTATION
                else None
            )
        except (cv2.error, ValueError):
            return None
        color.setflags(write=False)
        gray.setflags(write=False)
        if processed is not None:
            processed.setflags(write=False)
        preprocess_ms = max(0.0, (time.monotonic() - preprocess_started) * 1000.0)
        with self._telemetry_lock:
            self._record_timing_locked("preprocess", preprocess_ms)
            self._telemetry["frames_sampled"] += 1
            self._telemetry["derived_frame_count"] += 2 + int(processed is not None)
            self._telemetry["derived_frame_bytes"] += int(
                color.nbytes
                + gray.nbytes
                + (processed.nbytes if processed is not None else 0)
            )
        with self.frame_lock:
            self.frames.append((frame_epoch, gray))
            self.color_frames.append((frame_epoch, color))
            self.evidence_frames.append(MotionEvidenceFrame(
                image=color,
                captured_at_epoch=frame_epoch,
                captured_at_monotonic=captured_at_monotonic,
                sequence=capture_sequence,
                capture_generation=capture_generation,
                lifecycle_generation=lifecycle_generation,
            ))
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
            self.evidence_frames.clear()
            self.processed_frames.clear()
            self.qualification_results.clear()
            self.recent_accepted_results.clear()
            self._last_submitted_ema_candidate_at = 0.0
            self.last_continuous_result = None
            self.last_processed_at = 0.0
            self.primary_last_processed_at = 0.0
            self._pending_analysis_at = 0.0
            self._analysis_request_deferred = False
        self.limiter.cancel(self.camera_id)
        with self._visual_lock:
            self.ema_v2.reset()
            self.ema_verification.reset()
        self.events.reset_timebase()
        with self._telemetry_lock:
            self._telemetry["clock_discontinuity_resets"] += 1
        self.qualification.reset_runtime(
            clear_observation_evidence=self.evidence.clear,
        )

    def samples(self) -> list[tuple[float, np.ndarray]]:
        with self.frame_lock:
            return [
                (timestamp, self._share_frame(frame, "samples"))
                for timestamp, frame in self.frames
            ]

    def evidence_frame_near(
        self,
        captured_at_epoch: float,
        *,
        sequence: int,
        capture_generation: int,
        lifecycle_generation: int,
        maximum_delta_seconds: float = 1.0,
    ) -> MotionEvidenceFrame | None:
        """Select the intended EMA frame without crossing a capture generation."""
        with self.frame_lock:
            candidates = tuple(self.evidence_frames)
        if sequence > 0:
            exact = next(
                (
                    item
                    for item in candidates
                    if item.sequence == sequence
                    and item.capture_generation == capture_generation
                    and item.lifecycle_generation == lifecycle_generation
                ),
                None,
            )
            if (
                exact is not None
                and abs(exact.captured_at_epoch - captured_at_epoch)
                <= maximum_delta_seconds
            ):
                return exact
        compatible = [
            item
            for item in candidates
            if item.capture_generation == capture_generation
            and item.lifecycle_generation == lifecycle_generation
        ]
        if not compatible:
            return None
        nearest = min(
            compatible,
            key=lambda item: abs(item.captured_at_epoch - captured_at_epoch),
        )
        if abs(nearest.captured_at_epoch - captured_at_epoch) > maximum_delta_seconds:
            return None
        return nearest

    def samples_since(self, captured_at: float) -> list[tuple[float, np.ndarray]]:
        with self.frame_lock:
            return [
                (timestamp, self._share_frame(frame, "samples_since"))
                for timestamp, frame in self.frames
                if timestamp >= captured_at
            ]

    def qualification_results_since(
        self,
        captured_at: float,
    ) -> list[tuple[float, MotionQualificationResult]]:
        """Return already-computed EMA results without replaying frame state."""
        with self.frame_lock:
            return [
                (timestamp, result)
                for timestamp, result in self.qualification_results
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
        elapsed_seconds = max(
            0.001,
            time.monotonic() - self._telemetry_started_monotonic,
        )
        result["copy_mb_per_second"] = round(
            float(result.get("copy_bytes") or 0)
            / 1_000_000.0
            / elapsed_seconds,
            3,
        )
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
            return self.ema_v2.snapshot()

    def visual_backup_readiness(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> bool:
        with self._visual_lock:
            return self.ema_v2.evaluate(
                result,
                captured_at,
                time.monotonic(),
                self.qualification.visual_backup_policy(),
                detection_enabled=True,
            ).scene_ready

    @property
    def visual_backup_scene_ready(self) -> bool:
        with self._visual_lock:
            return self.ema_v2.scene_ready

    @property
    def visual_backup_stable_samples(self) -> int:
        with self._visual_lock:
            return self.ema_v2.observation_count

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
                    self.state.increment_stat("analysis_worker_errors", 1)
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
                    self.state.increment_stat("analysis_worker_errors", 1)
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
                        captured_at_monotonic=work.captured_at_monotonic,
                        capture_sequence=work.capture_sequence,
                        capture_generation=work.capture_generation,
                        lifecycle_generation=work.lifecycle_generation,
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
                self.qualification.observe_frame(frame, captured_at)
                if self.qualification.continuous_primary_required() and (
                    self.qualification.continuous_primary_due(
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
                    self.capture_debug(captured_at)
            except Exception:
                self.state.increment_stat("analysis_worker_errors", 1)
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
        if not self.qualification.continuous_primary_required():
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
            self.state.record_analysis_wait(max(0.0, wait_seconds * 1000.0))
            qualification_started = time.monotonic()
            try:
                self.analyze_continuous(captured_at)
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
            self.state.increment_stat("analysis_frames_dropped", superseded)
            with self._telemetry_lock:
                self._telemetry["mailbox_replacements"] += superseded
        return work

    def analyze_continuous(self, captured_at: float) -> None:
        _mode, sensitivity, _frame_width = self.qualification.settings()
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
        
        # Temporal filtering: skip analysis if frame is stable (no significant change)
        if len(samples) >= 2:
            prev_frame = samples[-2][1]
            curr_frame = samples[-1][1]
            
            # Convert to grayscale if needed for comparison
            if prev_frame.ndim == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
                curr_gray = curr_frame
            
            # Calculate pixel-wise difference
            frame_diff = cv2.absdiff(prev_gray, curr_gray)

            # Count pixels with significant change (> 5 value difference)
            # Use NumPy here instead of cv2.countNonZero on a boolean mask, which
            # OpenCV rejects with a type error.
            changed_pixels = int(np.count_nonzero(frame_diff > 5))
            total_pixels = frame_diff.size
            pixel_change_ratio = changed_pixels / max(1, total_pixels)
            
            # If frame is stable, skip expensive analysis
            if pixel_change_ratio < self.temporal_filter_threshold:
                with self._telemetry_lock:
                    self._telemetry["temporal_filter_skips"] = self._telemetry.get(
                        "temporal_filter_skips", 0
                    ) + 1
                stable_result = MotionQualificationResult(
                    accepted=False,
                    score=0.0,
                    threshold=1.0,
                    reason="temporal_stable_scene",
                    frame_count=len(samples),
                    features={"pixel_change_ratio": pixel_change_ratio},
                    telemetry={},
                )
                with self.frame_lock:
                    # A cheap temporal observation still satisfies the primary
                    # cadence and must not leave every subsequent quiet frame due.
                    self.primary_last_processed_at = captured_at
                if self.qualification.trigger_mode() == "camera_rescue":
                    self._observe_stable_scene_for_visual_backup(
                        stable_result,
                        captured_at,
                    )
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
                result = self.qualification.run_pipeline(
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
        persist_candidate: dict[str, Any] | None = None
        with self.frame_lock:
            self.last_continuous_result = result
            self.primary_last_processed_at = captured_at
            if (
                self.qualification_results
                and self.qualification_results[-1][0] == captured_at
            ):
                self.qualification_results[-1] = (captured_at, result)
            else:
                self.qualification_results.append((captured_at, result))
            if (
                result.accepted
                and result.reason not in VISUAL_BACKUP_EXCLUDED_REASONS
            ):
                self.recent_accepted_results.append((captured_at, result))
                if (
                    self._ema_candidate_sink is not None
                    and captured_at - self._last_submitted_ema_candidate_at >= 0.5
                ):
                    persist_candidate = {
                        "accepted": bool(result.accepted),
                        "score": float(result.score),
                        "threshold": float(result.threshold),
                        "reason": str(result.reason),
                        "frame_count": int(result.frame_count),
                        "features": dict(result.features),
                        "telemetry": dict(result.telemetry),
                    }
        self._record_continuous_stats(result)
        if self._stopping():
            return
        trigger_mode = self.qualification.trigger_mode()
        if trigger_mode in {"camera_rescue", "adaptive"}:
            self.consider_visual_backup(
                result,
                samples,
                captured_at,
                trigger_mode=trigger_mode,
            )
            self._persist_ema_route_candidate(captured_at, persist_candidate)
            return

    def _observe_stable_scene_for_visual_backup(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> None:
        """Advance visual-backup readiness without running route/audit work."""
        observed_monotonic = time.monotonic()
        with self._visual_lock:
            policy = self.qualification.visual_backup_policy()
            detection_enabled = self.state.detection_enabled()
            self.ema_v2.evaluate(
                result,
                captured_at,
                observed_monotonic,
                policy,
                detection_enabled=detection_enabled,
            )
            self.ema_verification.evaluate(
                result,
                captured_at,
                observed_monotonic,
                replace(
                    policy,
                    minimum_score=min(
                        float(policy.minimum_score),
                        float(result.threshold),
                    ),
                    score_margin=0.0,
                    minimum_consecutive=policy.minimum_consecutive + 2,
                    grace_seconds=policy.grace_seconds + 1.0,
                ),
                detection_enabled=detection_enabled,
            )

    def _persist_ema_route_candidate(
        self,
        captured_at: float,
        payload: dict[str, Any] | None,
    ) -> None:
        if payload is None or self._ema_candidate_sink is None:
            return
        # Admission has already run. This optional restart cache must never
        # stand in front of immediate detector work or wait materially on the
        # security ledger's writer.
        self._last_submitted_ema_candidate_at = captured_at
        try:
            outcome = self._ema_candidate_sink(self.camera_id, captured_at, payload)
            if outcome in {
                EmaCandidateSubmitResult.STOPPED,
                EmaCandidateSubmitResult.OVERSIZE_DROPPED,
            }:
                self.state.increment_stat("ema_candidate_persist_failures", 1)
            elif outcome is EmaCandidateSubmitResult.OVERFLOW_DROPPED:
                self.state.increment_stat("ema_candidate_overflow_drops", 1)
        except Exception:
            self.state.increment_stat("ema_candidate_persist_failures", 1)
            now_monotonic = time.monotonic()
            if (
                now_monotonic - self._last_ema_candidate_failure_log_monotonic
                >= 60.0
            ):
                self._last_ema_candidate_failure_log_monotonic = now_monotonic
                LOGGER.exception(
                    "could not persist EMA route candidate for %s",
                    self.camera_id,
                )

    def consider_visual_backup(
        self,
        result: MotionQualificationResult,
        samples: list[tuple[float, np.ndarray]],
        captured_at: float,
        *,
        trigger_mode: str = "camera_rescue",
    ) -> None:
        if self._stopping():
            return
        settings = self.qualification.visual_backup_settings()
        illumination_probe_allowed = bool(
            self.state.detection_enabled()
            and result.reason == "illumination_change"
            and result.features.get("illumination_would_reject")
            and should_verify_suppression(
                (
                    f"illumination:{self.camera_id}:"
                    f"{int(captured_at // max(5.0, float(settings['cooldown_seconds'])))}"
                ),
                self.qualification.suppression_verification_rate(),
            )
        )
        if illumination_probe_allowed:
            result = MotionQualificationResult(
                accepted=True,
                score=result.score,
                threshold=result.threshold,
                reason="illumination_verification_probe",
                frame_count=result.frame_count,
                features={**result.features, "illumination_verification_probe": True},
                telemetry=dict(result.telemetry),
            )
        observed_monotonic = time.monotonic()
        if trigger_mode == "adaptive":
            if not self.state.detection_enabled() or not result.accepted:
                return
            qualified = EmaQualified(
                camera_id=self.camera_id,
                captured_at=captured_at,
                observed_monotonic=observed_monotonic,
                result=result,
                required_score=result.threshold,
                qualifying_samples=1,
                window_samples=1,
                candidate_started_at=captured_at,
            )
            decision = EmaSignalDecision(
                action=EmaSignalAction.QUALIFIED,
                result=result,
                required_score=result.threshold,
                scene_ready=True,
                qualifying_samples=1,
                window_samples=1,
                qualified=qualified,
            )
        else:
            with self._visual_lock:
                policy = self.qualification.visual_backup_policy()
                decision = self.ema_v2.evaluate(
                    result,
                    captured_at,
                    observed_monotonic,
                    policy,
                    detection_enabled=self.state.detection_enabled(),
                )
                try:
                    watch = (
                        self._route_watch_provider(self.camera_id, captured_at)
                        if self._route_watch_provider is not None
                        else None
                    )
                except Exception:
                    watch = None
                    LOGGER.exception(
                        "route detection watch lookup failed for %s",
                        self.camera_id,
                    )
                try:
                    effectiveness = (
                        self._onvif_effectiveness_provider()
                        if self._onvif_effectiveness_provider is not None
                        else {}
                    )
                except Exception:
                    effectiveness = {}
                    LOGGER.exception(
                        "ONVIF effectiveness lookup failed for %s",
                        self.camera_id,
                    )
                onvif_degraded = bool(effectiveness.get("signal_degraded"))
                onvif_unavailable = bool(
                    effectiveness.get("signal_effectiveness_status")
                    == "transport_unavailable"
                )
                verification_reason = (
                    "route_watch"
                    if watch is not None
                    else "onvif_degraded"
                    if onvif_degraded
                    else "onvif_unavailable"
                    if onvif_unavailable
                    else "persistent_ema"
                )
                watch_payload = (
                    watch.as_dict()
                    if watch is not None and hasattr(watch, "as_dict")
                    else {}
                )
                verification_result = MotionQualificationResult(
                    accepted=result.accepted,
                    score=result.score,
                    threshold=result.threshold,
                    reason=result.reason,
                    frame_count=result.frame_count,
                    features={
                        **result.features,
                        "security_verification": True,
                        "security_verification_bypass_limits": watch is not None,
                        "security_verification_reason": verification_reason,
                        "route_detection_watch": watch_payload,
                        "onvif_signal_effectiveness_status": effectiveness.get(
                            "signal_effectiveness_status"
                        ),
                    },
                    telemetry=dict(result.telemetry),
                )
                # The ordinary edge remains the efficient primary path.  A
                # second conditioner guarantees that accepted, persistent
                # motion is eventually verified even below the operator's
                # rescue score.  Route evidence or proven ONVIF degradation
                # accelerates that verification; otherwise it requires more
                # persistence to keep nuisance cost bounded.
                verification_policy = replace(
                    policy,
                    minimum_score=min(
                        float(policy.minimum_score),
                        float(result.threshold),
                    ),
                    score_margin=0.0,
                    minimum_consecutive=(
                        1
                        if watch is not None
                        else policy.minimum_consecutive
                        if onvif_degraded or onvif_unavailable
                        else policy.minimum_consecutive + 2
                    ),
                    grace_seconds=(
                        0.0
                        if watch is not None
                        else policy.grace_seconds
                        if onvif_degraded or onvif_unavailable
                        else policy.grace_seconds + 1.0
                    ),
                )
                verification = self.ema_verification.evaluate(
                    verification_result,
                    captured_at,
                    observed_monotonic,
                    verification_policy,
                    detection_enabled=self.state.detection_enabled(),
                )
                route_verification = bool(
                    watch is not None
                    and verification_result.accepted
                    and verification_result.reason
                    not in VISUAL_BACKUP_EXCLUDED_REASONS
                    and self.state.detection_enabled()
                )
                if route_verification:
                    qualified = EmaQualified(
                        camera_id=self.camera_id,
                        captured_at=captured_at,
                        observed_monotonic=observed_monotonic,
                        result=verification_result,
                        required_score=float(result.threshold),
                        qualifying_samples=1,
                        window_samples=1,
                        candidate_started_at=captured_at,
                    )
                    decision = EmaSignalDecision(
                        action=EmaSignalAction.QUALIFIED,
                        result=verification_result,
                        required_score=float(result.threshold),
                        scene_ready=True,
                        qualifying_samples=1,
                        window_samples=1,
                        qualified=qualified,
                    )
                    self.ema_v2.clear_candidate()
                    self.ema_verification.clear_candidate()
                elif decision.action is EmaSignalAction.QUALIFIED:
                    self.ema_verification.clear_candidate()
                elif verification.action is EmaSignalAction.QUALIFIED:
                    self.ema_v2.clear_candidate()
                    decision = verification
        result = decision.result
        if decision.action in {EmaSignalAction.DISABLED, EmaSignalAction.REJECTED}:
            if decision.count_nonpromotion:
                self.state.increment_stat("visual_backup_not_promoted", 1)
                self._observe_visual_backup_nonpromotion(
                    result,
                    decision.required_score,
                    samples,
                    captured_at,
                    settings,
                )
            else:
                self._flush_visual_backup_nonpromotion(settings)
            return
        # A stronger candidate, a matching camera notification, or startup
        # learning means the preceding low-score observations were not a
        # completed missed-rescue episode.
        self._discard_visual_backup_nonpromotion()
        if decision.action == EmaSignalAction.LEARNING:
            self.state.increment_stat("visual_backup_not_ready", 1)
            if decision.readiness_audit_needed:
                self.record_visual_backup_readiness_audit(result, captured_at)
            return
        if decision.action in {EmaSignalAction.ACCUMULATING, EmaSignalAction.QUALIFIED}:
            self.state.increment_stat("visual_backup_candidates", 1)
        if decision.action == EmaSignalAction.ACCUMULATING:
            return
        if decision.action != EmaSignalAction.QUALIFIED or decision.qualified is None:
            return
        route_details = result.features.get("route_detection_watch")
        standalone_route = bool(
            result.features.get("security_verification_bypass_limits")
            and isinstance(route_details, dict)
            and route_details.get("source_camera_id")
            and route_details.get("source_event_id")
        )
        episode = None
        if standalone_route:
            # A route observation represents a distinct physical occurrence.
            # It must not be absorbed by an unrelated request that happens to
            # own the camera's ordinary EMA episode.  The upstream durable
            # event makes this identity stable across retries and restarts.
            route_identity = (
                f"route:{self.camera_id}:"
                f"{route_details['source_camera_id']}:"
                f"{int(route_details['source_event_id'])}"
            )
            intent = DetectionIntent(
                intent_id=route_identity,
                episode_id=route_identity,
                camera_id=self.camera_id,
                generation=self.state.lifecycle_generation(),
                event_at=captured_at,
                created_monotonic=observed_monotonic,
                primary_source=MotionSource.EMA,
                sources=(MotionSource.EMA,),
                ema=decision.qualified,
            )
        else:
            episode = self.events.episode_controller.observe_ema(
                decision.qualified,
                generation=self.state.lifecycle_generation(),
            )
            intent = episode.intent
        matched_onvif = bool(
            intent is not None
            and MotionSource.CAMERA in intent.sources
        )
        observer = self._onvif_effectiveness_observer
        if observer is not None and not standalone_route:
            try:
                observer(matched_onvif, captured_at)
            except Exception:
                # Health instrumentation must never interfere with detection.
                LOGGER.exception(
                    "ONVIF effectiveness observer failed for %s",
                    self.camera_id,
                )
        if episode is not None and episode.reason is EpisodeDecisionReason.MERGED_WITH_REQUEST:
            if (
                episode.intent is not None
                and MotionSource.CAMERA in episode.intent.sources
            ):
                self.state.increment_stat("visual_backup_onvif_matches", 1)
            elif trigger_mode == "adaptive":
                self.state.increment_stat("adaptive_triggers_deferred", 1)
            return
        followup = bool(
            episode is not None
            and episode.reason is EpisodeDecisionReason.FOLLOWUP_RESERVED
        )
        if episode is not None and episode.reason in {
            EpisodeDecisionReason.FOLLOWUP_DUPLICATE,
            EpisodeDecisionReason.FOLLOWUP_RATE_LIMITED,
            EpisodeDecisionReason.FOLLOWUP_LIMIT_REACHED,
        }:
            stat = {
                EpisodeDecisionReason.FOLLOWUP_DUPLICATE: "active_followup_deduplicated",
                EpisodeDecisionReason.FOLLOWUP_RATE_LIMITED: "active_followup_rate_limited",
                EpisodeDecisionReason.FOLLOWUP_LIMIT_REACHED: "active_followup_episode_limited",
            }[episode.reason]
            self.state.increment_stat(stat, 1)
            return
        if episode is not None and episode.reason is EpisodeDecisionReason.EMA_RATE_LIMITED:
            self.state.increment_stat("visual_backup_rate_limited", 1)
            return
        if episode is not None and episode.reason not in {
            EpisodeDecisionReason.REQUEST_RESERVED,
            EpisodeDecisionReason.FOLLOWUP_RESERVED,
        }:
            self.state.increment_stat("visual_backup_not_promoted", 1)
            return
        if intent is None:
            return

        trigger_enqueued = False
        try:
            if self._stopping():
                return
            fused = MotionQualificationResult(
                accepted=True,
                score=result.score,
                threshold=result.threshold,
                reason=(
                    result.reason
                    if result.reason == "illumination_verification_probe"
                    else "ema_v2_qualified"
                ),
                frame_count=result.frame_count,
                features={
                    **result.features,
                    "visual_backup": trigger_mode == "camera_rescue" and not followup,
                    "active_event_followup": followup,
                    "ema_v2": True,
                    "motion_episode_id": intent.episode_id,
                    "visual_backup_required_score": round(
                        decision.required_score, 4
                    ),
                    "visual_backup_consecutive": decision.qualifying_samples,
                    "visual_backup_window_samples": decision.window_samples,
                    "visual_backup_grace_seconds": settings["grace_seconds"],
                },
                telemetry=dict(result.telemetry),
            )
            self.last_continuous_result = fused
            event_at = datetime.fromtimestamp(captured_at, timezone.utc)
            with self.frame_lock:
                compatible_evidence = tuple(
                    item
                    for item in self.evidence_frames
                    if item.lifecycle_generation == intent.generation
                    and item.capture_generation > 0
                )
            nearest_evidence = min(
                compatible_evidence,
                key=lambda item: abs(item.captured_at_epoch - captured_at),
                default=None,
            )
            evidence_frame = (
                nearest_evidence
                if nearest_evidence is not None
                and abs(nearest_evidence.captured_at_epoch - captured_at) <= 1.0
                else None
            )
            trigger_enqueued = self._enqueue_trigger(
                MotionTrigger(
                    topic=(
                        "adaptive/active_followup"
                        if followup
                        else "adaptive/visual_backup"
                        if trigger_mode == "camera_rescue"
                        else "adaptive/motion"
                    ),
                    message=(
                        "new credible motion during active EMA episode"
                        if followup
                        else "adaptive visual backup after missing camera notice"
                        if trigger_mode == "camera_rescue"
                        else "adaptive motion episode"
                    ),
                    event_at=event_at,
                    received_at=captured_at,
                    prequalified=fused,
                    episode_id=intent.episode_id,
                    detection_intent_id=intent.intent_id,
                    lifecycle_generation=intent.generation,
                    evidence_frame_at_epoch=(
                        evidence_frame.captured_at_epoch
                        if evidence_frame is not None
                        else captured_at
                    ),
                    evidence_frame_sequence=(
                        evidence_frame.sequence if evidence_frame is not None else 0
                    ),
                    evidence_capture_generation=(
                        evidence_frame.capture_generation
                        if evidence_frame is not None
                        else 0
                    ),
                )
            )
            if not standalone_route:
                self.events.episode_controller.acknowledge_admission(
                    intent.intent_id,
                    admitted=trigger_enqueued,
                    occurred_monotonic=time.monotonic(),
                )
            if not trigger_enqueued and not standalone_route:
                if followup:
                    self.state.increment_stat("active_followup_queue_rejected", 1)
                return
            self.state.increment_stat(
                "active_followup_triggers" if followup else "visual_backup_triggers",
                1,
            )
            if result.features.get("security_verification"):
                self.state.increment_stat("security_verification_triggers", 1)
                if result.features.get("route_detection_watch"):
                    self.state.increment_stat("route_watch_triggers", 1)
            self.state.increment_stat(
                "illumination_verification_probes",
                int(illumination_probe_allowed),
            )
            self._publish_motion(
                event_at,
                "active_followup"
                if followup
                else "visual_backup"
                if trigger_mode == "camera_rescue"
                else "adaptive",
            )
        finally:
            if not trigger_enqueued:
                try:
                    self.events.episode_controller.acknowledge_admission(
                        intent.intent_id,
                        admitted=False,
                        occurred_monotonic=time.monotonic(),
                    )
                except ValueError:
                    pass

    def consider_route_watch(self, watch: Any) -> bool:
        """Replay recent accepted EMA evidence when route confirmation is late."""
        if self._stopping() or getattr(watch, "target_camera_id", "") != self.camera_id:
            return False
        eligible_at = float(getattr(watch, "eligible_at", 0.0))
        expires_at = float(getattr(watch, "expires_at", 0.0))
        with self.frame_lock:
            candidates = [
                item
                for item in self.recent_accepted_results
                if eligible_at <= item[0] <= expires_at
                and item[1].reason not in VISUAL_BACKUP_EXCLUDED_REASONS
            ]
            color_samples = [
                (captured_at, frame)
                for captured_at, frame in self.color_frames
                if eligible_at <= captured_at <= expires_at
            ]
        if self._ema_candidate_source is not None:
            try:
                durable_candidates = self._ema_candidate_source(
                    self.camera_id,
                    eligible_at,
                    expires_at,
                )
            except Exception:
                durable_candidates = []
                LOGGER.exception(
                    "could not load EMA route candidates for %s",
                    self.camera_id,
                )
            known_times = {captured_at for captured_at, _result in candidates}
            for candidate_at, payload in durable_candidates:
                if candidate_at in known_times or not isinstance(payload, dict):
                    continue
                try:
                    restored = MotionQualificationResult(
                        accepted=bool(payload.get("accepted")),
                        score=float(payload.get("score") or 0.0),
                        threshold=float(payload.get("threshold") or 0.0),
                        reason=str(payload.get("reason") or "qualified"),
                        frame_count=int(payload.get("frame_count") or 0),
                        features=(
                            dict(payload.get("features") or {})
                            if isinstance(payload.get("features"), dict)
                            else {}
                        ),
                        telemetry=(
                            dict(payload.get("telemetry") or {})
                            if isinstance(payload.get("telemetry"), dict)
                            else {}
                        ),
                    )
                except (TypeError, ValueError):
                    continue
                if (
                    restored.accepted
                    and restored.reason not in VISUAL_BACKUP_EXCLUDED_REASONS
                ):
                    candidates.append((float(candidate_at), restored))
        if not candidates:
            self.state.increment_stat("route_watch_without_recent_ema", 1)
            return False
        captured_at, result = max(
            candidates,
            key=lambda item: (float(item[1].score), item[0]),
        )
        self.consider_visual_backup(
            result,
            color_samples,
            captured_at,
            trigger_mode="camera_rescue",
        )
        return True

    def capture_debug(self, captured_at: float) -> None:
        samples = self.samples()
        frames = [frame for _timestamp, frame in samples]
        if len(frames) < 2:
            return
        _mode, sensitivity, _frame_width = self.qualification.settings()
        try:
            self.qualification.run_pipeline(
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
        self.state.increment_stat("continuous_frames", 1)
        self.state.increment_stat("continuous_candidates", int(result.accepted))
        illumination_available = bool(
            result.features.get("illumination_evidence_available")
        )
        illumination_candidate = bool(result.features.get("illumination_would_reject"))
        self.state.increment_stat("illumination_evaluations", int(illumination_available))
        self.state.increment_stat("illumination_candidates", int(illumination_candidate))
        self.state.increment_stat(
            "illumination_filtered",
            int(
                illumination_candidate
                and self.qualification.illumination_filter_enabled()
                and not result.accepted
            ),
        )

    def _enqueue_trigger(self, trigger: MotionTrigger) -> bool:
        return self.events.enqueue(
            trigger,
            evict_oldest=False,
            on_trigger=lambda name: self.state.increment_stat(name, 1),
            on_drop=lambda name: self.state.increment_stat(name, 1),
        )

    def record_visual_backup_readiness_audit(
        self,
        result: MotionQualificationResult,
        captured_at: float,
    ) -> None:
        event_at = datetime.fromtimestamp(captured_at, timezone.utc)
        try:
            self.audit_recorder.record_audit(
                snapshot_path=self.media.sample_rejected_motion(event_at, result),
                event_at=event_at,
                mode="camera_rescue",
                sensitivity=self.qualification.settings()[1],
                score=result.score,
                threshold=result.threshold,
                reason="startup_not_ready",
                object_detected=None,
                trigger_count=0,
                features={
                    **audit_features(result),
                    "visual_backup_scene_ready": False,
                    "visual_backup_warmup_seconds": (
                        self.config.visual_backup_warmup_seconds
                    ),
                },
                category="visual_backup",
            )
        except Exception:
            LOGGER.exception(
                "failed to record visual backup readiness audit for %s",
                self.camera_id,
            )

    def _observe_visual_backup_nonpromotion(
        self,
        result: MotionQualificationResult,
        required_score: float,
        samples: list[tuple[float, np.ndarray]],
        captured_at: float,
        settings: dict[str, float | int],
    ) -> None:
        """Accumulate only credible motion that specifically missed the score gate."""
        if result.score >= required_score or not samples:
            self._flush_visual_backup_nonpromotion(settings)
            return
        frame = samples[-1][1]
        expected_interval = 1.0 / max(
            0.5,
            min(self.config.sample_fps, self.config.camera_mode_background_fps),
        )
        episode = self._visual_nonpromotion_episode
        if (
            episode is not None
            and captured_at - episode.last_seen_at > expected_interval * 2.5
        ):
            self._flush_visual_backup_nonpromotion(settings)
            episode = None
        if episode is None:
            self._visual_nonpromotion_episode = _VisualBackupNonpromotionEpisode(
                started_at=captured_at,
                last_seen_at=captured_at,
                observation_count=1,
                peak_at=captured_at,
                peak_result=result,
                peak_required_score=required_score,
                peak_frame=frame,
            )
            return
        episode.last_seen_at = captured_at
        episode.observation_count += 1
        if result.score > episode.peak_result.score:
            episode.peak_at = captured_at
            episode.peak_result = result
            episode.peak_required_score = required_score
            episode.peak_frame = frame

    def _discard_visual_backup_nonpromotion(self) -> None:
        self._visual_nonpromotion_episode = None

    def _flush_visual_backup_nonpromotion(
        self,
        settings: dict[str, float | int],
    ) -> None:
        episode = self._visual_nonpromotion_episode
        self._visual_nonpromotion_episode = None
        if episode is None:
            return
        minimum_observations = max(2, int(settings["minimum_consecutive"]))
        if episode.observation_count < minimum_observations:
            return
        event_at = datetime.fromtimestamp(episode.peak_at, timezone.utc)
        try:
            snapshot_path = self.media.sample_rejected_motion_frame(
                event_at,
                episode.peak_result,
                episode.peak_frame,
            )
            self.audit_recorder.record_audit(
                snapshot_path=snapshot_path,
                event_at=event_at,
                mode="camera_rescue",
                sensitivity=self.qualification.settings()[1],
                score=episode.peak_result.score,
                threshold=episode.peak_result.threshold,
                reason="visual_backup_below_threshold",
                object_detected=None,
                trigger_count=0,
                features={
                    **audit_features(episode.peak_result),
                    "visual_backup_scene_ready": True,
                    "visual_backup_required_score": round(
                        episode.peak_required_score,
                        4,
                    ),
                    "visual_backup_episode_started_at": datetime.fromtimestamp(
                        episode.started_at,
                        timezone.utc,
                    ).isoformat(),
                    "visual_backup_episode_ended_at": datetime.fromtimestamp(
                        episode.last_seen_at,
                        timezone.utc,
                    ).isoformat(),
                    "visual_backup_episode_duration_seconds": round(
                        max(0.0, episode.last_seen_at - episode.started_at),
                        3,
                    ),
                    "visual_backup_credible_frames": episode.observation_count,
                    "visual_backup_peak_at": event_at.isoformat(),
                    "visual_backup_camera_notice_received": False,
                },
                category="visual_backup",
            )
        except Exception:
            LOGGER.exception(
                "failed to record visual backup non-promotion audit for %s",
                self.camera_id,
            )

    def _publish_motion(self, event_at: datetime, source: str) -> None:
        timestamp = event_at.isoformat()
        self.state.set_last_motion_at(timestamp)
        self.state.publish_event(
            "motion",
            MotionObserved(
                camera_id=self.camera_id,
                timestamp=timestamp,
                source=source,
            ).to_payload(),
        )

    def _stopping(self) -> bool:
        return bool(
            self._stop_requested.is_set()
            or (self._stop_event is not None and self._stop_event.is_set())
        )
