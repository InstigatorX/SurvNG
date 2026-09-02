from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any

import numpy as np

from ..config import CameraConfig, ObjectTrackingConfig
from ..detector import detection_failure
from ..domain_events import TrackingCompleted
from ..security import redact_secret_text
from ..video_frames import DecodedVideoFrame, VideoFrameReference
from ..visual_quality import image_quality
from ..zones import apply_depth_zone_filters, apply_detection_zones
from .geometry import (
    _box,
    _confidence,
    _detect_tracking_objects,
    _encode_appearance,
    _encoder_supports_label,
    _inference_deferred,
    _iou,
    _labeled_tracking_objects,
    _rescale_detection_boxes,
)
from .registry import ObjectTrackerRegistry, build_builtin_object_tracker_registry
from .types import (
    AppearanceEncoder,
    AppearanceIndexWriter,
    CatchupFrameProvider,
    FrameProvider,
    FrameSample,
    ObjectDetectorBackend,
    TrackingCoverFrameProvider,
    TrackingCoverPromoter,
    TrackingPublisher,
    TrackingSnapshotWriter,
    TrackingUpdate,
    LiveDetectionsProvider,
)

LOGGER = logging.getLogger("survng.app.object_tracking")
TRACKING_STOP_TIMEOUT_SECONDS = 18.0
TRACKING_CATCHUP_SETTLE_SECONDS = 5.0
TRACKING_CATCHUP_RETRY_SECONDS = 0.25
# Small open-segment handoff gaps are expected; escalate only when large
# or repeated, or when catch-up itself fails (exception path below).
COVERAGE_GAP_WARNING_SECONDS = 10.0
# Keep aligned with tracking_frames.TRACKING_OPEN_SEGMENT_BRIDGE_SECONDS.
# Anything older cannot be reconstructed from retained live history.
TRACKING_MAX_RECOVERABLE_HANDOFF_AGE_SECONDS = 12.0


def _adaptive_tracking_fps(
    config: ObjectTrackingConfig,
    tracked_objects: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    important_transition: bool,
    stable_frames: int,
) -> tuple[float, int]:
    uncertain = (
        important_transition
        or not tracked_objects
        or any(
            item.get("track_state") != "confirmed"
            for item in tracked_objects
        )
        or any(item.get("state") != "confirmed" for item in summaries)
    )
    next_stable_frames = 0 if uncertain else stable_frames + 1
    if (
        config.adaptive_sampling_enabled
        and next_stable_frames >= config.adaptive_stable_frames
    ):
        return config.stable_sample_fps, next_stable_frames
    return config.sample_fps, next_stable_frames


def _tracking_persistence_due(
    interval_seconds: float,
    last_persisted_at: float,
    now: float,
    *,
    important_transition: bool,
) -> bool:
    return (
        important_transition
        or interval_seconds == 0
        or now - last_persisted_at >= interval_seconds
    )


@dataclass(frozen=True, slots=True)
class _TrackingCoverCandidate:
    captured_at: float
    tracked_objects: tuple[dict[str, Any], ...]
    primary_track_id: int
    subject_area_ratio: float
    edge_clearance_ratio: float
    quality_score: float
    detection_confidence: float
    fully_framed: bool
    frame_reference: VideoFrameReference | None = None

    @property
    def score(self) -> tuple[int, float, float, float, float]:
        return (
            int(self.fully_framed),
            self.subject_area_ratio,
            self.quality_score,
            self.detection_confidence,
            self.captured_at,
        )

class ObjectTrackingSession:
    def __init__(
        self,
        camera: CameraConfig,
        config: ObjectTrackingConfig,
        detector: ObjectDetectorBackend,
        frame_provider: FrameProvider,
        update_event: TrackingUpdate,
        publisher: TrackingPublisher | None,
        limiter: threading.BoundedSemaphore,
        tracker_registry: ObjectTrackerRegistry | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
        appearance_indexer: AppearanceIndexWriter | None = None,
        catchup_frame_provider: CatchupFrameProvider | None = None,
        cover_frame_provider: TrackingCoverFrameProvider | None = None,
        snapshot_writer: TrackingSnapshotWriter | None = None,
        cover_promoter: TrackingCoverPromoter | None = None,
        live_detections_provider: LiveDetectionsProvider | None = None,
    ) -> None:
        self.camera = camera
        self.config = config
        self.detector = detector
        self.frame_provider = frame_provider
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder
        self.appearance_indexer = appearance_indexer
        self.catchup_frame_provider = catchup_frame_provider
        self.cover_frame_provider = cover_frame_provider
        self.snapshot_writer = snapshot_writer
        self.cover_promoter = cover_promoter
        self.live_detections_provider = live_detections_provider
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_id: int | None = None
        self._pending_start: tuple[
            int, datetime, list[dict[str, Any]], np.ndarray | None
        ] | None = None
        self._deadline = 0.0
        self._accepting = False
        self._status: dict[str, Any] = self._idle_status()
        self._status["enabled"] = config.enabled
        self._reid_recovery_base = 0
        self._reid_recovery_base_by_label: dict[str, int] = {}
        self._reid_avoided_base = 0
        self._reid_avoided_base_by_label: dict[str, int] = {}
        self._reid_attempt_base = 0
        self._reid_attempt_base_by_reason: dict[str, int] = {}
        self._frame_width = 0
        self._frame_height = 0
        self._catchup_frames_processed = 0
        self._coverage_gap_count = 0
        self._maximum_coverage_gap_seconds = 0.0
        self._completion_reason = ""
        self._effective_sample_fps = config.sample_fps
        self._cover_baseline: _TrackingCoverCandidate | None = None
        self._cover_candidate: _TrackingCoverCandidate | None = None
        self._cover_promotion: dict[str, Any] | None = None

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {
            "enabled": False,
            "active": False,
            "event_id": None,
            "track_count": 0,
            "confirmed_tracks": 0,
            "frames_processed": 0,
            "effective_sample_fps": 0.0,
            "catchup_frames_processed": 0,
            "coverage_gap_count": 0,
            "maximum_coverage_gap_seconds": 0.0,
            "coverage_incomplete": False,
            "completion_reason": "",
            "last_error": "",
            "reid_failures": 0,
            "reid_attempts": 0,
            "reid_successes": 0,
            "reid_inference_ms": 0.0,
            "reid_average_ms": 0.0,
            "reid_attempts_by_label": {},
            "reid_attempts_by_reason": {},
            "reid_recoveries": 0,
            "reid_recoveries_by_label": {},
            "reid_avoided_geometry_matches": 0,
            "reid_avoided_by_label": {},
            "last_reid_error": "",
            "appearance_vectors_indexed": 0,
            "appearance_index_error": "",
            "capacity_requests": 0,
            "capacity_waits": 0,
            "capacity_timeouts": 0,
            "capacity_wait_seconds_total": 0.0,
            "capacity_wait_seconds_max": 0.0,
            "capacity_wait_seconds_last": 0.0,
            "cover_promoted": False,
            "cover_promotion_attempted": False,
            "cover_promotion_result": "",
            "cover_verification_inference_ms": 0.0,
        }

    def _enrich_tracking_depth(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        *,
        frame_offset_s: float,
    ) -> list[dict[str, Any]]:
        """Sample depth for every accepted tracking frame when available.

        The isolated depth supervisor sheds overlapping optional work, so a
        busy device returns the unmodified detections instead of accumulating
        stale tracking samples.
        """
        if not objects or not getattr(
            getattr(self.detector.config, "depth", None),
            "enabled",
            False,
        ):
            return objects
        estimate_depth = getattr(
            self.detector,
            "estimate_depth_for_objects",
            None,
        )
        if not callable(estimate_depth):
            return objects
        enriched, _depth_metadata = estimate_depth(
            frame,
            objects,
            frame_offset_s=frame_offset_s,
        )
        apply_depth_zone_filters(self.camera, enriched)
        return enriched

    def _tracking_detections_for_frame(
        self,
        frame: np.ndarray,
        *,
        catchup: bool,
    ) -> list[dict[str, Any]]:
        """Live ticks use gvadetect sidecar boxes; catch-up still runs OpenVINO."""
        if not catchup and self.live_detections_provider is not None:
            return _labeled_tracking_objects(self.live_detections_provider())
        return _detect_tracking_objects(
            self.detector,
            frame,
            self.config.low_confidence_threshold,
        )

    def start(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None = None,
    ) -> bool:
        with self._transition_lock:
            return self._start_session(event_id, event_at, initial_objects, initial_frame)

    def _start_session(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None,
    ) -> bool:
        initial_objects = [
            item
            for item in initial_objects
            if self.config.tracks_label(item.get("label"))
        ]
        if not self.config.enabled or not any(
            item.get("label") and item.get("incident_eligible") is not False
            for item in initial_objects
        ):
            return False
        with self._lock:
            if not self._accepting:
                return False
            if self._thread is not None and self._thread.is_alive():
                if self._event_id == event_id:
                    self._deadline = max(self._deadline, time.monotonic() + self.config.max_session_seconds)
                    return True
                self._stop.set()
                # A blocked detector/catch-up read can take longer than the
                # handoff caller may wait.  Preserve only the newest incident;
                # the retiring worker starts it after its own cleanup.
                self._pending_start = (
                    event_id,
                    event_at,
                    [dict(item) for item in initial_objects],
                    initial_frame.copy() if initial_frame is not None else None,
                )
                LOGGER.info(
                    "object tracking for %s is stopping; queued latest event %d",
                    self.camera.id,
                    event_id,
                )
                return True
        with self._lock:
            if not self._accepting:
                return False
            self._stop = threading.Event()
            self._event_id = event_id
            self._deadline = time.monotonic() + self.config.max_session_seconds
            self._reid_recovery_base = int(self._status.get("reid_recoveries") or 0)
            self._reid_recovery_base_by_label = dict(
                self._status.get("reid_recoveries_by_label") or {}
            )
            self._reid_avoided_base = int(
                self._status.get("reid_avoided_geometry_matches") or 0
            )
            self._reid_avoided_base_by_label = dict(
                self._status.get("reid_avoided_by_label") or {}
            )
            self._reid_attempt_base = int(self._status.get("reid_attempts") or 0)
            self._reid_attempt_base_by_reason = dict(
                self._status.get("reid_attempts_by_reason") or {}
            )
            thread = threading.Thread(
                target=self._run,
                args=(
                    event_id,
                    event_at,
                    [dict(item) for item in initial_objects],
                    initial_frame.copy() if initial_frame is not None else None,
                    self._stop,
                ),
                name=f"object-tracking-{self.camera.id}",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._event_id = None
                raise
            return True

    def set_accepting(self, accepting: bool) -> None:
        with self._lock:
            self._accepting = bool(accepting and self.config.enabled)
            if not self._accepting:
                self._pending_start = None
        if not accepting:
            self.stop()

    def stop(self) -> bool:
        self.request_stop()
        return self.wait_stopped(TRACKING_STOP_TIMEOUT_SECONDS)

    def request_stop(self) -> None:
        """Stop accepting work and signal the session without joining it."""
        with self._lock:
            self._accepting = False
            self._pending_start = None
            self._stop.set()

    def wait_stopped(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the signalled session to exit."""
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
            if thread.is_alive():
                LOGGER.error("object tracking worker did not stop for %s", self.camera.id)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
                self._event_id = None
            return self._thread is None or not self._thread.is_alive()

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread is not None and self._thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._lock:
            pending_event_id = (
                int(self._pending_start[0]) if self._pending_start is not None else None
            )
            return {
                **self._status,
                "accepting": self._accepting,
                "worker_running": bool(self._thread is not None and self._thread.is_alive()),
                # Distinct from active event_id: handoff returned True but the
                # successor has not started until the prior worker finishes.
                "pending_event_id": pending_event_id,
            }

    def _consider_cover_candidate(
        self,
        frame: np.ndarray,
        captured_at: float,
        tracked_objects: list[dict[str, Any]],
        primary_track_ids: set[int],
        frame_reference: VideoFrameReference | None = None,
    ) -> None:
        """Remember metadata for the best later view without retaining its frame."""
        if (
            frame is None
            or not frame.size
            or self._frame_width <= 0
            or self._frame_height <= 0
        ):
            return
        primary: list[tuple[dict[str, Any], Box, float]] = []
        frame_area = float(self._frame_width * self._frame_height)
        for item in tracked_objects:
            try:
                track_id = int(item.get("track_id"))
            except (TypeError, ValueError):
                continue
            box = _box(item.get("box"))
            if track_id not in primary_track_ids or box is None:
                continue
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
            primary.append((item, box, area))
        if not primary:
            return
        item, box, area = max(primary, key=lambda value: value[2])
        clearance = max(
            0.0,
            min(
                box[0] / self._frame_width,
                box[1] / self._frame_height,
                (self._frame_width - box[2]) / self._frame_width,
                (self._frame_height - box[3]) / self._frame_height,
            ),
        )
        source_height, source_width = frame.shape[:2]
        scale_x = source_width / self._frame_width
        scale_y = source_height / self._frame_height
        x1 = max(0, min(source_width, int(np.floor(box[0] * scale_x))))
        y1 = max(0, min(source_height, int(np.floor(box[1] * scale_y))))
        x2 = max(0, min(source_width, int(np.ceil(box[2] * scale_x))))
        y2 = max(0, min(source_height, int(np.ceil(box[3] * scale_y))))
        crop = frame[y1:y2, x1:x2]
        quality = image_quality(crop).score if crop.size else 0.0
        candidate = _TrackingCoverCandidate(
            captured_at=float(captured_at),
            tracked_objects=tuple(dict(value) for value in tracked_objects),
            primary_track_id=int(item["track_id"]),
            subject_area_ratio=float(area),
            edge_clearance_ratio=float(clearance),
            quality_score=float(quality),
            detection_confidence=_confidence(item),
            fully_framed=clearance >= 0.01,
            frame_reference=frame_reference,
        )
        if self._cover_baseline is None:
            self._cover_baseline = candidate
        elif frame_reference is None or not frame_reference.exact:
            return
        if self._cover_candidate is None or candidate.score > self._cover_candidate.score:
            self._cover_candidate = candidate

    def _promote_cover_candidate(self, event_id: int) -> None:
        """Verify and persist one materially better tracked cover."""
        baseline = self._cover_baseline
        candidate = self._cover_candidate
        if (
            baseline is None
            or candidate is None
            or candidate.frame_reference is None
            or not candidate.frame_reference.exact
            or self.cover_frame_provider is None
            or self.snapshot_writer is None
            or self.cover_promoter is None
            or candidate.captured_at < baseline.captured_at + 0.25
            or not candidate.fully_framed
            or candidate.subject_area_ratio
            < max(baseline.subject_area_ratio * 1.5, baseline.subject_area_ratio + 0.0025)
            or candidate.quality_score < max(0.12, baseline.quality_score * 0.55)
        ):
            return
        self._cover_promotion = {
            "cover_promoted": False,
            "cover_promotion_attempted": True,
            "cover_promotion_result": "decode_unavailable",
            "cover_source": "object_tracking",
        }
        frame = self.cover_frame_provider(
            candidate.captured_at,
            self._frame_width,
            candidate.frame_reference,
        )
        if frame is None or not frame.size:
            return
        frame_height, frame_width = frame.shape[:2]
        expected_objects = [dict(item) for item in candidate.tracked_objects]
        for item in expected_objects:
            box = item.get("box")
            if isinstance(box, dict):
                item["box"] = dict(box)
        _rescale_detection_boxes(
            expected_objects,
            self._frame_width,
            self._frame_height,
            frame_width,
            frame_height,
        )
        inference_started = time.monotonic()
        try:
            detected_objects = _detect_tracking_objects(
                self.detector,
                frame,
                self.config.low_confidence_threshold,
                enrichment=True,
            )
        except Exception as error:
            self._cover_promotion.update({
                "cover_promotion_result": "detector_error",
                "cover_verification_error": redact_secret_text(error)[:160],
                "cover_verification_inference_ms": round(
                    (time.monotonic() - inference_started) * 1000.0,
                    3,
                ),
            })
            LOGGER.warning(
                "tracked cover verification failed for %s event %d: %s",
                self.camera.id,
                event_id,
                redact_secret_text(error),
            )
            return
        inference_ms = round((time.monotonic() - inference_started) * 1000.0, 3)
        if _inference_deferred(detected_objects):
            self._cover_promotion.update({
                "cover_promotion_result": "inference_deferred",
                "cover_verification_inference_ms": inference_ms,
            })
            return
        failure = detection_failure(detected_objects)
        if failure:
            self._cover_promotion.update({
                "cover_promotion_result": "detector_unavailable",
                "cover_verification_error": failure[:160],
                "cover_verification_inference_ms": inference_ms,
            })
            return
        tracked_objects = self._associate_cover_detections(
            expected_objects,
            detected_objects,
            candidate.primary_track_id,
            frame_width,
            frame_height,
        )
        primary = next(
            (
                item
                for item in tracked_objects
                if int(item.get("track_id") or 0) == candidate.primary_track_id
            ),
            None,
        )
        primary_box = _box(primary.get("box")) if primary is not None else None
        if primary_box is None:
            self._cover_promotion.update({
                "cover_promotion_result": "primary_subject_not_verified",
                "cover_verification_inference_ms": inference_ms,
                "cover_verification_detection_count": len(detected_objects),
            })
            return
        final_area = (
            (primary_box[2] - primary_box[0])
            * (primary_box[3] - primary_box[1])
            / float(frame_width * frame_height)
        )
        final_clearance = max(
            0.0,
            min(
                primary_box[0] / frame_width,
                primary_box[1] / frame_height,
                (frame_width - primary_box[2]) / frame_width,
                (frame_height - primary_box[3]) / frame_height,
            ),
        )
        crop_x1 = max(0, min(frame_width, int(np.floor(primary_box[0]))))
        crop_y1 = max(0, min(frame_height, int(np.floor(primary_box[1]))))
        crop_x2 = max(0, min(frame_width, int(np.ceil(primary_box[2]))))
        crop_y2 = max(0, min(frame_height, int(np.ceil(primary_box[3]))))
        verified_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        final_quality = image_quality(verified_crop).score if verified_crop.size else 0.0
        if (
            final_clearance < 0.01
            or final_area
            < max(baseline.subject_area_ratio * 1.5, baseline.subject_area_ratio + 0.0025)
            or final_quality < max(0.12, baseline.quality_score * 0.55)
        ):
            self._cover_promotion.update({
                "cover_promotion_result": "verified_frame_not_better",
                "cover_verification_inference_ms": inference_ms,
                "cover_verified_subject_area_ratio": round(final_area, 6),
                "cover_verified_edge_clearance_ratio": round(final_clearance, 6),
                "cover_verified_quality_score": round(final_quality, 6),
            })
            return
        snapshot_path = self.snapshot_writer(
            frame,
            datetime.fromtimestamp(candidate.captured_at, timezone.utc),
        )
        promoted = self.cover_promoter(
            event_id,
            snapshot_path=snapshot_path,
            captured_at=candidate.captured_at,
            frame_width=frame_width,
            frame_height=frame_height,
            tracked_objects=tracked_objects,
            cover_metrics={
                "snapshot_primary_subject": True,
                "snapshot_subject_area_ratio": round(final_area, 6),
                "snapshot_edge_clearance_ratio": round(final_clearance, 6),
                "snapshot_quality_score": round(final_quality, 6),
            },
        )
        if promoted is None:
            self._cover_promotion.update({
                "cover_promotion_result": "event_unavailable",
                "cover_verification_inference_ms": inference_ms,
            })
            return
        self._cover_promotion = {
            "cover_promoted": True,
            "cover_promotion_attempted": True,
            "cover_promotion_result": "promoted",
            "cover_source": "object_tracking",
            "cover_captured_at": datetime.fromtimestamp(
                candidate.captured_at,
                timezone.utc,
            ).isoformat(),
            "cover_primary_track_id": candidate.primary_track_id,
            "cover_subject_area_ratio": round(final_area, 6),
            "cover_previous_subject_area_ratio": round(
                baseline.subject_area_ratio,
                6,
            ),
            "cover_verification_inference_ms": inference_ms,
            "cover_verification_detection_count": len(detected_objects),
            "cover_verified_quality_score": round(final_quality, 6),
            "cover_frame_timestamp_source": "source_pts",
            "cover_frame_pts": candidate.frame_reference.pts,
            "cover_frame_time_base": (
                f"{candidate.frame_reference.time_base_num}/"
                f"{candidate.frame_reference.time_base_den}"
            ),
        }

    @staticmethod
    def _associate_cover_detections(
        expected_objects: list[dict[str, Any]],
        detected_objects: list[dict[str, Any]],
        primary_track_id: int,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        """Associate detections from the saved pixels with nominated tracks."""
        usable = [
            item
            for item in detected_objects
            if isinstance(item, dict)
            and item.get("label")
            and _box(item.get("box")) is not None
        ]
        used: set[int] = set()
        verified: list[dict[str, Any]] = []
        diagonal = max(1.0, float(np.hypot(frame_width, frame_height)))
        ordered = sorted(
            expected_objects,
            key=lambda item: int(int(item.get("track_id") or 0) == primary_track_id),
            reverse=True,
        )
        for expected in ordered:
            expected_box = _box(expected.get("box"))
            if expected_box is None:
                continue
            expected_center = (
                (expected_box[0] + expected_box[2]) / 2.0,
                (expected_box[1] + expected_box[3]) / 2.0,
            )
            expected_diagonal_ratio = float(np.hypot(
                expected_box[2] - expected_box[0],
                expected_box[3] - expected_box[1],
            )) / diagonal
            distance_limit = max(0.04, min(0.12, expected_diagonal_ratio * 0.75))
            matches: list[tuple[float, float, int, dict[str, Any]]] = []
            for index, detected in enumerate(usable):
                if index in used or str(detected.get("label")) != str(expected.get("label")):
                    continue
                detected_box = _box(detected.get("box"))
                if detected_box is None:
                    continue
                overlap = _iou(expected_box, detected_box)
                detected_center = (
                    (detected_box[0] + detected_box[2]) / 2.0,
                    (detected_box[1] + detected_box[3]) / 2.0,
                )
                distance = float(np.hypot(
                    expected_center[0] - detected_center[0],
                    expected_center[1] - detected_center[1],
                )) / diagonal
                if overlap < 0.05 and distance > distance_limit:
                    continue
                score = (
                    2.0 * overlap
                    + max(0.0, 1.0 - distance / distance_limit)
                    + 0.1 * _confidence(detected)
                )
                matches.append((score, overlap, index, detected))
            matches.sort(key=lambda item: item[0], reverse=True)
            if not matches:
                continue
            if (
                len(matches) > 1
                and matches[0][1] < 0.25
                and matches[0][0] - matches[1][0] < 0.15
            ):
                continue
            _score, _overlap, index, detected = matches[0]
            used.add(index)
            verified.append({
                **expected,
                "box": dict(detected["box"]),
                "confidence": float(detected.get("confidence") or 0.0),
                "cover_verification_confidence": float(
                    detected.get("confidence") or 0.0
                ),
            })
        return verified

    def _run(
        self,
        event_id: int,
        event_at: datetime,
        initial_objects: list[dict[str, Any]],
        initial_frame: np.ndarray | None,
        stop: threading.Event,
    ) -> None:
        wait_started = time.monotonic()
        acquired = self.limiter.acquire(blocking=False)
        if not acquired and self.config.capacity_wait_seconds > 0:
            wait_deadline = wait_started + self.config.capacity_wait_seconds
            while not acquired and not stop.is_set():
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    break
                acquired = self.limiter.acquire(timeout=min(0.1, remaining))
        capacity_wait_seconds = max(0.0, time.monotonic() - wait_started)
        with self._lock:
            requests = int(self._status.get("capacity_requests") or 0) + 1
            waits = int(self._status.get("capacity_waits") or 0)
            timeouts = int(self._status.get("capacity_timeouts") or 0)
            wait_total = float(self._status.get("capacity_wait_seconds_total") or 0.0)
            wait_max = float(self._status.get("capacity_wait_seconds_max") or 0.0)
            if capacity_wait_seconds >= 0.01:
                waits += 1
                wait_total += capacity_wait_seconds
                wait_max = max(wait_max, capacity_wait_seconds)
            if not acquired and not stop.is_set():
                timeouts += 1
            self._status = {
                **self._status,
                "capacity_requests": requests,
                "capacity_waits": waits,
                "capacity_timeouts": timeouts,
                "capacity_wait_seconds_total": round(wait_total, 3),
                "capacity_wait_seconds_max": round(wait_max, 3),
                "capacity_wait_seconds_last": round(capacity_wait_seconds, 3),
            }
        if stop.is_set() and not acquired:
            # Still drain a queued replacement; capacity-wait exits never reach
            # the acquired-path finally that normally starts _pending_start.
            self._finish_worker_and_start_pending(release_limiter=False)
            return
        if not acquired:
            skipped = {
                "implementation": self.config.implementation,
                "state": "skipped_capacity",
                "sample_fps": self.config.sample_fps,
                "effective_sample_fps": self.config.sample_fps,
                "lost_timeout_seconds": self.config.lost_timeout_seconds,
                "frames_processed": 0,
                "catchup_frames_processed": 0,
                "capacity_wait_seconds": round(capacity_wait_seconds, 3),
                "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tracks": [],
            }
            try:
                self.update_event(event_id, skipped, None)
            except Exception as exc:
                self._set_status(last_error=str(exc))
                LOGGER.exception(
                    "failed to persist capacity status for %s event %d",
                    self.camera.id,
                    event_id,
                )
            self._set_status(enabled=True, active=False, event_id=event_id, last_error="tracking capacity is busy")
            self._publish_safely(event_id, skipped)
            LOGGER.info(
                "object tracking skipped for %s event %d: capacity busy after %.3fs",
                self.camera.id,
                event_id,
                capacity_wait_seconds,
            )
            self._finish_worker_and_start_pending(release_limiter=False)
            return
        tracker: ObjectTrackerBackend | None = None
        frames_processed = 0
        try:
            with self._lock:
                # Capacity waiting must not consume the useful tracking window.
                self._deadline = max(
                    self._deadline,
                    time.monotonic() + self.config.max_session_seconds,
                )
            self._frame_width = 0
            self._frame_height = 0
            self._catchup_frames_processed = 0
            self._coverage_gap_count = 0
            self._maximum_coverage_gap_seconds = 0.0
            self._completion_reason = ""
            self._effective_sample_fps = self.config.sample_fps
            self._cover_baseline = None
            self._cover_candidate = None
            self._cover_promotion = None
            self._set_status(
                cover_promoted=False,
                cover_promotion_attempted=False,
                cover_promotion_result="",
                cover_verification_inference_ms=0.0,
            )
            tracker = self.tracker_registry.create(
                self.config.implementation,
                self.config,
                float(self.detector.config.confidence_threshold),
            )
            if initial_frame is not None:
                self._frame_height = int(initial_frame.shape[0])
                self._frame_width = int(initial_frame.shape[1])
                self._annotate_appearances(
                    initial_frame,
                    initial_objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
            # Preserve the actual selected recording sample time. Consensus may
            # choose a frame up to one second on either side of the event.
            seed_offsets = []
            for detected in initial_objects:
                try:
                    offset = float(detected.get("temporal_sample_offset_seconds"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(offset):
                    seed_offsets.append(offset)
            seed_epoch = event_at.timestamp() + (
                float(np.median(seed_offsets)) if seed_offsets else 0.0
            )
            captured_at = seed_epoch
            for detected in initial_objects:
                detected["_tracking_first_seen_at"] = seed_epoch
            initial_tracked = tracker.update(initial_objects, captured_at, confirm_new=True)
            primary_track_ids = {
                int(item["track_id"])
                for item in initial_tracked
                if item.get("track_id") is not None
                and item.get("snapshot_primary_subject") is True
            }
            if not primary_track_ids:
                primary_track_ids = {
                    int(item["track_id"])
                    for item in initial_tracked
                    if item.get("track_id") is not None
                }
            if initial_frame is not None:
                self._consider_cover_candidate(
                    initial_frame,
                    captured_at,
                    initial_tracked,
                    primary_track_ids,
                )
            self._set_status(enabled=True, active=True, event_id=event_id, last_error="")
            self._persist(
                event_id,
                tracker,
                captured_at,
                initial_tracked,
                frames_processed,
                "active",
                capacity_wait_seconds=capacity_wait_seconds,
            )
            last_persisted_at = time.monotonic()
            stable_frames = 0
            latest_tracked_objects = initial_tracked
            track_states = {
                int(item["track_id"]): str(item.get("track_state") or "confirmed")
                for item in initial_tracked
                if item.get("track_id") is not None
            }
            consecutive_failures = 0

            def interval() -> float:
                return 1.0 / max(0.01, self._effective_sample_fps)

            def process_frame(
                frame: np.ndarray,
                sample_epoch: float,
                *,
                catchup: bool,
                frame_reference: VideoFrameReference | None = None,
            ) -> bool:
                nonlocal consecutive_failures, frames_processed
                nonlocal last_persisted_at, latest_tracked_objects
                nonlocal stable_frames, track_states
                source_height = int(frame.shape[0])
                source_width = int(frame.shape[1])
                if self._frame_width <= 0 or self._frame_height <= 0:
                    self._frame_width = source_width
                    self._frame_height = source_height
                objects = self._tracking_detections_for_frame(frame, catchup=catchup)
                if _inference_deferred(objects):
                    return False
                failure = detection_failure(objects)
                if failure:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise RuntimeError(f"tracking detector failed repeatedly: {failure}")
                    return False
                consecutive_failures = 0
                objects = [
                    item
                    for item in objects
                    if self.config.tracks_label(item.get("label"))
                ]
                apply_detection_zones(
                    self.camera,
                    objects,
                    source_width,
                    source_height,
                    float(self.detector.config.confidence_threshold),
                    bool(getattr(self.detector.config, "require_incident_zone", True)),
                )
                objects = self._enrich_tracking_depth(
                    frame,
                    objects,
                    frame_offset_s=sample_epoch - event_at.timestamp(),
                )
                self._annotate_appearances(
                    frame,
                    objects,
                    lazy=self.config.implementation == "survng_hybrid",
                )
                _rescale_detection_boxes(
                    objects,
                    source_width,
                    source_height,
                    self._frame_width,
                    self._frame_height,
                )
                tracked = tracker.update(objects, sample_epoch)
                latest_tracked_objects = tracked
                summaries = tracker.summaries(sample_epoch)
                next_track_states = {
                    int(item["track_id"]): str(item.get("state") or "")
                    for item in summaries
                    if item.get("track_id") is not None
                }
                next_track_states.update({
                    int(item["track_id"]): str(item.get("track_state") or "")
                    for item in tracked
                    if item.get("track_id") is not None
                })
                important_transition = any(
                    track_id not in track_states
                    or track_states[track_id] != state
                    for track_id, state in next_track_states.items()
                )
                important_transition = important_transition or any(
                    track_id not in next_track_states
                    for track_id in track_states
                )
                self._effective_sample_fps, stable_frames = _adaptive_tracking_fps(
                    self.config,
                    tracked,
                    summaries,
                    important_transition=important_transition,
                    stable_frames=stable_frames,
                )
                track_states = next_track_states
                self._consider_cover_candidate(
                    frame,
                    sample_epoch,
                    tracked,
                    primary_track_ids,
                    frame_reference,
                )
                frames_processed += 1
                if catchup:
                    self._catchup_frames_processed += 1
                now_monotonic = time.monotonic()
                persist_due = _tracking_persistence_due(
                    self.config.persist_interval_seconds,
                    last_persisted_at,
                    now_monotonic,
                    important_transition=important_transition,
                )
                self._set_status(
                    enabled=True,
                    active=True,
                    event_id=event_id,
                    track_count=len(summaries),
                    confirmed_tracks=sum(
                        item.get("state") == "confirmed" for item in summaries
                    ),
                    frames_processed=frames_processed,
                    effective_sample_fps=self._effective_sample_fps,
                    catchup_frames_processed=self._catchup_frames_processed,
                )
                if persist_due:
                    self._persist(
                        event_id,
                        tracker,
                        sample_epoch,
                        tracked,
                        frames_processed,
                        "active",
                    )
                    last_persisted_at = now_monotonic
                return True

            def process_catchup_until(target_epoch: float) -> bool:
                """Consume newly finalized recording frames up to ``target_epoch``.

                The recorder index intentionally exposes only finalized segments. A
                delayed tracking session can therefore catch up to the end of the
                previous segment while the next segment is still being written. This
                helper is safe to call repeatedly as segments become available.
                """
                nonlocal captured_at, last_persisted_at
                if self.catchup_frame_provider is None or initial_frame is None:
                    return False
                catchup_interval = 1.0 / self.config.sample_fps
                catchup_start = captured_at + catchup_interval
                if target_epoch <= catchup_start:
                    return False
                advanced = False
                persisted_before_batch = last_persisted_at
                catchup_frames = iter(
                    self.catchup_frame_provider(
                        catchup_start,
                        target_epoch,
                        self.config.sample_fps,
                        min(1280, int(initial_frame.shape[1])),
                    )
                )
                try:
                    processed_this_tick = 0
                    for sample in catchup_frames:
                        sample_epoch, frame = sample
                        frame_reference = getattr(sample, "reference", None)
                        if stop.is_set() or time.monotonic() >= self._deadline:
                            break
                        if sample_epoch <= captured_at or sample_epoch > target_epoch:
                            continue
                        processed = process_frame(
                            frame,
                            sample_epoch,
                            catchup=True,
                            frame_reference=frame_reference,
                        )
                        # Count deferred/failed attempts toward the tick budget
                        # so shed inference cannot monopolize catch-up replay.
                        processed_this_tick += 1
                        if processed:
                            # Track expiry is based only on successfully
                            # analyzed media, never on a deferred/failed cursor.
                            captured_at = sample_epoch
                            advanced = True
                        if (
                            processed_this_tick
                            >= self.config.max_catchup_frames_per_tick
                        ):
                            break
                finally:
                    close_catchup = getattr(catchup_frames, "close", None)
                    if callable(close_catchup):
                        close_catchup()
                if (
                    advanced
                    and self.config.persist_interval_seconds > 0
                    and last_persisted_at == persisted_before_batch
                ):
                    # A replay batch can finish before the wall-clock cadence
                    # elapses. Flush once at its boundary, never once per frame.
                    self._persist(
                        event_id,
                        tracker,
                        captured_at,
                        latest_tracked_objects,
                        frames_processed,
                        "active",
                    )
                    last_persisted_at = time.monotonic()
                return advanced

            catchup_until = time.time()
            if (
                self.catchup_frame_provider is not None
                and initial_frame is not None
                and catchup_until - captured_at > interval() * 1.5
            ):
                # Start immediately after the actual selected sample. The
                # provider and loop still reject non-increasing timestamps.
                try:
                    process_catchup_until(catchup_until)
                except Exception:
                    LOGGER.exception(
                        "recorded tracking catch-up failed for %s event %d; continuing live",
                        self.camera.id,
                        event_id,
                    )
                catchup_until = time.time()
            if (
                self.catchup_frame_provider is not None
                and initial_frame is not None
                and catchup_until - captured_at
                > TRACKING_MAX_RECOVERABLE_HANDOFF_AGE_SECONDS
            ):
                # Do not turn minutes of queue delay into a fabricated
                # open-segment warning. Partial catch-up that still leaves the
                # cursor beyond recoverable live/open-segment history is the
                # same unrecoverable case as advancing nothing.
                self._completion_reason = "stale_handoff_without_recorded_coverage"
                final_epoch = time.time()
                self._persist(
                    event_id,
                    tracker,
                    final_epoch,
                    None,
                    frames_processed,
                    "interrupted",
                )
                self._set_status(
                    enabled=True,
                    active=False,
                    event_id=event_id,
                    track_count=len(tracker.summaries(final_epoch)),
                    confirmed_tracks=len(tracker.summaries(final_epoch)),
                    frames_processed=frames_processed,
                    catchup_frames_processed=self._catchup_frames_processed,
                )
                return

            next_sample = time.monotonic()
            frame_acquisition_deadline = min(
                self._deadline,
                time.monotonic() + 5.0,
            )
            last_frame_token: float | None = None
            # If recorded evidence already followed every confirmed track until
            # it naturally expired, the event has complete useful coverage.
            # Bridging to a much newer live frame would manufacture a timestamp
            # gap after the object left and incorrectly suppress cover promotion.
            live_bridge_required = tracker.has_live_tracks(captured_at)
            if not live_bridge_required:
                self._completion_reason = "object_exited_recorded_window"
            while live_bridge_required and not stop.is_set():
                with self._lock:
                    deadline = self._deadline
                if time.monotonic() >= deadline:
                    break
                wait_seconds = max(0.0, next_sample - time.monotonic())
                if stop.wait(wait_seconds):
                    break
                sample = self.frame_provider()
                now_epoch = time.time()
                if sample is None:
                    if time.monotonic() >= frame_acquisition_deadline and not tracker.has_live_tracks(now_epoch):
                        break
                    next_sample = time.monotonic() + interval()
                    continue
                frame, sample_epoch, frame_token = sample
                if self._catchup_frames_processed and sample_epoch <= captured_at:
                    next_sample = time.monotonic() + interval()
                    continue
                coverage_gap = sample_epoch - captured_at
                gap_backfilled = False
                if (
                    self.catchup_frame_provider is not None
                    and initial_frame is not None
                    and coverage_gap > self.config.lost_timeout_seconds
                ):
                    # A live frame far ahead of the last recorded sample would age
                    # every track out immediately. Give the recorder's currently-open
                    # segment a bounded opportunity to finalize, then replay the gap
                    # in timestamp order before returning to live frames.
                    settle_deadline = min(
                        deadline,
                        time.monotonic() + TRACKING_CATCHUP_SETTLE_SECONDS,
                    )
                    while (
                        not stop.is_set()
                        and sample_epoch - captured_at > interval() * 1.5
                        and time.monotonic() < settle_deadline
                    ):
                        try:
                            gap_backfilled = bool(
                                process_catchup_until(sample_epoch)
                                or gap_backfilled
                            )
                        except Exception:
                            LOGGER.exception(
                                "recorded tracking gap backfill failed for %s event %d",
                                self.camera.id,
                                event_id,
                            )
                            break
                        if sample_epoch - captured_at <= interval() * 1.5:
                            break
                        wait_for = min(
                            TRACKING_CATCHUP_RETRY_SECONDS,
                            max(0.0, settle_deadline - time.monotonic()),
                        )
                        if wait_for <= 0.0 or stop.wait(wait_for):
                            break
                    coverage_gap = sample_epoch - captured_at
                    if (
                        coverage_gap > self.config.lost_timeout_seconds
                        and tracker.has_live_tracks(captured_at)
                    ):
                        self._coverage_gap_count += 1
                        self._maximum_coverage_gap_seconds = max(
                            self._maximum_coverage_gap_seconds,
                            coverage_gap,
                        )
                        # Persist incomplete coverage either way; only the log
                        # severity distinguishes common open-segment handoff
                        # delays from larger/repeated gaps worth paging on.
                        log = (
                            LOGGER.warning
                            if (
                                coverage_gap >= COVERAGE_GAP_WARNING_SECONDS
                                or self._coverage_gap_count > 1
                            )
                            else LOGGER.info
                        )
                        if coverage_gap <= TRACKING_MAX_RECOVERABLE_HANDOFF_AGE_SECONDS:
                            gap_detail = "open recording segment not bridged"
                            reason = "missing_media_while_object_active"
                        else:
                            gap_detail = "tracking fell behind live"
                            reason = "tracking_fell_behind_live"
                        log(
                            "object tracking coverage gap for %s event %d: %.3fs "
                            "(%s)",
                            self.camera.id,
                            event_id,
                            coverage_gap,
                            gap_detail,
                        )
                        self._completion_reason = reason
                    elif not tracker.has_live_tracks(captured_at):
                        self._completion_reason = "object_exited_during_catchup"
                        break
                if gap_backfilled and sample_epoch <= captured_at + interval() * 0.5:
                    last_frame_token = frame_token
                    next_sample = time.monotonic() + interval()
                    continue
                if last_frame_token is not None and frame_token <= last_frame_token:
                    if not tracker.has_live_tracks(now_epoch):
                        break
                    next_sample = time.monotonic() + interval()
                    continue
                last_frame_token = frame_token
                if not process_frame(frame, sample_epoch, catchup=False):
                    next_sample = time.monotonic() + interval()
                    continue
                captured_at = sample_epoch
                if not tracker.has_live_tracks(sample_epoch):
                    self._completion_reason = "object_exited_live_window"
                    break
                next_sample = max(next_sample + interval(), time.monotonic())
            final_epoch = time.time()
            final_state = "interrupted" if self._coverage_gap_count else "complete"
            if not self._completion_reason:
                self._completion_reason = (
                    "session_stopped" if stop.is_set() else "tracking_window_complete"
                )
            if final_state == "complete" and not stop.is_set():
                try:
                    self._promote_cover_candidate(event_id)
                except Exception:
                    LOGGER.exception(
                        "tracked cover promotion failed for %s event %d",
                        self.camera.id,
                        event_id,
                    )
            self._persist(event_id, tracker, final_epoch, None, frames_processed, final_state)
            self._set_status(
                enabled=True,
                active=False,
                event_id=event_id,
                track_count=len(tracker.summaries(final_epoch)),
                confirmed_tracks=len(tracker.summaries(final_epoch)),
                frames_processed=frames_processed,
                catchup_frames_processed=self._catchup_frames_processed,
            )
        except Exception as exc:
            self._persist_failure(event_id, tracker, frames_processed, exc)
            self._set_status(
                enabled=True,
                active=False,
                event_id=event_id,
                last_error=redact_secret_text(exc)[:240],
            )
            LOGGER.exception("object tracking failed for %s event %d", self.camera.id, event_id)
        finally:
            self._finish_worker_and_start_pending(release_limiter=True)

    def _finish_worker_and_start_pending(self, *, release_limiter: bool) -> None:
        """Release this worker's slot ownership and start any queued successor.

        Capacity-wait exits never acquire the limiter, so they must still clear
        ``_thread`` and drain ``_pending_start`` or a replacement queued while
        waiting for capacity is orphaned until a later unrelated start.
        """
        if release_limiter:
            self.limiter.release()
        with self._transition_lock:
            pending_start = None
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._event_id = None
                    if self._accepting:
                        pending_start = self._pending_start
                    self._pending_start = None
            if pending_start is not None:
                self._start_session(*pending_start)

    def _persist(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend,
        captured_at: float,
        tracked_objects: list[dict[str, Any]] | None,
        frames_processed: int,
        state: str,
        capacity_wait_seconds: float | None = None,
    ) -> None:
        tracks = tracker.summaries(captured_at)
        diagnostics_method = getattr(tracker, "diagnostics", None)
        tracker_diagnostics = (
            diagnostics_method() if callable(diagnostics_method) else {}
        )
        avoided = int(tracker_diagnostics.get("reid_avoided_geometry_matches") or 0)
        avoided_by_label = dict(tracker_diagnostics.get("reid_avoided_by_label") or {})
        recoveries_by_label: dict[str, int] = {}
        for track in tracks:
            recoveries = int(track.get("reid_matches") or 0)
            if recoveries:
                label = str(track.get("label") or "unknown")
                recoveries_by_label[label] = recoveries_by_label.get(label, 0) + recoveries
        payload = {
            "implementation": self.config.implementation,
            "state": state,
            "sample_fps": self.config.sample_fps,
            "effective_sample_fps": self._effective_sample_fps,
            "capacity_wait_seconds": round(
                float(
                    capacity_wait_seconds
                    if capacity_wait_seconds is not None
                    else self._status.get("capacity_wait_seconds_last") or 0.0
                ),
                3,
            ),
            "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
            "lost_timeout_seconds": self.config.lost_timeout_seconds,
            "frames_processed": frames_processed,
            "catchup_frames_processed": self._catchup_frames_processed,
            "coverage_gap_count": self._coverage_gap_count,
            "maximum_coverage_gap_seconds": round(
                self._maximum_coverage_gap_seconds,
                3,
            ),
            "coverage_incomplete": self._coverage_gap_count > 0,
            "completion_reason": self._completion_reason,
            "updated_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "tracks": tracks,
            "reid_diagnostics": {
                **tracker_diagnostics,
                "inference_attempts": max(
                    0,
                    int(self.status().get("reid_attempts") or 0)
                    - self._reid_attempt_base,
                ),
                "inference_attempts_by_reason": {
                    reason: max(
                        0,
                        int(count)
                        - int(self._reid_attempt_base_by_reason.get(reason) or 0),
                    )
                    for reason, count in dict(
                        self.status().get("reid_attempts_by_reason") or {}
                    ).items()
                    if int(count)
                    - int(self._reid_attempt_base_by_reason.get(reason) or 0)
                    > 0
                },
            },
        }
        if self._cover_promotion:
            payload.update(self._cover_promotion)
        if self._frame_width > 0 and self._frame_height > 0:
            payload["frame_width"] = self._frame_width
            payload["frame_height"] = self._frame_height
        if self.update_event(event_id, payload, tracked_objects) is None:
            raise RuntimeError(f"tracking event {event_id} no longer exists")
        if state in {"complete", "interrupted"}:
            self._index_track_appearances(event_id, tracker)
        self._set_status(
            enabled=True,
            active=state == "active",
            event_id=event_id,
            track_count=len(tracks),
            confirmed_tracks=len(tracks),
            frames_processed=frames_processed,
            effective_sample_fps=self._effective_sample_fps,
            catchup_frames_processed=self._catchup_frames_processed,
            coverage_gap_count=self._coverage_gap_count,
            maximum_coverage_gap_seconds=round(
                self._maximum_coverage_gap_seconds,
                3,
            ),
            coverage_incomplete=self._coverage_gap_count > 0,
            completion_reason=self._completion_reason,
            reid_recoveries=(
                self._reid_recovery_base + sum(recoveries_by_label.values())
            ),
            reid_recoveries_by_label={
                label: int(self._reid_recovery_base_by_label.get(label) or 0)
                + int(recoveries_by_label.get(label) or 0)
                for label in (
                    self._reid_recovery_base_by_label.keys()
                    | recoveries_by_label.keys()
                )
            },
            reid_avoided_geometry_matches=self._reid_avoided_base + avoided,
            reid_avoided_by_label={
                label: int(self._reid_avoided_base_by_label.get(label) or 0)
                + int(avoided_by_label.get(label) or 0)
                for label in (
                    self._reid_avoided_base_by_label.keys()
                    | avoided_by_label.keys()
                )
            },
            **(self._cover_promotion or {}),
        )
        if state in {"complete", "interrupted"}:
            self._publish_safely(event_id, payload)

    def _index_track_appearances(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend,
    ) -> None:
        if self.appearance_indexer is None or self.appearance_encoder is None:
            return
        records_method = getattr(tracker, "appearance_records", None)
        identity_method = getattr(
            self.appearance_encoder,
            "model_identity_for_label",
            None,
        )
        if not callable(records_method) or not callable(identity_method):
            return
        prepared: list[dict[str, Any]] = []
        for record in records_method():
            identity = identity_method(str(record.get("label") or ""))
            if not isinstance(identity, dict) or not identity.get("model_fingerprint"):
                continue
            prepared.append({
                **record,
                "model_kind": identity.get("model_kind"),
                "model_fingerprint": identity.get("model_fingerprint"),
                "match_threshold": identity.get("match_threshold"),
                "created_at": record.get("last_seen"),
            })
        if not prepared:
            return
        try:
            indexed = self.appearance_indexer(
                event_id,
                self.camera.id,
                prepared,
            )
        except Exception as exc:
            error = redact_secret_text(exc)[:240]
            self._set_status(appearance_index_error=error)
            LOGGER.exception(
                "failed to index object appearances for %s event %d",
                self.camera.id,
                event_id,
            )
            return
        self._set_status(
            appearance_vectors_indexed=int(indexed),
            appearance_index_error="",
        )

    def _persist_failure(
        self,
        event_id: int,
        tracker: ObjectTrackerBackend | None,
        frames_processed: int,
        error: Exception,
    ) -> None:
        captured_at = time.time()
        payload = {
            "implementation": self.config.implementation,
            "state": "failed",
            "sample_fps": self.config.sample_fps,
            "effective_sample_fps": self._effective_sample_fps,
            "capacity_wait_seconds": round(
                float(self._status.get("capacity_wait_seconds_last") or 0.0),
                3,
            ),
            "capacity_wait_limit_seconds": self.config.capacity_wait_seconds,
            "lost_timeout_seconds": self.config.lost_timeout_seconds,
            "frames_processed": frames_processed,
            "catchup_frames_processed": self._catchup_frames_processed,
            "updated_at": datetime.fromtimestamp(captured_at, timezone.utc).isoformat(),
            "error": redact_secret_text(error)[:240],
            "tracks": tracker.summaries(captured_at) if tracker is not None else [],
        }
        try:
            self.update_event(event_id, payload, None)
        except Exception:
            LOGGER.exception(
                "failed to persist tracking failure for %s event %d",
                self.camera.id,
                event_id,
            )
        self._publish_safely(event_id, payload)

    def _publish_safely(self, event_id: int, payload: dict[str, Any]) -> None:
        if self.publisher is None:
            return
        try:
            self.publisher(
                "object_tracking",
                TrackingCompleted(
                    event_id=event_id,
                    camera_id=self.camera.id,
                    state=str(payload.get("state") or "complete"),
                    details=payload,
                ).to_payload(),
            )
        except Exception:
            LOGGER.exception(
                "object tracking notification failed for %s event %d",
                self.camera.id,
                event_id,
            )

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self._status = {**self._status, **values}

    def _annotate_appearances(
        self,
        frame: np.ndarray,
        objects: list[dict[str, Any]],
        *,
        lazy: bool = False,
    ) -> None:
        encoder = self.appearance_encoder
        if encoder is None or not encoder.enabled or not self.config.appearance_reid_enabled:
            return
        height, width = frame.shape[:2]
        candidates = sorted(
            (
                detected
                for detected in objects
                if _encoder_supports_label(
                    encoder,
                    self.config,
                    str(detected.get("label") or "").lower(),
                )
            ),
            key=_confidence,
            reverse=True,
        )[:self.config.reid_max_embeddings_per_frame]
        frame_state = {"failed": False}
        for detected in candidates:
            box = _box(detected.get("box"))
            if box is None:
                continue
            x1 = max(0, min(width - 1, int(box[0])))
            y1 = max(0, min(height - 1, int(box[1])))
            x2 = max(x1 + 1, min(width, int(box[2])))
            y2 = max(y1 + 1, min(height, int(box[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 16 or crop.shape[1] < 8:
                continue
            label = str(detected.get("label") or "").lower()
            if lazy:
                # The provider is consumed synchronously by tracker.update(), so
                # retaining a view avoids copying large vehicle crops.
                owned_crop = crop
                detected["_tracking_embedding_provider"] = (
                    lambda label=label, crop=owned_crop, detected=detected: self._encode_with_telemetry(
                        encoder,
                        label,
                        crop,
                        frame_state,
                        reason=str(
                            detected.get("_tracking_embedding_reason") or "unspecified"
                        ),
                    )
                )
                continue
            embedding = self._encode_with_telemetry(
                encoder,
                label,
                crop,
                frame_state,
                reason="eager",
            )
            if embedding is not None:
                detected["_tracking_embedding"] = embedding
            if frame_state["failed"]:
                break

    def _encode_with_telemetry(
        self,
        encoder: AppearanceEncoder,
        label: str,
        crop: np.ndarray,
        frame_state: dict[str, bool],
        *,
        reason: str,
    ) -> np.ndarray | None:
        if frame_state["failed"]:
            return None
        started_at = time.perf_counter()
        embedding: np.ndarray | None = None
        error = ""
        try:
            embedding = _encode_appearance(encoder, label, crop)
        except Exception as exc:
            error = redact_secret_text(exc)[:240]
            frame_state["failed"] = True
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        with self._lock:
            attempts = int(self._status.get("reid_attempts") or 0) + 1
            successes = int(self._status.get("reid_successes") or 0)
            failures = int(self._status.get("reid_failures") or 0)
            total_ms = float(self._status.get("reid_inference_ms") or 0.0) + elapsed_ms
            by_label = dict(self._status.get("reid_attempts_by_label") or {})
            by_label[label] = int(by_label.get(label) or 0) + 1
            by_reason = dict(self._status.get("reid_attempts_by_reason") or {})
            by_reason[reason] = int(by_reason.get(reason) or 0) + 1
            if embedding is not None:
                successes += 1
            if error:
                failures += 1
            self._status = {
                **self._status,
                "reid_attempts": attempts,
                "reid_successes": successes,
                "reid_failures": failures,
                "reid_inference_ms": round(total_ms, 3),
                "reid_average_ms": round(total_ms / attempts, 3),
                "reid_attempts_by_label": by_label,
                "reid_attempts_by_reason": by_reason,
                "last_reid_error": error or self._status.get("last_reid_error", ""),
            }
        if error:
            LOGGER.debug(
                "appearance ReID unavailable for %s: %s",
                self.camera.id,
                error,
            )
        return embedding

class ObjectTrackingSessionFactory:
    def __init__(
        self,
        config: ObjectTrackingConfig,
        detector: ObjectDetectorBackend,
        update_event: TrackingUpdate,
        publisher: TrackingPublisher | None,
        limiter: threading.BoundedSemaphore,
        tracker_registry: ObjectTrackerRegistry | None = None,
        appearance_encoder: AppearanceEncoder | None = None,
        appearance_indexer: AppearanceIndexWriter | None = None,
        cover_promoter: TrackingCoverPromoter | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.update_event = update_event
        self.publisher = publisher
        self.limiter = limiter
        self.tracker_registry = tracker_registry or build_builtin_object_tracker_registry()
        self.appearance_encoder = appearance_encoder
        self.appearance_indexer = appearance_indexer
        self.cover_promoter = cover_promoter
        # Fail configuration loading before any event tries to start a session.
        self.tracker_registry.require(config.implementation)

    def create(
        self,
        camera: CameraConfig,
        frame_provider: FrameProvider,
        catchup_frame_provider: CatchupFrameProvider | None = None,
        cover_frame_provider: TrackingCoverFrameProvider | None = None,
        snapshot_writer: TrackingSnapshotWriter | None = None,
        live_detections_provider: LiveDetectionsProvider | None = None,
    ) -> ObjectTrackingSession:
        return ObjectTrackingSession(
            camera=camera,
            config=self.config,
            detector=self.detector,
            frame_provider=frame_provider,
            update_event=self.update_event,
            publisher=self.publisher,
            limiter=self.limiter,
            tracker_registry=self.tracker_registry,
            appearance_encoder=self.appearance_encoder,
            appearance_indexer=self.appearance_indexer,
            catchup_frame_provider=catchup_frame_provider,
            cover_frame_provider=cover_frame_provider,
            snapshot_writer=snapshot_writer,
            cover_promoter=self.cover_promoter,
            live_detections_provider=live_detections_provider,
        )
