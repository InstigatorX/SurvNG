from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from ..evidence_work import (
    check_evidence_cancellation, evidence_cancelled,
    report_evidence_stage,
)
from ..config import CameraConfig, DetectorConfig
from ..scene_activity_evidence import scene_sample_records as _scene_sample_records
from ..evidence_gallery import GalleryBatch, encode_gallery
from ..face_candidates import FaceCandidate, FaceCandidateSample, collect_face_candidates
from ..face_detection_geometry import upper_body_window
from ..object_activity import ObjectActivityAttributor
from ..recorded_decoder import (
    build_decode_command,
    decoder_error_codecs,
    read_bmp_frames,
    recorded_decode_plans,
    run_decoder,
)
from ..recording_media import mp4_video_dimensions
from ..zones import apply_depth_zone_filters, apply_detection_zones
from .context import Frame
from .recorded_decode_budget import RecordedDecodeBudget, refinement_frame_count
from .recorded_evidence import (
    _RecordedDetectionSample,
    _box,
    _candidate_detection,
    _collect_temporal_evidence,
    _confidence,
    _image_quality,
    _refinement_early_exit_ready,
    _refinement_stage_complete,
    _representative_needs_refinement,
    _temporal_consensus,
    _temporal_motion_metrics,
    _temporal_trigger_evidence,
)
from .scene_evidence import scene_observation


LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_STAGES = (
    (-1.0, -0.5, 0.0, 0.5, 1.0),
    (1.5, 2.0, 2.5, 3.0),
    (3.5, 4.0, 4.5),
    (8.0, 8.5),
    (12.0, 12.5),
)
RECORDED_ROUTE_DENSE_EVENT_FRAME_STAGES = (
    (-1.0, -0.5, 0.0, 0.5, 1.0),
    (1.5, 2.0, 2.5, 3.0),
    (3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
    (6.5, 7.0, 7.5),
    (8.0, 8.5),
    (12.0, 12.5),
)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 24.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0
RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS = 6.0
# Finalization and edge indexing after a segment boundary.
RECORDED_SEGMENT_INDEX_MARGIN_SECONDS = 1.0
# FFmpeg's accurate seek discards frames before the seek point, so keep the
# seek strictly ahead of the first requested sample.
RECORDED_BATCH_SEEK_MARGIN_SECONDS = 0.25
# Shorter waits are cheaper to sleep through than to reschedule.
RECORDED_SPLIT_MINIMUM_WAIT_SECONDS = 1.0
FAST_LIVE_FRAME_MAX_AGE_SECONDS = 1.0
FAST_LIVE_FRAME_FUTURE_TOLERANCE_SECONDS = 0.5
RECORDED_LIVE_FALLBACK_EVENT_TOLERANCE_SECONDS = 1.5
# One decoder process may not outlive this, whatever the stage deadline allows.
RECORDED_DECODE_TIMEOUT_SECONDS = 8.0


def _flatten_refinement_stages(
    stages: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    return tuple(offset for stage in stages for offset in stage)


def _coerce_refinement_stages(raw: object) -> tuple[tuple[float, ...], ...] | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    stages: list[tuple[float, ...]] = []
    for stage in raw:
        if not isinstance(stage, (list, tuple)) or not stage:
            return None
        try:
            stages.append(tuple(float(offset) for offset in stage))
        except (TypeError, ValueError):
            return None
    return tuple(stages)


def _refinement_budget(config: Any, name: str, default: float) -> float:
    """Prefer detector config when present; else module defaults (patchable in tests)."""
    if config is None or not hasattr(config, name):
        return float(default)
    value = getattr(config, name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_recorded_refinement_plan(
    config: Any = None,
    *,
    qualification: dict[str, Any] | None = None,
    event_at: datetime | None = None,
    camera_id: str | None = None,
) -> tuple[tuple[tuple[float, ...], ...], float, float, float, float]:
    """Resolve recorded refinement stages and occupancy budgets."""
    if isinstance(qualification, dict) and qualification.get("scene_confirmation"):
        epoch = event_at.timestamp() if event_at is not None else 0.0
        window = qualification.get("scene_confirmation_window") or {}
        start = float(window.get("start_epoch", epoch-1.0))
        end = float(window.get("end_epoch", epoch+1.0))
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError("invalid scene confirmation window")
        # Endpoints preserve context; near-trigger samples recover brief acts.
        # Work remains bounded even when a caller requests a wider interval.
        absolute = {start+(end-start)*i/4 for i in range(5)}
        absolute.update(t for t in (epoch-.5,epoch,epoch+.5) if start <= t <= end)
        return (tuple(round(t-epoch,6) for t in sorted(absolute)),), 8.0, .25, .25, 0.0
    if isinstance(qualification, dict) and qualification.get("scene_discovery"):
        # Periodic discovery shares the normal decode/inference budgets, but
        # samples one complete frame rather than doing a cover-selection pass.
        return ((0.0,),), 2.0, 0.0, 0.25, 0.0
    stages = _coerce_refinement_stages(
        getattr(config, "event_refinement_stages", None)
    ) or RECORDED_EVENT_FRAME_STAGES
    watch = (
        qualification.get("features", {}).get("route_detection_watch")
        if isinstance(qualification, dict)
        and isinstance(qualification.get("features"), dict)
        else None
    )
    valid_route_watch = False
    if isinstance(watch, dict) and event_at is not None:
        try:
            verification_reason = str(
                (qualification or {}).get("features", {}).get(
                    "security_verification_reason", ""
                )
            ).strip().lower()
            target = str(watch.get("target_camera_id") or "").strip()
            source_event_id = int(watch.get("source_event_id") or 0)
            eligible_at = float(watch.get("eligible_at"))
            expires_at = float(watch.get("expires_at"))
            event_epoch = float(event_at.timestamp())
            path = tuple(str(item).strip() for item in (watch.get("route_path") or ()))
            valid_route_watch = bool(
                target
                and verification_reason == "route_watch"
                and (camera_id is None or target == str(camera_id).strip())
                and source_event_id > 0
                and math.isfinite(eligible_at)
                and math.isfinite(expires_at)
                and eligible_at <= event_epoch <= expires_at
                and path
                and path[-1] == target
                and len(path) == len(set(path))
            )
        except (TypeError, ValueError, OverflowError):
            valid_route_watch = False
    # Camera id is not part of DetectorConfig; callers may validate target
    # separately by supplying it as a private compatibility attribute.
    if valid_route_watch:
        stages = (
            _coerce_refinement_stages(
                getattr(config, "event_route_refinement_stages", None)
            )
            or RECORDED_ROUTE_DENSE_EVENT_FRAME_STAGES
        )
    return (
        stages,
        _refinement_budget(
            config,
            "event_refinement_retry_seconds",
            RECORDED_EVENT_RETRY_SECONDS,
        ),
        _refinement_budget(
            config,
            "event_refinement_settle_seconds",
            RECORDED_EVENT_SETTLE_SECONDS,
        ),
        _refinement_budget(
            config,
            "event_refinement_retry_interval_seconds",
            RECORDED_EVENT_RETRY_INTERVAL_SECONDS,
        ),
        _refinement_budget(
            config,
            "event_representative_refinement_timeout_seconds",
            RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS,
        ),
    )


@dataclass(frozen=True)
class RecordedDetectionResult:
    """Recorded evidence plus truthful phase timings for one sampling pass."""

    frame: Frame | None
    objects: list[dict[str, Any]]
    recording_path: str
    timings_ms: dict[str, float]
    refinement_pending: bool = False
    face_candidates: tuple[FaceCandidate, ...] = ()
    frame_captured_at_epoch: float | None = None
    frame_source: str = ""
    frame_timestamp_exact: bool = False
    review_image: dict[str, Any] | None = None
    gallery: GalleryBatch | None = None
    # Set when sampling stopped before a stage whose recording is not yet
    # indexed; the caller re-runs from ``resume_stage`` at ``resume_at``.
    resume_stage: int | None = None
    resume_at: float | None = None

    def __iter__(self):
        # Preserve the historical three-value provider contract for callers
        # that do not need refinement or timing metadata.
        yield self.frame
        yield self.objects
        yield self.recording_path


@dataclass(frozen=True, slots=True)
class TimestampedLiveFrame:
    """Provenance required before live evidence can become provisional truth."""

    frame: Frame
    captured_at_epoch: float
    captured_at_monotonic: float
    sequence: int
    camera_generation: int
    capture_generation: int
    source: str = "live"
    geometry_trusted: bool = True
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class _DecodedRecordedFrame:
    frame: Frame
    actual_offset: float
    exact_timestamp: bool


class MotionObjectDetectorBackend(Protocol):
    config: DetectorConfig

    def detect(
        self,
        frame: Frame,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def detect_faces(self, frame: Frame) -> list[dict[str, Any]]:
        ...

    def estimate_depth_for_objects(
        self,
        frame: Frame,
        objects: list[dict[str, Any]],
        *,
        frame_offset_s: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...


class MotionRecordingProvider(Protocol):
    ffmpeg_path: str
    hardware_acceleration: str

    def recording_at(self, camera_id: str, epoch: float) -> dict[str, Any] | None:
        ...


LiveFrameProvider = Callable[[], Frame | None]
TimestampedLiveFrameProvider = Callable[[], TimestampedLiveFrame | tuple[Frame, float] | None]
TimestampedEvidenceFrameProvider = Callable[
    [dict[str, Any]], TimestampedLiveFrame | tuple[Frame, float] | None
]
StopRequested = Callable[[], bool]


class _EventRecordedSampler:
    """Reuse recording discovery and decoded samples within one event workflow."""

    def __init__(
        self,
        *,
        camera_id: str,
        recorder: MotionRecordingProvider,
        frame_reader: Callable[..., Frame | None],
        batch_frame_reader: Callable[
            ..., tuple[dict[float, _DecodedRecordedFrame], int]
        ] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.recorder = recorder
        self.frame_reader = frame_reader
        self.batch_frame_reader = batch_frame_reader
        self._rows: list[dict[str, Any]] = sorted(
            (dict(row) for row in (rows or [])),
            key=lambda row: float(row.get("start_epoch", 0.0)),
        )
        self._frames: dict[tuple[str, float], _DecodedRecordedFrame | None] = {}

    def recording_at(self, epoch: float) -> dict[str, Any] | None:
        # Match Recorder.recording_at(): at a shared segment boundary the
        # newest segment is preferred, avoiding an EOF decode from its predecessor.
        for row in reversed(self._rows):
            start = float(row.get("start_epoch", 0.0))
            end = row.get("end_epoch")
            if end is None:
                duration = row.get("duration_seconds")
                end = start + float(duration) if duration is not None else start
            if start <= epoch <= float(end):
                return row
        row = self.recorder.recording_at(self.camera_id, epoch)
        if row is not None and row.get("start_epoch") is not None:
            normalized = dict(row)
            if normalized.get("end_epoch") is None:
                duration = normalized.get("duration_seconds")
                if duration is not None:
                    normalized["end_epoch"] = float(normalized["start_epoch"]) + float(duration)
            self._rows.append(normalized)
            self._rows.sort(
                key=lambda item: float(item.get("start_epoch", 0.0))
            )
            return normalized
        return row

    def frame_at(
        self,
        path: Path,
        offset_seconds: float,
        *,
        deadline: float | None,
    ) -> _DecodedRecordedFrame | None:
        key = (str(path), round(max(0.0, offset_seconds), 3))
        if key not in self._frames:
            frame = self.frame_reader(
                path,
                offset_seconds,
                deadline=deadline,
            )
            if frame is not None:
                self._frames[key] = _DecodedRecordedFrame(
                    frame=frame,
                    actual_offset=max(0.0, offset_seconds),
                    exact_timestamp=False,
                )
            return self._frames.get(key)
        return self._frames[key]

    def frames_at(
        self,
        path: Path,
        offsets_seconds: list[float],
        *,
        deadline: float | None,
    ) -> tuple[dict[float, _DecodedRecordedFrame], int, int]:
        """Return cached/batched frames and bounded fallback counts.

        The tuple contains frames keyed by the caller's original offsets,
        batch process count, and single-frame fallback count.
        """
        requested = list(dict.fromkeys(max(0.0, value) for value in offsets_seconds))
        frames: dict[float, _DecodedRecordedFrame] = {}
        missing: list[float] = []
        for offset in requested:
            key = (str(path), round(offset, 3))
            cached = self._frames.get(key)
            if cached is not None:
                frames[offset] = cached
            else:
                missing.append(offset)

        batch_processes = 0
        if missing and self.batch_frame_reader is not None:
            decoded, batch_processes = self.batch_frame_reader(
                path, missing, deadline=deadline
            )
            decoded_by_key = {
                round(max(0.0, offset), 3): frame
                for offset, frame in decoded.items()
            }
            for offset in missing:
                normalized_offset = round(offset, 3)
                frame = decoded_by_key.get(normalized_offset)
                if frame is None:
                    continue
                key = (str(path), normalized_offset)
                self._frames[key] = frame
                frames[offset] = frame

        fallback_count = 0
        for offset in missing:
            if offset in frames:
                continue
            fallback_count += 1
            frame = self.frame_at(path, offset, deadline=deadline)
            if frame is not None:
                frames[offset] = frame
        return frames, batch_processes, fallback_count


class RecordedMotionObjectDetector:
    """Combines recorded event-time detections into repeatable object evidence."""

    def __init__(
        self,
        camera: CameraConfig,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
        live_frame_provider: LiveFrameProvider,
        timestamped_live_frame_provider: TimestampedLiveFrameProvider | None = None,
        timestamped_evidence_frame_provider: TimestampedEvidenceFrameProvider | None = None,
        stop_requested: StopRequested = lambda: False,
        decode_budget: RecordedDecodeBudget | None = None,
        activity_attributor: ObjectActivityAttributor | None = None,
    ) -> None:
        self.camera = camera
        self.activity_attributor = activity_attributor
        self.detector = detector
        self.recorder = recorder
        self.live_frame_provider = live_frame_provider
        self.timestamped_live_frame_provider = timestamped_live_frame_provider
        self.timestamped_evidence_frame_provider = timestamped_evidence_frame_provider
        self._camera_stop_requested = stop_requested
        self.decode_budget = decode_budget
        self._scene_frames: deque[TimestampedLiveFrame] = deque(maxlen=4)
        self._scene_frames_lock = threading.Lock()

    def stop_requested(self) -> bool:
        return self._camera_stop_requested() or evidence_cancelled()

    def remember_scene_frame(self, sample: TimestampedLiveFrame) -> None:
        """Keep only scheduled discovery frames while their durable jobs wait.

        Capture owns the image; take a copy before another capture callback can
        reuse it. Evicted frames and jobs replayed after restart use recordings.
        """
        retained = replace(sample, frame=sample.frame.copy())
        with self._scene_frames_lock:
            self._scene_frames.append(retained)

    def recorded_evidence_ready_at(
        self,
        event_at: datetime,
        qualification: dict[str, Any] | None = None,
    ) -> float | None:
        """Predict when the first refinement stage's newest frame is indexed.

        Segments are indexed only once closed. The newest indexed main segment
        fixes the boundary phase; the prediction never exceeds one segment
        past the needed frame, so a stalled recorder delays work only boundedly.
        """
        qualification = qualification or {}
        if qualification.get("scene_discovery") or qualification.get("scene_confirmation"):
            return None
        config = self.detector.config
        stages, _retry, settle_seconds, *_ = resolve_recorded_refinement_plan(
            config, qualification=qualification, event_at=event_at, camera_id=self.camera.id,
        )
        if not stages or not stages[0]:
            return None
        initial = stages[0]
        adaptive = tuple(offset for offset in (0.0, 0.5, -0.5) if offset in initial)
        newest = (
            max(adaptive)
            if adaptive and config.recorded_adaptive_sampling
            else max(initial)
        )
        return self._predicted_index_epoch(event_at.timestamp() + newest, settle_seconds)

    def _index_refresh_generation(self) -> int | None:
        """Finished-discovery count to wait against, where the recorder has one."""
        generation = getattr(self.recorder, "recording_edge_refresh_generation", None)
        if not callable(generation) or not callable(
            getattr(self.recorder, "wait_for_recording_edge_refresh", None)
        ):
            return None
        return int(generation(self.camera.id))

    def _predicted_index_epoch(self, target: float, settle_seconds: float) -> float:
        ready = target + settle_seconds
        segment = float(getattr(self.recorder, "segment_seconds", 0.0) or 0.0)
        rows_between = getattr(self.recorder, "recording_rows_between", None)
        if segment <= 0 or not callable(rows_between):
            return ready
        try:
            rows = rows_between(
                self.camera.id, target - 4.0 * segment, target + segment, "main",
                discover_missing=False,
            )
        except Exception:
            LOGGER.debug("recording readiness lookup unavailable for %s", self.camera.id, exc_info=True)
            return ready
        ends = [float(row["end_epoch"]) for row in rows if row.get("end_epoch") is not None]
        if not ends:
            return ready
        latest = max(ends)
        if target <= latest:
            return ready
        closes = latest + math.ceil((target - latest) / segment) * segment
        return max(ready, min(closes, target + segment) + RECORDED_SEGMENT_INDEX_MARGIN_SECONDS)

    def detect(
        self,
        event_at: datetime,
        qualification: dict[str, Any] | None = None,
    ) -> RecordedDetectionResult:
        confirming_scene = bool(qualification and qualification.get("scene_confirmation"))
        if qualification and qualification.get("scene_discovery") and not confirming_scene:
            live = self._live_discovery_result(event_at)
            if live is not None:
                return live
        (
            stages,
            retry_seconds,
            settle_seconds,
            retry_interval_seconds,
            representative_timeout_seconds,
        ) = resolve_recorded_refinement_plan(
            self.detector.config,
            qualification=qualification,
            event_at=event_at,
            camera_id=self.camera.id,
        )
        sampling = qualification.get("evidence_sampling", {}) if isinstance(qualification, dict) else {}
        minimum_last_offset = None
        if isinstance(sampling, dict):
            try:
                requested = float(sampling.get("minimum_last_offset_seconds"))
                if math.isfinite(requested):
                    minimum_last_offset = min(max(_flatten_refinement_stages(stages)), max(0.0, requested))
            except (TypeError, ValueError):
                pass
            try:
                timeout = float(sampling.get("timeout_seconds"))
                if math.isfinite(timeout) and timeout >= 0:
                    retry_seconds = min(retry_seconds, timeout)
            except (TypeError, ValueError):
                pass
        route_dense = bool(
            qualification
            and isinstance(qualification.get("features"), dict)
            and isinstance(
                qualification["features"].get("route_detection_watch"), dict
            )
            and len(_flatten_refinement_stages(stages))
            > refinement_frame_count(
                _coerce_refinement_stages(self.detector.config.event_refinement_stages)
                or RECORDED_EVENT_FRAME_STAGES
            )
        )
        resumable = bool(
            qualification
            and qualification.get("refinement_split_allowed")
            and not confirming_scene
            and not qualification.get("scene_discovery")
            and minimum_last_offset is None
        )
        start_stage = 0
        if resumable:
            try:
                start_stage = min(max(0, int(qualification.get("refinement_resume_stage") or 0)), len(stages) - 1)
            except (TypeError, ValueError):
                start_stage = 0
        result = self._detect(
            event_at,
            stages=stages,
            retry_seconds=retry_seconds,
            settle_seconds=settle_seconds,
            retry_interval_seconds=retry_interval_seconds,
            representative_timeout_seconds=representative_timeout_seconds,
            allow_representative_refinement=not confirming_scene,
            refinement_pending=False,
            route_dense=route_dense,
            minimum_last_offset_seconds=minimum_last_offset,
            preserve_confirmation_counts=bool(qualification and qualification.get("scene_discovery")),
            scene_confirmation=confirming_scene,
            start_stage=start_stage,
            split_allowed=resumable,
            gallery_request=(qualification or {}).get("evidence_gallery"),
        )
        if confirming_scene and not any(item.get("samples") is not None for item in result.objects if item.get("status")=="scene_observations"):
            result=replace(result,objects=[*result.objects,{"status":"scene_observations","observations":[],
                           "samples":_scene_sample_records([],[],event_at.timestamp(),self.camera.id,_flatten_refinement_stages(stages)),
                           "coverage":{"state":"incomplete","sampled_frames":0}}])
        return result

    def _live_discovery_result(self, event_at: datetime) -> RecordedDetectionResult | None:
        """Sample fresh capture through shared inference without waiting for segment closure.

        Delayed/replayed jobs must still read their original recorded interval;
        a current image is not evidence of an old scene.
        """
        started = time.monotonic()
        with self._scene_frames_lock:
            sample = next((frame for frame in reversed(self._scene_frames)
                           if abs(frame.captured_at_epoch - event_at.timestamp()) <= 1e-6), None)
        retained = sample is not None
        if sample is None:
            sample = self.timestamped_live_frame_provider() if self.timestamped_live_frame_provider else None
        if not isinstance(sample, TimestampedLiveFrame):
            return None
        captured = float(sample.captured_at_epoch)
        age = time.time() - captured
        if not (
            sample.source == "live" and sample.sequence > 0
            and sample.camera_generation > 0 and sample.capture_generation > 0
            and math.isfinite(captured) and math.isfinite(sample.captured_at_monotonic)
            and age >= -FAST_LIVE_FRAME_FUTURE_TOLERANCE_SECONDS
            and (retained or age <= FAST_LIVE_FRAME_MAX_AGE_SECONDS)
            and abs(captured - event_at.timestamp()) <= RECORDED_LIVE_FALLBACK_EVENT_TOLERANCE_SECONDS
        ):
            return None
        timing = {"detector_request_ms": 0.0, "detection_enrichment_ms": 0.0}
        objects = self._detect_objects(sample.frame, timing=timing, enrich_faces=False, workload="refinement")
        zone_geometry_required = any(
            zone.enabled and zone.behavior in {"incident", "ignore"} and len(zone.points) >= 3
            for zone in self.camera.zones
        )
        for detected in objects:
            if not detected.get("label"):
                continue
            detected.update({
                "frame_source": "live_discovery", "frame_captured_at_epoch": captured,
                "frame_sequence": sample.sequence, "camera_generation": sample.camera_generation,
                "capture_generation": sample.capture_generation,
                "frame_geometry_trusted": sample.geometry_trusted,
            })
            if zone_geometry_required and not sample.geometry_trusted:
                detected["incident_eligible"] = False
                detected["fast_geometry_untrusted"] = True
        observations = [observation for detected in objects
                        if (observation := scene_observation(
                            detected, captured_at_epoch=captured, frame_source="live_discovery",
                        )) is not None]
        # Use the same confirmation policy as recorded discovery. One acquired
        # observation belongs to the scene even when it cannot justify an alert.
        _, presented = _temporal_consensus(
            [_RecordedDetectionSample(captured - event_at.timestamp(), sample.frame, objects, "")],
            self.detector.config.event_confirmation_frames,
            dict(self.detector.config.event_class_confirmation_frames),
        )
        return self._result(
            sample.frame, presented, "", timing, started, refinement_pending=False,
            frame_captured_at_epoch=captured, frame_source="live_discovery",
            scene_observations=observations,
        )

    def detect_initial(
        self,
        event_at: datetime,
        evidence: dict[str, Any] | None = None,
    ) -> RecordedDetectionResult:
        """Run one strictly fresh live-frame check and always refine later.

        Finalized recordings intentionally lag the live edge by a segment. The
        initial security path must not wait for that boundary. A stale or
        unavailable live frame therefore produces a provisional no-frame result
        rather than cancelling the authoritative recorded refinement.
        """
        workflow_started = time.monotonic()
        timing = {
            "recording_wait_ms": 0.0,
            "frame_decode_ms": 0.0,
            "detector_request_ms": 0.0,
            "detection_enrichment_ms": 0.0,
            "temporal_confirmation_wait_ms": 0.0,
            "recording_batch_processes": 0.0,
            "recording_fallback_samples": 0.0,
            "recording_samples_requested": 0.0,
            "recording_samples_decoded": 0.0,
        }
        evidence_provider = self.timestamped_evidence_frame_provider
        provider = self.timestamped_live_frame_provider
        evidence_token = evidence or {}
        has_evidence_token = bool(
            evidence_token.get("evidence_frame_at_epoch") is not None
            and int(evidence_token.get("evidence_frame_sequence") or 0) > 0
            and int(evidence_token.get("evidence_capture_generation") or 0) > 0
            and int(evidence_token.get("evidence_lifecycle_generation") or 0) > 0
        )
        evidence_required = bool(evidence_token.get("evidence_frame_required"))
        if has_evidence_token:
            sample = (
                evidence_provider(evidence_token)
                if evidence_provider is not None
                else None
            )
        elif evidence_required:
            sample = None
        else:
            sample = provider() if provider is not None else None
        if sample is None:
            return self._result(
                None,
                [{"status": "fast_frame_unavailable", "frame_source": "live_fast_path"}],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        if isinstance(sample, TimestampedLiveFrame):
            frame = sample.frame
            captured_at = float(sample.captured_at_epoch)
            sequence = int(sample.sequence)
            generation = int(sample.camera_generation)
            capture_generation = int(sample.capture_generation)
            source = str(sample.source or "live")
            geometry_trusted = bool(sample.geometry_trusted)
            provenance_valid = bool(
                source == "live"
                and sequence > 0
                and generation > 0
                and capture_generation > 0
                and math.isfinite(float(sample.captured_at_monotonic))
            )
        else:
            # Compatibility for external factories/tests. It remains fresh but
            # cannot claim capture-generation provenance.
            frame, captured_at = sample
            sequence = 0
            generation = 0
            capture_generation = 0
            source = "live"
            geometry_trusted = True
            provenance_valid = True
        if not math.isfinite(float(captured_at)) or not provenance_valid:
            return self._result(
                None,
                [{
                    "status": "fast_frame_invalid_provenance",
                    "frame_source": "live_fast_path",
                    "frame_sequence": sequence,
                    "camera_generation": generation,
                    "capture_generation": capture_generation,
                }],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        frame_age = time.time() - float(captured_at)
        if (
            frame_age > FAST_LIVE_FRAME_MAX_AGE_SECONDS
            or frame_age < -FAST_LIVE_FRAME_FUTURE_TOLERANCE_SECONDS
        ):
            return self._result(
                None,
                [{
                    "status": "fast_frame_stale",
                    "frame_source": "live_fast_path",
                    "frame_age_ms": round(frame_age * 1000.0, 3),
                }],
                "",
                timing,
                workflow_started,
                refinement_pending=True,
            )
        objects = self._detect_objects(
            frame,
            timing=timing,
            enrich_faces=False,
            workload="initial",
        )
        zone_geometry_required = any(
            zone.enabled
            and zone.behavior in {"incident", "ignore"}
            and len(zone.points) >= 3
            for zone in self.camera.zones
        )
        for detected in objects:
            if isinstance(detected, dict):
                detected.update({
                    "frame_source": "live_fast_path",
                    "provisional_detection": True,
                    "frame_captured_at_epoch": round(float(captured_at), 6),
                    "frame_age_ms": round(max(0.0, frame_age) * 1000.0, 3),
                    "frame_sequence": sequence,
                    "camera_generation": generation,
                    "capture_generation": capture_generation,
                    "frame_geometry_trusted": geometry_trusted,
                })
                if zone_geometry_required and not geometry_trusted:
                    # A substream may use a different crop/FOV than the main
                    # image on which zones were drawn. Preserve the fast
                    # observation, but wait for the main-recording refinement
                    # before allowing zone-based object admission or tracking.
                    detected["provisional_zone_eligible"] = bool(
                        detected.get("incident_eligible")
                    )
                    detected["incident_eligible"] = False
                    detected["fast_geometry_untrusted"] = True
        return self._result(
            frame,
            objects,
            "",
            timing,
            workflow_started,
            refinement_pending=True,
            frame_captured_at_epoch=float(captured_at),
            frame_source="live_fast_path",
            # Capture receipt time is bounded and fresh, but it is not a
            # source-PTS timestamp from the camera stream.
            frame_timestamp_exact=False,
        )

    def _detect(
        self,
        event_at: datetime,
        *,
        stages: tuple[tuple[float, ...], ...],
        retry_seconds: float,
        allow_representative_refinement: bool,
        refinement_pending: bool,
        route_dense: bool = False,
        settle_seconds: float = RECORDED_EVENT_SETTLE_SECONDS,
        retry_interval_seconds: float = RECORDED_EVENT_RETRY_INTERVAL_SECONDS,
        representative_timeout_seconds: float = RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS,
        minimum_last_offset_seconds: float | None = None,
        preserve_confirmation_counts: bool = False,
        scene_confirmation: bool = False,
        start_stage: int = 0,
        split_allowed: bool = False,
        gallery_request: dict | None = None,
    ) -> RecordedDetectionResult:
        workflow_started = time.monotonic()
        timing = {
            "recording_wait_ms": 0.0,
            "frame_decode_ms": 0.0,
            "detector_request_ms": 0.0,
            "detection_enrichment_ms": 0.0,
            "temporal_confirmation_wait_ms": 0.0,
            "recording_batch_processes": 0.0,
            "recording_fallback_samples": 0.0,
            "recording_samples_requested": 0.0,
            "recording_samples_decoded": 0.0,
            "decode_budget_wait_ms": 0.0,
            "refinement_early_exit_skipped": 0.0,
            "refinement_route_dense_plan": float(route_dense),
        }
        event_epoch = event_at.timestamp()
        planned_offsets = _flatten_refinement_stages(stages)
        timing["refinement_plan_samples"] = float(len(planned_offsets))
        initial_offsets = stages[0]
        adaptive_initial_offsets = tuple(
            offset for offset in (0.0, 0.5, -0.5) if offset in initial_offsets
        )
        newest_initial_offset = (
            max(adaptive_initial_offsets)
            if adaptive_initial_offsets
            and self.detector.config.recorded_adaptive_sampling
            else max(initial_offsets)
        )
        newest_needed = (
            event_epoch
            + newest_initial_offset
            + settle_seconds
        )
        wait_seconds = max(0.0, newest_needed - time.time()) if start_stage == 0 else 0.0
        if wait_seconds > 0:
            slept = min(wait_seconds, 3.0)
            remaining_wait = slept
            while remaining_wait > 0:
                check_evidence_cancellation()
                pause = min(remaining_wait, 0.05)
                time.sleep(pause)
                remaining_wait -= pause
            timing["temporal_confirmation_wait_ms"] += slept * 1000.0

        deadline = time.monotonic() + max(0.0, retry_seconds)
        report_evidence_stage("recording_lookup")
        prefetched_rows = self._prefetch_recording_rows(
            event_epoch=event_epoch,
            planned_offsets=planned_offsets,
            timing=timing,
        )
        memory_lease = None
        budget = self.decode_budget
        if budget is not None:
            maximum_frames = refinement_frame_count(stages)
            if (self.detector.config.face_recognition_enabled
                    and self.detector.config.face_evidence_enabled):
                maximum_frames += self.detector.config.face_evidence_max_extra_frames
            frame_bytes = self._frame_bytes_for_rows(prefetched_rows)
            if frame_bytes:
                budget.observe_frame_bytes(frame_bytes)
            budget_wait_started = time.monotonic()
            report_evidence_stage("decode_memory_wait")
            memory_lease = budget.reserve_workflow(
                maximum_frames=maximum_frames,
                frame_bytes=frame_bytes or None,
                camera_id=self.camera.id,
                incident_epoch=event_epoch,
                deadline=deadline,
                cancelled=self.stop_requested,
            )
            timing["decode_budget_wait_ms"] += (
                time.monotonic() - budget_wait_started
            ) * 1000.0
            if memory_lease is None:
                if self.stop_requested():
                    return self._result(
                        None,
                        [{"status": "cancelled"}],
                        "",
                        timing,
                        workflow_started,
                        refinement_pending=False,
                    )
                return self._result(
                    None,
                    [{"status": "no_recorded_frame", "reason": "decode_budget_timeout"}],
                    "",
                    timing,
                    workflow_started,
                    refinement_pending=refinement_pending,
                )

        try:
            return self._detect_with_budget(
                event_at,
                stages=stages,
                retry_seconds=retry_seconds,
                settle_seconds=settle_seconds,
                retry_interval_seconds=retry_interval_seconds,
                representative_timeout_seconds=representative_timeout_seconds,
                allow_representative_refinement=allow_representative_refinement,
                refinement_pending=refinement_pending,
                workflow_started=workflow_started,
                timing=timing,
                event_epoch=event_epoch,
                deadline=deadline,
                planned_offsets=planned_offsets,
                prefetched_rows=prefetched_rows,
                minimum_last_offset_seconds=minimum_last_offset_seconds,
                preserve_confirmation_counts=preserve_confirmation_counts,
                scene_confirmation=scene_confirmation,
                start_stage=start_stage,
                split_allowed=split_allowed,
                gallery_request=gallery_request,
            )
        finally:
            if memory_lease is not None:
                memory_lease.release()

    def _prefetch_recording_rows(
        self,
        *,
        event_epoch: float,
        planned_offsets: tuple[float, ...],
        timing: dict[str, float],
    ) -> list[dict[str, Any]]:
        rows_between = getattr(self.recorder, "recording_rows_between", None)
        if not callable(rows_between):
            return []
        lookup_started = time.monotonic()
        try:
            return list(rows_between(
                self.camera.id,
                event_epoch + min(planned_offsets) - 1.0,
                event_epoch + max(planned_offsets) + 1.0,
                "main",
                discover_missing=False,
            ))
        except Exception:
            LOGGER.debug(
                "recording range prefetch unavailable for %s",
                self.camera.id,
                exc_info=True,
            )
            return []
        finally:
            timing["recording_wait_ms"] += (
                time.monotonic() - lookup_started
            ) * 1000.0

    @staticmethod
    def _frame_bytes_for_rows(rows: list[dict[str, Any]]) -> int:
        frame_bytes = 0
        for row in rows:
            path_value = str(row.get("path") or "")
            if not path_value:
                continue
            dimensions = mp4_video_dimensions(Path(path_value))
            if dimensions is None:
                continue
            width, height = dimensions
            frame_bytes = max(frame_bytes, width * height * 3)
        return frame_bytes

    def _detect_with_budget(
        self,
        event_at: datetime,
        *,
        stages: tuple[tuple[float, ...], ...],
        retry_seconds: float,
        allow_representative_refinement: bool,
        refinement_pending: bool,
        workflow_started: float,
        timing: dict[str, float],
        event_epoch: float,
        deadline: float,
        planned_offsets: tuple[float, ...],
        prefetched_rows: list[dict[str, Any]],
        settle_seconds: float = RECORDED_EVENT_SETTLE_SECONDS,
        retry_interval_seconds: float = RECORDED_EVENT_RETRY_INTERVAL_SECONDS,
        representative_timeout_seconds: float = RECORDED_EVENT_REFINEMENT_TIMEOUT_SECONDS,
        minimum_last_offset_seconds: float | None = None,
        preserve_confirmation_counts: bool = False,
        scene_confirmation: bool = False,
        start_stage: int = 0,
        split_allowed: bool = False,
        gallery_request: dict | None = None,
    ) -> RecordedDetectionResult:
        refinement_deadline: float | None = None
        samples_by_offset: dict[float, _RecordedDetectionSample] = {}
        samples_by_source_frame: dict[tuple[str, float], _RecordedDetectionSample] = {}

        def ordered_samples() -> list[_RecordedDetectionSample]:
            # Different requested slots can resolve to one source frame. Keep
            # those slots satisfied, but count their shared evidence only once.
            unique: dict[int, _RecordedDetectionSample] = {}
            for offset in planned_offsets:
                sample = samples_by_offset.get(offset)
                if sample is not None:
                    unique.setdefault(id(sample), sample)
            return list(unique.values())

        samples: list[_RecordedDetectionSample] = []
        confirmation_limit = 5 if preserve_confirmation_counts else len(planned_offsets)
        default_required = max(
            1,
            min(
                confirmation_limit,
                self.detector.config.event_confirmation_frames,
            ),
        )
        class_confirmations = {
            str(label).strip().lower(): max(
                1,
                min(confirmation_limit, int(confirmations)),
            )
            for label, confirmations in dict(
                self.detector.config.event_class_confirmation_frames
            ).items()
        }
        # Cover-image recovery needs a later, settled view.  The early bridge
        # improves temporal confirmation but must not replace that recovery
        # pass when its subject is still clipped or obscured.
        representative_stage_index = next(
            (
                stage_index
                for stage_index, stage_offsets in enumerate(stages[1:], start=1)
                if any(math.isclose(offset, 4.0, abs_tol=0.001) for offset in stage_offsets)
            ),
            None,
        )
        if representative_stage_index is None and len(stages) > 1:
            representative_stage_index = 1
        representative_refinement_requested = False
        def frame_reader(
            path: Path,
            offset_seconds: float,
            *,
            deadline: float | None = None,
        ) -> Frame | None:
            return self._read_recorded_frame(
                path,
                offset_seconds,
                deadline=deadline,
                incident_epoch=event_epoch,
                budget_wait_sink=timing,
            )

        def batch_frame_reader(
            path: Path,
            offsets_seconds: list[float],
            *,
            deadline: float | None = None,
        ) -> tuple[dict[float, _DecodedRecordedFrame], int]:
            return self._read_recorded_frames(
                path,
                offsets_seconds,
                deadline=deadline,
                incident_epoch=event_epoch,
                budget_wait_sink=timing,
            )

        sampler = _EventRecordedSampler(
            camera_id=self.camera.id,
            recorder=self.recorder,
            frame_reader=frame_reader,
            batch_frame_reader=batch_frame_reader,
            rows=prefetched_rows,
        )

        for stage_index, stage_offsets in enumerate(stages):
            if stage_index < start_stage:
                continue
            if (
                representative_refinement_requested
                and representative_stage_index is not None
                and 0 < stage_index < representative_stage_index
            ):
                continue
            if (
                split_allowed
                and stage_index > start_stage
                and not representative_refinement_requested
                # Only an empty prefix can be dropped: later consensus never
                # needs a track that has not started.
                and not any(
                    _candidate_detection(detected)
                    for sample in samples_by_offset.values()
                    for detected in sample.objects
                )
            ):
                resume_at = self._predicted_index_epoch(event_epoch + max(stage_offsets), settle_seconds)
                if resume_at - time.time() > RECORDED_SPLIT_MINIMUM_WAIT_SECONDS:
                    timing["refinement_resume_stage"] = float(stage_index)
                    return replace(
                        self._result(
                            None, [{"status": "refinement_resumes_later"}], "",
                            timing, workflow_started, refinement_pending=False,
                        ),
                        resume_stage=stage_index,
                        resume_at=resume_at,
                    )
            adaptive_stage = bool(
                stage_index == 0
                and not scene_confirmation
                and self.detector.config.recorded_adaptive_sampling
            )
            if adaptive_stage:
                preferred_order = (0.0, 0.5, -0.5, -1.0, 1.0)
                ordered_stage_offsets = tuple(
                    offset for offset in preferred_order if offset in stage_offsets
                ) + tuple(
                    offset for offset in stage_offsets if offset not in preferred_order
                )
                adaptive_limit = 1
            else:
                ordered_stage_offsets = stage_offsets
                adaptive_limit = len(stage_offsets)
            # A just-closed segment is usually not indexed yet: the first miss
            # asks the indexer for it. Wait for that one discovery rather than a
            # full retry interval. Later misses mean the segment is still open
            # and keep the plain interval, so discovery is not requested faster.
            index_refresh_wait_available = True
            while True:
                if self.stop_requested():
                    return self._result(
                        None,
                        [{"status": "cancelled"}],
                        "",
                        timing,
                        workflow_started,
                        refinement_pending=False,
                    )
                index_generation = self._index_refresh_generation()
                stage_deadline = min(
                    deadline,
                    refinement_deadline
                    if refinement_deadline is not None
                    else deadline,
                )
                recording_missing = False
                future_sample = False
                stage_consensus = False
                pending_by_path: dict[Path, list[tuple[float, float, dict[str, Any]]]] = {}
                requested_stage_offsets = ordered_stage_offsets[:adaptive_limit]
                for sample_offset in requested_stage_offsets:
                    if sample_offset in samples_by_offset:
                        continue
                    if time.monotonic() >= stage_deadline:
                        break
                    target_epoch = event_epoch + sample_offset
                    if target_epoch + settle_seconds > time.time():
                        future_sample = True
                        continue
                    lookup_started = time.monotonic()
                    row = sampler.recording_at(target_epoch)
                    timing["recording_wait_ms"] += (
                        time.monotonic() - lookup_started
                    ) * 1000.0
                    if row is None:
                        recording_missing = True
                        continue
                    start_epoch = row.get("start_epoch")
                    if start_epoch is None:
                        continue
                    frame_offset = max(0.0, target_epoch - float(start_epoch))
                    pending_by_path.setdefault(Path(str(row["path"])), []).append(
                        (sample_offset, frame_offset, row)
                    )

                for path, pending in list(pending_by_path.items()):
                    if stage_consensus:
                        timing["refinement_early_exit_skipped"] += float(len(pending))
                        continue
                    if self.decode_budget is not None:
                        dimensions = mp4_video_dimensions(path)
                        if dimensions is not None:
                            width, height = dimensions
                            self.decode_budget.observe_frame_bytes(width * height * 3)
                    decode_started = time.monotonic()
                    frames, batch_processes, fallback_count = sampler.frames_at(
                        path,
                        [frame_offset for _, frame_offset, _ in pending],
                        deadline=stage_deadline,
                    )
                    timing["frame_decode_ms"] += (
                        time.monotonic() - decode_started
                    ) * 1000.0
                    timing["recording_batch_processes"] += batch_processes
                    timing["recording_fallback_samples"] += fallback_count
                    timing["recording_samples_requested"] += len(pending)
                    timing["recording_samples_decoded"] += len(frames)
                    for sample_index, (sample_offset, frame_offset, row) in enumerate(
                        pending
                    ):
                        if stage_consensus:
                            timing["refinement_early_exit_skipped"] += float(
                                len(pending) - sample_index
                            )
                            break
                        decoded = frames.get(frame_offset)
                        if decoded is None:
                            continue
                        source_key = (
                            (str(row["path"]), decoded.actual_offset)
                            if decoded.exact_timestamp
                            else None
                        )
                        sample = (
                            samples_by_source_frame.get(source_key)
                            if source_key is not None
                            else None
                        )
                        if sample is None:
                            objects = self._detect_objects(
                                decoded.frame,
                                timing=timing,
                                enrich_faces=False,
                                workload="refinement",
                            )
                            row_start = float(row.get("start_epoch") or 0.0)
                            actual_offset = (
                                row_start + decoded.actual_offset - event_epoch
                            )
                            sample = _RecordedDetectionSample(
                                offset=actual_offset,
                                requested_offset=sample_offset,
                                exact_timestamp=decoded.exact_timestamp,
                                frame=decoded.frame,
                                objects=objects,
                                recording_path=str(row["path"]),
                            )
                            if source_key is not None:
                                samples_by_source_frame[source_key] = sample
                        samples_by_offset[sample_offset] = sample
                        samples = ordered_samples()
                        if not scene_confirmation and (minimum_last_offset_seconds is None or any(
                            item.requested_offset is not None
                            and item.requested_offset >= minimum_last_offset_seconds
                            for item in samples
                        )) and _refinement_early_exit_ready(
                            stage_index=stage_index,
                            stage_offsets=stage_offsets,
                            samples_by_offset=samples_by_offset,
                            samples=samples,
                            minimum_confirmations=default_required,
                            class_confirmations=class_confirmations,
                        ):
                            if not (
                                representative_refinement_requested
                                and stage_index == representative_stage_index
                            ):
                                # Stop spending GPU on remaining offsets once every
                                # observed track is confirmed and the core window
                                # has already been sampled. The designated cover
                                # pass intentionally reads its whole stage so a
                                # better framed subject is not skipped.
                                stage_consensus = True
                                timing["refinement_early_exit_skipped"] += float(
                                    len(pending) - sample_index - 1
                                )

                samples = ordered_samples()
                if samples:
                    selected, objects = _temporal_consensus(
                        samples,
                        default_required,
                        class_confirmations,
                    )
                    if any(item.get("temporal_consensus") is True for item in objects) and (
                        minimum_last_offset_seconds is None or any(
                            item.requested_offset is not None
                            and item.requested_offset >= minimum_last_offset_seconds
                            for item in samples
                        )
                    ):
                        # Detection is already proven. If its cover image clips,
                        # obscures, or barely shows the active/new subject, spend
                        # one existing +4 second stage looking for a better frame.
                        # This is bounded and does not wait through the full
                        # tracking lifecycle or add a second inference pipeline.
                        representative_needs_refinement = bool(
                            stage_index == 0
                            and _representative_needs_refinement(objects)
                        )
                        causal_context_ready = bool(
                            stage_index != 0
                            or any(
                                sample.requested_offset is not None
                                and sample.requested_offset < 0.0
                                for sample in samples
                            )
                        )
                        unresolved_candidate = not _refinement_stage_complete(
                            samples,
                            default_required,
                            class_confirmations,
                        )
                        if (
                            adaptive_stage
                            and not stage_consensus
                            and (
                                not causal_context_ready
                                or representative_needs_refinement
                                or unresolved_candidate
                            )
                            and adaptive_limit < len(ordered_stage_offsets)
                        ):
                            adaptive_limit += 1
                            continue
                        if (
                            allow_representative_refinement
                            and representative_needs_refinement
                            and representative_stage_index is not None
                        ):
                            representative_refinement_requested = True
                            refinement_deadline = min(
                                deadline,
                                time.monotonic()
                                + representative_timeout_seconds,
                            )
                            LOGGER.info(
                                "recorded object representative refinement requested for %s",
                                self.camera.id,
                            )
                            break
                        if (
                            representative_refinement_requested
                            and stage_index < representative_stage_index
                        ):
                            # Preserve the later cover-quality pass even
                            # though the bridge has already confirmed the
                            # subject.
                            break
                        if stage_index:
                            LOGGER.info(
                                "recorded object selection completed for %s at stage +%.1fs",
                                self.camera.id,
                                max(stage_offsets),
                            )
                        return self._recorded_result(
                            selected,
                            objects,
                            samples,
                            timing,
                            workflow_started,
                            refinement_pending=bool(
                                refinement_pending and representative_needs_refinement
                            ),
                            event_epoch=event_epoch,
                            face_sampler=sampler,
                            face_deadline=deadline,
                            confirmation_offsets=planned_offsets if scene_confirmation else None,
                            gallery_request=gallery_request,
                        )
                if (
                    adaptive_stage
                    and not stage_consensus
                    and adaptive_limit < len(ordered_stage_offsets)
                ):
                    adaptive_limit += 1
                    continue
                if stage_consensus or all(
                    offset in samples_by_offset for offset in stage_offsets
                ):
                    break
                remaining = deadline - time.monotonic()
                if refinement_deadline is not None:
                    remaining = min(
                        remaining,
                        refinement_deadline - time.monotonic(),
                    )
                if remaining <= 0:
                    break
                slept = min(retry_interval_seconds, remaining)
                if (
                    recording_missing
                    and not future_sample
                    and index_refresh_wait_available
                    and index_generation is not None
                ):
                    index_refresh_wait_available = False
                    wait_started = time.monotonic()
                    self.recorder.wait_for_recording_edge_refresh(
                        self.camera.id, index_generation, slept,
                    )
                    slept = time.monotonic() - wait_started
                else:
                    time.sleep(slept)
                wait_key = (
                    "recording_wait_ms"
                    if recording_missing and not future_sample
                    else "temporal_confirmation_wait_ms"
                )
                timing[wait_key] += slept * 1000.0
            if (
                refinement_deadline is not None
                and time.monotonic() >= refinement_deadline
            ):
                break
            if time.monotonic() >= deadline:
                break

        if samples:
            selected, objects = _temporal_consensus(
                samples,
                default_required,
                class_confirmations,
            )
            return self._recorded_result(
                selected,
                objects,
                samples,
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
                event_epoch=event_epoch,
                face_sampler=sampler,
                face_deadline=deadline,
                confirmation_offsets=planned_offsets if scene_confirmation else None,
                gallery_request=gallery_request,
            )

        if scene_confirmation:
            return self._result(None,[{"status":"no_recorded_frame"}],"",timing,workflow_started,
                                refinement_pending=False, scene_samples=_scene_sample_records(
                                    [], [], event_epoch, self.camera.id, planned_offsets))
        return self._live_fallback_result(event_epoch, timing, workflow_started, refinement_pending)

    def _live_fallback_result(
        self, event_epoch: float, timing: dict[str, float],
        workflow_started: float, refinement_pending: bool,
    ) -> RecordedDetectionResult:
        fallback_sample = (
            self.timestamped_live_frame_provider()
            if self.timestamped_live_frame_provider is not None
            else None
        )
        fallback = (
            fallback_sample.frame
            if isinstance(fallback_sample, TimestampedLiveFrame)
            else None
        )
        fallback_captured_at = (
            float(fallback_sample.captured_at_epoch)
            if isinstance(fallback_sample, TimestampedLiveFrame)
            else None
        )
        fallback_valid = bool(
            isinstance(fallback_sample, TimestampedLiveFrame)
            and fallback_sample.source == "live"
            and fallback_sample.sequence > 0
            and fallback_sample.camera_generation > 0
            and fallback_sample.capture_generation > 0
            and math.isfinite(float(fallback_sample.captured_at_monotonic))
            and fallback_captured_at is not None
            and math.isfinite(fallback_captured_at)
            and abs(fallback_captured_at - event_epoch)
            <= RECORDED_LIVE_FALLBACK_EVENT_TOLERANCE_SECONDS
        )
        if fallback is None or not fallback_valid:
            return self._result(
                None,
                [{"status": "no_recorded_frame"}],
                "",
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
                face_candidates=(),
            )
        objects = self._detect_objects(
            fallback,
            timing=timing,
            workload="refinement",
        )
        if objects:
            for detected in objects:
                detected["frame_source"] = "live_fallback"
                detected["recording_status"] = "no_recorded_frame"
            return self._result(
                fallback,
                objects,
                "",
                timing,
                workflow_started,
                refinement_pending=refinement_pending,
                frame_captured_at_epoch=fallback_captured_at,
                frame_source="live_fallback",
            )
        return self._result(
            fallback,
            [{"status": "no_recorded_frame", "frame_source": "live_fallback"}],
            "",
            timing,
            workflow_started,
            refinement_pending=refinement_pending,
            frame_captured_at_epoch=fallback_captured_at,
            frame_source="live_fallback",
        )

    @staticmethod
    def _face_candidates(
        samples: list[_RecordedDetectionSample],
    ) -> tuple[FaceCandidate, ...]:
        return collect_face_candidates(
            FaceCandidateSample(
                offset_seconds=sample.offset,
                frame=sample.frame,
                objects=tuple(sample.objects),
            )
            for sample in samples
            if sample.frame is not None
        )

    def _recorded_result(
        self,
        selected: _RecordedDetectionSample,
        objects: list[dict[str, Any]],
        samples: list[_RecordedDetectionSample],
        timing: dict[str, float],
        workflow_started: float,
        *,
        refinement_pending: bool,
        event_epoch: float,
        face_sampler: _EventRecordedSampler | None = None,
        face_deadline: float | None = None,
        confirmation_offsets: tuple[float, ...] | None = None,
        gallery_request: dict | None = None,
    ) -> RecordedDetectionResult:
        frame = selected.frame
        if frame is None:
            raise ValueError("selected recorded frame was released too early")
        # Acquire the whole sampled scene before presentation enrichment can
        # replace the selected sample with a cover-oriented object projection.
        observations: list[dict[str, Any]] = []
        for track in _collect_temporal_evidence(samples):
            motion = _temporal_motion_metrics(track, samples)
            trigger_evidence = _temporal_trigger_evidence(track, samples)
            first_index = min(track.observations)
            first = scene_observation(
                track.observations[first_index],
                captured_at_epoch=event_epoch + samples[first_index].offset,
                frame_source="recorded_main",
                recording_path=samples[first_index].recording_path,
            )
            track_key = f"recorded:{first['observation_key']}" if first else None
            for index, detected in track.observations.items():
                sample = samples[index]
                observation = scene_observation(
                    detected, captured_at_epoch=event_epoch + sample.offset,
                    frame_source="recorded_main", recording_path=sample.recording_path,
                    offset_seconds=sample.offset,
                    frame_timestamp_exact=sample.exact_timestamp,
                    scene_track_key=track_key,
                )
                if observation is not None:
                    observation["temporal_motion"] = motion.as_dict()
                    observation["temporal_center_displacement_ratio"] = round(motion.raw_displacement_ratio, 5)
                    observation["temporal_center_path_ratio"] = round(motion.raw_path_ratio, 5)
                    observation["temporal_track_observations"] = motion.samples
                    observation["temporal_consensus"] = motion.samples >= 2
                    observation.update(trigger_evidence)
                    observation["snapshot_visible"] = sample is selected
                    observations.append(observation)
        # Dedicated face enrichment runs only after consensus. Collect
        # face_candidates afterward so persistence sees those faces.
        if any(item.get("temporal_consensus") is True for item in objects):
            objects = self._enrich_selected_faces(frame, objects, timing=timing)
            objects = self._enrich_depth(
                frame,
                objects,
                selected.offset,
                timing=timing,
            )
            selected.objects = list(objects)
            if face_sampler is not None:
                self._refine_face_evidence(samples, face_sampler, event_epoch, face_deadline, timing)
        # Optional face enrichment can discover more objects or decode extra
        # frames. Those observations join the same ledger. Off-cover objects
        # in the selected presentation projection already have their own
        # source observations above and must not be attributed to this frame.
        acquired_keys = {item["observation_key"] for item in observations}
        for sample in samples:
            for detected in sample.objects:
                if detected.get("snapshot_visible") is False:
                    continue
                observation = scene_observation(
                    detected, captured_at_epoch=event_epoch + sample.offset,
                    frame_source="recorded_main", recording_path=sample.recording_path,
                    offset_seconds=sample.offset, frame_timestamp_exact=sample.exact_timestamp,
                )
                if observation is not None and observation["observation_key"] not in acquired_keys:
                    observation["snapshot_visible"] = sample is selected
                    observations.append(observation)
                    acquired_keys.add(observation["observation_key"])
        face_candidates = self._face_candidates(samples)
        for detected in objects:
            if detected.get("label"):
                detected["frame_captured_at_epoch"] = event_epoch + float(
                    detected.get("temporal_sample_offset_seconds", selected.offset)
                )
                detected["frame_source"] = "recorded_main"
        attributor = self.activity_attributor
        if attributor is not None and observations:
            attributor.stamp_recorded(
                observations,
                event_key=datetime.fromtimestamp(event_epoch, timezone.utc).isoformat(),
                observed_at_epoch=event_epoch,
            )
        scene_samples = _scene_sample_records(samples,observations,event_epoch,
            self.camera.id, confirmation_offsets)
        gallery = encode_gallery(samples, selected, observations,
            camera_id=self.camera.id, event_epoch=event_epoch, request=gallery_request, timing=timing)
        self._release_nonselected_frames(samples, selected)
        review_image = ({"role":"additional_review","source":"recorded_main",
                         "captured_at_epoch":event_epoch+selected.offset,
                         "width":int(frame.shape[1]),"height":int(frame.shape[0]),
                         "analyzed_frame":True,"frame_timestamp_exact":selected.exact_timestamp,
                         "recording_path":selected.recording_path}
                        if confirmation_offsets is not None else None)
        return self._result(
            frame,
            objects,
            selected.recording_path,
            timing,
            workflow_started,
            refinement_pending=refinement_pending,
            face_candidates=face_candidates,
            frame_captured_at_epoch=event_epoch + selected.offset,
            frame_source="recorded_main",
            frame_timestamp_exact=selected.exact_timestamp,
            scene_observations=observations,
            scene_samples=scene_samples,
            review_image=review_image,
            gallery=gallery,
        )

    def _refine_face_evidence(
        self,
        samples: list[_RecordedDetectionSample],
        sampler: _EventRecordedSampler,
        event_epoch: float,
        deadline: float | None,
        timing: dict[str, float],
    ) -> None:
        """Spend a bounded part of the existing decode lease on identity evidence.

        Samples remain at recording resolution. Extra detection results are used
        only for face candidates; they cannot change event qualification/cover.
        """
        config = self.detector.config
        if not config.face_recognition_enabled or not config.face_evidence_enabled:
            return
        started = time.monotonic()
        deadline = min(
            deadline if deadline is not None else started,
            started + float(config.face_evidence_timeout_seconds),
        )
        max_extra = config.face_evidence_max_extra_frames
        timing["face_evidence_samples"] = 0.0
        timing["face_evidence_failed"] = 0.0

        def usefulness(sample: _RecordedDetectionSample) -> float:
            if sample.frame is None:
                return 0.0
            areas = []
            height, width = sample.frame.shape[:2]
            for item in sample.objects:
                if item.get("label") not in {"person", "pedestrian"} or item.get("incident_eligible") is False:
                    continue
                box = _box(item)
                if box is not None:
                    x1, y1, x2, y2 = box
                    crop = sample.frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
                    if crop.size:
                        areas.append(crop.shape[0] * crop.shape[1] * _image_quality(crop).score)
            return max(areas, default=0.0)

        ranked = sorted(((usefulness(sample), sample) for sample in samples), key=lambda item: item[0], reverse=True)
        anchors = [sample for value, sample in ranked if value > 0][:3]
        if not anchors:
            return
        try:
            # Existing full-resolution frames are cheaper than another decode.
            for sample in anchors:
                if self.stop_requested() or time.monotonic() >= deadline:
                    break
                check_evidence_cancellation()
                if any(item.get("label") == "face" for item in sample.objects):
                    continue
                confirmed = [
                    {**item, "temporal_consensus": True} for item in sample.objects
                    if item.get("label") in {"person", "pedestrian"}
                    and item.get("incident_eligible") is not False
                ]
                sample.objects = self._enrich_selected_faces(sample.frame, confirmed, timing=timing)
                timing["face_evidence_samples"] += 1
            existing = [sample.offset for sample in samples]
            offsets = []
            for delta in (-.4, .4, -.8, .8):
                for anchor in anchors[:2]:
                    target = round(anchor.offset + delta, 3)
                    if all(abs(target - other) >= .15 for other in existing + offsets):
                        offsets.append(target)
            for target in offsets[:max_extra]:
                if self.stop_requested() or time.monotonic() >= deadline:
                    break
                check_evidence_cancellation()
                if event_epoch + target + RECORDED_EVENT_SETTLE_SECONDS > time.time():
                    continue
                row = sampler.recording_at(event_epoch + target)
                if row is None or row.get("start_epoch") is None:
                    timing["face_evidence_recording_unavailable"] = timing.get("face_evidence_recording_unavailable", 0) + 1
                    continue
                relative = round(event_epoch + target - float(row["start_epoch"]), 3)
                decode_started = time.monotonic()
                frames, processes, fallbacks = sampler.frames_at(
                    Path(str(row["path"])), [relative], deadline=deadline,
                )
                timing["frame_decode_ms"] += (time.monotonic() - decode_started) * 1000
                timing["recording_batch_processes"] += processes
                timing["recording_fallback_samples"] += fallbacks
                timing["recording_samples_requested"] += 1
                timing["recording_samples_decoded"] += len(frames)
                decoded = frames.get(relative)
                # Nudged fallback frames do not establish independent timestamps.
                if decoded is None or not decoded.exact_timestamp:
                    timing["face_evidence_inexact_or_missing"] = timing.get("face_evidence_inexact_or_missing", 0) + 1
                    continue
                actual = float(row["start_epoch"]) + decoded.actual_offset - event_epoch
                if any(abs(actual - sample.offset) < .15 for sample in samples):
                    continue
                if time.monotonic() >= deadline:
                    break
                detected = self._detect_objects(decoded.frame, timing=timing)
                samples.append(_RecordedDetectionSample(
                    actual, decoded.frame, detected, str(row["path"]), target, True,
                ))
                timing["face_evidence_samples"] += 1
        except Exception:
            check_evidence_cancellation()
            timing["face_evidence_failed"] = 1.0
            LOGGER.exception("Optional face evidence refinement failed for camera %s", self.camera.id)
        finally:
            timing["face_evidence_ms"] = (time.monotonic() - started) * 1000
            timing["face_evidence_deadline_reached"] = float(time.monotonic() >= deadline)

    @staticmethod
    def _release_nonselected_frames(
        samples: list[_RecordedDetectionSample],
        selected: _RecordedDetectionSample,
    ) -> None:
        for sample in samples:
            if sample is not selected:
                sample.frame = None

    @staticmethod
    def _result(
        frame: Frame | None,
        objects: list[dict[str, Any]],
        recording_path: str,
        timing: dict[str, float],
        workflow_started: float,
        *,
        refinement_pending: bool,
        face_candidates: tuple[FaceCandidate, ...] = (),
        frame_captured_at_epoch: float | None = None,
        frame_source: str = "",
        frame_timestamp_exact: bool = False,
        scene_observations: list[dict[str, Any]] | None = None,
        scene_samples: list[dict[str, Any]] | None = None,
        review_image: dict[str, Any] | None = None,
        gallery: GalleryBatch | None = None,
    ) -> RecordedDetectionResult:
        normalized = {key: round(max(0.0, value), 3) for key, value in timing.items()}
        normalized["workflow_ms"] = round(
            max(0.0, (time.monotonic() - workflow_started) * 1000.0),
            3,
        )
        if scene_observations is None:
            scene_observations = [
                observation for detected in objects
                if (observation := scene_observation(
                    detected, captured_at_epoch=frame_captured_at_epoch,
                    frame_source=frame_source, recording_path=recording_path,
                    frame_timestamp_exact=frame_timestamp_exact,
                )) is not None
            ]
        # This batch is a ledger input, not an additional displayed object.
        if frame is not None or scene_observations or scene_samples:
            objects = [*objects, {
                "status": "scene_observations", "observations": scene_observations,
                **({"samples":scene_samples} if scene_samples is not None else {}),
                **({"review_image":review_image} if review_image is not None else {}),
                "coverage": {"sampled_frames": int(timing.get("recording_samples_decoded", 0))
                             or int(frame is not None),
                             "state": "pending" if refinement_pending else "sampled"},
            }]
        return RecordedDetectionResult(
            frame=frame,
            objects=objects,
            recording_path=recording_path,
            timings_ms=normalized,
            refinement_pending=refinement_pending,
            face_candidates=face_candidates,
            frame_captured_at_epoch=frame_captured_at_epoch,
            frame_source=frame_source,
            frame_timestamp_exact=frame_timestamp_exact,
            review_image=review_image,
            gallery=gallery,
        )

    def _detect_objects(
        self,
        frame: Frame,
        *,
        timing: dict[str, float] | None = None,
        enrich_faces: bool = True,
        workload: str = "refinement",
    ) -> list[dict[str, Any]]:
        check_evidence_cancellation()
        report_evidence_stage("inference")
        enrichment_started = time.monotonic()
        configured_threshold = float(self.detector.config.confidence_threshold)
        class_thresholds = dict(self.detector.config.event_class_confidence_thresholds)
        # Alert confidence and zones may change without changing what the
        # scene acquisition model observes.
        candidate_threshold = float(self.detector.config.event_candidate_confidence_threshold)
        detector_started = time.monotonic()
        detector_method = getattr(
            self.detector,
            "detect_initial" if workload == "initial" else "detect_refinement",
            self.detector.detect,
        )
        objects = detector_method(
            frame,
            confidence_threshold=candidate_threshold,
        )
        check_evidence_cancellation()
        detector_ms = (time.monotonic() - detector_started) * 1000.0
        if timing is not None:
            timing["detector_request_ms"] += detector_ms
        frame_height, frame_width = frame.shape[:2]
        detect_faces = getattr(self.detector, "detect_faces", None)
        if enrich_faces and callable(detect_faces):
            dedicated_faces = self._detect_faces_in_people(
                frame,
                objects,
                detect_faces,
                max_people=8,
            )
            objects = self._merge_dedicated_faces(objects, dedicated_faces)
        for detected in objects:
            if isinstance(detected, dict) and detected.get("label"):
                detected["detection_frame_width"] = int(frame_width)
                detected["detection_frame_height"] = int(frame_height)
                detected["scene_confirmation_threshold"] = float(
                    self.detector.config.tracking.confirmation_confidence_threshold
                )
                if str(detected.get("label") or "").strip().lower() == "face":
                    box = _box(detected)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        left = max(0, min(frame_width, int(math.floor(x1))))
                        top = max(0, min(frame_height, int(math.floor(y1))))
                        right = max(left, min(frame_width, int(math.ceil(x2))))
                        bottom = max(top, min(frame_height, int(math.ceil(y2))))
                        quality = _image_quality(frame[top:bottom, left:right])
                        detected["face_quality_score"] = round(quality.score, 4)
                        detected["face_sharpness_score"] = round(quality.sharpness, 4)
                        detected["face_exposure_score"] = round(quality.exposure, 4)
        apply_detection_zones(
            self.camera,
            objects,
            int(frame_width),
            int(frame_height),
            configured_threshold,
            self.detector.config.require_incident_zone,
            class_thresholds,
        )
        apply_depth_zone_filters(self.camera, objects)
        for detected in objects:
            if isinstance(detected, dict):
                detected["temporal_candidate_threshold"] = candidate_threshold
                detected["temporal_candidate_eligible"] = bool(
                    detected.get("label")
                    and _confidence(detected) >= candidate_threshold
                )
            if isinstance(detected, dict) and str(
                detected.get("label") or ""
            ).strip().lower() == "face":
                if detected.get("detection_source") == "dedicated_face":
                    detected["confidence_eligible"] = True
                detected["incident_eligible"] = False
                detected["auxiliary_detection"] = True
        if timing is not None:
            timing["detection_enrichment_ms"] += max(
                0.0,
                (time.monotonic() - enrichment_started) * 1000.0 - detector_ms,
            )
        return objects

    def _enrich_selected_faces(
        self,
        frame: Frame,
        objects: list[dict[str, Any]],
        *,
        timing: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run bounded face enrichment only after recorded consensus."""
        detect_faces = getattr(self.detector, "detect_faces", None)
        if not callable(detect_faces):
            return objects
        check_evidence_cancellation()
        report_evidence_stage("inference")
        enrichment_started = time.monotonic()
        confirmed_objects = [
            item
            for item in objects
            if item.get("temporal_consensus") is True
            and item.get("snapshot_visible") is not False
        ]
        dedicated_faces = self._detect_faces_in_people(
            frame,
            confirmed_objects,
            detect_faces,
            max_people=max(
                1,
                self.detector.config.face_enrich_max_people,
            ),
        )
        merged = self._merge_dedicated_faces(objects, dedicated_faces)
        frame_height, frame_width = frame.shape[:2]
        candidate_threshold = float(self.detector.config.event_candidate_confidence_threshold)
        for detected in merged:
            if (
                not isinstance(detected, dict)
                or str(detected.get("label") or "").strip().lower() != "face"
            ):
                continue
            detected.setdefault("detection_source", "dedicated_face")
            detected["detection_frame_width"] = int(frame_width)
            detected["detection_frame_height"] = int(frame_height)
            detected["confidence_eligible"] = True
            detected["incident_eligible"] = False
            detected["zone_eligible"] = False
            detected["temporal_eligible"] = False
            detected["temporal_consensus"] = False
            detected["auxiliary_detection"] = True
            detected["temporal_candidate_threshold"] = candidate_threshold
            detected["temporal_candidate_eligible"] = bool(
                _confidence(detected) >= candidate_threshold
            )
            box = _box(detected)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            left = max(0, min(frame_width, int(math.floor(x1))))
            top = max(0, min(frame_height, int(math.floor(y1))))
            right = max(left, min(frame_width, int(math.ceil(x2))))
            bottom = max(top, min(frame_height, int(math.ceil(y2))))
            quality = _image_quality(frame[top:bottom, left:right])
            detected["face_quality_score"] = round(quality.score, 4)
            detected["face_sharpness_score"] = round(quality.sharpness, 4)
            detected["face_exposure_score"] = round(quality.exposure, 4)
        if timing is not None:
            timing["detection_enrichment_ms"] += (
                time.monotonic() - enrichment_started
            ) * 1000.0
        return merged

    def _enrich_depth(
        self,
        frame: Frame,
        objects: list[dict[str, Any]],
        frame_offset_s: float,
        *,
        timing: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        depth_config = self.detector.config.depth
        if not depth_config.enabled:
            return objects
        estimate_depth = getattr(self.detector, "estimate_depth_for_objects", None)
        if not callable(estimate_depth):
            return objects
        check_evidence_cancellation()
        report_evidence_stage("inference")
        enrichment_started = time.monotonic()
        visible_objects = [
            item
            for item in objects
            if item.get("snapshot_visible") is not False
        ]
        eligible = [
            item
            for item in visible_objects
            if isinstance(item, dict)
            and item.get("label")
            and item.get("incident_eligible") is not False
        ]
        if not eligible:
            return objects
        enriched, metadata = estimate_depth(
            frame,
            visible_objects,
            frame_offset_s=frame_offset_s,
        )
        enriched_iter = iter(enriched)
        merged = [
            item if item.get("snapshot_visible") is False else next(enriched_iter, item)
            for item in objects
        ]
        merged.extend(enriched_iter)
        apply_depth_zone_filters(self.camera, merged)
        if timing is not None:
            timing["depth_enrichment_ms"] = round(
                (time.monotonic() - enrichment_started) * 1000.0,
                3,
            )
            if metadata.get("inference_ms") is not None:
                timing["depth_inference_ms"] = float(metadata["inference_ms"])
        return merged

    @staticmethod
    def _detect_faces_in_people(
        frame: Frame,
        objects: list[dict[str, Any]],
        detect_faces: Callable[[Frame], list[dict[str, Any]]],
        *,
        max_people: int,
    ) -> list[dict[str, Any]]:
        """Run the 300px face model where CCTV faces retain useful resolution."""
        frame_height, frame_width = frame.shape[:2]
        detected_faces: list[dict[str, Any]] = []
        people = [
            item
            for item in objects
            if isinstance(item, dict)
            and str(item.get("label") or "").strip().lower() == "person"
            and _box(item) is not None
        ]
        for person in sorted(people, key=_confidence, reverse=True)[:max_people]:
            check_evidence_cancellation()
            box = _box(person)
            if box is None:
                continue
            window = upper_body_window(box, frame_width, frame_height)
            if window is None:
                continue
            left, top, right, bottom = window
            for detected in detect_faces(frame[top:bottom, left:right]):
                if not isinstance(detected, dict):
                    continue
                face_box = _box(detected)
                if face_box is None:
                    continue
                fx1, fy1, fx2, fy2 = face_box
                detected_faces.append(
                    {
                        **detected,
                        "box": {
                            "x1": fx1 + left,
                            "y1": fy1 + top,
                            "x2": fx2 + left,
                            "y2": fy2 + top,
                        },
                        "parent_person_box": person.get("box"),
                    }
                )
        return detected_faces

    @staticmethod
    def _merge_dedicated_faces(
        objects: list[dict[str, Any]],
        dedicated_faces: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prefer dedicated face boxes while preserving unrelated detector output."""
        merged = [
            dict(item)
            for item in objects
            if isinstance(item, dict)
            and str(item.get("label") or "").strip().lower() != "face"
        ]
        candidates = [
            dict(item)
            for item in dedicated_faces
            if isinstance(item, dict) and _box(item) is not None
        ]
        valid_dedicated: list[dict[str, Any]] = []
        for item in sorted(candidates, key=_confidence, reverse=True):
            item_box = _box(item)
            if item_box is None:
                continue
            if any(
                RecordedMotionObjectDetector._box_iou(item_box, _box(existing)) >= 0.4
                for existing in valid_dedicated
            ):
                continue
            valid_dedicated.append(item)
        merged.extend(valid_dedicated)
        for item in objects:
            if (
                not isinstance(item, dict)
                or str(item.get("label") or "").strip().lower() != "face"
            ):
                continue
            generic_box = _box(item)
            if generic_box is None:
                continue
            overlaps = [
                RecordedMotionObjectDetector._box_iou(generic_box, _box(face))
                for face in valid_dedicated
                if _box(face) is not None
            ]
            if max(overlaps, default=0.0) < 0.25:
                merged.append({**item, "detection_source": "object_detector"})
        return merged

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float] | None,
    ) -> float:
        if second is None:
            return 0.0
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
        return intersection / union if union > 0 else 0.0

    def _acquire_decode_process(
        self,
        *,
        incident_epoch: float | None,
        deadline: float | None,
        budget_wait_sink: dict[str, float] | None,
    ):
        budget = self.decode_budget
        if budget is None:
            return None
        wait_started = time.monotonic()
        report_evidence_stage("decode_process_wait")
        lease = budget.acquire_process(
            incident_epoch=(
                float(incident_epoch)
                if incident_epoch is not None
                else time.time()
            ),
            deadline=deadline,
            cancelled=self.stop_requested,
        )
        if budget_wait_sink is not None:
            budget_wait_sink["decode_budget_wait_ms"] = (
                float(budget_wait_sink.get("decode_budget_wait_ms") or 0.0)
                + (time.monotonic() - wait_started) * 1000.0
            )
        return lease

    def _read_recorded_frame(
        self,
        path: Path,
        offset_seconds: float,
        *,
        deadline: float | None = None,
        incident_epoch: float | None = None,
        budget_wait_sink: dict[str, float] | None = None,
    ) -> Frame | None:
        if not path.exists():
            return None
        attempts = [0.0, -0.25, 0.25, -0.75, 0.75]
        last_error = ""
        decode_plans = recorded_decode_plans(self.recorder.hardware_acceleration)
        for nudge in attempts:
            sample_at = max(0.0, offset_seconds + nudge)
            for plan in decode_plans:
                backend = plan.backend
                timeout = RECORDED_DECODE_TIMEOUT_SECONDS
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        last_error = "recorded-frame deadline expired"
                        break
                    timeout = min(timeout, remaining)
                command = build_decode_command(
                    self.recorder.ffmpeg_path,
                    path,
                    plan=plan,
                    tolerate_corruption=True,
                    seek=f"{sample_at:.3f}",
                    copyts=True,
                    start_at_zero=True,
                    first_video_stream=True,
                    video_filter=plan.download_filter,
                    frames=1,
                )
                process_lease = self._acquire_decode_process(
                    incident_epoch=incident_epoch,
                    deadline=deadline,
                    budget_wait_sink=budget_wait_sink,
                )
                if self.decode_budget is not None and process_lease is None:
                    last_error = "decode process budget unavailable"
                    break
                try:
                    result = run_decoder(command, backend=backend, timeout=timeout)
                finally:
                    if process_lease is not None:
                        process_lease.release()
                self._record_decoder_errors(
                    stderr=result.stderr,
                    path=path,
                    backend=backend,
                )
                if not result.ok or not result.output:
                    if self.decode_budget is not None:
                        self.decode_budget.record_ffmpeg(backend, success=False)
                    last_error = result.describe()
                    continue
                array = np.frombuffer(result.output, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None:
                    if self.decode_budget is not None:
                        self.decode_budget.record_ffmpeg(backend, success=True)
                    return frame
                if self.decode_budget is not None:
                    self.decode_budget.record_ffmpeg(backend, success=False)
                last_error = f"{backend}: lossless frame decode returned no frame"
            if deadline is not None and time.monotonic() >= deadline:
                break
        LOGGER.debug(
            "skipped unreadable recording sample for %s at %.2fs: %s%s",
            self.camera.id,
            offset_seconds,
            path,
            f" ({last_error})" if last_error else "",
        )
        return None

    def _read_recorded_frames(
        self,
        path: Path,
        offsets_seconds: list[float],
        *,
        deadline: float | None = None,
        incident_epoch: float | None = None,
        budget_wait_sink: dict[str, float] | None = None,
    ) -> tuple[dict[float, _DecodedRecordedFrame], int]:
        """Decode several ordered samples from one segment in one process.

        The select expression emits exactly the first decoded frame at or
        after each requested segment-relative timestamp. Missing outputs are
        intentionally left to the established nudged single-frame fallback.
        """
        if not path.exists():
            return {}, 0
        offsets = sorted(dict.fromkeys(round(max(0.0, value), 3) for value in offsets_seconds))
        if not offsets:
            return {}, 0

        # Skip the segment prefix while preserving the timestamp/image pairing
        # through QSV's B-frame decoder. -start_at_zero removes the container's
        # initial offset, keeping select and showinfo on the segment timeline.
        seek_seconds = max(0, math.floor(offsets[0] - RECORDED_BATCH_SEEK_MARGIN_SECONDS))
        expression = "0"
        for index, offset in reversed(list(enumerate(offsets))):
            expression = (
                f"if(eq(selected_n\\,{index})\\,gte(t\\,{offset:.3f})\\,{expression})"
            )
        last_error = ""
        process_count = 0
        for plan in recorded_decode_plans(self.recorder.hardware_acceleration):
            backend = plan.backend
            timeout = RECORDED_DECODE_TIMEOUT_SECONDS
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            # Download only the selected surfaces from the hardware decoder.
            download_filter = f"{plan.download_filter}," if plan.download_filter else ""
            command = build_decode_command(
                self.recorder.ffmpeg_path,
                path,
                plan=plan,
                loglevel="info",
                tolerate_corruption=True,
                seek=str(seek_seconds) if seek_seconds else None,
                copyts=True,
                start_at_zero=True,
                first_video_stream=True,
                video_filter=f"select={expression},{download_filter}showinfo@event_sample",
                passthrough_timestamps=True,
                frames=len(offsets),
            )
            process_lease = self._acquire_decode_process(
                incident_epoch=incident_epoch,
                deadline=deadline,
                budget_wait_sink=budget_wait_sink,
            )
            if self.decode_budget is not None and process_lease is None:
                last_error = "decode process budget unavailable"
                break
            try:
                process_count += 1
                result = run_decoder(
                    command,
                    backend=backend,
                    timeout=timeout,
                    stdout_reader=read_bmp_frames,
                )
            finally:
                if process_lease is not None:
                    process_lease.release()
            stderr = result.stderr
            decoded_frames: list[Frame] = result.output or []
            self._record_decoder_errors(stderr=stderr, path=path, backend=backend)
            if not result.ok or not decoded_frames:
                if self.decode_budget is not None:
                    self.decode_budget.record_ffmpeg(backend, success=False)
                last_error = result.describe()
                continue
            actual_offsets = [
                round(float(match.group(1)), 6)
                for match in re.finditer(
                    rb"showinfo@event_sample[^\r\n]*\bpts_time:([-+0-9.eE]+)",
                    stderr,
                )
            ]
            if len(actual_offsets) < len(decoded_frames):
                # Frames pair with timestamps by position. With a timestamp
                # line missing, nothing says which frame lost it, so none of
                # this batch can be trusted; the single-frame path re-reads.
                if self.decode_budget is not None:
                    self.decode_budget.record_ffmpeg(backend, success=False)
                last_error = (
                    f"{backend}: {len(decoded_frames)} frames but only "
                    f"{len(actual_offsets)} timestamps"
                )
                continue
            decoded: dict[float, _DecodedRecordedFrame] = {}
            for offset, frame, actual_offset in zip(
                offsets,
                decoded_frames,
                actual_offsets,
                strict=False,
            ):
                decoded[offset] = _DecodedRecordedFrame(
                    frame=frame,
                    actual_offset=actual_offset,
                    exact_timestamp=True,
                )
            if decoded:
                if self.decode_budget is not None:
                    self.decode_budget.record_ffmpeg(backend, success=True)
                return decoded, process_count
            if self.decode_budget is not None:
                self.decode_budget.record_ffmpeg(backend, success=False)
            last_error = f"{backend}: batch frame decode returned no frames"
        LOGGER.debug(
            "recorded batch decode failed for %s camera=%s samples=%d%s",
            path,
            self.camera.id,
            len(offsets),
            f" ({last_error})" if last_error else "",
        )
        return {}, process_count

    def _record_decoder_errors(self, *, stderr: bytes, path: Path, backend: str) -> None:
        """Summarize controlled recorded-decoder stderr with camera context."""
        codec_lines = decoder_error_codecs(stderr)
        if not codec_lines:
            return
        source = next((part for part in path.parts if part in {"main", "live"}), "")
        counts = " ".join(
            f"{codec}={lines}" for codec, lines in sorted(codec_lines.items())
        )
        LOGGER.warning(
            "recorded decoder errors camera=%s source=%s recording=%s backend=%s %s",
            self.camera.id,
            source or "unknown",
            path.name,
            backend,
            counts,
        )
        if self.decode_budget is not None:
            self.decode_budget.record_decoder_errors(
                camera_id=self.camera.id,
                source=source,
                recording_name=path.name,
                backend=backend,
                codec_lines=dict(codec_lines),
            )



class RecordedMotionObjectDetectorFactory:
    def __init__(
        self,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
        decode_budget: RecordedDecodeBudget | None = None,
    ) -> None:
        self.detector = detector
        self.recorder = recorder
        self.decode_budget = decode_budget or RecordedDecodeBudget()

    def reconfigure_decode_budget(self, config: Any) -> None:
        self.decode_budget.reconfigure_from_detector_config(config)

    def create(
        self,
        camera: CameraConfig,
        live_frame_provider: LiveFrameProvider,
        timestamped_live_frame_provider: TimestampedLiveFrameProvider | None = None,
        timestamped_evidence_frame_provider: TimestampedEvidenceFrameProvider | None = None,
        stop_requested: StopRequested = lambda: False,
        activity_attributor: ObjectActivityAttributor | None = None,
    ) -> RecordedMotionObjectDetector:
        return RecordedMotionObjectDetector(
            camera=camera,
            detector=self.detector,
            recorder=self.recorder,
            live_frame_provider=live_frame_provider,
            timestamped_live_frame_provider=timestamped_live_frame_provider,
            timestamped_evidence_frame_provider=timestamped_evidence_frame_provider,
            stop_requested=stop_requested,
            decode_budget=self.decode_budget,
            activity_attributor=activity_attributor,
        )
