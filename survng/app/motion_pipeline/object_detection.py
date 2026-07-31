from __future__ import annotations

import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from ..config import CameraConfig
from ..ffmpeg_hw import recorded_frame_hw_args
from ..zones import apply_detection_zones, detection_threshold
from .context import Frame


LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_STAGES = (
    (-1.0, -0.5, 0.0, 0.5, 1.0),
    (4.0, 4.5),
    (8.0, 8.5),
    (12.0, 12.5),
)
RECORDED_EVENT_FRAME_OFFSETS = tuple(
    offset
    for stage in RECORDED_EVENT_FRAME_STAGES
    for offset in stage
)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 24.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0
TEMPORAL_ASSOCIATION_MIN_IOU = 0.05
TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO = 2.5


@dataclass(frozen=True)
class _RecordedDetectionSample:
    offset: float
    frame: Frame
    objects: list[dict[str, Any]]
    recording_path: str


@dataclass
class _TemporalDetectionEvidence:
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, sample_index: int, detected: dict[str, Any]) -> None:
        self.observations[sample_index] = detected

    @property
    def latest(self) -> dict[str, Any]:
        return self.observations[max(self.observations)]

    @property
    def label_votes(self) -> dict[str, int]:
        votes: dict[str, int] = {}
        for detected in self.observations.values():
            label = str(detected.get("label") or "").strip()
            if label:
                votes[label] = votes.get(label, 0) + 1
        return votes

    @property
    def winning_label(self) -> str:
        def label_score(label: str) -> tuple[int, float, float, str]:
            confidences = [
                _confidence(item)
                for item in self.observations.values()
                if str(item.get("label") or "").strip() == label
            ]
            return (
                len(confidences),
                float(median(confidences)),
                max(confidences, default=0.0),
                label,
            )

        return max(self.label_votes, key=label_score, default="")

    @property
    def winning_observations(self) -> list[dict[str, Any]]:
        label = self.winning_label
        return [
            item
            for item in self.observations.values()
            if str(item.get("label") or "").strip() == label
        ]

    @property
    def aggregate_confidence(self) -> float:
        return float(median(_confidence(item) for item in self.winning_observations))

    @property
    def peak_confidence(self) -> float:
        return max((_confidence(item) for item in self.winning_observations), default=0.0)


def _confidence(detected: dict[str, Any]) -> float:
    try:
        confidence = float(detected.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _box(detected: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = detected.get("box")
    if not isinstance(box, dict):
        return None
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _temporal_association_score(
    previous: dict[str, Any],
    detected: dict[str, Any],
) -> float | None:
    previous_box = _box(previous)
    detected_box = _box(detected)
    if previous_box is None or detected_box is None:
        return None
    px1, py1, px2, py2 = previous_box
    dx1, dy1, dx2, dy2 = detected_box
    intersection = max(0.0, min(px2, dx2) - max(px1, dx1)) * max(0.0, min(py2, dy2) - max(py1, dy1))
    union = (px2 - px1) * (py2 - py1) + (dx2 - dx1) * (dy2 - dy1) - intersection
    iou = intersection / union if union > 0 else 0.0
    if iou >= TEMPORAL_ASSOCIATION_MIN_IOU:
        return 2.0 + iou
    previous_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
    detected_center = ((dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0)
    center_distance = math.hypot(
        detected_center[0] - previous_center[0],
        detected_center[1] - previous_center[1],
    )
    scale = max(
        math.hypot(px2 - px1, py2 - py1),
        math.hypot(dx2 - dx1, dy2 - dy1),
        1.0,
    )
    distance_ratio = center_distance / scale
    if distance_ratio > TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO:
        return None
    return 1.0 - distance_ratio / TEMPORAL_ASSOCIATION_MAX_DISTANCE_RATIO


def _eligible_detection(detected: dict[str, Any]) -> bool:
    return bool(detected.get("label") and detected.get("incident_eligible") is not False)


def _candidate_detection(detected: dict[str, Any]) -> bool:
    """Return whether a detector result can corroborate a temporal object track.

    Zone admission is intentionally not required here. An object that enters an
    incident zone and then crosses its boundary is still the same object; the
    completed track only needs one admitted observation to become eligible.
    """
    return bool(detected.get("label") and _box(detected) is not None)


def _temporal_consensus(
    samples: list[_RecordedDetectionSample],
    minimum_confirmations: int,
    class_confirmations: dict[str, int] | None = None,
) -> tuple[_RecordedDetectionSample, list[dict[str, Any]]]:
    """Select one recorded frame and retain only repeatable object evidence."""
    evidence: list[_TemporalDetectionEvidence] = []
    assignments: dict[tuple[int, int], _TemporalDetectionEvidence] = {}
    for sample_index, sample in enumerate(samples):
        available = set(range(len(evidence)))
        candidates = [item for item in sample.objects if _candidate_detection(item)]
        for object_index, detected in sorted(
            enumerate(candidates),
            key=lambda item: _confidence(item[1]),
            reverse=True,
        ):
            matches = [
                (score, evidence_index)
                for evidence_index in available
                if (score := _temporal_association_score(evidence[evidence_index].latest, detected)) is not None
            ]
            if matches:
                _, evidence_index = max(matches)
                track = evidence[evidence_index]
                available.remove(evidence_index)
            else:
                track = _TemporalDetectionEvidence()
                evidence.append(track)
                evidence_index = len(evidence) - 1
            track.add(sample_index, detected)
            assignments[(sample_index, id(detected))] = track

    default_required = max(1, min(5, int(minimum_confirmations)))
    normalized_class_confirmations = {
        str(label).strip().lower(): max(1, min(5, int(confirmations)))
        for label, confirmations in (class_confirmations or {}).items()
    }
    required_by_track = {
        id(track): normalized_class_confirmations.get(track.winning_label.lower(), default_required)
        for track in evidence
    }
    confirmed_ids = {
        id(track)
        for track in evidence
        if len(track.winning_observations) >= required_by_track[id(track)]
        and any(_eligible_detection(item) for item in track.winning_observations)
    }

    def sample_score(item: tuple[int, _RecordedDetectionSample]) -> tuple[int, int, float, float, float]:
        sample_index, sample = item
        visible = [
            track
            for track in evidence
            if id(track) in confirmed_ids
            and sample_index in track.observations
            and str(track.observations[sample_index].get("label") or "").strip() == track.winning_label
        ]
        admitted_visible = [
            track for track in visible
            if _eligible_detection(track.observations[sample_index])
        ]
        raw_peak = max((_confidence(detected) for detected in sample.objects if _candidate_detection(detected)), default=0.0)
        return (
            len(admitted_visible),
            len(visible),
            sum(track.aggregate_confidence for track in visible),
            raw_peak,
            -abs(sample.offset),
        )

    selected_index, selected = max(enumerate(samples), key=sample_score)
    annotated: list[dict[str, Any]] = []
    for detected in selected.objects:
        if not _candidate_detection(detected):
            annotated.append(dict(detected))
            continue
        track = assignments.get((selected_index, id(detected)))
        if track is None:
            continue
        confirmed = id(track) in confirmed_ids
        label_confirmed_here = str(detected.get("label") or "").strip() == track.winning_label
        confirmed = confirmed and label_confirmed_here
        enriched = {
            **detected,
            "incident_eligible": confirmed,
            "temporal_consensus": confirmed,
            "temporal_sample_offset_seconds": selected.offset,
            "temporal_observations": len(track.winning_observations),
            "temporal_track_observations": len(track.observations),
            "temporal_incident_observations": sum(
                1 for item in track.winning_observations if _eligible_detection(item)
            ),
            "temporal_required_observations": required_by_track[id(track)],
            "temporal_samples": len(samples),
            "temporal_peak_confidence": round(track.peak_confidence, 4),
            "temporal_label_votes": track.label_votes,
        }
        if confirmed:
            enriched["confidence"] = round(track.aggregate_confidence, 4)
        annotated.append(enriched)

    LOGGER.debug(
        "recorded object consensus retained %d/%d candidates across %d frames (minimum %d)",
        len(confirmed_ids),
        len(evidence),
        len(samples),
        default_required,
    )
    return selected, annotated


class MotionObjectDetectorBackend(Protocol):
    config: Any

    def detect(
        self,
        frame: Frame,
        confidence_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        ...


class MotionRecordingProvider(Protocol):
    ffmpeg_path: str
    hardware_acceleration: str

    def recording_at(self, camera_id: str, epoch: float) -> dict[str, Any] | None:
        ...


LiveFrameProvider = Callable[[], Frame | None]


class RecordedMotionObjectDetector:
    """Combines recorded event-time detections into repeatable object evidence."""

    def __init__(
        self,
        camera: CameraConfig,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
        live_frame_provider: LiveFrameProvider,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.recorder = recorder
        self.live_frame_provider = live_frame_provider

    def detect(self, event_at: datetime) -> tuple[Frame | None, list[dict[str, Any]], str]:
        event_epoch = event_at.timestamp()
        initial_offsets = RECORDED_EVENT_FRAME_STAGES[0]
        newest_needed = event_epoch + max(initial_offsets) + RECORDED_EVENT_SETTLE_SECONDS
        wait_seconds = max(0.0, newest_needed - time.time())
        if wait_seconds > 0:
            time.sleep(min(wait_seconds, 3.0))

        deadline = time.monotonic() + max(0.0, RECORDED_EVENT_RETRY_SECONDS)
        samples_by_offset: dict[float, _RecordedDetectionSample] = {}
        default_required = max(
            1,
            min(
                len(RECORDED_EVENT_FRAME_OFFSETS),
                int(getattr(self.detector.config, "event_confirmation_frames", 2)),
            ),
        )
        class_confirmations = {
            str(label).strip().lower(): max(
                1,
                min(len(RECORDED_EVENT_FRAME_OFFSETS), int(confirmations)),
            )
            for label, confirmations in dict(
                getattr(self.detector.config, "event_class_confirmation_frames", {}) or {}
            ).items()
        }

        for stage_index, stage_offsets in enumerate(RECORDED_EVENT_FRAME_STAGES):
            while True:
                for sample_offset in stage_offsets:
                    if sample_offset in samples_by_offset:
                        continue
                    if time.monotonic() >= deadline:
                        break
                    target_epoch = event_epoch + sample_offset
                    if target_epoch + RECORDED_EVENT_SETTLE_SECONDS > time.time():
                        continue
                    row = self.recorder.recording_at(self.camera.id, target_epoch)
                    if row is None:
                        continue
                    start_epoch = row.get("start_epoch")
                    if start_epoch is None:
                        continue
                    frame_offset = max(0.0, target_epoch - float(start_epoch))
                    frame = self._read_recorded_frame(
                        Path(str(row["path"])),
                        frame_offset,
                        deadline=deadline,
                    )
                    if frame is None:
                        continue
                    objects = self._detect_objects(frame)
                    samples_by_offset[sample_offset] = _RecordedDetectionSample(
                        offset=sample_offset,
                        frame=frame,
                        objects=objects,
                        recording_path=str(row["path"]),
                    )

                samples = [
                    samples_by_offset[offset]
                    for offset in RECORDED_EVENT_FRAME_OFFSETS
                    if offset in samples_by_offset
                ]
                if samples:
                    selected, objects = _temporal_consensus(
                        samples,
                        default_required,
                        class_confirmations,
                    )
                    if any(item.get("temporal_consensus") is True for item in objects):
                        if stage_index:
                            LOGGER.info(
                                "late recorded object discovery confirmed for %s at stage +%.1fs",
                                self.camera.id,
                                max(stage_offsets),
                            )
                        return selected.frame, objects, selected.recording_path
                if all(offset in samples_by_offset for offset in stage_offsets):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(RECORDED_EVENT_RETRY_INTERVAL_SECONDS, remaining))
            if time.monotonic() >= deadline:
                break

        if samples:
            selected, objects = _temporal_consensus(
                samples,
                default_required,
                class_confirmations,
            )
            return selected.frame, objects, selected.recording_path

        fallback = self.live_frame_provider()
        if fallback is None:
            return None, [{"status": "no_recorded_frame"}], ""
        objects = self._detect_objects(fallback)
        if objects:
            for detected in objects:
                detected["frame_source"] = "live_fallback"
                detected["recording_status"] = "no_recorded_frame"
            return fallback, objects, ""
        return fallback, [{"status": "no_recorded_frame", "frame_source": "live_fallback"}], ""

    def _detect_objects(self, frame: Frame) -> list[dict[str, Any]]:
        configured_threshold = float(self.detector.config.confidence_threshold)
        threshold = detection_threshold(self.camera, configured_threshold)
        objects = self.detector.detect(frame, confidence_threshold=threshold)
        apply_detection_zones(
            self.camera,
            objects,
            int(frame.shape[1]),
            int(frame.shape[0]),
            configured_threshold,
            bool(getattr(self.detector.config, "require_incident_zone", True)),
        )
        return objects

    def _read_recorded_frame(
        self,
        path: Path,
        offset_seconds: float,
        *,
        deadline: float | None = None,
    ) -> Frame | None:
        if not path.exists():
            return None
        attempts = [0.0, -0.25, 0.25, -0.75, 0.75]
        last_error = ""
        hw_input_args, hw_filter_args = recorded_frame_hw_args(self.recorder.hardware_acceleration)
        decode_plans = [("hardware", hw_input_args, hw_filter_args)] if hw_input_args else []
        decode_plans.append(("cpu", [], []))
        for nudge in attempts:
            sample_at = max(0.0, offset_seconds + nudge)
            for backend, input_args, filter_args in decode_plans:
                timeout = 8.0
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        last_error = "recorded-frame deadline expired"
                        break
                    timeout = min(timeout, remaining)
                command = [
                    self.recorder.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-fflags",
                    "+discardcorrupt",
                    "-err_detect",
                    "ignore_err",
                    *input_args,
                    "-ss",
                    f"{sample_at:.3f}",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-frames:v",
                    "1",
                    *filter_args,
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ]
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    last_error = f"{backend} timed out"
                    continue
                if result.returncode != 0 or not result.stdout:
                    detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[0:2]
                    last_error = f"{backend}: {' '.join(detail)[:180]}"
                    continue
                array = np.frombuffer(result.stdout, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                last_error = f"{backend}: mjpeg decode returned no frame"
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

class RecordedMotionObjectDetectorFactory:
    def __init__(
        self,
        detector: MotionObjectDetectorBackend,
        recorder: MotionRecordingProvider,
    ) -> None:
        self.detector = detector
        self.recorder = recorder

    def create(
        self,
        camera: CameraConfig,
        live_frame_provider: LiveFrameProvider,
    ) -> RecordedMotionObjectDetector:
        return RecordedMotionObjectDetector(
            camera=camera,
            detector=self.detector,
            recorder=self.recorder,
            live_frame_provider=live_frame_provider,
        )
