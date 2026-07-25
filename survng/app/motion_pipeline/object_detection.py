from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from ..config import CameraConfig
from ..ffmpeg_hw import recorded_frame_hw_args
from ..zones import apply_detection_zones, detection_threshold
from .context import Frame


LOGGER = logging.getLogger(__name__)
RECORDED_EVENT_FRAME_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RECORDED_EVENT_SETTLE_SECONDS = 0.75
RECORDED_EVENT_RETRY_SECONDS = 12.0
RECORDED_EVENT_RETRY_INTERVAL_SECONDS = 1.0


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
    """Finds the strongest recorded frame and runs downstream object inference."""

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
        newest_needed = event_epoch + max(RECORDED_EVENT_FRAME_OFFSETS) + RECORDED_EVENT_SETTLE_SECONDS
        wait_seconds = max(0.0, newest_needed - time.time())
        if wait_seconds > 0:
            time.sleep(min(wait_seconds, 3.0))

        deadline = time.time() + RECORDED_EVENT_RETRY_SECONDS
        best_frame: Frame | None = None
        best_objects: list[dict[str, Any]] = []
        best_score = -1.0
        best_distance = float("inf")
        best_recording_path = ""

        while True:
            for sample_offset in RECORDED_EVENT_FRAME_OFFSETS:
                target_epoch = event_epoch + sample_offset
                row = self.recorder.recording_at(self.camera.id, target_epoch)
                if row is None:
                    continue
                start_epoch = row.get("start_epoch")
                if start_epoch is None:
                    continue
                frame_offset = max(0.0, target_epoch - float(start_epoch))
                frame = self._read_recorded_frame(Path(str(row["path"])), frame_offset)
                if frame is None:
                    continue
                objects = self._detect_objects(frame)
                score = self._object_score(objects)
                distance = abs(sample_offset)
                if score > best_score or (score == best_score and distance < best_distance):
                    best_frame = frame
                    best_objects = objects
                    best_score = score
                    best_distance = distance
                    best_recording_path = str(row["path"])

            if best_frame is not None or time.time() >= deadline:
                break
            time.sleep(RECORDED_EVENT_RETRY_INTERVAL_SECONDS)

        if best_frame is not None:
            return best_frame, best_objects, best_recording_path

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
        )
        return objects

    def _read_recorded_frame(self, path: Path, offset_seconds: float) -> Frame | None:
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
                        timeout=8,
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
        LOGGER.debug(
            "skipped unreadable recording sample for %s at %.2fs: %s%s",
            self.camera.id,
            offset_seconds,
            path,
            f" ({last_error})" if last_error else "",
        )
        return None

    @staticmethod
    def _object_score(objects: list[dict[str, Any]]) -> float:
        score = 0.0
        for detected in objects:
            label = detected.get("label")
            if not label or detected.get("incident_eligible") is False:
                continue
            confidence = detected.get("confidence")
            score = max(score, float(confidence) if isinstance(confidence, (float, int)) else 0.01)
        return score


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
