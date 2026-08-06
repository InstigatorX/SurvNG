"""Live and recorded frame acquisition for object tracking sessions."""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import cv2
import numpy as np

from .camera_capture import CameraCaptureService
from .config import CameraConfig
from .security import redact_secret_text
from .tracking_comparison import sampled_video_frames

LOGGER = logging.getLogger(__name__)
TRACKING_CATCHUP_SECONDS = 10.0
TRACKING_CATCHUP_FRAME_WIDTH = 640


class TrackingRecorder(Protocol):
    ffmpeg_path: str

    def recording_rows_between(
        self,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        *,
        source: str,
    ) -> list[dict[str, Any]]: ...


class TrackingFrameService:
    """Own timestamped live fallback and bounded recorded-frame catch-up."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        capture: CameraCaptureService,
        recorder: TrackingRecorder,
        stop_event: threading.Event,
        sample_fps: Callable[[], float],
    ) -> None:
        self.camera = camera
        self.capture = capture
        self.recorder = recorder
        self.stop_event = stop_event
        self.sample_fps = sample_fps
        self._lock = threading.Lock()
        self.frames: deque[tuple[float, np.ndarray]] = deque(
            maxlen=self.buffer_size(sample_fps())
        )
        self._last_sample_epoch = 0.0

    @staticmethod
    def buffer_size(sample_fps: float) -> int:
        return max(4, round(sample_fps * TRACKING_CATCHUP_SECONDS) + 2)

    def clear(self) -> None:
        with self._lock:
            self.frames.clear()
            self._last_sample_epoch = 0.0

    def resize(self, sample_fps: float) -> None:
        with self._lock:
            self.frames = deque(
                self.frames,
                maxlen=self.buffer_size(sample_fps),
            )
            self._last_sample_epoch = 0.0

    def latest(
        self,
        source: str = "main",
    ) -> tuple[np.ndarray, float, float] | None:
        normalized = self.camera.normalized_source(source)
        if self.stop_event.is_set():
            return None
        frame = self.capture.request_frame(normalized)
        if frame is None:
            return None
        return frame.image, frame.captured_at_epoch, frame.captured_at_monotonic

    def latest_with_fallback(self) -> tuple[np.ndarray, float, float] | None:
        """Prefer main detail while using the continuously warm live stream."""
        return self.latest("main") or self.latest("live")

    def remember(self, frame: np.ndarray, captured_at: float) -> None:
        interval = 1.0 / max(0.1, float(self.sample_fps()))
        with self._lock:
            if captured_at - self._last_sample_epoch < interval * 0.9:
                return
            self._last_sample_epoch = captured_at
        height, width = frame.shape[:2]
        if width > TRACKING_CATCHUP_FRAME_WIDTH:
            scale = TRACKING_CATCHUP_FRAME_WIDTH / width
            stored = cv2.resize(
                frame,
                (TRACKING_CATCHUP_FRAME_WIDTH, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            stored = frame.copy()
        with self._lock:
            self.frames.append((captured_at, stored))

    def recorded_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        if end_epoch <= start_epoch or frame_width <= 0:
            return
        samples: list[tuple[float, np.ndarray]] = []
        rows = self.recorder.recording_rows_between(
            self.camera.id,
            start_epoch,
            end_epoch,
            source="main",
        )
        interval = 1.0 / max(0.1, float(sample_fps))
        last_epoch = start_epoch - interval
        for row in rows:
            row_start = float(row.get("start_epoch") or 0.0)
            row_end = float(row.get("end_epoch") or row_start)
            sample_start = max(start_epoch, row_start)
            sample_end = min(end_epoch, row_end)
            duration = sample_end - sample_start
            if duration <= 0.0:
                continue
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                continue
            try:
                for captured_at, frame in sampled_video_frames(
                    path,
                    start_epoch=sample_start,
                    sample_fps=sample_fps,
                    duration_seconds=duration,
                    ffmpeg_path=self.recorder.ffmpeg_path,
                    maximum_width=frame_width,
                    start_offset_seconds=max(0.0, sample_start - row_start),
                    probe_path=path,
                ):
                    if captured_at <= last_epoch + interval * 0.5:
                        continue
                    if captured_at > end_epoch + 1e-6:
                        break
                    last_epoch = captured_at
                    samples.append((captured_at, frame))
            except RuntimeError as error:
                LOGGER.warning(
                    "recorded tracking catch-up skipped %s/%s: %s",
                    self.camera.id,
                    path.name,
                    redact_secret_text(error),
                )
        with self._lock:
            samples.extend(
                (captured_at, frame)
                for captured_at, frame in self.frames
                if start_epoch <= captured_at <= end_epoch
            )
        last_epoch = start_epoch - interval
        for captured_at, frame in sorted(samples, key=lambda sample: sample[0]):
            if captured_at <= last_epoch + interval * 0.5:
                continue
            last_epoch = captured_at
            yield captured_at, frame
