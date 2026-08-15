"""Live and recorded frame acquisition for object tracking sessions."""

from __future__ import annotations

import logging
import threading
from collections import deque
from heapq import merge
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
        self._generation = 0
        self.frames: deque[tuple[float, np.ndarray]] = deque(
            maxlen=self.buffer_size(sample_fps())
        )
        self._last_sample_epoch = 0.0

    @staticmethod
    def buffer_size(sample_fps: float) -> int:
        return max(4, round(sample_fps * TRACKING_CATCHUP_SECONDS) + 2)

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self.frames.clear()
            self._last_sample_epoch = 0.0

    def resize(self, sample_fps: float) -> None:
        with self._lock:
            self._generation += 1
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
            generation = self._generation
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
            if (
                generation != self._generation
                or captured_at != self._last_sample_epoch
            ):
                return
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
        rows = sorted(
            self.recorder.recording_rows_between(
                self.camera.id,
                start_epoch,
                end_epoch,
                source="main",
            ),
            key=lambda row: (
                float(row.get("start_epoch") or 0.0),
                float(row.get("end_epoch") or 0.0),
                str(row.get("path") or ""),
            ),
        )
        interval = 1.0 / max(0.1, float(sample_fps))
        with self._lock:
            buffered = tuple(
                sorted(
                    (
                        (captured_at, frame)
                        for captured_at, frame in self.frames
                        if start_epoch <= captured_at <= end_epoch
                    ),
                    key=lambda sample: sample[0],
                )
            )

        def recorded_samples() -> Iterator[tuple[float, np.ndarray]]:
            last_recorded_epoch = start_epoch - interval
            for row in rows:
                if self.stop_event.is_set():
                    return
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
                        if self.stop_event.is_set():
                            return
                        if captured_at <= last_recorded_epoch + interval * 0.5:
                            continue
                        if captured_at > end_epoch + 1e-6:
                            break
                        last_recorded_epoch = captured_at
                        yield captured_at, frame
                except RuntimeError as error:
                    LOGGER.warning(
                        "recorded tracking catch-up skipped %s/%s: %s",
                        self.camera.id,
                        path.name,
                        redact_secret_text(error),
                    )

        last_epoch = start_epoch - interval
        for captured_at, frame in merge(
            recorded_samples(),
            buffered,
            key=lambda sample: sample[0],
        ):
            if self.stop_event.is_set():
                return
            if captured_at <= last_epoch + interval * 0.5:
                continue
            if captured_at > end_epoch + 1e-6:
                break
            last_epoch = captured_at
            yield captured_at, frame

    def recorded_frame_at(
        self,
        captured_at: float,
        frame_width: int,
    ) -> np.ndarray | None:
        """Decode one full-detail frame nearest a nominated tracking timestamp.

        This deliberately bypasses the low-resolution in-memory history. Cover
        promotion happens after tracking completes, when the finalized main
        recording can provide a durable image at the camera's native aspect.
        """
        if self.stop_event.is_set() or captured_at <= 0.0 or frame_width <= 0:
            return None
        rows = sorted(
            self.recorder.recording_rows_between(
                self.camera.id,
                captured_at - 0.05,
                captured_at + 0.05,
                source="main",
            ),
            key=lambda row: (
                abs(float(row.get("start_epoch") or 0.0) - captured_at),
                str(row.get("path") or ""),
            ),
        )
        for row in rows:
            row_start = float(row.get("start_epoch") or 0.0)
            row_end = float(row.get("end_epoch") or row_start)
            if not (row_start <= captured_at <= row_end + 1e-6):
                continue
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                continue
            try:
                samples = iter(sampled_video_frames(
                    path,
                    start_epoch=captured_at,
                    sample_fps=1.0,
                    duration_seconds=0.1,
                    ffmpeg_path=self.recorder.ffmpeg_path,
                    maximum_width=frame_width,
                    start_offset_seconds=max(0.0, captured_at - row_start),
                    probe_path=path,
                ))
                try:
                    return next(samples, (None, None))[1]
                finally:
                    close_samples = getattr(samples, "close", None)
                    if callable(close_samples):
                        close_samples()
            except (OSError, RuntimeError) as error:
                LOGGER.warning(
                    "tracking cover frame decode skipped %s/%s: %s",
                    self.camera.id,
                    path.name,
                    redact_secret_text(error),
                )
        return None
