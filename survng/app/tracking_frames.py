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

from .camera_capture import CameraCaptureService, CapturedFrame
from .config import CameraConfig
from .security import redact_secret_text
from .tracking_comparison import sampled_video_frames, video_frame_at_reference
from .video_frames import DecodedVideoFrame, VideoFrameReference

LOGGER = logging.getLogger(__name__)
TRACKING_CATCHUP_SECONDS = 10.0
TRACKING_CATCHUP_FRAME_WIDTH = 640
# Bound for bridging an unfinalized main-segment tail from live history.
TRACKING_OPEN_SEGMENT_BRIDGE_SECONDS = 12.0


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


class CameraFrameTimeline:
    """Timestamped live/main history merged with finalized recordings.

    Prefer finalized main recordings for catch-up. When the next segment is
    still open (absent from the index), timestamp-ordered in-memory history
    bridges the unavailable tail: main samples when warm, otherwise continuous
    live samples (bounded by TRACKING_OPEN_SEGMENT_BRIDGE_SECONDS).
    """

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
        self._main_generation = 0
        self._live_generation = 0
        size = self.buffer_size(sample_fps())
        # Main-stream history (existing callers/tests use ``frames``).
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=size)
        self.live_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=size)
        self._last_main_sample_epoch = 0.0
        self._last_live_sample_epoch = 0.0

    @staticmethod
    def buffer_size(sample_fps: float) -> int:
        return max(
            4,
            round(
                sample_fps
                * max(
                    TRACKING_CATCHUP_SECONDS,
                    TRACKING_OPEN_SEGMENT_BRIDGE_SECONDS,
                )
            )
            + 2,
        )

    def clear(self, source: str | None = None) -> None:
        """Clear retained history for ``main``, ``live``, or both when omitted."""
        with self._lock:
            if source in (None, "main"):
                self._main_generation += 1
                self.frames.clear()
                self._last_main_sample_epoch = 0.0
            if source in (None, "live"):
                self._live_generation += 1
                self.live_frames.clear()
                self._last_live_sample_epoch = 0.0

    def resize(self, sample_fps: float) -> None:
        size = self.buffer_size(sample_fps)
        with self._lock:
            self._main_generation += 1
            self._live_generation += 1
            self.frames = deque(self.frames, maxlen=size)
            self.live_frames = deque(self.live_frames, maxlen=size)
            self._last_main_sample_epoch = 0.0
            self._last_live_sample_epoch = 0.0

    def latest(
        self,
        source: str = "main",
    ) -> tuple[np.ndarray, float, float] | None:
        frame = self.captured(source)
        if frame is None:
            return None
        return frame.image, frame.captured_at_epoch, frame.captured_at_monotonic

    def captured(self, source: str = "main") -> CapturedFrame | None:
        """Return the latest frame with complete capture-generation identity."""
        normalized = self.camera.normalized_source(source)
        if self.stop_event.is_set():
            return None
        return self.capture.request_frame(normalized)

    def latest_with_fallback(self) -> tuple[np.ndarray, float, float] | None:
        """Prefer main detail while using the continuously warm live stream."""
        return self.latest("main") or self.latest("live")

    def remember(
        self,
        frame: np.ndarray,
        captured_at: float,
        *,
        source: str = "main",
    ) -> None:
        """Retain a timestamped sample for open-segment catch-up bridging.

        ``source="main"`` stores detail frames when main capture is warm.
        ``source="live"`` stores the continuous live stream so an unfinalized
        main-segment tail can still be walked in timestamp order without waiting
        for segment finalization.
        """
        normalized = "live" if source == "live" else "main"
        interval = 1.0 / max(0.1, float(self.sample_fps()))
        with self._lock:
            if normalized == "live":
                if captured_at - self._last_live_sample_epoch < interval * 0.9:
                    return
                self._last_live_sample_epoch = captured_at
                generation = self._live_generation
            else:
                if captured_at - self._last_main_sample_epoch < interval * 0.9:
                    return
                self._last_main_sample_epoch = captured_at
                generation = self._main_generation
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
            if normalized == "live":
                if (
                    generation != self._live_generation
                    or captured_at != self._last_live_sample_epoch
                ):
                    return
                self.live_frames.append((captured_at, stored))
            else:
                if (
                    generation != self._main_generation
                    or captured_at != self._last_main_sample_epoch
                ):
                    return
                self.frames.append((captured_at, stored))

    def recorded_frames(
        self,
        start_epoch: float,
        end_epoch: float,
        sample_fps: float,
        frame_width: int,
    ) -> Iterator[tuple[float, np.ndarray] | DecodedVideoFrame]:
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
        live_bridge_start = max(
            start_epoch,
            end_epoch - TRACKING_OPEN_SEGMENT_BRIDGE_SECONDS,
        )
        with self._lock:
            main_buffered = tuple(
                sorted(
                    (
                        (captured_at, frame)
                        for captured_at, frame in self.frames
                        if start_epoch <= captured_at <= end_epoch
                    ),
                    key=lambda sample: sample[0],
                )
            )
            live_buffered = tuple(
                sorted(
                    (
                        (captured_at, frame)
                        for captured_at, frame in self.live_frames
                        if live_bridge_start <= captured_at <= end_epoch
                    ),
                    key=lambda sample: sample[0],
                )
            )

        def recorded_samples() -> Iterator[tuple[float, np.ndarray] | DecodedVideoFrame]:
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
                    for sample in sampled_video_frames(
                        path,
                        start_epoch=sample_start,
                        sample_fps=sample_fps,
                        duration_seconds=duration,
                        ffmpeg_path=self.recorder.ffmpeg_path,
                        maximum_width=frame_width,
                        start_offset_seconds=max(0.0, sample_start - row_start),
                        probe_path=path,
                    ):
                        captured_at, _frame = sample
                        if self.stop_event.is_set():
                            return
                        if captured_at <= last_recorded_epoch + interval * 0.5:
                            continue
                        if captured_at > end_epoch + 1e-6:
                            break
                        last_recorded_epoch = captured_at
                        yield sample
                except RuntimeError as error:
                    LOGGER.warning(
                        "recorded tracking catch-up skipped %s/%s: %s",
                        self.camera.id,
                        path.name,
                        redact_secret_text(error),
                    )

        last_epoch = start_epoch - interval
        # Preference on near-ties: finalized recordings, then main history,
        # then live history for the open-segment tail only.
        for sample in merge(
            recorded_samples(),
            main_buffered,
            live_buffered,
            key=lambda sample: sample[0],
        ):
            captured_at, _frame = sample
            if self.stop_event.is_set():
                return
            if captured_at <= last_epoch + interval * 0.5:
                continue
            if captured_at > end_epoch + 1e-6:
                break
            last_epoch = captured_at
            yield sample

    def recorded_frame_at(
        self,
        captured_at: float,
        frame_width: int,
        reference: VideoFrameReference | None = None,
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
        if reference is not None and reference.exact:
            for row in rows:
                path = Path(str(row.get("path") or ""))
                if (
                    path.is_file()
                    and reference.source_path.resolve() == path.resolve()
                ):
                    exact = video_frame_at_reference(
                        reference,
                        ffmpeg_path=self.recorder.ffmpeg_path,
                        maximum_width=frame_width,
                    )
                    return exact.frame if exact is not None else None
            return None
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
