"""Camera media and durable motion-evidence operations."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import cv2

from .config import CameraConfig
from .image_storage import DurableImageWriter
from .motion import MotionQualificationResult

LOGGER = logging.getLogger(__name__)
REJECTED_SAMPLE_LIMIT = 100


class RecordedMotionDetector(Protocol):
    """Detect objects using the best recorded or live frame for an event."""

    def detect(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]: ...

    def detect_initial(self, event_at: datetime) -> Any: ...


class CameraMediaService:
    """Own camera image encoding, evidence persistence, and frame detection."""

    def __init__(
        self,
        *,
        camera: CameraConfig,
        storage_dir: Path,
        image_writer: DurableImageWriter,
        motion_detector: RecordedMotionDetector,
        frame_provider: Callable[[str], Any | None],
        rejected_sample_rate: Callable[[], float],
        stop_requested: Callable[[], bool],
        random_value: Callable[[], float] = random.random,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        time_ns: Callable[[], int] = time.time_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.camera = camera
        self.storage_dir = storage_dir
        self.image_writer = image_writer
        self.motion_detector = motion_detector
        self.frame_provider = frame_provider
        self.rejected_sample_rate = rejected_sample_rate
        self.stop_requested = stop_requested
        self.random_value = random_value
        self.utc_now = utc_now
        self.time_ns = time_ns
        self.sleeper = sleeper
        self.snapshots_dir = storage_dir / "snapshots" / camera.id
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self, source: str = "live") -> bytes | None:
        frame = self.frame_provider(source)
        if frame is None:
            return None
        try:
            encoded, buffer = cv2.imencode(".jpg", frame)
        except (cv2.error, TypeError, ValueError, AttributeError):
            return None
        return buffer.tobytes() if encoded and buffer is not None else None

    def mjpeg_frames(self, fps: float = 4.0, source: str = "live") -> Iterator[bytes]:
        normalized = self.camera.normalized_source(source)
        delay = 1.0 / max(fps, 1.0)
        while not self.stop_requested():
            image = self.snapshot(normalized)
            if image is None:
                self.sleeper(delay)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + image
                + b"\r\n"
            )
            self.sleeper(delay)

    def sample_rejected_motion(
        self,
        event_at: datetime,
        result: MotionQualificationResult,
    ) -> str:
        sample_rate = self.rejected_sample_rate()
        if sample_rate <= 0 or self.random_value() > sample_rate:
            return ""
        frame = self.frame_provider("live")
        if frame is None:
            return ""
        directory = self.storage_dir / "motion_samples" / self.camera.id
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = event_at.strftime("%Y%m%d-%H%M%S-%f")
            path = self.image_writer.write(
                directory,
                f"{stamp}-{result.score:.3f}-{result.reason}",
                frame,
            )
            if path is None:
                LOGGER.warning(
                    "failed to encode rejected motion sample for %s",
                    self.camera.id,
                )
                return ""
            self._prune_rejected_samples(directory)
            return str(path)
        except OSError as error:
            LOGGER.debug(
                "failed to save rejected motion sample for %s: %s",
                self.camera.id,
                error,
            )
            return ""

    def detect_recorded_motion(
        self,
        event_at: datetime,
    ) -> tuple[Any | None, list[dict[str, Any]], str]:
        return self.motion_detector.detect(event_at)

    def detect_initial_recorded_motion(self, event_at: datetime) -> Any:
        return self.motion_detector.detect_initial(event_at)

    @staticmethod
    def read_image(path: str) -> Any | None:
        return cv2.imread(path)

    def write_snapshot(self, frame: Any, event_at: datetime | None = None) -> str:
        captured_at = self._utc(event_at or self.utc_now())
        event_stamp = captured_at.strftime("%Y%m%d-%H%M%S-%f")
        stamp = f"{event_stamp}-{self.time_ns() % 1_000_000_000:09d}"
        path = self.image_writer.write(self.snapshots_dir, stamp, frame)
        if path is None:
            LOGGER.warning("failed to encode snapshot for %s", self.camera.id)
            return ""
        return str(path)

    def _prune_rejected_samples(self, directory: Path) -> None:
        try:
            samples: list[tuple[int, Path]] = []
            for item in self.image_writer.stored_images(directory):
                try:
                    samples.append((item.stat().st_mtime_ns, item))
                except FileNotFoundError:
                    continue
            for _modified, stale in sorted(samples)[:-REJECTED_SAMPLE_LIMIT]:
                try:
                    stale.unlink(missing_ok=True)
                except OSError as error:
                    LOGGER.debug(
                        "failed to prune rejected motion sample %s: %s",
                        stale,
                        error,
                    )
        except OSError as error:
            LOGGER.debug(
                "failed to enumerate rejected motion samples for %s: %s",
                self.camera.id,
                error,
            )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
