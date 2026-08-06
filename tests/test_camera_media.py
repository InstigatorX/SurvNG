from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.camera_media import CameraMediaService, REJECTED_SAMPLE_LIMIT
from survng.app.config import CameraConfig, ImageStorageConfig
from survng.app.image_storage import DurableImageWriter
from survng.app.motion import MotionQualificationResult


def _service(
    storage_dir: Path,
    *,
    frame: np.ndarray | None = None,
    sample_rate: float = 1.0,
    random_value: float = 0.0,
    stopped: list[bool] | None = None,
    detector: Mock | None = None,
) -> CameraMediaService:
    camera = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://example.invalid/main",
        live_stream_url="rtsp://example.invalid/sub",
    )
    return CameraMediaService(
        camera=camera,
        storage_dir=storage_dir,
        image_writer=DurableImageWriter(ImageStorageConfig()),
        motion_detector=detector or Mock(),
        frame_provider=lambda _source: None if frame is None else frame.copy(),
        rejected_sample_rate=lambda: sample_rate,
        stop_requested=lambda: bool(stopped and stopped[0]),
        random_value=lambda: random_value,
        utc_now=lambda: datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
        time_ns=lambda: 123456789,
        sleeper=lambda _delay: None,
    )


def test_snapshot_returns_decodable_jpeg_without_persisting_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        service = _service(
            Path(tmpdir),
            frame=np.full((24, 32, 3), 91, dtype=np.uint8),
        )

        encoded = service.snapshot("live")

        assert encoded is not None
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (24, 32, 3)
        assert list(service.snapshots_dir.iterdir()) == []


def test_mjpeg_normalizes_source_and_stops_without_an_extra_frame() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        stopped = [False]
        service = _service(
            Path(tmpdir),
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            stopped=stopped,
        )
        service.snapshot = Mock(return_value=b"jpeg")
        frames = service.mjpeg_frames(source="unsupported")

        assert next(frames).endswith(b"jpeg\r\n")
        service.snapshot.assert_called_once_with("live")
        stopped[0] = True
        assert list(frames) == []


def test_rejected_sample_rate_is_evaluated_before_frame_acquisition() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = Mock(return_value=np.zeros((8, 8, 3), dtype=np.uint8))
        service = _service(Path(tmpdir), frame=None, sample_rate=0.25, random_value=0.5)
        service.frame_provider = provider
        result = MotionQualificationResult(False, 0.4, 0.5, "low_score", 2, {})

        assert service.sample_rejected_motion(datetime.now(timezone.utc), result) == ""
        provider.assert_not_called()


def test_rejected_samples_are_bounded_to_newest_hundred() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        service = _service(root, frame=np.zeros((8, 8, 3), dtype=np.uint8))
        directory = root / "motion_samples" / "gate"
        directory.mkdir(parents=True)
        for index in range(REJECTED_SAMPLE_LIMIT):
            path = directory / f"old-{index:03d}.webp"
            path.touch()
            timestamp = 1_000_000_000 + index
            os.utime(path, ns=(timestamp, timestamp))
        result = MotionQualificationResult(False, 0.4, 0.5, "low_score", 2, {})

        stored = service.sample_rejected_motion(
            datetime(2026, 8, 6, tzinfo=timezone.utc),
            result,
        )

        assert stored
        assert Path(stored).is_file()
        assert len(service.image_writer.stored_images(directory)) == REJECTED_SAMPLE_LIMIT
        assert not (directory / "old-000.webp").exists()


def test_snapshot_filename_uses_normalized_event_time_and_injected_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        service = _service(Path(tmpdir))
        eastern = timezone(timedelta(hours=-4))

        stored = service.write_snapshot(
            np.zeros((8, 8, 3), dtype=np.uint8),
            datetime(2026, 8, 6, 8, 30, 1, 234567, tzinfo=eastern),
        )

        path = Path(stored)
        assert path.is_file()
        assert path.name.startswith("20260806-123001-234567-123456789")
        decoded = service.read_image(stored)
        assert decoded is not None
        assert decoded.shape == (8, 8, 3)


def test_recorded_motion_detection_is_delegated_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        detector = Mock()
        expected = (np.zeros((2, 2, 3), dtype=np.uint8), [{"label": "person"}], "clip.mp4")
        detector.detect.return_value = expected
        service = _service(Path(tmpdir), detector=detector)
        event_at = datetime(2026, 8, 6, tzinfo=timezone.utc)

        result = service.detect_recorded_motion(event_at)

        assert result is expected
        detector.detect.assert_called_once_with(event_at)
