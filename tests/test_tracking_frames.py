from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from survng.app.config import CameraConfig
from survng.app.tracking_frames import TrackingFrameService


def _service(
    *,
    capture: Mock | None = None,
    recorder: Mock | None = None,
    sample_fps: float = 2.0,
) -> TrackingFrameService:
    effective_recorder = recorder if recorder is not None else Mock()
    effective_recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    if recorder is None:
        effective_recorder.recording_rows_between.return_value = []
    return TrackingFrameService(
        camera=CameraConfig(
            id="gate",
            name="Gate",
            stream_url="rtsp://example.invalid/main",
            live_stream_url="rtsp://example.invalid/sub",
        ),
        capture=capture or Mock(),
        recorder=effective_recorder,
        stop_event=threading.Event(),
        sample_fps=lambda: sample_fps,
    )


def test_latest_prefers_main_and_falls_back_with_original_timestamps() -> None:
    capture = Mock()
    live = SimpleNamespace(
        image=np.full((10, 20, 3), 7, dtype=np.uint8),
        captured_at_epoch=100.0,
        captured_at_monotonic=50.0,
    )
    capture.request_frame.side_effect = [None, live]
    service = _service(capture=capture)

    sample = service.latest_with_fallback()

    assert sample is not None
    assert int(sample[0][0, 0, 0]) == 7
    assert sample[1:] == (100.0, 50.0)
    assert [call.args[0] for call in capture.request_frame.call_args_list] == [
        "main",
        "live",
    ]


def test_buffer_sampling_is_bounded_resized_and_cleared() -> None:
    service = _service(sample_fps=2.0)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    service.remember(frame, 100.0)
    service.remember(frame, 100.1)
    service.remember(frame, 100.5)
    service.resize(3.0)

    assert [sample[0] for sample in service.frames] == [100.0, 100.5]
    assert service.frames[0][1].shape == (360, 640, 3)
    assert service.frames.maxlen == 32
    service.clear()
    assert not service.frames


def test_recorded_and_buffered_frames_merge_in_timestamp_order_without_duplicates() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 101.0,
        }]
        service = _service(recorder=recorder)
        service.frames.extend([
            (100.5, np.full((10, 20, 3), 5, dtype=np.uint8)),
            (101.0, np.full((10, 20, 3), 10, dtype=np.uint8)),
        ])
        decoded = [
            (100.0, np.zeros((10, 20, 3), dtype=np.uint8)),
            (100.5, np.zeros((10, 20, 3), dtype=np.uint8)),
        ]

        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            return_value=iter(decoded),
        ):
            samples = list(service.recorded_frames(100.0, 101.0, 2.0, 640))

    assert [sample[0] for sample in samples] == [100.0, 100.5, 101.0]
    assert int(samples[1][1][0, 0, 0]) == 0
    recorder.recording_rows_between.assert_called_once_with(
        "gate", 100.0, 101.0, source="main"
    )


def test_stopped_service_does_not_request_new_capture_frames() -> None:
    capture = Mock()
    service = _service(capture=capture)
    service.stop_event.set()

    assert service.latest("main") is None
    capture.request_frame.assert_not_called()


def test_clear_during_resize_does_not_reintroduce_a_stale_frame() -> None:
    service = _service(sample_fps=2.0)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    resizing = threading.Event()
    release = threading.Event()
    original_resize = cv2.resize

    def delayed_resize(*args, **kwargs):
        resizing.set()
        release.wait(timeout=1)
        return original_resize(*args, **kwargs)

    with patch("survng.app.tracking_frames.cv2.resize", side_effect=delayed_resize):
        writer = threading.Thread(target=service.remember, args=(frame, 100.0))
        writer.start()
        assert resizing.wait(timeout=1)
        service.clear()
        release.set()
        writer.join(timeout=1)

    assert not writer.is_alive()
    assert not service.frames


def test_recording_rows_are_sorted_before_cross_segment_deduplication() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        first = Path(tmpdir) / "first.mp4"
        second = Path(tmpdir) / "second.mp4"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        recorder.recording_rows_between.return_value = [
            {"path": str(second), "start_epoch": 101.0, "end_epoch": 102.0},
            {"path": str(first), "start_epoch": 100.0, "end_epoch": 101.0},
        ]
        service = _service(recorder=recorder)

        def decoded(path, **_kwargs):
            captured_at = 100.0 if path == first else 101.0
            return iter([(captured_at, np.zeros((4, 4, 3), dtype=np.uint8))])

        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            side_effect=decoded,
        ):
            samples = list(service.recorded_frames(100.0, 102.0, 1.0, 640))

    assert [captured_at for captured_at, _frame in samples] == [100.0, 101.0]


def test_recorded_catchup_streams_and_stops_before_decoding_later_segments() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    opened: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        first = Path(tmpdir) / "first.mp4"
        second = Path(tmpdir) / "second.mp4"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        recorder.recording_rows_between.return_value = [
            {"path": str(first), "start_epoch": 100.0, "end_epoch": 101.0},
            {"path": str(second), "start_epoch": 101.0, "end_epoch": 102.0},
        ]
        service = _service(recorder=recorder)

        def decoded(path, **_kwargs):
            opened.append(path.name)
            captured_at = 100.0 if path == first else 101.0
            yield captured_at, np.zeros((4, 4, 3), dtype=np.uint8)

        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            side_effect=decoded,
        ):
            samples = service.recorded_frames(100.0, 102.0, 1.0, 640)
            assert next(samples)[0] == 100.0
            assert opened == ["first.mp4"]
            service.stop_event.set()
            assert list(samples) == []

    assert opened == ["first.mp4"]
