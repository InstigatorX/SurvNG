from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from survng.app.config import CameraConfig
from survng.app.tracking_frames import CameraFrameTimeline
from survng.app.video_frames import DecodedVideoFrame, VideoFrameReference


def _service(
    *,
    capture: Mock | None = None,
    recorder: Mock | None = None,
    sample_fps: float = 2.0,
) -> CameraFrameTimeline:
    effective_recorder = recorder if recorder is not None else Mock()
    effective_recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    if recorder is None:
        effective_recorder.recording_rows_between.return_value = []
    return CameraFrameTimeline(
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
    assert service.frames.maxlen == 38
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


def test_recorded_catchup_uses_requested_window_as_post_seek_pts_origin() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 110.0,
        }]
        service = _service(recorder=recorder)

        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            return_value=iter([]),
        ) as sampled:
            list(service.recorded_frames(106.0, 107.0, 2.0, 640))

    assert sampled.call_args.kwargs["start_epoch"] == 106.0
    assert sampled.call_args.kwargs["start_offset_seconds"] == 6.0


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


def test_recorded_cover_frame_bypasses_buffer_and_decodes_nominated_main_frame() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 110.0,
        }]
        service = _service(recorder=recorder)
        service.frames.append((104.0, np.full((10, 20, 3), 3, dtype=np.uint8)))
        full_detail = np.full((1920, 2560, 3), 9, dtype=np.uint8)

        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            return_value=iter([(104.25, full_detail)]),
        ) as decoder:
            selected = service.recorded_frame_at(104.25, 2560)

    assert selected is full_detail
    recorder.recording_rows_between.assert_called_once_with(
        "gate", 104.2, 104.3, source="main"
    )
    assert decoder.call_args.kwargs["maximum_width"] == 2560
    assert decoder.call_args.kwargs["start_offset_seconds"] == 4.25


def test_recorded_cover_frame_uses_exact_reference_before_timestamp_fallback() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 110.0,
        }]
        service = _service(recorder=recorder)
        reference = VideoFrameReference(
            source_path=path,
            seek_offset_seconds=4.0,
            pts=3000,
            pts_seconds=0.25,
            time_base_num=1,
            time_base_den=90000,
            captured_at=104.25,
        )
        full_detail = np.full((1920, 2560, 3), 9, dtype=np.uint8)

        with (
            patch(
                "survng.app.tracking_frames.video_frame_at_reference",
                return_value=DecodedVideoFrame(104.25, full_detail, reference),
            ) as exact_decoder,
            patch("survng.app.tracking_frames.sampled_video_frames") as fallback,
        ):
            selected = service.recorded_frame_at(104.25, 2560, reference)

    assert selected is full_detail
    exact_decoder.assert_called_once_with(
        reference,
        ffmpeg_path="/usr/bin/ffmpeg",
        maximum_width=2560,
    )
    fallback.assert_not_called()


def test_recorded_cover_does_not_approximate_when_exact_reference_fails() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 110.0,
        }]
        service = _service(recorder=recorder)
        reference = VideoFrameReference(
            source_path=path,
            seek_offset_seconds=4.0,
            pts=3000,
            pts_seconds=0.25,
            time_base_num=1,
            time_base_den=90000,
            captured_at=104.25,
        )

        with (
            patch(
                "survng.app.tracking_frames.video_frame_at_reference",
                return_value=None,
            ),
            patch("survng.app.tracking_frames.sampled_video_frames") as fallback,
        ):
            selected = service.recorded_frame_at(104.25, 2560, reference)

    assert selected is None
    fallback.assert_not_called()

def test_live_history_bridges_open_segment_tail_when_recordings_end() -> None:
    """Finalized segment ends; open next segment is filled from live history."""
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "segment.mp4"
        path.write_bytes(b"segment")
        # Only the finalized segment is indexed; the open tail is absent.
        recorder.recording_rows_between.return_value = [{
            "path": str(path),
            "start_epoch": 100.0,
            "end_epoch": 103.0,
        }]
        service = _service(recorder=recorder, sample_fps=2.0)
        # Live history covers the unfinalized 3.5s tail after recordings end.
        for epoch, value in (
            (103.5, 21),
            (104.0, 22),
            (104.5, 23),
            (105.0, 24),
            (105.5, 25),
            (106.0, 26),
        ):
            service.remember(
                np.full((10, 20, 3), value, dtype=np.uint8),
                epoch,
                source="live",
            )
        decoded = [
            (100.0, np.full((10, 20, 3), 1, dtype=np.uint8)),
            (101.0, np.full((10, 20, 3), 2, dtype=np.uint8)),
            (102.0, np.full((10, 20, 3), 3, dtype=np.uint8)),
            (103.0, np.full((10, 20, 3), 4, dtype=np.uint8)),
        ]
        with patch(
            "survng.app.tracking_frames.sampled_video_frames",
            return_value=iter(decoded),
        ):
            samples = list(service.recorded_frames(100.0, 106.5, 2.0, 640))

    epochs = [sample[0] for sample in samples]
    assert epochs == [100.0, 101.0, 102.0, 103.0, 103.5, 104.0, 104.5, 105.0, 105.5, 106.0]
    # Live-bridged samples keep their retained pixels.
    assert int(samples[-1][1][0, 0, 0]) == 26
    assert int(samples[4][1][0, 0, 0]) == 21


def test_history_capacity_covers_the_open_segment_bridge_window() -> None:
    """The retained deque must not be shorter than the advertised bridge."""
    for sample_fps in (1.0, 2.0, 3.0, 5.0):
        size = CameraFrameTimeline.buffer_size(sample_fps)
        assert (size - 1) / sample_fps >= 12.0


def test_main_clear_preserves_live_bridge_history() -> None:
    service = _service(sample_fps=2.0)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    service.remember(frame, 100.0, source="main")
    service.remember(frame, 100.5, source="live")
    service.clear("main")
    assert not service.frames
    assert [sample[0] for sample in service.live_frames] == [100.5]


def test_live_capture_restart_is_a_tracking_continuity_boundary() -> None:
    service = _service(sample_fps=2.0)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    service.remember(frame, 100.0, source="main")
    service.remember(frame, 100.5, source="live")
    service.clear("live", captured_at=101.0)

    batch = service.read_recorded_frames(100.0, 102.0, 2.0, 640)

    assert [sample[0] for sample in batch] == [100.0, 100.5]
    assert batch.covered_through == 100.5
    assert batch.interruption == "capture_generation_changed"


def test_main_capture_restart_does_not_interrupt_continuous_live_bridge() -> None:
    service = _service(sample_fps=2.0)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    service.remember(frame, 100.0, source="live")
    service.clear("main", captured_at=100.5)
    service.remember(frame, 101.0, source="live")

    batch = service.read_recorded_frames(100.0, 101.0, 2.0, 640)

    assert [sample[0] for sample in batch] == [100.0, 101.0]
    assert batch.interruption is None


def test_recorder_epoch_rollover_is_a_tracking_continuity_boundary() -> None:
    recorder = Mock()
    recorder.ffmpeg_path = "/usr/bin/ffmpeg"
    recorder.recording_rows_between.return_value = []
    recorder.timestamp_health.return_value = {
        ("gate", "main"): {"last_rollover_at": "1970-01-01T00:01:41+00:00"}
    }
    service = _service(recorder=recorder, sample_fps=2.0)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    service.remember(frame, 100.0, source="live")
    service.remember(frame, 100.5, source="live")

    batch = service.read_recorded_frames(100.0, 102.0, 2.0, 640)

    assert [sample[0] for sample in batch] == [100.0, 100.5]
    assert batch.interruption == "recorder_epoch_changed"
