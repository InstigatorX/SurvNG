"""Refinement reads a fragmented segment while the recorder is still writing it."""
from __future__ import annotations

import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from survng.app.config import CameraConfig
from survng.app.motion_pipeline import object_detection
from survng.app.motion_pipeline.object_detection import (
    RECORDED_OPEN_SEGMENT_READY_SECONDS,
    RecordedMotionObjectDetector,
    _EventRecordedSampler,
)
from survng.app.recording_media import fragmented_readable_seconds, mp4_playback_metadata
from survng.app.recording_process.recorder import Recorder


def _segment_path(root: Path, camera: str, start: float) -> Path:
    stamp = datetime.fromtimestamp(start)
    directory = root / "recordings" / camera / "main" / stamp.strftime("%Y-%m-%d") / stamp.strftime("%H")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / stamp.strftime("%Y%m%d-%H%M%S.mp4")


def _running(recorder: Recorder, camera: str, *, fragmented: bool) -> None:
    key = (camera, "main")
    with recorder._lock:
        recorder.processes[key] = (Mock(poll=Mock(return_value=None)), Mock(), Mock())
        if fragmented:
            recorder._fragmented_keys.add(key)


@pytest.fixture(scope="module")
def fragmented_bytes(tmp_path_factory) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    path = tmp_path_factory.mktemp("open-segment") / "complete.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=160x120:rate=10:duration=8", "-c:v", "libx264", "-g", "10", "-bf", "2",
        "-movflags", Recorder._segment_layout_args(
            CameraConfig(id="x", name="X", stream_url="rtsp://x.invalid/x", recording_fragmented=True)
        )[1].removeprefix("movflags=").split(":")[0],
        "-y", str(path),
    ], check=True, capture_output=True, timeout=60)
    return path.read_bytes()


def _written_until(complete: bytes, fragments: int) -> bytes:
    """The file as it stands after FFmpeg has completed ``fragments`` fragments."""
    cursor, seen = 0, 0
    while cursor + 8 <= len(complete):
        size = int.from_bytes(complete[cursor:cursor + 4], "big")
        if complete[cursor + 4:cursor + 8] == b"moof":
            if seen == fragments:
                # Part-way through the next fragment: its header is written,
                # its media data is not all there yet.
                return complete[: cursor + size + 200]
            seen += 1
        cursor += size
    return complete


def test_plain_cameras_do_not_read_open_segments(tmp_path):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=False)
    assert not recorder.reads_open_segments("gate")
    assert recorder.open_recording("gate", time.time()) is None


def test_a_stopped_recorder_has_no_open_segment(tmp_path):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    recorder.processes[("gate", "main")][0].poll.return_value = 1
    assert not recorder.reads_open_segments("gate")
    assert recorder.open_recording("gate", time.time()) is None


def test_open_segment_grows_with_completed_fragments_and_is_never_indexed(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    assert recorder.reads_open_segments("gate")
    start = float(int(time.time()) - 4)
    path = _segment_path(tmp_path, "gate", start)

    path.write_bytes(_written_until(fragmented_bytes, 2))
    row = recorder.open_recording("gate", start + 1.5)
    assert row is not None and row["open"] and row["path"] == str(path)
    assert row["start_epoch"] == start
    assert row["end_epoch"] == pytest.approx(start + 2.0)

    # Before the open segment: not this file's business.
    assert recorder.open_recording("gate", start - 0.5) is None
    # Inside it but not written yet: the row says how far it has got.
    pending = recorder.open_recording("gate", start + 3.5)
    assert pending is not None and pending["end_epoch"] <= start + 3.5

    path.write_bytes(_written_until(fragmented_bytes, 4))
    assert recorder.open_recording("gate", start + 3.5)["end_epoch"] == pytest.approx(start + 4.0)

    assert recorder.recording_rows("gate", limit=8, source="main") == []
    assert recorder._take_recording_edge_refreshes() == {}


def test_the_open_segment_is_found_with_one_directory_listing(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    start = float(int(time.time()) - 3)
    _segment_path(tmp_path, "gate", start - 10).write_bytes(fragmented_bytes)
    _segment_path(tmp_path, "gate", start).write_bytes(_written_until(fragmented_bytes, 2))
    with patch.object(
        recorder, "_recent_hour_recording_files", wraps=recorder._recent_hour_recording_files,
    ) as listing:
        for offset in (0.5, 1.0, 1.5, 9.0, -2.0):
            recorder.open_recording("gate", start + offset)
    # Media storage is listed once per segment, not once per lookup.
    assert listing.call_count == 1


def test_no_open_file_is_remembered_instead_of_listed_on_every_lookup(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    # The camera stalled: FFmpeg is alive but its newest file is long closed.
    stale = float(int(time.time()) - 40)
    _segment_path(tmp_path, "gate", stale).write_bytes(fragmented_bytes)
    with patch.object(
        recorder, "_recent_hour_recording_files", wraps=recorder._recent_hour_recording_files,
    ) as listing:
        for _ in range(6):
            assert recorder.open_recording("gate", time.time() - 1.0) is None
        assert listing.call_count == 1
        # Looked for again later, not never.
        with recorder._lock:
            path, start, listed = recorder._open_segments[("gate", "main")]
            recorder._open_segments[("gate", "main")] = (path, start, listed - 5.0)
        recorder.open_recording("gate", time.time() - 1.0)
        assert listing.call_count == 2


def test_the_previous_process_last_file_is_not_taken_for_the_open_segment(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    now = time.time()
    # Killed 2 s ago; its file is recent enough to look active by age alone.
    old = _segment_path(tmp_path, "gate", float(int(now) - 6))
    old.write_bytes(_written_until(fragmented_bytes, 3))
    with recorder._lock:
        recorder._recorder_started_at[("gate", "main")] = now - 0.5
    # The new process has not created its file yet.
    assert recorder.open_recording("gate", now - 0.2) is None
    new = _segment_path(tmp_path, "gate", float(int(now)))
    new.write_bytes(_written_until(fragmented_bytes, 1))
    with recorder._lock:
        recorder._open_segments.pop(("gate", "main"))
    row = recorder.open_recording("gate", float(int(now)) + 0.5)
    assert row is not None and row["path"] == str(new)


def test_a_listing_that_spans_a_restart_does_not_restore_the_old_file(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    start = float(int(time.time()) - 3)
    _segment_path(tmp_path, "gate", start).write_bytes(_written_until(fragmented_bytes, 2))
    listed = recorder._recent_hour_recording_files

    def restart_during_listing(*args, **kwargs):
        files = listed(*args, **kwargs)
        with recorder._lock:
            recorder._recorder_generation[("gate", "main")] = 7
        return files

    with patch.object(recorder, "_recent_hour_recording_files", side_effect=restart_during_listing):
        recorder.open_recording("gate", start + 1.0)
    assert ("gate", "main") not in recorder._open_segments


def test_a_restarted_recorder_forgets_the_segment_it_was_writing(tmp_path, fragmented_bytes):
    recorder = Recorder("ffmpeg", tmp_path, segment_seconds=10)
    _running(recorder, "gate", fragmented=True)
    start = float(int(time.time()) - 6)
    old = _segment_path(tmp_path, "gate", start)
    old.write_bytes(_written_until(fragmented_bytes, 2))
    assert recorder.open_recording("gate", start + 1.0)["path"] == str(old)

    camera = CameraConfig(
        id="gate", name="Gate", stream_url="rtsp://camera.invalid/main", recording_fragmented=True,
    )
    process = Mock(pid=4321, stderr=iter(()))
    process.poll.return_value = None
    with recorder._lock:
        recorder.processes.clear()
    with (
        patch("survng.app.recording_process.recorder.subprocess.Popen", return_value=process),
        patch.object(recorder, "_audio_output_args", return_value=[]),
        patch.object(recorder, "_owned_ffmpeg_recorders", return_value={}),
        patch.object(recorder, "_kill_pid"),
        patch("survng.app.recording_process.recorder.os.killpg"),
    ):
        try:
            recorder.start(camera, "main")
            new = _segment_path(tmp_path, "gate", float(int(time.time())))
            new.write_bytes(_written_until(fragmented_bytes, 1))
            assert recorder.open_recording("gate", time.time() + 0.5)["path"] == str(new)
        finally:
            recorder.stop("gate", "main")
    assert not recorder.reads_open_segments("gate")


def test_readable_seconds_is_not_cached_as_the_file_grows(tmp_path, fragmented_bytes):
    path = tmp_path / "open.mp4"
    before = mp4_playback_metadata.__globals__["_cached_playback_metadata"].cache_info().currsize
    for fragments, expected in ((1, 1.0), (3, 3.0), (5, 5.0)):
        path.write_bytes(_written_until(fragmented_bytes, fragments))
        assert fragmented_readable_seconds(path) == pytest.approx(expected)
    after = mp4_playback_metadata.__globals__["_cached_playback_metadata"].cache_info().currsize
    assert after == before
    path.write_bytes(fragmented_bytes[:40])
    assert fragmented_readable_seconds(path) is None


class _OpenRecorder:
    """An index with nothing in it yet and one segment being written."""

    ffmpeg_path = "ffmpeg"
    hardware_acceleration = "off"
    segment_seconds = 10.0

    def __init__(self, start: float, readable: float, *, open_reads: bool = True) -> None:
        self.start = start
        self.readable = readable
        self.open_reads = open_reads
        self.index_lookups = 0

    def reads_open_segments(self, _camera_id: str) -> bool:
        return self.open_reads

    def open_recording(self, _camera_id: str, epoch: float):
        if not self.open_reads or epoch < self.start:
            return None
        return {
            "path": "open.mp4", "start_epoch": self.start,
            "end_epoch": self.start + self.readable, "open": True,
        }

    def recording_at(self, _camera_id: str, _epoch: float, request_refresh: bool = True):
        self.index_lookups += 1
        return None

    def recording_rows_between(self, *_args, **_kwargs):
        return []


def test_sampler_serves_the_open_segment_without_asking_the_indexer():
    recorder = _OpenRecorder(start=1000.0, readable=4.0)
    sampler = _EventRecordedSampler(
        camera_id="gate", recorder=recorder, frame_reader=lambda *args, **kwargs: None,
    )
    row = sampler.recording_at(1003.5)
    assert row is not None and row["open"]
    assert sampler.open_segment_lookups == 1 and not sampler.open_segment_pending
    # Not written yet: a miss that must not send the indexer to list storage.
    assert sampler.recording_at(1004.5) is None
    assert sampler.open_segment_pending
    assert recorder.index_lookups == 0
    # Older than the open segment: the index answers, as before.
    assert sampler.recording_at(999.0) is None
    assert recorder.index_lookups == 1
    # The readable end moved on; a row kept beyond one pass would hide that.
    recorder.readable = 5.0
    sampler._open_row = None
    assert sampler.recording_at(1004.5) is not None


def test_offsets_of_one_pass_share_one_read_of_the_open_segment():
    recorder = _OpenRecorder(start=1000.0, readable=4.0)
    calls: list[float] = []
    original = recorder.open_recording
    recorder.open_recording = lambda camera, epoch: calls.append(epoch) or original(camera, epoch)
    sampler = _EventRecordedSampler(
        camera_id="gate", recorder=recorder, frame_reader=lambda *args, **kwargs: None,
    )
    for epoch in (1001.0, 1001.5, 1002.0, 1003.0):
        assert sampler.recording_at(epoch) is not None
    assert calls == [1001.0]
    # Still routed correctly from the kept row: older goes to the index.
    assert sampler.recording_at(999.0) is None and recorder.index_lookups == 1


def test_the_last_frame_interval_of_the_written_data_waits_for_the_next_fragment():
    # Completed fragments end at 4.0 s; their last frame is one interval
    # earlier, so nothing can be decoded "at or after" 3.95 s yet.
    sampler = _EventRecordedSampler(
        camera_id="gate", recorder=_OpenRecorder(start=1000.0, readable=4.0),
        frame_reader=lambda *args, **kwargs: None,
    )
    assert sampler.recording_at(1003.95) is None and sampler.open_segment_pending
    assert sampler.open_segment_lookups == 0
    assert sampler.recording_at(1003.7) is not None


def test_a_failed_read_of_the_open_segment_is_not_followed_by_nudged_retries():
    single_reads: list[float] = []

    def single(_path, offset, **_kwargs):
        single_reads.append(offset)
        return None

    sampler = _EventRecordedSampler(
        camera_id="gate", recorder=_OpenRecorder(start=1000.0, readable=4.0),
        frame_reader=single,
        # FFmpeg 8 cannot open a file that ends inside a fragment header.
        batch_frame_reader=lambda path, offsets, **kwargs: ({}, 1),
    )
    frames, processes, fallbacks = sampler.frames_at(
        Path("open.mp4"), [2.5, 3.5], deadline=None, single_frame_fallback=False,
    )
    assert (frames, processes, fallbacks) == ({}, 1, 0) and single_reads == []
    # A closed segment keeps the nudged single-frame fallback.
    sampler.frames_at(Path("closed.mp4"), [2.5, 3.5], deadline=None)
    assert single_reads == [2.5, 3.5]


def _detector(recorder) -> RecordedMotionObjectDetector:
    return RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        SimpleNamespace(config=SimpleNamespace()),
        recorder,
        lambda: None,
    )


def test_refinement_is_due_a_fragment_after_the_frame_not_at_segment_close():
    target = 1_800_000_000.0
    fragmented = _detector(_OpenRecorder(start=target - 2.0, readable=1.0))
    assert fragmented._predicted_index_epoch(target, 0.75) == target + RECORDED_OPEN_SEGMENT_READY_SECONDS
    # A long settle still wins.
    assert fragmented._predicted_index_epoch(target, 3.0) == target + 3.0

    plain = _OpenRecorder(start=target - 2.0, readable=1.0, open_reads=False)
    # The newest indexed segment ended 2 s before the frame: wait for the next close.
    plain.recording_rows_between = lambda *args, **kwargs: [{"end_epoch": target - 2.0}]
    assert _detector(plain)._predicted_index_epoch(target, 0.75) == target + 8.0 + 1.0


def test_detector_reads_a_frame_from_the_segment_still_being_written(tmp_path, fragmented_bytes):
    recorder = Recorder(shutil.which("ffmpeg"), tmp_path, segment_seconds=10)
    recorder.hardware_acceleration = "off"
    _running(recorder, "gate", fragmented=True)
    start = float(int(time.time()) - 5)
    path = _segment_path(tmp_path, "gate", start)
    path.write_bytes(_written_until(fragmented_bytes, 4))
    backend = _detector(recorder)

    sampler = _EventRecordedSampler(
        camera_id="gate", recorder=recorder,
        frame_reader=backend._read_recorded_frame,
        batch_frame_reader=backend._read_recorded_frames,
    )
    row = sampler.recording_at(start + 2.5)
    assert row is not None and row["open"]
    frames, processes = backend._read_recorded_frames(Path(row["path"]), [2.5, 3.5])
    assert processes == 1 and set(frames) == {2.5, 3.5}
    for offset, item in frames.items():
        assert isinstance(item.frame, np.ndarray) and item.frame.shape == (120, 160, 3)
        assert offset <= item.actual_offset < offset + 0.1 + 1e-6
    # Beyond the completed fragments nothing is decodable yet, and nothing
    # was asked of the indexer for a file it must not index while open.
    assert sampler.recording_at(start + 4.5) is None and sampler.open_segment_pending
    assert recorder._take_recording_edge_refreshes() == {}


def test_a_frame_not_yet_written_is_polled_quickly_and_then_found(monkeypatch):
    event_epoch = 1_800_000_000.0
    recorder = _OpenRecorder(start=event_epoch - 3.0, readable=1.0)
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # FFmpeg completes the fragment while refinement waits.
        recorder.readable = 6.0

    class Detector:
        config = None

        def detect(self, _frame, confidence_threshold=None):
            return [{"label": "car", "confidence": 0.9, "box": (2, 2, 12, 12)}]

    from tests.test_recorded_object_detection import detector_config, detected

    Detector.config = detector_config(
        confidence_threshold=0.5, require_incident_zone=False,
        event_confirmation_frames=1, event_class_confirmation_frames={},
    )
    Detector.detect = lambda self, _frame, confidence_threshold=None: [detected("car", 0.8, (2, 2, 12, 12))]
    backend = RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        Detector(), recorder, lambda: None,
    )
    monkeypatch.setattr(object_detection, "RECORDED_EVENT_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(object_detection.time, "time", lambda: event_epoch + 20.0)
    monkeypatch.setattr(object_detection.time, "sleep", sleep)
    monkeypatch.setattr(
        backend, "_read_recorded_frames",
        lambda path, offsets, **kwargs: ({
            offset: object_detection._DecodedRecordedFrame(
                frame=np.zeros((20, 20, 3), dtype=np.uint8),
                actual_offset=offset,
                exact_timestamp=True,
            )
            for offset in offsets
        }, 1),
    )
    result = backend.detect(datetime.fromtimestamp(event_epoch, timezone.utc))

    assert result.frame_source == "recorded_main"
    assert sleeps and sleeps[0] == object_detection.RECORDED_OPEN_SEGMENT_RETRY_SECONDS
    assert recorder.index_lookups == 0
