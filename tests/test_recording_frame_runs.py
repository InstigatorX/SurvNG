import shutil
import subprocess

import numpy as np
import pytest

from survng.app.face_capture.pipeline import rows_for_window
from survng.app.recording_frame_runs import FrameRunError, stream_frames

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("stream") / "clip.mp4"
    subprocess.run([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=96x64:rate=10:duration=4", "-c:v", "libx264", "-g", "7", "-bf", "2", "-y", str(path),
    ], check=True, capture_output=True, timeout=20)
    return path


def test_every_frame_of_a_window_comes_once_in_order_with_its_time(clip):
    frames = list(stream_frames(FFMPEG, clip, 1.05, 2.55, size=(96, 64), consumer="test"))
    times = [pts for pts, _ in frames]
    assert times == pytest.approx([1.1 + 0.1 * i for i in range(15)], abs=1e-6)
    assert all(frame.shape == (64, 96, 3) and frame.dtype == np.uint8 for _, frame in frames)
    # Frames that differ in the source differ here too (no duplicates).
    assert len({frame.tobytes() for _, frame in frames}) == len(frames)


def test_a_window_matches_the_same_frames_decoded_from_the_start(clip):
    whole = dict(stream_frames(FFMPEG, clip, 0.0, 4.0, size=(96, 64), consumer="test"))
    window = dict(stream_frames(FFMPEG, clip, 2.0, 3.0, size=(96, 64), consumer="test"))
    assert window and all(np.array_equal(frame, whole[pts]) for pts, frame in window.items())


def test_a_missing_file_raises_a_frame_run_error(tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    with pytest.raises(FrameRunError):
        list(stream_frames(FFMPEG, tmp_path / "gone.mp4", 0.0, 1.0, size=(96, 64), consumer="test"))


def test_frames_keep_their_own_times_and_size_across_a_resolution_change(tmp_path):
    # FFmpeg rebuilds the filter graph at the change and showinfo counts from
    # 0 again; later frames must not take a neighbour's time.
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    parts = []
    for index, (size, duration) in enumerate((("96x64", 0.5), ("48x32", 1.5))):
        part = tmp_path / f"{index}.ts"
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"testsrc2=size={size}:rate=10:duration={duration}", "-c:v", "libx264", "-g", "5",
            "-output_ts_offset", str(index * 3.0), "-f", "mpegts", "-y", str(part),
        ], check=True, capture_output=True, timeout=20)
        parts.append(part.read_bytes())
    joined = tmp_path / "joined.ts"
    joined.write_bytes(b"".join(parts))
    frames = list(stream_frames(FFMPEG, joined, 0.0, 10.0, size=(96, 64), consumer="test"))
    assert len(frames) == 20
    assert all(frame.shape == (64, 96, 3) for _, frame in frames)
    times = [pts for pts, _ in frames]
    assert None not in times
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert times[:5] == pytest.approx([0.1 * i for i in range(5)], abs=1e-6)


def test_window_rows_are_looked_up_once_per_source_and_filtered():
    calls = []
    segments = [{"start_epoch": 100.0 + 10 * i, "duration_seconds": 10.0, "path": f"{i}.mp4"} for i in range(4)]

    def rows(source, start, end):
        calls.append((source, start, end))
        return segments

    cached = rows_for_window(rows, 100.0, 140.0)
    assert [r["path"] for r in cached("main", 112.0, 118.0)] == ["1.mp4"]
    assert [r["path"] for r in cached("main", 119.0, 121.0)] == ["1.mp4", "2.mp4"]
    cached("live", 100.0, 101.0)
    assert calls == [("main", 100.0, 140.0), ("live", 100.0, 140.0)]


def test_an_unprobeable_segment_is_noted_and_the_next_one_still_streams(clip, tmp_path):
    from survng.app.face_capture.pipeline import main_stream_source

    segments = [
        {"start_epoch": 100.0, "duration_seconds": 1.0, "path": str(tmp_path / "gone.mp4")},
        {"start_epoch": 101.0, "duration_seconds": 4.0, "path": str(clip)},
    ]
    source = main_stream_source(lambda _source, _start, _end: segments, 100.0, FFMPEG)
    offsets = [offset for offset, _ in source(0.5, 2.0)]
    assert offsets and min(offsets) >= 1.0
    assert len(source.errors) == 1 and "gone.mp4" in source.errors[0]
