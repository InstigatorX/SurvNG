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
