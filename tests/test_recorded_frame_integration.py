"""Real FFmpeg round trips preserve recorded pixels and source timestamps."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from survng.app.tracking_comparison import sampled_video_frames, video_frame_at_reference


@pytest.fixture(scope="module")
def source_clip(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    path = tmp_path_factory.mktemp("tracking-audit") / "sample.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=96x64:rate=10:duration=5", "-c:v", "libx264", "-y", str(path),
    ], check=True, capture_output=True, timeout=20)
    return ffmpeg, path


def test_exact_reference_round_trip(source_clip):
    ffmpeg, path = source_clip
    for offset in (0.0, 1.25, 2.8):
        samples = list(sampled_video_frames(
            path, start_epoch=100 + offset, start_offset_seconds=offset,
            sample_fps=2, duration_seconds=1, ffmpeg_path=ffmpeg,
        ))
        assert samples
        for sample in samples:
            assert sample.reference.exact
            assert offset <= sample.reference.pts_seconds < offset + 1
            assert sample.captured_at == pytest.approx(100 + sample.reference.pts_seconds)
            decoded = video_frame_at_reference(sample.reference, ffmpeg_path=ffmpeg, maximum_width=640)
            assert decoded is not None
            assert np.array_equal(decoded.frame, sample.frame)
