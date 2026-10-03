"""Recorded decode budgets apply to hardware attempts and software fallbacks."""

from io import BytesIO
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from survng.app.config import CameraConfig
from survng.app.motion_pipeline import object_detection


@pytest.mark.parametrize("mode", ["off", "qsv", "vaapi"])
@pytest.mark.parametrize("batch", [False, True])
def test_recorded_decode_limits_survive_hardware_fallback(tmp_path, monkeypatch, mode, batch):
    path = tmp_path / "segment.mp4"
    path.touch()
    expected = np.full((12, 16, 3), 37, dtype=np.uint8)
    ok, encoded = cv2.imencode(".bmp", expected)
    assert ok
    commands = []

    def result_for(command):
        commands.append(command)
        # An unavailable hardware decoder must not remove the CPU budget from
        # the software retry, nor prevent successful evidence extraction.
        failed = "-hwaccel" in command
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout=b"" if failed else encoded.tobytes(),
            stderr=b"" if failed else b"[showinfo@event_sample] n: 0 pts: 500 pts_time:0.5\n",
        )

    def run(command, **kwargs):
        return result_for(command)

    def popen(command, **kwargs):
        result = result_for(command)
        return SimpleNamespace(
            returncode=result.returncode,
            stdout=BytesIO(result.stdout),
            stderr=BytesIO(result.stderr),
            wait=lambda timeout: result.returncode,
        )

    monkeypatch.setattr(object_detection, "run_evidence_process", run)
    monkeypatch.setattr(object_detection.subprocess, "Popen", popen)
    backend = object_detection.RecordedMotionObjectDetector(
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
        SimpleNamespace(config=SimpleNamespace()),
        SimpleNamespace(ffmpeg_path="ffmpeg", hardware_acceleration=mode),
        lambda: None,
    )
    if batch:
        frames, process_count = backend._read_recorded_frames(path, [0.5])
        actual = frames[0.5].frame
        assert frames[0.5].actual_offset == 0.5
        assert process_count == len(commands)
    else:
        actual = backend._read_recorded_frame(path, 0.5)
    assert np.array_equal(actual, expected)
    assert len(commands) == (1 if mode == "off" else 2)
    for command in commands:
        before_input = command[:command.index("-i")]
        after_input = command[command.index("-i") + 2:]
        assert before_input[before_input.index("-threads:v") + 1] == "2"
        assert before_input[before_input.index("-filter_threads") + 1] == "1"
        assert after_input[after_input.index("-threads:v") + 1] == "1"
    assert "-hwaccel" not in commands[-1]
