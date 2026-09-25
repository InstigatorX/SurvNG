from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from survng.app.evidence_work import (
    EvidenceWorkPreempted,
    cancellable_evidence_work,
    run_evidence_process,
)


@pytest.mark.parametrize("batch", [False, True])
def test_cancel_reaps_owned_decoder(monkeypatch, tmp_path, batch):
    stop = threading.Event()
    started = threading.Event()
    processes = []
    errors = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)

    def decode():
        try:
            with cancellable_evidence_work(stop.is_set, optional=False):
                if batch:
                    from types import SimpleNamespace
                    from survng.app.config import CameraConfig
                    from survng.app.motion_pipeline.object_detection import RecordedMotionObjectDetector
                    from survng.app.motion_pipeline.recorded_decode_budget import RecordedDecodeBudget
                    budget = RecordedDecodeBudget(max_processes=1)
                    backend = RecordedMotionObjectDetector(
                        CameraConfig(id="gate", name="Gate", stream_url="rtsp://example.invalid/main"),
                        SimpleNamespace(config=SimpleNamespace()),
                        SimpleNamespace(ffmpeg_path="ffmpeg", hardware_acceleration="off"),
                        lambda: None, decode_budget=budget,
                    )
                    recording = tmp_path / "sample.mp4"
                    recording.touch()
                    try:
                        backend._read_recorded_frames(recording, [.5, 1.0])
                    finally:
                        assert budget.status()["active_processes"] == 0
                else:
                    run_evidence_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=20)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=decode)
    thread.start()
    try:
        assert started.wait(2)
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], EvidenceWorkPreempted)
        assert processes[0].poll() is not None
    finally:
        stop.set()
        for process in processes:
            if process.poll() is None:
                process.kill()
        thread.join(2)


def test_scoped_decoder_preserves_output_and_deadline():
    with cancellable_evidence_work(lambda: False, optional=False):
        result = run_evidence_process([sys.executable, "-c", "print('frame')"], timeout=2)
        assert result.returncode == 0
        assert result.stdout == b"frame\n"
        with pytest.raises(subprocess.TimeoutExpired):
            run_evidence_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=.05)
