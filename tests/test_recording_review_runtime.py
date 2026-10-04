from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from survng.app.evidence_work import check_evidence_cancellation
from survng.app.evidence_work import run_evidence_process
from survng.app.motion_pipeline.recorded_decode_budget import RecordedDecodeBudget
from survng.app.recording_review_runtime import RecordingReviewRuntime


@pytest.fixture
def runtime_fixture(tmp_path):
    path = tmp_path / "recording.mp4"
    path.write_bytes(b"recorded fixture")
    os.utime(path, (0, 0))
    row = {"path": str(path), "start_epoch": 0.0, "end_epoch": 60.0,
           "size_bytes": path.stat().st_size, "modified_at": path.stat().st_mtime,
           "stream_fingerprint": "h264"}
    held = set()

    def acquire(_row, **_kwargs):
        token = str(len(acquire.mock_calls))
        held.add(token)
        return token

    acquire = Mock(side_effect=acquire)
    recorder = SimpleNamespace(
        ffmpeg_path="ffmpeg",
        recording_rows_between=Mock(return_value=[row]),
        acquire_recording_for_playback=acquire,
        release_recording_playback=Mock(side_effect=held.remove),
    )
    detector = SimpleNamespace(detect_enrichment=Mock(return_value=[]), detect=Mock())
    budget = RecordedDecodeBudget(max_processes=2, memory_budget_bytes=256 << 20)
    config = {"enabled": True, "confidence_threshold": 0.45}
    runtime = RecordingReviewRuntime(
        recorder=recorder, detector_provider=lambda: detector,
        decode_budget=budget, detector_config_provider=lambda: config,
    )
    manifest = runtime.manifest("gate", "main", 0, 60)
    job = {"camera_id": "gate", "source": "main", "start_epoch": 0, "end_epoch": 60, "manifest": manifest}
    return SimpleNamespace(runtime=runtime, recorder=recorder, detector=detector, budget=budget,
                           config=config, path=path, job=job, held=held)


def assert_released(fixture):
    assert not fixture.held
    status = fixture.budget.status()
    assert status["active_processes"] == 0
    assert status["reserved_bytes"] == 0


def test_manifest_reads_index_only_and_rejects_active_segments(runtime_fixture, monkeypatch):
    f = runtime_fixture
    f.recorder.recording_rows_between.assert_called_once_with("gate", 0, 60, source="main", discover_missing=False)
    row = f.job["manifest"][0]
    assert row["size_bytes"] == f.path.stat().st_size
    assert row["modified_at"] == f.path.stat().st_mtime
    # GET/polling must never touch recording storage, including network mounts.
    monkeypatch.setattr(Path, "stat", Mock(side_effect=AssertionError("filesystem access during manifest read")))
    assert len(f.runtime.manifest("gate", "main", 0, 60)) == 1
    f.recorder.recording_rows_between.return_value[0]["modified_at"] = 10**12
    assert f.runtime.manifest("gate", "main", 0, 60) == []


@pytest.mark.parametrize("start,end,source", [(0, 61, "main"), (1, 1, "main"), (math.nan, 60, "main"), (0, 60, "url")])
def test_manifest_rejects_unbounded_requests(runtime_fixture, start, end, source):
    with pytest.raises(ValueError):
        runtime_fixture.runtime.manifest("gate", source, start, end)


def test_manifest_rejects_excessive_segment_count(runtime_fixture):
    f = runtime_fixture
    f.recorder.recording_rows_between.return_value *= 129
    with pytest.raises(ValueError, match="too many segments"):
        f.runtime.manifest("gate", "main", 0, 60)


def test_twelve_low_priority_samples_have_only_safe_normalized_fields(runtime_fixture, monkeypatch):
    f = runtime_fixture
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    monkeypatch.setattr(f.runtime, "_decode", Mock(return_value=frame))

    def infer(_frame):
        assert f.held
        assert f.budget.status()["active_processes"] == 0
        assert f.budget.status()["reserved_bytes"] > 0
        return [{"label": "person", "confidence": 0.9,
                 "box": {"x1": 20, "y1": 10, "x2": 100, "y2": 90},
                 "event_id": 7, "identity": "must not propagate"}]

    f.detector.detect_enrichment.side_effect = infer
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert len(results) == 12
    assert [row["timestamp"] for row in results] == [2.5 + 5 * i for i in range(12)]
    assert all(row["sample_time_kind"] == "requested" for row in results)
    assert results[0]["objects"] == [{"label": "person", "confidence": 0.9,
                                      "box": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}}]
    assert all(row["status"] == "sampled" for row in results)
    assert f.detector.detect_enrichment.call_count == 12
    f.detector.detect.assert_not_called()
    assert_released(f)


@pytest.mark.parametrize("marker,expected", [("inference_deferred", "deferred"), ("detector_unavailable", "failed"), ("inference_error", "failed")])
def test_busy_or_failed_inference_is_not_clean_negative(runtime_fixture, monkeypatch, marker, expected):
    f = runtime_fixture
    monkeypatch.setattr(f.runtime, "_decode", Mock(return_value=np.zeros((20, 20, 3), dtype=np.uint8)))
    f.detector.detect_enrichment.return_value = [{"status": marker, "error": "/private/model/path"}]
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert all(row["status"] == expected for row in results)
    assert "/private" not in str(results)
    assert_released(f)


def test_missing_low_priority_detector_never_falls_back_to_raw_detection(runtime_fixture):
    f = runtime_fixture
    del f.detector.detect_enrichment
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert all(row["status"] == "failed" for row in results)
    f.detector.detect.assert_not_called()
    f.recorder.acquire_recording_for_playback.assert_not_called()
    assert_released(f)


def test_cancel_between_samples_holds_no_resources(runtime_fixture, monkeypatch):
    f = runtime_fixture
    monkeypatch.setattr(f.runtime, "_decode", Mock(return_value=np.zeros((20, 20, 3), dtype=np.uint8)))
    stop = threading.Event()
    samples = f.runtime.analyze(f.job, stop)
    assert next(samples)["status"] == "sampled"
    assert_released(f)
    stop.set()
    assert list(samples) == []
    assert f.detector.detect_enrichment.call_count == 1


def test_cancel_during_decode_releases_all_leases(runtime_fixture, monkeypatch):
    f = runtime_fixture
    stop = threading.Event()

    def cancelled_decode(*_args):
        stop.set()
        check_evidence_cancellation()

    monkeypatch.setattr(f.runtime, "_decode", cancelled_decode)
    assert list(f.runtime.analyze(f.job, stop)) == []
    f.detector.detect_enrichment.assert_not_called()
    assert_released(f)


def test_cancel_reaps_blocked_decoder_process(runtime_fixture, monkeypatch):
    f = runtime_fixture
    stop = threading.Event()
    entered = threading.Event()
    owned = []
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        owned.append(process)
        entered.set()
        return process

    def blocked_decode(_command, *, timeout):
        # Use a sleeping child to reproduce a decoder that stops emitting data.
        return run_evidence_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout=timeout)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr("survng.app.recording_review_runtime.run_evidence_process", blocked_decode)
    results = []
    thread = threading.Thread(target=lambda: results.extend(f.runtime.analyze(f.job, stop)))
    thread.start()
    try:
        assert entered.wait(2)
        began = time.monotonic()
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
        assert time.monotonic() - began < 2
        assert len(owned) == 1 and owned[0].poll() is not None
        assert results == []
        f.detector.detect_enrichment.assert_not_called()
        assert_released(f)
    finally:
        stop.set()
        thread.join(2)
        for process in owned:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)


def test_recording_changed_after_manifest_is_not_analyzed(runtime_fixture, monkeypatch):
    f = runtime_fixture
    decode = Mock()
    monkeypatch.setattr(f.runtime, "_decode", decode)
    f.path.write_bytes(b"different recording bytes")
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert all(row["reason"] == "recording_changed" for row in results)
    decode.assert_not_called()
    assert_released(f)


def test_retention_winning_race_returns_unavailable(runtime_fixture):
    f = runtime_fixture
    f.recorder.acquire_recording_for_playback.side_effect = None
    f.recorder.acquire_recording_for_playback.return_value = None
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert all(row["reason"] == "recording_unavailable" for row in results)
    f.detector.detect_enrichment.assert_not_called()
    assert_released(f)


def test_shared_decoder_saturation_is_deferred_without_starting_ffmpeg(runtime_fixture, monkeypatch):
    f = runtime_fixture
    monkeypatch.setattr("survng.app.recording_review_runtime.ADMISSION_WAIT_SECONDS", 0)
    first = f.budget.acquire_process(incident_epoch=1)
    second = f.budget.acquire_process(incident_epoch=2)
    try:
        results = list(f.runtime.analyze(f.job, threading.Event()))
        assert all(row["status"] == "deferred" and row["reason"] == "decoder_busy" for row in results)
        f.detector.detect_enrichment.assert_not_called()
    finally:
        first.release()
        second.release()
    assert_released(f)


def test_analysis_identity_changes_with_model_and_policy_but_exposes_no_path(runtime_fixture, tmp_path):
    f = runtime_fixture
    model = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    model.write_bytes(b"xml")
    weights.write_bytes(b"weights")
    f.config["model_xml"] = str(model)
    initial = f.runtime.analysis_identity()
    assert len(initial) == 64 and "/" not in initial
    assert initial == f.runtime.analysis_identity()
    weights.write_bytes(b"changed weights")
    updated = f.runtime.analysis_identity()
    assert initial != updated
    f.config["confidence_threshold"] = 0.8
    assert updated != f.runtime.analysis_identity()


def test_isolated_identity_pins_loaded_model_until_worker_reload(runtime_fixture, tmp_path, monkeypatch):
    f = runtime_fixture
    model = tmp_path / "weights.bin"
    model.write_bytes(b"loaded weights")
    f.config["model_path"] = str(model)
    state = {"instances": [{"index": 1, "worker_pid": 100, "generation": 1}]}
    f.detector.isolation_status = lambda: state
    runtime = RecordingReviewRuntime(
        recorder=f.recorder, detector_provider=lambda: f.detector, decode_budget=f.budget,
        detector_config_provider=lambda: f.config,
    )
    original = runtime.analysis_identity()
    model.write_bytes(b"replacement weights not yet loaded")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", Mock(side_effect=AssertionError("steady-state identity must not stat models")))
        assert runtime.analysis_identity() == original
    state["instances"][0].update(worker_pid=200, generation=2)
    reloaded = runtime.analysis_identity()
    assert reloaded != original
    state["instances"][0].update(worker_pid=300, generation=3)
    assert runtime.analysis_identity() == reloaded


def test_model_identity_tracks_partial_pool_reload_without_relabeling_old_worker(runtime_fixture, tmp_path):
    f = runtime_fixture
    model = tmp_path / "weights.bin"
    model.write_bytes(b"old model")
    f.config["model_path"] = str(model)
    instances = [{"index": 1, "worker_pid": 100, "generation": 1},
                 {"index": 2, "worker_pid": 101, "generation": 1}]
    f.detector.isolation_status = lambda: {"instances": instances}
    original = f.runtime.analysis_identity()
    model.write_bytes(b"replacement model")
    instances[0]["worker_pid"] = None
    assert f.runtime.analysis_identity() == original
    instances[0].update(worker_pid=200, generation=2)
    mixed = f.runtime.analysis_identity()
    assert mixed != original
    instances[1].update(worker_pid=201, generation=2)
    reloaded = f.runtime.analysis_identity()
    assert reloaded not in {original, mixed}
    # Volatile status fields never invalidate completed review results.
    instances[0]["pending_requests"] = 17
    assert f.runtime.analysis_identity() == reloaded


def test_decode_error_even_with_image_is_not_success(runtime_fixture, monkeypatch):
    f = runtime_fixture
    ok, jpeg = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
    assert ok
    monkeypatch.setattr("survng.app.recording_review_runtime.run_evidence_process", Mock(return_value=
                        subprocess.CompletedProcess([], 0, jpeg.tobytes(), b"[hevc] corrupt slice")))
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert all(row["reason"] == "decode_failed" for row in results)
    f.detector.detect_enrichment.assert_not_called()
    assert_released(f)


def test_job_deadline_marks_unprocessed_samples_deferred(runtime_fixture, monkeypatch):
    f = runtime_fixture
    monkeypatch.setattr("survng.app.recording_review_runtime.MAX_RUN_SECONDS", 0)
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert len(results) == 12
    assert all(row["status"] == "deferred" and row["reason"] == "review_time_budget" for row in results)
    f.detector.detect_enrichment.assert_not_called()
    assert_released(f)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_real_recording_decode_preserves_aspect_ratio_and_thread_budgets(runtime_fixture, monkeypatch):
    f = runtime_fixture
    created = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=red:s=160x90:r=5", "-t", "3", "-c:v", "mpeg4", "-threads:v", "1", str(f.path),
    ], capture_output=True, timeout=10, check=False)
    assert created.returncode == 0, created.stderr.decode()
    os.utime(f.path, (0, 0))
    f.recorder.recording_rows_between.return_value[0].update(end_epoch=3, size_bytes=f.path.stat().st_size)
    f.job["manifest"] = f.runtime.manifest("gate", "main", 0, 60)
    shapes = []

    def infer(frame):
        shapes.append(frame.shape)
        assert frame[:, :, 2].mean() > 200
        return []

    f.detector.detect_enrichment.side_effect = infer
    commands = []

    def record_command(command, *, timeout):
        commands.append(command)
        return run_evidence_process(command, timeout=timeout)

    monkeypatch.setattr("survng.app.recording_review_runtime.run_evidence_process", record_command)
    results = list(f.runtime.analyze(f.job, threading.Event()))
    assert results[0]["status"] == "sampled"
    assert all(row["status"] == "unavailable" for row in results[1:])
    assert shapes == [(360, 640, 3)]
    assert len(commands) == 1
    input_end = commands[0].index("-i")
    input_args, output_args = commands[0][:input_end], commands[0][input_end:]
    assert input_args[input_args.index("-threads:v") + 1] == "2"
    assert input_args[input_args.index("-filter_threads") + 1] == "1"
    assert output_args[output_args.index("-threads:v") + 1] == "1"
    assert_released(f)
