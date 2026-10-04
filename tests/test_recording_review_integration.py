"""Real FFmpeg through the isolated review queue, with no incident admission."""
import json
import os
import shutil
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from survng.app.event_store import EventStore
from survng.app.config import AppConfig
from survng.app.config_application import manager_owned_config
from survng.app.motion_pipeline.recorded_decode_budget import RecordedDecodeBudget
from survng.app.recording_review import RecordingReviewService
from survng.app.recording_review_runtime import RecordingReviewRuntime


def test_review_is_opt_in_and_toggle_uses_owned_manager_lifecycle():
    current = AppConfig()
    assert current.recording_review.enabled is False
    enabled = current.model_copy(deep=True)
    enabled.recording_review.enabled = True
    assert manager_owned_config(current) != manager_owned_config(enabled)


def test_recording_without_incident_can_be_reviewed_and_cached(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for recorded review integration")
    video = tmp_path / "minute.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                    "color=c=gray:s=64x64:r=1:d=60", "-threads", "1", "-c:v", "mpeg4", str(video)],
                   check=True, capture_output=True, timeout=10)
    os.utime(video, (1000, 1000))
    row = {"path": str(video), "start_epoch": 0, "end_epoch": 60,
           "size_bytes": video.stat().st_size, "modified_at": video.stat().st_mtime}
    leases = set()

    def acquire(*_args, **_kwargs):
        token = object()
        leases.add(token)
        return token

    recorder = SimpleNamespace(ffmpeg_path=ffmpeg,
                               recording_rows_between=Mock(return_value=[row]),
                               acquire_recording_for_playback=acquire,
                               release_recording_playback=leases.remove)
    detector = SimpleNamespace(detect_enrichment=Mock(return_value=[{
        "label": "person", "confidence": .9, "box": {"x1": 10, "y1": 10, "x2": 30, "y2": 30},
    }]), detect=Mock(side_effect=AssertionError("raw inference must not run")))
    budget = RecordedDecodeBudget(max_processes=1, memory_budget_bytes=128 << 20)
    runtime = RecordingReviewRuntime(recorder=recorder, detector_provider=lambda: detector,
                                     decode_budget=budget, detector_config_provider=lambda: {"enabled": True})
    events = EventStore(tmp_path / "events")
    service = RecordingReviewService(tmp_path, manifest_provider=runtime.manifest,
                                     analysis_identity=runtime.analysis_identity, analyze=runtime.analyze)
    service.start()
    try:
        assert service.status("archive", "main", 15)["state"] == "unreviewed"
        detector.detect_enrichment.assert_not_called()
        admitted = service.request("archive", "main", 15)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = service.status("archive", "main", 15)
            if result["state"] not in {"queued", "analyzing"}:
                break
            time.sleep(.02)
        assert result["state"] == "sampled", result
        assert result["sample_count"] == 12
        assert len(result["observations"]) == 12
        assert detector.detect_enrichment.call_count == 12
        reopened = service.request("archive", "main", 45)
        assert reopened["request_id"] == admitted["request_id"]
        assert reopened["state"] == "sampled"
        assert detector.detect_enrichment.call_count == 12
        assert str(tmp_path) not in json.dumps(result)
        with events._connect() as connection:
            for table in ("events", "scene_incidents", "scene_analysis_jobs", "scene_notification_outbox"):
                assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    finally:
        service.stop()
    assert not leases
    assert budget.status()["active_processes"] == 0
    assert budget.status()["reserved_bytes"] == 0
