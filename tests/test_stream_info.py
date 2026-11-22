from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Response

import survng.app.camera_api_routes as camera_routes
from survng.app.camera_api_routes import CameraApiDependencies, create_camera_api_router
from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.stream_info import RecordingStreamInfo, probe_video, runtime_stream


def _runner(payload: dict, returncode: int = 0):
    def run(command, **kwargs):
        assert kwargs["timeout"] > 0 and kwargs["check"] is False
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload), stderr="boom" if returncode else "")
    return run


def test_probe_reads_size_rate_and_codec() -> None:
    payload = {"streams": [{"codec_name": "HEVC", "width": 3840, "height": 2160, "avg_frame_rate": "30000/1001"}]}
    assert probe_video(Path("/x.mp4"), "ffmpeg", runner=_runner(payload)) == {
        "width": 3840, "height": 2160, "fps": 29.97, "codec": "hevc",
    }
    # avg_frame_rate 0/0 falls back to r_frame_rate.
    fallback = {"streams": [{"width": 640, "height": 360, "avg_frame_rate": "0/0", "r_frame_rate": "15/1"}]}
    assert probe_video(Path("/x.mp4"), "ffmpeg", runner=_runner(fallback))["fps"] == 15.0


@pytest.mark.parametrize("payload, code", [
    ({"streams": []}, 0),
    ({"streams": [{"width": 0, "height": 0}]}, 0),
    ({"streams": [{"width": 640, "height": 360}]}, 1),
])
def test_probe_returns_none_for_unusable_output(payload, code) -> None:
    assert probe_video(Path("/x.mp4"), "ffmpeg", runner=_runner(payload, code)) is None


def test_probe_survives_missing_binary_and_timeout() -> None:
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ffprobe")

    def slow(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 10)

    assert probe_video(Path("/x.mp4"), "ffmpeg", runner=missing) is None
    assert probe_video(Path("/x.mp4"), "ffmpeg", runner=slow) is None


def test_runtime_stream_uses_decoder_measurements() -> None:
    status = {
        "stream_dimensions": {"live": {"width": 1280, "height": 720}},
        "capture_stats": {"live": {"fps": 14.96}},
    }
    assert runtime_stream(status, "live") == {"width": 1280, "height": 720, "fps": 14.96, "measured": "decoder"}
    assert runtime_stream(status, "main") is None


def test_recording_info_is_cached_and_failures_retry_sooner() -> None:
    now = [0.0]
    probe = Mock(side_effect=[None, {"width": 1920, "height": 1080, "fps": 20.0, "codec": "h264"}])
    info = RecordingStreamInfo(clock=lambda: now[0], probe=probe)

    assert info.main("gate", Path("/a.mp4"), "ffmpeg") is None
    now[0] = 30.0
    assert info.main("gate", Path("/a.mp4"), "ffmpeg") is None  # failure cached briefly
    now[0] = 61.0
    assert info.main("gate", Path("/a.mp4"), "ffmpeg")["fps"] == 20.0
    now[0] = 500.0
    assert info.main("gate", Path("/a.mp4"), "ffmpeg")["measured"] == "recording"
    assert probe.call_count == 2
    # No closed recording yet: nothing to probe.
    assert RecordingStreamInfo(probe=probe).main("porch", None, "ffmpeg") is None
    assert probe.call_count == 2


def _route(worker, monkeypatch, probe=None):
    manager = SimpleNamespace(
        workers={"gate": worker},
        camera=lambda camera_id: object() if camera_id == "gate" else None,
    )
    monkeypatch.setattr(camera_routes, "RECORDING_STREAM_INFO", RecordingStreamInfo(probe=probe or Mock(return_value=None)))
    bundle = create_camera_api_router(CameraApiDependencies(
        get_manager=lambda: manager,
        manager_lock=threading.RLock(),
        manager_access=ManagerAccessCoordinator(),
    ))
    return bundle.handlers["stream_info"]


def test_route_prefers_decoded_main_and_falls_back_to_the_recording(monkeypatch) -> None:
    decoded = SimpleNamespace(status=lambda: {
        "stream_dimensions": {"main": {"width": 2560, "height": 1440}, "live": {"width": 640, "height": 360}},
        "capture_stats": {"main": {"fps": 15.0}, "live": {"fps": 15.0}},
    }, latest_closed_main_recording=Mock(), recording_ffmpeg_path=Mock())
    payload = _route(decoded, monkeypatch)("gate", Response())
    assert payload["main"]["measured"] == "decoder" and payload["main"]["width"] == 2560
    decoded.latest_closed_main_recording.assert_not_called()

    probe = Mock(return_value={"width": 3840, "height": 2160, "fps": 20.0, "codec": "h265"})
    recorded = SimpleNamespace(status=lambda: {
        "stream_dimensions": {"live": {"width": 1280, "height": 720}},
        "capture_stats": {"live": {"fps": 15.0}},
    }, latest_closed_main_recording=lambda: Path("/rec/main.mp4"), recording_ffmpeg_path=lambda: "/usr/bin/ffmpeg")
    payload = _route(recorded, monkeypatch, probe)("gate", Response())
    assert payload == {
        "main": {"width": 3840, "height": 2160, "fps": 20.0, "codec": "h265", "measured": "recording"},
        "live": {"width": 1280, "height": 720, "fps": 15.0, "measured": "decoder"},
    }
    probe.assert_called_once_with(Path("/rec/main.mp4"), "/usr/bin/ffmpeg")


def test_route_rejects_unknown_camera(monkeypatch) -> None:
    handler = _route(SimpleNamespace(status=dict), monkeypatch)
    with pytest.raises(HTTPException) as error:
        handler("porch", Response())
    assert error.value.status_code == 404


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg") or not __import__("shutil").which("ffprobe"), reason="ffmpeg not installed")
def test_probe_reads_a_real_recording(tmp_path) -> None:
    clip = tmp_path / "main.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=12:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True, timeout=60,
    )
    assert probe_video(clip, "ffmpeg") == {"width": 320, "height": 180, "fps": 12.0, "codec": "h264"}
