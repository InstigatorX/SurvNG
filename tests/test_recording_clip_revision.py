from types import SimpleNamespace
from pathlib import Path
from fastapi import HTTPException

from survng.app.recording_media_runtime import RecordingMediaRuntime


def test_growing_clip_uses_new_frozen_recording_revision(tmp_path):
    rows = [{"path": "first.mp4", "start_epoch": 120, "end_epoch": 150,
             "duration_seconds": 30, "size_bytes": 50, "modified_at": 1}]
    recorder = SimpleNamespace(recording_rows_between=lambda *args, **kwargs: rows)
    manager = SimpleNamespace(recorder=recorder)
    runtime = RecordingMediaRuntime(SimpleNamespace(get_manager=lambda: manager))
    runtime._event_clip_path = lambda *args, **kwargs: tmp_path / "clip.mp4"
    offline = set()
    def storage_path(value, **kwargs):
        if value in offline:
            raise HTTPException(status_code=404, detail="offline")
        return Path(value)
    runtime._recording_storage_path = storage_path
    inputs = []
    def build(event, **kwargs):
        inputs.append(event["_recording_rows"])
        kwargs["output_path"].write_bytes(b"generated")
    runtime._build_event_clip = build
    event = {"id": 120, "camera_id": "gate", "created_at": "1970-01-01T00:02:00+00:00"}
    first = runtime._ensure_event_clip(event, before=0, after=120)
    assert runtime._ensure_event_clip(event, before=0, after=120) == first
    assert len(inputs) == 1
    rows.append({"path": "second.mp4", "start_epoch": 150, "end_epoch": 240,
                 "duration_seconds": 90, "size_bytes": 100, "modified_at": 2})
    second = runtime._ensure_event_clip(event, before=0, after=120)
    assert first != second
    assert len(inputs[0]) == 1 and len(inputs[1]) == 2
    assert runtime._ensure_event_clip(event, before=0, after=120) == second
    assert len(inputs) == 2
    offline.add("second.mp4")
    assert runtime._ensure_event_clip(event, before=0, after=120) == first
    offline.clear()
    assert runtime._ensure_event_clip(event, before=0, after=120) == second


def test_vaapi_event_clip_decodes_on_the_gpu_and_keeps_a_software_decode_attempt(tmp_path):
    runtime = RecordingMediaRuntime(SimpleNamespace(
        get_config=lambda: SimpleNamespace(ffmpeg_path="ffmpeg"),
    ))
    runtime._ffmpeg_vaapi_info = lambda: {"device": "/dev/dri/renderD129"}
    arguments = ("hevc", tmp_path / "clip.ffconcat", 1.5, 20.0, tmp_path / "clip.mp4")

    hardware = runtime._event_clip_vaapi_command(*arguments)
    software = runtime._event_clip_vaapi_command(*arguments, hardware_decode=False)

    # Decoder options are input options, on the device the probe selected.
    assert hardware[hardware.index("-vaapi_device") + 1] == "/dev/dri/renderD129"
    assert hardware.index("-hwaccel_output_format") < hardware.index("-i")
    assert hardware[hardware.index("-hwaccel_output_format") + 1] == "vaapi"
    assert hardware[hardware.index("-vf") + 1] == "format=nv12|vaapi,hwupload"
    assert "-hwaccel" not in software
    assert software[software.index("-vf") + 1] == "format=nv12,hwupload"
    assert "h264_vaapi" in hardware and "h264_vaapi" in software


def _clip_attempts(tmp_path, monkeypatch, pix_fmt):
    import subprocess

    from survng.app import recording_media_runtime

    source = tmp_path / "first.mp4"
    source.write_bytes(b"recording")
    rows = [{"path": str(source), "start_epoch": 120, "end_epoch": 150, "duration_seconds": 30}]
    runtime = RecordingMediaRuntime(SimpleNamespace(
        get_config=lambda: SimpleNamespace(ffmpeg_path="ffmpeg"),
        get_manager=lambda: SimpleNamespace(),
    ))
    runtime._recording_storage_path = lambda value, **kwargs: Path(value)
    runtime._write_concat_file = lambda *args, **kwargs: tmp_path / "clip.ffconcat"
    runtime._probe_video_stream = lambda _path: ("h264", pix_fmt)
    runtime._event_clip_vaapi_enabled = lambda _codec: True
    runtime._event_clip_qsv_enabled = lambda _codec: False
    runtime._ffmpeg_vaapi_info = lambda: {"device": "/dev/dri/renderD128"}
    attempts = []

    def run(command, **kwargs):
        attempts.append(command)
        # A 20 s clip: hardware decode gets real time with a 30 s floor,
        # the established commands keep their 60 s floor.
        assert kwargs["timeout"] == (30.0 if "-hwaccel" in command else 60.0)
        if "-hwaccel" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        Path(command[-1]).write_bytes(b"clip")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(recording_media_runtime.subprocess, "run", run)
    output = tmp_path / "clip.mp4"
    event = {"id": 1, "camera_id": "gate", "created_at": "1970-01-01T00:02:00+00:00", "_recording_rows": rows}

    runtime._build_event_clip(event, 0, 20, output)

    assert output.read_bytes() == b"clip"
    return attempts


def test_stalled_hardware_event_clip_attempt_falls_back_instead_of_timing_out(tmp_path, monkeypatch):
    attempts = _clip_attempts(tmp_path, monkeypatch, "yuv420p")

    assert len(attempts) == 2 and "-hwaccel" in attempts[0] and "-hwaccel" not in attempts[1]


def test_ten_bit_event_clip_skips_the_hardware_decode_attempt(tmp_path, monkeypatch):
    attempts = _clip_attempts(tmp_path, monkeypatch, "yuv420p10le")

    assert len(attempts) == 1 and "-hwaccel" not in attempts[0]
    assert attempts[0][attempts[0].index("-vf") + 1] == "format=nv12,hwupload"
