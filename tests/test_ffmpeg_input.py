from __future__ import annotations

from survng.app.config import CameraConfig
from survng.app.ffmpeg_input import ffmpeg_input_args, ffmpeg_url_input_args
from survng.app.security import redact_secret_text


def test_http_userinfo_stays_in_url_so_ffmpeg_can_negotiate_digest() -> None:
    args = ffmpeg_url_input_args(
        "https://admin:s3cret@camera.local:8443/Streaming/channels/101"
    )
    assert args[-2:] == [
        "-i",
        "https://admin:s3cret@camera.local:8443/Streaming/channels/101",
    ]
    assert "-headers" not in args
    assert "Authorization" not in " ".join(args)


def test_rtsp_still_uses_url_userinfo_for_ffmpeg_compat() -> None:
    args = ffmpeg_url_input_args("rtsp://admin:s3cret@camera.local/stream")
    assert args[-2:] == ["-i", "rtsp://admin:s3cret@camera.local/stream"]
    assert "-rtsp_transport" in args
    assert "-headers" not in args


def test_ffmpeg_input_args_for_camera_live_source() -> None:
    camera = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://camera/main",
        live_stream_url="rtsp://camera/sub",
    )
    assert ffmpeg_input_args(camera, "live") == [
        "-fflags",
        "+genpts",
        "-dts_error_threshold",
        "10",
        "-rtsp_transport",
        "tcp",
        "-i",
        "rtsp://camera/sub",
    ]


def test_redact_secret_text_covers_recorder_style_lines() -> None:
    line = "Opening rtsp://admin:hunter2@192.168.1.20/h264Preview_01_main for reading"
    assert "hunter2" not in redact_secret_text(line)
    assert "rtsp://admin:***@192.168.1.20/h264Preview_01_main" in redact_secret_text(line)
