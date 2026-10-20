from __future__ import annotations

import base64

from survng.app.config import CameraConfig
from survng.app.ffmpeg_input import (
    ffmpeg_input_args,
    ffmpeg_url_input_args,
    stream_url_without_userinfo,
)
from survng.app.security import redact_secret_text


def test_http_credentials_move_to_authorization_header() -> None:
    args = ffmpeg_url_input_args("http://admin:s3cret@camera.local/video")
    assert "s3cret" not in " ".join(args)
    assert args[-2:] == ["-i", "http://camera.local/video"]
    assert "-headers" in args
    header = args[args.index("-headers") + 1]
    expected = base64.b64encode(b"admin:s3cret").decode("ascii")
    assert header == f"Authorization: Basic {expected}\r\n"


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


def test_stream_url_without_userinfo() -> None:
    assert (
        stream_url_without_userinfo("rtsps://user:pass@[::1]:8554/cam")
        == "rtsps://[::1]:8554/cam"
    )


def test_redact_secret_text_covers_recorder_style_lines() -> None:
    line = "Opening rtsp://admin:hunter2@192.168.1.20/h264Preview_01_main for reading"
    assert "hunter2" not in redact_secret_text(line)
    assert "rtsp://admin:***@192.168.1.20/h264Preview_01_main" in redact_secret_text(line)
