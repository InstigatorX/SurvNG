"""FFmpeg input and timestamp-repair helpers for URL-backed cameras."""

from __future__ import annotations

from .config import CameraConfig


def ffmpeg_input_args(camera: CameraConfig, source: str) -> list[str]:
    return [
        "-fflags",
        "+genpts",
        "-dts_error_threshold",
        "10",
        "-rtsp_transport",
        "tcp",
        "-i",
        camera.source_url(source),
    ]


def ffmpeg_timestamp_repair_args(camera: CameraConfig) -> list[str]:
    del camera  # Signature kept for call-site compatibility.
    missing_pts = (
        "if(eq(PTS\\,NOPTS)\\,"
        "if(eq(DTS\\,NOPTS)\\,"
        "if(eq(PREV_OUTPTS\\,NOPTS)\\,0\\,PREV_OUTPTS+max(DURATION\\,1))\\,DTS)\\,PTS)"
    )
    missing_dts = (
        "if(eq(DTS\\,NOPTS)\\,"
        "if(eq(PTS\\,NOPTS)\\,"
        "if(eq(PREV_OUTDTS\\,NOPTS)\\,0\\,PREV_OUTDTS+max(DURATION\\,1))\\,PTS)\\,DTS)"
    )
    return ["-bsf:v", f"setts=pts={missing_pts}:dts={missing_dts}"]
