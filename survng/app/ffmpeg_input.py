"""FFmpeg input and timestamp-repair helpers for URL-backed cameras."""

from __future__ import annotations

from urllib.parse import urlsplit

from .config import CameraConfig


def ffmpeg_url_input_args(url: str, *, rtsp_transport: str = "tcp") -> list[str]:
    """Build FFmpeg input args for a camera URL.

    HTTP(S) and RTSP userinfo stay in the URL. FFmpeg answers a Digest
    challenge only from that userinfo (``ff_http_auth_create_response`` returns
    nothing when the URL has no ``user:pass``). A pre-set ``Authorization:
    Basic`` header cannot be upgraded after ``401``, so cameras that require
    Digest never open. Prefer a credential-free go2rtc restream when process
    arguments must not contain secrets.
    """
    parsed = urlsplit(str(url or "").strip())
    scheme = parsed.scheme.lower()
    args: list[str] = [
        "-fflags",
        "+genpts",
        "-dts_error_threshold",
        "10",
    ]
    input_url = str(url or "").strip()
    if scheme in {"rtsp", "rtsps"}:
        args.extend(["-rtsp_transport", rtsp_transport])
    args.extend(["-i", input_url])
    return args


def ffmpeg_input_args(camera: CameraConfig, source: str) -> list[str]:
    return ffmpeg_url_input_args(camera.source_url(source))


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
