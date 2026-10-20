"""FFmpeg input and timestamp-repair helpers for URL-backed cameras."""

from __future__ import annotations

import base64
from urllib.parse import unquote, urlunsplit, urlsplit

from .config import CameraConfig


def _basic_authorization_header(username: str, password: str) -> str:
    token = base64.b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("ascii")
    return f"Authorization: Basic {token}\r\n"


def stream_url_without_userinfo(url: str) -> str:
    """Return URL with userinfo removed (credentials must be supplied separately)."""
    parsed = urlsplit(url)
    if parsed.username is None and parsed.password is None:
        return url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def ffmpeg_url_input_args(url: str, *, rtsp_transport: str = "tcp") -> list[str]:
    """Build FFmpeg input args, keeping secrets out of argv when the protocol allows.

    HTTP(S) credentials move to an Authorization header so `-i` has no userinfo.
    RTSP/RTSPS still require userinfo in the URL for current FFmpeg builds; callers
    should prefer a credential-free go2rtc restream for those cameras.
    """
    parsed = urlsplit(str(url or "").strip())
    scheme = parsed.scheme.lower()
    args: list[str] = [
        "-fflags",
        "+genpts",
        "-dts_error_threshold",
        "10",
    ]
    username = unquote(parsed.username) if parsed.username is not None else None
    password = unquote(parsed.password) if parsed.password is not None else None
    input_url = str(url or "").strip()
    if scheme in {"http", "https"} and (username is not None or password is not None):
        input_url = stream_url_without_userinfo(input_url)
        args.extend(
            [
                "-headers",
                _basic_authorization_header(username or "", password or ""),
            ]
        )
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
