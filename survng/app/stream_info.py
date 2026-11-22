"""Main/live stream size and frame rate for Admin's camera header.

The live stream is always decoded, so its size and rate come from the
capture runtime. The main stream is usually recorded without decoding, so
when the runtime has no measurement its size and rate are read from the
newest closed main recording with ffprobe. Probes are cached per camera and
run outside the manager lock.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0
CACHE_SECONDS = 600.0
FAILURE_CACHE_SECONDS = 60.0


def _ffprobe_for(ffmpeg_path: str) -> str:
    ffmpeg = Path(ffmpeg_path or "ffmpeg")
    sibling = ffmpeg.with_name("ffprobe")
    return str(sibling) if ffmpeg.name == "ffmpeg" and sibling.is_file() else "ffprobe"


def _rate(value: object) -> float:
    numerator, separator, denominator = str(value or "0").partition("/")
    try:
        if separator:
            bottom = float(denominator)
            return float(numerator) / bottom if bottom else 0.0
        return float(numerator)
    except ValueError:
        return 0.0


def probe_video(path: Path, ffmpeg_path: str, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any] | None:
    """Width, height, fps and codec of the first video stream, or None."""
    try:
        result = runner(
            [
                _ffprobe_for(ffmpeg_path), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("could not probe recording stream: %s", redact_secret_text(error)[:200])
        return None
    if result.returncode != 0:
        LOGGER.warning("ffprobe could not read a recording: %s", redact_secret_text(result.stderr or "")[:200])
        return None
    try:
        streams = json.loads(result.stdout or "{}").get("streams") or []
    except json.JSONDecodeError:
        return None
    video = streams[0] if streams else None
    if not isinstance(video, dict):
        return None
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2) if fps > 0 else None,
        "codec": str(video.get("codec_name") or "").lower(),
    }


def runtime_stream(status: dict[str, Any], source: str) -> dict[str, Any] | None:
    """Size and measured rate for a decoded source, or None when not decoded."""
    dimensions = (status.get("stream_dimensions") or {}).get(source) or {}
    width = int(dimensions.get("width") or 0)
    height = int(dimensions.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    fps = float(((status.get("capture_stats") or {}).get(source) or {}).get("fps") or 0.0)
    return {"width": width, "height": height, "fps": round(fps, 2) if fps > 0 else None, "measured": "decoder"}


class RecordingStreamInfo:
    """Per-camera cache of the newest closed main recording's video stream."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, probe: Callable[..., Any] = probe_video) -> None:
        self._clock = clock
        self._probe = probe
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict[str, Any] | None]] = {}

    def main(self, camera_id: str, path: Path | None, ffmpeg_path: str) -> dict[str, Any] | None:
        now = self._clock()
        with self._lock:
            cached = self._cache.get(camera_id)
            if cached is not None and cached[0] > now:
                return cached[1]
        value = None
        if path is not None:
            probed = self._probe(path, ffmpeg_path)
            if probed is not None:
                value = {**probed, "measured": "recording"}
        ttl = CACHE_SECONDS if value is not None else FAILURE_CACHE_SECONDS
        with self._lock:
            self._cache[camera_id] = (now + ttl, value)
        return value
