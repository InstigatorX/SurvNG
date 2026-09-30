#!/usr/bin/env python3
"""Compare the FFmpeg live capture path with the experimental GStreamer path.

Both decode a local H.264 file as fast as the rate cap allows. This is not a
paced RTSP session. FFmpeg remains the default; the numbers decide whether
the GStreamer backend should replace it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.app.camera_capture import (
    CaptureOpenLimiter,
    FfmpegCaptureBackend,
    FfmpegCaptureHandle,
    FfmpegCaptureOptions,
)
from survng.app.gstreamer_capture import (
    GStreamerCaptureBackend,
    GStreamerCaptureHandle,
    GStreamerCaptureOptions,
)


def _cpu_seconds(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[11]) + int(fields[12])) / ticks


def _rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _source(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=20:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "20",
            str(path),
        ],
        check=True,
    )


def _drain(handle, pid: int) -> dict[str, float | int]:
    frames = 0
    payload = 0
    read_ms: list[float] = []
    rss = 0
    child_cpu = 0.0
    python_before = _cpu_seconds(os.getpid()) or 0.0
    started = time.perf_counter()
    try:
        while frames < 80:
            read_started = time.perf_counter()
            ok, frame = handle.read()
            latest = _cpu_seconds(pid)
            if latest is not None:
                child_cpu = latest
            if not ok or frame is None:
                break
            read_ms.append((time.perf_counter() - read_started) * 1000.0)
            frames += 1
            payload += int(frame.nbytes)
            rss = max(rss, _rss_bytes(pid))
    finally:
        handle.close()
    elapsed = time.perf_counter() - started
    python_cpu = (_cpu_seconds(os.getpid()) or 0.0) - python_before
    safe = max(1, frames)
    return {
        "frames": frames,
        "elapsed_s": round(elapsed, 3),
        "child_ms_per_frame": round(1000.0 * child_cpu / safe, 3),
        "python_ms_per_frame": round(1000.0 * python_cpu / safe, 3),
        "read_ms_p95": round(float(np.percentile(read_ms, 95)), 3) if read_ms else 0.0,
        "payload_bytes": payload,
        "child_rss_bytes": rss,
    }


def main() -> None:
    source = Path("/tmp/survng-gstreamer-capture-bench.mp4")
    _source(source)
    summaries = []
    for _repeat in range(3):
        ffmpeg = FfmpegCaptureBackend(
            CaptureOpenLimiter(1),
            FfmpegCaptureOptions(
                frame_transport="rawvideo",
                frame_rate=lambda: 10.0,
                read_timeout_ms=2000,
                hardware_acceleration="off",
            ),
        )
        ffmpeg_handle = ffmpeg.create_handle()
        assert isinstance(ffmpeg_handle, FfmpegCaptureHandle)
        if not ffmpeg.open(ffmpeg_handle, str(source), lambda: False, open_timeout_ms=10000):
            raise RuntimeError(ffmpeg_handle.error_detail())
        assert ffmpeg_handle._process is not None
        summaries.append(("ffmpeg", _drain(ffmpeg_handle, ffmpeg_handle._process.pid)))

        gstreamer = GStreamerCaptureBackend(
            CaptureOpenLimiter(1),
            GStreamerCaptureOptions(
                frame_rate=lambda: 10.0,
                read_timeout_ms=2000,
                open_timeout_ms=10000,
            ),
        )
        gst_handle = gstreamer.create_handle()
        assert isinstance(gst_handle, GStreamerCaptureHandle)
        if not gstreamer.open(gst_handle, str(source), lambda: False):
            raise RuntimeError(gst_handle.error_detail())
        assert gst_handle._process is not None
        summaries.append(("gstreamer", _drain(gst_handle, gst_handle._process.pid)))
    for name, row in summaries:
        print(name, row)


if __name__ == "__main__":
    main()
