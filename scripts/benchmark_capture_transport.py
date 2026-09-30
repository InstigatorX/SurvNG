#!/usr/bin/env python3
"""Compare live-capture BMP and rawvideo transport cost on one local file.

This is a saturated decode, not a paced RTSP session. Both transports use the
production FfmpegCaptureBackend command, so the difference is BMP encode/parse
versus fixed-size raw BGR delivery.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from survng.app.camera_capture import (
    CaptureOpenLimiter,
    FfmpegCaptureBackend,
    FfmpegCaptureHandle,
    FfmpegCaptureOptions,
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


def _source(path: Path, width: int, height: int, fps: int, duration: float) -> None:
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
            f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            str(path),
        ],
        check=True,
    )


def _run(path: Path, transport: str, frame_rate: float) -> dict[str, float | int]:
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(
            frame_transport=transport,
            frame_rate=lambda: frame_rate,
            read_timeout_ms=2000,
        ),
    )
    handle = backend.create_handle()
    assert isinstance(handle, FfmpegCaptureHandle)
    python_before = _cpu_seconds(os.getpid())
    started = time.perf_counter()
    opened = backend.open(handle, str(path), lambda: False, open_timeout_ms=10000)
    if not opened or handle._process is None:
        raise RuntimeError(f"{transport} capture did not open {path}")
    frames = 0
    payload_bytes = 0
    ffmpeg_rss = 0
    read_ms: list[float] = []
    pid = handle._process.pid
    ffmpeg_cpu = _cpu_seconds(pid) or 0.0
    try:
        while True:
            read_started = time.perf_counter()
            ok, frame = handle.read()
            latest_cpu = _cpu_seconds(pid)
            if latest_cpu is not None:
                ffmpeg_cpu = latest_cpu
            if not ok or frame is None:
                break
            read_ms.append((time.perf_counter() - read_started) * 1000.0)
            frames += 1
            payload_bytes += int(frame.nbytes)
            if frames == 1 or frames % 8 == 0:
                ffmpeg_rss = max(ffmpeg_rss, _rss_bytes(pid))
        elapsed = time.perf_counter() - started
        ffmpeg_rss = max(ffmpeg_rss, _rss_bytes(pid))
    finally:
        handle.close()
    python_cpu = _cpu_seconds(os.getpid()) - python_before
    safe_frames = max(1, frames)
    return {
        "frames": frames,
        "elapsed_s": round(elapsed, 4),
        "ffmpeg_cpu_s": round(ffmpeg_cpu, 4),
        "python_cpu_s": round(python_cpu, 4),
        "ffmpeg_ms_per_frame": round(1000.0 * ffmpeg_cpu / safe_frames, 3),
        "python_ms_per_frame": round(1000.0 * python_cpu / safe_frames, 3),
        "payload_bytes": payload_bytes,
        "ffmpeg_rss_bytes": ffmpeg_rss,
        "python_rss_bytes": _rss_bytes(os.getpid()),
        "read_ms_p50": round(float(np.percentile(read_ms, 50)), 3) if read_ms else 0.0,
        "read_ms_p95": round(float(np.percentile(read_ms, 95)), 3) if read_ms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--frame-rate", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    source = Path("/tmp/survng-capture-transport-bench.mp4")
    _source(source, args.width, args.height, args.fps, args.duration)
    print(
        f"source {args.width}x{args.height} @{args.fps}fps "
        f"duration={args.duration}s select={args.frame_rate}fps"
    )
    for transport in ("bmp", "rawvideo"):
        runs = [
            _run(source, transport, args.frame_rate) for _ in range(args.repeats)
        ]
        ffmpeg_ms = [float(run["ffmpeg_ms_per_frame"]) for run in runs]
        python_ms = [float(run["python_ms_per_frame"]) for run in runs]
        print(
            transport,
            {
                "frames": runs[-1]["frames"],
                "ffmpeg_ms_per_frame_median": round(float(np.median(ffmpeg_ms)), 3),
                "python_ms_per_frame_median": round(float(np.median(python_ms)), 3),
                "payload_bytes": runs[-1]["payload_bytes"],
                "ffmpeg_rss_bytes": max(int(run["ffmpeg_rss_bytes"]) for run in runs),
                "python_rss_bytes": max(int(run["python_rss_bytes"]) for run in runs),
                "runs": runs,
            },
        )


if __name__ == "__main__":
    main()
