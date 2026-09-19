"""Timestamped recorded-video decoding helpers."""
from __future__ import annotations
import queue
import json
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
import numpy as np
from .video_frames import DecodedVideoFrame, VideoFrameReference

def sampled_video_frames(
    path: Path,
    *,
    start_epoch: float,
    sample_fps: float,
    duration_seconds: float,
    ffmpeg_path: str,
    maximum_width: int = 640,
    start_offset_seconds: float = 0.0,
    concat_input: bool = False,
    probe_path: Path | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Sample frames using ``start_epoch`` as the epoch at the seek point.

    Source PTS remains file-relative; ``start_epoch`` is the wall-clock epoch
    of ``start_offset_seconds``. Trimming occurs before timestamp collection
    so every sampled timestamp corresponds to an emitted image.
    """
    if not ffmpeg_path:
        raise ValueError("ffmpeg_path is required for video frame sampling")
    yield from _ffmpeg_sampled_video_frames(
        path,
        start_epoch=start_epoch,
        sample_fps=sample_fps,
        duration_seconds=duration_seconds,
        ffmpeg_path=ffmpeg_path,
        maximum_width=maximum_width,
        start_offset_seconds=start_offset_seconds,
        concat_input=concat_input,
        probe_path=probe_path,
    )


def _ffmpeg_sampled_video_frames(
    path: Path,
    *,
    start_epoch: float,
    sample_fps: float,
    duration_seconds: float,
    ffmpeg_path: str,
    maximum_width: int,
    start_offset_seconds: float,
    concat_input: bool,
    probe_path: Path | None,
) -> Iterator[DecodedVideoFrame]:
    # ffprobe is isolated and substantially
    # faster under a full camera workload. A constituent file is used when the
    # decoder input itself is an ffconcat manifest.
    source_width, source_height, time_base_num, time_base_den = _ffprobe_video_metadata(
        probe_path or path,
        ffmpeg_path,
    )
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("comparison video dimensions are unavailable")
    output_width = max(2, min(source_width, max(64, int(maximum_width))))
    output_height = max(2, int(round(source_height * output_width / source_width)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    frame_bytes = output_width * output_height * 3
    input_options = ["-f", "concat", "-safe", "0"] if concat_input else []
    duration = max(0.1, float(duration_seconds))
    offset = max(0.0, float(start_offset_seconds))
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "info",
        *input_options,
        "-i", str(path),
        "-vf", (
            f"trim=start={offset:.6f}:end={offset + duration:.6f},"
            f"scale={output_width}:{output_height},showinfo@source,"
            f"fps={max(0.1, float(sample_fps)):.6f},showinfo@sampled,"
            "setpts=PTS-STARTPTS"
        ),
        "-t", f"{duration:.6f}",
        "-fps_mode", "passthrough",
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 2,
    )
    timestamps: queue.Queue[tuple[int, float] | None] = queue.Queue()
    frame_pattern = re.compile(
        r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+).*"
        r"\bchecksum:([0-9A-Fa-f]+)"
    )

    def read_timestamps() -> None:
        stderr = process.stderr
        if stderr is None:
            timestamps.put(None)
            return
        try:
            source_by_checksum: dict[str, list[tuple[int, float]]] = {}
            last_source_time = float("-inf")
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace")
                match = frame_pattern.search(line)
                if not match:
                    continue
                pts = int(match.group(1))
                pts_seconds = float(match.group(2))
                checksum = match.group(3).upper()
                if "showinfo@source" in line:
                    source_by_checksum.setdefault(checksum, []).append(
                        (pts, pts_seconds)
                    )
                    continue
                if "showinfo@sampled" not in line:
                    continue
                candidates = source_by_checksum.get(checksum, [])
                eligible = [
                    item for item in candidates if item[1] >= last_source_time - 1e-9
                ]
                if not eligible:
                    timestamps.put(None)
                    return
                source_pts, source_time = min(
                    eligible,
                    key=lambda item: abs(item[1] - pts_seconds),
                )
                last_source_time = source_time
                candidates.remove((source_pts, source_time))
                timestamps.put((source_pts, source_time))
        finally:
            timestamps.put(None)

    timestamp_thread = threading.Thread(
        target=read_timestamps,
        name="survng-frame-pts",
        daemon=True,
    )
    timestamp_thread.start()
    try:
        if process.stdout is None:
            raise RuntimeError("comparison decoder output is unavailable")
        while True:
            payload = bytearray()
            while len(payload) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError("comparison decoder returned a partial frame")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
            try:
                timestamp = timestamps.get(timeout=5.0)
            except queue.Empty as exc:
                raise RuntimeError("comparison decoder frame timestamp timed out") from exc
            if timestamp is None:
                raise RuntimeError("comparison decoder frame timestamp is unavailable")
            pts, pts_seconds = timestamp
            captured_at = start_epoch + pts_seconds - offset
            yield DecodedVideoFrame(
                captured_at,
                frame,
                VideoFrameReference(
                    source_path=path,
                    seek_offset_seconds=max(0.0, float(start_offset_seconds)),
                    pts=pts,
                    pts_seconds=pts_seconds,
                    time_base_num=time_base_num,
                    time_base_den=time_base_den,
                    captured_at=captured_at,
                ),
            )
        return_code = process.wait(timeout=5.0)
        if return_code != 0:
            raise RuntimeError("comparison video decoder failed")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        timestamp_thread.join(timeout=1.0)
        if process.stderr is not None:
            process.stderr.close()


def _ffprobe_video_metadata(
    path: Path,
    ffmpeg_path: str,
) -> tuple[int, int, int, int]:
    ffprobe_path = str(Path(ffmpeg_path).with_name("ffprobe"))
    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,time_base",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if result.returncode == 0 and isinstance(streams, list) and streams:
            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            time_base = str(stream.get("time_base") or "1/1").split("/", 1)
            time_base_num = int(time_base[0])
            time_base_den = int(time_base[1]) if len(time_base) > 1 else 1
            if width > 0 and height > 0:
                return width, height, time_base_num, max(1, time_base_den)
    except (json.JSONDecodeError, OSError, TypeError, ValueError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError("comparison video dimensions are unavailable")


def video_frame_at_reference(
    reference: VideoFrameReference,
    *,
    ffmpeg_path: str,
    maximum_width: int,
) -> DecodedVideoFrame | None:
    """Re-decode the exact source PTS identified during recorded sampling."""
    if not reference.exact or maximum_width <= 0:
        return None
    source_width, source_height, _time_base_num, _time_base_den = (
        _ffprobe_video_metadata(reference.source_path, ffmpeg_path)
    )
    output_width = max(2, min(source_width, max(64, int(maximum_width))))
    output_height = max(2, int(round(source_height * output_width / source_width)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    command = [
        ffmpeg_path,
        "-nostdin",
        "-v", "error",
        "-i", str(reference.source_path),
        "-vf", (
            f"select='eq(pts\\,{reference.pts})',"
            f"scale={output_width}:{output_height}"
        ),
        "-frames:v", "1",
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    expected_bytes = output_width * output_height * 3
    if result.returncode != 0 or len(result.stdout) != expected_bytes:
        return None
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        (output_height, output_width, 3)
    ).copy()
    return DecodedVideoFrame(reference.captured_at, frame, reference)
