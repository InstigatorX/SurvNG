"""Recorded video sampling and exact source-frame retrieval."""
from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .ffmpeg_hw import (
    RECORDED_FRAME_INPUT_THREAD_ARGS,
    RECORDED_FRAME_OUTPUT_THREAD_ARGS,
    dri_render_device,
    qsv_enabled,
    vaapi_enabled,
)
from .video_frames import DecodedVideoFrame, VideoFrameReference

LOGGER = logging.getLogger(__name__)
INPUT_SEEK_MARGIN_SECONDS = 0.25


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
    hardware_acceleration: str | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Sample frames using ``start_epoch`` as the epoch at the seek point.

    Source PTS remains file-relative; ``start_epoch`` is the wall-clock epoch
    of ``start_offset_seconds``. Trimming occurs before timestamp collection
    so every sampled timestamp corresponds to an emitted image.

    ``hardware_acceleration`` follows recorded-frame policy: only explicit
    ``qsv`` or ``vaapi`` decodes and scales on the GPU (through VAAPI).
    Hardware pixels are not bit-identical to the software decode, so callers
    that re-decode an exact frame reference for pixel equality must leave it
    unset. A hardware failure before the first frame retries in software.
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
        hardware_acceleration=hardware_acceleration,
    )


def _sampled_decode_plans(
    hardware_acceleration: str | None,
    output_width: int,
    output_height: int,
) -> list[tuple[str, list[str], str]]:
    """Return (backend, input args, pre-sampling scale filter) attempts.

    Explicit QSV also uses VAAPI here: scale_qsv re-times frames onto a
    1/fps time base, which would break exact source-PTS frame references.
    """
    plans: list[tuple[str, list[str], str]] = []
    if qsv_enabled(hardware_acceleration) or vaapi_enabled(hardware_acceleration):
        plans.append((
            "vaapi",
            [
                "-vaapi_device", dri_render_device(),
                "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi",
            ],
            f"scale_vaapi=w={output_width}:h={output_height}:format=nv12,hwdownload,format=nv12",
        ))
    plans.append(("cpu", [], ""))
    return plans


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
    hardware_acceleration: str | None = None,
) -> Iterator[DecodedVideoFrame]:
    # ffprobe is isolated and substantially
    # faster under a full camera workload. A constituent file is used when the
    # decoder input itself is an ffconcat manifest.
    source_width, source_height, time_base_num, time_base_den, source_start = _ffprobe_video_metadata(
        probe_path or path,
        ffmpeg_path,
    )
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("comparison video dimensions are unavailable")
    output_width = max(2, min(source_width, max(64, int(maximum_width))))
    output_height = max(2, int(round(source_height * output_width / source_width)))
    output_width -= output_width % 2
    output_height -= output_height % 2
    input_options = ["-f", "concat", "-safe", "0"] if concat_input else []
    duration = max(0.1, float(duration_seconds))
    offset = max(0.0, float(start_offset_seconds))
    if not concat_input and (probe_path or path) == path and source_start == 0.0 and offset > INPUT_SEEK_MARGIN_SECONDS:
        # The demuxer starts at the preceding keyframe instead of decoding the
        # segment from its beginning. -copyts keeps the source timeline, so
        # trim, showinfo PTS and frame references are unchanged; a container
        # start offset would shift that timeline, so only zero-start files seek.
        input_options = ["-ss", f"{offset - INPUT_SEEK_MARGIN_SECONDS:.6f}", "-copyts"]
    trim = f"trim=start={offset:.6f}:end={offset + duration:.6f}"
    sampling = (
        f"showinfo@source,fps={max(0.1, float(sample_fps)):.6f},showinfo@sampled"
    )
    plans = _sampled_decode_plans(hardware_acceleration, output_width, output_height)
    for index, (backend, decode_args, hardware_scale) in enumerate(plans):
        # Source and sampled checksums must see system-memory frames. Hardware
        # scales before download; software scales only the sampled frames.
        video_filter = (
            f"{trim},{hardware_scale},{sampling},setpts=PTS-STARTPTS"
            if hardware_scale
            else f"{trim},{sampling},scale={output_width}:{output_height},setpts=PTS-STARTPTS"
        )
        command = [
            ffmpeg_path,
            "-nostdin",
            "-v", "info",
            *RECORDED_FRAME_INPUT_THREAD_ARGS,
            *decode_args,
            *input_options,
            "-i", str(path),
            "-vf", video_filter,
            "-t", f"{duration:.6f}",
            "-fps_mode", "passthrough",
            "-an", "-sn", "-dn",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            *RECORDED_FRAME_OUTPUT_THREAD_ARGS,
            "pipe:1",
        ]
        yielded = False
        try:
            for sample in _run_sampled_decoder(
                command,
                path=path,
                start_epoch=start_epoch,
                offset=offset,
                start_offset_seconds=start_offset_seconds,
                output_width=output_width,
                output_height=output_height,
                time_base_num=time_base_num,
                time_base_den=time_base_den,
            ):
                yielded = True
                yield sample
            return
        except (OSError, RuntimeError) as error:
            if yielded or index == len(plans) - 1:
                raise
            _warn_sampled_hardware_fallback(backend, error)


_hardware_fallback_warned: set[str] = set()
_hardware_fallback_lock = threading.Lock()


def _warn_sampled_hardware_fallback(backend: str, error: BaseException) -> None:
    with _hardware_fallback_lock:
        first = backend not in _hardware_fallback_warned
        _hardware_fallback_warned.add(backend)
    LOGGER.log(
        logging.WARNING if first else logging.DEBUG,
        "%s sampled-frame decode failed; using software decode (%s)",
        backend,
        error,
    )


def _run_sampled_decoder(
    command: list[str],
    *,
    path: Path,
    start_epoch: float,
    offset: float,
    start_offset_seconds: float,
    output_width: int,
    output_height: int,
    time_base_num: int,
    time_base_den: int,
) -> Iterator[DecodedVideoFrame]:
    frame_bytes = output_width * output_height * 3
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
                "-show_entries", "stream=width,height,time_base:format=start_time",
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
            container = payload.get("format")
            try:
                start = float(container.get("start_time")) if isinstance(container, dict) else None
            except (TypeError, ValueError):
                start = None
            if width > 0 and height > 0:
                return width, height, time_base_num, max(1, time_base_den), start
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
    source_width, source_height, _time_base_num, _time_base_den, _start = (
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
        *RECORDED_FRAME_INPUT_THREAD_ARGS,
        "-i", str(reference.source_path),
        "-vf", (
            f"select='eq(pts\\,{reference.pts})',"
            f"scale={output_width}:{output_height}"
        ),
        "-frames:v", "1",
        "-an", "-sn", "-dn",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        *RECORDED_FRAME_OUTPUT_THREAD_ARGS,
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
