"""Decode recorded frames a GOP run at a time, for any consumer.

A run is the frames ``[start, stop)`` of one file, decoded once from the
keyframe before ``start`` (see ``recording_frame_steps.frame_run``). Timeline
frame stepping encodes them for the browser; face capture searches them. Both
go through the shared recorded-frame decoder for thread limits, process
handling and failure classes.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .recorded_decoder import (
    FAILURE_BITSTREAM,
    FAILURE_MISSING_INPUT,
    FAILURE_OTHER,
    FAILURE_SPAWN,
    FAILURE_TIMEOUT,
    FAILURE_UNFINALIZED_INPUT,
    FAILURE_UNREADABLE_INPUT,
    DecodePlan,
    build_decode_command,
    read_bmp_frames,
    run_decoder,
    warn_decoder_errors,
    warn_decoder_failure,
)
from .recording_frame_steps import (
    FRAME_TIME_TOLERANCE_SECONDS,
    SegmentFrames,
    parse_packet_listing,
    showinfo_times,
)
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)

# Failures that belong to the file, not to the decoder: another attempt on
# the CPU may still read it, and the GPU stays in use for other files.
STREAM_FAILURES = frozenset({
    FAILURE_BITSTREAM, FAILURE_MISSING_INPUT, FAILURE_UNFINALIZED_INPUT, FAILURE_UNREADABLE_INPUT,
})
FRAME_LIST_TIMEOUT_SECONDS = 8.0


class FrameRunError(RuntimeError):
    """No frames: ``failure`` is the recorded-decoder failure class."""

    def __init__(self, failure: str, message: str, *, hardware_failed: bool = False) -> None:
        super().__init__(message)
        self.failure = failure
        self.hardware_failed = hardware_failed


@dataclass(frozen=True, slots=True)
class FrameRun:
    """Decoded frames as (seconds from the file's start time, BGR image)."""

    frames: list[tuple[float, np.ndarray]]
    # A hardware attempt failed for a reason other than the file; callers
    # that share a GPU back off from it for a while.
    hardware_failed: bool = False


def raw_bgr_frames(stream, size: tuple[int, int]) -> list[np.ndarray]:
    """Whole bgr24 frames of ``size`` from a raw decoder pipe; a partial tail is dropped."""
    width, height = size
    frame_bytes = width * height * 3
    frames: list[np.ndarray] = []
    while True:
        chunks: list[bytes] = []
        remaining = frame_bytes
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            return frames
        frames.append(np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape((height, width, 3)))


def list_segment_frames(ffprobe_path: str, source_path: Path) -> SegmentFrames:
    """The frames one recording file presents, from its packet index (a demux, not a decode)."""
    command = [
        ffprobe_path, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags:format=start_time", "-of", "json", str(source_path),
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=FRAME_LIST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameRunError(FAILURE_TIMEOUT, f"frame list of {source_path.name} timed out") from exc
    except OSError as exc:
        raise FrameRunError(FAILURE_SPAWN, f"frame list of {source_path.name} failed to start") from exc
    frames = parse_packet_listing((result.stdout or b"").decode("utf-8", errors="replace"))
    if result.returncode != 0 or not frames.times:
        error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise FrameRunError(
            FAILURE_OTHER,
            f"frame list of {source_path.name} failed: {redact_secret_text(error[-300:]) or 'no video frames'}",
        )
    return frames


def decode_frame_run(
    ffmpeg_path: str,
    source_path: Path,
    frames: SegmentFrames,
    run_start: int,
    run_stop: int,
    width: int,
    *,
    size: tuple[int, int] | None,
    plans: tuple[DecodePlan, ...],
    consumer: str,
    hardware_timeout: float,
    cpu_timeout: float,
) -> FrameRun:
    """Decode frames ``[run_start, run_stop)`` of ``source_path`` once.

    ``plans`` are tried in order; a hardware plan is followed by the CPU.
    With ``size`` frames come back at that size as raw bgr24; without it the
    CPU scales them to at most ``width`` and they travel as BMP.
    """
    count = run_stop - run_start
    # Accurate seek: decoding starts at the keyframe before this time and
    # frames before it are dropped. FFmpeg seeks from the file's start time,
    # and -copyts -start_at_zero reports each frame's PTS from it too: the
    # same origin as the frame list.
    seek = f"{max(0.0, frames.times[run_start] - FRAME_TIME_TOLERANCE_SECONDS):.6f}"
    result = None
    hardware_failed = False
    for plan in plans:
        if plan.backend == "cpu":
            scale = f"scale={size[0]}:{size[1]}" if size else f"scale='min({width},iw)':-2"
            video_filter = f"{scale},showinfo"
        else:
            video_filter = f"{plan.download_filter},showinfo"
        command = build_decode_command(
            ffmpeg_path, source_path, plan=plan, loglevel="info", seek=seek, copyts=True,
            start_at_zero=True, first_video_stream=True, video_filter=video_filter,
            passthrough_timestamps=True, frames=count,
            # Raw frames of a known size skip a BMP encode and decode each.
            transport="raw" if size else "bmp",
        )
        on_gpu = plan.backend != "cpu"
        result = run_decoder(
            command, backend=plan.backend,
            # A run takes well under a second on the GPU; one that takes this
            # long is hung, and the CPU still has its own budget.
            timeout=hardware_timeout if on_gpu else cpu_timeout,
            stdout_reader=(lambda stream: raw_bgr_frames(stream, size)) if size else read_bmp_frames,
        )
        warn_decoder_errors(consumer, source_path, plan.backend, result.stderr)
        if result.ok and result.output and (not on_gpu or len(result.output) >= count):
            break
        if on_gpu:
            if result.ok or result.failure in STREAM_FAILURES:
                # Frames missing, or a problem with this file: the CPU tries
                # it, and the GPU stays in use for other files.
                warn_decoder_failure(consumer, source_path, result)
                continue
            hardware_failed = True
            LOGGER.warning(
                "%s frame run: hardware decode failed for %s, using the CPU: %s",
                consumer, source_path.name, redact_secret_text(result.describe()),
            )
            continue
        warn_decoder_failure(consumer, source_path, result)
    if result is None or not result.ok or not result.output:
        raise FrameRunError(
            result.failure if result is not None else FAILURE_OTHER,
            f"frame run of {source_path.name} failed",
            hardware_failed=hardware_failed,
        )
    decoded = list(zip(showinfo_times(result.stderr), result.output))
    return FrameRun(decoded, hardware_failed)
