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

from .recorded_decoder import (
    DecodePlan,
    StreamingDecode,
    build_decode_command,
    recorded_decode_plans,
    run_decoder,
    warn_decoder_errors,
    warn_decoder_failure,
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
    duration = max(0.1, float(duration_seconds))
    offset = max(0.0, float(start_offset_seconds))
    seek: str | None = None
    if not concat_input and (probe_path or path) == path and source_start == 0.0 and offset > INPUT_SEEK_MARGIN_SECONDS:
        # The demuxer starts at the preceding keyframe instead of decoding the
        # segment from its beginning. -copyts keeps the source timeline, so
        # trim, showinfo PTS and frame references are unchanged; a container
        # start offset would shift that timeline, so only zero-start files seek.
        seek = f"{offset - INPUT_SEEK_MARGIN_SECONDS:.6f}"
    trim = f"trim=start={offset:.6f}:end={offset + duration:.6f}"
    sampling = (
        f"showinfo@source,fps={max(0.1, float(sample_fps)):.6f},showinfo@sampled"
    )
    plans = recorded_decode_plans(
        hardware_acceleration,
        scale=(output_width, output_height),
    )
    for index, plan in enumerate(plans):
        # Source and sampled checksums must see system-memory frames. Hardware
        # scales before download; software scales only the sampled frames.
        video_filter = (
            f"{trim},{plan.download_filter},{sampling},setpts=PTS-STARTPTS"
            if plan.download_filter
            else f"{trim},{sampling},scale={output_width}:{output_height},setpts=PTS-STARTPTS"
        )
        command = build_decode_command(
            ffmpeg_path,
            path,
            plan=plan,
            loglevel="info",
            concat_input=concat_input,
            seek=seek,
            copyts=seek is not None,
            video_filter=video_filter,
            duration=f"{duration:.6f}",
            passthrough_timestamps=True,
            transport="raw",
        )
        yielded = False
        try:
            for sample in _run_sampled_decoder(
                command,
                plan=plan,
                path=path,
                start_epoch=start_epoch,
                offset=offset,
                start_offset_seconds=start_offset_seconds,
                output_width=output_width,
                output_height=output_height,
                time_base_num=time_base_num,
                time_base_den=time_base_den,
                sample_interval_seconds=1.0 / max(0.1, float(sample_fps)),
            ):
                yielded = True
                yield sample
            return
        except (OSError, RuntimeError) as error:
            if yielded or index == len(plans) - 1:
                raise
            _warn_sampled_hardware_fallback(plan.backend, error)


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


_FRAME_HEAD_PATTERN = re.compile(
    rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:([-+0-9.eE]+)"
)
_FRAME_CHECKSUM_PATTERN = re.compile(rb"\bchecksum:([0-9A-Fa-f]+)")


class _SampledTimestamps:
    """Pair each sampled frame with the source frame it was copied from.

    The fps filter re-times frames, so the sampled PTS is not the source PTS.
    Both showinfo taps report a checksum of the same pixels; the nearest
    not-yet-used source frame with that checksum is the exact reference.

    showinfo writes a frame line in pieces, and a decoder thread reporting
    bitstream damage can write between them. The head (``n: pts: pts_time:``)
    then ends early and the checksum arrives on a later line of the same tap,
    or not at all. Every sampled head that is seen still yields exactly one
    timestamp, in order, so a torn line no longer shifts later images onto a
    neighbour's timestamp; a frame whose checksum was lost is paired by time
    and marked inexact. A head written without its ``showinfo@`` tag cannot
    be attributed to a tap and is still missed.
    """

    def __init__(self, pairing_tolerance_seconds: float) -> None:
        self.queue: queue.Queue[tuple[int, float, bool] | None] = queue.Queue()
        self.inexact = 0
        self.torn = 0
        self._tolerance = max(0.0, float(pairing_tolerance_seconds))
        # [pts, pts_time, checksum or None]; removed once matched by checksum.
        self._sources: list[list] = []
        self._open_source: list | None = None
        self._open_sampled: list | None = None
        self._last_source_time = float("-inf")
        self._ended = False

    def line(self, raw_line: bytes) -> bool:
        if b"showinfo@" not in raw_line:
            return False
        if self._ended:
            return True
        source = b"showinfo@source" in raw_line
        if not source and b"showinfo@sampled" not in raw_line:
            return True
        head = _FRAME_HEAD_PATTERN.search(raw_line)
        checksum_match = _FRAME_CHECKSUM_PATTERN.search(raw_line)
        checksum = checksum_match.group(1).upper() if checksum_match else None
        if head is None:
            # The tail of a torn line completes the head still waiting for it.
            if checksum is None:
                return True
            if source and self._open_source is not None:
                self._open_source[2] = checksum
                self._open_source = None
            elif not source and self._open_sampled is not None:
                self._open_sampled[2] = checksum
                self._resolve_sampled()
            return True
        try:
            frame = [int(head.group(1)), float(head.group(2)), checksum]
        except ValueError:
            return True
        if checksum is None:
            self.torn += 1
        if source:
            self._sources.append(frame)
            self._open_source = frame if checksum is None else None
            return True
        self._resolve_sampled()
        self._open_sampled = frame
        if checksum is not None:
            self._resolve_sampled()
        return True

    def _resolve_sampled(self) -> None:
        sampled, self._open_sampled = self._open_sampled, None
        if sampled is None or self._ended:
            return
        _pts, sampled_time, checksum = sampled
        eligible = [
            item for item in self._sources
            if item[1] >= self._last_source_time - 1e-9
        ]
        exact = [
            item for item in eligible
            if checksum is not None and item[2] == checksum
        ]
        if exact:
            chosen = min(exact, key=lambda item: abs(item[1] - sampled_time))
        else:
            # The checksum on one side was lost. Only source frames that could
            # still be this image qualify: unidentified ones when the sampled
            # checksum is known, any when it is not.
            nearby = [
                item for item in eligible
                if (checksum is None or item[2] is None)
                and abs(item[1] - sampled_time) <= self._tolerance
            ]
            if not nearby:
                self.end()
                return
            chosen = min(nearby, key=lambda item: abs(item[1] - sampled_time))
            self.inexact += 1
            # A guess must not consume the frame a later image can prove is
            # its own, nor move the ordering floor past it.
            self.queue.put((chosen[0], chosen[1], False))
            return
        self._last_source_time = chosen[1]
        self._sources.remove(chosen)
        self.queue.put((chosen[0], chosen[1], True))

    def end(self) -> None:
        if self._ended:
            return
        self._resolve_sampled()
        if not self._ended:
            self._ended = True
            self.queue.put(None)


def _run_sampled_decoder(
    command: list[str],
    *,
    plan: DecodePlan,
    path: Path,
    start_epoch: float,
    offset: float,
    start_offset_seconds: float,
    output_width: int,
    output_height: int,
    time_base_num: int,
    time_base_den: int,
    sample_interval_seconds: float,
) -> Iterator[DecodedVideoFrame]:
    frame_bytes = output_width * output_height * 3
    timestamps = _SampledTimestamps(sample_interval_seconds)
    decoder = StreamingDecode(
        command,
        backend=plan.backend,
        buffer_bytes=frame_bytes * 2,
        on_stderr_line=timestamps.line,
        on_stderr_end=timestamps.end,
    )
    try:
        while True:
            payload = decoder.read_exact(frame_bytes)
            if decoder.stalled:
                break
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise RuntimeError("comparison decoder returned a partial frame")
            frame = np.frombuffer(payload, dtype=np.uint8).reshape((output_height, output_width, 3)).copy()
            try:
                timestamp = timestamps.queue.get(timeout=5.0)
            except queue.Empty as exc:
                raise RuntimeError("comparison decoder frame timestamp timed out") from exc
            if timestamp is None:
                raise RuntimeError("comparison decoder frame timestamp is unavailable")
            pts, pts_seconds, exact = timestamp
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
                    exact=exact,
                ),
            )
        result = decoder.finish()
        if timestamps.inexact:
            LOGGER.debug(
                "sampled decode of %s paired %d frames by time after torn showinfo lines",
                path,
                timestamps.inexact,
            )
        warn_decoder_errors(
            "tracking", path, plan.backend, result.stderr,
            interleaved_lines=timestamps.torn,
        )
        if not result.ok:
            raise RuntimeError(f"comparison video decoder failed ({result.describe()})")
    finally:
        decoder.close()


def _ffprobe_video_metadata(
    path: Path,
    ffmpeg_path: str,
) -> tuple[int, int, int, int, float | None]:
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
    plan = DecodePlan("cpu")
    command = build_decode_command(
        ffmpeg_path,
        reference.source_path,
        plan=plan,
        video_filter=(
            f"select='eq(pts\\,{reference.pts})',"
            f"scale={output_width}:{output_height}"
        ),
        frames=1,
        transport="raw",
    )
    result = run_decoder(command, backend=plan.backend, timeout=10.0)
    warn_decoder_errors("tracking-exact-frame", reference.source_path, plan.backend, result.stderr)
    if not result.ok:
        warn_decoder_failure("tracking-exact-frame", reference.source_path, result)
        return None
    expected_bytes = output_width * output_height * 3
    if len(result.output or b"") != expected_bytes:
        return None
    frame = np.frombuffer(result.output, dtype=np.uint8).reshape(
        (output_height, output_width, 3)
    ).copy()
    return DecodedVideoFrame(reference.captured_at, frame, reference)
