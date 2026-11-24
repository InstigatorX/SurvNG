"""Frame-by-frame stepping through a recording: which frames exist, and runs of them.

The Timeline steps a paused player one frame at a time. "One frame" is the
next frame the file really holds, not ``t + 1/fps``: cameras run at 19.84 or
20.07 fps and segments start wherever the recorder cut them. The frame times
come from the container's packet index (no decoding), so a whole segment's
list costs a demux, not a decode.

An exact frame in the middle of a GOP can only be decoded from the keyframe
before it. So frames are decoded a run at a time, one GOP (or a bounded slice
of a long one) in one FFmpeg process, and the run is cached: the first step
into a GOP pays for the decode, the steps after it are files on disk.
"""
from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass

# Two segments whose index times are at most this far apart are one
# recording to step through: a frame is about 0.05 s, a real gap is seconds.
FRAME_STEP_CONTIGUOUS_SECONDS = 1.0
# A run never holds more frames than this. Cameras send a keyframe about
# every second (20 frames); a camera with a long GOP has it cut into slices,
# each decoded from the keyframe.
FRAME_RUN_MAX_FRAMES = 30
# Frame times requested back by the page are printed to the microsecond and
# offset by a wall-clock start; this absorbs that rounding. Far below any
# frame interval.
FRAME_TIME_TOLERANCE_SECONDS = 0.0005

_SHOWINFO_PTS = re.compile(rb"Parsed_showinfo[^\r\n]*\bpts_time:\s*([-+0-9.eE]+)")


@dataclass(frozen=True, slots=True)
class SegmentFrames:
    """The presented video frames of one file, in display order.

    ``times`` are seconds from the file's start time (the same origin the
    preview and strip endpoints use); ``keyframes`` are indices into it.
    """

    times: tuple[float, ...]
    keyframes: tuple[int, ...]
    start_time: float = 0.0


def parse_packet_listing(text: str) -> SegmentFrames:
    """Frames from ``ffprobe -show_entries packet=pts_time,flags:format=start_time -of json``.

    Packets the demuxer marks for discard (outside an edit list) are never
    presented and are left out; so are packets without a time.
    """
    try:
        payload = json.loads(text or "{}")
    except json.JSONDecodeError:
        return SegmentFrames((), ())
    try:
        start_time = float((payload.get("format") or {}).get("start_time") or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0
    if not math.isfinite(start_time):
        start_time = 0.0
    frames: dict[float, bool] = {}
    for packet in payload.get("packets") or []:
        flags = str(packet.get("flags") or "")
        if "D" in flags:
            continue
        try:
            pts = float(packet.get("pts_time"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pts):
            continue
        offset = round(pts - start_time, 6)
        frames[offset] = frames.get(offset, False) or flags.startswith("K")
    times = tuple(sorted(frames))
    keyframes = tuple(index for index, time in enumerate(times) if frames[time])
    return SegmentFrames(times, keyframes, start_time)


def frame_index_at(times: tuple[float, ...] | list[float], offset: float) -> int:
    """The frame on screen at ``offset``: the last at or before it, else the first."""
    if not times:
        raise ValueError("no frames")
    return max(0, bisect_right(times, offset + FRAME_TIME_TOLERANCE_SECONDS) - 1)


def frame_run(frames: SegmentFrames, index: int, max_frames: int = FRAME_RUN_MAX_FRAMES) -> tuple[int, int]:
    """``[start, stop)`` frame indices decoded together with frame ``index``.

    A run starts at a keyframe (or a fixed slice of the frames after one) and
    stops before the next keyframe, so a run never decodes another run's
    GOP. A file without keyframe flags is cut into fixed slices from its
    first frame.
    """
    count = len(frames.times)
    if not 0 <= index < count:
        raise IndexError(index)
    keyframes = frames.keyframes or (0,)
    position = bisect_right(keyframes, index) - 1
    gop_start = keyframes[position] if position >= 0 else 0
    gop_stop = keyframes[position + 1] if position + 1 < len(keyframes) else count
    limit = max(1, int(max_frames))
    start = gop_start + ((index - gop_start) // limit) * limit
    return start, min(gop_stop, start + limit)


def showinfo_times(stderr: bytes) -> list[float]:
    """Frame times FFmpeg's ``showinfo`` reported, in output order."""
    return [float(match) for match in _SHOWINFO_PTS.findall(stderr or b"")]


def scaled_size(source: tuple[int, int] | None, width: int) -> tuple[int, int] | None:
    """Even output size at most ``width`` wide, keeping the aspect; never upscaled."""
    if source is None:
        return None
    source_width, source_height = source
    if source_width <= 0 or source_height <= 0:
        return None
    target_width = min(int(width), source_width)
    target_width -= target_width % 2
    target_height = round(source_height * target_width / source_width / 2) * 2
    return max(2, target_width), max(2, target_height)
