from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

from .recording_mp4 import iter_mp4_boxes

# Bump when fragment bytes change so disk and browser caches expire together.
RECORDING_FMP4_VERSION = 8


def event_clip_window(
    configured_before: float,
    configured_after: float,
    before: float | None,
    after: float | None,
) -> tuple[float, float]:
    before_value = float(configured_before if before is None else before)
    after_value = float(configured_after if after is None else after)
    if not math.isfinite(before_value) or not math.isfinite(after_value):
        raise ValueError("event clip window must be finite")
    safe_before = max(0.0, min(before_value, 3600.0))
    safe_after = max(0.0, min(after_value, 3600.0))
    if safe_before + safe_after <= 0:
        raise ValueError("event clip window is empty")
    return safe_before, safe_after


def concatenated_clip_timing(
    rows: list[dict[str, Any]],
    window_start: float,
    window_end: float,
) -> tuple[float, float]:
    if not rows:
        raise ValueError("recording window is empty")
    first_start = float(rows[0]["start_epoch"])
    last_end = float(rows[-1]["end_epoch"])
    local_start = max(0.0, window_start - first_start)
    available_duration = sum(max(0.0, float(row["duration_seconds"])) for row in rows)
    tail_trim = max(0.0, last_end - window_end)
    duration = available_duration - local_start - tail_trim
    if duration <= 0:
        raise ValueError("recording window has no playable duration")
    return local_start, duration


def playback_segment_duration(
    row_start: float,
    row_duration: float,
    window_end: float,
    trim_end: bool = False,
) -> float:
    duration = max(0.1, min(float(row_duration), 300.0))
    if not trim_end:
        return duration
    return max(0.1, min(duration, float(window_end) - float(row_start)))


def _read_exact(handle: Any, size: int) -> bytes:
    """Read ``size`` bytes from an unbuffered handle, which may return fewer per call."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_mp4_box(path: Path, wanted: bytes, max_size: int = 32 * 1024 * 1024) -> bytes:
    try:
        file_size = path.stat().st_size
        # Unbuffered: a buffered handle reads a whole block (up to the NFS
        # read size) for each 16-byte box header.
        with path.open("rb", buffering=0) as handle:
            cursor = 0
            while cursor + 8 <= file_size:
                handle.seek(cursor)
                header_data = handle.read(16)
                if len(header_data) < 8:
                    return b""
                size = struct.unpack_from(">I", header_data, 0)[0]
                box_type = header_data[4:8]
                header_size = 8
                if size == 1:
                    if len(header_data) < 16:
                        return b""
                    size = struct.unpack_from(">Q", header_data, 8)[0]
                    header_size = 16
                elif size == 0:
                    size = file_size - cursor
                if size < header_size or cursor + size > file_size:
                    return b""
                if box_type == wanted:
                    if size > max_size:
                        return b""
                    handle.seek(cursor)
                    return _read_exact(handle, size)
                cursor += size
    except OSError:
        return b""
    return b""


def _media_timescale(data: bytes, payload: int, box_end: int) -> bytes:
    if payload + 4 > box_end:
        return b""
    version = data[payload]
    offset = payload + (20 if version == 1 else 12)
    if offset + 4 > box_end:
        return b""
    return data[offset:offset + 4]


def _descriptor(data: bytes, offset: int, end: int) -> tuple[int, int, int] | None:
    if offset >= end:
        return None
    tag = data[offset]
    cursor = offset + 1
    size = 0
    for _ in range(4):
        if cursor >= end:
            return None
        value = data[cursor]
        cursor += 1
        size = (size << 7) | (value & 0x7F)
        if not value & 0x80:
            return tag, cursor, min(cursor + size, end)
    return None


def _decoder_specific_info(data: bytes, offset: int, end: int) -> bytes:
    descriptor = _descriptor(data, offset, end)
    if descriptor is None:
        return b""
    tag, payload, descriptor_end = descriptor
    if tag == 0x05:
        return data[payload:descriptor_end]
    if tag == 0x03:
        if payload + 3 > descriptor_end:
            return b""
        flags = data[payload + 2]
        cursor = payload + 3
        if flags & 0x80:
            cursor += 2
        if flags & 0x40:
            if cursor >= descriptor_end:
                return b""
            cursor += 1 + data[cursor]
        if flags & 0x20:
            cursor += 2
    elif tag == 0x04:
        if payload + 13 > descriptor_end:
            return b""
        cursor = payload + 13
    else:
        return b""
    while cursor < descriptor_end:
        child = _descriptor(data, cursor, descriptor_end)
        if child is None:
            break
        child_tag, child_payload, child_end = child
        if child_tag == 0x05:
            return data[child_payload:child_end]
        nested = _decoder_specific_info(data, cursor, descriptor_end)
        if nested:
            return nested
        cursor = child_end
    return b""


def _normalized_sample_description(
    data: bytes,
    payload: int,
    box_end: int,
    handler: bytes,
) -> bytes:
    descriptors: list[bytes] = []
    for entry_type, _, entry_payload, entry_end in iter_mp4_boxes(data, payload + 8, box_end):
        if handler == b"vide" and entry_payload + 78 <= entry_end:
            fixed = data[entry_payload + 24:entry_payload + 28]
            child_start = entry_payload + 78
        elif handler == b"soun" and entry_payload + 28 <= entry_end:
            fixed = data[entry_payload + 16:entry_payload + 20] + data[entry_payload + 24:entry_payload + 28]
            version = struct.unpack_from(">H", data, entry_payload + 8)[0]
            child_start = entry_payload + 28 + (16 if version == 1 else 36 if version == 2 else 0)
        else:
            descriptors.append(entry_type + data[entry_payload:entry_end])
            continue
        children: list[bytes] = []
        for child_type, child_start_offset, child_payload, child_end in iter_mp4_boxes(data, child_start, entry_end):
            if child_type == b"btrt":
                continue
            if child_type == b"esds":
                decoder_config = _decoder_specific_info(data, child_payload + 4, child_end)
                children.append(child_type + decoder_config)
            else:
                children.append(data[child_start_offset:child_end])
        descriptors.append(entry_type + fixed + b"".join(children))
    return b"".join(descriptors)


def _track_stream_descriptor(data: bytes, payload: int, box_end: int) -> bytes:
    handler = b""
    timescale = b""
    sample_description_box: tuple[int, int] | None = None
    for box_type, _, child_payload, child_end in iter_mp4_boxes(data, payload, box_end):
        if box_type == b"hdlr" and child_payload + 12 <= child_end:
            handler = data[child_payload + 8:child_payload + 12]
        elif box_type == b"mdhd":
            timescale = _media_timescale(data, child_payload, child_end)
        elif box_type == b"minf":
            for minf_type, _, minf_payload, minf_end in iter_mp4_boxes(data, child_payload, child_end):
                if minf_type != b"stbl":
                    continue
                for stbl_type, stbl_start, _, stbl_end in iter_mp4_boxes(data, minf_payload, minf_end):
                    if stbl_type == b"stsd":
                        sample_description_box = (stbl_start + 8, stbl_end)
                        break
    if not handler or sample_description_box is None:
        return b""
    stsd_payload, stsd_end = sample_description_box
    sample_description = _normalized_sample_description(data, stsd_payload, stsd_end, handler)
    return handler + timescale + sample_description


def _track_video_dimensions(data: bytes, payload: int, box_end: int) -> tuple[int, int] | None:
    handler = b""
    sample_description_box: tuple[int, int] | None = None
    for box_type, _, child_payload, child_end in iter_mp4_boxes(data, payload, box_end):
        if box_type != b"mdia":
            continue
        for mdia_type, _, mdia_payload, mdia_end in iter_mp4_boxes(
            data, child_payload, child_end
        ):
            if mdia_type == b"hdlr" and mdia_payload + 12 <= mdia_end:
                handler = data[mdia_payload + 8:mdia_payload + 12]
            elif mdia_type == b"minf":
                for minf_type, _, minf_payload, minf_end in iter_mp4_boxes(
                    data, mdia_payload, mdia_end
                ):
                    if minf_type != b"stbl":
                        continue
                    for stbl_type, stbl_start, _, stbl_end in iter_mp4_boxes(
                        data, minf_payload, minf_end
                    ):
                        if stbl_type == b"stsd":
                            sample_description_box = (stbl_start + 8, stbl_end)
                            break
    if handler != b"vide" or sample_description_box is None:
        return None
    stsd_payload, stsd_end = sample_description_box
    for _entry_type, _, entry_payload, entry_end in iter_mp4_boxes(
        data, stsd_payload + 8, stsd_end
    ):
        if entry_payload + 28 > entry_end:
            continue
        width, height = struct.unpack_from(">HH", data, entry_payload + 24)
        if width > 0 and height > 0:
            return width, height
    return None


def _track_video_duration(data: bytes, payload: int, box_end: int) -> float | None:
    handler = b""
    scale = 0
    ticks = 0
    for kind, _, start, end in iter_mp4_boxes(data, payload, box_end):
        if kind == b"hdlr" and start + 12 <= end:
            handler = data[start + 8:start + 12]
        elif kind == b"mdhd" and start + 4 <= end:
            version = data[start]
            offset = start + (20 if version == 1 else 12)
            if version not in (0, 1) or offset + 4 > end:
                continue
            scale = struct.unpack_from(">I", data, offset)[0]
        elif kind == b"minf":
            for child, _, child_start, child_end in iter_mp4_boxes(data, start, end):
                if child != b"stbl":
                    continue
                for table, _, table_start, table_end in iter_mp4_boxes(data, child_start, child_end):
                    if table != b"stts" or table_start + 8 > table_end:
                        continue
                    count = struct.unpack_from(">I", data, table_start + 4)[0]
                    if table_start + 8 + count * 8 > table_end:
                        continue
                    # mdhd can include composition preroll; the sample durations
                    # describe the footage carried across the clip boundary.
                    ticks = sum(samples * delta for samples, delta in struct.iter_unpack(
                        ">II", data[table_start + 8:table_start + 8 + count * 8]
                    ))
    return ticks / scale if handler == b"vide" and scale and ticks else None


def _fragmented_video_track(moov: bytes) -> tuple[int, int, int] | None:
    """(track id, timescale, default sample duration) of the video track.

    Only a fragmented file declares ``mvex``; its sample tables are empty and
    the samples are described by the ``moof`` boxes that follow ``moov``.
    """
    tracks: dict[int, tuple[bytes, int]] = {}
    defaults: dict[int, int] = {}
    for box_type, _, payload, box_end in iter_mp4_boxes(moov):
        if box_type != b"moov":
            continue
        for child, _, child_payload, child_end in iter_mp4_boxes(moov, payload, box_end):
            if child == b"mvex":
                for item, _, start, end in iter_mp4_boxes(moov, child_payload, child_end):
                    if item == b"trex" and start + 16 <= end:
                        track_id, _index, duration = struct.unpack_from(">III", moov, start + 4)
                        defaults[track_id] = duration
            elif child == b"trak":
                track_id = 0
                handler = b""
                scale = 0
                for item, _, start, end in iter_mp4_boxes(moov, child_payload, child_end):
                    if item == b"tkhd" and start + 4 <= end:
                        offset = start + (20 if moov[start] == 1 else 12)
                        if offset + 4 <= end:
                            track_id = struct.unpack_from(">I", moov, offset)[0]
                    elif item == b"mdia":
                        for media, _, media_start, media_end in iter_mp4_boxes(moov, start, end):
                            if media == b"hdlr" and media_start + 12 <= media_end:
                                handler = moov[media_start + 8:media_start + 12]
                            elif media == b"mdhd" and media_start + 4 <= media_end:
                                offset = media_start + (20 if moov[media_start] == 1 else 12)
                                if moov[media_start] in (0, 1) and offset + 4 <= media_end:
                                    scale = struct.unpack_from(">I", moov, offset)[0]
                if track_id:
                    tracks[track_id] = (handler, scale)
    if not defaults:
        return None
    for track_id, (handler, scale) in tracks.items():
        if handler == b"vide" and scale:
            return track_id, scale, defaults.get(track_id, 0)
    return None


def _fragment_track_ticks(moof: bytes, track_id: int, default_duration: int) -> int:
    """Sum the sample durations one ``moof`` carries for ``track_id``."""
    ticks = 0
    for box_type, _, payload, box_end in iter_mp4_boxes(moof):
        if box_type != b"moof":
            continue
        for child, _, child_payload, child_end in iter_mp4_boxes(moof, payload, box_end):
            if child != b"traf":
                continue
            fragment_default = default_duration
            selected = False
            for item, _, start, end in iter_mp4_boxes(moof, child_payload, child_end):
                if item == b"tfhd" and start + 8 <= end:
                    flags = struct.unpack_from(">I", moof, start)[0] & 0xFFFFFF
                    selected = struct.unpack_from(">I", moof, start + 4)[0] == track_id
                    offset = start + 8
                    if flags & 0x1:
                        offset += 8
                    if flags & 0x2:
                        offset += 4
                    if flags & 0x8 and offset + 4 <= end:
                        fragment_default = struct.unpack_from(">I", moof, offset)[0]
                elif item == b"trun" and selected and start + 8 <= end:
                    flags = struct.unpack_from(">I", moof, start)[0] & 0xFFFFFF
                    count = struct.unpack_from(">I", moof, start + 4)[0]
                    offset = start + 8
                    if flags & 0x1:
                        offset += 4
                    if flags & 0x4:
                        offset += 4
                    if not flags & 0x100:
                        ticks += count * fragment_default
                        continue
                    stride = 4 * sum(1 for bit in (0x100, 0x200, 0x400, 0x800) if flags & bit)
                    if offset + count * stride > end:
                        # A run that overruns its box is not counted at all.
                        continue
                    ticks += sum(
                        struct.unpack_from(">I", moof, offset + index * stride)[0]
                        for index in range(count)
                    )
    return ticks


def _fragmented_video_duration(path: Path, moov: bytes) -> float | None:
    """Video duration of a fragmented MP4, from the fragments written so far.

    Reads are unbuffered: a 16-byte header per top-level box and each small
    fragment header, never the media data. A fragment FFmpeg has not finished
    writing is not counted, which makes the result the readable length of an
    open file. A layout this does not recognise, or an implausible total from
    damaged headers, yields None rather than a wrong duration.
    """
    track = _fragmented_video_track(moov)
    if track is None:
        return None
    scanned = _scan_fragments(path, track, 0, 0)
    return _fragment_seconds(track, scanned[1]) if scanned is not None else None


def _fragment_seconds(track: tuple[int, int, int], ticks: int) -> float | None:
    duration = ticks / track[1] if ticks else None
    if duration is not None and duration > _MAX_FRAGMENTED_DURATION_SECONDS:
        return None
    return duration


def _scan_fragments(
    path: Path, track: tuple[int, int, int], cursor: int, ticks: int,
) -> tuple[int, int] | None:
    """Count complete fragments from ``cursor`` on: (where to resume, ticks).

    ``cursor`` is a top-level box boundary; the one returned is the first box
    that was not complete (or the end of the file), so a later call on a file
    that has grown reads only what was added.
    """
    track_id, _scale, default_duration = track
    try:
        file_size = path.stat().st_size
        with path.open("rb", buffering=0) as handle:
            while cursor + 8 <= file_size:
                handle.seek(cursor)
                header = handle.read(16)
                if len(header) < 8:
                    break
                size = struct.unpack_from(">I", header, 0)[0]
                header_size = 8
                if size == 1:
                    if len(header) < 16:
                        break
                    size = struct.unpack_from(">Q", header, 8)[0]
                    header_size = 16
                elif size == 0:
                    size = file_size - cursor
                if size < header_size or cursor + size > file_size:
                    break
                if header[4:8] == b"moof":
                    if size > _MAX_FRAGMENT_HEADER_BYTES:
                        return None
                    handle.seek(cursor)
                    moof = _read_exact(handle, size)
                    # Its media data must be complete too.
                    handle.seek(cursor + size)
                    data_header = handle.read(16)
                    if len(data_header) < 8:
                        break
                    data_size = struct.unpack_from(">I", data_header, 0)[0]
                    if data_size == 1 and len(data_header) >= 16:
                        data_size = struct.unpack_from(">Q", data_header, 8)[0]
                    if data_header[4:8] != b"mdat":
                        # Not the moof-then-mdat layout FFmpeg writes here.
                        return None
                    if cursor + size + data_size > file_size:
                        break
                    ticks += _fragment_track_ticks(moof, track_id, default_duration)
                cursor += size
    except OSError:
        return None
    return cursor, ticks


# (video track, where the scan stopped, ticks counted so far)
FragmentScan = tuple[tuple[int, int, int], int, int]


def fragmented_readable_progress(
    path: Path, resume: FragmentScan | None = None,
) -> tuple[float | None, FragmentScan | None]:
    """Video seconds completed so far in a fragmented segment, resumably.

    Not cached by file size: the answer changes every time FFmpeg finishes a
    fragment, and caching each size of a growing file would crowd finished
    segments out of the metadata cache. Instead the caller keeps the returned
    scan for this path and hands it back, and only the fragments added since
    are read. None seconds means no index yet, or not a fragmented file.
    """
    if resume is None:
        moov = _read_mp4_box(path, b"moov")
        track = _fragmented_video_track(moov) if moov else None
        if track is None:
            return None, None
        resume = (track, 0, 0)
    track, cursor, ticks = resume
    try:
        if path.stat().st_size < cursor:
            # Not the file the scan was made on (replaced, or truncated).
            return fragmented_readable_progress(path)
    except OSError:
        return None, None
    scanned = _scan_fragments(path, track, cursor, ticks)
    if scanned is None:
        # Unknown layout or a failed read: nothing to resume from.
        return None, None
    return _fragment_seconds(track, scanned[1]), (track, scanned[0], scanned[1])


def fragmented_readable_seconds(path: Path) -> float | None:
    """``fragmented_readable_progress`` from the start of the file."""
    return fragmented_readable_progress(path)[0]


_MAX_FRAGMENT_HEADER_BYTES = 4 * 1024 * 1024
# Three times the longest configurable segment; damaged headers can claim more.
_MAX_FRAGMENTED_DURATION_SECONDS = 900.0


@lru_cache(maxsize=4096)
def _cached_playback_metadata(path_value: str, modified_ns: int, size: int) -> tuple[str, float | None]:
    del modified_ns, size
    moov = _read_mp4_box(Path(path_value), b"moov")
    if not moov:
        return "", None
    descriptors: list[bytes] = []
    video_duration = None
    for box_type, _, payload, box_end in iter_mp4_boxes(moov):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in iter_mp4_boxes(moov, payload, box_end):
            if child_type != b"trak":
                continue
            for trak_type, _, trak_payload, trak_end in iter_mp4_boxes(moov, child_payload, child_end):
                if trak_type != b"mdia":
                    continue
                if video_duration is None:
                    video_duration = _track_video_duration(moov, trak_payload, trak_end)
                descriptor = _track_stream_descriptor(moov, trak_payload, trak_end)
                if descriptor:
                    descriptors.append(descriptor)
    if video_duration is None:
        video_duration = _fragmented_video_duration(Path(path_value), moov)
    if not descriptors:
        return "", video_duration
    digest = hashlib.sha256()
    for descriptor in descriptors:
        digest.update(struct.pack(">I", len(descriptor)))
        digest.update(descriptor)
    return digest.hexdigest(), video_duration


def mp4_stream_fingerprint(path: Path) -> str:
    return mp4_playback_metadata(path)[0]


def mp4_playback_metadata(path: Path) -> tuple[str, float | None]:
    """Read cached stream identity and video duration from MP4 headers only."""
    try:
        stat = path.stat()
    except OSError:
        return "", None
    return _cached_playback_metadata(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _cached_mp4_video_dimensions(
    path_value: str,
    modified_ns: int,
    size: int,
) -> tuple[int, int] | None:
    del modified_ns, size
    moov = _read_mp4_box(Path(path_value), b"moov")
    if not moov:
        return None
    for box_type, _, payload, box_end in iter_mp4_boxes(moov):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in iter_mp4_boxes(moov, payload, box_end):
            if child_type != b"trak":
                continue
            dimensions = _track_video_dimensions(moov, child_payload, child_end)
            if dimensions is not None:
                return dimensions
    return None


def mp4_video_dimensions(path: Path) -> tuple[int, int] | None:
    """Return cached video dimensions from a finalized MP4, if available."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return _cached_mp4_video_dimensions(
        str(path.resolve()), stat.st_mtime_ns, stat.st_size
    )


def resolve_stream_fingerprints(values: list[str | None]) -> list[str]:
    # Unknown metadata must remain unknown. Borrowing a neighboring segment's
    # fingerprint can make HLS reuse an incompatible init section exactly when
    # a camera changes codec, resolution, or audio parameters.
    return [str(value or "") for value in values]


def hls_map_transition(previous: str | None, current: str, map_uri: str) -> tuple[list[str], str]:
    # A known identical fingerprint can safely reuse the current map. Unknown
    # segments each receive their own map because compatibility is unproven.
    if previous is not None and current and current == previous:
        return [], current
    lines = ["#EXT-X-DISCONTINUITY"] if previous is not None else []
    lines.append(f'#EXT-X-MAP:URI="{map_uri}"')
    return lines, current
