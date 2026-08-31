from __future__ import annotations

import hashlib
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any


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


def _boxes(data: bytes, start: int = 0, end: int | None = None):
    limit = len(data) if end is None else min(end, len(data))
    cursor = start
    while cursor + 8 <= limit:
        size = struct.unpack_from(">I", data, cursor)[0]
        box_type = data[cursor + 4:cursor + 8]
        header = 8
        if size == 1 and cursor + 16 <= limit:
            size = struct.unpack_from(">Q", data, cursor + 8)[0]
            header = 16
        elif size == 0:
            size = limit - cursor
        if size < header or cursor + size > limit:
            break
        yield box_type, cursor, cursor + header, cursor + size
        cursor += size


def _read_mp4_box(path: Path, wanted: bytes, max_size: int = 32 * 1024 * 1024) -> bytes:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
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
                    return handle.read(size)
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
    for entry_type, _, entry_payload, entry_end in _boxes(data, payload + 8, box_end):
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
        for child_type, child_start_offset, child_payload, child_end in _boxes(data, child_start, entry_end):
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
    for box_type, _, child_payload, child_end in _boxes(data, payload, box_end):
        if box_type == b"hdlr" and child_payload + 12 <= child_end:
            handler = data[child_payload + 8:child_payload + 12]
        elif box_type == b"mdhd":
            timescale = _media_timescale(data, child_payload, child_end)
        elif box_type == b"minf":
            for minf_type, _, minf_payload, minf_end in _boxes(data, child_payload, child_end):
                if minf_type != b"stbl":
                    continue
                for stbl_type, stbl_start, _, stbl_end in _boxes(data, minf_payload, minf_end):
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
    for box_type, _, child_payload, child_end in _boxes(data, payload, box_end):
        if box_type != b"mdia":
            continue
        for mdia_type, _, mdia_payload, mdia_end in _boxes(
            data, child_payload, child_end
        ):
            if mdia_type == b"hdlr" and mdia_payload + 12 <= mdia_end:
                handler = data[mdia_payload + 8:mdia_payload + 12]
            elif mdia_type == b"minf":
                for minf_type, _, minf_payload, minf_end in _boxes(
                    data, mdia_payload, mdia_end
                ):
                    if minf_type != b"stbl":
                        continue
                    for stbl_type, stbl_start, _, stbl_end in _boxes(
                        data, minf_payload, minf_end
                    ):
                        if stbl_type == b"stsd":
                            sample_description_box = (stbl_start + 8, stbl_end)
                            break
    if handler != b"vide" or sample_description_box is None:
        return None
    stsd_payload, stsd_end = sample_description_box
    for _entry_type, _, entry_payload, entry_end in _boxes(
        data, stsd_payload + 8, stsd_end
    ):
        if entry_payload + 28 > entry_end:
            continue
        width, height = struct.unpack_from(">HH", data, entry_payload + 24)
        if width > 0 and height > 0:
            return width, height
    return None


@lru_cache(maxsize=4096)
def _cached_stream_fingerprint(path_value: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    moov = _read_mp4_box(Path(path_value), b"moov")
    if not moov:
        return ""
    descriptors: list[bytes] = []
    for box_type, _, payload, box_end in _boxes(moov):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in _boxes(moov, payload, box_end):
            if child_type != b"trak":
                continue
            for trak_type, _, trak_payload, trak_end in _boxes(moov, child_payload, child_end):
                if trak_type != b"mdia":
                    continue
                descriptor = _track_stream_descriptor(moov, trak_payload, trak_end)
                if descriptor:
                    descriptors.append(descriptor)
    if not descriptors:
        return ""
    digest = hashlib.sha256()
    for descriptor in descriptors:
        digest.update(struct.pack(">I", len(descriptor)))
        digest.update(descriptor)
    return digest.hexdigest()


def mp4_stream_fingerprint(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    return _cached_stream_fingerprint(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


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
    for box_type, _, payload, box_end in _boxes(moov):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in _boxes(moov, payload, box_end):
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
