"""Read MP4 box boundaries and adjust timestamps in staged playback fragments."""
from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from pathlib import Path


def iter_mp4_boxes(
    data: bytes | bytearray | mmap.mmap, start: int = 0, end: int | None = None,
) -> Iterator[tuple[bytes, int, int, int]]:
    """Yield each box type and the offsets of its start, payload, and end."""
    limit = len(data) if end is None else min(end, len(data))
    cursor = start
    while cursor + 8 <= limit:
        size = struct.unpack_from(">I", data, cursor)[0]
        box_type = bytes(data[cursor + 4:cursor + 8])
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


def mp4_track_timescales(init_data: bytes | bytearray) -> dict[int, int]:
    timescales: dict[int, int] = {}
    for box_type, _, payload, box_end in iter_mp4_boxes(init_data):
        if box_type != b'moov':
            continue
        for child_type, _, child_payload, child_end in iter_mp4_boxes(init_data, payload, box_end):
            if child_type != b'trak':
                continue
            track_id = None
            timescale = None
            for trak_type, _, trak_payload, trak_end in iter_mp4_boxes(init_data, child_payload, child_end):
                if trak_type == b'tkhd':
                    version = init_data[trak_payload]
                    offset = trak_payload + (20 if version == 1 else 12)
                    if offset + 4 <= trak_end:
                        track_id = struct.unpack_from('>I', init_data, offset)[0]
                elif trak_type == b'mdia':
                    for mdia_type, _, mdia_payload, mdia_end in iter_mp4_boxes(init_data, trak_payload, trak_end):
                        if mdia_type != b'mdhd':
                            continue
                        version = init_data[mdia_payload]
                        offset = mdia_payload + (20 if version == 1 else 12)
                        if offset + 4 <= mdia_end:
                            timescale = struct.unpack_from('>I', init_data, offset)[0]
            if track_id and timescale:
                timescales[track_id] = timescale
    return timescales

def _fmp4_start_delays(init_data: bytearray, timescales: dict[int, int]) -> dict[int, int]:
    """Move positive track delays out of per-clip init edit lists."""
    delays = {}
    for kind, _, payload, end in iter_mp4_boxes(init_data):
        if kind != b'moov':
            continue
        children = list(iter_mp4_boxes(init_data, payload, end))
        movie_scale = 0
        for child, _, start, stop in children:
            if child == b'mvhd':
                offset = start + (20 if init_data[start] == 1 else 12)
                if offset + 4 <= stop:
                    movie_scale = struct.unpack_from('>I', init_data, offset)[0]
        if not movie_scale:
            continue
        for child, _, start, stop in children:
            if child != b'trak':
                continue
            track = None
            edits = None
            for entry, box_start, a, b in iter_mp4_boxes(init_data, start, stop):
                if entry == b'tkhd':
                    offset = a + (20 if init_data[a] == 1 else 12)
                    if offset + 4 <= b:
                        track = struct.unpack_from('>I', init_data, offset)[0]
                elif entry == b'edts':
                    edits = (box_start, a, b)
            if track not in timescales or edits is None:
                continue
            box_start, a, b = edits
            for entry, _, edit_start, edit_end in iter_mp4_boxes(init_data, a, b):
                if entry != b'elst' or edit_start + 8 > edit_end:
                    continue
                version = init_data[edit_start]
                count = struct.unpack_from('>I', init_data, edit_start + 4)[0]
                fmt = '>Qqhh' if version == 1 else '>Iihh'
                size = struct.calcsize(fmt)
                if version not in (0, 1) or count not in (1, 2) or edit_start + 8 + count * size > edit_end:
                    continue
                entries = [struct.unpack_from(fmt, init_data, edit_start + 8 + i * size) for i in range(count)]
                # Keep decoder preroll and nontrivial edits intact. The HLS
                # muxer's positive-delay form is an empty edit followed by
                # an unbounded, unshifted media edit at normal speed.
                if entries[-1] != (0, 0, 1, 0):
                    continue
                if count == 2 and entries[0][1:] != (-1, 1, 0):
                    continue
                delay = entries[0][0] if count == 2 else 0
                delays[track] = round(delay * timescales[track] / movie_scale)
                # Retain byte positions; the media now carries this delay.
                init_data[box_start + 4:box_start + 8] = b'free'
    return delays

def offset_fmp4_timestamps(init_path: Path, media_path: Path, seconds: float) -> None:
    """Shift a staged fragment in place before it is published to the playback cache."""
    original_init = init_path.read_bytes()
    init_data = bytearray(original_init)
    timescales = mp4_track_timescales(init_data)
    if not timescales:
        raise RuntimeError('fragment init has no track timescales')
    start_delays = _fmp4_start_delays(init_data, timescales)
    adjusted = 0
    with media_path.open('r+b') as media_file, mmap.mmap(media_file.fileno(), 0) as data:
        for box_type, _, payload, box_end in iter_mp4_boxes(data):
            if box_type == b'sidx':
                # The seek index and media must describe the same timeline.
                # Its timescale is independent of the track's mdhd timescale;
                # retain the original presentation/decode timing difference.
                if payload + 4 > box_end or data[payload] not in (0, 1):
                    raise RuntimeError('fragment has an invalid sidx version')
                version = data[payload]
                if payload + (32 if version == 1 else 24) > box_end:
                    raise RuntimeError('fragment has a truncated sidx box')
                timescale = struct.unpack_from('>I', data, payload + 8)[0]
                if not timescale:
                    raise RuntimeError('fragment sidx has no timescale')
                value_format = '>Q' if version == 1 else '>I'
                value_offset = payload + 12
                current = struct.unpack_from(value_format, data, value_offset)[0]
                next_value = current + round(seconds * timescale)
                if next_value >= 1 << (64 if version == 1 else 32):
                    raise RuntimeError(f'fragment timestamp exceeds version {version} sidx')
                struct.pack_into(value_format, data, value_offset, next_value)
                continue
            if box_type != b'moof':
                continue
            for child_type, _, child_payload, child_end in iter_mp4_boxes(data, payload, box_end):
                if child_type != b'traf':
                    continue
                track_id = None
                tfdt = None
                for traf_type, _, traf_payload, traf_end in iter_mp4_boxes(data, child_payload, child_end):
                    if traf_type == b'tfhd' and traf_payload + 8 <= traf_end:
                        track_id = struct.unpack_from('>I', data, traf_payload + 4)[0]
                    elif traf_type == b'tfdt':
                        tfdt = (traf_payload, traf_end)
                if not track_id or not tfdt or track_id not in timescales:
                    continue
                tfdt_payload, tfdt_end = tfdt
                version = data[tfdt_payload]
                value_offset = tfdt_payload + 4
                increment = round(seconds * timescales[track_id]) + start_delays.get(track_id, 0)
                if version == 1 and value_offset + 8 <= tfdt_end:
                    current = struct.unpack_from('>Q', data, value_offset)[0]
                    struct.pack_into('>Q', data, value_offset, current + increment)
                    adjusted += 1
                elif version == 0 and value_offset + 4 <= tfdt_end:
                    current = struct.unpack_from('>I', data, value_offset)[0]
                    next_value = current + increment
                    if next_value > 4294967295:
                        raise RuntimeError('fragment timestamp exceeds version 0 tfdt')
                    struct.pack_into('>I', data, value_offset, next_value)
                    adjusted += 1
        data.flush()
    if not adjusted:
        raise RuntimeError('fragment has no adjustable tfdt boxes')
    if init_data != original_init:
        init_path.write_bytes(init_data)
