"""Served-entry records that keep growing day playlists append-only.

A growing (``EVENT``) day playlist is re-read by the player every target
duration or so. The playlist is rebuilt from the recording index each time,
and the index can change under it: a transient storage error hides a row,
index validation marks one unplayable, or a duration first listed as nominal
is corrected. HLS only allows a server to append to such a playlist (RFC 8216
section 6.2.1), so the route keeps, per playlist URL, the entries it has
already served and only appends to them.

The record is in-process and bounded:

* It is keyed by the playlist request itself (camera, source, window, start
  offset and the page's ``reload`` token). The player re-reads one URL; a page
  that loads the window again sends a new token and so gets a fresh playlist
  that matches the rows the page has just read.
* It lives with the router, not the manager, so it survives a manager reload.
  It does not survive a process restart: the next re-read is then a first read.
* Only windows up to ``GROWING_MAX_WINDOW_SECONDS`` are tracked. A record
  expires ``GROWING_RECORD_LINGER_SECONDS`` after its window stops growing;
  the re-read that ends the playlist marks it finished, and retries get the
  same finished playlist until then. The least recently read records are
  evicted when there are more than ``max_records`` or their entries exceed
  ``max_entries`` in total; an evicted record makes the next re-read a first
  read, which is logged.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable
from urllib.parse import quote

from .recording_media import RECORDING_FMP4_VERSION, hls_map_transition

GROWING_RECORD_MAX_RECORDS = 256
GROWING_RECORD_MAX_ENTRIES = 16_384
# An unreadable row that is new to a growing playlist holds back the rows
# after it for this long (it is usually an NFS hiccup) and is then skipped, so
# one file that stays unreadable cannot stop the playlist growing.
GROWING_HOLD_SECONDS = 30.0
# How long a record outlives the time its window can still grow, so the re-read
# that finishes the playlist is built from what was served.
GROWING_RECORD_LINGER_SECONDS = 120.0
# Only growing playlists of windows up to this long are tracked (the page
# plays 15-minute windows); longer ones are rebuilt on every read.
GROWING_MAX_WINDOW_SECONDS = 3600.0
GROWING_EVICTED_KEYS = 1024
GROWING_LOG_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class GrowingPlaylistKey:
    camera_id: str
    source: str
    start_epoch: str
    end_epoch: str
    start: str
    reload: str

    def window(self) -> tuple[str, str, str, str]:
        return (self.camera_id, self.source, self.start_epoch, self.end_epoch)


@dataclass(frozen=True, slots=True)
class ServedEntry:
    name: str
    start_epoch: float
    duration_seconds: float
    fragment_id: str
    media_offset: float
    lines: tuple[str, ...]


@dataclass(slots=True)
class GrowingPlaylistRecord:
    key: GrowingPlaylistKey
    header: tuple[str, ...]
    query: str
    target_duration: int
    expires_at: float
    entries: list[ServedEntry] = field(default_factory=list)
    by_name: dict[str, ServedEntry] = field(default_factory=dict)
    media_end: float = 0.0
    previous_fingerprint: str | None = None
    # Unreadable new rows: name -> monotonic time first seen unreadable.
    held: dict[str, float] = field(default_factory=dict)
    # Set by the re-read that ends the playlist; it is then served unchanged.
    finished: bool = False

    def append(self, row: dict, fingerprint: str) -> ServedEntry:
        entry_lines, self.previous_fingerprint = day_playlist_entry_lines(
            row,
            fingerprint,
            self.previous_fingerprint,
            self.query,
            self.media_end,
            growing=True,
        )
        entry = ServedEntry(
            name=str(row["name"]),
            start_epoch=float(row["start_epoch"]),
            duration_seconds=float(row["duration_seconds"]),
            fragment_id=row_fragment_id(row),
            media_offset=self.media_end,
            lines=tuple(entry_lines),
        )
        self.entries.append(entry)
        self.by_name[entry.name] = entry
        self.media_end += entry.duration_seconds
        self.held.pop(entry.name, None)
        return entry

    def body(self, *, finished: bool) -> str:
        header = list(self.header)
        if finished:
            header = [
                "#EXT-X-PLAYLIST-TYPE:VOD"
                if line == "#EXT-X-PLAYLIST-TYPE:EVENT"
                else line
                for line in header
            ]
        lines = header + [
            line for entry in self.entries for line in entry.lines
        ]
        if finished:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What a re-read found that differs from what was already served."""

    appended: tuple[str, ...] = ()
    kept_unreadable: tuple[str, ...] = ()
    kept_missing: tuple[str, ...] = ()
    kept_changed: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    out_of_order: tuple[str, ...] = ()
    over_target: tuple[str, ...] = ()
    # Why a row could not be checked, by name (from the fragment source).
    reasons: dict[str, str] = field(default_factory=dict)


def row_fragment_id(row: dict) -> str:
    return str(
        getattr(row.get("_encoded_fragment"), "source_identity", "") or ""
    )


def row_unavailable_reason(row: dict) -> str:
    fragment = row.get("_encoded_fragment")
    return str(getattr(fragment, "unavailable_reason", "") or "unavailable")


def row_available(row: dict) -> bool:
    fragment = row.get("_encoded_fragment")
    return bool(getattr(fragment, "available", True))


def day_playlist_entry_lines(
    row: dict,
    stream_fingerprint: str,
    previous_fingerprint: str | None,
    query: str,
    media_offset: float,
    *,
    growing: bool = False,
) -> tuple[list[str], str]:
    """Render one day-playlist segment, exactly as the route always has."""
    row_start = float(row["start_epoch"])
    segment_name = quote(str(row["name"]), safe="")
    fragment_id = row_fragment_id(row)
    segment_query = (
        f"{query}&media_offset={media_offset:.3f}"
        f"&fragment_id={quote(fragment_id, safe='')}"
        f"&v={RECORDING_FMP4_VERSION}"
    )
    if growing:
        # Lets the segment routes find this entry's served record.
        segment_query += "&growing=true"
    map_lines, current = hls_map_transition(
        previous_fingerprint,
        stream_fingerprint,
        f"day/segment/{segment_name}/init.mp4?{segment_query}",
    )
    lines = list(map_lines)
    lines.extend(
        [
            "#EXT-X-PROGRAM-DATE-TIME:"
            f"{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
            f"#EXTINF:{float(row['duration_seconds']):.3f},",
            f"day/segment/{segment_name}/media.m4s?{segment_query}",
        ]
    )
    return lines, current


def merge_rows(
    record: GrowingPlaylistRecord,
    rows: list[dict],
    fingerprints: list[str],
    *,
    now_monotonic: float,
    hold_seconds: float = GROWING_HOLD_SECONDS,
) -> MergeReport:
    """Append new readable rows to ``record``; never change served entries."""
    kept_unreadable: list[str] = []
    kept_missing: list[str] = []
    kept_changed: list[str] = []
    reasons: dict[str, str] = {}
    fresh = {str(row.get("name")): row for row in rows}
    for entry in record.entries:
        row = fresh.get(entry.name)
        if row is None:
            kept_missing.append(entry.name)
        elif not row_available(row):
            kept_unreadable.append(entry.name)
            reasons[entry.name] = row_unavailable_reason(row)
        elif (
            f"{float(row['duration_seconds']):.3f}"
            != f"{entry.duration_seconds:.3f}"
            or float(row["start_epoch"]) != entry.start_epoch
            or row_fragment_id(row) != entry.fragment_id
        ):
            kept_changed.append(entry.name)

    appended: list[str] = []
    held: list[str] = []
    skipped: list[str] = []
    out_of_order: list[str] = []
    over_target: list[str] = []
    last_start = record.entries[-1].start_epoch if record.entries else -math.inf
    for row, fingerprint in zip(rows, fingerprints):
        name = str(row.get("name"))
        if name in record.by_name:
            continue
        start = float(row["start_epoch"])
        if start <= last_start:
            # Indexed after a later row was served: it cannot be inserted.
            out_of_order.append(name)
            continue
        if not row_available(row):
            reasons[name] = row_unavailable_reason(row)
            first_seen = record.held.setdefault(name, now_monotonic)
            if now_monotonic - first_seen < hold_seconds:
                held.append(name)
                break
            skipped.append(name)
            continue
        if math.ceil(float(row["duration_seconds"])) > record.target_duration:
            over_target.append(name)
        record.append(row, fingerprint)
        appended.append(name)
        last_start = start
    # A held row that has left the index will not be appended; forget it.
    for name in [name for name in record.held if name not in fresh]:
        record.held.pop(name, None)
    return MergeReport(
        appended=tuple(appended),
        kept_unreadable=tuple(kept_unreadable),
        kept_missing=tuple(kept_missing),
        kept_changed=tuple(kept_changed),
        held=tuple(held),
        skipped=tuple(skipped),
        out_of_order=tuple(out_of_order),
        over_target=tuple(over_target),
        reasons=reasons,
    )


class RateLimitedLog:
    """Log one message per (kind, scope) per interval, counting the rest."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        interval_seconds: float = GROWING_LOG_INTERVAL_SECONDS,
        max_scopes: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._interval = interval_seconds
        self._max_scopes = max_scopes
        self._clock = clock
        self._lock = threading.Lock()
        self._state: OrderedDict[tuple[str, str], tuple[float, int]] = OrderedDict()

    def log(
        self,
        level: int,
        kind: str,
        scope: str,
        message: str,
        *args: object,
    ) -> bool:
        now = self._clock()
        with self._lock:
            key = (kind, scope)
            last, suppressed = self._state.get(key, (-math.inf, 0))
            if now - last < self._interval:
                self._state[key] = (last, suppressed + 1)
                return False
            self._state[key] = (now, 0)
            self._state.move_to_end(key)
            while len(self._state) > self._max_scopes:
                self._state.popitem(last=False)
        if suppressed:
            message += " (%d similar messages suppressed)"
            args = (*args, suppressed)
        self._logger.log(level, message, *args)
        return True


class GrowingPlaylistRegistry:
    """Bounded, thread-safe store of growing playlist records."""

    def __init__(
        self,
        *,
        max_records: int = GROWING_RECORD_MAX_RECORDS,
        max_entries: int = GROWING_RECORD_MAX_ENTRIES,
        clock: Callable[[], float] = lambda: time.time(),
    ) -> None:
        self._max_records = max(1, int(max_records))
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self.lock = threading.RLock()
        self._records: OrderedDict[GrowingPlaylistKey, GrowingPlaylistRecord] = (
            OrderedDict()
        )
        # Recently evicted keys, so a re-read that must rebuild can say so.
        self._evicted: OrderedDict[GrowingPlaylistKey, None] = OrderedDict()

    def get(self, key: GrowingPlaylistKey) -> GrowingPlaylistRecord | None:
        """Return a live record and mark it recently used (hold ``lock``)."""
        self._expire()
        record = self._records.get(key)
        if record is not None:
            self._records.move_to_end(key)
        return record

    def put(self, record: GrowingPlaylistRecord) -> list[GrowingPlaylistKey]:
        """Store ``record`` (hold ``lock``); return the keys evicted for room."""
        self._records[record.key] = record
        self._records.move_to_end(record.key)
        return self.enforce_bounds(keep=record.key)

    def was_evicted(self, key: GrowingPlaylistKey) -> bool:
        """Whether ``key`` lost its record to the bounds (hold ``lock``)."""
        return self._evicted.pop(key, False) is None

    def _note_evicted(self, key: GrowingPlaylistKey) -> None:
        self._evicted[key] = None
        self._evicted.move_to_end(key)
        while len(self._evicted) > GROWING_EVICTED_KEYS:
            self._evicted.popitem(last=False)

    def enforce_bounds(
        self,
        *,
        keep: GrowingPlaylistKey | None = None,
    ) -> list[GrowingPlaylistKey]:
        evicted: list[GrowingPlaylistKey] = []
        total = sum(len(record.entries) for record in self._records.values())
        for key in list(self._records):
            if (
                len(self._records) <= self._max_records
                and total <= self._max_entries
            ):
                break
            if key == keep:
                continue
            total -= len(self._records.pop(key).entries)
            evicted.append(key)
        if keep in self._records and total > self._max_entries:
            # One record larger than the whole budget is not kept at all.
            self._records.pop(keep)
            evicted.append(keep)
        for key in evicted:
            self._note_evicted(key)
        return evicted

    def served_entry(
        self,
        window: tuple[str, str, str, str],
        name: str,
        fragment_id: str,
    ) -> ServedEntry | None:
        """Find an entry served under ``window`` with this exact identity."""
        with self.lock:
            self._expire()
            for key, record in self._records.items():
                if key.window() != window:
                    continue
                entry = record.by_name.get(name)
                if entry is not None and entry.fragment_id == fragment_id:
                    return entry
        return None

    def keys(self) -> Iterable[GrowingPlaylistKey]:
        with self.lock:
            return list(self._records)

    def _expire(self) -> None:
        now = self._clock()
        for key in [
            key for key, record in self._records.items()
            if record.expires_at <= now
        ]:
            self._records.pop(key, None)
