"""Transport-neutral access to finalized, codec-preserving media fragments."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator, Protocol

from .recording_media import (
    RECORDING_FMP4_VERSION,
    playback_segment_duration,
    resolve_stream_fingerprints,
)


class FragmentSourceError(RuntimeError):
    pass


class InvalidFragmentWindow(FragmentSourceError):
    pass


class FragmentUnavailable(FragmentSourceError):
    pass


class FragmentCancelled(FragmentSourceError):
    pass


class FragmentSourceClosed(FragmentSourceError):
    pass


@dataclass(frozen=True, slots=True)
class FragmentWindow:
    camera_id: str
    source: str
    start_epoch: float
    end_epoch: float

    def __post_init__(self) -> None:
        if not str(self.camera_id or "").strip():
            raise InvalidFragmentWindow("camera_id is required")
        if self.source not in {"main", "live"}:
            raise InvalidFragmentWindow("source must be main or live")
        if (
            not math.isfinite(self.start_epoch)
            or not math.isfinite(self.end_epoch)
            or self.end_epoch <= self.start_epoch
        ):
            raise InvalidFragmentWindow("fragment window is invalid")


@dataclass(frozen=True, slots=True)
class StreamDescription:
    fingerprint: str | None
    init_path: Path | None = None


@dataclass(frozen=True, slots=True)
class EncodedFragment:
    sequence: int
    segment_name: str
    wall_start_epoch: float
    wall_end_epoch: float
    media_start_seconds: float
    media_duration_seconds: float
    keyframe_aligned: bool
    stream: StreamDescription
    source_identity: str
    source_path: Path
    media_path: Path | None = None
    discontinuity_before: bool = False
    finalized: bool = True


@dataclass(frozen=True, slots=True)
class FragmentSourceInfo:
    window: FragmentWindow
    available_start_epoch: float | None
    available_end_epoch: float | None
    fragment_count: int
    follow_supported: bool = False
    provenance: str = "indexed_mp4"


class EncodedFragmentSource(Protocol):
    def describe(self, window: FragmentWindow) -> FragmentSourceInfo: ...

    def fragments(
        self,
        window: FragmentWindow,
        *,
        trim_end: bool = False,
        cancel: threading.Event | None = None,
    ) -> Iterator[EncodedFragment]: ...

    def materialize(
        self,
        fragment: EncodedFragment,
        *,
        cancel: threading.Event | None = None,
    ) -> EncodedFragment: ...

    def close(self) -> None: ...


class IndexedMp4FragmentSource:
    """Expose finalized recording-index rows as lazily materialized fMP4."""

    def __init__(
        self,
        *,
        row_loader: Callable[
            [str, float, float, str], list[dict]
        ],
        path_resolver: Callable[[object], Path],
        remuxer: Callable[[Path, float, float], tuple[Path, Path]],
    ) -> None:
        self._row_loader = row_loader
        self._path_resolver = path_resolver
        self._remuxer = remuxer
        self._lock = threading.Lock()
        self._closed = False

    def describe(self, window: FragmentWindow) -> FragmentSourceInfo:
        fragments = list(self.fragments(window))
        return FragmentSourceInfo(
            window=window,
            available_start_epoch=(
                fragments[0].wall_start_epoch if fragments else None
            ),
            available_end_epoch=(
                fragments[-1].wall_end_epoch if fragments else None
            ),
            fragment_count=len(fragments),
        )

    def fragments(
        self,
        window: FragmentWindow,
        *,
        trim_end: bool = False,
        cancel: threading.Event | None = None,
    ) -> Iterator[EncodedFragment]:
        self._ensure_open()
        self._check_cancel(cancel)
        rows = self._row_loader(
            window.camera_id,
            window.start_epoch,
            window.end_epoch,
            window.source,
        )
        fingerprints = resolve_stream_fingerprints(
            [row.get("stream_fingerprint") for row in rows]
        )
        media_offset = 0.0
        previous_fingerprint: str | None = None
        for sequence, (row, fingerprint) in enumerate(
            zip(rows, fingerprints),
            start=1,
        ):
            self._check_cancel(cancel)
            try:
                path = self._path_resolver(row.get("path"))
                stat = path.stat()
                wall_start = float(row["start_epoch"])
                duration = playback_segment_duration(
                    wall_start,
                    float(row["duration_seconds"]),
                    window.end_epoch,
                    trim_end,
                )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise FragmentUnavailable(
                    "recording fragment is unavailable"
                ) from exc
            name = str(row.get("name") or path.name)
            identity = hashlib.sha256(
                (
                    f"v{RECORDING_FMP4_VERSION}:{path.resolve()}:"
                    f"{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}:"
                    f"{media_offset:.3f}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            discontinuity = (
                sequence > 1
                and (
                    not previous_fingerprint
                    or not fingerprint
                    or fingerprint != previous_fingerprint
                )
            )
            yield EncodedFragment(
                sequence=sequence,
                segment_name=name,
                wall_start_epoch=wall_start,
                wall_end_epoch=float(
                    row.get("end_epoch") or wall_start + duration
                ),
                media_start_seconds=media_offset,
                media_duration_seconds=duration,
                keyframe_aligned=bool(row.get("playable", True)),
                stream=StreamDescription(fingerprint=fingerprint),
                source_identity=identity,
                source_path=path,
                discontinuity_before=discontinuity,
            )
            previous_fingerprint = fingerprint
            media_offset += duration

    def materialize(
        self,
        fragment: EncodedFragment,
        *,
        cancel: threading.Event | None = None,
    ) -> EncodedFragment:
        self._ensure_open()
        self._check_cancel(cancel)
        init_path, media_path = self._remuxer(
            fragment.source_path,
            fragment.media_duration_seconds,
            fragment.media_start_seconds,
        )
        self._check_cancel(cancel)
        return replace(
            fragment,
            stream=replace(fragment.stream, init_path=init_path),
            media_path=media_path,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise FragmentSourceClosed("fragment source is closed")

    @staticmethod
    def _check_cancel(cancel: threading.Event | None) -> None:
        if cancel is not None and cancel.is_set():
            raise FragmentCancelled("fragment request was cancelled")
