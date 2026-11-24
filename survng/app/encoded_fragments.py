"""Transport-neutral access to finalized, codec-preserving media fragments."""

from __future__ import annotations

import errno
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
    # False only in a tolerant listing: the row is indexed but its file could
    # not be checked. Such a fragment has no identity and must not be served.
    available: bool = True
    unavailable_reason: str = ""


class EncodedFragmentSource(Protocol):
    def fragments(
        self,
        window: FragmentWindow,
        *,
        trim_end: bool = False,
        cancel: threading.Event | None = None,
        tolerate_unavailable: bool = False,
    ) -> Iterator[EncodedFragment]: ...

    def materialize(
        self,
        fragment: EncodedFragment,
        *,
        cancel: threading.Event | None = None,
    ) -> EncodedFragment: ...

    def close(self) -> None: ...


def _unavailable_reason(exc: BaseException) -> str:
    """A short, secret-free description of why a file could not be checked."""
    name = type(exc).__name__
    code = getattr(exc, "errno", None)
    if code is not None:
        return f"{name}[{errno.errorcode.get(code, code)}]"
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"{name}[{status}]"
    return name


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
        tolerant_row_loader: Callable[
            [str, float, float, str], list[dict]
        ] | None = None,
        unavailable_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._row_loader = row_loader
        # Used by tolerant listings: it also returns rows whose file cannot be
        # checked now, marked ``_storage_unavailable``, instead of hiding them.
        self._tolerant_row_loader = tolerant_row_loader or row_loader
        # Errors the path resolver raises for a missing or refused file, in
        # addition to OSError, that a tolerant listing reports as unavailable.
        self._unavailable_errors = tuple(unavailable_errors)
        self._path_resolver = path_resolver
        self._remuxer = remuxer
        self._lock = threading.Lock()
        self._closed = False

    def fragments(
        self,
        window: FragmentWindow,
        *,
        trim_end: bool = False,
        cancel: threading.Event | None = None,
        tolerate_unavailable: bool = False,
    ) -> Iterator[EncodedFragment]:
        """List the window's fragments.

        By default any row whose file cannot be checked fails the listing
        with ``FragmentUnavailable``. With ``tolerate_unavailable`` such a row
        is yielded with ``available=False`` instead (and adds nothing to the
        media offset), so a caller that already served the other rows can
        keep serving them.
        """
        self._ensure_open()
        self._check_cancel(cancel)
        loader = self._tolerant_row_loader if tolerate_unavailable else self._row_loader
        rows = loader(
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
            if tolerate_unavailable and row.get("_storage_unavailable"):
                yield self._unavailable(
                    sequence, row, wall_start, duration, fingerprint,
                    str(row["_storage_unavailable"]),
                )
                continue
            try:
                path = self._path_resolver(row.get("path"))
                stat = path.stat()
            except (KeyError, OSError, TypeError, ValueError) as exc:
                if not tolerate_unavailable:
                    raise FragmentUnavailable(
                        "recording fragment is unavailable"
                    ) from exc
                yield self._unavailable(
                    sequence, row, wall_start, duration, fingerprint,
                    _unavailable_reason(exc),
                )
                continue
            except self._unavailable_errors as exc:
                # The resolver's own error (an HTTP 404/403): strict listings
                # keep propagating it unchanged.
                if not tolerate_unavailable:
                    raise
                yield self._unavailable(
                    sequence, row, wall_start, duration, fingerprint,
                    _unavailable_reason(exc),
                )
                continue
            name = str(row.get("name") or path.name)
            identity = hashlib.sha256(
                (
                    f"v{RECORDING_FMP4_VERSION}:{path.resolve()}:"
                    f"{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}"
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

    @staticmethod
    def _unavailable(
        sequence: int,
        row: dict,
        wall_start: float,
        duration: float,
        fingerprint: str,
        reason: str,
    ) -> EncodedFragment:
        raw_path = Path(str(row.get("path") or ""))
        return EncodedFragment(
            sequence=sequence,
            segment_name=str(row.get("name") or raw_path.name),
            wall_start_epoch=wall_start,
            wall_end_epoch=float(row.get("end_epoch") or wall_start + duration),
            media_start_seconds=0.0,
            media_duration_seconds=duration,
            keyframe_aligned=bool(row.get("playable", True)),
            stream=StreamDescription(fingerprint=fingerprint),
            source_identity="",
            source_path=raw_path,
            available=False,
            unavailable_reason=reason,
        )

    def materialize(
        self,
        fragment: EncodedFragment,
        *,
        cancel: threading.Event | None = None,
    ) -> EncodedFragment:
        if not fragment.available:
            raise FragmentUnavailable("recording fragment is unavailable")
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
