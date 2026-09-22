from __future__ import annotations

import threading
from pathlib import Path

import pytest

from survng.app.encoded_fragments import (
    FragmentCancelled,
    FragmentSourceClosed,
    FragmentWindow,
    IndexedMp4FragmentSource,
    InvalidFragmentWindow,
)


def test_indexed_fragment_source_preserves_wall_time_and_compacts_media_time(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    rows = [
        {
            "path": str(first),
            "name": first.name,
            "start_epoch": 100.0,
            "end_epoch": 110.0,
            "duration_seconds": 10.0,
            "stream_fingerprint": "h264-a",
            "playable": True,
        },
        {
            "path": str(second),
            "name": second.name,
            "start_epoch": 120.0,
            "end_epoch": 130.0,
            "duration_seconds": 10.0,
            "stream_fingerprint": "h264-a",
            "playable": True,
        },
    ]
    remuxed = []
    source = IndexedMp4FragmentSource(
        row_loader=lambda *_args: rows,
        path_resolver=lambda value: Path(str(value)),
        remuxer=lambda path, duration, offset: (
            remuxed.append((path, duration, offset))
            or (tmp_path / "init.mp4", tmp_path / "media.m4s")
        ),
    )

    fragments = list(
        source.fragments(FragmentWindow("gate", "main", 100.0, 130.0))
    )

    assert [item.wall_start_epoch for item in fragments] == [100.0, 120.0]
    assert [item.media_start_seconds for item in fragments] == [0.0, 10.0]
    assert not fragments[1].discontinuity_before
    materialized = source.materialize(fragments[1])
    assert materialized.stream.init_path == tmp_path / "init.mp4"
    assert materialized.media_path == tmp_path / "media.m4s"
    assert remuxed == [(second, 10.0, 10.0)]


def test_indexed_fragment_source_marks_unknown_and_changed_streams(
    tmp_path: Path,
) -> None:
    paths = [
        tmp_path / "a.mp4",
        tmp_path / "b.mp4",
        tmp_path / "c.mp4",
    ]
    for path in paths:
        path.write_bytes(b"fragment")
    rows = [
        {
            "path": str(path),
            "name": path.name,
            "start_epoch": float(index * 10),
            "end_epoch": float(index * 10 + 10),
            "duration_seconds": 10.0,
            "stream_fingerprint": fingerprint,
        }
        for index, (path, fingerprint) in enumerate(
            zip(paths, ("h264-a", "", "h265-b")),
            start=1,
        )
    ]
    source = IndexedMp4FragmentSource(
        row_loader=lambda *_args: rows,
        path_resolver=lambda value: Path(str(value)),
        remuxer=lambda *_args: paths[:2],
    )
    fragments = list(
        source.fragments(FragmentWindow("gate", "main", 0.0, 100.0))
    )
    assert [item.discontinuity_before for item in fragments] == [
        False,
        True,
        True,
    ]
def test_fragment_source_validates_cancellation_and_close(tmp_path: Path) -> None:
    with pytest.raises(InvalidFragmentWindow):
        FragmentWindow("gate", "main", 10.0, 10.0)

    source = IndexedMp4FragmentSource(
        row_loader=lambda *_args: [],
        path_resolver=lambda value: Path(str(value)),
        remuxer=lambda *_args: (tmp_path / "init", tmp_path / "media"),
    )
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(FragmentCancelled):
        list(
            source.fragments(
                FragmentWindow("gate", "main", 0.0, 1.0),
                cancel=cancelled,
            )
        )

    source.close()
    with pytest.raises(FragmentSourceClosed):
        list(source.fragments(FragmentWindow("gate", "main", 0.0, 1.0)))
