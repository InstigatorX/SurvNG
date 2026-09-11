"""Exact video metadata survives process/cache turnover without reopening media."""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from survng.app.recorder import Recorder

PARSER = "survng.app.recording_process.index.mp4_playback_metadata"


def index_row(path):
    stat = path.stat()
    return dict(path=str(path), name=path.name, size_bytes=stat.st_size,
                modified_at=stat.st_mtime, start_epoch=100.0,
                duration_seconds=10.0, end_epoch=110.0, source="main")


def age(path):
    old = time.time() - 30
    os.utime(path, (old, old))


@pytest.fixture
def indexed(tmp_path):
    recorder = Recorder("ffmpeg", tmp_path)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"recording")
    age(path)
    recorder._store_recording_rows("gate", "main", [index_row(path)])
    return recorder, path


def stored(recorder, path):
    with recorder._index_connection() as connection:
        return dict(connection.execute("SELECT * FROM recordings WHERE path = ?", (str(path),)).fetchone())


def test_legacy_read_persists_exact_duration_and_survives_restart(indexed):
    recorder, path = indexed
    rows = [stored(recorder, path)]
    with patch(PARSER, return_value=("video-v1", 9.73)) as parse:
        recorder.resolve_recording_playback_metadata(rows)
    parse.assert_called_once_with(path)
    assert rows[0]["duration_seconds"] == 9.73
    persisted = stored(recorder, path)
    assert persisted["duration_seconds"] == 10  # Availability estimate is a separate value.
    assert persisted["playback_duration_seconds"] == 9.73
    assert persisted["stream_fingerprint"] == "video-v1"
    reopened = Recorder("ffmpeg", path.parent)
    rows = reopened.recording_rows_between("gate", 100, 110, discover_missing=False)
    with patch(PARSER, side_effect=AssertionError("cached playback must not read MP4 headers")):
        reopened.resolve_recording_playback_metadata(rows)
    assert rows[0]["end_epoch"] == 109.73


def test_nullable_column_migrates_without_inventing_legacy_metadata(indexed):
    recorder, path = indexed
    with recorder._index_connection() as connection:
        connection.execute("ALTER TABLE recordings DROP COLUMN playback_duration_seconds")
    reopened = Recorder("ffmpeg", path.parent)
    assert stored(reopened, path)["playback_duration_seconds"] is None


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_same_path_replacement_invalidates_metadata(indexed, change):
    recorder, path = indexed
    with patch(PARSER, return_value=("video-v1", 9.73)):
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    with recorder._index_connection() as connection:
        connection.execute("UPDATE recordings SET validated=1, duration_seconds=9.9, end_epoch=109.9")
    if change == "size":
        old = path.stat().st_mtime
        path.write_bytes(b"longer replacement recording")
        os.utime(path, (old, old))
    else:
        old = path.stat().st_mtime - 1
        os.utime(path, (old, old))
    # Even before the indexer sees the replacement, stat mismatch forbids reuse.
    with patch(PARSER, return_value=("video-v2", 8.5)) as parse:
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    parse.assert_called_once()
    assert stored(recorder, path)["playback_duration_seconds"] == 9.73  # No write against mismatched identity.
    recorder._store_recording_rows("gate", "main", [index_row(path)])
    row = stored(recorder, path)
    assert row["playback_duration_seconds"] is None
    assert row["stream_fingerprint"] == "" and row["fingerprint_checked"] == 0
    assert row["validated"] == 0 and row["duration_seconds"] == 10 and row["end_epoch"] == 110
    with patch(PARSER, return_value=("video-v2", 8.5)):
        recorder.resolve_recording_playback_metadata([row])
    assert stored(recorder, path)["playback_duration_seconds"] == 8.5


@pytest.mark.parametrize("result", [("", None), ("video", float("nan")), ("video", -1)])
def test_failed_or_invalid_metadata_is_not_saved(indexed, result):
    recorder, path = indexed
    with patch(PARSER, return_value=result):
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    assert stored(recorder, path)["playback_duration_seconds"] is None


def test_file_changing_during_header_read_is_not_saved(indexed):
    recorder, path = indexed
    def parse(_path):
        path.write_bytes(b"changed while reading")
        return "old-video", 9.73
    with patch(PARSER, side_effect=parse):
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    assert stored(recorder, path)["playback_duration_seconds"] is None


def test_concurrent_index_change_rejects_old_metadata_write(indexed):
    recorder, path = indexed
    def parse(_path):
        with recorder._index_connection() as connection:
            connection.execute("UPDATE recordings SET modified_at=modified_at+1 WHERE path=?", (str(path),))
        return "old-video", 9.73
    with patch(PARSER, side_effect=parse):
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    assert stored(recorder, path)["playback_duration_seconds"] is None


def test_unfinalized_file_is_not_saved(indexed):
    recorder, path = indexed
    os.utime(path, None)
    recorder._store_recording_rows("gate", "main", [index_row(path)])
    with patch(PARSER, return_value=("video", 9.73)):
        recorder.resolve_recording_playback_metadata([stored(recorder, path)])
    assert stored(recorder, path)["playback_duration_seconds"] is None


def test_existing_fingerprint_pass_saves_video_duration(indexed):
    recorder, path = indexed
    recorder.queue_stream_fingerprints([stored(recorder, path)])
    with patch(PARSER, return_value=("video", 9.73)):
        assert recorder._backfill_stream_fingerprints(limit=1) == 1
    assert stored(recorder, path)["playback_duration_seconds"] == 9.73


@pytest.mark.parametrize("fingerprint,duration", [("", 9.73), ("video", None)])
def test_partial_metadata_keeps_existing_fingerprint_pass_semantics(indexed, fingerprint, duration):
    recorder, path = indexed
    recorder.queue_stream_fingerprints([stored(recorder, path)])
    with patch(PARSER, return_value=(fingerprint, duration)):
        recorder._backfill_stream_fingerprints(limit=1)
    row = stored(recorder, path)
    assert row["fingerprint_checked"] == 1 and row["stream_fingerprint"] == fingerprint
    assert row["playback_duration_seconds"] == duration
    recorder.queue_stream_fingerprints([index_row(path)])
    assert not recorder._fingerprint_pending
    if duration is not None:
        with patch(PARSER, side_effect=AssertionError("exact duration with unknown codec still reuses metadata")):
            recorder.resolve_recording_playback_metadata([row])


def test_recent_discovery_queues_only_new_eligible_metadata(tmp_path):
    recorder = Recorder("ffmpeg", tmp_path)
    rows = []
    for name, seconds_ago in [("new", 20), ("complete", 30), ("unplayable", 40), ("historical", 3600)]:
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(b"recording")
        age(path)
        row = index_row(path)
        row.update(start_epoch=time.time() - seconds_ago, end_epoch=time.time() - seconds_ago + 10)
        rows.append(row)
    recorder._store_recording_rows("gate", "main", rows)
    with recorder._index_connection() as connection:
        connection.execute("UPDATE recordings SET fingerprint_checked=1, stream_fingerprint='known', playback_duration_seconds=9.73 WHERE name='complete.mp4'")
        connection.execute("UPDATE recordings SET playable=0 WHERE name='unplayable.mp4'")
    with (patch.object(recorder, "_wanted_keys", return_value=[("gate", "main")]),
          patch.object(recorder, "_recent_hour_recording_files", return_value=[]),
          patch.object(recorder, "_recording_rows_for_files", return_value=rows)):
        recorder.refresh_recording_index({}, run_maintenance=False)
        recorder.refresh_recording_index({}, run_maintenance=False)
        assert list(recorder._fingerprint_pending) == [rows[0]["path"]]
        with patch(PARSER, return_value=("new-video", 9.73)) as parse:
            recorder._backfill_stream_fingerprints(limit=20)
        parse.assert_called_once_with(Path(rows[0]["path"]))
        recorder.refresh_recording_index({}, run_maintenance=False)
        assert not recorder._fingerprint_pending
