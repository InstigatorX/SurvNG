"""Retention planning reads index pages, not table rows.

Production timing (InstigatorX/SurvNG retention probe): the recording plan
took 93 s on a 2.6 GB recording index, scanning a non-covering index and
reading every segment's table row; the snapshot plan and candidate scan took
10-25 s reading every snapshot-bearing event row (with its objects_json).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from survng.app.config import CameraConfig, RecordingRetentionConfig
from survng.app.event_store import EventStore
from survng.app.face_store.store import FaceStore
from survng.app.recording_process.index import RecordingIndexMixin
from survng.app.recording_retention import (
    GROUP_TOTALS_SQL,
    PROTECTED_EXPIRED_SQL,
    RETENTION_TOTALS_INDEX,
    RecordingRetentionService,
)

DAY = 86400.0


class _Index(RecordingIndexMixin):
    def __init__(self, root: Path) -> None:
        self.index_path = root / "recordings.sqlite3"
        self.recordings_dir = root / "recordings"
        self.media_storage = None


def _recording_index(tmp_path: Path) -> Path:
    index = _Index(tmp_path)
    index._init_recording_index()
    return index.index_path


def _connect(path: Path):
    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection
    return connect


def _plan_steps(connection: sqlite3.Connection, sql: str) -> list[str]:
    return [row[3] for row in connection.execute("explain query plan " + sql)]


def test_recording_totals_come_from_the_covering_index(tmp_path: Path) -> None:
    path = _recording_index(tmp_path)
    connection = sqlite3.connect(path)
    connection.execute(
        "create temp table retention_policy (camera_id text not null, source text not null, "
        "cutoff real not null, primary key (camera_id, source)) without rowid"
    )
    connection.execute("create temp table retention_protected (path text primary key) without rowid")

    totals = _plan_steps(connection, GROUP_TOTALS_SQL.format(hint=f"INDEXED BY {RETENTION_TOTALS_INDEX}"))
    protected = _plan_steps(connection, PROTECTED_EXPIRED_SQL)
    indexes = [row[1] for row in connection.execute("pragma index_list(recordings)")]

    assert any(f"COVERING INDEX {RETENTION_TOTALS_INDEX}" in step for step in totals), totals
    assert protected[0] == "SCAN protected", protected  # protected paths drive the join
    assert "recordings_retention_expiry" not in indexes  # superseded by the covering index


def test_recording_plan_splits_expired_segments_by_protection(tmp_path: Path) -> None:
    path = _recording_index(tmp_path)
    now = time.time()
    rows = []
    # (camera, source, age in days, size, protected)
    layout = [
        ("gate", "main", 10, 100, False), ("gate", "main", 9, 200, True),
        ("gate", "main", 1, 400, False), ("gate", "live", 30, 800, False),
        ("gate", "live", 25, 1600, True), ("yard", "main", 8, 3200, False),
        ("yard", "main", 2, 6400, True),
    ]
    protected_paths = set()
    for index, (camera, source, age, size, protected) in enumerate(layout):
        segment = str(tmp_path / "recordings" / camera / source / f"{index}.mp4")
        start = now - age * DAY
        rows.append((segment, camera, source, f"{index}.mp4", size, start, start, 10.0, start + 10.0))
        if protected:
            protected_paths.add(segment)
    with sqlite3.connect(path) as connection:
        connection.executemany(
            "insert into recordings(path, camera_id, source, name, size_bytes, modified_at, "
            "start_epoch, duration_seconds, end_epoch) values (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    service = RecordingRetentionService(
        tmp_path, tmp_path / "recordings", _connect(path),
        RecordingRetentionConfig(main_days=7, live_days=21, storage_limit_tb=1),
        protected_paths_provider=lambda: set(protected_paths),
    )
    service._cameras = {
        camera: CameraConfig(id=camera, name=camera, stream_url="rtsp://camera/main")
        for camera in ("gate", "yard")
    }

    groups = {
        (item["camera_id"], item["source"]): item
        for item in service.plan(now_epoch=now)["per_camera"]
    }

    assert (groups[("gate", "main")]["expired_files"], groups[("gate", "main")]["expired_bytes"]) == (1, 100)
    assert (groups[("gate", "main")]["protected_expired_files"],
            groups[("gate", "main")]["protected_expired_bytes"]) == (1, 200)
    assert groups[("gate", "main")]["file_count"] == 3
    assert groups[("gate", "main")]["bytes"] == 700
    assert (groups[("gate", "live")]["expired_files"], groups[("gate", "live")]["expired_bytes"]) == (1, 800)
    assert groups[("gate", "live")]["protected_expired_bytes"] == 1600
    assert (groups[("yard", "main")]["expired_files"], groups[("yard", "main")]["expired_bytes"]) == (1, 3200)
    assert groups[("yard", "main")]["protected_expired_files"] == 0  # protected but not expired


def test_snapshot_planning_reads_covering_and_pinned_indexes(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    FaceStore(tmp_path, start_recognition=False)
    ranked = EventStore._latest_snapshot_refs_cte()
    connection = sqlite3.connect(store.db_path)
    plan = _plan_steps(connection, ranked + " select count(*) from ranked where snapshot_rank = 1")
    pinned = _plan_steps(
        connection,
        "select distinct snapshot_path from face_observations where reference_pinned = 1",
    )

    assert any(step.startswith("SCAN events USING COVERING INDEX") for step in plan), plan
    assert any("idx_face_observations_pinned_snapshot" in step for step in pinned), pinned
