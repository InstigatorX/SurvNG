"""A reload must not re-queue a job the previous backfill worker still holds.

A settings reload builds a new DeferredAppearanceBackfill while the old
worker is running. Restart recovery (every running job back to queued) runs
once per process; a stalled job is re-queued when a worker starts, but only
after STALLED_JOB_SECONDS without progress.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

from survng.app.appearance_backfill import STALLED_JOB_SECONDS, DeferredAppearanceBackfill
from survng.app.appearance_index import AppearanceIndex
from survng.app.config import ObjectTrackingConfig
from survng.app.main_database import forget_startup_recovery


def _service(database: Path, root: Path) -> DeferredAppearanceBackfill:
    return DeferredAppearanceBackfill(
        database, root, ObjectTrackingConfig(), Mock(), AppearanceIndex(database), Mock(),
    )


def _running_job(database: Path, event_id: int, updated_at: str) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute("insert or ignore into events values (?)", (event_id,))
        connection.execute(
            "insert into appearance_backfill_jobs(event_id, camera_id, state, available_at, "
            "created_at, updated_at) values (?, 'gate', 'running', 0, ?, ?)",
            (event_id, updated_at, updated_at),
        )


def _state(database: Path, event_id: int) -> str:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "select state from appearance_backfill_jobs where event_id = ?", (event_id,)
        ).fetchone()[0]


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "events.db"
    with sqlite3.connect(database) as connection:
        connection.execute("create table events (id integer primary key)")
    return database


def test_reload_leaves_a_job_the_old_worker_holds(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _service(database, tmp_path)
    _running_job(database, 7, datetime.now(timezone.utc).isoformat())

    _service(database, tmp_path)  # reload: same process
    assert _state(database, 7) == "running"

    forget_startup_recovery()
    _service(database, tmp_path)  # new process
    assert _state(database, 7) == "queued"


def test_starting_a_worker_requeues_only_stalled_jobs(tmp_path: Path) -> None:
    database = _database(tmp_path)
    service = _service(database, tmp_path)
    now = datetime.now(timezone.utc)
    _running_job(database, 7, (now - timedelta(seconds=STALLED_JOB_SECONDS + 60)).isoformat())
    _running_job(database, 8, now.isoformat())

    assert service._requeue_stalled_jobs() == 1
    assert _state(database, 7) == "queued"
    assert _state(database, 8) == "running"
