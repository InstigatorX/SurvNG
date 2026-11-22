"""Restart-only recovery runs once per process; interrupted-work marking does not.

A settings reload builds a new EventStore over the same database while the
previous one is still running. Its _init_db used to repeat the orphan repair
and audit backfill there, holding the event-store lock and the main database
write lock for 1-5 s on a production database. Marking work interrupted stays
on every construction: a reload is refused while AI work is registered, so a
row still in a running state then has no live worker.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from survng.app.event_store import EventStore
from survng.app.event_store import jobs as event_jobs
from survng.app.main_database import forget_startup_recovery


def _cancelling_run(store: EventStore) -> int:
    run = store.create_calibration_run(
        mode="standard", camera_ids=["gate"], configuration_fingerprint="a" * 64,
    )
    store.update_calibration_run(int(run["id"]), status="cancelling")
    return int(run["id"])


def _orphan_audits(store: EventStore, count: int = 1) -> None:
    """Audit rows pointing at events that no longer exist (legacy, no FK)."""
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("pragma foreign_keys = off")
        connection.executemany(
            "insert into motion_audits(event_id, related_event_id, camera_id, snapshot_path, "
            "created_at, mode, sensitivity, score, threshold, reason) "
            "values (?, null, 'gate', '', '2026-01-01T00:00:00+00:00', 'audit', 'balanced', "
            "0.1, 0.5, 'test')",
            [(10_000 + index,) for index in range(count)],
        )


def _orphaned(store: EventStore) -> int:
    with sqlite3.connect(store.db_path) as connection:
        return connection.execute(
            "select count(*) from motion_audits where event_id is not null"
        ).fetchone()[0]


def test_reload_marks_interrupted_work_but_skips_the_orphan_repair(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    run_id = _cancelling_run(store)
    _orphan_audits(store)

    reloaded = EventStore(tmp_path)

    run = reloaded.get_calibration_run(run_id)
    assert run is not None and run["status"] == "interrupted"
    assert _orphaned(reloaded) == 1


def test_a_new_process_repairs_orphans(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    _orphan_audits(store)

    forget_startup_recovery()
    restarted = EventStore(tmp_path)

    assert _orphaned(restarted) == 0


def test_a_replaced_database_file_is_recovered_again(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    _orphan_audits(store)
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(store.db_path) as source, sqlite3.connect(replacement) as copy:
        source.backup(copy)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{store.db_path}{suffix}").unlink(missing_ok=True)
    replacement.replace(store.db_path)  # new inode, as a restore would produce

    assert _orphaned(EventStore(tmp_path)) == 0


def test_a_database_without_this_process_token_is_recovered(tmp_path: Path) -> None:
    # A recreated file can reuse the inode number; its contents still lack
    # the token this process wrote when it recovered the original.
    store = EventStore(tmp_path)
    _orphan_audits(store)
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("delete from survng_startup_recovery")

    assert _orphaned(EventStore(tmp_path)) == 0


def test_orphan_repair_runs_in_rowid_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EventStore(tmp_path)
    _orphan_audits(store, count=7)
    monkeypatch.setattr(event_jobs, "ORPHAN_REPAIR_BATCH_ROWS", 2)

    forget_startup_recovery()
    restarted = EventStore(tmp_path)

    assert _orphaned(restarted) == 0
