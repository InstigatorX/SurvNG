"""Restart recovery runs once per process, not on every settings reload.

A settings reload builds a new EventStore over the same database while the
previous one is still running. Its _init_db used to repeat restart recovery
and full-table orphan repairs there, holding the event-store lock and the
main database write lock for 1-5 s on a production database, and marking
work that was still running as interrupted by a restart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.main_database import forget_startup_recovery


def _cancelling_run(store: EventStore) -> int:
    run = store.create_calibration_run(
        mode="standard", camera_ids=["gate"], configuration_fingerprint="a" * 64,
    )
    store.update_calibration_run(int(run["id"]), status="cancelling")
    return int(run["id"])


def test_reload_in_the_same_process_skips_restart_recovery(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    run_id = _cancelling_run(store)

    reloaded = EventStore(tmp_path)

    run = reloaded.get_calibration_run(run_id)
    assert run is not None and run["status"] == "cancelling"


def test_a_new_process_still_recovers(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    run_id = _cancelling_run(store)

    forget_startup_recovery()
    restarted = EventStore(tmp_path)

    run = restarted.get_calibration_run(run_id)
    assert run is not None and run["status"] == "interrupted"


def test_a_replaced_database_file_is_recovered_again(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    run_id = _cancelling_run(store)
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(store.db_path) as source, sqlite3.connect(replacement) as copy:
        source.backup(copy)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{store.db_path}{suffix}").unlink(missing_ok=True)
    replacement.replace(store.db_path)  # new inode, as a restore would produce

    run = EventStore(tmp_path).get_calibration_run(run_id)
    assert run is not None and run["status"] == "interrupted"
