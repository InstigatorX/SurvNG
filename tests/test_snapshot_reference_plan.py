"""Snapshot reference probes must use indexes.

``EventStore._delete_snapshot_if_unreferenced`` runs these probes while
holding the event-store lock that every camera's event writes share. On a
large production database a full scan of ``events`` or ``face_observations``
held that lock for 5-10 s per replaced snapshot.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.event_store.store import (
    FACE_SNAPSHOT_REFERENCE_SQL,
    SNAPSHOT_REFERENCE_SQL,
)
from survng.app.face_store.store import FaceStore


def _plan(db_path: Path, sql: str, parameters: tuple[str, ...]) -> list[str]:
    connection = sqlite3.connect(db_path)
    try:
        return [row[3] for row in connection.execute("explain query plan " + sql, parameters)]
    finally:
        connection.close()


def test_snapshot_reference_probes_search_indexes(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    FaceStore(tmp_path, start_recognition=False)

    steps = _plan(store.db_path, SNAPSHOT_REFERENCE_SQL, ("snapshots/gate/a.jpg",) * 7)
    face_steps = _plan(store.db_path, FACE_SNAPSHOT_REFERENCE_SQL, ("snapshots/gate/a.jpg",))

    # "SCAN CONSTANT ROW" is the table-less outer select of the exists() chain.
    table_scans = [
        step for step in steps + face_steps
        if step.startswith("SCAN ") and step != "SCAN CONSTANT ROW"
    ]
    assert table_scans == []
    assert any("events USING" in step and "snapshot_path" in step for step in steps)
    assert any("face_observations USING" in step for step in face_steps)


def test_replaced_snapshot_is_kept_while_referenced_and_deleted_after(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    snapshot = tmp_path / "snapshots" / "gate" / "cover.jpg"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"jpeg")
    event = store.add_event("gate", "motion", snapshot_path=str(snapshot))

    store._delete_snapshot_if_unreferenced(str(snapshot))
    assert snapshot.exists()

    with sqlite3.connect(store.db_path) as connection:
        connection.execute("update events set snapshot_path = '' where id = ?", (event["id"],))
    store._delete_snapshot_if_unreferenced(str(snapshot))
    assert not snapshot.exists()
