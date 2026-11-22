"""Scene writes that run under the main write lock must use indexes.

Production timings attributed a 19 s write-lock hold to the acquisition
snapshot refresh (it walked every observation without a snapshot) and 3-4 s
holds to subject joins (full scans of appearance_embeddings).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from survng.app.appearance_index import AppearanceIndex
from survng.app.event_store import EventStore
from survng.app.event_store.scene_acquisition import REFRESH_SAMPLE_OBSERVATION_SNAPSHOTS_SQL
from survng.app.event_store.scenes import (
    JOIN_SUBJECT_ALIASES_SQL,
    JOIN_SUBJECT_EMBEDDINGS_SQL,
)


def _plan(db_path: Path, sql: str) -> list[str]:
    connection = sqlite3.connect(db_path)
    try:
        return [row[3] for row in connection.execute("explain query plan " + sql, ("a", "b"))]
    finally:
        connection.close()


def test_scene_write_statements_search_indexes(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    AppearanceIndex(store.db_path)

    refresh = _plan(store.db_path, REFRESH_SAMPLE_OBSERVATION_SNAPSHOTS_SQL)
    assert not any("acquired_observation_snapshot" in step for step in refresh), refresh
    assert any(
        step.startswith("SEARCH acquired_observations USING INDEX sqlite_autoindex")
        for step in refresh
    ), refresh

    for sql, index in (
        (JOIN_SUBJECT_EMBEDDINGS_SQL, "idx_appearance_scene_object"),
        (JOIN_SUBJECT_ALIASES_SQL, "scene_track_alias_object"),
    ):
        steps = _plan(store.db_path, sql)
        assert not any(step.startswith("SCAN ") for step in steps), steps
        assert any(index in step for step in steps), steps


def test_acquisition_refresh_still_fills_only_missing_snapshots(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    with sqlite3.connect(store.db_path) as connection:
        connection.executemany(
            "insert into acquired_observations values(?,?,?,?,?,?)",
            [("o1", "gate", 1.0, "{}", "", ""), ("o2", "gate", 1.0, "{}", "kept.jpg", ""),
             ("o3", "gate", 1.0, "{}", "", "")],
        )
        connection.executemany(
            "insert into acquired_sample_observations values(?,?)",
            [("s1", "o1"), ("s1", "o2"), ("s2", "o3")],
        )
        connection.execute(REFRESH_SAMPLE_OBSERVATION_SNAPSHOTS_SQL, ("new.jpg", "s1"))
        rows = dict(connection.execute("select id, snapshot_path from acquired_observations"))

    assert rows == {"o1": "new.jpg", "o2": "kept.jpg", "o3": ""}
