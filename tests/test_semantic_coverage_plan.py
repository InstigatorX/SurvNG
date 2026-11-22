"""Semantic coverage counts must come from covering indexes.

``SemanticIndex.coverage`` runs on every semantic status read (telemetry,
runtime monitor, status API). Without a covering index, count(distinct
event_id) for one model generation read every embedding row, each carrying a
multi-KB vector blob: about 4 s per call on a production database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.semantic_search import SemanticIndex, SemanticModelIdentity

IDENTITY = SemanticModelIdentity("test", "model-a", "prep-a", 2)


def _plan(db_path: Path, sql: str, parameters: tuple) -> list[str]:
    connection = sqlite3.connect(db_path)
    try:
        return [row[3] for row in connection.execute("explain query plan " + sql, parameters)]
    finally:
        connection.close()


def test_coverage_queries_use_covering_indexes(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    index = SemanticIndex(store.db_path)

    for identity in (IDENTITY, None):
        sql, parameters = index.coverage_query(identity)
        steps = _plan(store.db_path, sql, parameters)
        with_table = [step for step in steps if "semantic_embeddings" in step]
        assert with_table, steps
        assert all("COVERING INDEX" in step for step in with_table), (identity, steps)


def test_coverage_counts_rows_and_distinct_events_per_generation(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    index = SemanticIndex(store.db_path)
    first = store.add_event("gate", "motion")["id"]
    second = store.add_event("gate", "motion")["id"]
    connection = sqlite3.connect(store.db_path)
    rows = [
        (first, "a", "model-a", "prep-a"),
        (first, "b", "model-a", "prep-a"),
        (second, "c", "model-a", "prep-a"),
        (second, "d", "model-b", "prep-a"),
    ]
    connection.executemany(
        "insert into semantic_embeddings(event_id,camera_id,captured_at,source_kind,source_key,"
        "implementation,model_fingerprint,preprocessing_fingerprint,embedding_size,embedding_blob,"
        "created_at) values (?, 'gate', '2026-11-22T00:00:00', 'object', ?, 'test', ?, ?, 2, x'00', '2026')",
        rows,
    )
    connection.commit()
    connection.close()

    assert index.coverage(IDENTITY) == {"evidence_count": 3, "event_count": 2}
    assert index.coverage() == {"evidence_count": 4, "event_count": 2}
