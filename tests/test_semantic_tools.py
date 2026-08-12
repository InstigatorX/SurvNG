from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_script(name: str, filename: str):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_script("survng_semantic_index_builder", "build-semantic-index.py")
EVALUATOR = _load_script("survng_semantic_evaluator", "evaluate-semantic-search.py")


def test_event_pages_are_stable_and_complete(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            create table events (
                id integer primary key, camera_id text, kind text,
                snapshot_path text, recording_path text, objects_json text,
                created_at text
            )
            """
        )
        connection.executemany(
            "insert into events values (?, 'gate', 'object', 'x.webp', '', '[]', ?)",
            [
                (1, "2026-08-11T12:00:00+00:00"),
                (2, "2026-08-11T12:00:00+00:00"),
                (3, "2026-08-11T13:00:00+00:00"),
                (4, "2026-08-11T14:00:00+00:00"),
                (5, "2026-08-11T14:00:00+00:00"),
            ],
        )

    pages = list(BUILDER._event_pages(database, 2))

    assert [[row["id"] for row in page] for page in pages] == [[5, 4], [3, 2], [1]]


def test_evaluator_reports_metrics_and_source() -> None:
    def search(query: str, limit: int):
        assert query == "white delivery truck"
        assert limit == 10
        return [
            {"score": 0.9, "event": {"id": 20}, "snapshot_url": "/20"},
            {"score": 0.8, "event": {"id": 30}, "snapshot_url": "/30"},
            {"score": 0.7, "event": {"id": 10}, "snapshot_url": "/10"},
        ], 12.345

    report = EVALUATOR.evaluate(
        {
            "name": "reviewed",
            "queries": [{
                "id": "truck",
                "query": "white delivery truck",
                "judged": True,
                "relevant_event_ids": [10, 20],
            }],
        },
        search=search,
        source={"kind": "local_generation", "generation": "abc"},
    )

    assert report["source"] == {"kind": "local_generation", "generation": "abc"}
    assert report["base_url"] == ""
    result = report["queries"][0]
    assert result["result_event_ids"] == [20, 30, 10]
    assert result["precision_at_5"] == {"matches": 2, "precision": 0.6667, "recall": 1.0}
    assert result["reciprocal_rank"] == 1.0
    assert report["summary"]["median_latency_ms"] == 12.35


def test_index_builder_retries_only_database_lock_errors() -> None:
    class LockedOnce:
        calls = 0

        def index_event(self, event):
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return 3

    service = LockedOnce()
    sleeps: list[float] = []

    written, retries = BUILDER._index_event_with_retry(
        service, {"id": 1}, sleep=sleeps.append
    )

    assert (written, retries) == (3, 1)
    assert sleeps == [BUILDER.SQLITE_LOCK_RETRY_INITIAL_SECONDS]

    class Broken:
        def index_event(self, event):
            raise sqlite3.OperationalError("no such table")

    try:
        BUILDER._index_event_with_retry(Broken(), {"id": 1}, sleep=sleeps.append)
    except sqlite3.OperationalError as exc:
        assert "no such table" in str(exc)
    else:
        raise AssertionError("non-lock SQLite errors must not be retried")


def test_example_benchmark_is_valid_json() -> None:
    path = Path(__file__).parents[1] / "docs" / "semantic-search-benchmark.example.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
