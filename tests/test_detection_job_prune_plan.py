"""Detection-job pruning must not parse a week of payloads every minute.

The single prune query searched every terminal job newer than the six-hour
discovery window, json_extract-ing each payload and sorting them, to find a
batch; with a backlog it repeated each minute and held the jobs lock for
1-2 s in production. It is now two index-driven queries.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from survng.app.event_store import EventStore
from survng.app.event_store.jobs import (
    PRUNE_EXPIRED_DISCOVERY_JOBS_SQL,
    PRUNE_EXPIRED_JOBS_SQL,
    _UNFINISHED_SPEECH_LOOK_SQL,
)


def test_discovery_prune_walks_only_its_partial_index(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    connection = sqlite3.connect(store.jobs_db_path)
    try:
        discovery = [row[3] for row in connection.execute(
            "explain query plan " + PRUNE_EXPIRED_DISCOVERY_JOBS_SQL, ("cutoff", 250)
        )]
        expired = [row[3] for row in connection.execute(
            "explain query plan " + PRUNE_EXPIRED_JOBS_SQL, ("cutoff", 250)
        )]
    finally:
        connection.close()

    assert discovery == [
        "SEARCH detection_jobs USING INDEX idx_detection_jobs_discovery_terminal (updated_at<?)"
    ]
    assert any("idx_detection_jobs_terminal" in step for step in expired), expired


def test_failed_speech_looks_read_only_open_obligations(tmp_path: Path) -> None:
    """Run every few seconds per camera; must not parse each failed payload."""
    store = EventStore(tmp_path)
    connection = sqlite3.connect(store.jobs_db_path)
    try:
        plan = [row[3] for row in connection.execute(
            "explain query plan select id from detection_jobs "
            "INDEXED BY idx_detection_jobs_unfinished_speech_look where camera_id=? and "
            + _UNFINISHED_SPEECH_LOOK_SQL, ("gate",)
        )]
    finally:
        connection.close()
    assert plan == [
        "SEARCH detection_jobs USING INDEX idx_detection_jobs_unfinished_speech_look (camera_id=?)"
    ]

    store.enqueue_detection_job(
        job_id="open", camera_id="gate", dedupe_key="intent:open",
        payload={"event_at": "2026-08-01T12:00:00+00:00",
                 "qualification": {"trigger_source": "speech"}, "existing_event_id": 5},
    )
    with store._connect_jobs() as connection:
        connection.execute("update detection_jobs set state='failed' where id='open'")
    assert [job["id"] for job in store.failed_speech_looks("gate")] == ["open"]
    store.finish_speech_look_job("open")
    assert store.failed_speech_looks("gate") == []


def test_prune_keeps_the_same_rows_as_before(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    now = datetime.now(timezone.utc)
    jobs = {
        # id: (age, state, qualification, extra payload)
        "old": (timedelta(days=8), "completed", {}, {}),
        "recent": (timedelta(days=3), "completed", {}, {}),
        "old-discovery": (timedelta(hours=7), "completed", {"scene_discovery": True}, {}),
        "fresh-discovery": (timedelta(hours=1), "completed", {"scene_discovery": True}, {}),
        "queued-old": (timedelta(days=8), "queued", {}, {}),
        "unfinished-speech": (
            timedelta(days=8), "failed", {"trigger_source": "speech"}, {"existing_event_id": 5},
        ),
    }
    for index, (job_id, (_age, _state, qualification, extra)) in enumerate(jobs.items()):
        store.enqueue_detection_job(
            job_id=job_id, camera_id="gate", dedupe_key=f"intent:{index}",
            payload={"event_at": "2026-08-01T12:00:00+00:00",
                     "qualification": qualification, **extra},
        )
    with store._connect_jobs() as connection:
        for job_id, (age, state, _qualification, _extra) in jobs.items():
            connection.execute(
                "update detection_jobs set state=?, updated_at=? where id=?",
                (state, (now - age).isoformat(), job_id),
            )

    assert store.prune_detection_jobs(force=True) == 2

    with store._connect_jobs() as connection:
        remaining = {row["id"] for row in connection.execute("select id from detection_jobs")}
    assert remaining == {"recent", "fresh-discovery", "queued-old", "unfinished-speech"}
