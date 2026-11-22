"""Call-site attribution of slow lock holds and database transactions."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from survng.app import main_database
from survng.app.local_observability import _database_snapshot
from survng.app.main_database import connect_main_database
from survng.app.perf_samples import SlowSiteTable, TimedLock, caller_site


def test_caller_site_names_this_function() -> None:
    site = caller_site(0)

    assert "test_slow_lock_sites.py:" in site
    assert site.endswith(" test_caller_site_names_this_function")


def test_slow_site_table_ignores_fast_operations_and_aggregates_slow_ones() -> None:
    table = SlowSiteTable("probe", 100.0)

    table.record("a.py:1 fast", 99.9)
    table.record("b.py:2 slow", 150.0, write_lock_ms=20.0)
    table.record("b.py:2 slow", 450.0, write_lock_ms=5.0)

    snapshot = table.snapshot()
    assert snapshot["threshold_ms"] == 100.0
    assert [site["site"] for site in snapshot["sites"]] == ["b.py:2 slow"]
    row = snapshot["sites"][0]
    assert row["count"] == 2
    assert row["max_ms"] == 450.0
    assert row["total_ms"] == 600.0
    assert row["last_ms"] == 450.0
    assert row["max_write_lock_ms"] == 20.0
    assert row["last_write_lock_ms"] == 5.0
    assert "last_logged_at" not in row


def test_slow_site_table_is_bounded_and_keeps_the_slowest_sites() -> None:
    table = SlowSiteTable("probe", 0.0, limit=2)

    table.record("one", 10.0)
    table.record("two", 30.0)
    table.record("three", 5.0)   # slower than nothing kept: dropped
    table.record("four", 50.0)   # replaces "one", the smallest maximum

    snapshot = table.snapshot()
    assert [site["site"] for site in snapshot["sites"]] == ["four", "two"]
    assert snapshot["dropped"] == 2


def test_slow_site_warnings_are_rate_limited_per_site(caplog: pytest.LogCaptureFixture) -> None:
    table = SlowSiteTable("probe", 0.0, log_interval_seconds=3600.0)

    with caplog.at_level(logging.WARNING, logger="survng.app.perf_samples"):
        for _ in range(5):
            table.record("hot.py:1 site", 2000.0)
        table.record("other.py:2 site", 2000.0)

    messages = [record.getMessage() for record in caplog.records]
    assert sum("hot.py:1 site" in message for message in messages) == 1
    assert sum("other.py:2 site" in message for message in messages) == 1


def _hold_for(lock: TimedLock, seconds: float) -> None:
    with lock:
        time.sleep(seconds)


def test_timed_lock_attributes_long_holds_to_the_acquiring_site() -> None:
    lock = TimedLock(threading.Lock(), slow_holds=SlowSiteTable("lock", 20.0))

    _hold_for(lock, 0.04)
    with lock:
        pass

    slow = lock.histograms()["slow_holds"]["sites"]
    assert len(slow) == 1
    assert slow[0]["site"].endswith(" _hold_for")
    assert slow[0]["max_ms"] >= 35.0


def test_timed_lock_without_attribution_reports_no_slow_holds() -> None:
    lock = TimedLock(threading.Lock())

    _hold_for(lock, 0.0)

    assert "slow_holds" not in lock.histograms()


def _slow_write(connect, seconds: float) -> None:
    with connect() as connection:
        connection.execute("insert into t(value) values (1)")
        time.sleep(seconds)


def test_slow_transaction_is_attributed_with_its_write_lock_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = SlowSiteTable("database transaction", 20.0)
    monkeypatch.setattr(main_database, "SLOW_TRANSACTIONS", table)
    monkeypatch.setattr(main_database, "SLOW_TRANSACTION_MS", 20.0)
    write_lock = TimedLock(threading.RLock(), slow_holds=SlowSiteTable("write lock", 20.0))
    path = tmp_path / "survng.sqlite3"

    def connect():
        return connect_main_database(path, timeout=1.0, write_lock=write_lock)

    with connect() as connection:
        connection.execute("create table t (value integer not null)")
    _slow_write(connect, 0.05)

    sites = table.snapshot()["sites"]
    assert len(sites) == 1
    assert sites[0]["site"].startswith("main ")
    assert sites[0]["site"].endswith(" _slow_write")
    assert sites[0]["max_write_lock_ms"] >= 40.0
    held = write_lock.histograms()["slow_holds"]["sites"]
    # The write lock is attributed to the statement that first wrote.
    assert held and held[0]["site"].endswith(" _slow_write")
    assert "slow_transactions" in main_database.database_metrics_snapshot()


def test_status_projection_exposes_sanitized_slow_sites() -> None:
    lock = TimedLock(threading.Lock(), slow_holds=SlowSiteTable("lock", 0.0))
    _hold_for(lock, 0.0)

    class Store:
        def lock_histograms(self):
            return {"events": lock.histograms(), "jobs": TimedLock().histograms()}

    raw = {
        "domains": {},
        "slow_transactions": {
            "threshold_ms": 1000.0,
            "dropped": 0,
            "sites": [
                {"site": "main survng/app/x.py:10 f", "count": 1, "max_ms": 1500.0,
                 "max_write_lock_ms": 3.0, "secret": "nope"},
                {"site": "bad\nsite with newline", "count": 1, "max_ms": 9000.0},
            ],
        },
    }
    snapshot = _database_snapshot(raw, lock, Store())

    transactions = snapshot["slow_transactions"]["sites"]
    assert [row["site"] for row in transactions] == ["main survng/app/x.py:10 f"]
    assert "secret" not in transactions[0]
    assert snapshot["main_write_lock"]["slow_holds"]["sites"][0]["site"].endswith(" _hold_for")
    assert snapshot["event_store_locks"]["events"]["slow_holds"]["sites"]
    assert snapshot["event_store_locks"]["jobs"]["slow_holds"] is None
    json.dumps(snapshot, allow_nan=False)
