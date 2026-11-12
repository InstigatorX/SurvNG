from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from survng.app import main_database
from survng.app.event_store import EventStore
from survng.app.face_detection_dataset import FaceDetectionDataset
from survng.app.main_database import (
    ClosingDatabaseConnection,
    closing_connection_factory,
    connect_main_database,
    database_metrics,
    database_metrics_snapshot,
)
from survng.app.media_exports import MediaExportStore
from survng.app.perf_samples import TimedLock
from survng.app.recording_process.index import RecordingIndexMixin
from survng.app.telemetry_store import TelemetryStore


@pytest.fixture()
def database(tmp_path: Path):
    path = tmp_path / "survng.sqlite3"
    lock = TimedLock(threading.RLock())

    def connect():
        return connect_main_database(path, timeout=1.0, write_lock=lock)

    with connect() as connection:
        connection.execute("create table t (value integer not null)")
    return connect, lock


def _main_count() -> int:
    return database_metrics("main").transactions.snapshot()["count"]


def test_write_transaction_records_duration_and_lock_hold(database) -> None:
    connect, lock = database
    before = _main_count()
    holds = lock.histograms()["hold"]["count"]

    with connect() as connection:
        connection.execute("insert into t(value) values (1)")
        time.sleep(0.02)

    assert _main_count() == before + 1
    histograms = lock.histograms()
    assert histograms["hold"]["count"] == holds + 1
    assert histograms["hold"]["max_ms"] >= 15.0
    assert database_metrics("main").transactions.snapshot()["max_ms"] >= 15.0
    # Reads do not take the shared writer lock.
    with connect() as connection:
        connection.execute("select count(*) from t").fetchone()
    assert lock.histograms()["hold"]["count"] == holds + 1


def test_contended_writer_records_wait(database) -> None:
    connect, lock = database
    started = threading.Event()

    def writer() -> None:
        started.set()
        with connect() as connection:
            connection.execute("insert into t(value) values (2)")

    with lock:
        worker = threading.Thread(target=writer)
        worker.start()
        assert started.wait(2.0)
        time.sleep(0.1)
    worker.join(2.0)
    assert not worker.is_alive()
    wait = lock.histograms()["wait"]
    assert lock.histograms()["contended"] >= 1
    assert wait["count"] >= 1 and wait["max_ms"] >= 50.0


def test_exception_rolls_back_releases_lock_and_reraises_original(database) -> None:
    connect, lock = database
    failed = database_metrics("main").snapshot()["transactions_failed"]

    class Original(Exception):
        pass

    with pytest.raises(Original) as raised:
        with connect() as connection:
            connection.execute("insert into t(value) values (3)")
            raise Original("boom")
    assert str(raised.value) == "boom"
    assert database_metrics("main").snapshot()["transactions_failed"] == failed + 1
    assert lock.acquire(blocking=False)  # Released, not leaked to this thread.
    lock.release()
    _assert_free_from_other_thread(lock)
    with connect() as connection:
        assert connection.execute("select count(*) from t").fetchone()[0] == 0


def test_sql_error_and_busy_error_are_counted_and_propagate(database) -> None:
    connect, lock = database
    busy = database_metrics("main").snapshot()["busy_errors"]

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with connect() as connection:
            connection.execute("insert into t(value) values (4)")
            connection.execute("insert into missing(value) values (5)")
    assert database_metrics("main").snapshot()["busy_errors"] == busy
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with connect() as connection:
            connection.execute("insert into t(value) values (6)")
            raise sqlite3.OperationalError("database is locked")
    assert database_metrics("main").snapshot()["busy_errors"] == busy + 1
    _assert_free_from_other_thread(lock)


def test_metric_failures_cannot_break_or_mask_a_transaction(database, monkeypatch) -> None:
    connect, lock = database

    def explode(*_args, **_kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(main_database.DatabaseMetrics, "record_transaction", explode)
    monkeypatch.setattr(lock._hold_histogram, "record", explode)
    monkeypatch.setattr(lock._wait_histogram, "record", explode)
    failures = main_database._METRIC_FAILURES

    with connect() as connection:
        connection.execute("insert into t(value) values (7)")
    with pytest.raises(ValueError, match="original"):
        with connect() as connection:
            connection.execute("insert into t(value) values (8)")
            raise ValueError("original")

    assert main_database._METRIC_FAILURES >= failures + 2
    assert lock.histograms()["metric_failures"] >= 2
    _assert_free_from_other_thread(lock)
    with connect() as connection:
        assert connection.execute("select value from t").fetchall() == [(7,)]


def test_domain_factories_are_labelled_and_bounded(tmp_path: Path) -> None:
    class _Index(RecordingIndexMixin):
        def __init__(self, path):
            self.index_path = path

    events = EventStore(tmp_path)
    dataset = FaceDetectionDataset.__new__(FaceDetectionDataset)
    dataset.database_path = tmp_path / "face-dataset.sqlite3"
    expected = {
        "main": events._connect,
        "jobs": events._connect_jobs,
        "telemetry": TelemetryStore(tmp_path)._connect,
        "recording_index": _Index(tmp_path / "recording-index.sqlite3")._index_connection,
        "exports": MediaExportStore(tmp_path)._connect,
        "face_dataset": dataset._connect,
    }
    for domain, factory in expected.items():
        connection = factory()
        assert isinstance(connection, ClosingDatabaseConnection)
        assert connection.metrics_domain == domain
        before = database_metrics(domain).transactions.snapshot()["count"]
        with connection:
            connection.execute("select 1").fetchone()
        assert database_metrics(domain).transactions.snapshot()["count"] == before + 1

    snapshot = database_metrics_snapshot()
    assert set(expected) <= set(snapshot["domains"])
    assert closing_connection_factory("jobs") is closing_connection_factory("jobs")
    with pytest.raises(ValueError):
        closing_connection_factory("SELECT * FROM events")


def _assert_free_from_other_thread(lock) -> None:
    result: list[bool] = []

    def probe() -> None:
        acquired = lock.acquire(blocking=False)
        result.append(acquired)
        if acquired:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(2.0)
    assert result == [True]
