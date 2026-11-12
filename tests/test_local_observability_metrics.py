from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.config import AppConfig
from survng.app.local_observability import (
    LocalObservabilityServer,
    build_runtime_status,
    request_runtime_status,
)
from survng.app.main_database import closing_connection_factory
from survng.app.perf_samples import TimedLock
from survng.app.runtime_metrics import EventLoopLagMonitor, LatencyHistogram


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_histogram_bucket_edges_are_inclusive_upper_bounds() -> None:
    histogram = LatencyHistogram((1, 10, 100), window_seconds=60, slot_seconds=10)
    for value in (0.0, 1.0, 1.0001, 10.0, 100.0, 100.5, -5.0, float("nan"), float("inf")):
        histogram.record(value)
    snapshot = histogram.snapshot()
    # -5 clamps to 0; NaN and inf are ignored.
    assert snapshot["bucket_upper_bounds_ms"] == [1.0, 10.0, 100.0]
    assert snapshot["bucket_counts"] == [3, 2, 1, 1]
    assert snapshot["count"] == 7
    assert snapshot["max_ms"] == 100.5
    assert snapshot["recent"]["count"] == 7
    assert snapshot["recent"]["p50_ms"] == 10.0
    # Overflow ranks report the observed window maximum, not infinity.
    assert snapshot["recent"]["p99_ms"] == 100.5
    json.dumps(snapshot, allow_nan=False)


def test_histogram_percentile_is_capped_at_observed_maximum() -> None:
    histogram = LatencyHistogram((1, 100), window_seconds=60, slot_seconds=10)
    histogram.record(3.0)
    assert histogram.snapshot()["recent"]["p50_ms"] == 3.0


def test_histogram_window_expires_with_injected_clock_and_memory_is_fixed() -> None:
    clock, wall = _Clock(), _Clock(5_000.0)
    histogram = LatencyHistogram(
        (1, 10, 100), window_seconds=60, slot_seconds=10, clock=clock, wall_clock=wall
    )
    histogram.record(80.0)
    clock.now += 30
    wall.now += 30
    histogram.record(5.0)
    recent = histogram.snapshot()["recent"]
    assert recent["window_seconds"] == 60
    assert recent["count"] == 2
    assert recent["max_ms"] == 80.0
    assert recent["max_at_epoch_seconds"] == 5_000.0

    clock.now += 45  # The 80 ms slot is now older than the window.
    recent = histogram.snapshot()["recent"]
    assert recent["count"] == 1
    assert recent["max_ms"] == 5.0

    clock.now += 600
    snapshot = histogram.snapshot()
    assert snapshot["recent"]["count"] == 0
    assert snapshot["recent"]["max_ms"] is None
    assert snapshot["recent"]["p99_ms"] is None
    assert snapshot["count"] == 2 and snapshot["max_ms"] == 80.0

    slots = len(histogram._slot_counts)
    for index in range(10_000):
        clock.now += 1.7
        histogram.record(index % 200)
    assert len(histogram._slot_counts) == slots == 6
    assert all(len(counts) == 4 for counts in histogram._slot_counts)
    assert histogram.snapshot()["count"] == 10_002


def test_histogram_concurrent_records_are_not_lost() -> None:
    histogram = LatencyHistogram()
    threads = [
        threading.Thread(target=lambda: [histogram.record(i % 50) for i in range(5_000)])
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    snapshot = histogram.snapshot()
    assert snapshot["count"] == 40_000
    assert sum(snapshot["bucket_counts"]) == 40_000
    assert snapshot["recent"]["count"] == 40_000


def test_event_loop_monitor_records_a_blocked_loop_and_stops_cleanly() -> None:
    async def exercise() -> EventLoopLagMonitor:
        monitor = EventLoopLagMonitor(0.01, stall_log_threshold_ms=10_000)
        monitor.start()
        monitor.start()  # Idempotent while running.
        assert monitor.running
        await asyncio.sleep(0.05)
        time.sleep(0.2)  # Deliberately block the loop.
        await asyncio.sleep(0.05)
        await monitor.stop()
        assert not monitor.running
        await monitor.stop()  # Safe to repeat.
        return monitor

    monitor = asyncio.run(exercise())
    snapshot = monitor.snapshot()
    assert snapshot["running"] is False
    assert snapshot["interval_ms"] == 10.0
    assert snapshot["count"] >= 2
    # The 200 ms block is late by at least ~190 ms: 250 ms bucket or above.
    assert snapshot["max_ms"] >= 150.0
    bounds = snapshot["bucket_upper_bounds_ms"]
    assert sum(snapshot["bucket_counts"][bounds.index(250.0):]) >= 1
    assert snapshot["recent"]["max_ms"] >= 150.0


def test_event_loop_monitor_restarts_on_a_new_loop() -> None:
    monitor = EventLoopLagMonitor(0.01)

    async def run_once() -> None:
        monitor.start()
        await asyncio.sleep(0.03)
        await monitor.stop()

    asyncio.run(run_once())
    first = monitor.snapshot()["count"]
    asyncio.run(run_once())
    assert monitor.snapshot()["count"] > first


def test_socket_snapshot_carries_continuous_lag_and_stops_its_monitor(tmp_path: Path) -> None:
    monitor = EventLoopLagMonitor(0.01)

    async def exercise() -> None:
        server = LocalObservabilityServer(
            lambda: {"schema_version": 1}, tmp_path / "run" / "obs.sock", lag_monitor=monitor
        )
        await server.start()
        try:
            assert monitor.running
            await asyncio.sleep(0.05)
            payload = await asyncio.to_thread(request_runtime_status, server.socket_path)
        finally:
            await server.stop()
        assert not monitor.running
        loop = payload["event_loop"]
        assert loop["probe_lag_ms"] >= 0 and loop["tasks"] >= 2
        lag = loop["lag"]
        assert lag["running"] is True and lag["interval_ms"] == 10.0
        assert lag["count"] >= 1
        assert set(lag["recent"]) == {
            "window_seconds", "count", "max_ms", "max_at_epoch_seconds",
            "p50_ms", "p95_ms", "p99_ms",
        }
        assert len(lag["bucket_counts"]) == len(lag["bucket_upper_bounds_ms"]) + 1

    asyncio.run(exercise())


def test_runtime_status_reports_database_metrics_without_sensitive_text(tmp_path: Path) -> None:
    secret_path = tmp_path / "secret-dir" / "exports.sqlite3"
    secret_path.parent.mkdir()
    connection = sqlite3.connect(secret_path, factory=closing_connection_factory("exports"))
    try:
        with connection:
            connection.execute("select 'secret-sql'").fetchone()
            raise sqlite3.OperationalError("database is locked: secret-sql")
    except sqlite3.OperationalError:
        pass
    write_lock = TimedLock(threading.RLock())
    with write_lock:
        pass
    manager = SimpleNamespace(
        statuses=Mock(return_value=[]),
        detector_status=Mock(return_value={}),
        recorder=SimpleNamespace(retention_status=Mock(return_value={})),
        inference=SimpleNamespace(tracking_limiter=SimpleNamespace(status=Mock(return_value={}))),
        database_write_lock=write_lock,
    )

    payload = build_runtime_status(
        AppConfig(), manager, instance_id="one", uptime_seconds=1, stopping=False
    )

    databases = payload["databases"]
    exports = databases["domains"]["exports"]
    assert exports["busy_errors"] >= 1
    assert exports["transactions_failed"] >= 1
    assert exports["transaction_ms"]["count"] >= 1
    lock = databases["main_write_lock"]
    assert lock["available"] is True
    assert lock["acquisitions"] == 1
    assert lock["hold_ms"]["count"] == 1
    assert lock["wait_ms"]["count"] == 0
    encoded = json.dumps(payload, allow_nan=False)
    assert "secret" not in encoded and str(tmp_path) not in encoded

    # A plain RLock (older wiring or tests) is reported as unavailable.
    manager.database_write_lock = threading.RLock()
    payload = build_runtime_status(
        AppConfig(), manager, instance_id="one", uptime_seconds=1, stopping=False
    )
    assert payload["databases"]["main_write_lock"] == {"available": False}
