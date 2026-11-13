"""Connection support for SurvNG's single shared SQLite writer."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from .runtime_metrics import LatencyHistogram

LOGGER = logging.getLogger(__name__)
MAX_DATABASE_METRIC_DOMAINS = 16
_DOMAIN_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")

_WRITE_PREFIXES = (
    "alter ",
    "begin immediate",
    "create ",
    "delete ",
    "drop ",
    "insert ",
    "pragma journal_mode",
    "replace ",
    "update ",
    "vacuum",
)


class DatabaseMetrics:
    """Per-database transaction duration and outcome counters (fixed memory)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self.transactions = LatencyHistogram()
        self._failed = 0
        self._busy = 0

    def record_transaction(self, duration_ms: float, error: BaseException | None) -> None:
        busy = isinstance(error, sqlite3.OperationalError) and _is_busy_error(error)
        with self._lock:
            if error is not None:
                self._failed += 1
            if busy:
                self._busy += 1
        self.transactions.record(duration_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            failed, busy = self._failed, self._busy
        return {
            "transactions_failed": failed,
            "busy_errors": busy,
            "transaction_ms": self.transactions.snapshot(),
        }


def _is_busy_error(error: sqlite3.OperationalError) -> bool:
    # sqlite3 reports SQLITE_BUSY/SQLITE_LOCKED as "database is locked",
    # "database table is locked" or "database is busy".
    text = str(error).lower()
    return "locked" in text or "busy" in text


_METRICS: dict[str, DatabaseMetrics] = {}
_METRICS_LOCK = threading.Lock()
_METRIC_FAILURES = 0


def database_metrics(name: str) -> DatabaseMetrics:
    """Return the process-wide metrics for one static database domain name."""
    metrics = _METRICS.get(name)
    if metrics is not None:
        return metrics
    if not _DOMAIN_RE.fullmatch(name):
        raise ValueError(f"invalid database metric domain: {name!r}")
    with _METRICS_LOCK:
        metrics = _METRICS.get(name)
        if metrics is None:
            if len(_METRICS) >= MAX_DATABASE_METRIC_DOMAINS:
                raise ValueError("too many database metric domains")
            metrics = _METRICS[name] = DatabaseMetrics(name)
        return metrics


def database_metrics_snapshot() -> dict[str, Any]:
    with _METRICS_LOCK:
        domains = dict(_METRICS)
    return {
        "metric_failures": _METRIC_FAILURES,
        "domains": {name: metrics.snapshot() for name, metrics in sorted(domains.items())},
    }


def _record_transaction_safely(
    domain: str, started: float | None, error: BaseException | None
) -> None:
    """Metrics never raise into, or replace an exception of, a transaction."""
    global _METRIC_FAILURES
    if started is None:
        return
    try:
        database_metrics(domain).record_transaction(
            (time.perf_counter() - started) * 1000.0, error
        )
    except Exception as metric_error:
        _METRIC_FAILURES += 1
        if _METRIC_FAILURES == 1:
            LOGGER.warning(
                "database metric recording failed (%s)", type(metric_error).__name__
            )


class ClosingDatabaseConnection(sqlite3.Connection):
    """A transaction context that also owns and closes its connection."""

    metrics_domain = "other"
    _transaction_started: float | None = None

    def __enter__(self) -> "ClosingDatabaseConnection":
        result = super().__enter__()
        self._transaction_started = time.perf_counter()
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        error = exc_value if isinstance(exc_value, BaseException) else None
        try:
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                self.close()
        except BaseException as exit_error:
            error = exit_error
            raise
        finally:
            _record_transaction_safely(
                self.metrics_domain, self._transaction_started, error
            )


_FACTORIES: dict[str, type[ClosingDatabaseConnection]] = {}


def closing_connection_factory(domain: str) -> type[ClosingDatabaseConnection]:
    """A ``sqlite3.connect`` factory whose transactions report under ``domain``."""
    factory = _FACTORIES.get(domain)
    if factory is None:
        database_metrics(domain)  # validates the static name and bounds the set
        factory = _FACTORIES.setdefault(
            domain,
            type(
                f"ClosingDatabaseConnection_{domain}",
                (ClosingDatabaseConnection,),
                {"metrics_domain": domain, "__module__": __name__},
            ),
        )
    return factory


class MainDatabaseConnection(ClosingDatabaseConnection):
    """Serialize only write transactions while preserving concurrent WAL reads."""

    metrics_domain = "main"
    _write_lock: threading.RLock | None = None
    _writer_lock_held = False

    def set_write_lock(self, write_lock: threading.RLock) -> None:
        self._write_lock = write_lock

    def _before_statement(self, statement: str) -> None:
        normalized = statement.lstrip().lower()
        if self._writer_lock_held or not normalized.startswith(_WRITE_PREFIXES):
            return
        if self._write_lock is None:
            raise RuntimeError("main database connection has no write lock")
        self._write_lock.acquire()
        self._writer_lock_held = True

    def _release_write_lock(self) -> None:
        if self._writer_lock_held:
            self._writer_lock_held = False
            assert self._write_lock is not None
            self._write_lock.release()

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        self._before_statement(sql)
        return super().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        self._before_statement(sql)
        return super().executemany(sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        self._before_statement(sql_script)
        return super().executescript(sql_script)

    def commit(self) -> None:
        try:
            super().commit()
        finally:
            self._release_write_lock()

    def rollback(self) -> None:
        try:
            super().rollback()
        finally:
            self._release_write_lock()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._release_write_lock()


def connect_main_database(
    database_path: Path,
    *,
    timeout: float,
    write_lock: threading.RLock,
) -> MainDatabaseConnection:
    connection = sqlite3.connect(
        database_path,
        timeout=timeout,
        factory=MainDatabaseConnection,
    )
    connection.set_write_lock(write_lock)
    return connection
