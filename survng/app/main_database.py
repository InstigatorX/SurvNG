"""Connection support for SurvNG's single shared SQLite writer."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from .perf_samples import SlowSiteTable, caller_site
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


# Transactions slower than this are attributed to their call site (diagnostic).
SLOW_TRANSACTION_MS = 1000.0
# A statement at least this slow inside a transaction has its call site kept.
SLOW_STATEMENT_MS = 100.0
# A slow read that never took the shared write lock blocks no other writer, so
# it is recorded for status but logged at DEBUG; holding the write lock this
# long is a WARNING.
def held_write_lock_too_long(_duration_ms: float, extra: dict[str, float]) -> bool:
    return extra.get("write_lock_ms", 0.0) >= SLOW_TRANSACTION_MS


SLOW_TRANSACTIONS = SlowSiteTable(
    "database transaction",
    SLOW_TRANSACTION_MS,
    warn_when=held_write_lock_too_long,
)

# Startup recovery (repairing orphaned rows, backfills, re-queueing work a
# crashed process held) belongs to a process start, not to a store instance:
# a settings reload rebuilds every store over the same database, and repeating
# full-table repairs there held the shared write lock for seconds. A database
# counts as recovered by this process only when both hold:
#   - its file (path, device, inode) was registered after a recovery commit;
#   - it carries this process's token, written in that same transaction.
# A replaced file fails the first; a recreated file that reused the inode
# number fails the second.
_STARTUP_RECOVERED: set[tuple[str, str, int, int]] = set()
_STARTUP_RECOVERED_LOCK = threading.Lock()
_PROCESS_RECOVERY_TOKEN = uuid.uuid4().hex


def _database_identity(scope: str, path: Path) -> tuple[str, str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (scope, str(path.resolve()), stat.st_dev, stat.st_ino)


def startup_recovery_pending(
    connection: sqlite3.Connection, scope: str, path: Path,
) -> bool:
    """True until this process recovered ``scope`` on this database file."""
    identity = _database_identity(scope, Path(path))
    with _STARTUP_RECOVERED_LOCK:
        registered = identity is not None and identity in _STARTUP_RECOVERED
        token = _PROCESS_RECOVERY_TOKEN
    if not registered:
        return True
    try:
        row = connection.execute(
            "select process_token from survng_startup_recovery where scope = ?", (scope,)
        ).fetchone()
    except sqlite3.OperationalError:  # table not created yet: never recovered
        return True
    return row is None or row[0] != token


def stamp_startup_recovery(connection: sqlite3.Connection, scope: str) -> None:
    """Record, inside the recovery transaction, that this process recovered ``scope``."""
    connection.execute(
        "create table if not exists survng_startup_recovery "
        "(scope text primary key, process_token text not null)"
    )
    connection.execute(
        "insert or replace into survng_startup_recovery(scope, process_token) values (?, ?)",
        (scope, _PROCESS_RECOVERY_TOKEN),
    )


def mark_startup_recovered(scope: str, path: Path) -> None:
    """Register the file once the transaction holding the stamp has committed."""
    identity = _database_identity(scope, Path(path))
    if identity is not None:
        with _STARTUP_RECOVERED_LOCK:
            _STARTUP_RECOVERED.add(identity)


def forget_startup_recovery() -> None:
    """Behave as a new process would (tests simulate restarts with this)."""
    global _PROCESS_RECOVERY_TOKEN
    with _STARTUP_RECOVERED_LOCK:
        _STARTUP_RECOVERED.clear()
        _PROCESS_RECOVERY_TOKEN = uuid.uuid4().hex



class UnsupportedDatabaseError(RuntimeError):
    """A database older than the minimum this build opens (docs/current-only.md)."""


def require_schema_columns(
    connection: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> None:
    """Stop on a table SurvNG 1.2.0 never opened, before anything relies on it.

    Startup no longer adds columns that 1.2.0 already added, so an older table
    would otherwise fail later with an unexplained SQLite error.
    """
    present = {str(row[1]) for row in connection.execute(f"pragma table_info({table})")}
    missing = [column for column in columns if column not in present]
    if missing:
        raise UnsupportedDatabaseError(
            f"Database table {table} predates SurvNG 1.2.0 (missing {', '.join(missing)}). "
            "Open it with SurvNG 1.2.x first; see docs/current-only.md."
        )


def create_index_logged(connection: sqlite3.Connection, name: str, sql: str) -> None:
    """Run ``create index if not exists``, logging a first-time build.

    Building an index reads its whole table while holding the write lock.
    Stores create their indexes while the manager is constructed, before any
    camera starts, and the statement is a no-op afterwards; on a large
    database the first start after an upgrade is still slow, so say why.
    """
    exists = connection.execute(
        "select 1 from sqlite_master where type = 'index' and name = ?", (name,)
    ).fetchone()
    if exists:
        return
    LOGGER.info("Building database index %s once; this reads its whole table", name)
    started = time.perf_counter()
    connection.execute(sql)
    elapsed = time.perf_counter() - started
    # A build on an empty or small table is instant; only a slow one is worth
    # a warning (and the journal).
    LOGGER.log(
        logging.WARNING if elapsed >= 1.0 else logging.INFO,
        "Built database index %s in %.1f s", name, elapsed,
    )

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
        "slow_transactions": SLOW_TRANSACTIONS.snapshot(),
    }


def _record_transaction_safely(
    domain: str,
    started: float | None,
    error: BaseException | None,
    site: str | None = None,
    timings_ms: dict[str, float] | None = None,
    slowest_site: str = "",
) -> None:
    """Metrics never raise into, or replace an exception of, a transaction."""
    global _METRIC_FAILURES
    if started is None:
        return
    try:
        duration_ms = (time.perf_counter() - started) * 1000.0
        database_metrics(domain).record_transaction(duration_ms, error)
        if site is not None and duration_ms >= SLOW_TRANSACTION_MS:
            SLOW_TRANSACTIONS.record(
                f"{domain} {site}", duration_ms, slowest_site=slowest_site,
                **(timings_ms or {"write_lock_ms": 0.0}),
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
    _transaction_site: str | None = None
    # Milliseconds this connection held the shared write lock (main only).
    _write_lock_held_ms = 0.0
    # Milliseconds spent committing or rolling back.
    _commit_ms = 0.0

    def __enter__(self) -> "ClosingDatabaseConnection":
        result = super().__enter__()
        self._transaction_site = caller_site(1)
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
            # sqlite3's own __exit__ commits (or rolls back) without calling
            # the Python-level commit(), so its cost is measured here.
            exit_started = time.perf_counter()
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                self._commit_ms += (time.perf_counter() - exit_started) * 1000.0
                self.close()
        except BaseException as exit_error:
            error = exit_error
            raise
        finally:
            _record_transaction_safely(
                self.metrics_domain,
                self._transaction_started,
                error,
                self._transaction_site,
                *self._transaction_timings(),
            )

    def _transaction_timings(self) -> tuple[dict[str, float], str]:
        return {"write_lock_ms": self._write_lock_held_ms}, ""


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
    _write_lock_acquired_at = 0.0
    # Where a slow transaction's time went (diagnostic): inside SQLite
    # statements, in COMMIT (fsync, WAL checkpoint), or neither (Python work
    # between statements, row fetching, or waiting for the GIL).
    _statements_ms = 0.0
    _slowest_statement_ms = 0.0
    _slowest_statement_site = ""

    def _timed(self, call: Any, *arguments: Any) -> Any:
        started = time.perf_counter()
        try:
            return call(*arguments)
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self._statements_ms += elapsed
            if elapsed > self._slowest_statement_ms:
                self._slowest_statement_ms = elapsed
                if elapsed >= SLOW_STATEMENT_MS:
                    self._slowest_statement_site = caller_site(0)

    def _transaction_timings(self) -> tuple[dict[str, float], str]:
        return {
            "write_lock_ms": self._write_lock_held_ms,
            "statements_ms": self._statements_ms,
            "slowest_statement_ms": self._slowest_statement_ms,
            "commit_ms": self._commit_ms,
        }, self._slowest_statement_site

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
        self._write_lock_acquired_at = time.perf_counter()

    def _release_write_lock(self) -> None:
        if self._writer_lock_held:
            self._writer_lock_held = False
            self._write_lock_held_ms += (
                time.perf_counter() - self._write_lock_acquired_at
            ) * 1000.0
            assert self._write_lock is not None
            self._write_lock.release()

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        self._before_statement(sql)
        return self._timed(super().execute, sql, parameters)

    def executemany(self, sql: str, parameters: Any, /) -> sqlite3.Cursor:
        self._before_statement(sql)
        return self._timed(super().executemany, sql, parameters)

    def executescript(self, sql_script: str, /) -> sqlite3.Cursor:
        self._before_statement(sql_script)
        return self._timed(super().executescript, sql_script)

    def commit(self) -> None:
        started = time.perf_counter()
        try:
            super().commit()
        finally:
            self._commit_ms += (time.perf_counter() - started) * 1000.0
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
