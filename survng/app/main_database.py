"""Connection support for SurvNG's single shared SQLite writer."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


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


class MainDatabaseConnection(sqlite3.Connection):
    """Serialize only write transactions while preserving concurrent WAL reads."""

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

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
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
