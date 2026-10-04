"""Worker-scoped, read-only SQLite connections for recurring queue probes."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


_local = threading.local()


@contextmanager
def database_polling_session():
    """Reuse page/statement caches only for this worker's lifetime.

    Probes remain autocommit reads: no snapshot is held between polls, and
    mutations still use the normal locked claim transaction. On worker failure
    or shutdown every connection closes before recovery starts a new session.
    """
    previous = getattr(_local, "connections", None)
    connections = {}
    _local.connections = connections
    try:
        yield
    finally:
        _local.connections = previous
        for connection in connections.values():
            connection.close()


@contextmanager
def polling_connection(path: Path):
    connections = getattr(_local, "connections", None)
    key = str(path.absolute())
    connection = connections.get(key) if connections is not None else None
    if connection is None:
        # mode=ro also prevents accidentally creating a missing ledger.
        connection = sqlite3.connect(
            Path(key).as_uri() + "?mode=ro", uri=True, timeout=10.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        if connections is not None:
            connections[key] = connection
    try:
        yield connection
    finally:
        if connections is None:
            connection.close()
