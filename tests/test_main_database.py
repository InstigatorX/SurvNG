from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from survng.app.main_database import connect_main_database


class MainDatabaseConnectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "survng.sqlite3"
        self.write_lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("create table values_table (value integer not null)")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _connect(self):
        return connect_main_database(
            self.path,
            timeout=0.01,
            write_lock=self.write_lock,
        )

    def test_writer_waits_for_shared_lock_without_blocking_reader(self) -> None:
        writer_started = threading.Event()
        writer_finished = threading.Event()

        def write() -> None:
            writer_started.set()
            with self._connect() as connection:
                connection.execute("insert into values_table(value) values (1)")
            writer_finished.set()

        with self.write_lock:
            worker = threading.Thread(target=write)
            worker.start()
            self.assertTrue(writer_started.wait(timeout=1.0))
            self.assertFalse(writer_finished.wait(timeout=0.05))
            with self._connect() as connection:
                self.assertEqual(
                    connection.execute("select count(*) from values_table").fetchone()[0],
                    0,
                )

        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(writer_finished.is_set())

    def test_rollback_releases_shared_writer_lock(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            with self._connect() as connection:
                connection.execute("insert into values_table(value) values (1)")
                connection.execute("insert into missing_table(value) values (2)")

        with self._connect() as connection:
            connection.execute("insert into values_table(value) values (3)")
            values = connection.execute("select value from values_table").fetchall()

        self.assertEqual(values, [(3,)])


if __name__ == "__main__":
    unittest.main()
