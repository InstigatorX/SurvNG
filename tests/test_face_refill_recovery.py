"""Face recognition keeps durable pending work recoverable after storage faults."""
import sqlite3
import threading
from unittest.mock import Mock, patch

import pytest


@pytest.mark.parametrize("phase", ["startup", "idle", "after_job"])
def test_face_recognition_retries_failed_refill(tmp_path, phase):
    from survng.app.face_store import FaceStore
    store = FaceStore(tmp_path, recognizer=Mock(enabled=True), start_recognition=False)
    recovered = threading.Event()
    errors = []
    calls = 0

    def refill():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("injected refill failure")
        recovered.set()

    def run():
        try:
            store._recognition_loop()
        except BaseException as error:
            errors.append(error)
            recovered.set()  # Fail immediately, while still joining the thread.

    with patch.object(store, "_queue_pending_recognition", side_effect=refill), patch.object(
        store, "_recognize_observation", return_value=False,
    ):
        if phase == "startup":
            store.start()
            thread = store._recognition_thread
        else:
            store._recognition_refill_needed.set()
            if phase == "after_job":
                store._queue_recognition(1)
            thread = threading.Thread(target=run)
            thread.start()
        try:
            assert recovered.wait(5), "recognition never recovered pending work"
            assert not errors
            assert calls >= 2
            assert thread.is_alive()
        finally:
            store._recognition_stop.set()
            store._recognition_queue.put_nowait(None)
            thread.join(2)
        assert not thread.is_alive()
