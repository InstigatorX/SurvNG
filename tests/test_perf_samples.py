from __future__ import annotations

import threading

from survng.app.perf_samples import TimedLock


def test_timed_lock_samples_only_contended_waits() -> None:
    lock = TimedLock(threading.RLock())
    with lock:
        with lock:
            pass
    assert lock.snapshot()["acquisitions"] == 2
    assert lock.snapshot()["contended"] == 0

    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with lock:
            held.set()
            release.wait(2.0)

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(2.0)
    threading.Timer(0.05, release.set).start()
    with lock:
        pass
    thread.join(2.0)

    snapshot = lock.snapshot()
    assert snapshot["acquisitions"] == 4
    assert snapshot["contended"] == 1
    assert snapshot["wait_total_ms"] >= 40.0
    assert snapshot["wait_p99_ms"] >= 40.0
    assert snapshot["hold_p99_ms"] is not None
    assert lock.acquire(blocking=False)
    lock.release()


def test_timed_lock_non_blocking_failure_is_not_an_acquisition() -> None:
    lock = TimedLock()
    lock.acquire()
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(lock.acquire(blocking=False)))
    thread.start()
    thread.join(2.0)
    lock.release()
    assert result == [False]
    assert lock.snapshot()["acquisitions"] == 1
