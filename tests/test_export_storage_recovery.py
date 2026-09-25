"""Transient storage failures must not retire long-lived consumers."""
import sqlite3
import threading
from unittest.mock import Mock, patch

import pytest

from survng.app.media_exports import MediaExportManager


@pytest.fixture
def exports(tmp_path):
    return MediaExportManager(
        tmp_path / "storage", tmp_path / "database", recorder=lambda: None,
        ffmpeg_path=lambda: "ffmpeg", hardware_backend=lambda: "cpu",
    )


@pytest.mark.parametrize("failure", ["lookup", "cleanup", "failure_write", "cancel_write"])
def test_export_worker_survives_storage_failure(exports, failure):
    jobs = [exports.store.create({
        "kind": "recording", "camera_id": "gate", "source": "main",
        "start_epoch": 100, "end_epoch": 110,
    }) for _ in range(2)]
    if failure == "cancel_write":
        exports.store.update(jobs[0]["id"], cancel_requested=1)
    for job in jobs:
        exports._queue.put_nowait(job["id"])
    exports._queue.put_nowait(None)
    target = {
        "lookup": (exports.store, "get"), "cleanup": (exports, "cleanup"),
        "failure_write": (exports.store, "update"), "cancel_write": (exports.store, "update"),
    }[failure]
    original = getattr(*target)
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise sqlite3.OperationalError("injected temporary storage failure")
        return original(*args, **kwargs)

    execute = Mock(side_effect=[RuntimeError("injected transcode failure"), None]
                   if failure == "failure_write" else None)
    with patch.object(*target, side_effect=fail_once), patch.object(exports, "_execute", execute):
        exports._run()
    assert failed
    assert execute.call_count == (1 if failure == "cancel_write" else 2)
    if failure in {"failure_write", "cancel_write"}:
        assert exports.store.get(jobs[0]["id"])["status"] == (
            "failed" if failure == "failure_write" else "cancelled"
        )
    assert exports._active_job_id == ""


def test_export_shutdown_interrupts_failed_lookup_retry(exports):
    entered = threading.Event()
    exports._queue.put_nowait("pending")
    def unavailable(_job_id):
        entered.set()
        raise sqlite3.OperationalError("database unavailable")
    errors = []
    def run():
        try:
            exports._run()
        except BaseException as error:
            errors.append(error)
    with patch.object(exports.store, "get", side_effect=unavailable):
        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(2)
        finally:
            exports._stop.set()
            thread.join(2)
    assert not thread.is_alive()
    assert not errors


def test_export_shutdown_persists_cancellation_when_storage_is_available(exports):
    job = exports.store.create({
        "kind": "recording", "camera_id": "gate", "source": "main",
        "start_epoch": 100, "end_epoch": 110,
    })
    exports._queue.put_nowait(job["id"])
    def cancelled(_job, _cancel):
        exports._stop.set()
        raise InterruptedError
    with patch.object(exports, "_execute", side_effect=cancelled):
        exports._run()
    assert exports.store.get(job["id"])["status"] == "cancelled"
