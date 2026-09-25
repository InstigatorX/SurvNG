from __future__ import annotations

import threading

import pytest

from survng.app.media_sessions import (
    MediaAdmissionPolicy,
    MediaResourceClass,
    MediaSessionAdmissionError,
    MediaSessionKind,
    MediaSessionManager,
    MediaSessionRequest,
)


def _request(camera_id: str = "gate") -> MediaSessionRequest:
    return MediaSessionRequest(
        MediaSessionKind.GO2RTC_WEBRTC,
        camera_id=camera_id,
        source="live",
        resources={MediaResourceClass.RELAY: 1},
    )


def test_media_sessions_enforce_global_and_camera_capacity_atomically() -> None:
    manager = MediaSessionManager(
        MediaAdmissionPolicy(
            global_limits={MediaResourceClass.RELAY: 2},
            per_camera_limits={MediaResourceClass.RELAY: 1},
        )
    )
    first = manager.acquire(_request(), blocking=False)

    with pytest.raises(MediaSessionAdmissionError, match="camera gate"):
        manager.acquire(_request(), blocking=False)

    second = manager.acquire(_request("yard"), blocking=False)
    with pytest.raises(MediaSessionAdmissionError, match="capacity"):
        manager.acquire(_request("door"), blocking=False)

    snapshot = manager.snapshot()
    assert snapshot["active"] == 2
    assert snapshot["by_resource"]["relay"]["used"] == 2
    first.close()
    second.close()
    assert manager.wait_idle(0.1)


def test_media_session_waiter_wakes_on_release() -> None:
    manager = MediaSessionManager(
        MediaAdmissionPolicy(global_limits={MediaResourceClass.RELAY: 1})
    )
    first = manager.acquire(_request(), blocking=False)
    acquired = threading.Event()
    leases = []

    def wait_for_capacity() -> None:
        leases.append(manager.acquire(_request("yard"), timeout=1.0))
        acquired.set()

    waiter = threading.Thread(target=wait_for_capacity)
    waiter.start()
    assert not acquired.wait(0.05)
    first.close()
    assert acquired.wait(1.0)
    leases[0].close()
    waiter.join(1.0)


def test_media_session_cancellation_is_scoped_and_callbacks_run() -> None:
    manager = MediaSessionManager()
    gate = manager.acquire(_request(), blocking=False)
    yard = manager.acquire(_request("yard"), blocking=False)
    reasons = []
    gate.cancellation.add_callback(reasons.append)

    assert manager.cancel_camera("gate", "camera_power_off") == 1
    assert gate.cancelled()
    assert not yard.cancelled()
    assert reasons == ["camera_power_off"]

    gate.close()
    yard.close()
    assert manager.snapshot()["counters"]["cancel_requested"] == 1


def test_media_session_snapshots_do_not_expose_request_metadata() -> None:
    manager = MediaSessionManager()
    lease = manager.acquire(
        MediaSessionRequest(
            MediaSessionKind.RECORDING_REMUX,
            camera_id="gate",
            resources={MediaResourceClass.REMUX_PROCESS: 1},
            metadata={"url": "rtsp://admin:secret@example/live"},
        ),
        blocking=False,
    )
    lease.set_phase("remuxing")
    lease.attach_process(123)
    lease.add_usage(bytes_out=200, frames=3)

    snapshot = manager.snapshot(include_sessions=True)
    text = repr(snapshot)
    assert "secret" not in text
    assert snapshot["sessions"][0]["phase"] == "remuxing"
    assert snapshot["sessions"][0]["bytes_out"] == 200
    lease.close()


def test_media_session_lease_close_is_idempotent() -> None:
    manager = MediaSessionManager()
    lease = manager.acquire(_request(), blocking=False)
    lease.close()
    lease.close()

    snapshot = manager.snapshot()
    assert snapshot["active"] == 0
    assert snapshot["counters"]["completed"] == 1
