"""An ONVIF worker stops within a bound even when its camera does not answer."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

from survng.app.config import CameraConfig
from survng.app.onvif_events import OnvifEventListener
from survng.app.onvif_transport import CancellableSession


def _camera(host: str = "example.invalid", port: int = 80) -> CameraConfig:
    return CameraConfig.model_validate({
        "id": "gate",
        "name": "Gate",
        "stream_url": "rtsp://example.invalid/main",
        "onvif": {"enabled": True, "host": host, "port": port, "username": "admin"},
    })


@contextmanager
def _blackhole():
    """A local port whose accept queue is full, so new connects hang (Linux)."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    port = server.getsockname()[1]
    fillers = []
    try:
        for _ in range(8):
            filler = socket.socket()
            filler.setblocking(False)
            try:
                filler.connect(("127.0.0.1", port))
            except BlockingIOError:
                pass
            fillers.append(filler)
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(("127.0.0.1", port))
            hanging = False
        except (socket.timeout, OSError):
            hanging = True
        finally:
            probe.close()
        if not hanging:
            pytest.skip("this kernel does not hold connects on a full accept queue")
        yield port
    finally:
        for filler in fillers:
            filler.close()
        server.close()


@contextmanager
def _silent_server():
    """Accepts connections and never replies; counts accepted connections."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.05)
        while not stop.is_set():
            try:
                accepted.append(server.accept()[0])
            except (socket.timeout, OSError):
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1], accepted
    finally:
        stop.set()
        thread.join(1)
        for connection in accepted:
            connection.close()
        server.close()


def _post_in_thread(session, url, timeout):
    result: dict = {}

    def run() -> None:
        started = time.monotonic()
        try:
            session.post(url, data=b"x", timeout=timeout)
            result["error"] = None
        except Exception as error:  # noqa: BLE001 - the error type is asserted
            result["error"] = error
        result["seconds"] = time.monotonic() - started

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def test_aborting_connects_interrupts_a_connect_that_never_completes():
    with _blackhole() as port:
        session = CancellableSession()
        thread, result = _post_in_thread(session, f"http://127.0.0.1:{port}/onvif/device_service", (10, 10))
        time.sleep(0.3)
        session.abort_connecting()
        thread.join(2)
        assert not thread.is_alive()
        assert result["seconds"] < 1.5
        assert isinstance(result["error"], requests.exceptions.ConnectionError)


def test_aborting_connects_leaves_an_established_request_alone_and_cancel_interrupts_it():
    with _silent_server() as (port, accepted):
        session = CancellableSession()
        thread, result = _post_in_thread(session, f"http://127.0.0.1:{port}/", (3, 10))
        deadline = time.monotonic() + 2
        while not accepted and time.monotonic() < deadline:
            time.sleep(0.02)
        assert accepted
        # Connected and waiting for a reply: a healthy camera's Unsubscribe
        # must not be cut off by the early abort.
        session.abort_connecting()
        thread.join(0.3)
        assert thread.is_alive()
        session.cancel()
        thread.join(2)
        assert not thread.is_alive()
        assert result["seconds"] < 1.5
        assert isinstance(result["error"], requests.exceptions.ConnectionError)


def test_an_interrupt_wakes_a_blocked_read_and_still_allows_a_new_connection():
    with _silent_server() as (port, accepted):
        session = CancellableSession()
        thread, result = _post_in_thread(session, f"http://127.0.0.1:{port}/", (3, 10))
        deadline = time.monotonic() + 2
        while not accepted and time.monotonic() < deadline:
            time.sleep(0.02)
        session.interrupt()
        thread.join(2)
        assert not thread.is_alive() and result["seconds"] < 1.5
        # The woken worker can still open a connection to send Unsubscribe.
        thread, result = _post_in_thread(session, f"http://127.0.0.1:{port}/", (3, 0.3))
        thread.join(2)
        assert len(accepted) == 2


def test_a_cancelled_session_refuses_new_connections():
    with _silent_server() as (port, accepted):
        session = CancellableSession()
        session.cancel()
        started = time.monotonic()
        with pytest.raises(requests.exceptions.ConnectionError):
            session.post(f"http://127.0.0.1:{port}/", data=b"x", timeout=(3, 3))
        assert time.monotonic() - started < 0.5
        time.sleep(0.1)
        assert accepted == []


def test_a_listener_stops_promptly_while_its_camera_does_not_answer(tmp_path):
    """End to end through python-onvif: the first SOAP call hangs in connect."""
    with _blackhole() as port:
        listener = OnvifEventListener(_camera("127.0.0.1", port), Mock(), cache_dir=tmp_path)
        listener.start()
        time.sleep(0.5)
        started = time.monotonic()
        listener.stop()
        assert time.monotonic() - started < 1.5
        assert not listener.running
        assert listener.abandoned_workers == 0


def test_a_worker_stuck_past_its_stop_is_abandoned_fenced_and_replaced():
    callback = Mock()
    listener = OnvifEventListener(_camera(), callback)
    release = threading.Event()
    calls = []
    notification = SimpleNamespace(Topic="motion", Message="motion")
    pullpoint = Mock()
    pullpoint.PullMessages.return_value = SimpleNamespace(NotificationMessage=[notification])
    old_manager, new_manager = Mock(), Mock()
    old_transport = SimpleNamespace(session=Mock())

    def subscribe(*_args, generation=None, stop_event=None, **_kwargs):
        calls.append(generation)
        with listener._lifecycle_lock:
            if len(calls) == 1:
                listener._transport, listener._transport_generation = old_transport, generation
                listener._subscription_manager = old_manager
            else:
                listener._subscription_manager = new_manager
            listener._subscription_generation = generation
        if len(calls) == 1:
            # I/O that neither the stop nor the transport close interrupts.
            release.wait(5)
        else:
            stop_event.wait(5)
        return pullpoint

    with (
        patch.object(listener, "_subscribe", side_effect=subscribe),
        patch("survng.app.onvif_events.STOP_GRACE_SECONDS", 0.01),
        patch("survng.app.onvif_events.STOP_FORCE_SECONDS", 0.1),
    ):
        listener.start()
        time.sleep(0.1)
        stuck = listener._thread
        started = time.monotonic()
        assert listener.wait_stopped(1.0, listener.request_stop()) is True
        assert time.monotonic() - started < 0.5
        assert not listener.running
        assert listener.abandoned_workers == 1
        # The count belongs to the camera, not this listener object, so it
        # survives a manager reload that replaces the listener.
        assert OnvifEventListener(_camera(), Mock()).abandoned_workers == 1

        # The camera can start a new worker while the old one is still blocked.
        listener.start()
        assert listener.running
        deadline = time.monotonic() + 2
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) == 2 and calls[1] != calls[0]

        # The old worker's late reply is fenced: no motion, no state changes.
        release.set()
        stuck.join(2)
        assert not stuck.is_alive()
        callback.assert_not_called()
        assert listener.notifications_received == 0
        assert listener.abandoned_workers == 0
        assert listener.running
        # It released its own subscription, not the new worker's.
        old_manager.Unsubscribe.assert_called_once_with()
        new_manager.Unsubscribe.assert_not_called()
        old_transport.session.close.assert_called()

        listener.stop()
        assert not listener.running


def test_unreachable_endpoints_are_told_apart_from_camera_faults():
    unreachable = RuntimeError(
        "Unknown error: HTTPConnectionPool(host='192.0.2.1', port=8000): Max retries "
        "exceeded with url: /onvif/device_service (Caused by ConnectTimeoutError(...))"
    )
    assert OnvifEventListener._is_unreachable_error(unreachable)
    from urllib3.exceptions import NewConnectionError

    try:
        try:
            raise NewConnectionError(None, "boom")
        except NewConnectionError as cause:
            raise RuntimeError("Unknown error") from cause
    except RuntimeError as wrapped:
        # The cause chain is examined, not only the outer message.
        assert OnvifEventListener._is_unreachable_error(wrapped)
    fault = RuntimeError("Sender not Authorized")
    assert not OnvifEventListener._is_unreachable_error(fault)


def test_speech_is_replaced_in_place_and_not_after_shutdown_claimed_it():
    from survng.app.manager import AppManager

    manager = AppManager.__new__(AppManager)
    manager._lifecycle_lock = threading.RLock()
    manager._stopping = False
    manager._closed = False
    manager._started = True
    manager._speech_swap_lock = threading.Lock()
    manager._speech_closed = False
    old, new = Mock(), Mock()
    manager.speech = old
    manager._mqtt_connected = Mock()
    with patch.object(AppManager, "_build_speech", return_value=new):
        manager.reconfigure_speech(Mock())
    old.stop.assert_called_once_with()
    new.start.assert_called_once_with()
    assert manager.speech is new
    manager._mqtt_connected.assert_called_once_with()

    manager._speech_closed = True
    with patch.object(AppManager, "_build_speech", return_value=Mock()), pytest.raises(RuntimeError):
        manager.reconfigure_speech(Mock())
    assert manager.speech is new


def test_an_environment_proxy_connection_can_be_interrupted_too():
    with _blackhole() as port:
        session = CancellableSession()
        session.proxies = {"http": f"http://127.0.0.1:{port}"}
        thread, result = _post_in_thread(session, "http://192.0.2.10/onvif/device_service", (10, 10))
        time.sleep(0.3)
        session.abort_connecting()
        thread.join(2)
        assert not thread.is_alive() and result["seconds"] < 1.5


def test_a_worker_woken_by_the_interrupt_is_not_abandoned_on_a_spent_budget(tmp_path):
    with _silent_server() as (port, _accepted):
        listener = OnvifEventListener(_camera(), Mock())

        def subscribe(*_args, generation=None, stop_event=None, **_kwargs):
            session = CancellableSession()
            with listener._lifecycle_lock:
                listener._transport = SimpleNamespace(session=session)
                listener._transport_generation = generation
            # Established and waiting on a reply: only the interrupt wakes it.
            session.post(f"http://127.0.0.1:{port}/", data=b"x", timeout=(3, 10))
            return Mock()

        with patch.object(listener, "_subscribe", side_effect=subscribe):
            listener.start()
            time.sleep(0.3)
            assert listener.wait_stopped(0.0, listener.request_stop()) is True
        assert listener.abandoned_workers == 0
        assert not listener.running


def test_settings_saves_are_refused_while_a_saved_change_waits_for_a_restart(monkeypatch):
    from fastapi import HTTPException

    from survng.app import main

    save = Mock()
    monkeypatch.setattr(main, "save_config_with_history", save)
    main.APPLICATION_STOPPING.set()
    try:
        with pytest.raises(HTTPException) as raised:
            main.save_settings_change(main.config, assign_ids=False)
        assert raised.value.status_code == 503
        save.assert_not_called()

        # Without a supervisor the restart is manual, and the Restart button
        # is exactly how to do it.
        main.RESTART_REQUIRED.set()
        with pytest.raises(HTTPException) as raised:
            main.save_settings_change(main.config, assign_ids=False)
        assert "Restart SurvNG" in raised.value.detail
        monkeypatch.setattr(main, "get_manager", Mock())
        monkeypatch.setattr(main, "_active_storage_tasks", lambda _manager: [])
        monkeypatch.setattr(main.threading, "Thread", Mock())
        assert main._request_server_restart()["status"] == "restart_scheduled"
    finally:
        main.APPLICATION_STOPPING.clear()
        main.RESTART_REQUIRED.clear()
        main.SERVER_RESTART_SCHEDULED = False
