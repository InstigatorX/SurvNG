"""An unrecoverable in-process failure stops the server and exits non-zero."""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from survng.app import process_exit


@pytest.fixture(autouse=True)
def _clean_registry():
    process_exit.detach_server()
    yield
    process_exit.detach_server()


def test_restart_request_stops_the_attached_server_once(caplog: pytest.LogCaptureFixture) -> None:
    server = SimpleNamespace(should_exit=False)
    process_exit.attach_server(server)

    with caplog.at_level(logging.CRITICAL, logger="survng.app.process_exit"):
        assert process_exit.request_supervisor_restart("reload aborted") is True
        assert process_exit.request_supervisor_restart("again") is True

    assert server.should_exit is True
    assert process_exit.requested_exit_status() == process_exit.SUPERVISOR_RESTART_EXIT_STATUS
    assert len(caplog.records) == 1
    assert "reload aborted" in caplog.records[0].getMessage()


def test_without_a_managed_server_the_process_keeps_running() -> None:
    assert process_exit.request_supervisor_restart("reload aborted") is False
    assert process_exit.requested_exit_status() == process_exit.SUPERVISOR_RESTART_EXIT_STATUS


class _FakeServer:
    def __init__(self, config: Any, *, on_run=None, started: bool = True) -> None:
        self.config = config
        self.should_exit = False
        self.started = False
        self._on_run = on_run
        self._started = started

    def run(self) -> None:
        self.started = self._started
        if self._on_run is not None:
            self._on_run()


def _serve(monkeypatch: pytest.MonkeyPatch, **server_options: Any) -> list[_FakeServer]:
    import survng.app.__main__ as server_main

    servers: list[_FakeServer] = []

    def make_server(config: Any) -> _FakeServer:
        servers.append(_FakeServer(config, **server_options))
        return servers[-1]

    monkeypatch.setattr(server_main.uvicorn, "Config", lambda **kwargs: kwargs)
    monkeypatch.setattr(server_main.uvicorn, "Server", make_server)
    server_main._run_server(app="survng.app.main:app", port=8088)
    return servers


def test_entrypoint_exits_with_the_restart_status_after_a_requested_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exited:
        _serve(
            monkeypatch,
            on_run=lambda: process_exit.request_supervisor_restart("reload aborted"),
        )
    assert exited.value.code == process_exit.SUPERVISOR_RESTART_EXIT_STATUS


def test_entrypoint_returns_normally_after_an_ordinary_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    servers = _serve(monkeypatch)
    assert servers[0].config == {"app": "survng.app.main:app", "port": 8088}


def test_entrypoint_reports_a_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from uvicorn.main import STARTUP_FAILURE

    with pytest.raises(SystemExit) as exited:
        _serve(monkeypatch, started=False)
    assert exited.value.code == STARTUP_FAILURE


def test_reload_mode_still_uses_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import survng.app.__main__ as server_main

    captured: dict[str, Any] = {}
    monkeypatch.setattr(server_main.uvicorn, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(server_main.uvicorn, "Server", lambda *_a, **_k: pytest.fail("Server"))
    server_main._run_server(app="survng.app.main:app", reload=True)
    assert captured["reload"] is True
    assert sys.modules["survng.app.process_exit"]._server is None


def test_restart_deadline_ends_a_process_that_does_not_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    exited: list[int] = []
    done = threading.Event()
    monkeypatch.setattr(process_exit, "RESTART_SHUTDOWN_BUDGET_SECONDS", 0.05)
    monkeypatch.setattr(process_exit, "_force_exit", lambda status: (exited.append(status), done.set()))
    process_exit.attach_server(SimpleNamespace(should_exit=False), graceful_shutdown_seconds=0)

    process_exit.request_supervisor_restart("reload aborted")

    assert done.wait(5.0)
    assert exited == [process_exit.SUPERVISOR_RESTART_EXIT_STATUS]


def test_no_deadline_is_armed_without_a_managed_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_exit, "_force_exit", lambda status: pytest.fail("forced exit"))
    process_exit.request_supervisor_restart("reload aborted")
    assert process_exit._deadline_timer is None


_SERVE_SCRIPT = """
import sys, threading, time
sys.path.insert(0, sys.argv[1])
from survng.app import process_exit
import survng.app.__main__ as server_main

process_exit.RESTART_SHUTDOWN_BUDGET_SECONDS = 1.0
stuck = sys.argv[2] == "stuck"

async def app(scope, receive, send):
    if scope["type"] != "lifespan":
        return
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            if stuck:
                threading.Thread(target=time.sleep, args=(60,)).start()  # non-daemon
            threading.Timer(0.2, process_exit.request_supervisor_restart, args=("test",)).start()
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            print("lifespan shutdown ran", flush=True)
            await send({"type": "lifespan.shutdown.complete"})
            return

server_main._run_server(
    app=app, host="127.0.0.1", port=0, timeout_graceful_shutdown=0.5, log_level="warning",
)
"""


@pytest.mark.parametrize("mode", ["clean", "stuck"])
def test_real_server_exits_with_the_restart_status(tmp_path, mode: str) -> None:
    import subprocess
    import time
    from pathlib import Path

    script = tmp_path / "serve.py"
    script.write_text(_SERVE_SCRIPT)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).resolve().parents[1]), mode],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == process_exit.SUPERVISOR_RESTART_EXIT_STATUS, completed.stderr
    assert "lifespan shutdown ran" in completed.stdout
    # The stuck thread sleeps 60 s; the deadline (0.5 s + 1 s) ends the process first.
    assert time.monotonic() - started < 20


def test_reload_manager_wires_the_supervisor_restart_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    from survng.app import main

    captured: dict[str, Any] = {}

    class Lifecycle:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def reload(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(main, "ManagerGenerationLifecycle", Lifecycle)
    main.reload_manager(main.config.model_copy(deep=True), persist=False)

    assert captured["hooks"].request_supervisor_restart is process_exit.request_supervisor_restart
