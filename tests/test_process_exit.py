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
