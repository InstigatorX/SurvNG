"""Stop the server so the process supervisor restarts SurvNG.

Some failures leave the application unable to recover in-process: a reload
that could not finish stopping the previous manager generation refuses to
start another one, because the old generation may still own cameras,
recorders or inference workers. Before this module the process stayed up with
every camera stopped until someone restarted it by hand.

``request_supervisor_restart`` asks the running Uvicorn server to shut down
gracefully, so the normal final shutdown still retries cleanup, and the
entrypoint then exits with ``SUPERVISOR_RESTART_EXIT_STATUS``. A deadline
ends the process with that status if shutdown or a leftover non-daemon
thread holds it longer. systemd
(``Restart=on-failure``) and Docker (``restart: unless-stopped``) both restart
a process that exits with a non-zero status. Raising SIGTERM instead would not
work: Uvicorn re-raises a captured SIGTERM after shutdown, and systemd treats
death by SIGTERM as a clean stop that ``on-failure`` does not restart.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

# EX_TEMPFAIL: a temporary failure that a restart is expected to clear.
SUPERVISOR_RESTART_EXIT_STATUS = 75
# Time allowed for the application shutdown after the server stops accepting
# work. The failures that request a restart can leave non-daemon worker
# threads running, and those would keep a normal interpreter exit waiting
# forever, so the process is ended outright once the deadline passes.
RESTART_SHUTDOWN_BUDGET_SECONDS = 60.0

_lock = threading.Lock()
_server: Any = None
_exit_deadline_seconds: float | None = None
_requested_status: int | None = None
_deadline_timer: threading.Timer | None = None
_force_exit = os._exit


def attach_server(server: Any, *, graceful_shutdown_seconds: float = 30.0) -> None:
    """Register the Uvicorn server the entrypoint is running."""
    global _server, _exit_deadline_seconds
    with _lock:
        _server = server
        _exit_deadline_seconds = (
            max(0.0, float(graceful_shutdown_seconds)) + RESTART_SHUTDOWN_BUDGET_SECONDS
        )


def detach_server() -> None:
    global _server, _requested_status, _exit_deadline_seconds, _deadline_timer
    with _lock:
        timer = _deadline_timer
        _server = None
        _requested_status = None
        _exit_deadline_seconds = None
        _deadline_timer = None
    if timer is not None:
        timer.cancel()


def _exit_after_deadline(status: int, deadline: float) -> None:
    LOGGER.critical(
        "SurvNG did not finish shutting down within %.0f s of a restart request; "
        "exiting with status %d",
        deadline,
        status,
    )
    for handler in logging.getLogger().handlers + logging.getLogger("uvicorn.error").handlers:
        try:
            handler.flush()
        except Exception:  # best effort: the process ends on the next line
            pass
    _force_exit(status)


def request_supervisor_restart(reason: str) -> bool:
    """Stop the server and exit non-zero; True when a managed server will stop.

    Only the first request logs and records the exit status. Without a
    managed server (``--reload`` or an embedding host) the process keeps
    running and an operator must restart it.
    """
    global _requested_status, _deadline_timer
    with _lock:
        first = _requested_status is None
        if first:
            _requested_status = SUPERVISOR_RESTART_EXIT_STATUS
        server = _server
        deadline = _exit_deadline_seconds
        if first and server is not None and deadline is not None:
            _deadline_timer = threading.Timer(
                deadline, _exit_after_deadline, args=(SUPERVISOR_RESTART_EXIT_STATUS, deadline),
            )
            _deadline_timer.daemon = True
            _deadline_timer.name = "survng-restart-deadline"
            _deadline_timer.start()
    if server is None:
        if first:
            LOGGER.critical(
                "SurvNG cannot recover in-process (%s) and is not running under a "
                "managed server; restart SurvNG through its supervisor",
                reason,
            )
        return False
    if first:
        LOGGER.critical(
            "SurvNG cannot recover in-process (%s); stopping so its supervisor "
            "restarts it (exit status %d)",
            reason,
            SUPERVISOR_RESTART_EXIT_STATUS,
        )
    server.should_exit = True
    return True


def requested_exit_status() -> int | None:
    with _lock:
        return _requested_status
