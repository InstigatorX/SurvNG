"""Stop the server so the process supervisor restarts SurvNG.

Some failures leave the application unable to recover in-process: a reload
that could not finish stopping the previous manager generation refuses to
start another one, because the old generation may still own cameras,
recorders or inference workers. Before this module the process stayed up with
every camera stopped until someone restarted it by hand.

``request_supervisor_restart`` asks the running Uvicorn server to shut down
gracefully, so the normal final shutdown still retries cleanup, and the
entrypoint then exits with ``SUPERVISOR_RESTART_EXIT_STATUS``. systemd
(``Restart=on-failure``) and Docker (``restart: unless-stopped``) both restart
a process that exits with a non-zero status. Raising SIGTERM instead would not
work: Uvicorn re-raises a captured SIGTERM after shutdown, and systemd treats
death by SIGTERM as a clean stop that ``on-failure`` does not restart.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

# EX_TEMPFAIL: a temporary failure that a restart is expected to clear.
SUPERVISOR_RESTART_EXIT_STATUS = 75

_lock = threading.Lock()
_server: Any = None
_requested_status: int | None = None


def attach_server(server: Any) -> None:
    """Register the Uvicorn server the entrypoint is running."""
    global _server
    with _lock:
        _server = server


def detach_server() -> None:
    global _server, _requested_status
    with _lock:
        _server = None
        _requested_status = None


def request_supervisor_restart(reason: str) -> bool:
    """Stop the server and exit non-zero; True when a managed server will stop.

    Only the first request logs and records the exit status. Without a
    managed server (``--reload`` or an embedding host) the process keeps
    running and an operator must restart it.
    """
    global _requested_status
    with _lock:
        first = _requested_status is None
        if first:
            _requested_status = SUPERVISOR_RESTART_EXIT_STATUS
        server = _server
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
