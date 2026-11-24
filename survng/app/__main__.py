"""Run SurvNG with optional HTTPS from configured certificate files."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from uvicorn.main import STARTUP_FAILURE

from . import process_exit
from .bind_policy import assert_network_bind_allowed
from .config import load_config
from .local_observability import SOCKET_ENVIRONMENT_VARIABLE, default_socket_path
from .process_logging import uvicorn_log_config
from .tls import uvicorn_tls_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="SurvNG web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--loop", default="asyncio")
    parser.add_argument("--timeout-graceful-shutdown", type=float, default=30.0)
    parser.add_argument(
        "--observability-socket",
        default=str(default_socket_path()),
        help="owner-only Unix socket used by survngctl status",
    )
    args = parser.parse_args()
    os.environ[SOCKET_ENVIRONMENT_VARIABLE] = args.observability_socket
    config = load_config()
    assert_network_bind_allowed(args.host, config)
    tls = uvicorn_tls_kwargs(config, args.port)
    kwargs = {
        "app": "survng.app.main:app",
        "host": args.host,
        "port": int(tls["port"]),
        "reload": args.reload,
        "loop": args.loop,
        "timeout_graceful_shutdown": args.timeout_graceful_shutdown,
        # Forwarded headers are interpreted by SecurityBoundaryMiddleware,
        # which applies the configured trusted-proxy policy dynamically.
        "proxy_headers": False,
        # Adds the redacted, rate-limited stderr handler (the journal).
        "log_config": uvicorn_log_config(),
    }
    if tls.get("ssl_certfile") and tls.get("ssl_keyfile"):
        kwargs["ssl_certfile"] = tls["ssl_certfile"]
        kwargs["ssl_keyfile"] = tls["ssl_keyfile"]
    _run_server(**kwargs)


def _run_server(**kwargs: object) -> None:
    """Run Uvicorn, exiting non-zero when the application asked for a restart.

    Equivalent to ``uvicorn.run`` for one worker, but keeps the ``Server`` so
    ``process_exit.request_supervisor_restart`` can stop it gracefully.
    """
    if kwargs.get("reload"):
        uvicorn.run(**kwargs)  # type: ignore[arg-type]
        return
    server = uvicorn.Server(uvicorn.Config(**kwargs))  # type: ignore[arg-type]
    process_exit.attach_server(
        server,
        graceful_shutdown_seconds=float(kwargs.get("timeout_graceful_shutdown") or 30.0),  # type: ignore[arg-type]
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        sys.exit(STARTUP_FAILURE)
    status = process_exit.requested_exit_status()
    if status is not None:
        sys.exit(status)


if __name__ == "__main__":
    main()
