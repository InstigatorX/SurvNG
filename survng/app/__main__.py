"""Run SurvNG with optional HTTPS from configured certificate files."""

from __future__ import annotations

import argparse

import uvicorn

from .config import load_config
from .tls import uvicorn_tls_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="SurvNG web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--loop", default="asyncio")
    parser.add_argument("--timeout-graceful-shutdown", type=float, default=30.0)
    args = parser.parse_args()
    config = load_config()
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
    }
    if tls.get("ssl_certfile") and tls.get("ssl_keyfile"):
        kwargs["ssl_certfile"] = tls["ssl_certfile"]
        kwargs["ssl_keyfile"] = tls["ssl_keyfile"]
    uvicorn.run(**kwargs)


if __name__ == "__main__":
    main()
