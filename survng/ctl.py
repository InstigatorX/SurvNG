"""Local SurvNG operator CLI."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

PROTOCOL_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def default_socket_path() -> Path:
    """Resolve the local observer socket without importing the server package."""
    configured = os.environ.get("SURVNG_OBSERVABILITY_SOCKET", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.geteuid() == 0:
        return Path("/run/survng/observability.sock")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir:
        return Path(runtime_dir) / "survng" / "observability.sock"
    return Path("/tmp") / f"survng-{os.geteuid()}" / "observability.sock"


def request_runtime_status(socket_path: Path, *, timeout: float) -> dict:
    """Read the fixed status protocol using only the Python standard library."""
    request = json.dumps(
        {"version": PROTOCOL_VERSION, "command": "status"}, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(max(0.1, float(timeout)))
        client.connect(str(socket_path))
        client.sendall(request)
        with client.makefile("rb") as response_file:
            raw = response_file.readline(MAX_RESPONSE_BYTES + 1)
    if not raw or len(raw) > MAX_RESPONSE_BYTES or not raw.endswith(b"\n"):
        raise RuntimeError("invalid response from SurvNG observability socket")
    response = json.loads(raw)
    if not isinstance(response, dict) or not response.get("ok"):
        detail = response.get("error") if isinstance(response, dict) else None
        raise RuntimeError(str(detail or "SurvNG runtime status is unavailable"))
    payload = response.get("status")
    if not isinstance(payload, dict):
        raise RuntimeError("SurvNG observability response did not contain a status object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="survngctl", description="SurvNG host-local controls")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="read effective in-memory runtime status")
    status.add_argument(
        "--socket",
        type=Path,
        default=default_socket_path(),
        help="observability Unix socket path",
    )
    status.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    status.add_argument("--compact", action="store_true", help="print compact JSON")
    args = parser.parse_args(argv)

    try:
        payload = request_runtime_status(args.socket, timeout=args.timeout)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"survngctl: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            payload,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
            sort_keys=not args.compact,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
