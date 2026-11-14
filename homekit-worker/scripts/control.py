#!/usr/bin/env python3
"""Owner-only lab client. Pairing material is displayed only on explicit request."""
import argparse
import json
from pathlib import Path
import socket
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--state-dir", required=True, type=Path)
parser.add_argument("command", choices=["status", "pairing", "motion"])
parser.add_argument("--seconds", type=int, default=10)
args = parser.parse_args()
request = {"version": 1, "id": uuid.uuid4().hex, "method": args.command, "params": {"seconds": args.seconds}}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(5)
    client.connect(str(args.state_dir / "control.sock"))
    client.sendall(json.dumps(request).encode() + b"\n")
    response = b""
    while b"\n" not in response:
        chunk = client.recv(4096)
        if not chunk or len(response) + len(chunk) > 65536:
            raise SystemExit("invalid control response")
        response += chunk
value = json.loads(response)
if value.get("id") != request["id"] or value.get("version") != 1:
    raise SystemExit("invalid control response")
if value.get("error"):
    raise SystemExit(value["error"])
print(json.dumps(value["result"], indent=2))
