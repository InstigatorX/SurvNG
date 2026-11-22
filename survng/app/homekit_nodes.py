"""Enrolled HomeKit encoder nodes.

The node secret and camera source URLs stay on the authenticated channel.
Status and the installer contain neither.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import platform
import shutil
from pathlib import Path
import secrets
import select
import socket
import stat
import tarfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from . import homekit_build
from .homekit_build import WORKER_ROOT, agent_archive as agent_archive


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("unsafe HomeKit state directory")

ONLINE_SECONDS = 20
SNAPSHOT_LIMIT = 4 * 1024 * 1024
TOKEN_SECONDS = 30 * 60
# Match the local encoder's per-viewer queue bound in camera-source.ts.
FEED_CLIENT_BUFFER = 2 * 1024 * 1024
SYNC_REQUEST_SECONDS = 5 * 60


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _matches(value: str, digest: str) -> bool:
    return hmac.compare_digest(_hash(value), digest)


def _version(value: object) -> str | None:
    if isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def assignment_revision(config, camera) -> str:
    return _hash(json.dumps([camera.source_url(camera.homekit.source), bool(camera.homekit.audio),
        config.capture_rtsp_transport, bool(camera.homekit.allow_software_encoding), camera.homekit.encoder], separators=(",", ":")))


def clean_profile(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    width, height = raw.get("width"), raw.get("height")
    tiers, resolutions = raw.get("tiers"), raw.get("resolutions")
    if type(width) is not int or type(height) is not int or not 2 <= width <= 4096 or not 2 <= height <= 4096:
        return None
    if not isinstance(tiers, list) or not 1 <= len(tiers) <= 3 or not isinstance(resolutions, list) or not 1 <= len(resolutions) <= 8:
        return None
    cleaned = []
    for tier in tiers:
        if not isinstance(tier, dict):
            return None
        item = {}
        for key in ("identifier", "quality", "width", "height", "frameRate", "targetAverageBitrate", "peakBitrate"):
            value = tier.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100000:
                return None
            item[key] = value
        if (item["identifier"] not in {1, 2, 3} or item["quality"] != item["identifier"] + 1
                or any(existing["identifier"] == item["identifier"] for existing in cleaned)
                or not all(2 <= item[key] <= 1920 and item[key] % 2 == 0 for key in ("width", "height"))
                or item["width"] * item["height"] > 1920 * 1080
                or item["frameRate"] not in {15, 30}
                or not 0 < item["targetAverageBitrate"] <= item["peakBitrate"] <= 20000):
            return None
        cleaned.append(item)
    clean_resolutions = []
    for resolution in resolutions:
        if (not isinstance(resolution, list) or len(resolution) != 3
                or not all(isinstance(part, int) and not isinstance(part, bool) and 0 < part <= 4096 for part in resolution)):
            return None
        clean_resolutions.append(resolution)
        if not any(resolution[0] == tier["width"] and resolution[1] == tier["height"]
                and resolution[2] in {15, 30} and resolution[2] <= tier["frameRate"] for tier in cleaned):
            return None
    return {"width": width, "height": height, "tiers": cleaned, "resolutions": clean_resolutions}


def public_api_root(base_url: str, base_path: str = "") -> str:
    """Host the admin reached, plus the /survng prefix when the API lives under it."""
    root = install_origin(str(base_url).strip())
    prefix = str(base_path or "").strip().rstrip("/")
    if prefix:
        prefix = "/" + prefix.lstrip("/")
        if not urlsplit(root).path.endswith(prefix):
            root += prefix
    return install_origin(root)


def install_origin(origin: str) -> str:
    parsed = urlsplit(origin.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid installer origin")
    if any(character in origin for character in " \t\r\n'\"`$\\"):
        raise ValueError("invalid installer origin")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def install_script(origin: str, token: str) -> str:
    origin = install_origin(origin)
    if not token or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for character in token):
        raise ValueError("invalid enrollment token")
    return f"""#!/bin/sh
set -eu
umask 077
if [ "$(id -u)" -ne 0 ]; then
  echo "Run this installer as root inside the new container." >&2
  exit 1
fi
ORIGIN='{origin}'
TOKEN='{token}'
if [ "$(uname -m)" != '{platform.machine()}' ]; then
  echo "This encoder must match the server architecture ({platform.machine()})." >&2
  exit 1
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
trap 'exit 1' HUP INT TERM
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl xz-utils libdrm2
if ! command -v node >/dev/null 2>&1 || [ "$(node --version | cut -d. -f1)" != "v24" ]; then
  case "$(uname -m)" in
    x86_64) node_arch=x64 ;;
    aarch64) node_arch=arm64 ;;
    *) echo "This installer supports x86_64 and arm64." >&2; exit 1 ;;
  esac
  curl -fsSL "https://nodejs.org/dist/v24.22.3/node-v24.22.3-linux-${{node_arch}}.tar.xz" -o "$work/survng-node.tar.xz"
  tar -xJf "$work/survng-node.tar.xz" -C /usr/local --strip-components=1
  rm -f "$work/survng-node.tar.xz"
fi
cred=$(curl -fsSL -X POST "$ORIGIN/api/homekit/enroll/$TOKEN")
secret=$(printf '%s\\n' "$cred" | sed -n 's/^SURVNG_NODE_SECRET=//p')
case "$secret" in
  *[!A-Za-z0-9_-]*|"") echo "Enrollment was rejected." >&2; exit 1 ;;
esac
install -d -m 0700 /etc/survng /opt/survng-homekit-node
printf '%s\\nSURVNG_ORIGIN=%s\\nSURVNG_FFMPEG=%s\\n' "$cred" "$ORIGIN" /opt/survng-homekit-node/bin/ffmpeg > /etc/survng/homekit-node.env
chmod 0600 /etc/survng/homekit-node.env
curl -fsSL -H "Authorization: Bearer $secret" -o "$work/survng-homekit-agent.tgz" "$ORIGIN/api/homekit/nodes/agent"
agent_checksum=$(sha256sum "$work/survng-homekit-agent.tgz")
agent_checksum=${{agent_checksum%% *}}
rm -rf /opt/survng-homekit-node/dist /opt/survng-homekit-node/node_modules
tar -xzf "$work/survng-homekit-agent.tgz" -C /opt/survng-homekit-node
rm -f "$work/survng-homekit-agent.tgz"
curl -fsSL -H "Authorization: Bearer $secret" -o "$work/survng-homekit-ffmpeg.tgz" "$ORIGIN/api/homekit/nodes/ffmpeg"
if tar -tzf "$work/survng-homekit-ffmpeg.tgz" | grep -vxE 'ffmpeg|ffprobe'; then
  echo "Encoder tools were rejected." >&2
  exit 1
fi
install -d -m 0755 /opt/survng-homekit-node/bin
tar -xzf "$work/survng-homekit-ffmpeg.tgz" -C /opt/survng-homekit-node/bin
chmod 0755 /opt/survng-homekit-node/bin/ffmpeg /opt/survng-homekit-node/bin/ffprobe
rm -f "$work/survng-homekit-ffmpeg.tgz"
/opt/survng-homekit-node/bin/ffmpeg -hide_banner -encoders > "$work/survng-ffmpeg-encoders"
grep -q libx265 "$work/survng-ffmpeg-encoders"
grep -q libopus "$work/survng-ffmpeg-encoders"
grep -q ' aac ' "$work/survng-ffmpeg-encoders"
grep -q mjpeg "$work/survng-ffmpeg-encoders"
rm -f "$work/survng-ffmpeg-encoders"
curl -fsSL -H "Authorization: Bearer $secret" -o "$work/survng-homekit-runtime.tgz" "$ORIGIN/api/homekit/nodes/runtime"
if tar -tzf "$work/survng-homekit-runtime.tgz" | grep -vxE 'dri|dri/iHD_drv_video.so|lib[A-Za-z0-9._+-]+[.]so([.][A-Za-z0-9._+-]+)?'; then
  echo "Encoder runtime was rejected." >&2
  exit 1
fi
rm -rf /opt/survng-homekit-node/lib
install -d -m 0755 /opt/survng-homekit-node/lib
tar -xzf "$work/survng-homekit-runtime.tgz" -C /opt/survng-homekit-node/lib
rm -f "$work/survng-homekit-runtime.tgz"
node_bin=$(command -v node)
cat > /etc/systemd/system/survng-homekit-node.service <<EOF
[Unit]
Description=SurvNG HomeKit encoder
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/survng/homekit-node.env
Environment=LD_LIBRARY_PATH=/opt/survng-homekit-node/lib
Environment=LIBVA_DRIVERS_PATH=/opt/survng-homekit-node/lib/dri
WorkingDirectory=/opt/survng-homekit-node
ExecStart=$node_bin dist/src/agent.js
Restart=always
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
sync_body=$(curl -fsSL -m 25 -H "Authorization: Bearer $secret" -H "content-type: application/json" -d '{{"name":"encoder"}}' "$ORIGIN/api/homekit/nodes/sync") || sync_body='{{}}'
printf '%s' "$sync_body" | node -e '
const fs = require("fs");
let releases = {{}};
try {{ releases = JSON.parse(fs.readFileSync(0, "utf8")).releases || {{}}; }}
catch {{ console.error("Could not read the encoder tools release metadata."); }}
fs.writeFileSync("/opt/survng-homekit-node/.releases.json", JSON.stringify({{
  agent: process.argv[1], ffmpeg: String(releases.ffmpeg || ""), runtime: String(releases.runtime || ""),
}}), {{ mode: 0o600 }});
' "$agent_checksum"
systemctl daemon-reload
systemctl enable --now survng-homekit-node.service
echo "SurvNG HomeKit encoder is installed and checking in."
"""


class _TierFeed:
    def __init__(self, path: Path, bump) -> None:
        self.path, self.bump = path, bump
        self.clients: set[socket.socket] = set()
        self._outbound: dict[socket.socket, bytearray] = {}
        self.closed = False
        self.lock = threading.Lock()
        self._bound = False
        self.thread = threading.Thread(target=self._accept, name=f"homekit-feed-{path.parent.name}-{path.stem}", daemon=True)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def start(self) -> None:
        if self.path.exists() or self.path.is_symlink():
            self.path.unlink()
        self.server.bind(str(self.path))
        self._bound = True
        os.chmod(self.path, 0o600)
        self.server.listen(4)
        self.server.settimeout(0.5)
        self.thread.start()

    def _accept(self) -> None:
        while not self.closed:
            try:
                client, _ = self.server.accept()
            except (TimeoutError, OSError):
                continue
            client.setblocking(False)
            with self.lock:
                if self.closed:
                    client.close()
                    return
                self.clients.add(client)
            self.bump()

    def wanted(self) -> bool:
        with self.lock:
            dead = []
            for client in self.clients:
                readable, _, _ = select.select([client], [], [], 0)
                if not readable:
                    continue
                try:
                    if not client.recv(1, socket.MSG_PEEK):
                        dead.append(client)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    dead.append(client)
            for client in dead:
                self._drop_client(client)
            return bool(self.clients)

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self.lock:
            dead = []
            for client in self.clients:
                pending = self._outbound.setdefault(client, bytearray())
                try:
                    self._flush(client, pending)
                    # A full kernel buffer is backpressure, not a disconnect.
                    if len(pending) + len(data) > FEED_CLIENT_BUFFER:
                        dead.append(client)
                        continue
                    pending.extend(data)
                    self._flush(client, pending)
                except OSError:
                    dead.append(client)
                    continue
                if not pending:
                    self._outbound.pop(client, None)
            for client in dead:
                self._drop_client(client)

    def _flush(self, client: socket.socket, pending: bytearray) -> None:
        view = memoryview(pending)
        sent = 0
        try:
            while sent < len(view):
                try:
                    count = client.send(view[sent:])
                except InterruptedError:
                    continue
                if count == 0:
                    raise ConnectionError("HomeKit feed closed")
                sent += count
        except BlockingIOError:
            pass
        finally:
            view.release()
            if sent:
                del pending[:sent]

    def _drop_client(self, client: socket.socket) -> None:
        self.clients.discard(client)
        self._outbound.pop(client, None)
        try:
            client.close()
        except OSError:
            pass

    def close(self) -> None:
        with self.lock:
            self.closed = True
        try:
            self.server.close()
        except OSError:
            pass
        with self.lock:
            for client in list(self.clients):
                self._drop_client(client)
        if self.thread.ident is not None:
            self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("HomeKit feed shutdown incomplete")
        if self._bound:
            self.path.unlink(missing_ok=True)
            self._bound = False


@dataclass(frozen=True, eq=False)
class EncoderUploadClaim:
    revision: str


@dataclass
class _SyncRequest:
    nonce: str
    version: str
    expires_at: float
    failed: bool = False

    def pending(self, now: float) -> bool:
        return not self.failed and now < self.expires_at


class NodeRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "nodes.json"
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self.updates = threading.Condition()
        self.generation = 0
        self.closed = True
        self.error: str | None = None
        self.tokens: list[dict] = []
        self.nodes: dict[str, dict] = {}
        self.feeds: dict[tuple[str, int], _TierFeed] = {}
        self.uploads: dict[tuple[str, int], EncoderUploadClaim] = {}
        self.snapshots: dict[str, tuple[threading.Event, Path, str]] = {}
        self.sync_requests: dict[str, _SyncRequest] = {}

    def _bump(self) -> None:
        with self.updates:
            self.generation += 1
            self.updates.notify_all()

    def _load(self) -> None:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            self.tokens = []
            self.nodes = {}
            return
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("unsafe HomeKit node state")
        saved = json.loads(self.path.read_text())
        if not isinstance(saved, dict) or saved.get("version") != 1:
            raise RuntimeError("unsupported HomeKit node state")
        tokens, nodes = saved.get("tokens", []), saved.get("nodes", [])
        if not isinstance(tokens, list) or not isinstance(nodes, list):
            raise RuntimeError("invalid HomeKit node state")
        for node in nodes:
            if (not isinstance(node, dict) or not isinstance(node.get("id"), str)
                    or len(node["id"]) != 16 or any(character not in "0123456789abcdef" for character in node["id"])
                    or not isinstance(node.get("secretHash"), str)
                    or not isinstance(node.get("hardware", ""), str)
                    or not isinstance(node.get("name", ""), str) or not isinstance(node.get("ffmpeg", ""), str)
                    or not math.isfinite(float(node.get("lastSeen") or 0))
                    or any(not isinstance(node.get(key, {}), dict) for key in ("profiles", "backends", "fps", "revisions"))):
                raise RuntimeError("invalid HomeKit node state")
            node["lastSeen"] = float(node.get("lastSeen") or 0)
            node["profiles"] = {key: profile for key, value in node.get("profiles", {}).items() if (profile := clean_profile(value))}
            node["backends"] = {key: value for key, value in node.get("backends", {}).items()
                if isinstance(value, str) and value in {"qsv", "software", "unavailable"}}
            node["runningVersion"] = _version(node.get("runningVersion"))
        for token in tokens:
            if (not isinstance(token, dict) or not isinstance(token.get("hash"), str)
                    or not isinstance(token.get("origin"), str) or not math.isfinite(float(token.get("expires") or 0))):
                raise RuntimeError("invalid HomeKit enrollment state")
            token["expires"] = float(token.get("expires") or 0)
        self.tokens = tokens
        self.nodes = {item["id"]: item for item in nodes}

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("HomeKit registry is closed")

    def _save(self) -> None:
        self._require_open()
        payload = {"version": 1, "tokens": self.tokens, "nodes": list(self.nodes.values())}
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        os.chmod(self.path, 0o600)

    def begin_enrollment(self, origin: str) -> dict:
        origin = install_origin(origin)
        token = secrets.token_urlsafe(32)
        expires = time.time() + TOKEN_SECONDS
        with self.lock:
            self._require_open()
            self.tokens = [item for item in self.tokens if item.get("expires", 0) > time.time()]
            self.tokens.append({"hash": _hash(token), "expires": expires, "origin": origin})
            self._save()
        return {"command": f"curl -fsSL '{origin}/api/homekit/enroll/{token}' | sh", "expires": int(expires)}

    def _take_token(self, token: str) -> None:
        now = time.time()
        found = False
        kept = []
        for item in self.tokens:
            if item.get("expires", 0) <= now:
                continue
            if not found and isinstance(item.get("hash"), str) and _matches(token, item["hash"]):
                found = True
                continue
            kept.append(item)
        self.tokens = kept
        if not found:
            raise KeyError("enrollment token is not valid")

    def enrollment_script(self, token: str, origin: str | None = None) -> str:
        with self.lock:
            self._require_open()
            now = time.time()
            match = next((item for item in self.tokens
                if item.get("expires", 0) > now and isinstance(item.get("hash"), str) and isinstance(item.get("origin"), str)
                and _matches(token, item["hash"])), None)
            if match is None:
                raise KeyError("enrollment token is not valid")
            chosen = install_origin(origin) if origin else match["origin"]
            match["origin"] = chosen
            self._save()
            return install_script(chosen, token)

    def complete_enrollment(self, token: str, name: str) -> dict:
        name = "".join(character for character in name if character.isprintable() and character not in "\r\n")[:64] or "encoder"
        with self.lock:
            self._require_open()
            self._take_token(token)
            node_id = secrets.token_hex(8)
            secret = secrets.token_urlsafe(32)
            self.nodes[node_id] = {"id": node_id, "name": name, "secretHash": _hash(secret), "hardware": "",
                "ffmpeg": "", "lastSeen": 0, "profiles": {}, "backends": {}, "fps": {}}
            self._save()
        return {"id": node_id, "secret": secret,
            "environment": f"SURVNG_NODE_ID={node_id}\nSURVNG_NODE_SECRET={secret}\n"}

    def authenticate(self, secret: str) -> dict | None:
        if not secret:
            return None
        with self.lock:
            if self.closed:
                return None
            for node in self.nodes.values():
                digest = node.get("secretHash")
                if isinstance(digest, str) and _matches(secret, digest):
                    return node
        return None

    def update_report(self, node_id: str, report: dict) -> None:
        with self.lock:
            self._require_open()
            node = self.nodes.get(node_id)
            if node is None:
                raise KeyError(node_id)
            hardware = report.get("hardware")
            if isinstance(hardware, str) and hardware in {"qsv", "software"}:
                node["hardware"] = hardware
            ffmpeg = report.get("ffmpeg")
            if isinstance(ffmpeg, str):
                node["ffmpeg"] = "".join(character for character in ffmpeg if character.isprintable())[:80]
            name = report.get("name")
            if isinstance(name, str) and name.strip():
                node["name"] = "".join(character for character in name if character.isprintable())[:64]
            profiles = report.get("profiles") if isinstance(report.get("profiles"), dict) else {}
            backends = report.get("backends") if isinstance(report.get("backends"), dict) else {}
            fps = report.get("fps") if isinstance(report.get("fps"), dict) else {}
            revisions = report.get("revisions") if isinstance(report.get("revisions"), dict) else {}
            node["profiles"] = {key: profile for key, value in profiles.items() if (profile := clean_profile(value))}
            node["backends"] = {key: value for key, value in backends.items()
                if isinstance(value, str) and value in {"qsv", "software", "unavailable"}}
            node["fps"] = {key: value for key, value in fps.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 120}
            node["revisions"] = {key: value for key, value in revisions.items()
                if isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)}
            node["runningVersion"] = _version(report.get("runningVersion"))
            request = self.sync_requests.get(node_id)
            if request is not None:
                if node["runningVersion"] == request.version:
                    del self.sync_requests[node_id]
                elif report.get("failedSyncRequest") == request.nonce:
                    request.failed = True
            node["lastSeen"] = time.time()
            self._save()

    def request_sync(self, node_id: str) -> None:
        with self.lock:
            self._require_open()
            node = self.nodes[node_id]
        available = available_agent_version()
        with self.lock:
            self._require_open()
            if self.nodes.get(node_id) is not node:
                raise KeyError(node_id)
            if time.time() - node["lastSeen"] > ONLINE_SECONDS:
                raise RuntimeError("HomeKit encoder is offline")
            if available is None:
                raise RuntimeError("HomeKit encoder release is unavailable")
            if node.get("runningVersion") == available:
                raise RuntimeError("HomeKit encoder is already current")
            now = time.monotonic()
            queued = self.sync_requests.get(node_id)
            if queued is not None and queued.version == available and queued.pending(now):
                return
            self.sync_requests[node_id] = _SyncRequest(secrets.token_hex(16), available, now + SYNC_REQUEST_SECONDS)
            self._bump()

    def pending_sync_request(self, node: dict, available: str | None) -> str | None:
        with self.lock:
            if self.closed or self.nodes.get(node["id"]) is not node:
                return None
            request = self.sync_requests.get(node["id"])
            if request is not None and request.version == available and request.pending(time.monotonic()):
                return request.nonce
            return None

    def sync(self, node: dict, report: dict, assignments: list[dict] | Callable[[], list[dict]], timeout: float) -> dict:
        with self.updates:
            seen = self.generation
        with self.lock:
            if self.nodes.get(node["id"]) is not node:
                raise KeyError(node["id"])
            self.update_report(node["id"], report)
        deadline = time.monotonic() + max(0, timeout)
        with self.updates:
            while time.monotonic() < deadline and self.generation == seen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.updates.wait(remaining)
        current = assignments() if callable(assignments) else assignments
        with self.lock:
            if self.nodes.get(node["id"]) is not node:
                raise KeyError(node["id"])
            return self.assignment_payload(node["id"], current)

    def assignment_payload(self, node_id: str, assignments: list[dict]) -> dict:
        with self.lock:
            if self.closed or node_id not in self.nodes:
                raise KeyError(node_id)
            owned = [item for item in assignments if item.get("encoder") == node_id]
            feeds = []
            for item in owned:
                for tier, feed in list(self.feeds.items()):
                    if tier[0] == item["id"] and feed.wanted():
                        feeds.append({"camera": item["id"], "tier": tier[1]})
            snapshots = [camera_id for camera_id, _waiter in self.snapshots.items() if any(item["id"] == camera_id for item in owned)]
            public = [{key: item[key] for key in ("id", "url", "audio", "transport", "allowSoftwareEncoding", "revision")} for item in owned]
            return {"assignments": public, "feeds": feeds, "snapshots": snapshots,
                "snapshotRequests": {camera_id: self.snapshots[camera_id][2] for camera_id in snapshots}}

    def decision(self, camera, revision: str | None = None) -> str:
        encoder = camera.homekit.encoder or "local"
        if encoder == "local":
            return "local"
        with self.lock:
            node = self.nodes.get(encoder)
            if node is None or time.time() - float(node.get("lastSeen") or 0) > ONLINE_SECONDS:
                return "offline"
            if revision is not None and (node.get("revisions") or {}).get(camera.id) != revision:
                return "waiting"
            backend = (node.get("backends") or {}).get(camera.id)
            if node.get("hardware") != "qsv" and not camera.homekit.allow_software_encoding:
                return "software_not_allowed"
            if backend == "software" and not camera.homekit.allow_software_encoding:
                return "software_not_allowed"
            if backend == "unavailable":
                return "unavailable"
            if backend not in {"qsv", "software"}:
                return "waiting"
            if not (node.get("profiles") or {}).get(camera.id):
                return "waiting"
            return "remote"

    def remote_for(self, camera, revision: str | None = None) -> dict | None:
        with self.lock:
            if self.decision(camera, revision) != "remote":
                return None
            node = self.nodes[camera.homekit.encoder]
            profile = node["profiles"][camera.id]
            backend = node["backends"].get(camera.id, "software")
            encoding = {"backend": backend}
            if backend == "software":
                encoding["fallbackReason"] = "qsv_unavailable"
            elif backend == "qsv":
                encoding["device"] = "/dev/dri/renderD128"
            directory = self.prepare_feeds(camera.id, [tier["identifier"] for tier in profile["tiers"]])
            return {"feeds": str(directory), "profile": profile, "encoding": encoding}

    def prepare_feeds(self, camera_id: str, tiers: list[int]) -> Path:
        directory = self.directory / "feeds" / camera_id
        with self.lock:
            self._require_open()
            _private_directory(self.directory / "feeds")
            _private_directory(directory)
            for tier in tiers:
                key = (camera_id, tier)
                existing = self.feeds.get(key)
                if existing is not None:
                    if existing.closed:
                        raise RuntimeError("HomeKit feed shutdown incomplete")
                    continue
                feed = _TierFeed(directory / f"{tier}.sock", self._bump)
                self.feeds[key] = feed
                try:
                    feed.start()
                except BaseException as start_error:
                    try:
                        feed.close()
                    except Exception as close_error:
                        raise RuntimeError("HomeKit feed startup cleanup incomplete") from BaseExceptionGroup(
                            "HomeKit feed startup", [start_error, close_error],
                        )
                    del self.feeds[key]
                    raise
        return directory

    def write_feed(self, camera_id: str, tier: int, data: bytes) -> None:
        with self.lock:
            if not self.closed:
                feed = self.feeds.get((camera_id, tier))
                if feed:
                    feed.write(data)

    def feed_allowed(self, node: dict, camera_id: str, tier: str, assignments: list[dict], revision: str) -> bool:
        if not revision or tier not in {"1", "2", "3"}:
            return False
        with self.lock:
            return not self.closed and self.nodes.get(node["id"]) is node and any(
                item["id"] == camera_id and item["encoder"] == node["id"] and item["revision"] == revision
                for item in assignments)

    def claim_feed(self, node: dict, camera_id: str, tier: str, assignments: list[dict], revision: str) -> EncoderUploadClaim | None:
        with self.lock:
            if not self.feed_allowed(node, camera_id, tier, assignments, revision):
                return None
            key = (camera_id, int(tier))
            feed = self.feeds.get(key)
            if feed is None or not feed.wanted():
                return None
            claim = EncoderUploadClaim(revision)
            self.uploads[key] = claim
            return claim

    def write_upload(self, node: dict, camera_id: str, tier: str, claim: EncoderUploadClaim, data: bytes, assignments: list[dict]) -> bool:
        with self.lock:
            if not self.feed_allowed(node, camera_id, tier, assignments, claim.revision):
                return False
            key = (camera_id, int(tier))
            feed = self.feeds.get(key)
            if self.uploads.get(key) is not claim or feed is None or not feed.wanted():
                return False
            feed.write(data)
            return True

    def release_upload(self, camera_id: str, tier: str, claim: EncoderUploadClaim) -> None:
        with self.lock:
            key = (camera_id, int(tier))
            if self.uploads.get(key) is claim:
                self.uploads.pop(key)

    def encoder_status(self, camera) -> dict | None:
        with self.lock:
            node = self.nodes.get(camera.homekit.encoder)
            if node is None:
                return None
            backend = (node.get("backends") or {}).get(camera.id)
            if backend not in {"qsv", "software"}:
                return None
            status: dict = {"backend": backend}
            if backend == "software":
                status["fallbackReason"] = "qsv_unavailable"
            fps = (node.get("fps") or {}).get(camera.id)
            if isinstance(fps, (int, float)):
                status["sources"] = [{"fps": fps}]
            return status

    def begin_snapshot(self, camera_id: str, destination: Path) -> tuple[threading.Event, Path, str]:
        with self.lock:
            self._require_open()
            if camera_id in self.snapshots:
                raise RuntimeError("snapshot already pending")
            waiter = (threading.Event(), destination, secrets.token_hex(16))
            self.snapshots[camera_id] = waiter
            self._bump()
            return waiter

    def end_snapshot(self, camera_id: str, waiter: tuple) -> None:
        with self.lock:
            if self.snapshots.get(camera_id) is waiter:
                self.snapshots.pop(camera_id)
                self._bump()

    def accept_snapshot(self, node: dict, camera_id: str, data: bytes, assignments: list[dict], request_id: str = "") -> bool:
        if not data or len(data) > SNAPSHOT_LIMIT or not data.startswith(b"\xff\xd8"):
            return False
        with self.lock:
            if self.closed or self.nodes.get(node["id"]) is not node or not any(
                    item["id"] == camera_id and item["encoder"] == node["id"] for item in assignments):
                return False
            waiter = self.snapshots.get(camera_id)
            if waiter is None or waiter[2] != request_id or waiter[0].is_set():
                return False
            temporary = waiter[1].with_suffix(".tmp")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temporary, waiter[1])
            waiter[0].set()
            return True

    def remove(self, node_id: str, assigned: list[str]) -> None:
        if assigned:
            raise RuntimeError("encoder is assigned to a camera")
        with self.lock:
            self._require_open()
            if node_id not in self.nodes:
                raise KeyError(node_id)
            self.nodes.pop(node_id)
            self.sync_requests.pop(node_id, None)
            self._save()
            self._bump()

    def public_status(self, cameras) -> list[dict]:
        with self.lock:
            if not self.nodes:
                return []
        available = available_agent_version()
        now = time.time()
        monotonic_now = time.monotonic()
        with self.lock:
            rows = []
            for node_id, node in self.nodes.items():
                assigned = [camera.id for camera in cameras if camera.homekit.encoder == node_id]
                running = node.get("runningVersion")
                request = self.sync_requests.get(node_id)
                if available is None:
                    update_state = "unavailable"
                elif running == available:
                    update_state = "current"
                elif request is not None and request.version == available:
                    update_state = "syncing" if request.pending(monotonic_now) else "failed"
                else:
                    update_state = "unknown" if running is None else "outdated"
                rows.append({"id": node_id, "name": node.get("name") or "Encoder",
                    "online": now - float(node.get("lastSeen") or 0) <= ONLINE_SECONDS,
                    "hardware": node.get("hardware") if node.get("hardware") in {"qsv", "software"} else "unknown",
                    "lastSeen": int(node.get("lastSeen") or 0), "cameras": assigned,
                    "runningVersion": running, "availableVersion": available, "updateState": update_state})
            return rows

    def close(self) -> None:
        with self._close_lock:
            with self.lock:
                if self.closed and not self.feeds:
                    return
                self.closed = True
                feeds = list(self.feeds.items())
                self.uploads.clear()
                self.snapshots.clear()
                self.sync_requests.clear()
                self._bump()
            first_error = None
            for key, feed in feeds:
                try:
                    feed.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    with self.lock:
                        del self.feeds[key]
            if first_error is not None:
                raise RuntimeError("HomeKit feed shutdown incomplete") from first_error

    def reopen(self) -> None:
        with self.lock:
            if not self.closed:
                return
            if self.feeds:
                raise RuntimeError("HomeKit feed shutdown incomplete")
            try:
                _private_directory(self.directory)
                self._load()
                self.closed = False
                self.error = None
            except (OSError, ValueError, RuntimeError, TypeError):
                # A broken optional integration must not stop camera capture.
                # Preserve the original state for repair; never replace it.
                self.closed = True
                self.error = "node_state_unavailable"
                self.nodes.clear()
                self.tokens.clear()


_release_lock = threading.Lock()
_ffmpeg_stamp_cache: tuple[tuple, str] | None = None


def _file_signature(paths: list[Path]) -> tuple:
    rows = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        info = path.stat()
        rows.append((str(path), info.st_mtime_ns, info.st_size))
    return tuple(rows)


def agent_release(root: Path = WORKER_ROOT) -> tuple[str, bytes]:
    """Read the checksum and archive from the same verified release."""
    release, problem = homekit_build.verified_release(root)
    if release is None:
        raise RuntimeError(problem)
    return homekit_build.read_release_archive(release)


def available_agent_version(root: Path = WORKER_ROOT) -> str | None:
    try:
        release, _ = homekit_build.verified_release(root)
        return homekit_build.release_archive_version(release) if release is not None else None
    except (OSError, RuntimeError):
        return None


def ffmpeg_binaries(ffmpeg_path: str) -> tuple[Path, Path]:
    ffmpeg = Path(shutil.which(ffmpeg_path) or ffmpeg_path).resolve()
    if ffmpeg.name != "ffmpeg" or not ffmpeg.is_file() or ffmpeg.is_symlink():
        raise ValueError("configured ffmpeg is not a regular file")
    ffprobe = ffmpeg.with_name("ffprobe")
    if not ffprobe.is_file() or ffprobe.is_symlink():
        raise ValueError("ffprobe is not beside ffmpeg")
    return ffmpeg, ffprobe


def ffmpeg_release(ffmpeg_path: str) -> str:
    global _ffmpeg_stamp_cache
    ffmpeg, ffprobe = ffmpeg_binaries(ffmpeg_path)
    signature = _file_signature([ffmpeg, ffprobe])
    with _release_lock:
        if _ffmpeg_stamp_cache and _ffmpeg_stamp_cache[0] == signature:
            return _ffmpeg_stamp_cache[1]
    digest = hashlib.sha256()
    for path in (ffmpeg, ffprobe):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    stamp = digest.hexdigest()
    with _release_lock:
        _ffmpeg_stamp_cache = (signature, stamp)
    return stamp


RUNTIME_LIB_DIR = Path("/usr/lib/x86_64-linux-gnu")
RUNTIME_SONAMES = ("libva.so.2", "libva-drm.so.2", "libigdgmm.so.12", "libmfx-gen.so.1.2", "libvpl.so.2")
_runtime_stamp_cache: tuple[tuple, str] | None = None


def _local_library(path: Path, root: Path) -> Path:
    if not path.exists():
        raise ValueError("missing media runtime")
    target = path.resolve()
    if not target.is_file() or target.is_symlink() or not target.is_relative_to(root):
        raise ValueError("media runtime is not a local file")
    return target


def runtime_files(lib_dir: Path = RUNTIME_LIB_DIR) -> list[tuple[str, Path]]:
    """libva, the iHD driver, and the oneVPL runtime from the primary host."""
    root = lib_dir.resolve()
    members = [(name, _local_library(lib_dir / name, root)) for name in RUNTIME_SONAMES]
    members.append(("dri/iHD_drv_video.so", _local_library(lib_dir / "dri" / "iHD_drv_video.so", root)))
    return members


def runtime_release(lib_dir: Path = RUNTIME_LIB_DIR) -> str:
    global _runtime_stamp_cache
    files = runtime_files(lib_dir)
    signature = _file_signature([path for _, path in files])
    with _release_lock:
        if _runtime_stamp_cache and _runtime_stamp_cache[0] == signature:
            return _runtime_stamp_cache[1]
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    stamp = digest.hexdigest()
    with _release_lock:
        _runtime_stamp_cache = (signature, stamp)
    return stamp


def write_runtime_archive(destination, lib_dir: Path = RUNTIME_LIB_DIR) -> None:
    with tarfile.open(fileobj=destination, mode="w|gz") as archive:
        for name, path in runtime_files(lib_dir):
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = int(path.stat().st_mtime)
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def runtime_archive(lib_dir: Path = RUNTIME_LIB_DIR) -> bytes:
    buffer = io.BytesIO()
    write_runtime_archive(buffer, lib_dir)
    return buffer.getvalue()


def runtime_archive_iter(lib_dir: Path = RUNTIME_LIB_DIR):
    runtime_files(lib_dir)
    read_fd, write_fd = os.pipe()

    def write() -> None:
        try:
            with os.fdopen(write_fd, "wb") as raw:
                write_runtime_archive(raw, lib_dir)
        except Exception:
            try:
                os.close(write_fd)
            except OSError:
                pass

    threading.Thread(target=write, name="homekit-runtime-archive", daemon=True).start()
    with os.fdopen(read_fd, "rb") as raw:
        while chunk := raw.read(1024 * 1024):
            yield chunk


def node_releases(ffmpeg_path: str, root: Path = WORKER_ROOT, *, lib_dir: Path | None = None) -> dict:
    releases: dict[str, str] = {}
    agent = available_agent_version(root)
    if agent is not None:
        releases["agent"] = agent
    try:
        releases["ffmpeg"] = ffmpeg_release(ffmpeg_path)
    except (OSError, ValueError):
        pass
    try:
        releases["runtime"] = runtime_release(lib_dir or RUNTIME_LIB_DIR)
    except (OSError, ValueError):
        pass
    return releases


def _tool_member(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if member.name not in {"ffmpeg", "ffprobe"} or not member.isfile() or member.issym() or member.islnk():
        return None
    member.uname = member.gname = ""
    member.mode = 0o755
    return member


def write_ffmpeg_archive(ffmpeg_path: str, destination) -> None:
    ffmpeg, ffprobe = ffmpeg_binaries(ffmpeg_path)
    with tarfile.open(fileobj=destination, mode="w|gz") as archive:
        archive.add(ffmpeg, arcname="ffmpeg", filter=_tool_member)
        archive.add(ffprobe, arcname="ffprobe", filter=_tool_member)


def ffmpeg_archive(ffmpeg_path: str) -> bytes:
    buffer = io.BytesIO()
    write_ffmpeg_archive(ffmpeg_path, buffer)
    return buffer.getvalue()


def ffmpeg_archive_iter(ffmpeg_path: str):
    ffmpeg_binaries(ffmpeg_path)
    read_fd, write_fd = os.pipe()

    def write() -> None:
        try:
            with os.fdopen(write_fd, "wb") as raw:
                write_ffmpeg_archive(ffmpeg_path, raw)
        except Exception:
            try:
                os.close(write_fd)
            except OSError:
                pass

    threading.Thread(target=write, name="homekit-ffmpeg-archive", daemon=True).start()
    with os.fdopen(read_fd, "rb") as raw:
        while chunk := raw.read(1024 * 1024):
            yield chunk
