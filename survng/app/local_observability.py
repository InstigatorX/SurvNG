"""Host-only, read-only access to a deliberately small runtime snapshot."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import socket
import stat
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .security import redact_secret_text

if TYPE_CHECKING:
    from .config import AppConfig


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_IO_TIMEOUT_SECONDS = 5.0
SOCKET_ENVIRONMENT_VARIABLE = "SURVNG_OBSERVABILITY_SOCKET"
MAX_RECENT_LOG_ROWS = 100
MAX_RECENT_LOG_BYTES = 32 * 1024
MAX_RECENT_LOG_MESSAGE_CHARACTERS = 500
_LOG_URL_RE = re.compile(
    r"\b(?:rtsp|rtsps|rtmp|http|https)://[^\s\"'<>]+",
    re.IGNORECASE,
)


def default_socket_path() -> Path:
    """Return the deterministic per-service local socket path.

    Native SurvNG runs as root by default and uses ``/run/survng``.  A
    non-root development process gets a private directory in its runtime
    directory, or a UID-specific private directory under ``/tmp``.
    """
    configured = os.environ.get(SOCKET_ENVIRONMENT_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.geteuid() == 0:
        return Path("/run/survng/observability.sock")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir:
        return Path(runtime_dir) / "survng" / "observability.sock"
    return Path("/tmp") / f"survng-{os.geteuid()}" / "observability.sock"


def _number(value: object, default: int | float = 0) -> int | float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return default


def _optional_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def _camera_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    tracking = raw.get("object_tracking")
    tracking = tracking if isinstance(tracking, dict) else {}
    lifecycle = raw.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    return {
        "id": str(raw.get("id") or ""),
        "name": str(raw.get("name") or ""),
        "enabled": bool(raw.get("expected_enabled", lifecycle.get("enabled", True))),
        "running": bool(raw.get("running")),
        "connected": bool(raw.get("connected")),
        "connectivity": str(raw.get("capture_connectivity") or "unknown"),
        "last_frame_age_seconds": _optional_number(raw.get("last_frame_age_seconds")),
        "recording": bool(raw.get("recording")),
        "recording_enabled": bool(raw.get("recording_enabled", True)),
        "detection_enabled": bool(raw.get("detection_enabled")),
        "onvif_connected": bool(raw.get("onvif_connected")),
        "tracking": {
            "active": bool(tracking.get("active")),
            "running": bool(tracking.get("running")),
            "capacity_requests": int(_number(tracking.get("capacity_requests"))),
            "capacity_waits": int(_number(tracking.get("capacity_waits"))),
            "capacity_timeouts": int(_number(tracking.get("capacity_timeouts"))),
            "capacity_wait_seconds_last": float(
                _number(tracking.get("capacity_wait_seconds_last"))
            ),
        },
    }


def _detector_snapshot(config: AppConfig, raw: dict[str, Any]) -> dict[str, Any]:
    runtime = raw.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    isolation = raw.get("isolation")
    isolation = isolation if isinstance(isolation, dict) else {}
    lifecycle = raw.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    ready_value = raw.get("ready")
    if ready_value is None:
        ready_value = bool(
            raw.get("loaded_backend")
            or raw.get("openvino_loaded")
            or raw.get("opencv_loaded")
            or raw.get("coreml_loaded")
        )
        if isolation.get("enabled"):
            ready_value = bool(ready_value and isolation.get("worker_alive"))
    return {
        "enabled": bool(config.detector.enabled),
        "backend": str(config.detector.backend),
        "device": str(raw.get("configured_device") or config.detector.device),
        "ready": bool(ready_value),
        "runtime": {
            "queue_depth": int(_number(runtime.get("queue_depth"))),
            "active_inferences": int(_number(runtime.get("active_inferences"))),
            "total_inferences": int(_number(runtime.get("total_inferences"))),
            "failed_inferences": int(_number(runtime.get("failed_inferences"))),
            "last_inference_ms": _number(runtime.get("last_inference_ms")),
            "average_inference_ms": _number(runtime.get("average_inference_ms")),
            "detection_fps": _number(runtime.get("detection_fps")),
        },
        "workers": {
            "configured": int(
                _number(
                    isolation.get("configured_workers"),
                    raw.get("object_worker_count") or 0,
                )
            ),
            "alive": int(
                _number(
                    isolation.get("alive_workers"),
                    int(bool(isolation.get("worker_alive"))),
                )
            ),
            "pending_requests": int(_number(isolation.get("pending_requests"))),
        },
        "lifecycle": {
            "core_started": bool(lifecycle.get("core_started")),
            "core_ready": bool(lifecycle.get("core_ready")),
            "auxiliary_started": bool(lifecycle.get("auxiliary_started")),
            "closed": bool(lifecycle.get("closed")),
        },
    }


def _storage_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    plan = raw.get("plan")
    plan = plan if isinstance(plan, dict) else {}
    storage = plan.get("storage")
    storage = storage if isinstance(storage, dict) else {}
    return {
        "state": str(raw.get("state") or "unknown"),
        "last_plan_at": raw.get("last_plan_at"),
        "last_run_at": raw.get("last_run_at"),
        "total_bytes": int(_number(storage.get("total_bytes"))),
        "used_bytes": int(_number(storage.get("used_bytes"))),
        "free_bytes": int(_number(storage.get("free_bytes"))),
        "free_percent": _number(storage.get("free_percent")),
        "emergency": bool(storage.get("emergency")),
    }


def _safe_log_text(value: object, *, limit: int) -> str:
    redacted = redact_secret_text(value)
    redacted = _LOG_URL_RE.sub("[redacted-url]", redacted)
    return redacted[:limit]


def _recent_log_snapshot(
    log_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return newest-first-priority log context with strict field and size bounds."""
    selected: list[dict[str, str]] = []
    # Account for the JSON array delimiters as part of the hard byte ceiling.
    serialized_bytes = 2
    source_count = len(log_rows)
    for raw in reversed(log_rows):
        if not isinstance(raw, Mapping):
            continue
        row = {
            "timestamp": _safe_log_text(raw.get("time", ""), limit=64),
            "level": _safe_log_text(raw.get("level", ""), limit=16),
            "logger": _safe_log_text(raw.get("logger", ""), limit=128),
            "message": _safe_log_text(
                raw.get("message", ""),
                limit=MAX_RECENT_LOG_MESSAGE_CHARACTERS,
            ),
        }
        row_bytes = len(
            json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        separator_bytes = 1 if selected else 0
        if (
            len(selected) >= MAX_RECENT_LOG_ROWS
            or serialized_bytes + separator_bytes + row_bytes > MAX_RECENT_LOG_BYTES
        ):
            break
        selected.append(row)
        serialized_bytes += separator_bytes + row_bytes
    selected.reverse()
    return {
        "entries": selected,
        "truncated": len(selected) < source_count,
        "serialized_bytes": serialized_bytes,
        "limits": {
            "max_entries": MAX_RECENT_LOG_ROWS,
            "max_bytes": MAX_RECENT_LOG_BYTES,
        },
    }


def build_runtime_status(
    config: AppConfig,
    manager: Any,
    *,
    instance_id: str,
    uptime_seconds: float,
    stopping: bool,
    log_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build an allowlisted snapshot; never serialize config/status objects whole."""
    try:
        raw_cameras = manager.statuses()
    except Exception as error:
        LOGGER.warning(
            "local observability could not read camera status (%s)",
            type(error).__name__,
        )
        raw_cameras = []
    cameras = [
        _camera_snapshot(item)
        for item in raw_cameras
        if isinstance(item, dict)
    ]

    try:
        raw_detector = manager.detector_status()
    except Exception as error:
        LOGGER.warning(
            "local observability could not read detector status (%s)",
            type(error).__name__,
        )
        raw_detector = {}

    try:
        raw_retention = manager.recorder.retention_status()
    except Exception as error:
        LOGGER.warning(
            "local observability could not read storage status (%s)",
            type(error).__name__,
        )
        raw_retention = {}

    try:
        limiter = manager.inference.tracking_limiter.status()
        limiter = limiter if isinstance(limiter, dict) else {}
    except Exception as error:
        LOGGER.warning(
            "local observability could not read tracking capacity (%s)",
            type(error).__name__,
        )
        limiter = {}

    tracking_config = config.detector.tracking
    active_camera_ids = [
        camera["id"] for camera in cameras if camera["tracking"]["active"]
    ]
    return {
        "schema_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "process": {
            "instance_id": str(instance_id),
            "pid": os.getpid(),
            "uptime_seconds": round(max(0.0, float(uptime_seconds)), 1),
            "stopping": bool(stopping),
        },
        "tracking": {
            "settings": {
                "enabled": bool(tracking_config.enabled),
                "max_active_cameras": int(tracking_config.max_active_cameras),
                "adaptive_burst_enabled": bool(tracking_config.adaptive_burst_enabled),
                "burst_max_active_cameras": int(
                    tracking_config.burst_max_active_cameras
                ),
                "capacity_wait_seconds": float(tracking_config.capacity_wait_seconds),
                "deferred_reid_enabled": bool(tracking_config.deferred_reid_enabled),
            },
            "capacity": {
                "active": int(_number(limiter.get("active"))),
                "baseline": int(
                    _number(limiter.get("baseline"), tracking_config.max_active_cameras)
                ),
                "burst_limit": int(
                    _number(
                        limiter.get("burst_limit"),
                        tracking_config.burst_max_active_cameras,
                    )
                ),
                "burst_enabled": bool(
                    limiter.get("burst_enabled", tracking_config.adaptive_burst_enabled)
                ),
                "burst_admissions": int(_number(limiter.get("burst_admissions"))),
                "burst_denials": int(_number(limiter.get("burst_denials"))),
                "active_camera_ids": active_camera_ids,
            },
            "activity_since_restart": {
                "requests": sum(
                    camera["tracking"]["capacity_requests"] for camera in cameras
                ),
                "waits": sum(camera["tracking"]["capacity_waits"] for camera in cameras),
                "timeouts": sum(
                    camera["tracking"]["capacity_timeouts"] for camera in cameras
                ),
            },
        },
        "detector": _detector_snapshot(config, raw_detector),
        "storage": _storage_snapshot(raw_retention),
        "cameras": cameras,
        "recent_logs": _recent_log_snapshot(log_rows),
    }


class LocalObservabilityServer:
    """One-command JSON protocol over an owner-only Unix-domain socket."""

    def __init__(
        self,
        status_provider: Callable[[], dict[str, Any]],
        socket_path: Path | str | None = None,
    ) -> None:
        self.status_provider = status_provider
        self.socket_path = Path(socket_path) if socket_path is not None else default_socket_path()
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.Task[None]] = set()
        self._socket_identity: tuple[int, int] | None = None

    @staticmethod
    def _prepare_parent(path: Path) -> None:
        parent = path.parent
        if parent.exists():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise RuntimeError(f"observability socket parent is not a directory: {parent}")
            if info.st_uid != os.geteuid():
                raise PermissionError(
                    "observability socket parent is not owned by this service: "
                    f"{parent}"
                )
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise PermissionError(f"observability socket parent must be mode 0700: {parent}")
            return
        parent.mkdir(mode=0o700, parents=True)
        parent.chmod(0o700)

    async def _remove_stale_socket(self) -> None:
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise RuntimeError(
                f"refusing to replace non-socket observability path: {self.socket_path}"
            )
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path)), timeout=0.25
            )
        except (ConnectionRefusedError, FileNotFoundError):
            self.socket_path.unlink(missing_ok=True)
            return
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                f"observability socket is already in use: {self.socket_path}"
            ) from error
        writer.close()
        await writer.wait_closed()
        raise RuntimeError(f"another SurvNG observability server is active: {self.socket_path}")

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._handle(reader, writer))
        self._clients.add(task)
        task.add_done_callback(self._clients.discard)

    @staticmethod
    def _peer_is_allowed(writer: asyncio.StreamWriter) -> bool:
        if not hasattr(socket, "SO_PEERCRED"):
            return True
        connected = writer.get_extra_info("socket")
        if connected is None:
            return False
        try:
            credentials = connected.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            _peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
        except (OSError, struct.error):
            return False
        return peer_uid in {0, os.geteuid()}

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        response: dict[str, Any]
        try:
            if not self._peer_is_allowed(writer):
                response = {"ok": False, "error": "local peer is not authorized"}
            else:
                raw = await asyncio.wait_for(
                    reader.readline(), timeout=DEFAULT_IO_TIMEOUT_SECONDS
                )
                if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                    raise ValueError("invalid request framing")
                request = json.loads(raw)
                if not isinstance(request, dict) or request.get("command") != "status":
                    raise ValueError("unsupported command")
                if int(request.get("version") or 0) != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol version")
                status_payload = await asyncio.to_thread(self.status_provider)
                response = {"ok": True, "status": status_payload}
        except (ValueError, json.JSONDecodeError, asyncio.TimeoutError) as error:
            response = {"ok": False, "error": str(error)}
        except asyncio.CancelledError:
            raise
        except Exception as error:
            LOGGER.warning(
                "local observability request failed (%s)", type(error).__name__
            )
            response = {"ok": False, "error": "runtime status is unavailable"}
        try:
            encoded = json.dumps(
                response, separators=(",", ":"), allow_nan=False
            ).encode("utf-8") + b"\n"
            writer.write(encoded)
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass

    async def start(self) -> None:
        if self._server is not None:
            return
        if not self.socket_path.is_absolute():
            raise ValueError("observability socket path must be absolute")
        self._prepare_parent(self.socket_path)
        await self._remove_stale_socket()
        try:
            self._server = await asyncio.start_unix_server(
                self._accept,
                path=str(self.socket_path),
                limit=MAX_REQUEST_BYTES + 1,
            )
            info = self.socket_path.lstat()
            self._socket_identity = (info.st_dev, info.st_ino)
            self.socket_path.chmod(0o600)
        except BaseException:
            await self.stop()
            raise
        LOGGER.info("local observability ready at %s", self.socket_path)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        tasks = tuple(self._clients)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clients.clear()
        try:
            info = self.socket_path.lstat()
        except FileNotFoundError:
            return
        identity = (info.st_dev, info.st_ino)
        if stat.S_ISSOCK(info.st_mode) and identity == self._socket_identity:
            self.socket_path.unlink(missing_ok=True)
        self._socket_identity = None


def request_runtime_status(
    socket_path: Path | str | None = None,
    *,
    timeout: float = DEFAULT_IO_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Synchronously request a status snapshot for the CLI."""
    path = Path(socket_path) if socket_path is not None else default_socket_path()
    request = json.dumps(
        {"version": PROTOCOL_VERSION, "command": "status"}, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(max(0.1, float(timeout)))
        client.connect(str(path))
        client.sendall(request)
        response_file = client.makefile("rb")
        raw = response_file.readline(MAX_RESPONSE_BYTES + 1)
    if not raw or len(raw) > MAX_RESPONSE_BYTES or not raw.endswith(b"\n"):
        raise RuntimeError("invalid response from SurvNG observability socket")
    response = json.loads(raw)
    if not isinstance(response, dict) or not response.get("ok"):
        detail = response.get("error") if isinstance(response, dict) else None
        raise RuntimeError(str(detail or "SurvNG runtime status is unavailable"))
    status_payload = response.get("status")
    if not isinstance(status_payload, dict):
        raise RuntimeError("SurvNG observability response did not contain a status object")
    return status_payload
