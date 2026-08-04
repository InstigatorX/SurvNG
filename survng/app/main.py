from __future__ import annotations

import json
import hashlib
import hmac
import logging
import math
import mmap
import asyncio
import functools
import queue
import os
import re
import signal
import secrets
import platform
import shutil
import time
import socket
import struct
import subprocess
import tempfile
import threading
import weakref
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import cv2
import numpy as np

from .config import AppConfig, CameraConfig, CameraMotionQualificationConfig, DetectionZone, camera_by_id, load_config, normalize_config, save_config, slugify_camera_id
from .audit_ai import (
    AuditAiAdvisor,
    AuditAiChange,
    AuditAiError,
    ai_provider_configured,
    motion_audit_interpretation,
    motion_paradigm_context,
    validate_tuning_value,
)
from .assistant import (
    AssistantAnswer,
    AssistantChatRequest,
    AssistantEvidence,
    AssistantProvider,
    AssistantToolCall,
    IncidentVisualReviewer,
)
from .assistant_investigation import correlate_incident_timeline
from .camera_intelligence import (
    aggregate_camera_intelligence,
    compare_camera_intelligence_results,
    select_balanced_samples,
)
from .detector import detection_failure, objects_to_json
from .manager import AppManager, validate_motion_pipeline_configuration
from .motion_pipeline import (
    analysis_preset_selections,
    guided_fusion_settings,
    identify_analysis_preset,
    motion_pipeline_catalog,
    resolve_motion_pipeline_graphs,
)
from .motion_ai_review import aggregate_motion_ai_review
from .media_exports import MediaExportManager
from .go2rtc import Go2RtcError
from .incident_utils import (
    DEFAULT_INCIDENT_GAP_SECONDS,
    event_epoch,
    event_snapshot_path,
    incident_event_groups,
    snapshot_media_type,
    stable_incident_id,
    stable_incident_key,
)
from .recording_media import (
    concatenated_clip_timing,
    event_clip_window,
    hls_map_transition,
    playback_segment_duration,
    resolve_stream_fingerprints,
)
from .object_tracking import ultralytics_botsort_dependency_status
from .tracking_comparison import TrackingComparisonRunner, sampled_video_frames
from .zones import apply_detection_zones, detection_threshold
from .security import redact_secret_text
from .storage_maintenance import StorageMaintenanceRunner, StorageReconciler

config = load_config()
manager = AppManager(config)
LOGGER = logging.getLogger(__name__)
LOG_LINES: deque[dict] = deque(maxlen=1000)
SECRET_PLACEHOLDER = "__SURVNG_SECRET_SET__"
RECORDING_LOOKUP_LIMIT = 20000
RECORDING_FMP4_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
RECORDING_FMP4_LOCKS_GUARD = threading.Lock()
EVENT_CLIP_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
EVENT_CLIP_LOCKS_GUARD = threading.Lock()
RECORDING_DAY_CACHE: dict[tuple[str, str, int, int], tuple[float, list[dict]]] = {}
RECORDING_DAY_CACHE_LOCK = threading.Lock()
RECORDING_DAY_CACHE_SECONDS = 30.0
RECORDING_PLAYBACK_WINDOW_SECONDS = 15 * 60
RECORDING_PREVIEW_INTERVAL_SECONDS = 5.0
RECORDING_PREVIEW_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
RECORDING_PREVIEW_MAX_BYTES = 256 * 1024 * 1024
RECORDING_PREVIEW_BUILD_LIMITER = threading.BoundedSemaphore(1)
RECORDING_PREVIEW_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
RECORDING_PREVIEW_LOCKS_GUARD = threading.Lock()
RECORDING_PREVIEW_MAINTENANCE_LOCK = threading.Lock()
RECORDING_PREVIEW_LAST_MAINTENANCE = 0.0
RECORDING_CACHE_MAINTENANCE_LOCK = threading.Lock()
RECORDING_CACHE_LAST_MAINTENANCE = 0.0
RECORDING_CACHE_METRICS_LOCK = threading.Lock()
RECORDING_CACHE_METRICS = {
    "playback_hits": 0,
    "playback_misses": 0,
    "playback_remuxes": 0,
    "playback_failures": 0,
    "playback_remux_ms": 0.0,
    "playback_last_remux_ms": 0.0,
    "prewarm_hits": 0,
    "prewarm_misses": 0,
    "prewarm_remuxes": 0,
    "prewarm_failures": 0,
    "prewarm_remux_ms": 0.0,
    "prewarm_last_remux_ms": 0.0,
}
RECORDING_PREWARM_STOP = threading.Event()
RECORDING_PREWARM_THREAD: threading.Thread | None = None
RECORDING_PREWARM_PROCESS_LOCK = threading.Lock()
RECORDING_PREWARM_PROCESS: subprocess.Popen | None = None
FACE_OBSERVATIONS_SYNCED = False
FACE_OBSERVATIONS_SYNC_LOCK = threading.Lock()
FACE_OBSERVATIONS_SYNC_THREAD_LOCK = threading.Lock()
FACE_OBSERVATIONS_SYNC_THREAD: threading.Thread | None = None
MANAGER_RELOAD_LOCK = threading.RLock()
APPLICATION_STOPPING = threading.Event()
CONFIG_PROBE_LIMITER = threading.BoundedSemaphore(2)
AUDIT_AI_LIMITER = threading.BoundedSemaphore(1)
ASSISTANT_LIMITER = threading.BoundedSemaphore(2)
AI_ACTIVITY_LOCK = threading.Lock()
AI_ACTIVE_OPERATIONS: dict[str, int] = {}
AI_RECOMMENDATION_SECRET = secrets.token_bytes(32)
AI_RECOMMENDATION_MAX_AGE_SECONDS = 60 * 60
EVENT_CLIP_BUILD_LIMITER = threading.BoundedSemaphore(2)
TRACKING_COMPARISON_LIMITER = threading.BoundedSemaphore(1)
PROCESS_STARTED_MONOTONIC = time.monotonic()
PROCESS_INSTANCE_ID = secrets.token_hex(12)
GPU_SAMPLE_LOCK = threading.Lock()
GPU_SAMPLE: dict[str, object] = {"at": 0.0, "pids": (), "engines": {}}
TELEMETRY_HISTORY_LOCK = threading.Lock()
TELEMETRY_HISTORY: deque[dict[str, object]] = deque(maxlen=360)
TELEMETRY_HISTORY_STATE: dict[str, float] = {"last_sample_at": 0.0}
TELEMETRY_PERSISTED_CACHE_LOCK = threading.Lock()
TELEMETRY_PERSISTED_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
STORAGE_MAINTENANCE = StorageMaintenanceRunner()
MEDIA_EXPORTS_LOCK = threading.Lock()
MEDIA_EXPORTS: MediaExportManager | None = None


class RecordingPrewarmCancelled(Exception):
    pass


def _begin_ai_operation(kind: str) -> None:
    with AI_ACTIVITY_LOCK:
        AI_ACTIVE_OPERATIONS[kind] = AI_ACTIVE_OPERATIONS.get(kind, 0) + 1


def _end_ai_operation(kind: str) -> None:
    with AI_ACTIVITY_LOCK:
        remaining = AI_ACTIVE_OPERATIONS.get(kind, 0) - 1
        if remaining > 0:
            AI_ACTIVE_OPERATIONS[kind] = remaining
        else:
            AI_ACTIVE_OPERATIONS.pop(kind, None)


def _active_ai_operations() -> dict[str, int]:
    with AI_ACTIVITY_LOCK:
        return dict(AI_ACTIVE_OPERATIONS)


class ConfiguredBasePathMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        base_path = config.base_path
        path = scope.get("path", "")
        if base_path and (path == base_path or path.startswith(f"{base_path}/")):
            scope = dict(scope)
            scope["path"] = path[len(base_path):] or "/"
            scope["raw_path"] = scope["path"].encode("utf-8")
        await self.app(scope, receive, send)


def _scope_header(scope: dict, name: bytes) -> str:
    for header_name, value in scope.get("headers", []):
        if header_name.lower() == name:
            return value.decode("latin-1").strip()
    return ""


def _same_origin_request(scope: dict) -> bool:
    origin = _scope_header(scope, b"origin")
    if not origin:
        return _scope_header(scope, b"sec-fetch-site").lower() != "cross-site"
    host = _scope_header(scope, b"host").lower()
    if not host:
        return False
    try:
        parsed = urlsplit(origin)
        parsed_host = urlsplit(f"//{host}")
        origin_port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        forwarded_scheme = _scope_header(scope, b"x-forwarded-proto").split(",", 1)[0].strip().lower()
        request_scheme = forwarded_scheme or str(scope.get("scheme") or "").lower()
        request_scheme = {"ws": "http", "wss": "https"}.get(request_scheme, request_scheme)
        request_port = parsed_host.port or (443 if request_scheme == "https" else 80)
    except ValueError:
        return False
    return (
        parsed.scheme.lower() in {"http", "https"}
        and (not request_scheme or parsed.scheme.lower() == request_scheme)
        and parsed.hostname is not None
        and parsed.hostname.lower() == str(parsed_host.hostname or "").lower()
        and origin_port == request_port
        and not parsed.username
        and not parsed.password
        and not parsed_host.username
        and not parsed_host.password
        and parsed.path == ""
        and not parsed.query
        and not parsed.fragment
    )


def _api_scope_path(scope: dict) -> str:
    path = str(scope.get("path") or "")
    base_path = config.base_path
    if base_path and (path == base_path or path.startswith(f"{base_path}/")):
        path = path[len(base_path):] or "/"
    return path


class SecurityBoundaryMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        scope_type = scope.get("type")
        if (
            scope_type == "http"
            and APPLICATION_STOPPING.is_set()
            and str(scope.get("method") or "GET").upper() not in {"GET", "HEAD", "OPTIONS"}
            and _api_scope_path(scope).startswith("/api/")
        ):
            response = JSONResponse(
                {"detail": "SurvNG is shutting down; configuration changes are temporarily unavailable"},
                status_code=503,
                headers={"Cache-Control": "no-store", "Retry-After": "10"},
            )
            await response(scope, receive, send)
            return
        if scope_type == "websocket" and not _same_origin_request(scope):
            await send({"type": "websocket.close", "code": 1008})
            return
        if (
            scope_type == "http"
            and _api_scope_path(scope).startswith("/api/")
            and not _same_origin_request(scope)
        ):
            response = JSONResponse(
                {"detail": "cross-origin API requests are not allowed"},
                status_code=403,
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "Referrer-Policy": "same-origin",
                },
            )
            await response(scope, receive, send)
            return

        async def send_with_security_headers(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _value in headers}
                additions = {
                    b"x-content-type-options": b"nosniff",
                    b"referrer-policy": b"same-origin",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                    b"x-frame-options": b"SAMEORIGIN",
                }
                if _api_scope_path(scope).startswith("/api/"):
                    additions[b"cache-control"] = b"no-store"
                headers.extend(
                    (name, value)
                    for name, value in additions.items()
                    if name not in existing
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


def public_url(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{config.base_path}{normalized}"


def frontend_response(filename: str) -> HTMLResponse:
    html = Path("survng/static", filename).read_text(encoding="utf-8")
    html = html.replace('="/static/', f'="{config.base_path}/static/')
    runtime_config = f"<script>window.__SURVNG_BASE_PATH__={json.dumps(config.base_path)};</script>"
    html = html.replace("</head>", f"  {runtime_config}\n  </head>")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


class MemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        LOG_LINES.append({
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_log_message(message),
        })


def redact_log_message(message: str) -> str:
    return redact_secret_text(message)


def _mask_url_password(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    userinfo, separator, host = parsed.netloc.rpartition("@")
    if not separator or ":" not in userinfo:
        return value
    username, _password = userinfo.split(":", 1)
    return urlunsplit(parsed._replace(netloc=f"{username}:{SECRET_PLACEHOLDER}@{host}"))


def _encoded_url_password(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    userinfo, separator, _host = parsed.netloc.rpartition("@")
    if not separator or ":" not in userinfo:
        return None
    return userinfo.split(":", 1)[1]


def _restore_url_password(masked: str | None, current: str | None, field: str) -> str | None:
    if not masked or _encoded_url_password(masked) != SECRET_PLACEHOLDER:
        return masked
    current_password = _encoded_url_password(current)
    if current_password is None:
        raise ValueError(f"{field} contains a masked secret without an existing value")
    parsed = urlsplit(masked)
    userinfo, _separator, host = parsed.netloc.rpartition("@")
    username, _masked_password = userinfo.split(":", 1)
    return urlunsplit(parsed._replace(netloc=f"{username}:{current_password}@{host}"))


def _restore_secret(masked: str, current: str, field: str) -> str:
    if masked != SECRET_PLACEHOLDER:
        return masked
    if not current:
        raise ValueError(f"{field} contains a masked secret without an existing value")
    return current


def _camera_uses_masked_secret(camera: CameraConfig) -> bool:
    return (
        _encoded_url_password(camera.stream_url) == SECRET_PLACEHOLDER
        or _encoded_url_password(camera.live_stream_url) == SECRET_PLACEHOLDER
        or camera.onvif.password == SECRET_PLACEHOLDER
        or camera.baichuan.password == SECRET_PLACEHOLDER
    )


def _restore_camera_secrets(
    incoming: CameraConfig,
    current: CameraConfig | None,
) -> CameraConfig:
    restored = incoming.model_copy(deep=True)
    if not _camera_uses_masked_secret(restored):
        return restored
    if current is None:
        raise ValueError("new cameras must provide their own credentials")
    restored.stream_url = str(
        _restore_url_password(restored.stream_url, current.stream_url, "stream_url")
        or ""
    )
    restored.live_stream_url = _restore_url_password(
        restored.live_stream_url,
        current.live_stream_url,
        "live_stream_url",
    )
    restored.onvif.password = _restore_secret(
        restored.onvif.password,
        current.onvif.password,
        "onvif.password",
    )
    restored.baichuan.password = _restore_secret(
        restored.baichuan.password,
        current.baichuan.password,
        "baichuan.password",
    )
    return restored


def _camera_secret_identity_matches(incoming: CameraConfig, current: CameraConfig) -> bool:
    if _mask_url_password(incoming.stream_url) == _mask_url_password(current.stream_url):
        return True
    if (
        incoming.live_stream_url
        and _mask_url_password(incoming.live_stream_url)
        == _mask_url_password(current.live_stream_url)
    ):
        return True
    return any(
        incoming_host
        and incoming_host == current_host
        and incoming_user == current_user
        for incoming_host, current_host, incoming_user, current_user in (
            (
                incoming.onvif.host,
                current.onvif.host,
                incoming.onvif.username,
                current.onvif.username,
            ),
            (
                incoming.baichuan.host,
                current.baichuan.host,
                incoming.baichuan.username,
                current.baichuan.username,
            ),
        )
    )


def _restore_config_secrets(incoming: AppConfig, current: AppConfig) -> AppConfig:
    restored = incoming.model_copy(deep=True)
    restored.mqtt.password = _restore_secret(
        restored.mqtt.password,
        current.mqtt.password,
        "mqtt.password",
    )
    restored.audit_ai.api_key = _restore_secret(
        restored.audit_ai.api_key,
        current.audit_ai.api_key,
        "audit_ai.api_key",
    )
    current_by_id = {camera.id: camera for camera in current.cameras}
    same_shape = len(restored.cameras) == len(current.cameras)
    restored.cameras = [
        _restore_camera_secrets(
            camera,
            current_by_id.get(camera.id)
            or (
                current.cameras[index]
                if same_shape
                and _camera_secret_identity_matches(camera, current.cameras[index])
                else None
            ),
        )
        for index, camera in enumerate(restored.cameras)
    ]
    return AppConfig.model_validate(restored.model_dump(mode="json"))


def _redacted_camera_payload(camera: CameraConfig) -> dict:
    payload = camera.model_dump(mode="json")
    payload["stream_url"] = _mask_url_password(camera.stream_url)
    payload["live_stream_url"] = _mask_url_password(camera.live_stream_url)
    payload["onvif"]["password"] = SECRET_PLACEHOLDER if camera.onvif.password else ""
    payload["baichuan"]["password"] = SECRET_PLACEHOLDER if camera.baichuan.password else ""
    return payload


def _redacted_config_payload(active_config: AppConfig) -> dict:
    payload = active_config.model_dump(mode="json")
    payload["mqtt"]["password"] = SECRET_PLACEHOLDER if active_config.mqtt.password else ""
    payload["audit_ai"]["api_key"] = SECRET_PLACEHOLDER if active_config.audit_ai.api_key else ""
    payload["cameras"] = [
        _redacted_camera_payload(camera)
        for camera in active_config.cameras
    ]
    return payload


def install_memory_log_handler() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "name", "") == "survng-memory-log" for handler in root.handlers):
        return
    handler = MemoryLogHandler()
    handler.name = "survng-memory-log"
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(min(root.level or logging.INFO, logging.INFO))


install_memory_log_handler()


class CameraFeatureState(BaseModel):
    enabled: bool


class ConfigProbeRequest(BaseModel):
    camera_id: str = Field(default="", max_length=128)
    host: str = Field(min_length=1, max_length=255, pattern=r"^[^\s/@?#]+$")
    username: str = Field(default="", max_length=256)
    password: str = Field(default="", max_length=1024)
    onvif_port: int = Field(default=8000, ge=1, le=65535)
    baichuan_port: int = Field(default=9000, ge=1, le=65535)



def _ffmpeg_sibling_tool(name: str) -> str:
    ffmpeg = Path(config.ffmpeg_path)
    if ffmpeg.name == "ffmpeg":
        sibling = ffmpeg.with_name(name)
        if sibling.exists():
            return str(sibling)
    return name


def _ffprobe_path() -> str:
    return _ffmpeg_sibling_tool("ffprobe")


def _ffplay_path() -> str:
    return _ffmpeg_sibling_tool("ffplay")

def normalize_source(source: str) -> str:
    return "main" if source == "main" else "live"


def recording_source(source: str = "main") -> str:
    return "live" if source == "live" else "main"


def _require_recording_camera(camera_id: str) -> None:
    if manager.camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="camera not found")


def _validate_recording_range(
    start_epoch: float,
    end_epoch: float,
    maximum_seconds: float,
    detail: str,
) -> None:
    if not math.isfinite(start_epoch) or not math.isfinite(end_epoch):
        raise HTTPException(status_code=400, detail=detail)
    try:
        datetime.fromtimestamp(start_epoch, timezone.utc)
        datetime.fromtimestamp(end_epoch, timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise HTTPException(status_code=400, detail=detail) from None
    if end_epoch <= start_epoch or end_epoch - start_epoch > maximum_seconds:
        raise HTTPException(status_code=400, detail=detail)


class StorageTasksActiveError(RuntimeError):
    def __init__(self, tasks: list[str]) -> None:
        self.tasks = tasks
        super().__init__(
            "configuration change was not applied because storage work is active: "
            f"{', '.join(tasks)}. Wait for it to finish or cancel it from Maintenance."
        )


class AiOperationsActiveError(RuntimeError):
    def __init__(self, operations: dict[str, int]) -> None:
        self.operations = operations
        labels = ", ".join(
            f"{name.replace('_', ' ')} ({count})"
            for name, count in sorted(operations.items())
        )
        super().__init__(
            "configuration change was not applied because AI analysis is active: "
            f"{labels}. Wait for it to finish and try again."
        )


def _active_storage_tasks(active_manager: AppManager) -> list[str]:
    tasks: list[str] = []
    maintenance = STORAGE_MAINTENANCE.status()
    if maintenance.get("status") in {"running", "cancelling"}:
        mode = str(maintenance.get("mode") or "maintenance").replace("_", " ")
        tasks.append(f"storage {mode}")
    if MEDIA_EXPORTS is not None:
        exports = MEDIA_EXPORTS.active_jobs()
        if exports:
            kinds = sorted({str(job.get("kind") or "media") for job in exports})
            tasks.append(f"media {'/'.join(kinds)} export")
    return tasks


def reload_manager(
    next_config: AppConfig,
    *,
    assign_ids: bool = False,
    persist: bool = True,
) -> AppConfig:
    global config, manager, FACE_OBSERVATIONS_SYNCED
    effective_config = normalize_config(
        next_config.model_copy(deep=True),
        assign_ids=assign_ids,
    )
    with MANAGER_RELOAD_LOCK:
        if APPLICATION_STOPPING.is_set():
            raise RuntimeError("configuration reload refused while SurvNG is shutting down")
        previous_config = config
        previous_manager = manager
        active_storage_tasks = _active_storage_tasks(previous_manager)
        if active_storage_tasks:
            raise StorageTasksActiveError(active_storage_tasks)
        active_ai_operations = _active_ai_operations()
        if active_ai_operations:
            raise AiOperationsActiveError(active_ai_operations)
        prewarmer_was_running = bool(
            RECORDING_PREWARM_THREAD is not None
            and RECORDING_PREWARM_THREAD.is_alive()
        )
        candidate = AppManager(effective_config)
        previous_stop_attempted = False
        runtime_preferences = previous_manager.runtime_preferences()
        try:
            _stop_recording_prewarmer()
            previous_stop_attempted = True
            previous_manager.stop_all_with_runtime_preferences()
            candidate.apply_runtime_preferences(runtime_preferences)
            candidate.start_all()
            candidate.apply_runtime_preferences(runtime_preferences, persist=True)
            if persist:
                save_config(effective_config, assign_ids=False)
        except BaseException as reload_error:
            try:
                candidate.stop_all()
            except Exception:
                logging.getLogger(__name__).exception(
                    "failed to clean up replacement manager after reload failure"
                )
            if not previous_stop_attempted:
                if prewarmer_was_running:
                    _start_recording_prewarmer()
                raise RuntimeError(
                    "configuration reload failed before the active manager was stopped"
                ) from reload_error
            try:
                recovery = AppManager(previous_config)
                recovery.apply_runtime_preferences(runtime_preferences, persist=True)
                recovery.start_all()
            except BaseException as recovery_error:
                raise RuntimeError(
                    "configuration reload failed and the previous manager could not be restored"
                ) from recovery_error
            config = previous_config
            manager = recovery
            FACE_OBSERVATIONS_SYNCED = False
            _start_face_observation_sync()
            if prewarmer_was_running:
                _start_recording_prewarmer()
            if not isinstance(reload_error, Exception):
                raise
            raise RuntimeError(
                "configuration reload failed; the previous configuration was restored"
            ) from reload_error

        config = effective_config
        manager = candidate
        FACE_OBSERVATIONS_SYNCED = False
        _start_face_observation_sync()
        with RECORDING_DAY_CACHE_LOCK:
            RECORDING_DAY_CACHE.clear()
        _ffmpeg_qsv_info.cache_clear()
        _ffmpeg_vaapi_info.cache_clear()
        if prewarmer_was_running:
            _start_recording_prewarmer()
        return effective_config


HOT_CONFIG_FIELDS = frozenset({
    "base_path",
    "event_clip_before_seconds",
    "event_clip_after_seconds",
    "incident_thumbnail_annotations",
    "image_storage",
    "recording_cache_max_gb",
    "recording_cache_max_days",
    "recording_cache_prewarm",
    "audit_ai",
    "mqtt",
    "retention",
})
RECORDER_CONFIG_FIELDS = frozenset({
    "ffmpeg_path",
    "hardware_acceleration",
    "recording_segment_seconds",
})
DETECTOR_HOT_POLICY_FIELDS = frozenset({
    "confidence_threshold",
    "event_confirmation_frames",
    "event_class_confirmation_frames",
    "event_class_confidence_thresholds",
    "require_incident_zone",
    "face_max_observations",
    "face_match_threshold",
    "face_min_size",
    "face_max_references",
})
TRACKING_SESSION_FIELDS = frozenset({
    "enabled",
    "implementation",
    "excluded_labels",
    "sample_fps",
    "max_session_seconds",
    "lost_timeout_seconds",
    "min_confirmations",
    "low_confidence_threshold",
    "match_iou_threshold",
    "match_center_distance_ratio",
    "max_active_cameras",
    "capacity_wait_seconds",
    "max_tracks_per_session",
    "reid_max_age_seconds",
    "reid_max_embeddings_per_frame",
    "reid_refresh_interval_frames",
    "reid_match_threshold",
    "vehicle_reid_match_threshold",
    "vehicle_reid_labels",
    "botsort_match_threshold",
    "botsort_proximity_threshold",
    "botsort_fuse_score",
})
DETECTOR_OBJECT_ENGINE_FIELDS = frozenset({
    "enabled",
    "backend",
    "model_path",
    "model_xml",
    "coreml_model_path",
    "labels_path",
    "device",
    "nms_threshold",
    "warmup_enabled",
    "labels",
})
DETECTOR_OBJECT_TRACKING_RESET_FIELDS = frozenset({
    "enabled",
    "backend",
    "model_path",
    "model_xml",
    "coreml_model_path",
    "labels_path",
    "nms_threshold",
    "labels",
})
DETECTOR_FACE_ENGINE_FIELDS = frozenset({
    "face_recognition_enabled",
    "face_embedding_model_path",
    "face_landmark_model_path",
    "face_recognition_device",
})
DETECTOR_SHARED_ENGINE_FIELDS = frozenset({
    "cache_enabled",
    "cache_dir",
})
TRACKING_REID_ENGINE_FIELDS = frozenset({
    "reid_enabled",
    "reid_model_path",
    "reid_device",
    "vehicle_reid_enabled",
    "vehicle_reid_model_path",
    "vehicle_reid_device",
})


def _detector_without_fields(detector: dict, fields: frozenset[str]) -> dict:
    return {
        key: value
        for key, value in detector.items()
        if key not in fields
    }


def _manager_owned_config(config_value: AppConfig) -> dict:
    """Return only settings that require rebuilding camera-owned services."""
    payload = config_value.model_dump(mode="json")
    for field_name in HOT_CONFIG_FIELDS | RECORDER_CONFIG_FIELDS:
        payload.pop(field_name, None)
    for camera in payload.get("cameras", []):
        camera.pop("retention", None)
    payload["detector"] = _detector_without_fields(
        payload.get("detector", {}),
        DETECTOR_HOT_POLICY_FIELDS
        | DETECTOR_OBJECT_ENGINE_FIELDS
        | DETECTOR_FACE_ENGINE_FIELDS
        | DETECTOR_SHARED_ENGINE_FIELDS,
    )
    tracking = payload["detector"].get("tracking")
    if isinstance(tracking, dict):
        payload["detector"]["tracking"] = _detector_without_fields(
            tracking,
            TRACKING_SESSION_FIELDS | TRACKING_REID_ENGINE_FIELDS,
        )
    return payload


def _hot_config_changes(current: AppConfig, incoming: AppConfig) -> list[str]:
    changed = [
        field_name
        for field_name in sorted(HOT_CONFIG_FIELDS)
        if getattr(current, field_name) != getattr(incoming, field_name)
    ]
    current_retention = {camera.id: camera.retention for camera in current.cameras}
    incoming_retention = {camera.id: camera.retention for camera in incoming.cameras}
    if current_retention != incoming_retention and "retention" not in changed:
        changed.append("retention")
    return changed


def apply_config_update(
    next_config: AppConfig,
    *,
    assign_ids: bool = False,
    persist: bool = True,
) -> tuple[AppConfig, dict[str, object]]:
    """Apply configuration at the narrowest safe runtime boundary."""
    global config
    effective_config = normalize_config(
        next_config.model_copy(deep=True),
        assign_ids=assign_ids,
    )
    if _manager_owned_config(config) != _manager_owned_config(effective_config):
        effective = reload_manager(
            effective_config,
            assign_ids=False,
            persist=persist,
        )
        return effective, {
            "apply_mode": "manager_reload",
            "camera_workers_restarted": True,
            "subsystems_restarted": ["manager"],
        }

    with MANAGER_RELOAD_LOCK:
        previous_config = config
        changes = _hot_config_changes(previous_config, effective_config)
        mqtt_changed = previous_config.mqtt != effective_config.mqtt
        retention_changed = "retention" in changes
        image_storage_changed = "image_storage" in changes
        detector_policy_changed = any(
            getattr(previous_config.detector, field_name)
            != getattr(effective_config.detector, field_name)
            for field_name in DETECTOR_HOT_POLICY_FIELDS
        )
        tracking_session_changed = any(
            getattr(previous_config.detector.tracking, field_name)
            != getattr(effective_config.detector.tracking, field_name)
            for field_name in TRACKING_SESSION_FIELDS
        )
        inference_roles: set[str] = set()
        object_engine_changed = any(
            getattr(previous_config.detector, field_name)
            != getattr(effective_config.detector, field_name)
            for field_name in DETECTOR_OBJECT_ENGINE_FIELDS
        )
        if object_engine_changed:
            inference_roles.add("object")
        if any(
            getattr(previous_config.detector, field_name)
            != getattr(effective_config.detector, field_name)
            for field_name in DETECTOR_FACE_ENGINE_FIELDS
        ):
            inference_roles.add("face")
        if any(
            getattr(previous_config.detector, field_name)
            != getattr(effective_config.detector, field_name)
            for field_name in DETECTOR_SHARED_ENGINE_FIELDS
        ):
            inference_roles.update({"object", "face", "reid"})
        if any(
            getattr(previous_config.detector.tracking, field_name)
            != getattr(effective_config.detector.tracking, field_name)
            for field_name in TRACKING_REID_ENGINE_FIELDS
        ):
            inference_roles.add("reid")
        reid_tracking_refresh = (
            "reid" in inference_roles
            and previous_config.detector.tracking
            != effective_config.detector.tracking
        )
        object_tracking_refresh = any(
            getattr(previous_config.detector, field_name)
            != getattr(effective_config.detector, field_name)
            for field_name in DETECTOR_OBJECT_TRACKING_RESET_FIELDS
        )
        inference_tracking_refresh = (
            reid_tracking_refresh
            or object_tracking_refresh
            or (tracking_session_changed and bool(inference_roles))
        )
        recorder_changes = [
            field_name
            for field_name in sorted(RECORDER_CONFIG_FIELDS)
            if getattr(previous_config, field_name) != getattr(effective_config, field_name)
        ]
        if recorder_changes and MEDIA_EXPORTS is not None:
            active_exports = MEDIA_EXPORTS.active_jobs()
            if active_exports:
                kinds = sorted({str(job.get("kind") or "media") for job in active_exports})
                raise StorageTasksActiveError([f"media {'/'.join(kinds)} export"])
        if persist:
            save_config(effective_config, assign_ids=False)
        manager.config = effective_config
        recorder_attempted = False
        mqtt_attempted = False
        retention_attempted = False
        image_storage_attempted = False
        detector_policy_attempted = False
        tracking_session_applied = False
        inference_applied = False
        try:
            if recorder_changes:
                recorder_attempted = True
                manager.reconfigure_recorders(effective_config)
            if mqtt_changed:
                mqtt_attempted = True
                manager.reconfigure_mqtt(effective_config.mqtt)
            if retention_changed:
                retention_attempted = True
                manager.recorder.reconfigure_retention(
                    effective_config.retention,
                    effective_config.cameras,
                )
            if image_storage_changed:
                image_storage_attempted = True
                manager.reconfigure_image_storage(effective_config.image_storage)
            if inference_roles:
                manager.reconfigure_inference(
                    effective_config.detector,
                    inference_roles,
                    refresh_tracking=inference_tracking_refresh,
                )
                inference_applied = True
            if tracking_session_changed and not inference_tracking_refresh:
                manager.reconfigure_object_tracking(effective_config.detector)
                tracking_session_applied = True
            if detector_policy_changed:
                detector_policy_attempted = True
                manager.reconfigure_detector_policy(effective_config.detector)
        except BaseException:
            manager.config = previous_config
            if detector_policy_attempted:
                try:
                    manager.reconfigure_detector_policy(previous_config.detector)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back detector policy configuration"
                    )
            if tracking_session_applied:
                try:
                    manager.reconfigure_object_tracking(previous_config.detector)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back object tracking configuration"
                    )
            if inference_applied:
                try:
                    manager.reconfigure_inference(
                        previous_config.detector,
                        inference_roles,
                        refresh_tracking=inference_tracking_refresh,
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back inference configuration"
                    )
            if image_storage_attempted:
                try:
                    manager.reconfigure_image_storage(previous_config.image_storage)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back image storage configuration"
                    )
            if retention_attempted:
                try:
                    manager.recorder.reconfigure_retention(
                        previous_config.retention,
                        previous_config.cameras,
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back recording retention configuration"
                    )
            if mqtt_attempted:
                try:
                    manager.reconfigure_mqtt(previous_config.mqtt)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back MQTT configuration"
                    )
            if recorder_attempted:
                try:
                    manager.reconfigure_recorders(previous_config)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to roll back recorder configuration"
                    )
            if persist:
                try:
                    save_config(previous_config, assign_ids=False)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "failed to restore persisted configuration after hot-apply failure"
                    )
            raise
        config = effective_config
        restarted = [
            name
            for name, changed
            in (("recorders", bool(recorder_changes)), ("mqtt", mqtt_changed))
            if changed
        ]
        if tracking_session_changed or inference_tracking_refresh:
            restarted.append("tracking_sessions")
        restarted.extend(
            f"{role}_inference"
            for role in ("object", "face", "reid")
            if role in inference_roles
        )
        hot_updated = changes + recorder_changes
        if detector_policy_changed:
            hot_updated.append("detector_policy")
        return effective_config, {
            "apply_mode": "targeted" if restarted else "hot" if hot_updated else "unchanged",
            "camera_workers_restarted": False,
            "subsystems_restarted": restarted,
            "hot_updated": hot_updated,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    APPLICATION_STOPPING.clear()
    loop = asyncio.get_running_loop()
    early_onvif_thread: threading.Thread | None = None
    early_onvif_lock = threading.Lock()
    media_exports: MediaExportManager | None = None

    def release_onvif_before_server_drain() -> None:
        nonlocal early_onvif_thread
        APPLICATION_STOPPING.set()
        with early_onvif_lock:
            if early_onvif_thread is not None and early_onvif_thread.is_alive():
                return
            active_manager = manager

            def release() -> None:
                logging.getLogger("uvicorn.error").info(
                    "SurvNG early shutdown: releasing ONVIF subscriptions before connection drain"
                )
                try:
                    active_manager.release_onvif_subscriptions()
                except Exception:
                    logging.getLogger("uvicorn.error").exception(
                        "early ONVIF subscription release was incomplete"
                    )

            early_onvif_thread = threading.Thread(
                target=release,
                name="early-onvif-shutdown",
                daemon=False,
            )
            early_onvif_thread.start()

    early_signal_installed = False
    try:
        loop.add_signal_handler(signal.SIGUSR1, release_onvif_before_server_drain)
        early_signal_installed = True
    except (NotImplementedError, RuntimeError):
        logging.getLogger(__name__).warning(
            "early ONVIF shutdown signal is unavailable on this platform"
        )
    manager.start_all()
    try:
        media_exports = _media_export_manager()
        media_exports.start()
        _start_face_observation_sync()
        _start_recording_prewarmer()
        yield
    finally:
        APPLICATION_STOPPING.set()
        if early_signal_installed:
            loop.remove_signal_handler(signal.SIGUSR1)
        try:
            if not STORAGE_MAINTENANCE.stop(timeout=5.0):
                logging.getLogger("uvicorn.error").warning(
                    "storage maintenance did not stop before application shutdown"
                )
        finally:
            try:
                if media_exports is not None and not media_exports.stop(timeout=10.0):
                    logging.getLogger("uvicorn.error").warning(
                        "media export worker did not stop before application shutdown"
                    )
            finally:
                try:
                    _stop_recording_prewarmer()
                finally:
                    try:
                        with MANAGER_RELOAD_LOCK:
                            manager.stop_all()
                    finally:
                        if early_onvif_thread is not None and early_onvif_thread.is_alive():
                            early_onvif_thread.join()
                        try:
                            manager.detector.stop_resource_tracker()
                        except Exception:
                            logging.getLogger("uvicorn.error").exception(
                                "final multiprocessing resource tracker cleanup failed"
                            )


app = FastAPI(title="SurvNG", lifespan=lifespan)


@app.exception_handler(StorageTasksActiveError)
async def storage_tasks_active_handler(
    _request: Request,
    error: StorageTasksActiveError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(error), "active_storage_tasks": error.tasks},
    )


@app.exception_handler(AiOperationsActiveError)
async def ai_operations_active_handler(
    _request: Request,
    error: AiOperationsActiveError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": str(error), "active_ai_operations": error.operations},
    )


app.add_middleware(ConfiguredBasePathMiddleware)
app.add_middleware(SecurityBoundaryMiddleware)
app.mount("/static", StaticFiles(directory="survng/static"), name="static")


@app.get("/api/health", include_in_schema=False)
def health() -> dict[str, str]:
    """Cheap liveness check that never probes cameras, media, or network storage."""
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse("survng/static/favicon.ico", media_type="image/x-icon")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    return FileResponse("survng/static/apple-touch-icon.png", media_type="image/png")


@app.get("/")
def index() -> HTMLResponse:
    return frontend_response("index.html")


@app.get("/recordings")
def recordings_page() -> HTMLResponse:
    return frontend_response("recordings.html")


@app.get("/recordings/exports")
def recording_exports_page() -> HTMLResponse:
    return frontend_response("recordings.html")


@app.get("/config")
def config_page() -> HTMLResponse:
    return frontend_response("config.html")


@app.get("/incidents")
def incidents_page() -> HTMLResponse:
    return frontend_response("index.html")


@app.get("/faces")
def faces_page() -> HTMLResponse:
    return frontend_response("index.html")


@app.get("/api/cameras")
def cameras() -> list[dict]:
    return manager.statuses()


def _sse_message(event_type: str, payload: object, event_id: str | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(payload, separators=(',', ':'), default=str)}")
    return "\n".join(lines) + "\n\n"


@app.get("/api/events/stream")
async def application_event_stream(request: Request) -> StreamingResponse:
    active_manager = manager
    subscriber = active_manager.state_events.subscribe()

    async def generate():
        try:
            yield "retry: 3000\n\n"
            last_event_id = request.headers.get("last-event-id", "")
            replay = active_manager.state_events.events_after(last_event_id)
            replayed_ids: set[str] = set()
            snapshot_sequence: int | None = None
            if replay is None:
                # Establish the snapshot boundary first. Events accepted after
                # this cursor remain queued and are delivered after the
                # snapshots, so a change racing snapshot generation cannot be
                # lost. Events at or before it are represented by snapshots.
                connection_cursor = active_manager.state_events.cursor
                snapshot_sequence = active_manager.state_events.sequence(connection_cursor)
                yield _sse_message("cameras_state", await asyncio.to_thread(active_manager.statuses))
                yield _sse_message("system_state", await asyncio.to_thread(system_status))
            else:
                for event in replay:
                    replayed_ids.add(event.id)
                    yield _sse_message(event.type, event.data, event.id)
                connection_cursor = replay[-1].id if replay else last_event_id
            yield _sse_message("connected", {"instance": active_manager.state_events.instance_id}, connection_cursor)
            next_system_update = time.monotonic() + 5.0
            stream_deadline = time.monotonic() + 6.0
            while True:
                if await request.is_disconnected():
                    return
                if time.monotonic() >= stream_deadline:
                    return
                try:
                    event = subscriber.get_nowait()
                except queue.Empty:
                    if time.monotonic() >= next_system_update:
                        yield _sse_message("system_state", await asyncio.to_thread(system_status))
                        next_system_update = time.monotonic() + 5.0
                    await asyncio.sleep(0.2)
                    continue
                if event is None:
                    return
                if event.id in replayed_ids:
                    continue
                event_sequence = active_manager.state_events.sequence(event.id)
                if (
                    snapshot_sequence is not None
                    and event_sequence is not None
                    and event_sequence <= snapshot_sequence
                ):
                    continue
                yield _sse_message(event.type, event.data, event.id)
        finally:
            active_manager.state_events.unsubscribe(subscriber)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/config")
def get_config() -> dict:
    return _redacted_config_payload(config)


@app.get("/api/motion/pipeline/catalog")
def get_motion_pipeline_catalog() -> dict:
    return motion_pipeline_catalog(manager.motion_pipeline_registry)


@app.get("/api/logs")
def logs(limit: int = 300, level: str = "", q: str = "") -> dict:
    safe_limit = max(1, min(limit, 1000))
    wanted_level = level.strip().upper()
    query = q.strip().lower()
    rows = list(LOG_LINES)[-safe_limit:]
    if wanted_level:
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            min_index = levels.index(wanted_level)
            allowed = set(levels[min_index:])
            rows = [row for row in rows if row.get("level") in allowed]
        except ValueError:
            rows = [row for row in rows if row.get("level") == wanted_level]
    if query:
        rows = [row for row in rows if query in f"{row.get('level', '')} {row.get('logger', '')} {row.get('message', '')}".lower()]
    return {"lines": rows[-safe_limit:], "total": len(LOG_LINES)}


class StorageMaintenanceRequest(BaseModel):
    apply: bool = False
    full: bool = False


class RecordingRetentionRequest(BaseModel):
    apply: bool = False


class MediaExportRequest(BaseModel):
    kind: str = Field(default="recording", pattern=r"^(recording|timelapse)$")
    camera_id: str = Field(min_length=1, max_length=128)
    source: str = Field(default="main", pattern=r"^(main|live)$")
    start_epoch: float
    end_epoch: float
    sample_interval_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    output_fps: int = Field(default=30, ge=1, le=60)
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=0, ge=0, le=2160)
    label: str = Field(default="", max_length=120)


class MediaExportProtectionRequest(BaseModel):
    protected: bool


class MediaExportMetadataRequest(BaseModel):
    label: str = Field(default="", max_length=120)


class MediaExportBatchRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=100)
    action: str = Field(pattern=r"^(protect|unprotect|delete)$")


@app.get("/api/retention/status")
def recording_retention_status() -> dict:
    return manager.recorder.retention_status()


@app.post("/api/retention/run", status_code=202)
def run_recording_retention(request: RecordingRetentionRequest) -> dict:
    return manager.recorder.request_retention_run(apply=request.apply)


@app.get("/api/maintenance/storage")
def storage_maintenance_status() -> dict:
    return STORAGE_MAINTENANCE.status()


@app.post("/api/maintenance/storage", status_code=202)
def start_storage_maintenance(request: StorageMaintenanceRequest) -> dict:
    active_manager = manager
    try:
        return STORAGE_MAINTENANCE.start(
            lambda cancel_event, progress: StorageReconciler(
                active_manager.storage_dir,
                active_manager.events.db_path,
                active_manager.recorder,
                cancel_event=cancel_event,
                progress=progress,
            ),
            apply=request.apply,
            full=request.full,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.delete("/api/maintenance/storage", status_code=202)
def cancel_storage_maintenance() -> dict:
    try:
        return STORAGE_MAINTENANCE.cancel()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error



def _run_ffmpeg_list(args: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            [config.ffmpeg_path, "-hide_banner", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return result.stdout or ""
    except Exception:
        return ""


def _dri_render_devices() -> list[str]:
    return sorted(str(path) for path in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []


@functools.lru_cache(maxsize=1)
def _ffmpeg_qsv_info() -> dict:
    hwaccels = _run_ffmpeg_list(["-hwaccels"])
    encoders = _run_ffmpeg_list(["-encoders"])
    decoders = _run_ffmpeg_list(["-decoders"])
    render_devices = _dri_render_devices()
    qsv_encoders = sorted({name for name in ("h264_qsv", "hevc_qsv", "av1_qsv", "mjpeg_qsv") if name in encoders})
    qsv_decoders = sorted({name for name in ("h264_qsv", "hevc_qsv", "av1_qsv", "mjpeg_qsv") if name in decoders})
    listed = "qsv" in hwaccels and "h264_qsv" in encoders
    runtime_usable = False
    runtime_error = ""
    if listed:
        probe_args = [config.ffmpeg_path, "-hide_banner", "-v", "error"]
        if render_devices:
            probe_args.extend(["-qsv_device", render_devices[0]])
        probe_args.extend(["-f", "lavfi", "-i", "color=size=64x64:rate=1", "-frames:v", "1", "-c:v", "h264_qsv", "-f", "null", "-"])
        try:
            probe = subprocess.run(
                probe_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
            runtime_usable = probe.returncode == 0
            runtime_error = "" if runtime_usable else (probe.stderr or "QSV runtime probe failed").strip()[-500:]
        except Exception as exc:
            runtime_error = str(exc) or "QSV runtime probe failed"
    return {
        "available": bool(listed and runtime_usable),
        "listed": bool(listed),
        "runtime_usable": runtime_usable,
        "runtime_error": runtime_error,
        "hwaccel_listed": "qsv" in hwaccels,
        "encoders": qsv_encoders,
        "decoders": qsv_decoders,
        "render_devices": render_devices,
    }


@functools.lru_cache(maxsize=1)
def _ffmpeg_vaapi_info() -> dict:
    hwaccels = _run_ffmpeg_list(["-hwaccels"])
    encoders = _run_ffmpeg_list(["-encoders"])
    decoders = _run_ffmpeg_list(["-decoders"])
    filters = _run_ffmpeg_list(["-filters"])
    render_devices = _dri_render_devices()
    vaapi_encoders = sorted({name for name in ("h264_vaapi", "hevc_vaapi", "av1_vaapi", "mjpeg_vaapi", "mpeg2_vaapi", "vp8_vaapi", "vp9_vaapi") if name in encoders})
    vaapi_decoders = sorted({name for name in ("h264_vaapi", "hevc_vaapi", "av1_vaapi", "mjpeg_vaapi", "mpeg2_vaapi", "vp8_vaapi", "vp9_vaapi") if name in decoders})
    vaapi_filters = sorted({name for name in ("hwupload", "scale_vaapi") if name in filters})
    listed = "vaapi" in hwaccels and "h264_vaapi" in encoders and "hwupload" in filters
    runtime_usable = False
    runtime_error = ""
    if listed and render_devices:
        probe_args = [
            config.ffmpeg_path,
            "-hide_banner",
            "-v",
            "error",
            "-vaapi_device",
            render_devices[0],
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=1",
            "-frames:v",
            "1",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-f",
            "null",
            "-",
        ]
        try:
            probe = subprocess.run(
                probe_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
            )
            runtime_usable = probe.returncode == 0
            runtime_error = "" if runtime_usable else (probe.stderr or "VAAPI runtime probe failed").strip()[-500:]
        except Exception as exc:
            runtime_error = str(exc) or "VAAPI runtime probe failed"
    elif listed:
        runtime_error = "No /dev/dri/renderD* render device found"
    return {
        "available": bool(listed and runtime_usable),
        "listed": bool(listed),
        "runtime_usable": runtime_usable,
        "runtime_error": runtime_error,
        "hwaccel_listed": "vaapi" in hwaccels,
        "encoders": vaapi_encoders,
        "decoders": vaapi_decoders,
        "filters": vaapi_filters,
        "render_devices": render_devices,
        "device": render_devices[0] if render_devices else "",
    }


def _hardware_acceleration_mode() -> str:
    mode = str(getattr(config, "hardware_acceleration", "auto") or "auto").lower()
    return mode if mode in {"auto", "vaapi", "qsv", "off"} else "auto"


def _media_export_hardware_backend() -> str:
    """Resolve the configured, currently usable H.264 export encoder."""
    mode = _hardware_acceleration_mode()
    if mode == "off":
        return "cpu"
    if mode in {"auto", "vaapi"}:
        info = _ffmpeg_vaapi_info()
        if info.get("available") and "h264_vaapi" in set(info.get("encoders") or []):
            return "vaapi"
        if mode == "vaapi":
            return "cpu"
    if mode in {"auto", "qsv"}:
        info = _ffmpeg_qsv_info()
        if info.get("available") and "h264_qsv" in set(info.get("encoders") or []):
            return "qsv"
    return "cpu"


def _media_export_hardware_device(backend: str) -> str:
    info = _ffmpeg_qsv_info() if backend == "qsv" else _ffmpeg_vaapi_info()
    devices = info.get("render_devices") or []
    return str(devices[0]) if devices else str(info.get("device") or "")


def _media_export_manager() -> MediaExportManager:
    global MEDIA_EXPORTS
    with MEDIA_EXPORTS_LOCK:
        if MEDIA_EXPORTS is None:
            MEDIA_EXPORTS = MediaExportManager(
                storage_dir=manager.storage_dir,
                database_dir=manager.database_dir,
                recorder=lambda: manager.recorder,
                ffmpeg_path=lambda: config.ffmpeg_path,
                hardware_backend=_media_export_hardware_backend,
                hardware_device=_media_export_hardware_device,
            )
        return MEDIA_EXPORTS


def _probe_video_codec(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                _ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
        return (result.stdout or "").strip().lower()
    except Exception:
        return ""


def _mp4_boxes(data: bytes | bytearray, start: int = 0, end: int | None = None):
    limit = len(data) if end is None else min(end, len(data))
    cursor = start
    while cursor + 8 <= limit:
        size = struct.unpack_from(">I", data, cursor)[0]
        box_type = bytes(data[cursor + 4:cursor + 8])
        header = 8
        if size == 1 and cursor + 16 <= limit:
            size = struct.unpack_from(">Q", data, cursor + 8)[0]
            header = 16
        elif size == 0:
            size = limit - cursor
        if size < header or cursor + size > limit:
            break
        yield box_type, cursor, cursor + header, cursor + size
        cursor += size


def _mp4_track_timescales(init_data: bytes) -> dict[int, int]:
    timescales: dict[int, int] = {}
    for box_type, _, payload, box_end in _mp4_boxes(init_data):
        if box_type != b"moov":
            continue
        for child_type, _, child_payload, child_end in _mp4_boxes(init_data, payload, box_end):
            if child_type != b"trak":
                continue
            track_id = None
            timescale = None
            for trak_type, _, trak_payload, trak_end in _mp4_boxes(init_data, child_payload, child_end):
                if trak_type == b"tkhd":
                    version = init_data[trak_payload]
                    offset = trak_payload + (20 if version == 1 else 12)
                    if offset + 4 <= trak_end:
                        track_id = struct.unpack_from(">I", init_data, offset)[0]
                elif trak_type == b"mdia":
                    for mdia_type, _, mdia_payload, mdia_end in _mp4_boxes(init_data, trak_payload, trak_end):
                        if mdia_type != b"mdhd":
                            continue
                        version = init_data[mdia_payload]
                        offset = mdia_payload + (20 if version == 1 else 12)
                        if offset + 4 <= mdia_end:
                            timescale = struct.unpack_from(">I", init_data, offset)[0]
            if track_id and timescale:
                timescales[track_id] = timescale
    return timescales


def _offset_fmp4_timestamps(init_path: Path, media_path: Path, seconds: float) -> None:
    if seconds <= 0:
        return
    timescales = _mp4_track_timescales(init_path.read_bytes())
    if not timescales:
        raise RuntimeError("fragment init has no track timescales")
    adjusted = 0
    with media_path.open("r+b") as media_file, mmap.mmap(media_file.fileno(), 0) as data:
        for box_type, _, payload, box_end in _mp4_boxes(data):
            if box_type != b"moof":
                continue
            for child_type, _, child_payload, child_end in _mp4_boxes(data, payload, box_end):
                if child_type != b"traf":
                    continue
                track_id = None
                tfdt = None
                for traf_type, _, traf_payload, traf_end in _mp4_boxes(data, child_payload, child_end):
                    if traf_type == b"tfhd" and traf_payload + 8 <= traf_end:
                        track_id = struct.unpack_from(">I", data, traf_payload + 4)[0]
                    elif traf_type == b"tfdt":
                        tfdt = (traf_payload, traf_end)
                if not track_id or not tfdt or track_id not in timescales:
                    continue
                tfdt_payload, tfdt_end = tfdt
                version = data[tfdt_payload]
                value_offset = tfdt_payload + 4
                increment = round(seconds * timescales[track_id])
                if version == 1 and value_offset + 8 <= tfdt_end:
                    current = struct.unpack_from(">Q", data, value_offset)[0]
                    struct.pack_into(">Q", data, value_offset, current + increment)
                    adjusted += 1
                elif version == 0 and value_offset + 4 <= tfdt_end:
                    current = struct.unpack_from(">I", data, value_offset)[0]
                    next_value = current + increment
                    if next_value > 0xFFFFFFFF:
                        raise RuntimeError("fragment timestamp exceeds version 0 tfdt")
                    struct.pack_into(">I", data, value_offset, next_value)
                    adjusted += 1
        data.flush()
    if not adjusted:
        raise RuntimeError("fragment has no adjustable tfdt boxes")


def _event_clip_cache_suffix(source_codec: str, backend: str) -> str:
    codec = source_codec or "unknown"
    return f"a3-{backend}-{codec}"


def _event_clip_vaapi_enabled(source_codec: str) -> bool:
    mode = _hardware_acceleration_mode()
    if mode not in {"auto", "vaapi"}:
        return False
    if source_codec not in {"h264", "hevc"}:
        return False
    info = _ffmpeg_vaapi_info()
    has_encoder = "h264_vaapi" in set(info.get("encoders") or [])
    return bool(info.get("available") and has_encoder)


def _event_clip_qsv_enabled(source_codec: str) -> bool:
    mode = _hardware_acceleration_mode()
    if mode == "off":
        return False
    if mode == "auto" and _ffmpeg_vaapi_info().get("available"):
        return False
    if mode not in {"auto", "qsv"}:
        return False
    if source_codec not in {"h264", "hevc"}:
        return False
    info = _ffmpeg_qsv_info()
    decoder = f"{source_codec}_qsv"
    has_decoder = decoder in set(info.get("decoders") or [])
    has_encoder = "h264_qsv" in set(info.get("encoders") or [])
    return bool(info.get("available") and has_decoder and has_encoder)


def _event_clip_cpu_command(concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]


def _event_clip_vaapi_command(source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    info = _ffmpeg_vaapi_info()
    device = str(info.get("device") or "/dev/dri/renderD128")
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-vaapi_device",
        device,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        "format=nv12,hwupload",
        "-c:v",
        "h264_vaapi",
        "-qp",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]


def _event_clip_qsv_command(source_codec: str, concat_path: Path, local_start: float, duration: float, tmp_path: Path) -> list[str]:
    decoder = "hevc_qsv" if source_codec == "hevc" else "h264_qsv"
    info = _ffmpeg_qsv_info()
    render_devices = info.get("render_devices") or []
    device_args = ["-qsv_device", render_devices[0]] if render_devices else []
    return [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        *device_args,
        "-hwaccel",
        "qsv",
        "-hwaccel_output_format",
        "qsv",
        "-c:v",
        decoder,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-ss",
        f"{local_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "h264_qsv",
        "-preset",
        "veryfast",
        "-global_quality",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-y",
        str(tmp_path),
    ]

@app.get("/api/accelerator")
def accelerator() -> dict:
    system = platform.system()
    machine = platform.machine()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    openvino_devices: list[str] = []
    openvino_error = ""
    coreml_available = False
    coreml_error = ""
    openvino_probe = manager.detector.probe_devices()
    openvino_devices = list(openvino_probe.get("devices") or [])
    openvino_error = str(openvino_probe.get("error") or "")

    if is_macos:
        try:
            import coremltools  # noqa: F401

            coreml_available = True
        except Exception as exc:
            coreml_error = str(exc) or "Core ML probe failed"

    recommended_backend = "coreml" if is_apple_silicon and coreml_available else "openvino"
    vaapi_info = _ffmpeg_vaapi_info()
    qsv_info = _ffmpeg_qsv_info()

    return {
        "system": system,
        "machine": machine,
        "is_macos": is_macos,
        "is_apple_silicon": is_apple_silicon,
        "has_nvidia": shutil.which("nvidia-smi") is not None,
        "ffmpeg_path": config.ffmpeg_path,
        "ffprobe_path": _ffprobe_path(),
        "ffplay_path": _ffplay_path(),
        "openvino_devices": openvino_devices,
        "openvino_error": openvino_error,
        "coreml_available": coreml_available,
        "coreml_error": coreml_error,
        "recommended_openvino_device": "GPU" if "GPU" in openvino_devices else "CPU",
        "recommended_detector_backend": recommended_backend,
        "ffmpeg_hardware_acceleration": {
            "configured": _hardware_acceleration_mode(),
            "ffmpeg_path": config.ffmpeg_path,
            "ffprobe_path": _ffprobe_path(),
            "ffplay_path": _ffplay_path(),
            "vaapi": vaapi_info,
            "qsv": qsv_info,
        },
    }


@app.get("/api/detector/status")
def detector_status() -> dict:
    return manager.detector_status()


@app.get("/api/object-tracking/catalog")
def object_tracking_catalog() -> dict:
    ultralytics_status = ultralytics_botsort_dependency_status()
    return {
        "active": config.detector.tracking.implementation,
        "implementations": [
            {
                "id": "survng_hybrid",
                "name": "SurvNG Hybrid",
                "available": True,
                "description": "Lightweight geometry tracking with SurvNG appearance recovery.",
            },
            {
                "id": "ultralytics_botsort",
                "name": "Ultralytics BoT-SORT",
                "available": ultralytics_status["available"],
                "description": "Official Kalman/BoT-SORT association using SurvNG person embeddings.",
                "installed_version": ultralytics_status["installed_version"],
                "required_version": ultralytics_status["required_version"],
                "reason": ultralytics_status["reason"] or "",
            },
        ],
    }


@app.get("/api/detector/models")
def detector_models() -> dict:
    models: list[dict] = []
    search_roots = sorted({Path("openvino_model"), *Path(".").glob("*_openvino_model")})
    for root in search_roots:
        if not root.exists():
            continue
        for xml_path in sorted(root.rglob("*.xml")):
            bin_path = xml_path.with_suffix(".bin")
            metadata_path = xml_path.parent / "metadata.yaml"
            item = {
                "path": str(xml_path),
                "name": xml_path.stem,
                "bin_path": str(bin_path),
                "bin_present": bin_path.exists(),
                "metadata_path": str(metadata_path) if metadata_path.exists() else "",
                "task": "",
                "classes": [],
                "input_shape": [],
                "output_shapes": [],
                "valid": False,
                "error": "",
            }
            if metadata_path.exists():
                try:
                    import yaml

                    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                    names = metadata.get("names") or {}
                    if isinstance(names, dict):
                        item["classes"] = [str(value) for _, value in sorted(names.items(), key=lambda entry: int(entry[0]))]
                    elif isinstance(names, list):
                        item["classes"] = [str(value) for value in names]
                    item["task"] = str(metadata.get("task") or "")
                except Exception as exc:
                    item["error"] = f"Metadata: {exc}"
            inspection = manager.detector.inspect_model(str(xml_path))
            item["input_shape"] = list(inspection.get("input_shape") or [])
            item["output_shapes"] = list(inspection.get("output_shapes") or [])
            if inspection.get("error"):
                item["error"] = str(inspection["error"])
            else:
                item["valid"] = bin_path.exists()
            models.append(item)
    return {"models": models, "active_path": config.detector.resolved_model_path()}


@app.get("/api/event-clip/settings")
def event_clip_settings() -> dict:
    before, after = _event_clip_window(None, None)
    return {"before_seconds": before, "after_seconds": after}


@app.get("/api/recordings/cache/status")
def recording_cache_status() -> dict:
    root = manager.storage_dir / "playback-cache" / "fmp4"
    files = [path for path in root.glob("*/*") if path.is_file()] if root.exists() else []
    existing_files: list[Path] = []
    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
            existing_files.append(path)
        except OSError:
            continue
    with RECORDING_CACHE_METRICS_LOCK:
        metrics = dict(RECORDING_CACHE_METRICS)
    for origin in ("playback", "prewarm"):
        remuxes = int(metrics[f"{origin}_remuxes"])
        metrics[f"{origin}_avg_remux_ms"] = round(
            float(metrics[f"{origin}_remux_ms"]) / remuxes,
            1,
        ) if remuxes else 0.0
        metrics[f"{origin}_last_remux_ms"] = round(float(metrics[f"{origin}_last_remux_ms"]), 1)
        metrics.pop(f"{origin}_remux_ms", None)
    return {
        "entries": len({path.parent for path in existing_files}),
        "bytes": total_bytes,
        "max_bytes": int(float(config.recording_cache_max_gb) * 1024 * 1024 * 1024),
        "max_days": int(config.recording_cache_max_days),
        "prewarm": bool(config.recording_cache_prewarm),
        "metrics": metrics,
    }


@app.get("/api/system/status")
def system_status() -> dict:
    storage_path = manager.storage_dir
    storage_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(storage_path)
    cameras = manager.statuses()
    detector = manager.detector_status()
    return {
        "instance_id": PROCESS_INSTANCE_ID,
        "storage": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round((usage.used / usage.total) * 100, 1) if usage.total else 0,
        },
        "detector": detector,
        "cameras": {
            "total": len(cameras),
            "online": sum(1 for camera in cameras if camera.get("running")),
            "recording": sum(1 for camera in cameras if camera.get("recording")),
        },
        "mqtt": manager.mqtt_status(),
        "go2rtc": manager.go2rtc_status(),
    }


def _linux_memory_status() -> dict[str, int | float]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {"total_bytes": 0, "available_bytes": 0, "used_bytes": 0, "used_percent": 0.0}
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "used_percent": round((used / total) * 100, 1) if total else 0.0,
    }


def _process_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _cgroup_memory_status(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    process_cgroup: Path = Path("/proc/self/cgroup"),
) -> dict[str, int]:
    """Return service-wide cgroup v2 memory without conflating file cache with heap."""
    empty = {
        "total_bytes": 0,
        "application_bytes": 0,
        "file_cache_bytes": 0,
        "reclaimable_file_cache_bytes": 0,
        "kernel_bytes": 0,
    }
    try:
        relative = next(
            line.split(":", 2)[2].strip().lstrip("/")
            for line in process_cgroup.read_text(encoding="utf-8").splitlines()
            if line.startswith("0::")
        )
        base = cgroup_root / relative
        total = int((base / "memory.current").read_text(encoding="utf-8").strip())
        stats = {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1)
                for line in (base / "memory.stat").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    except (OSError, StopIteration, ValueError):
        return empty
    shmem = max(0, stats.get("shmem", 0))
    return {
        "total_bytes": max(0, total),
        "application_bytes": max(0, stats.get("anon", 0)) + shmem,
        "file_cache_bytes": max(0, stats.get("file", 0) - shmem),
        "reclaimable_file_cache_bytes": max(0, stats.get("inactive_file", 0) - shmem),
        "kernel_bytes": max(0, stats.get("kernel", 0)),
    }


def _database_bytes() -> int:
    total = 0
    for path in manager.database_dir.glob("*.sqlite3*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _read_integer(path: Path, *, scale: int = 1) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0) * scale
    except (OSError, ValueError):
        return None


def _drm_worker_counters(pid: int) -> dict[str, object]:
    engines: dict[str, int] = {}
    allocated_bytes = 0
    resident_bytes = 0
    clients: set[str] = set()
    driver = ""
    try:
        fdinfo_paths = list(Path(f"/proc/{pid}/fdinfo").iterdir())
    except OSError:
        return {"engines": engines, "allocated_bytes": 0, "resident_bytes": 0, "driver": ""}
    for path in fdinfo_paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        values: dict[str, str] = {}
        for line in lines:
            if not line.startswith("drm-") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key] = value.strip()
        if not values:
            continue
        driver = values.get("drm-driver", driver)
        client_id = values.get("drm-client-id", str(path))
        for key, value in values.items():
            if not key.startswith("drm-engine-"):
                continue
            try:
                nanoseconds = int(value.split()[0])
            except (ValueError, IndexError):
                continue
            name = key.removeprefix("drm-engine-")
            engines[name] = engines.get(name, 0) + nanoseconds
        if client_id in clients:
            continue
        clients.add(client_id)
        for key, target in (("drm-total-system0", "allocated"), ("drm-resident-system0", "resident")):
            try:
                byte_count = int(values.get(key, "0").split()[0]) * 1024
            except (ValueError, IndexError):
                byte_count = 0
            if target == "allocated":
                allocated_bytes += byte_count
            else:
                resident_bytes += byte_count
    return {
        "engines": engines,
        "allocated_bytes": allocated_bytes,
        "resident_bytes": resident_bytes,
        "driver": driver,
    }


def _gpu_status(detector: dict) -> dict[str, object]:
    workers = detector.get("workers") or {}
    pids = tuple(sorted({
        int(worker.get("worker_pid") or 0)
        for worker in workers.values()
        if isinstance(worker, dict) and worker.get("worker_alive") and int(worker.get("worker_pid") or 0) > 0
    })) if isinstance(workers, dict) else ()
    engines: dict[str, int] = {}
    allocated_bytes = 0
    resident_bytes = 0
    driver = ""
    for pid in pids:
        counters = _drm_worker_counters(pid)
        driver = str(counters.get("driver") or driver)
        allocated_bytes += int(counters.get("allocated_bytes") or 0)
        resident_bytes += int(counters.get("resident_bytes") or 0)
        for name, value in dict(counters.get("engines") or {}).items():
            engines[str(name)] = engines.get(str(name), 0) + int(value)

    sampled_at = time.monotonic()
    engine_usage: dict[str, float] = {}
    with GPU_SAMPLE_LOCK:
        previous_at = float(GPU_SAMPLE.get("at") or 0.0)
        previous_pids = tuple(GPU_SAMPLE.get("pids") or ())
        previous_engines = dict(GPU_SAMPLE.get("engines") or {})
        elapsed_ns = (sampled_at - previous_at) * 1_000_000_000
        if pids and pids == previous_pids and elapsed_ns > 0:
            for name, value in engines.items():
                delta = value - int(previous_engines.get(name, value))
                if delta >= 0:
                    engine_usage[name] = round(min(100.0, (delta / elapsed_ns) * 100.0), 1)
        GPU_SAMPLE.update({"at": sampled_at, "pids": pids, "engines": engines})

    card = Path("/sys/class/drm/card0")
    vendor_id = _read_integer(card / "device/vendor")
    device_id = _read_integer(card / "device/device")
    current_frequency = _read_integer(card / "gt_act_freq_mhz")
    maximum_frequency = _read_integer(card / "gt_max_freq_mhz")
    temperature_millidegrees = next(
        (value for value in (_read_integer(path) for path in (card / "device/hwmon").glob("hwmon*/temp1_input")) if value is not None),
        None,
    )
    vendor_names = {0x8086: "Intel", 0x1002: "AMD", 0x10DE: "NVIDIA"}
    vendor = vendor_names.get(vendor_id, f"0x{vendor_id:04x}" if vendor_id is not None else "Unknown")
    utilization = round(min(100.0, sum(engine_usage.values())), 1) if engine_usage else None
    return {
        "available": bool(card.exists() or engines),
        "scope": "SurvNG inference workers",
        "vendor": vendor,
        "device_id": f"0x{device_id:04x}" if device_id is not None else "",
        "driver": driver,
        "worker_pids": list(pids),
        "utilization_percent": utilization,
        "engine_utilization": engine_usage,
        "allocated_bytes": allocated_bytes,
        "resident_bytes": resident_bytes,
        "current_frequency_mhz": current_frequency,
        "maximum_frequency_mhz": maximum_frequency,
        "temperature_celsius": round(temperature_millidegrees / 1000.0, 1) if temperature_millidegrees is not None else None,
        "sample_ready": utilization is not None,
    }


def _record_telemetry_history(sample: dict[str, object], sampled_at: float) -> list[dict[str, object]]:
    """Keep one rolling system sample per five seconds for up to one hour."""
    with TELEMETRY_HISTORY_LOCK:
        last_sample_at = TELEMETRY_HISTORY_STATE["last_sample_at"]
        if not TELEMETRY_HISTORY or sampled_at - last_sample_at >= 5.0:
            TELEMETRY_HISTORY.append(sample)
            TELEMETRY_HISTORY_STATE["last_sample_at"] = sampled_at
        else:
            TELEMETRY_HISTORY[-1] = sample
        return [dict(item) for item in TELEMETRY_HISTORY]


def _persisted_telemetry_history(camera_id: str) -> dict[str, Any]:
    """Cache database-backed graph series for one sampling interval."""
    cache_key = (id(manager.events), camera_id)
    now = time.monotonic()
    with TELEMETRY_PERSISTED_CACHE_LOCK:
        cached = TELEMETRY_PERSISTED_CACHE.get(cache_key)
        if cached is not None and now - float(cached["at"]) < 55.0:
            return cached["value"]
    value = {
        "runtime": {
            "short": manager.events.runtime_telemetry_history(
                hours=2, bucket_minutes=1, camera_id=camera_id
            ),
            "long": manager.events.runtime_telemetry_history(
                hours=168, bucket_minutes=15, camera_id=camera_id
            ),
        },
        "tracking": {
            "short": manager.events.tracking_capacity_activity(
                hours=2, bucket_minutes=1, camera_id=camera_id
            ),
            "long": manager.events.tracking_capacity_activity(
                hours=168, bucket_minutes=15, camera_id=camera_id
            ),
        },
    }
    with TELEMETRY_PERSISTED_CACHE_LOCK:
        if len(TELEMETRY_PERSISTED_CACHE) >= 32:
            TELEMETRY_PERSISTED_CACHE.clear()
        TELEMETRY_PERSISTED_CACHE[cache_key] = {"at": now, "value": value}
    return value


@app.get("/api/telemetry")
def telemetry(hours: int = 24, camera_id: str = "") -> dict:
    """Operational history and runtime counters for the Config telemetry view."""
    storage_path = manager.storage_dir
    storage_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(storage_path)
    camera_statuses = manager.statuses()
    detector = manager.detector_status()
    gpu = _gpu_status(detector)
    activity = manager.events.telemetry_activity(hours=hours)
    selected_camera_id = str(camera_id or "").strip()[:128]
    persisted_history = _persisted_telemetry_history(selected_camera_id)
    runtime_history = persisted_history["runtime"]
    tracking_capacity_history = persisted_history["tracking"]
    per_camera_activity = activity.get("by_camera", {})
    load_1m, load_5m, load_15m = os.getloadavg()
    memory = _linux_memory_status()
    process_rss_bytes = _process_rss_bytes()
    service_memory = _cgroup_memory_status()
    cpu_count = os.cpu_count() or 1
    generated_at = datetime.now(timezone.utc).isoformat()
    cameras = []
    for status in camera_statuses:
        status_camera_id = str(status.get("id") or "")
        motion = status.get("motion_qualification") or {}
        tracking = status.get("object_tracking") or {}
        cameras.append({
            "id": status_camera_id,
            "name": status.get("name") or status_camera_id,
            "connected": bool(status.get("connected")),
            "frame_fresh": bool(status.get("frame_fresh")),
            "last_frame_age_seconds": status.get("last_frame_age_seconds"),
            "recording": bool(status.get("recording") or status.get("sub_recording")),
            "recording_timestamps": dict(status.get("recording_timestamp_health") or {}),
            "detection_enabled": bool(status.get("detection_enabled")),
            "onvif": {
                "enabled": bool(status.get("onvif_enabled")),
                "connected": bool(status.get("onvif_connected")),
                "notifications": int(status.get("onvif_notifications_received") or 0),
                "motion_events": int(status.get("onvif_motion_events_received") or 0),
                "poll_errors": int(status.get("onvif_poll_errors") or 0),
                "poll_timeouts": int(status.get("onvif_poll_timeouts") or 0),
                "renewals": int(status.get("onvif_renewals") or 0),
                "renewal_errors": int(status.get("onvif_renewal_errors") or 0),
                "last_motion_at": status.get("onvif_last_motion_event_at"),
                "last_error": status.get("onvif_last_error") or "",
            },
            "motion": {
                "mode": motion.get("mode") or "",
                "triggers": int(motion.get("triggers") or 0),
                "bursts": int(motion.get("bursts") or 0),
                "passed": int(motion.get("passed") or 0),
                "rejected": int(motion.get("audit_rejected") or 0),
                "suppressed": int(motion.get("suppressed") or 0),
                "dropped": int(motion.get("dropped_triggers") or 0),
                "analysis_frames_dropped": int(motion.get("analysis_frames_dropped") or 0),
                "queue_depth": int(motion.get("queue_depth") or 0),
                "visual_backup_candidates": int(motion.get("visual_backup_candidates") or 0),
                "visual_backup_triggers": int(motion.get("visual_backup_triggers") or 0),
                "visual_backup_onvif_matches": int(motion.get("visual_backup_onvif_matches") or 0),
                "visual_backup_rate_limited": int(motion.get("visual_backup_rate_limited") or 0),
                "visual_backup_not_ready": int(motion.get("visual_backup_not_ready") or 0),
                "visual_backup_uncorrelated_objects": int(motion.get("visual_backup_uncorrelated_objects") or 0),
            },
            "tracking": {
                "active": bool(tracking.get("active")),
                "frames_processed": int(tracking.get("frames_processed") or 0),
                "track_count": int(tracking.get("track_count") or 0),
                "capacity_requests": int(tracking.get("capacity_requests") or 0),
                "capacity_waits": int(tracking.get("capacity_waits") or 0),
                "capacity_timeouts": int(tracking.get("capacity_timeouts") or 0),
                "capacity_wait_seconds_total": float(tracking.get("capacity_wait_seconds_total") or 0.0),
                "capacity_wait_seconds_max": float(tracking.get("capacity_wait_seconds_max") or 0.0),
                "capacity_wait_seconds_last": float(tracking.get("capacity_wait_seconds_last") or 0.0),
                "reid_attempts": int(tracking.get("reid_attempts") or 0),
                "reid_successes": int(tracking.get("reid_successes") or 0),
                "reid_failures": int(tracking.get("reid_failures") or 0),
                "reid_average_ms": float(tracking.get("reid_average_ms") or 0.0),
                "reid_attempts_by_label": dict(tracking.get("reid_attempts_by_label") or {}),
                "reid_attempts_by_reason": dict(tracking.get("reid_attempts_by_reason") or {}),
                "reid_recoveries": int(tracking.get("reid_recoveries") or 0),
                "reid_recoveries_by_label": dict(tracking.get("reid_recoveries_by_label") or {}),
                "reid_avoided_geometry_matches": int(
                    tracking.get("reid_avoided_geometry_matches") or 0
                ),
                "reid_avoided_by_label": dict(tracking.get("reid_avoided_by_label") or {}),
            },
            "capture": dict(status.get("capture_stats") or {}),
            "activity": per_camera_activity.get(
                status_camera_id,
                {
                    "last_hour": {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}},
                    "last_24h": {"events": 0, "object_incidents": 0, "objects": 0, "labels": {}},
                    "hourly": [],
                },
            ),
        })
    detector_runtime = detector.get("runtime") or {}
    history = _record_telemetry_history(
        {
            "sampled_at": generated_at,
            "cpu_load_percent": round(min(100.0, (load_1m / cpu_count) * 100.0), 2),
            "memory_used_percent": float(memory.get("used_percent") or 0.0),
            "storage_used_percent": round((usage.used / usage.total) * 100.0, 3) if usage.total else 0.0,
            "process_rss_bytes": process_rss_bytes,
            "service_application_bytes": service_memory["application_bytes"],
            "service_file_cache_bytes": service_memory["file_cache_bytes"],
            "gpu_utilization_percent": gpu.get("utilization_percent"),
            "inference_ms": detector_runtime.get("average_inference_ms"),
            "detection_fps": detector_runtime.get("detection_fps"),
        },
        time.monotonic(),
    )
    return {
        "generated_at": generated_at,
        "system": {
            "uptime_seconds": round(max(0.0, time.monotonic() - PROCESS_STARTED_MONOTONIC), 1),
            "cpu_count": cpu_count,
            "load_average": {"one": round(load_1m, 2), "five": round(load_5m, 2), "fifteen": round(load_15m, 2)},
            "memory": memory,
            "process_rss_bytes": process_rss_bytes,
            "service_memory": service_memory,
            "storage": {
                "path": str(storage_path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": round((usage.used / usage.total) * 100, 1) if usage.total else 0.0,
            },
            "database": {"path": str(manager.database_dir), "bytes": _database_bytes()},
        },
        "detector": detector,
        "gpu": gpu,
        "history": history,
        "runtime_history": runtime_history,
        "tracking_capacity_history": tracking_capacity_history,
        "tracking_capacity": {
            "limit": int(config.detector.tracking.max_active_cameras),
            "wait_seconds": float(config.detector.tracking.capacity_wait_seconds),
            "active": sum(1 for item in cameras if item["tracking"]["active"]),
            "requests_since_restart": sum(item["tracking"]["capacity_requests"] for item in cameras),
            "waits_since_restart": sum(item["tracking"]["capacity_waits"] for item in cameras),
            "timeouts_since_restart": sum(item["tracking"]["capacity_timeouts"] for item in cameras),
        },
        "activity": activity,
        "cameras": cameras,
    }


@app.put("/api/config")
def put_config(next_config: AppConfig) -> dict:
    try:
        next_config = _restore_config_secrets(next_config, config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        validate_motion_pipeline_configuration(next_config)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    effective_config, apply_result = apply_config_update(next_config, assign_ids=True)
    return {"ok": True, "cameras": len(effective_config.cameras), **apply_result}


@app.put("/api/config/cameras/{camera_id}/zones")
def put_camera_zones(camera_id: str, zones: list[DetectionZone]) -> dict:
    global config
    with MANAGER_RELOAD_LOCK:
        next_config = config.model_copy(deep=True)
        camera = camera_by_id(next_config, camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="camera not found")
        worker = manager.workers.get(camera_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="camera worker not found")
        current_camera = camera_by_id(config, camera_id)
        previous_zones = [zone.model_dump(mode="json") for zone in (current_camera.zones if current_camera else [])]
        camera.zones = zones
        next_zone_payload = [zone.model_dump(mode="json") for zone in camera.zones]
        try:
            manager.update_camera_zones(camera_id, camera.zones, previous_zones)
            save_config(next_config, assign_ids=False)
        except Exception:
            try:
                manager.update_camera_zones(
                    camera_id,
                    [DetectionZone.model_validate(zone) for zone in previous_zones],
                    next_zone_payload,
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "failed to roll back camera zones for %s",
                    camera_id,
                )
            raise
        config = next_config
        manager.config = next_config
    return {
        "ok": True,
        "camera_id": camera.id,
        "zones": [zone.model_dump(mode="json") for zone in camera.zones],
        "workers_restarted": False,
    }


@app.put("/api/config/cameras/order")
def put_camera_order(camera_ids: list[str]) -> dict:
    global config
    with MANAGER_RELOAD_LOCK:
        existing_ids = [camera.id for camera in config.cameras]
        if len(camera_ids) != len(existing_ids) or len(set(camera_ids)) != len(camera_ids):
            raise HTTPException(status_code=400, detail="camera order must contain every camera exactly once")
        if set(camera_ids) != set(existing_ids):
            raise HTTPException(status_code=400, detail="camera order does not match configured cameras")
        camera_by_identifier = {camera.id: camera for camera in config.cameras}
        next_config = config.model_copy(deep=True)
        next_config.cameras = [camera_by_identifier[camera_id].model_copy(deep=True) for camera_id in camera_ids]
        save_config(next_config, assign_ids=False)

        config = next_config
        manager.config = next_config
        manager.workers = {camera_id: manager.workers[camera_id] for camera_id in camera_ids}
    return {"ok": True, "camera_ids": camera_ids}


@app.put("/api/config/cameras/{camera_id}")
def put_camera(camera_id: str, camera_settings: CameraConfig) -> dict:
    with MANAGER_RELOAD_LOCK:
        next_config = config.model_copy(deep=True)
        existing_index = next((index for index, item in enumerate(next_config.cameras) if item.id == camera_id), None)
        existing = next_config.cameras[existing_index] if existing_index is not None else None
        try:
            camera_settings = _restore_camera_secrets(camera_settings, existing)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        used_ids = {item.id for item in next_config.cameras if item.id != camera_id}
        base_id = slugify_camera_id(camera_settings.name or camera_settings.id)
        next_id = base_id
        suffix = 2
        while next_id in used_ids:
            next_id = f"{base_id}-{suffix}"
            suffix += 1
        camera_settings.id = next_id
        camera_settings.zones = existing.zones if existing is not None else []
        if existing_index is None:
            next_config.cameras.append(camera_settings)
        else:
            next_config.cameras[existing_index] = camera_settings
        try:
            validate_motion_pipeline_configuration(next_config)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        _effective, apply_result = apply_config_update(next_config)
    return {
        "ok": True,
        "camera": _redacted_camera_payload(camera_settings),
        **apply_result,
    }


@app.delete("/api/config/cameras/{camera_id}")
def delete_camera(camera_id: str) -> dict:
    with MANAGER_RELOAD_LOCK:
        next_config = config.model_copy(deep=True)
        remaining = [camera for camera in next_config.cameras if camera.id != camera_id]
        if len(remaining) == len(next_config.cameras):
            raise HTTPException(status_code=404, detail="camera not found")
        next_config.cameras = remaining
        reload_manager(next_config)
    return {"ok": True, "camera_id": camera_id}


@app.post("/api/config/probe")
def probe_config(payload: ConfigProbeRequest) -> dict:
    if not CONFIG_PROBE_LIMITER.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="too many camera probes are already running",
            headers={"Retry-After": "2"},
        )
    try:
        host = payload.host.strip()
        username = payload.username.strip()
        password = payload.password
        if password == SECRET_PLACEHOLDER:
            existing = camera_by_id(config, payload.camera_id)
            if existing is None:
                raise HTTPException(
                    status_code=422,
                    detail="masked probe credentials require an existing camera",
                )
            password = existing.onvif.password or existing.baichuan.password
        result = {
            "host": host,
            "onvif": {
                "port": payload.onvif_port,
                "reachable": _tcp_reachable(host, payload.onvif_port),
                "capabilities": {},
                "error": "",
            },
            "baichuan": {
                "port": payload.baichuan_port,
                "reachable": _tcp_reachable(host, payload.baichuan_port),
            },
            "reolink_likely": False,
        }
        result["reolink_likely"] = bool(result["baichuan"]["reachable"])

        if result["onvif"]["reachable"] and username and password:
            try:
                from onvif import ONVIFCamera
                from zeep import Transport

                transport = Transport(operation_timeout=5)
                camera = ONVIFCamera(
                    host,
                    payload.onvif_port,
                    username,
                    password,
                    transport=transport,
                )
                device = camera.create_devicemgmt_service()
                capabilities = device.GetCapabilities({"Category": "All"})
                result["onvif"]["capabilities"] = {
                    "media": bool(getattr(capabilities, "Media", None)),
                    "events": bool(getattr(capabilities, "Events", None)),
                    "ptz": bool(getattr(capabilities, "PTZ", None)),
                    "analytics": bool(getattr(capabilities, "Analytics", None)),
                }
            except Exception as exc:
                error = redact_log_message(str(exc) or "ONVIF capability probe failed")
                result["onvif"]["error"] = error[:500]
        return result
    finally:
        CONFIG_PROBE_LIMITER.release()


@app.get("/api/events")
def events(limit: int = 100) -> list[dict]:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        rows = active_manager.events.recent(limit)
    return [_event_row(row) for row in rows]


def _motion_audit_row(row: dict, storage_dir: Path) -> dict:
    audit = dict(row)
    try:
        features = json.loads(str(audit.pop("features_json", "{}") or "{}"))
    except (json.JSONDecodeError, TypeError):
        features = {}
    audit["features"] = features if isinstance(features, dict) else {}
    snapshot_path = str(audit.pop("snapshot_path", "") or "")
    try:
        event_snapshot_path(storage_dir, {"snapshot_path": snapshot_path})
        audit["has_snapshot"] = True
    except (FileNotFoundError, PermissionError):
        audit["has_snapshot"] = False
    raw_outcome = audit.get("object_detected")
    audit["object_detected"] = None if raw_outcome is None else bool(raw_outcome)
    audit["interpretation"] = motion_audit_interpretation(
        reason=audit.get("reason"),
        event_id=audit.get("event_id"),
        object_detected=audit["object_detected"],
    )
    return audit


class AuditAiApplyRequest(BaseModel):
    changes: list[AuditAiChange] = Field(default_factory=list, max_length=8)
    confirmed: bool = False
    configuration_fingerprint: str = Field(default="", max_length=64)
    recommendation_proof: str = Field(default="", max_length=256)


class IncidentAiApplyRequest(BaseModel):
    changes: list[AuditAiChange] = Field(default_factory=list, max_length=8)
    confirmed: bool = False
    configuration_fingerprint: str = Field(default="", max_length=64)
    recommendation_proof: str = Field(default="", max_length=256)


class MotionAiReviewRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    hours: float = Field(default=24.0, ge=1.0, le=168.0)
    record_limit: int = Field(default=100, ge=20, le=100)
    image_limit: int = Field(default=12, ge=4, le=24)


class CameraIntelligenceApplyRequest(IncidentAiApplyRequest):
    evaluation_hours: float = Field(default=24.0, ge=24.0, le=168.0)


class CameraIntelligenceFollowupRequest(BaseModel):
    image_limit: int = Field(default=12, ge=4, le=24)


class TrackingComparisonVerdictRequest(BaseModel):
    verdict: str = Field(
        pattern=r"^(survng_hybrid|ultralytics_botsort|inconclusive)$"
    )


def _audit_ai_context(
    audit: dict,
    active_config: AppConfig,
    active_manager: AppManager,
) -> dict:
    camera_id = str(audit.get("camera_id") or "")
    camera = camera_by_id(active_config, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="audit camera not found")
    try:
        features = json.loads(str(audit.get("features_json") or "{}"))
    except (json.JSONDecodeError, TypeError):
        features = {}
    if not isinstance(features, dict):
        features = {}
    pipeline_telemetry = features.pop("pipeline_telemetry", {})
    event = active_manager.events.get(int(audit["event_id"])) if audit.get("event_id") else None
    detected_objects: list[dict] = []
    qualification: dict = {}
    object_tracking: dict[str, Any] = {}
    if event:
        try:
            entries = json.loads(str(event.get("objects_json") or "[]"))
        except json.JSONDecodeError:
            entries = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            if entry.get("label"):
                detected_objects.append({
                    "label": entry.get("label"),
                    "confidence": entry.get("confidence"),
                    "box": entry.get("box"),
                    "zones": entry.get("zones") or entry.get("zone_matches") or [],
                    "incident_eligible": entry.get("incident_eligible", True),
                    "temporal_consensus": entry.get("temporal_consensus"),
                    "temporal_sample_offset_seconds": entry.get(
                        "temporal_sample_offset_seconds"
                    ),
                    "temporal_observations": entry.get("temporal_observations"),
                    "temporal_track_observations": entry.get(
                        "temporal_track_observations"
                    ),
                    "temporal_incident_observations": entry.get(
                        "temporal_incident_observations"
                    ),
                    "temporal_required_observations": entry.get(
                        "temporal_required_observations"
                    ),
                    "temporal_samples": entry.get("temporal_samples"),
                    "temporal_peak_confidence": entry.get(
                        "temporal_peak_confidence"
                    ),
                    "temporal_label_votes": entry.get("temporal_label_votes"),
                    "temporal_center_displacement_ratio": entry.get("temporal_center_displacement_ratio"),
                    "temporal_center_path_ratio": entry.get("temporal_center_path_ratio"),
                    "temporal_first_observation_offset_seconds": entry.get("temporal_first_observation_offset_seconds"),
                    "temporal_last_observation_offset_seconds": entry.get("temporal_last_observation_offset_seconds"),
                    "temporal_newly_appeared": entry.get("temporal_newly_appeared"),
                    "motion_correlated": entry.get("motion_correlated"),
                    "motion_correlation": entry.get("motion_correlation"),
                    "motion_correlation_threshold": entry.get("motion_correlation_threshold"),
                    "motion_temporal_evidence_available": entry.get("motion_temporal_evidence_available"),
                    "track_id": entry.get("track_id"),
                    "track_state": entry.get("track_state"),
                    "track_observations": entry.get("track_observations"),
                })
            if entry.get("status") == "motion_qualification":
                candidate = entry.get("motion_qualification")
                qualification = candidate if isinstance(candidate, dict) else {}
            if entry.get("status") == "object_tracking":
                candidate = entry.get("object_tracking")
                object_tracking = candidate if isinstance(candidate, dict) else {}
    override = camera.motion_qualification
    graphs = resolve_motion_pipeline_graphs(active_config.motion_qualification, override)
    effective_mode = (
        active_config.motion_qualification.mode
        if override.mode == "inherit"
        else override.mode
    )
    fusion = guided_fusion_settings(graphs.fusion)
    mog2_available = any(
        stage.implementation == "opencv_mog2_evidence"
        and bool(stage.options.get("enabled", True))
        for stage in graphs.observation
    )
    require_incident_zone = (
        active_config.detector.require_incident_zone
        if camera.require_incident_zone is None
        else camera.require_incident_zone
    )
    suppression_verification_rate = (
        active_config.motion_qualification.suppression_verification_rate
        if override.suppression_verification_rate is None
        else override.suppression_verification_rate
    )
    effective = {
        "mode": effective_mode,
        "sensitivity": active_config.motion_qualification.sensitivity if override.sensitivity == "inherit" else override.sensitivity,
        "stationary_object_tolerance": (
            active_config.motion_qualification.stationary_object_tolerance
            if override.stationary_object_tolerance == "inherit"
            else override.stationary_object_tolerance
        ),
        "frame_width": override.frame_width or active_config.motion_qualification.frame_width,
        "borderline_rescue_enabled": (
            active_config.motion_qualification.borderline_rescue_enabled
            if override.borderline_rescue_enabled is None
            else override.borderline_rescue_enabled
        ),
        "borderline_margin": (
            active_config.motion_qualification.borderline_margin
            if override.borderline_margin is None
            else override.borderline_margin
        ),
        "mog2_available": mog2_available,
        "mog2_validation_enabled": mog2_available and "mog2" in fusion.get("sources", []),
        "mog2_history_seconds": active_config.motion_qualification.mog2_history_seconds,
        "sample_fps": active_config.motion_qualification.sample_fps,
        "window_seconds": active_config.motion_qualification.window_seconds,
        "post_trigger_seconds": active_config.motion_qualification.post_trigger_seconds,
        "burst_quiet_seconds": active_config.motion_qualification.burst_quiet_seconds,
        "camera_mode_background_fps": active_config.motion_qualification.camera_mode_background_fps,
        "visual_backup_warmup_seconds": active_config.motion_qualification.visual_backup_warmup_seconds,
        "visual_backup_grace_seconds": active_config.motion_qualification.visual_backup_grace_seconds,
        "visual_backup_min_score": active_config.motion_qualification.visual_backup_min_score,
        "visual_backup_score_margin": active_config.motion_qualification.visual_backup_score_margin,
        "visual_backup_min_consecutive": active_config.motion_qualification.visual_backup_min_consecutive,
        "visual_backup_cooldown_seconds": active_config.motion_qualification.visual_backup_cooldown_seconds,
        "visual_backup_max_triggers_5m": active_config.motion_qualification.visual_backup_max_triggers_5m,
        "rejected_sample_rate": active_config.motion_qualification.rejected_sample_rate,
        "suppression_verification_rate": suppression_verification_rate,
        "analysis_preset": identify_analysis_preset(graphs.qualification),
        "object_confirmation_frames": (
            active_config.detector.event_confirmation_frames
        ),
        "object_class_confirmation_frames": dict(
            active_config.detector.event_class_confirmation_frames
        ),
        "object_class_confidence_thresholds": dict(
            active_config.detector.event_class_confidence_thresholds
        ),
        "incident_eligibility_policy": (
            "zones_only" if require_incident_zone else "zones_plus_full_frame"
        ),
        "object_tracking": {
            "enabled": active_config.detector.tracking.enabled,
            "implementation": active_config.detector.tracking.implementation,
            "sample_fps": active_config.detector.tracking.sample_fps,
            "reid_enabled": active_config.detector.tracking.reid_enabled,
            "vehicle_reid_enabled": active_config.detector.tracking.vehicle_reid_enabled,
        },
        "fusion": fusion,
        "pipeline_origins": graphs.origins,
    }
    recent, _ = active_manager.events.motion_audits(limit=50, camera_id=camera_id)
    reason_counts: dict[str, int] = {}
    object_matches = 0
    for row in recent:
        reason = str(row.get("reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        object_matches += int(row.get("object_detected") == 1)
    interpretation = motion_audit_interpretation(
        reason=audit.get("reason"),
        event_id=audit.get("event_id"),
        object_detected=audit.get("object_detected"),
    )
    related_prior_event: dict[str, Any] | None = None
    if interpretation["category"] in {"duplicate_active_event", "duplicate_event_cooldown"}:
        related_id = audit.get("related_event_id")
        prior = active_manager.events.get(int(related_id)) if related_id else None
        try:
            audit_at = datetime.fromisoformat(str(audit.get("created_at")))
        except (TypeError, ValueError):
            audit_at = None
        if prior is None and audit_at is not None:
            nearby_events = active_manager.events.for_camera_range(
                camera_id,
                (audit_at - timedelta(seconds=60)).isoformat(),
                audit_at.isoformat(),
                limit=50,
            )
            prior = nearby_events[-1] if nearby_events else None
        if prior is not None:
            try:
                prior_entries = json.loads(str(prior.get("objects_json") or "[]"))
            except (json.JSONDecodeError, TypeError):
                prior_entries = []
            prior_objects = [
                {
                    "label": entry.get("label"),
                    "confidence": entry.get("confidence"),
                    "incident_eligible": entry.get("incident_eligible", True),
                }
                for entry in prior_entries if isinstance(entry, dict) and entry.get("label")
            ] if isinstance(prior_entries, list) else []
            try:
                if audit_at is None:
                    raise ValueError("audit timestamp unavailable")
                seconds_before = max(
                    0.0,
                    (audit_at - datetime.fromisoformat(str(prior.get("created_at")))).total_seconds(),
                )
            except (TypeError, ValueError):
                seconds_before = None
            related_prior_event = {
                "event_id": prior.get("id"),
                "created_at": prior.get("created_at"),
                "seconds_before": seconds_before,
                "objects": prior_objects,
            }
    return {
        "motion_paradigm": motion_paradigm_context(
            mode=effective_mode,
            onvif_enabled=camera.onvif.enabled,
            has_live_substream=bool(camera.live_stream_url),
            fusion=fusion,
            mog2_available=mog2_available,
            require_incident_zone=require_incident_zone,
        ),
        "audit": {
            "id": audit.get("id"),
            "camera_id": camera_id,
            "created_at": audit.get("created_at"),
            "score": audit.get("score"),
            "threshold": audit.get("threshold"),
            "reason": audit.get("reason"),
            "category": audit.get("category") or "qualification",
            "mode": audit.get("mode"),
            "sensitivity": audit.get("sensitivity"),
            "decision_id": audit.get("decision_id"),
            "event_id": audit.get("event_id"),
            "related_event_id": audit.get("related_event_id"),
            "features": features,
            "trigger_count": audit.get("trigger_count"),
            "object_detected": None if audit.get("object_detected") is None else bool(audit.get("object_detected")),
            "qualification": qualification,
            "pipeline_telemetry": pipeline_telemetry,
        },
        "decision_outcome": {
            "interpretation": interpretation,
            "filtered_before_object_detection": bool(
                not audit.get("event_id") and audit.get("object_detected") is None
            ),
            "object_detection_ran": bool(
                audit.get("event_id") or audit.get("object_detected") is not None
            ),
            "object_detection_completed": audit.get("object_detected") is not None,
            "object_detected": (
                None
                if audit.get("object_detected") is None
                else bool(audit.get("object_detected"))
            ),
            "eligible_object_found": audit.get("object_detected") == 1,
            "visual_backup": audit.get("category") == "visual_backup",
        },
        "related_prior_event": related_prior_event,
        "detected_objects": detected_objects,
        "object_tracking": object_tracking,
        "effective_settings": effective,
        "recent_camera_audits": {
            "sample_size": len(recent),
            "object_matches": object_matches,
            "reason_counts": reason_counts,
        },
        "setting_bounds": {
            "frame_width": [240, 960],
            "sample_fps": [2, 10],
            "window_seconds": [0.8, 4],
            "post_trigger_seconds": [0.5, 6],
            "burst_quiet_seconds": [0.1, 2],
            "borderline_margin": [0, 0.10],
            "mog2_history_seconds": [5, 300],
            "visual_backup_warmup_seconds": [0, 120],
            "visual_backup_grace_seconds": [0, 5],
            "visual_backup_min_score": [0, 1],
            "visual_backup_score_margin": [0, 0.5],
            "visual_backup_min_consecutive": [2, 10],
            "visual_backup_cooldown_seconds": [5, 300],
            "visual_backup_max_triggers_5m": [1, 30],
            "sensitivity": ["high", "balanced", "low"],
            "stationary_object_tolerance": ["low", "balanced", "high"],
            "analysis_preset": ["adaptive", "modular", "classic"],
        },
    }


def _apply_pipeline_ai_change(
    next_config: AppConfig,
    camera: CameraConfig,
    change: AuditAiChange,
    value: object,
) -> None:
    target = (
        next_config.motion_qualification
        if change.scope == "global"
        else camera.motion_qualification
    )
    if change.setting == "analysis_preset":
        target.pipeline.qualification = analysis_preset_selections(str(value))
        return
    setattr(target, change.setting, value)


@app.get("/api/motion-audit")
def motion_audit(
    limit: int = 24,
    offset: int = 0,
    camera_id: str = "",
    outcome: str = "all",
    category: str = "all",
) -> dict:
    if outcome not in {"all", "object", "clear", "not_run"}:
        raise HTTPException(status_code=400, detail="invalid motion audit outcome")
    if category not in {"all", "qualification", "visual_backup"}:
        raise HTTPException(status_code=400, detail="invalid motion audit category")
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        rows, total = active_manager.events.motion_audits(
            limit=limit,
            offset=offset,
            camera_id=camera_id,
            outcome=outcome,
            category=category,
        )
        storage_dir = active_manager.storage_dir
    return {
        "items": [_motion_audit_row(row, storage_dir) for row in rows],
        "total": total,
        "limit": max(1, min(int(limit), 100)),
        "offset": max(0, int(offset)),
    }


@app.get("/api/motion-effectiveness")
def motion_effectiveness(days: float = 7.0) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_events = manager.events
    return active_events.motion_effectiveness(days=days)


@app.get("/api/motion-audit/{audit_id}/snapshot.jpg")
def motion_audit_snapshot(audit_id: int) -> FileResponse:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        audit = active_manager.events.get_motion_audit(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail="motion audit entry not found")
    try:
        snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    media_type = snapshot_media_type(snapshot_path)
    return FileResponse(
        snapshot_path,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=300"},
    )


@app.post("/api/motion-audit/{audit_id}/ai-analyze")
async def motion_audit_ai_analyze(audit_id: int) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        audit_config = active_config.audit_ai
        audit = active_manager.events.get_motion_audit(audit_id)
        if audit is None:
            raise HTTPException(status_code=404, detail="motion audit entry not found")
        if not AUDIT_AI_LIMITER.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="an AI audit request is already running",
                headers={"Retry-After": "5"},
            )
        _begin_ai_operation("motion_audit")
    try:
        snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
        analysis_context = _audit_ai_context(audit, active_config, active_manager)
        advice = await asyncio.to_thread(
            AuditAiAdvisor(audit_config).analyze,
            snapshot_path,
            analysis_context,
        )
        camera = camera_by_id(active_config, str(audit.get("camera_id") or ""))
        if camera is None:
            raise AuditAiError("audit camera is unavailable")
        changes, _previews = _assistant_motion_change_previews(
            active_config,
            camera,
            [change for change in advice.changes if change.scope == "camera"],
        )
        advice.changes = changes
        configuration_fingerprint = _assistant_motion_config_fingerprint(
            active_config,
            camera,
        )
        recommendation_proof = _issue_ai_recommendation_token(
            kind="motion_audit",
            record_id=audit_id,
            camera_id=camera.id,
            configuration_fingerprint=configuration_fingerprint,
            changes=changes,
        )
    except AuditAiError as exc:
        raise HTTPException(status_code=502, detail=redact_secret_text(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    finally:
        _end_ai_operation("motion_audit")
        AUDIT_AI_LIMITER.release()
    return {
        "audit_id": audit_id,
        "camera_id": audit.get("camera_id"),
        "provider": audit_config.provider,
        "model": audit_config.model or "",
        "motion_paradigm": analysis_context["motion_paradigm"],
        "advice": advice.model_dump(mode="json"),
        "configuration_fingerprint": configuration_fingerprint,
        "recommendation_proof": recommendation_proof,
        "apply_requires_confirmation": True,
    }


@app.post("/api/motion-audit/{audit_id}/ai-apply")
def motion_audit_ai_apply(audit_id: int, request: AuditAiApplyRequest) -> dict:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="explicit confirmation is required")
    if not request.changes:
        raise HTTPException(status_code=400, detail="no recommendation changes supplied")
    if any(change.scope != "camera" for change in request.changes):
        raise HTTPException(
            status_code=400,
            detail="single motion reviews may only change settings for the reviewed camera",
        )
    with MANAGER_RELOAD_LOCK:
        if not config.audit_ai.allow_apply_recommendations:
            raise HTTPException(status_code=403, detail="applying AI recommendations is disabled")
        audit = manager.events.get_motion_audit(audit_id)
        if audit is None:
            raise HTTPException(status_code=404, detail="motion audit entry not found")
        next_config = config.model_copy(deep=True)
        camera = camera_by_id(next_config, str(audit.get("camera_id") or ""))
        if camera is None:
            raise HTTPException(status_code=404, detail="audit camera not found")
        current_fingerprint = _assistant_motion_config_fingerprint(next_config, camera)
        if request.configuration_fingerprint != current_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="motion settings changed after this review; run AI analysis again",
            )
        if not _verify_ai_recommendation_token(
            request.recommendation_proof,
            kind="motion_audit",
            record_id=audit_id,
            camera_id=camera.id,
            configuration_fingerprint=current_fingerprint,
            changes=request.changes,
        ):
            raise HTTPException(
                status_code=409,
                detail="AI recommendations are expired or do not match this review",
            )
        applied: list[dict] = []
        try:
            for change in request.changes:
                value = validate_tuning_value(change.setting, change.value)
                _apply_pipeline_ai_change(next_config, camera, change, value)
                applied.append({**change.model_dump(mode="json"), "value": value})
            validate_motion_pipeline_configuration(next_config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        next_config = AppConfig.model_validate(next_config.model_dump(mode="json"))
        _effective_config, apply_result = apply_config_update(next_config)
    return {
        "ok": True,
        "audit_id": audit_id,
        "camera_id": camera.id,
        "applied": applied,
        "workers_restarted": bool(apply_result["camera_workers_restarted"]),
        "apply_mode": apply_result["apply_mode"],
    }


def _run_motion_ai_review(
    review_id: int,
    audits: list[dict[str, Any]],
    active_config: AppConfig,
    active_manager: AppManager,
) -> None:
    analyses: list[tuple[dict[str, Any], Any]] = []
    images_available = 0
    failures = 0
    consecutive_failures = 0
    first_error = ""
    current_settings: dict[str, Any] = {}
    review_context: dict[str, Any] = {}
    try:
        active_manager.events.update_motion_ai_review(review_id, status="running")
        advisor = AuditAiAdvisor(active_config.audit_ai)
        candidates: list[tuple[dict[str, Any], Path]] = []
        for audit in audits:
            try:
                snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
            except (FileNotFoundError, PermissionError):
                continue
            candidates.append((audit, snapshot_path))
        images_available = len(candidates)
        active_manager.events.update_motion_ai_review(
            review_id,
            status="running",
            images_available=images_available,
            analyzed=0,
            failed=0,
        )
        for audit, snapshot_path in candidates:
            try:
                context = _audit_ai_context(audit, active_config, active_manager)
                if not current_settings:
                    current_settings = dict(context.get("effective_settings") or {})
                    review_context = {
                        "motion_paradigm": context.get("motion_paradigm") or {},
                        "effective_settings": current_settings,
                    }
                context["camera_review"] = {
                    "purpose": "aggregate camera tuning review",
                    "instruction": "Recommend only camera-scoped changes supported by this image.",
                }
                advice = advisor.analyze(snapshot_path, context)
            except (AuditAiError, FileNotFoundError, PermissionError) as exc:
                failures += 1
                consecutive_failures += 1
                if not first_error:
                    first_error = redact_secret_text(exc)
                active_manager.events.update_motion_ai_review(
                    review_id,
                    status="running",
                    images_available=images_available,
                    analyzed=len(analyses),
                    failed=failures,
                )
                if consecutive_failures >= 3:
                    break
                continue
            analyses.append((audit, advice))
            consecutive_failures = 0
            active_manager.events.update_motion_ai_review(
                review_id,
                status="running",
                images_available=images_available,
                analyzed=len(analyses),
                failed=failures,
            )

        if not analyses:
            if images_available == 0:
                error = "No retained motion-audit images are available for this camera"
            else:
                error = first_error or "AI analysis did not complete for any retained image"
            active_manager.events.update_motion_ai_review(
                review_id,
                status="failed",
                images_available=images_available,
                analyzed=0,
                failed=failures,
                error=error,
            )
            return
        result = aggregate_motion_ai_review(
            analyses,
            audits_considered=len(audits),
            images_available=images_available,
            failed=failures,
            current_settings=current_settings,
            review_context=review_context,
        )
        active_manager.events.update_motion_ai_review(
            review_id,
            status="completed",
            images_available=images_available,
            analyzed=len(analyses),
            failed=failures,
            result=result,
            error=first_error if failures else "",
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("motion AI review %s failed", review_id)
        try:
            active_manager.events.update_motion_ai_review(
                review_id,
                status="failed",
                images_available=images_available,
                analyzed=len(analyses),
                failed=failures,
                error=redact_secret_text(exc),
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "failed to persist motion AI review %s failure",
                review_id,
            )
    finally:
        AUDIT_AI_LIMITER.release()


def _camera_intelligence_candidates(
    camera: CameraConfig,
    active_manager: AppManager,
    *,
    hours: float,
    record_limit: int,
    image_limit: int,
) -> tuple[list[dict[str, Any]], int]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()
    audits, _total = active_manager.events.motion_audits(
        limit=record_limit,
        camera_id=camera.id,
    )
    recent_audits = [
        audit for audit in audits
        if not audit.get("created_at") or str(audit.get("created_at")) >= cutoff_iso
    ]
    end_iso = datetime.now(timezone.utc).isoformat()
    if hasattr(active_manager.events, "recent_for_camera_range"):
        event_rows = active_manager.events.recent_for_camera_range(
            camera.id,
            cutoff_iso,
            end_iso,
            limit=max(500, record_limit * 8),
        )
    elif hasattr(active_manager.events, "for_camera_range"):
        event_rows = list(reversed(active_manager.events.for_camera_range(
            camera.id,
            cutoff_iso,
            end_iso,
            limit=max(500, record_limit * 8),
        )))
    else:
        event_rows = []
    incident_summaries = _incident_rows(
        [_event_row(row) for row in event_rows],
        DEFAULT_INCIDENT_GAP_SECONDS,
    )[:record_limit]
    incidents = _incidents_with_faces(
        _hydrate_incidents(incident_summaries, active_manager),
        active_manager,
    ) if incident_summaries else []

    candidates: list[dict[str, Any]] = []
    incident_event_ids: set[int] = set()
    for incident in incidents:
        event_id = int(incident.get("representative_event_id") or 0)
        if event_id <= 0:
            continue
        incident_event_ids.update(
            int(event.get("id") or 0)
            for event in incident.get("events") or []
        )
        candidates.append({
            "kind": "incident",
            "camera_id": camera.id,
            "record_id": event_id,
            "event_id": event_id,
            "created_at": incident.get("start_at"),
            "category": (
                "recognized_incident"
                if incident.get("has_objects") or incident.get("labels")
                else "motion_only_incident"
            ),
        })
    for audit in recent_audits:
        reason = str(audit.get("reason") or "")
        if reason in {"event_state_active", "event_state_cooldown"}:
            continue
        linked_event_id = int(audit.get("event_id") or 0)
        if linked_event_id and linked_event_id in incident_event_ids:
            continue
        if audit.get("category") == "visual_backup":
            category = "visual_backup"
        elif audit.get("object_detected") == 0:
            category = "possible_miss"
        elif not linked_event_id:
            category = "motion_filtered"
        else:
            category = "other"
        candidates.append({
            "kind": "motion_decision",
            "camera_id": camera.id,
            "record_id": int(audit.get("id") or 0),
            "audit_id": int(audit.get("id") or 0),
            "created_at": audit.get("created_at"),
            "category": category,
            "audit": audit,
        })
    candidates.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    record_pool = select_balanced_samples(candidates, record_limit)
    return select_balanced_samples(record_pool, image_limit), len(record_pool)


def _run_camera_intelligence_review(
    review_id: int,
    camera_id: str,
    samples: list[dict[str, Any]],
    records_considered: int,
    hours: float,
    active_config: AppConfig,
    active_manager: AppManager,
    evaluation_id: int = 0,
    baseline_result: dict[str, Any] | None = None,
) -> None:
    analyses: list[dict[str, Any]] = []
    failed = 0
    consecutive_failures = 0
    first_error = ""
    try:
        active_manager.events.update_motion_ai_review(
            review_id,
            status="running",
            images_available=len(samples),
            analyzed=0,
            failed=0,
        )
        audit_advisor = AuditAiAdvisor(active_config.audit_ai)
        for sample in samples:
            try:
                if sample["kind"] == "incident":
                    evidence = _assistant_visual_incident_evidence(
                        int(sample["event_id"]),
                        active_config,
                        active_manager,
                    )
                    if evidence is None:
                        raise AuditAiError("incident evidence is unavailable")
                    advice = evidence.client_data["advice"]
                    verdict = {
                        "detection_consistent": "consistent",
                        "probable_missed_detection": "likely_miss",
                        "probable_misclassification": "likely_misclassification",
                        "probable_false_positive": "likely_false_alarm",
                    }.get(str(advice.get("verdict")), "uncertain")
                    analyses.append({
                        **sample,
                        "verdict": verdict,
                        "confidence": advice.get("confidence"),
                        "summary": advice.get("summary"),
                        "visible_subjects": advice.get("visible_subjects") or [],
                        "detector_assessment": advice.get("detector_assessment"),
                        "tracking_assessment": advice.get("tracking_assessment"),
                        "changes": advice.get("changes") or [],
                        "image_url": evidence.image_url,
                    })
                    consecutive_failures = 0
                else:
                    audit = sample["audit"]
                    snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
                    context = _audit_ai_context(audit, active_config, active_manager)
                    context["camera_intelligence_review"] = {
                        "purpose": "balanced cross-incident camera review",
                        "instruction": "Recommend only bounded camera-scoped changes this image supports; SurvNG will require repeated support across images.",
                    }
                    advice = audit_advisor.analyze(snapshot_path, context)
                    has_subject = bool(advice.visible_subjects)
                    verdict = (
                        "likely_miss"
                        if advice.verdict == "real_motion"
                        and has_subject
                        and audit.get("object_detected") != 1
                        else "likely_false_alarm"
                        if advice.verdict == "noise"
                        else "consistent"
                        if advice.verdict == "real_motion"
                        else "uncertain"
                    )
                    audit_id = int(audit["id"])
                    analyses.append({
                        **sample,
                        "verdict": verdict,
                        "confidence": advice.confidence,
                        "summary": advice.summary,
                        "visible_subjects": advice.visible_subjects,
                        "detector_assessment": (
                            "missed" if verdict == "likely_miss" else "unavailable"
                        ),
                        "tracking_assessment": "unavailable",
                        "changes": [
                            change.model_dump(mode="json")
                            for change in advice.changes
                            if change.scope == "camera"
                        ],
                        "image_url": f"/api/motion-audit/{audit_id}/snapshot.jpg",
                    })
                    consecutive_failures = 0
            except (AuditAiError, FileNotFoundError, PermissionError, ValueError) as exc:
                failed += 1
                consecutive_failures += 1
                if not first_error:
                    first_error = redact_secret_text(exc)
            active_manager.events.update_motion_ai_review(
                review_id,
                status="running",
                images_available=len(samples),
                analyzed=len(analyses),
                failed=failed,
            )
            if consecutive_failures >= 3:
                first_error = first_error or "Camera review stopped after repeated analysis failures"
                break

        if not analyses:
            active_manager.events.update_motion_ai_review(
                review_id,
                status="failed",
                images_available=len(samples),
                analyzed=0,
                failed=failed,
                error=first_error or "No retained images could be reviewed",
            )
            if evaluation_id:
                active_manager.events.reset_camera_intelligence_followup(
                    evaluation_id,
                    first_error or "No retained images could be reviewed",
                )
            return
        result = aggregate_camera_intelligence(
            analyses,
            records_considered=records_considered,
            selected_images=len(samples),
            failed=failed,
            hours=hours,
        )
        camera = camera_by_id(active_config, camera_id)
        if camera is not None:
            proposed_changes = [
                AuditAiChange(
                    scope="camera",
                    setting=item["setting"],
                    value=item["value"],
                    reason=(item.get("reasons") or ["Repeated review evidence supports this change."])[0],
                )
                for item in result.get("recommendations") or []
            ]
            _normalized, previews = _assistant_motion_change_previews(
                active_config, camera, proposed_changes
            )
            preview_by_key = {
                (item["setting"], json.dumps(item["proposed"], sort_keys=True)): item
                for item in previews
            }
            recommendations = []
            for item in result.get("recommendations") or []:
                preview = preview_by_key.get((
                    item["setting"], json.dumps(item["value"], sort_keys=True)
                ))
                if preview:
                    recommendations.append({**item, **preview})
            result["recommendations"] = recommendations
            result["configuration_fingerprint"] = _assistant_motion_config_fingerprint(
                active_config, camera
            )
            result["can_apply"] = bool(
                recommendations and active_config.audit_ai.allow_apply_recommendations
            )
        if evaluation_id and baseline_result is not None:
            comparison = compare_camera_intelligence_results(
                baseline_result,
                result,
            )
            result["effectiveness_comparison"] = comparison
            active_manager.events.complete_camera_intelligence_evaluation(
                evaluation_id,
                followup_result=result,
                comparison=comparison,
            )
        active_manager.events.update_motion_ai_review(
            review_id,
            status="completed",
            images_available=len(samples),
            analyzed=len(analyses),
            failed=failed,
            result=result,
            error=first_error if failed else "",
        )
    except Exception as exc:
        LOGGER.exception("camera intelligence review %s failed", review_id)
        try:
            active_manager.events.update_motion_ai_review(
                review_id,
                status="failed",
                analyzed=len(analyses),
                failed=failed,
                error=redact_secret_text(exc),
            )
        except Exception:
            LOGGER.exception(
                "failed to persist camera intelligence review %s failure",
                review_id,
            )
        if evaluation_id:
            try:
                active_manager.events.reset_camera_intelligence_followup(
                    evaluation_id,
                    redact_secret_text(exc),
                )
            except Exception:
                LOGGER.exception(
                    "failed to reset camera intelligence evaluation %s",
                    evaluation_id,
                )
    finally:
        _end_ai_operation("camera_intelligence")
        AUDIT_AI_LIMITER.release()


@app.post("/api/motion-ai-reviews")
def start_motion_ai_review(request: MotionAiReviewRequest) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        camera = camera_by_id(active_config, request.camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="camera not found")
        if not active_config.audit_ai.enabled:
            raise HTTPException(status_code=400, detail="AI audit advisor is disabled")
        if not ai_provider_configured(active_config.audit_ai):
            raise HTTPException(status_code=400, detail="AI audit API key is not configured")
        samples, records_considered = _camera_intelligence_candidates(
            camera,
            active_manager,
            hours=request.hours,
            record_limit=request.record_limit,
            image_limit=request.image_limit,
        )
        if not samples:
            raise HTTPException(
                status_code=404,
                detail="this camera has no recent incidents or motion decisions to review",
            )
        if not AUDIT_AI_LIMITER.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="an AI audit or camera review is already running",
                headers={"Retry-After": "5"},
            )
        _begin_ai_operation("camera_intelligence")
        try:
            review = active_manager.events.create_motion_ai_review(
                camera.id,
                records_considered,
            )
        except BaseException:
            _end_ai_operation("camera_intelligence")
            AUDIT_AI_LIMITER.release()
            raise
    thread = threading.Thread(
        target=_run_camera_intelligence_review,
        args=(
            int(review["id"]),
            camera.id,
            samples,
            records_considered,
            request.hours,
            active_config,
            active_manager,
        ),
        name=f"camera-intelligence-{camera.id}",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException:
        _end_ai_operation("camera_intelligence")
        AUDIT_AI_LIMITER.release()
        active_manager.events.update_motion_ai_review(
            int(review["id"]),
            status="failed",
            error="AI review worker could not start",
        )
        raise
    return review


@app.get("/api/motion-ai-reviews/latest")
def latest_motion_ai_review(camera_id: str) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_config = config
        active_events = manager.events
        camera = camera_by_id(active_config, camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    review = active_events.latest_motion_ai_review(camera.id)
    return review or {"camera_id": camera.id, "status": "never"}


@app.get("/api/motion-ai-reviews/{review_id}")
def motion_ai_review(review_id: int) -> dict:
    with MANAGER_RELOAD_LOCK:
        review = manager.events.get_motion_ai_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="motion AI review not found")
    return review


@app.post("/api/motion-ai-reviews/{review_id}/apply")
def camera_intelligence_apply(
    review_id: int,
    request: CameraIntelligenceApplyRequest,
) -> dict:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="explicit confirmation is required")
    if not request.changes:
        raise HTTPException(status_code=400, detail="no recommendation changes supplied")
    if any(change.scope != "camera" for change in request.changes):
        raise HTTPException(
            status_code=400,
            detail="camera intelligence may only change settings for the reviewed camera",
        )
    with MANAGER_RELOAD_LOCK:
        if not config.audit_ai.allow_apply_recommendations:
            raise HTTPException(
                status_code=403,
                detail="applying AI recommendations is disabled",
            )
        review = manager.events.get_motion_ai_review(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="camera intelligence review not found")
        result = review.get("result") or {}
        if review.get("status") != "completed" or result.get("review_type") != "camera_intelligence":
            raise HTTPException(status_code=409, detail="camera intelligence review is not complete")
        next_config = config.model_copy(deep=True)
        camera = camera_by_id(next_config, str(review.get("camera_id") or ""))
        if camera is None:
            raise HTTPException(status_code=404, detail="reviewed camera not found")
        current_fingerprint = _assistant_motion_config_fingerprint(next_config, camera)
        if request.configuration_fingerprint != current_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="motion settings changed after this review; run camera intelligence again",
            )

        persisted: dict[tuple[str, str], dict[str, Any]] = {}
        for recommendation in result.get("recommendations") or []:
            setting = str(recommendation.get("setting") or "")
            value = recommendation.get("proposed", recommendation.get("value"))
            persisted[(setting, json.dumps(value, sort_keys=True))] = recommendation
        approved_changes: list[AuditAiChange] = []
        for submitted in request.changes:
            try:
                normalized_value = validate_tuning_value(
                    submitted.setting,
                    submitted.value,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            recommendation = persisted.get((
                submitted.setting,
                json.dumps(normalized_value, sort_keys=True),
            ))
            if recommendation is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"{submitted.setting} is not an unchanged recommendation from this review",
                )
            approved_changes.append(AuditAiChange(
                scope="camera",
                setting=submitted.setting,
                value=normalized_value,
                reason=str((recommendation.get("reasons") or [recommendation.get("reason") or "Repeated evidence supports this change."])[0]),
            ))
        try:
            changes, previews = _assistant_motion_change_previews(
                next_config,
                camera,
                approved_changes,
            )
            if not changes:
                raise ValueError("recommendations do not change active settings")
            for change in changes:
                _apply_pipeline_ai_change(next_config, camera, change, change.value)
            validate_motion_pipeline_configuration(next_config)
            next_config = AppConfig.model_validate(next_config.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        active_events = manager.events
        _effective_config, apply_result = apply_config_update(next_config)
        evaluation = active_events.create_camera_intelligence_evaluation(
            camera_id=camera.id,
            baseline_review_id=review_id,
            evaluation_hours=request.evaluation_hours,
            applied_changes=previews,
            baseline_result=result,
        )
    return {
        "ok": True,
        "review_id": review_id,
        "camera_id": camera.id,
        "applied": previews,
        "workers_restarted": bool(apply_result["camera_workers_restarted"]),
        "apply_mode": apply_result["apply_mode"],
        "effectiveness_evaluation": evaluation,
    }


@app.get("/api/camera-intelligence/evaluations/latest")
def latest_camera_intelligence_evaluation(camera_id: str) -> dict:
    with MANAGER_RELOAD_LOCK:
        camera = camera_by_id(config, camera_id)
        active_events = manager.events
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    evaluation = active_events.latest_camera_intelligence_evaluation(camera.id)
    return evaluation or {"camera_id": camera.id, "status": "never"}


@app.post("/api/camera-intelligence/evaluations/{evaluation_id}/follow-up")
def start_camera_intelligence_followup(
    evaluation_id: int,
    request: CameraIntelligenceFollowupRequest,
) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        evaluation = active_manager.events.get_camera_intelligence_evaluation(
            evaluation_id
        )
        if evaluation is None:
            raise HTTPException(status_code=404, detail="effectiveness evaluation not found")
        if evaluation.get("status") != "ready":
            detail = (
                f"follow-up evidence is still being collected until {evaluation.get('ready_at')}"
                if evaluation.get("status") == "collecting"
                else "effectiveness follow-up is already running or complete"
            )
            raise HTTPException(status_code=409, detail=detail)
        camera = camera_by_id(active_config, str(evaluation.get("camera_id") or ""))
        if camera is None:
            raise HTTPException(status_code=404, detail="reviewed camera not found")
        if (
            not active_config.audit_ai.enabled
            or not ai_provider_configured(active_config.audit_ai)
        ):
            raise HTTPException(status_code=400, detail="AI analysis is not configured")
        try:
            applied_at = datetime.fromisoformat(
                str(evaluation["applied_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="evaluation start time is invalid") from exc
        if applied_at.tzinfo is None:
            applied_at = applied_at.replace(tzinfo=timezone.utc)
        elapsed_hours = max(
            1.0,
            (datetime.now(timezone.utc) - applied_at).total_seconds() / 3600.0,
        )
        samples, records_considered = _camera_intelligence_candidates(
            camera,
            active_manager,
            hours=min(168.0, elapsed_hours),
            record_limit=100,
            image_limit=request.image_limit,
        )
        if not samples:
            raise HTTPException(status_code=404, detail="no follow-up camera images are available")
        if not AUDIT_AI_LIMITER.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="another AI camera review is already running",
                headers={"Retry-After": "5"},
            )
        _begin_ai_operation("camera_intelligence")
        try:
            review = active_manager.events.create_motion_ai_review(
                camera.id,
                records_considered,
            )
            active_manager.events.start_camera_intelligence_followup(
                evaluation_id,
                int(review["id"]),
            )
        except BaseException:
            _end_ai_operation("camera_intelligence")
            AUDIT_AI_LIMITER.release()
            raise
    thread = threading.Thread(
        target=_run_camera_intelligence_review,
        args=(
            int(review["id"]),
            camera.id,
            samples,
            records_considered,
            min(168.0, elapsed_hours),
            active_config,
            active_manager,
            evaluation_id,
            evaluation.get("baseline_result") or {},
        ),
        name=f"camera-effectiveness-{camera.id}",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException:
        _end_ai_operation("camera_intelligence")
        AUDIT_AI_LIMITER.release()
        active_manager.events.update_motion_ai_review(
            int(review["id"]),
            status="failed",
            error="Effectiveness review worker could not start",
        )
        active_manager.events.reset_camera_intelligence_followup(
            evaluation_id,
            "Effectiveness review worker could not start",
        )
        raise
    return active_manager.events.get_camera_intelligence_evaluation(evaluation_id) or {}


@app.get("/api/events/{event_id}/snapshot.jpg")
def event_snapshot(event_id: int, download: bool = False) -> FileResponse:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        event = active_manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    try:
        snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    media_type = snapshot_media_type(snapshot_path)
    return FileResponse(
        snapshot_path,
        media_type=media_type,
        filename=snapshot_path.name if download else None,
        headers={"Cache-Control": "private, max-age=3600"},
    )


def _jpeg_thumbnail(frame: np.ndarray, width: int, quality: int) -> bytes:
    frame_height, frame_width = frame.shape[:2]
    if frame_width > width:
        target_height = max(1, round(frame_height * width / frame_width))
        frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode thumbnail")
    return encoded.tobytes()


@app.get("/api/events/{event_id}/thumbnail.jpg")
def event_thumbnail(event_id: int, width: int = 640, quality: int = 82) -> FileResponse:
    safe_width = max(160, min(int(width), 1280))
    safe_quality = max(50, min(int(quality), 92))
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        event = active_manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    try:
        snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
        stat = snapshot_path.stat()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    identity = f"{snapshot_path}:{stat.st_size}:{stat.st_mtime_ns}:{safe_width}:{safe_quality}"

    def build() -> bytes:
        frame = cv2.imread(str(snapshot_path))
        if frame is None:
            raise HTTPException(status_code=404, detail="snapshot is unavailable")
        return _jpeg_thumbnail(frame, safe_width, safe_quality)

    cached = active_manager.image_cache.get_or_create("events", identity, build)
    return FileResponse(
        cached,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


@app.get("/api/events/{event_id}/appearance-matches")
def event_appearance_matches(
    event_id: int,
    hours: float = 24.0,
    limit: int = 12,
    cross_camera_only: bool = True,
) -> dict[str, Any]:
    bounded_hours = max(0.25, min(float(hours), 24.0 * 30.0))
    bounded_limit = max(1, min(int(limit), 100))
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        try:
            anchor_at = datetime.fromisoformat(
                str(event.get("created_at") or "").replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="event timestamp is invalid") from exc
        if anchor_at.tzinfo is None:
            anchor_at = anchor_at.replace(tzinfo=timezone.utc)
        matches = active_manager.appearance_index.matches(
            event_id,
            start_at=(anchor_at - timedelta(hours=bounded_hours)).isoformat(),
            end_at=(anchor_at + timedelta(hours=bounded_hours)).isoformat(),
            cross_camera_only=bool(cross_camera_only),
            limit=bounded_limit,
        )
    return {
        "event_id": event_id,
        "hours": bounded_hours,
        "cross_camera_only": bool(cross_camera_only),
        "matches": matches,
    }


@app.get("/api/appearance-index/status")
def appearance_index_status() -> dict[str, Any]:
    with MANAGER_RELOAD_LOCK:
        return manager.appearance_index.status()


@app.get("/api/incidents")
def incidents(limit: int = 200, gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS) -> list[dict]:
    bounded_limit = max(1, min(limit, 200))
    bounded_gap = max(5, min(gap_seconds, 300))
    summaries = _recent_incident_summaries(bounded_limit, bounded_gap)
    return _incidents_with_faces(_hydrate_incidents(summaries))


@app.get("/api/incidents/feed")
def incident_feed(
    event_type: str = "object",
    camera_id: str = "",
    object_label: str = "",
    zone: str = "",
    limit: int = 18,
    offset: int = 0,
    gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
) -> dict:
    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, min(offset, 100_000))
    bounded_gap = max(5, min(gap_seconds, 300))
    page, has_more, scanned = _recent_filtered_incident_summaries(
        limit=bounded_limit,
        offset=bounded_offset,
        gap_seconds=bounded_gap,
        event_type=event_type,
        camera_id=camera_id,
        object_label=object_label,
        zone=zone,
    )
    facets = {
        "camera_ids": sorted({str(item.get("camera_id") or "") for item in scanned if item.get("camera_id")}),
        "labels": sorted({str(label) for item in scanned for label in item.get("labels", []) if label}),
        "zones": sorted({str(item_zone) for item in scanned for item_zone in item.get("zones", []) if item_zone}),
    }
    return {
        "items": [_incident_list_payload(item) for item in page],
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": has_more,
        "facets": facets,
    }


@app.get("/api/incidents/detail")
def incident_detail(event_ids: str, gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS) -> dict:
    try:
        requested_ids = list(dict.fromkeys(
            int(value.strip())
            for value in event_ids.split(",")
            if value.strip()
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="event_ids must be comma-separated integers") from exc
    if not requested_ids or len(requested_ids) > 200 or any(event_id <= 0 for event_id in requested_ids):
        raise HTTPException(status_code=422, detail="event_ids must contain 1 to 200 positive integers")

    rows = manager.events.get_many(requested_ids)
    if {int(row["id"]) for row in rows} != set(requested_ids):
        raise HTTPException(status_code=404, detail="incident events were not found")
    bounded_gap = max(5, min(gap_seconds, 300))
    summaries = _incident_rows([_event_row(row) for row in rows], bounded_gap)
    if len(summaries) != 1:
        raise HTTPException(status_code=422, detail="event_ids do not identify one incident")
    hydrated = _incidents_with_faces(_hydrate_incidents(summaries))
    if not hydrated:
        raise HTTPException(status_code=404, detail="incident was not found")
    return hydrated[0]


def _filter_incidents_by_event_type(incidents: list[dict], event_type: str) -> list[dict]:
    if event_type == "object":
        return [item for item in incidents if item.get("has_objects")]
    if event_type == "motion":
        return [item for item in incidents if not item.get("has_objects")]
    return incidents


def _filter_incident_summaries(
    summaries: list[dict],
    event_type: str,
    camera_id: str = "",
    object_label: str = "",
    zone: str = "",
) -> list[dict]:
    filtered = _filter_incidents_by_event_type(summaries, event_type)
    if camera_id:
        filtered = [item for item in filtered if item.get("camera_id") == camera_id]
    if object_label:
        filtered = [item for item in filtered if object_label in item.get("labels", [])]
    if zone:
        filtered = [item for item in filtered if zone in item.get("zones", [])]
    return filtered


@app.get("/api/incidents/search")
def incident_search(
    day: str = "",
    time_zone: str = "America/New_York",
    camera_id: str = "",
    event_type: str = "motion",
    object_label: str = "",
    zone: str = "",
    limit: int = 18,
    offset: int = 0,
    gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS,
) -> dict:
    try:
        selected_zone = ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="unknown timezone") from exc
    if day:
        try:
            selected_date = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="day must use YYYY-MM-DD") from exc
    else:
        selected_date = datetime.now(selected_zone).date()
        day = selected_date.isoformat()
    day_start = datetime.combine(selected_date, datetime.min.time(), selected_zone)
    day_end = day_start + timedelta(days=1)
    bounded_gap = max(5, min(gap_seconds, 300))
    query_start = day_start.astimezone(timezone.utc) - timedelta(seconds=bounded_gap)
    query_end = day_end.astimezone(timezone.utc) + timedelta(seconds=bounded_gap)
    compact_rows = [
        _event_row(row)
        for row in manager.events.between_compact(query_start.isoformat(), query_end.isoformat())
    ]
    day_start_epoch = day_start.timestamp()
    day_end_epoch = day_end.timestamp()
    day_incidents = [
        incident
        for incident in _incident_rows(compact_rows, gap_seconds=bounded_gap)
        if incident["last_epoch"] >= day_start_epoch and incident["start_epoch"] < day_end_epoch
    ]
    facets = {
        "camera_ids": sorted({str(item.get("camera_id") or "") for item in day_incidents if item.get("camera_id")}),
        "labels": sorted({str(label) for item in day_incidents for label in item.get("labels", []) if label}),
        "zones": sorted({str(item_zone) for item in day_incidents for item_zone in item.get("zones", []) if item_zone}),
    }
    filtered = _filter_incidents_by_event_type(day_incidents, event_type)
    if camera_id:
        filtered = [item for item in filtered if item.get("camera_id") == camera_id]
    if object_label:
        filtered = [item for item in filtered if object_label in item.get("labels", [])]
    if zone:
        filtered = [item for item in filtered if zone in item.get("zones", [])]
    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, offset)
    page_summaries = filtered[bounded_offset:bounded_offset + bounded_limit]
    return {
        "items": [_incident_list_payload(item) for item in page_summaries],
        "total": len(filtered),
        "limit": bounded_limit,
        "offset": bounded_offset,
        "day": day,
        "time_zone": time_zone,
        "start_at": day_start.astimezone(timezone.utc).isoformat(),
        "end_at": day_end.astimezone(timezone.utc).isoformat(),
        "facets": facets,
    }


def _assistant_catalog(active_config: AppConfig, active_manager: AppManager) -> dict[str, Any]:
    face_store = getattr(active_manager, "faces", None)
    try:
        people = face_store.people() if face_store is not None else []
    except Exception:
        LOGGER.warning("assistant catalog could not read recognized face names")
        people = []
    return {
        "cameras": [
            {"id": camera.id, "name": camera.name}
            for camera in active_config.cameras
        ],
        "object_labels": list(active_manager.detector.labels),
        "zones": sorted({
            zone.name
            for camera in active_config.cameras
            for zone in camera.zones
            if zone.enabled
        }),
        "recognized_faces": [
            {
                "id": int(person.get("id") or 0),
                "name": str(person.get("name") or "")[:128],
            }
            for person in people[:200]
            if str(person.get("name") or "").strip()
        ],
    }


def _assistant_system_evidence(active_manager: AppManager) -> AssistantEvidence:
    status = system_status()
    cameras = active_manager.statuses()
    unhealthy = [
        {
            "camera_id": str(camera.get("id") or ""),
            "running": bool(camera.get("running")),
            "connected": bool(camera.get("connected")),
            "frame_fresh": bool(camera.get("frame_fresh")),
            "recording": bool(camera.get("recording") or camera.get("sub_recording")),
            "last_frame_age_seconds": camera.get("last_frame_age_seconds"),
            "last_error": str(camera.get("last_error") or "")[:500],
        }
        for camera in cameras
        if not camera.get("running")
        or not camera.get("frame_fresh")
        or not (camera.get("recording") or camera.get("sub_recording"))
    ]
    detector = status.get("detector") or {}
    runtime = detector.get("runtime") or {}
    retention = active_manager.recorder.retention_status()
    retention_plan = retention.get("plan") or {}
    retention_reclaim = retention_plan.get("reclaim") or {}
    last_retention_run = retention.get("last_run") or {}
    payload = {
        "cameras": status.get("cameras"),
        "unhealthy_cameras": unhealthy,
        "storage": status.get("storage"),
        "detector": {
            "enabled": detector.get("enabled"),
            "backend": detector.get("loaded_backend"),
            "device": detector.get("loaded_device"),
            "ready": bool(detector.get("openvino_loaded") or detector.get("coreml_loaded")),
            "average_inference_ms": runtime.get("average_inference_ms"),
            "queue_depth": runtime.get("queue_depth"),
            "failed_inferences": runtime.get("failed_inferences"),
            "active_workers": runtime.get("active_workers"),
            "configured_workers": runtime.get("configured_workers"),
            "reid_ready": bool((detector.get("reid") or {}).get("loaded")),
        },
        "mqtt": {
            key: (status.get("mqtt") or {}).get(key)
            for key in (
                "enabled", "connected", "last_error", "publish_failures",
                "pending_incidents", "server_lifecycle",
            )
        },
        "go2rtc": status.get("go2rtc"),
        "retention": {
            "state": retention.get("state"),
            "enabled": retention.get("enabled"),
            "automatic_cleanup": retention.get("automatic_cleanup"),
            "last_plan_at": retention.get("last_plan_at"),
            "last_run_at": retention.get("last_run_at"),
            "error": retention.get("error"),
            "planned_reclaim_bytes": retention_reclaim.get("planned_bytes"),
            "last_deleted_files": last_retention_run.get("deleted_files"),
            "last_deleted_bytes": last_retention_run.get("deleted_bytes"),
        },
    }
    total = int((status.get("cameras") or {}).get("total") or 0)
    online = int((status.get("cameras") or {}).get("online") or 0)
    return AssistantEvidence(
        evidence_id="E-system",
        kind="system_health",
        title="Current SurvNG health",
        summary=f"{online}/{total} cameras online; {len(unhealthy)} cameras need attention.",
        data=payload,
        href="/config#telemetry",
    )


def _assistant_camera_evidence(
    active_manager: AppManager,
    camera_id: str,
) -> list[AssistantEvidence]:
    requested = camera_id.strip().lower()
    evidence: list[AssistantEvidence] = []
    for camera in active_manager.statuses():
        current_id = str(camera.get("id") or "")
        if requested and current_id.lower() != requested:
            continue
        motion = camera.get("motion_qualification") or {}
        tracking = camera.get("object_tracking") or {}
        data = {
            "camera_id": current_id,
            "name": camera.get("name") or current_id,
            "running": bool(camera.get("running")),
            "connected": bool(camera.get("connected")),
            "frame_fresh": bool(camera.get("frame_fresh")),
            "last_frame_age_seconds": camera.get("last_frame_age_seconds"),
            "recording": bool(camera.get("recording")),
            "sub_recording": bool(camera.get("sub_recording")),
            "detection_enabled": bool(camera.get("detection_enabled")),
            "onvif": {
                "enabled": bool(camera.get("onvif_enabled")),
                "connected": bool(camera.get("onvif_connected")),
                "notifications": int(camera.get("onvif_notifications_received") or 0),
                "motion_events": int(camera.get("onvif_motion_events_received") or 0),
                "poll_errors": int(camera.get("onvif_poll_errors") or 0),
                "poll_timeouts": int(camera.get("onvif_poll_timeouts") or 0),
                "last_motion_at": camera.get("onvif_last_motion_event_at"),
                "last_error": str(camera.get("onvif_last_error") or "")[:500],
            },
            "motion": {
                key: motion.get(key)
                for key in (
                    "mode", "sensitivity", "triggers", "passed", "audit_rejected",
                    "suppressed", "dropped_triggers", "queue_depth",
                    "visual_backup_triggers", "visual_backup_not_ready",
                    "visual_backup_uncorrelated_objects",
                )
            },
            "tracking": {
                key: tracking.get(key)
                for key in (
                    "active", "frames_processed", "track_count", "reid_attempts",
                    "reid_successes", "reid_failures", "coverage_incomplete",
                )
            },
        }
        healthy = data["running"] and data["frame_fresh"]
        evidence.append(AssistantEvidence(
            evidence_id=f"E-camera-{current_id}",
            kind="camera_health",
            title=str(data["name"]),
            summary=(
                f"{current_id} is {'healthy' if healthy else 'not healthy'}; "
                f"recording is {'active' if data['recording'] or data['sub_recording'] else 'inactive'}."
            ),
            data=data,
            href=f"/?camera={quote(current_id, safe='')}",
        ))
    return evidence


def _assistant_configuration_evidence(active_config: AppConfig) -> AssistantEvidence:
    assistant_provider = AssistantProvider(active_config.audit_ai)
    data = {
        "ai": {
            "enabled": active_config.audit_ai.enabled,
            "assistant_enabled": active_config.audit_ai.assistant_enabled,
            "provider": active_config.audit_ai.provider,
            "analysis_and_fast_model": assistant_provider.model_for_tier("fast"),
            "deep_reasoning_model": assistant_provider.model_for_tier("deep"),
            "deep_reasoning_uses_separate_model": (
                assistant_provider.model_for_tier("deep")
                != assistant_provider.model_for_tier("fast")
            ),
            "assistant_read_only": False,
            "supported_actions": ["create_media_export"],
        },
        "recording": {
            "segment_seconds": active_config.recording_segment_seconds,
            "cache_max_gb": active_config.recording_cache_max_gb,
            "cache_max_days": active_config.recording_cache_max_days,
            "prewarm": active_config.recording_cache_prewarm,
            "retention": active_config.retention.model_dump(mode="json"),
        },
        "motion": active_config.motion_qualification.model_dump(mode="json"),
        "detector": {
            "enabled": active_config.detector.enabled,
            "backend": active_config.detector.backend,
            "device": active_config.detector.device,
            "confidence_threshold": active_config.detector.confidence_threshold,
            "nms_threshold": active_config.detector.nms_threshold,
            "event_confirmation_frames": active_config.detector.event_confirmation_frames,
            "event_class_confirmation_frames": active_config.detector.event_class_confirmation_frames,
            "event_class_confidence_thresholds": active_config.detector.event_class_confidence_thresholds,
            "zone_only_incident_eligibility": active_config.detector.require_incident_zone,
            "tracking": active_config.detector.tracking.model_dump(mode="json"),
        },
        "mqtt": {
            "enabled": active_config.mqtt.enabled,
            "tls": active_config.mqtt.tls,
            "discovery_enabled": active_config.mqtt.discovery_enabled,
            "incident_events_enabled": active_config.mqtt.incident_events_enabled,
            "server_status_enabled": active_config.mqtt.server_status_enabled,
        },
        "cameras": [
            {
                "id": camera.id,
                "name": camera.name,
                "record": camera.record,
                "record_sub": camera.record_sub,
                "retention": camera.retention.model_dump(mode="json"),
                "zone_only_incident_eligibility": camera.require_incident_zone,
                "motion": camera.motion_qualification.model_dump(mode="json"),
                "onvif_enabled": camera.onvif.enabled,
                "zone_names": [zone.name for zone in camera.zones if zone.enabled],
            }
            for camera in active_config.cameras
        ],
    }
    return AssistantEvidence(
        evidence_id="E-config",
        kind="configuration",
        title="Active safe configuration",
        summary=f"Credential-free configuration for {len(active_config.cameras)} cameras.",
        data=data,
        href="/config",
    )


def _assistant_event_objects(event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    qualification: dict[str, Any] = {}
    tracking: dict[str, Any] = {}
    for item in event.get("objects") or []:
        if not isinstance(item, dict):
            continue
        if item.get("label"):
            objects.append({
                key: item.get(key)
                for key in (
                    "label", "confidence", "incident_eligible", "zones", "box",
                    "temporal_observations", "temporal_required_observations",
                    "temporal_center_displacement_ratio", "motion_correlated",
                    "motion_correlation", "track_id", "track_state",
                )
            })
        elif isinstance(item.get("motion_qualification"), dict):
            qualification = item["motion_qualification"]
        elif isinstance(item.get("object_tracking"), dict):
            tracking = item["object_tracking"]
    return objects, qualification, tracking


def _assistant_incident_payload(incident: dict[str, Any]) -> dict[str, Any]:
    events = []
    for event in incident.get("events") or []:
        objects, qualification, tracking = _assistant_event_objects(event)
        events.append({
            "id": event.get("id"),
            "created_at": event.get("created_at"),
            "kind": event.get("kind"),
            "topic": event.get("topic"),
            "trigger_source": event.get("trigger_source"),
            "objects": objects,
            "motion_qualification": qualification,
            "object_tracking": tracking,
            "recording_available": bool(event.get("recording_path")),
            "faces": event.get("faces") or [],
        })
    return {
        "incident_id": incident.get("id"),
        "representative_event_id": incident.get("representative_event_id"),
        "camera_id": incident.get("camera_id"),
        "start_at": incident.get("start_at"),
        "end_at": incident.get("end_at"),
        "duration_seconds": incident.get("duration_seconds"),
        "event_count": incident.get("event_count"),
        "trigger_source": incident.get("trigger_source"),
        "labels": incident.get("labels") or [],
        "zones": incident.get("zones") or [],
        "motion_observations": [
            {
                key: observation.get(key)
                for key in (
                    "id", "created_at", "category", "reason", "score", "threshold",
                    "object_detected", "trigger_count", "interpretation",
                )
            }
            for observation in incident.get("motion_observations") or []
            if isinstance(observation, dict)
        ],
        "events": events,
    }


def _assistant_incident_for_event(
    event_id: int,
    active_manager: AppManager,
) -> dict[str, Any] | None:
    row = active_manager.events.get(event_id)
    if row is None:
        return None
    try:
        anchor = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    # Fetch enough context to include delayed discoveries and a longer active
    # incident. Grouping still enforces the normal incident gap.
    start = (anchor - timedelta(minutes=15)).isoformat()
    end = (anchor + timedelta(minutes=15)).isoformat()
    rows = [
        _event_row(candidate)
        for candidate in active_manager.events.for_camera_range(
            str(row.get("camera_id") or ""),
            start,
            end,
            limit=2000,
        )
    ]
    for summary in _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS):
        if any(int(event.get("id") or 0) == event_id for event in summary.get("events") or []):
            hydrated = _incidents_with_faces(
                _hydrate_incidents([summary], active_manager),
                active_manager,
            )
            return hydrated[0] if hydrated else summary
    return None


def _assistant_incident_evidence(
    incident: dict[str, Any],
    evidence_event_id: int | None = None,
) -> AssistantEvidence:
    payload = _assistant_incident_payload(incident)
    event_ids = [str(event.get("id")) for event in incident.get("events") or [] if event.get("id")]
    query = quote(",".join(event_ids), safe=",")
    event_id = evidence_event_id or int(
        incident.get("representative_event_id")
        or (event_ids[0] if event_ids else 0)
    )
    image_event_id = int(
        incident.get("representative_event_id")
        or event_id
    )
    return AssistantEvidence(
        evidence_id=f"E-incident-{event_id}",
        kind="incident",
        title=f"{incident.get('camera_id')} · {incident.get('start_at')}",
        summary=(
            f"{len(payload['events'])} event(s); labels: "
            f"{', '.join(payload['labels']) if payload['labels'] else 'motion only'}."
        ),
        data=payload,
        href=f"/incidents?event_ids={query}",
        image_url=(
            f"/api/events/{image_event_id}/thumbnail.jpg?width=960&quality=82"
            if image_event_id > 0
            else ""
        ),
    )


def _assistant_inspect_incident(
    event_id: int,
    active_manager: AppManager,
) -> AssistantEvidence | None:
    incident = _assistant_incident_for_event(event_id, active_manager)
    if incident is None:
        return None
    return _assistant_incident_evidence(incident, event_id)


def _assistant_motion_change_current_value(
    active_config: AppConfig,
    camera: CameraConfig,
    change: AuditAiChange,
) -> object:
    if change.setting == "analysis_preset":
        if change.scope == "global":
            return identify_analysis_preset(
                active_config.motion_qualification.pipeline.qualification
            )
        graphs = resolve_motion_pipeline_graphs(
            active_config.motion_qualification,
            camera.motion_qualification,
        )
        return identify_analysis_preset(graphs.qualification)
    if change.scope == "global":
        return getattr(active_config.motion_qualification, change.setting)
    override = getattr(camera.motion_qualification, change.setting)
    if override is None or override == "inherit":
        return getattr(active_config.motion_qualification, change.setting)
    return override


def _assistant_motion_change_previews(
    active_config: AppConfig,
    camera: CameraConfig,
    changes: list[AuditAiChange],
) -> tuple[list[AuditAiChange], list[dict[str, Any]]]:
    unique: list[AuditAiChange] = []
    previews: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for change in changes:
        key = (change.scope, change.setting)
        if key in seen:
            continue
        seen.add(key)
        proposed = validate_tuning_value(change.setting, change.value)
        current = _assistant_motion_change_current_value(
            active_config, camera, change
        )
        if current == proposed:
            continue
        normalized = change.model_copy(update={"value": proposed})
        unique.append(normalized)
        previews.append({
            "scope": change.scope,
            "setting": change.setting,
            "current": current,
            "proposed": proposed,
            "reason": change.reason,
        })
    return unique, previews


def _assistant_motion_config_fingerprint(
    active_config: AppConfig,
    camera: CameraConfig,
) -> str:
    payload = {
        "global": active_config.motion_qualification.model_dump(mode="json"),
        "camera": camera.motion_qualification.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ai_recommendation_payload(
    *,
    kind: str,
    record_id: int,
    camera_id: str,
    configuration_fingerprint: str,
    changes: list[AuditAiChange],
    issued_at: int,
) -> bytes:
    normalized = [
        {
            "scope": change.scope,
            "setting": change.setting,
            "value": validate_tuning_value(change.setting, change.value),
            "reason": change.reason,
        }
        for change in changes
    ]
    payload = {
        "version": 1,
        "kind": kind,
        "record_id": int(record_id),
        "camera_id": camera_id,
        "configuration_fingerprint": configuration_fingerprint,
        "changes": normalized,
        "issued_at": int(issued_at),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _issue_ai_recommendation_token(
    *,
    kind: str,
    record_id: int,
    camera_id: str,
    configuration_fingerprint: str,
    changes: list[AuditAiChange],
) -> str:
    issued_at = int(time.time())
    signature = hmac.new(
        AI_RECOMMENDATION_SECRET,
        _ai_recommendation_payload(
            kind=kind,
            record_id=record_id,
            camera_id=camera_id,
            configuration_fingerprint=configuration_fingerprint,
            changes=changes,
            issued_at=issued_at,
        ),
        hashlib.sha256,
    ).hexdigest()
    return f"v1.{issued_at}.{signature}"


def _verify_ai_recommendation_token(
    token: str,
    *,
    kind: str,
    record_id: int,
    camera_id: str,
    configuration_fingerprint: str,
    changes: list[AuditAiChange],
) -> bool:
    try:
        version, raw_issued_at, supplied_signature = token.split(".", 2)
        issued_at = int(raw_issued_at)
    except (AttributeError, TypeError, ValueError):
        return False
    now = int(time.time())
    if (
        version != "v1"
        or issued_at > now + 30
        or now - issued_at > AI_RECOMMENDATION_MAX_AGE_SECONDS
    ):
        return False
    expected = hmac.new(
        AI_RECOMMENDATION_SECRET,
        _ai_recommendation_payload(
            kind=kind,
            record_id=record_id,
            camera_id=camera_id,
            configuration_fingerprint=configuration_fingerprint,
            changes=changes,
            issued_at=issued_at,
        ),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected)


def _assistant_visual_incident_evidence(
    event_id: int,
    active_config: AppConfig,
    active_manager: AppManager,
) -> AssistantEvidence | None:
    incident = _assistant_incident_for_event(event_id, active_manager)
    if incident is None:
        return None
    camera = camera_by_id(active_config, str(incident.get("camera_id") or ""))
    if camera is None:
        return None
    candidate_ids = list(dict.fromkeys([
        int(incident.get("representative_event_id") or 0),
        *[
            int(event.get("id") or 0)
            for event in incident.get("events") or []
        ],
    ]))
    snapshot_path = None
    source_event_id = 0
    for candidate_id in candidate_ids:
        if candidate_id <= 0:
            continue
        raw_event = active_manager.events.get(candidate_id)
        if raw_event is None:
            continue
        try:
            snapshot_path = event_snapshot_path(active_manager.storage_dir, raw_event)
            source_event_id = candidate_id
            break
        except (FileNotFoundError, PermissionError):
            continue
    if snapshot_path is None:
        raise AuditAiError("incident has no retained image available for visual review")

    config_evidence = _assistant_configuration_evidence(active_config).data
    camera_configuration = next(
        (
            item for item in config_evidence.get("cameras", [])
            if item.get("id") == camera.id
        ),
        {},
    )
    camera_evidence = _assistant_camera_evidence(active_manager, camera.id)
    context = {
        "incident": _assistant_incident_payload(incident),
        "camera_health": camera_evidence[0].data if camera_evidence else {},
        "active_configuration": {
            "global_motion": config_evidence.get("motion") or {},
            "detector": config_evidence.get("detector") or {},
            "camera": camera_configuration,
        },
        "image_source_event_id": source_event_id,
        "limitations": [
            "The image is one representative moment, not the full recording.",
            "Tracking begins after the initial object decision.",
            "Only bounded motion settings may be proposed from this review.",
        ],
    }
    advice = IncidentVisualReviewer(active_config.audit_ai).review(
        snapshot_path,
        context,
    )
    changes, previews = _assistant_motion_change_previews(
        active_config,
        camera,
        [change for change in advice.changes if change.scope == "camera"],
    )
    advice_payload = advice.model_dump(mode="json")
    advice_payload["changes"] = [change.model_dump(mode="json") for change in changes]
    configuration_fingerprint = _assistant_motion_config_fingerprint(
        active_config,
        camera,
    )
    details = {
        "event_id": event_id,
        "source_event_id": source_event_id,
        "camera_id": camera.id,
        "advice": advice_payload,
        "proposals": previews,
        "can_apply": bool(
            previews and active_config.audit_ai.allow_apply_recommendations
        ),
        "apply_requires_confirmation": True,
        "configuration_fingerprint": configuration_fingerprint,
        "recommendation_proof": _issue_ai_recommendation_token(
            kind="incident_visual",
            record_id=event_id,
            camera_id=camera.id,
            configuration_fingerprint=configuration_fingerprint,
            changes=changes,
        ),
    }
    incident_evidence = _assistant_incident_evidence(incident, event_id)
    return AssistantEvidence(
        evidence_id=f"E-visual-{event_id}",
        kind="incident_visual_review",
        title=f"Visual review · {camera.name}",
        summary=(
            f"{advice.verdict.replace('_', ' ')} "
            f"({round(advice.confidence * 100)}% confidence); "
            f"{len(previews)} bounded setting proposal(s)."
        ),
        data={
            **{
                key: value
                for key, value in details.items()
                if key != "recommendation_proof"
            },
            "incident_evidence": incident_evidence.data,
        },
        href=incident_evidence.href,
        image_url=f"/api/events/{source_event_id}/thumbnail.jpg?width=960&quality=82",
        client_data=details,
    )


def _assistant_parse_datetime(value: str, selected_zone: ZoneInfo) -> datetime | None:
    if not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=selected_zone)
    return parsed.astimezone(timezone.utc)


def _assistant_media_export_evidence(
    call: AssistantToolCall,
    request: AssistantChatRequest,
    active_config: AppConfig,
) -> AssistantEvidence:
    try:
        selected_zone = ZoneInfo(request.context.time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        selected_zone = ZoneInfo("America/New_York")
    export_kind = call.export_kind
    camera_id = call.camera_id or request.context.camera_id
    camera = camera_by_id(active_config, camera_id) if camera_id else None
    start = _assistant_parse_datetime(call.start_at, selected_zone)
    end = _assistant_parse_datetime(call.end_at, selected_zone)
    questions: list[str] = []
    suggestions: list[str] = []
    if not export_kind:
        questions.append("Should this be a normal video clip or a timelapse?")
        suggestions.extend(["Make it a timelapse", "Make it a normal video clip"])
    if camera is None:
        if camera_id:
            questions.append(f"I couldn't find a camera named {camera_id}. Which camera should I use?")
        else:
            questions.append("Which camera should I use?")
        suggestions.extend(camera.name for camera in active_config.cameras[:4])
    if start is None and end is None:
        questions.append("What date and start/end times should the export cover?")
    elif start is None:
        questions.append("What time should the export start?")
    elif end is None:
        questions.append("What time should the export end?")
    if questions:
        question = " ".join(questions)
        return AssistantEvidence(
            evidence_id="E-export-clarification",
            kind="media_export_clarification",
            title="More export details needed",
            summary=question,
            data={"questions": questions, "suggestions": list(dict.fromkeys(suggestions))[:4]},
            href="/recordings",
            client_data={"questions": questions},
        )

    assert camera is not None and start is not None and end is not None
    if end <= start:
        return AssistantEvidence(
            evidence_id="E-export-clarification",
            kind="media_export_clarification",
            title="Export times need correction",
            summary="The end time must be after the start time. What start and end times should I use?",
            data={
                "questions": ["What corrected start and end times should I use?"],
                "suggestions": [],
            },
            href="/recordings",
        )
    maximum = timedelta(days=1 if export_kind == "recording" else 7)
    if end - start > maximum:
        label = "24 hours" if export_kind == "recording" else "7 days"
        return AssistantEvidence(
            evidence_id="E-export-clarification",
            kind="media_export_clarification",
            title="Export range is too long",
            summary=f"That {export_kind} exceeds the {label} limit. What shorter range should I use?",
            data={
                "questions": [f"What range within {label} should I use?"],
                "suggestions": [],
            },
            href="/recordings",
        )

    source = recording_source(call.source or "main")
    options: dict[str, object] = {"height": call.height or 0}
    if export_kind == "timelapse":
        options = {
            "sample_interval_seconds": call.sample_interval_seconds or 30.0,
            "output_fps": call.output_fps or 30,
            **(
                {"height": call.height or 720}
                if call.height is not None or call.width is None
                else {"width": call.width}
            ),
        }
    try:
        job = _media_export_manager().create({
            "kind": export_kind,
            "camera_id": camera.id,
            "source": source,
            "start_epoch": start.timestamp(),
            "end_epoch": end.timestamp(),
            "options": options,
            "origin": "assistant",
        })
    except RuntimeError as exc:
        LOGGER.warning("assistant could not queue media export: %s", exc)
        return AssistantEvidence(
            evidence_id="E-export-clarification",
            kind="media_export_clarification",
            title="Export could not be queued",
            summary="The export queue is unavailable right now. Please try again shortly.",
            data={"questions": [], "suggestions": ["Try the export again"]},
            href="/recordings",
        )
    job_id = str(job["id"])
    local_start = start.astimezone(selected_zone)
    local_end = end.astimezone(selected_zone)
    kind_label = "timelapse" if export_kind == "timelapse" else "video clip"
    summary = (
        f"Queued a {kind_label} for {camera.name} from "
        f"{local_start.strftime('%b %-d, %-I:%M %p')} to {local_end.strftime('%b %-d, %-I:%M %p')}."
    )
    return AssistantEvidence(
        evidence_id=f"E-export-{job_id[:12]}",
        kind="media_export_job",
        title=f"{camera.name} {kind_label}",
        summary=summary,
        data={
            "job_id": job_id,
            "kind": export_kind,
            "camera_id": camera.id,
            "source": source,
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "status": job.get("status"),
            "options": options,
        },
        href=(
            f"/recordings?camera={quote(camera.id, safe='')}&at={start.timestamp():.3f}"
            f"&source={source}"
        ),
        client_data={
            "media_export": {
                "id": job_id,
                "kind": export_kind,
                "camera_id": camera.id,
                "source": source,
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "status": str(job.get("status") or "queued"),
                "phase": str(job.get("phase") or "Queued"),
                "progress": float(job.get("progress") or 0),
            },
        },
    )


def _assistant_media_export_answer(evidence: AssistantEvidence) -> AssistantAnswer:
    if evidence.kind == "media_export_clarification":
        return AssistantAnswer(
            answer=f"{evidence.summary} [{evidence.evidence_id}]",
            citations=[evidence.evidence_id],
            suggestions=list(evidence.data.get("suggestions") or [])[:4],
        )
    return AssistantAnswer(
        answer=(
            f"I started the requested export. It will appear here when the MP4 is ready, "
            f"and you can leave this panel open while it runs. [{evidence.evidence_id}]"
        ),
        citations=[evidence.evidence_id],
        suggestions=[],
    )


def _assistant_search_incidents(
    call: AssistantToolCall,
    time_zone: str,
    active_manager: AppManager,
) -> list[AssistantEvidence]:
    try:
        selected_zone = ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        selected_zone = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    start = _assistant_parse_datetime(call.start_at, selected_zone) or now - timedelta(hours=24)
    end = _assistant_parse_datetime(call.end_at, selected_zone) or now
    if end <= start:
        start, end = end, start
    start = max(start, end - timedelta(days=31))
    rows = [
        _event_row(row)
        for row in active_manager.events.between_compact(start.isoformat(), end.isoformat())
    ]
    summaries = _filter_incident_summaries(
        _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS),
        call.event_type,
        call.camera_id,
        call.object_label,
        call.zone,
    )
    candidate_summaries = summaries[: min(250, max(call.limit * 8, call.limit))]
    hydrated = _incidents_with_faces(
        _hydrate_incidents(candidate_summaries, active_manager),
        active_manager,
    )
    filtered: list[dict[str, Any]] = []
    wanted_face = call.face_name.strip().lower()
    for incident in hydrated:
        payload = _assistant_incident_payload(incident)
        detections = [
            obj
            for event in payload["events"]
            for obj in event["objects"]
            if obj.get("incident_eligible") is not False
        ]
        if call.minimum_confidence is not None and not any(
            float(obj.get("confidence") or 0) >= call.minimum_confidence
            for obj in detections
        ):
            continue
        if wanted_face:
            face_names = {
                str(face.get("name") or "").strip().lower()
                for event in payload["events"]
                for face in event.get("faces") or []
                if isinstance(face, dict)
            }
            if wanted_face not in face_names:
                continue
        filtered.append(incident)
        if len(filtered) >= call.limit:
            break
    query_summary = AssistantEvidence(
        evidence_id="E-search",
        kind="incident_search",
        title="Incident search",
        summary=f"Found {len(filtered)} matching incident(s) in the searched window.",
        data={
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "camera_id": call.camera_id,
            "event_type": call.event_type,
            "object_label": call.object_label,
            "zone": call.zone,
            "minimum_confidence": call.minimum_confidence,
            "face_name": call.face_name,
            "returned": len(filtered),
            "candidate_count": len(summaries),
            "scanned_candidates": len(candidate_summaries),
            "candidate_scan_limited": len(candidate_summaries) < len(summaries),
        },
        href="/incidents",
    )
    evidence = [query_summary]
    for incident in filtered:
        event_id = int(incident.get("representative_event_id") or 0)
        if event_id:
            evidence.append(_assistant_incident_evidence(incident, event_id))
    return evidence


def _assistant_recent_activity_summary(
    call: AssistantToolCall,
    time_zone: str,
    active_manager: AppManager,
) -> AssistantEvidence:
    try:
        selected_zone = ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        selected_zone = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    start = _assistant_parse_datetime(call.start_at, selected_zone) or now - timedelta(hours=24)
    end = _assistant_parse_datetime(call.end_at, selected_zone) or now
    if end <= start:
        start, end = end, start
    start = max(start, end - timedelta(days=31))
    rows = [
        _event_row(row)
        for row in active_manager.events.between_compact(start.isoformat(), end.isoformat())
    ]
    summaries = _filter_incident_summaries(
        _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS),
        "object",
        call.camera_id,
        call.object_label,
        call.zone,
    )

    def counts(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            if value:
                result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))

    duration_minutes = max(0, round((end - start).total_seconds() / 60))
    camera_ids = {str(item.get("camera_id") or "") for item in summaries}
    recent = [
        {
            "event_id": int(item.get("representative_event_id") or 0),
            "camera_id": str(item.get("camera_id") or ""),
            "started_at": item.get("start_at"),
            "labels": list(item.get("labels") or []),
            "zones": list(item.get("zones") or []),
            "trigger_source": item.get("trigger_source") or "camera",
        }
        for item in summaries[:8]
    ]
    return AssistantEvidence(
        evidence_id="E-activity",
        kind="recent_activity_summary",
        title="Recent activity",
        summary=(
            f"{len(summaries)} incidents across {len(camera_ids - {''})} cameras "
            f"during the last {duration_minutes} minutes."
        ),
        data={
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "duration_minutes": duration_minutes,
            "incident_count": len(summaries),
            "object_incident_count": len(summaries),
            "camera_counts": counts([str(item.get("camera_id") or "") for item in summaries]),
            "object_label_counts": counts([
                str(label)
                for item in summaries
                for label in item.get("labels") or []
            ]),
            "zone_counts": counts([
                str(zone)
                for item in summaries
                for zone in item.get("zones") or []
            ]),
            "trigger_counts": counts([
                str(item.get("trigger_source") or "camera") for item in summaries
            ]),
            "recent_notable_incidents": recent,
            "filters": {
                "camera_id": call.camera_id,
                "event_type": "object",
                "object_label": call.object_label,
                "zone": call.zone,
            },
        },
        href="/incidents",
    )


def _assistant_activity_followups(evidence: AssistantEvidence) -> list[str]:
    data = evidence.data
    minutes = max(1, int(data.get("duration_minutes") or 0))
    if minutes % 1440 == 0:
        count = minutes // 1440
        period = f"last {count} day{'s' if count != 1 else ''}"
    elif minutes % 60 == 0:
        count = minutes // 60
        period = f"last {count} hour{'s' if count != 1 else ''}"
    else:
        period = f"last {minutes} minutes"
    followups: list[str] = []
    camera_counts = data.get("camera_counts") or {}
    if camera_counts:
        busiest = next(iter(camera_counts))
        followups.append(f"What happened on {busiest} in the {period}?")
    if int(data.get("object_incident_count") or 0):
        followups.append(f"Show me the object incidents from the {period}")
    triggers = data.get("trigger_counts") or {}
    if any(key in triggers for key in ("adaptive", "visual_backup", "adaptive/visual_backup")):
        followups.append(f"Which incidents did the visual motion check rescue in the {period}?")
    return followups[:3]


def _assistant_prioritize_trace_candidates(
    candidate_summaries: list[dict[str, Any]],
    appearance_matches: list[dict[str, Any]],
    appearance_event_ids: set[int],
    distance_from_anchor: Callable[[dict[str, Any]], float],
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Keep strongest appearance evidence before filling a bounded temporal scan."""
    candidate_summaries = sorted(candidate_summaries, key=distance_from_anchor)
    appearance_summaries = [
        summary
        for summary in candidate_summaries
        if any(
            int(event.get("id") or 0) in appearance_event_ids
            for event in summary.get("events") or []
        )
    ]
    appearance_score_by_event = {
        int(item.get("event_id") or 0): float(item.get("similarity") or 0.0)
        for item in appearance_matches
        if item.get("visually_similar")
    }
    appearance_summaries.sort(
        key=lambda summary: max(
            (
                appearance_score_by_event.get(int(event.get("id") or 0), 0.0)
                for event in summary.get("events") or []
            ),
            default=0.0,
        ),
        reverse=True,
    )
    retained_ids = {id(summary) for summary in appearance_summaries}
    retained_appearance = appearance_summaries[:min(100, limit)]
    return retained_appearance + [
        summary
        for summary in candidate_summaries
        if id(summary) not in retained_ids
    ][:max(0, limit - len(retained_appearance))]


def _assistant_trace_across_cameras(
    call: AssistantToolCall,
    request: AssistantChatRequest,
    active_manager: AppManager,
) -> list[AssistantEvidence]:
    event_id = call.event_id or request.context.incident_event_id
    if not event_id and not call.face_name.strip() and not call.object_label.strip():
        return []
    anchor = (
        _assistant_incident_for_event(int(event_id), active_manager)
        if event_id
        else None
    )
    if event_id and anchor is None:
        return []
    try:
        selected_zone = ZoneInfo(request.context.time_zone)
    except (ZoneInfoNotFoundError, ValueError):
        selected_zone = ZoneInfo("America/New_York")
    now = datetime.now(timezone.utc)
    try:
        anchor_at = datetime.fromisoformat(
            str((anchor or {}).get("start_at") or "").replace("Z", "+00:00")
        )
    except ValueError:
        anchor_at = now
    if anchor_at.tzinfo is None:
        anchor_at = anchor_at.replace(tzinfo=timezone.utc)
    default_start = (
        anchor_at - timedelta(minutes=15) if anchor else now - timedelta(hours=24)
    )
    default_end = anchor_at + timedelta(minutes=15) if anchor else now
    start = _assistant_parse_datetime(call.start_at, selected_zone) or default_start
    end = _assistant_parse_datetime(call.end_at, selected_zone) or default_end
    if end <= start:
        start, end = end, start
    start = max(start, end - timedelta(hours=24))
    appearance_matches: list[dict[str, Any]] = []
    appearance_index = getattr(active_manager, "appearance_index", None)
    if event_id and appearance_index is not None:
        try:
            appearance_matches = appearance_index.matches(
                int(event_id),
                start_at=start.isoformat(),
                end_at=end.isoformat(),
                cross_camera_only=True,
                limit=100,
            )
        except Exception:
            LOGGER.exception(
                "cross-camera appearance lookup failed for event %s",
                event_id,
            )
    appearance_event_ids = {
        int(item.get("event_id") or 0)
        for item in appearance_matches
        if item.get("visually_similar")
    }
    rows = [
        _event_row(row)
        for row in active_manager.events.between_compact(
            start.isoformat(),
            end.isoformat(),
        )
    ]
    summaries = _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS)
    wanted_label = call.object_label.strip().lower()
    anchor_labels = {
        str(label).strip().lower()
        for label in (anchor or {}).get("labels") or []
        if str(label).strip()
    }
    target_labels = {wanted_label} if wanted_label else anchor_labels
    candidate_summaries = [
        summary
        for summary in summaries
        if not target_labels
        or target_labels & {
            str(label).strip().lower()
            for label in summary.get("labels") or []
        }
        or bool(call.face_name.strip())
        or any(
            int(event.get("id") or 0) in appearance_event_ids
            for event in summary.get("events") or []
        )
    ]
    def distance_from_anchor(summary: dict[str, Any]) -> float:
        parsed = _assistant_parse_datetime(
            str(summary.get("start_at") or ""),
            selected_zone,
        )
        return (
            abs(parsed.timestamp() - anchor_at.timestamp())
            if parsed is not None
            else float("inf")
        )

    candidate_summaries = _assistant_prioritize_trace_candidates(
        candidate_summaries,
        appearance_matches,
        appearance_event_ids,
        distance_from_anchor,
    )

    candidates = _incidents_with_faces(
        _hydrate_incidents(candidate_summaries, active_manager),
        active_manager,
    )
    correlation_anchor = anchor or {
        "representative_event_id": 0,
        "camera_id": "",
        "start_at": start.isoformat(),
        "labels": [call.object_label] if call.object_label else [],
        "faces": [],
    }
    matches = correlate_incident_timeline(
        correlation_anchor,
        candidates,
        object_label=call.object_label,
        face_name=call.face_name,
        limit=min(call.limit, 12),
    )
    incident_by_event_id: dict[int, dict[str, Any]] = {}
    for incident in candidates:
        representative_id = int(incident.get("representative_event_id") or 0)
        if representative_id > 0:
            incident_by_event_id[representative_id] = incident
        for event in incident.get("events") or []:
            candidate_event_id = int(event.get("id") or 0)
            if candidate_event_id > 0:
                incident_by_event_id[candidate_event_id] = incident
    matches_by_event_id = {
        int(item.get("event_id") or 0): item
        for item in matches
    }
    for appearance in appearance_matches:
        if not appearance.get("visually_similar"):
            continue
        matched_incident = incident_by_event_id.get(int(appearance["event_id"]))
        if matched_incident is None:
            continue
        representative_id = int(
            matched_incident.get("representative_event_id")
            or appearance["event_id"]
        )
        similarity = float(appearance.get("similarity") or 0.0)
        reason = (
            f"{str(appearance.get('model_kind') or 'object').title()} appearance "
            f"is {round(similarity * 100)}% similar using the same ReID model"
        )
        existing = matches_by_event_id.get(representative_id)
        if existing is not None:
            existing.setdefault("reasons", []).append(reason)
            existing["appearance_similarity"] = round(similarity, 4)
            if existing.get("match_strength") not in {
                "confirmed_identity",
                "possible_identity",
            }:
                existing["match_strength"] = "appearance_similarity"
                existing["confidence"] = round(similarity, 3)
            continue
        matched_at = str(matched_incident.get("start_at") or appearance.get("created_at") or "")
        matched_epoch = _assistant_parse_datetime(matched_at, selected_zone)
        item = {
            "incident": matched_incident,
            "event_id": representative_id,
            "camera_id": str(matched_incident.get("camera_id") or appearance.get("camera_id") or ""),
            "start_at": matched_at,
            "seconds_from_anchor": round(
                (matched_epoch.timestamp() - anchor_at.timestamp())
                if matched_epoch is not None
                else 0.0,
                1,
            ),
            "match_strength": "appearance_similarity",
            "confidence": round(similarity, 3),
            "appearance_similarity": round(similarity, 4),
            "reasons": [reason],
        }
        matches.append(item)
        matches_by_event_id[representative_id] = item
    strength_rank = {
        "confirmed_identity": 4,
        "possible_identity": 3,
        "appearance_similarity": 2,
        "context_candidate": 1,
    }
    matches = sorted(
        sorted(
            matches,
            key=lambda item: (
                -strength_rank.get(str(item.get("match_strength") or ""), 0),
                -float(item.get("confidence") or 0.0),
                abs(float(item.get("seconds_from_anchor") or 0.0)),
            ),
        )[:min(call.limit, 12)],
        key=lambda item: str(item.get("start_at") or ""),
    )
    confirmed = sum(
        item["match_strength"] == "confirmed_identity" for item in matches
    )
    possible = sum(
        item["match_strength"] == "possible_identity" for item in matches
    )
    contextual = sum(
        item["match_strength"] == "context_candidate" for item in matches
    )
    appearance_similar = sum(
        item["match_strength"] == "appearance_similarity" for item in matches
    )
    timeline_data = {
        "anchor_event_id": int(event_id) if event_id else None,
        "anchor_camera_id": (anchor or {}).get("camera_id"),
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "object_label": call.object_label,
        "face_name": call.face_name,
        "matches": [
            {key: item.get(key) for key in (
                "event_id", "camera_id", "start_at", "seconds_from_anchor",
                "match_strength", "confidence", "reasons",
                "appearance_similarity",
            )}
            for item in matches
        ],
        "limitations": [
            "Confirmed recognized faces can link incidents across cameras.",
            "Possible face matches remain uncertain.",
            "Shared person, vehicle, or animal labels plus nearby time provide context only.",
            "Appearance similarity uses durable, model-versioned ReID vectors and is stronger than a shared class label, but it is not proof of identity.",
            "Camera angle, lighting, occlusion, and visually similar subjects can change the score.",
            "Only the strongest 12 candidates and at most 24 hours are returned.",
        ],
    }
    trace = AssistantEvidence(
        evidence_id=f"E-trace-{event_id or 'search'}",
        kind="cross_camera_timeline",
        title="Cross-camera investigation timeline",
        summary=(
            f"Found {len(matches)} bounded timeline candidate(s): {confirmed} confirmed "
            f"identity, {possible} possible identity, {appearance_similar} appearance-similar, "
            f"and {contextual} context-only."
        ),
        data=timeline_data,
        href=(
            _assistant_incident_evidence(anchor, int(event_id)).href
            if anchor and event_id
            else "/incidents"
        ),
        client_data={"timeline": timeline_data},
    )
    evidence = [trace]
    if anchor and event_id:
        evidence.append(_assistant_incident_evidence(anchor, int(event_id)))
    for item in matches:
        incident = item["incident"]
        evidence.append(
            _assistant_incident_evidence(incident, int(item["event_id"]))
        )
    return evidence


def _assistant_execute_tool(
    call: AssistantToolCall,
    request: AssistantChatRequest,
    active_config: AppConfig,
    active_manager: AppManager,
) -> list[AssistantEvidence]:
    if call.name == "get_system_health":
        return [_assistant_system_evidence(active_manager)]
    if call.name == "get_camera_health":
        camera_id = call.camera_id or request.context.camera_id
        return _assistant_camera_evidence(active_manager, camera_id)
    if call.name == "explain_configuration":
        return [_assistant_configuration_evidence(active_config)]
    if call.name == "inspect_incident":
        event_id = call.event_id or request.context.incident_event_id
        item = _assistant_inspect_incident(int(event_id), active_manager) if event_id else None
        return [item] if item is not None else []
    if call.name == "analyze_incident_visual":
        event_id = call.event_id or request.context.incident_event_id
        if not event_id:
            return []
        if not AUDIT_AI_LIMITER.acquire(blocking=False):
            raise AuditAiError("another visual AI review is already running")
        try:
            item = _assistant_visual_incident_evidence(
                int(event_id), active_config, active_manager
            )
        finally:
            AUDIT_AI_LIMITER.release()
        return [item] if item is not None else []
    if call.name == "search_incidents":
        return _assistant_search_incidents(
            call,
            request.context.time_zone,
            active_manager,
        )
    if call.name == "summarize_recent_activity":
        return [_assistant_recent_activity_summary(
            call,
            request.context.time_zone,
            active_manager,
        )]
    if call.name == "trace_across_cameras":
        return _assistant_trace_across_cameras(call, request, active_manager)
    if call.name == "create_media_export":
        return [_assistant_media_export_evidence(call, request, active_config)]
    return []


@app.get("/api/assistant/status")
def assistant_status() -> dict[str, Any]:
    ai = config.audit_ai
    configured = bool(
        ai.enabled
        and ai.assistant_enabled
        and ai_provider_configured(ai)
    )
    provider = AssistantProvider(ai)
    return {
        "enabled": bool(ai.assistant_enabled),
        "configured": configured,
        "provider": ai.provider,
        "fast_model": provider.model_for_tier("fast"),
        "reasoning_model": provider.model_for_tier("deep"),
        "read_only": False,
        "media_exports": True,
    }


@app.post("/api/assistant/chat")
async def assistant_chat(request: AssistantChatRequest) -> dict[str, Any]:
    with MANAGER_RELOAD_LOCK:
        active_config = config
        active_manager = manager
        ai = active_config.audit_ai
        if not ai.assistant_enabled:
            raise HTTPException(status_code=409, detail="SurvNG Assistant is disabled")
        if not ai.enabled:
            raise HTTPException(status_code=409, detail="AI features are disabled in Admin")
        if not ai_provider_configured(ai):
            raise HTTPException(status_code=409, detail="AI provider credentials are not configured")
        if not ASSISTANT_LIMITER.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="SurvNG Assistant is busy; try again shortly")
        _begin_ai_operation("assistant")

    def run() -> dict[str, Any]:
        provider = AssistantProvider(ai)
        catalog = _assistant_catalog(active_config, active_manager)
        try:
            selected_zone = ZoneInfo(request.context.time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo("America/New_York")
        now = datetime.now(timezone.utc).astimezone(selected_zone)
        plan = provider.plan(request, catalog, now.isoformat())
        evidence: list[AssistantEvidence] = []
        seen: set[str] = set()
        for call in plan.tool_calls:
            for item in _assistant_execute_tool(
                call,
                request,
                active_config,
                active_manager,
            ):
                if item.evidence_id not in seen:
                    seen.add(item.evidence_id)
                    evidence.append(item)
        media_export = next(
            (
                item for item in evidence
                if item.kind in {"media_export_job", "media_export_clarification"}
            ),
            None,
        )
        answer = (
            _assistant_media_export_answer(media_export)
            if media_export is not None
            else provider.answer(request, evidence, plan.reasoning_tier)
        )
        activity = next(
            (item for item in evidence if item.kind == "recent_activity_summary"),
            None,
        )
        suggestions = (
            _assistant_activity_followups(activity)
            if activity is not None
            else answer.suggestions
        )
        return {
            "message": answer.answer,
            "citations": answer.citations,
            "suggestions": suggestions,
            "evidence": [item.client_payload() for item in evidence],
            "tools": [call.name for call in plan.tool_calls],
            "reasoning_tier": plan.reasoning_tier,
            "model": provider.model_for_tier(plan.reasoning_tier),
            "read_only": False,
        }

    try:
        return await asyncio.to_thread(run)
    except AuditAiError as exc:
        LOGGER.warning("SurvNG Assistant provider failure: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("SurvNG Assistant request failed")
        raise HTTPException(status_code=500, detail="SurvNG Assistant could not complete the request") from exc
    finally:
        _end_ai_operation("assistant")
        ASSISTANT_LIMITER.release()


@app.post("/api/incidents/{event_id}/ai-apply")
def incident_ai_apply(event_id: int, request: IncidentAiApplyRequest) -> dict:
    if not request.confirmed:
        raise HTTPException(
            status_code=400,
            detail="explicit confirmation is required",
        )
    if not request.changes:
        raise HTTPException(status_code=400, detail="no recommendation changes supplied")
    if any(change.scope != "camera" for change in request.changes):
        raise HTTPException(
            status_code=400,
            detail="incident reviews may only change camera-scoped motion settings",
        )
    with MANAGER_RELOAD_LOCK:
        if not config.audit_ai.allow_apply_recommendations:
            raise HTTPException(
                status_code=403,
                detail="applying AI recommendations is disabled",
            )
        event = manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="incident event not found")
        next_config = config.model_copy(deep=True)
        camera = camera_by_id(next_config, str(event.get("camera_id") or ""))
        if camera is None:
            raise HTTPException(status_code=404, detail="incident camera not found")
        current_fingerprint = _assistant_motion_config_fingerprint(
            next_config, camera
        )
        if request.configuration_fingerprint != current_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="motion settings changed after this review; run visual analysis again",
            )
        if not _verify_ai_recommendation_token(
            request.recommendation_proof,
            kind="incident_visual",
            record_id=event_id,
            camera_id=camera.id,
            configuration_fingerprint=current_fingerprint,
            changes=request.changes,
        ):
            raise HTTPException(
                status_code=409,
                detail="AI recommendations are expired or do not match this visual review",
            )
        try:
            changes, previews = _assistant_motion_change_previews(
                next_config,
                camera,
                request.changes,
            )
            if not changes:
                raise ValueError("recommendations do not change active settings")
            for change in changes:
                _apply_pipeline_ai_change(
                    next_config,
                    camera,
                    change,
                    validate_tuning_value(change.setting, change.value),
                )
            validate_motion_pipeline_configuration(next_config)
            next_config = AppConfig.model_validate(
                next_config.model_dump(mode="json")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _effective_config, apply_result = apply_config_update(next_config)
    return {
        "ok": True,
        "event_id": event_id,
        "camera_id": camera.id,
        "applied": previews,
        "workers_restarted": bool(apply_result["camera_workers_restarted"]),
        "apply_mode": apply_result["apply_mode"],
    }


class FacePersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=1000)
    observation_id: int | None = Field(default=None, gt=0)


class FaceAssignment(BaseModel):
    person_id: int | None = Field(default=None, gt=0)


def _sync_face_observations(limit: int = 5000) -> int:
    global FACE_OBSERVATIONS_SYNCED
    with MANAGER_RELOAD_LOCK, FACE_OBSERVATIONS_SYNC_LOCK:
        if FACE_OBSERVATIONS_SYNCED:
            return 0
        active_manager = manager
        inserted = active_manager.faces.ingest_events(
            active_manager.events.recent(max(1, min(limit, 20000)))
        )
        FACE_OBSERVATIONS_SYNCED = True
        return inserted


def _start_face_observation_sync() -> None:
    global FACE_OBSERVATIONS_SYNC_THREAD
    with FACE_OBSERVATIONS_SYNC_THREAD_LOCK:
        if FACE_OBSERVATIONS_SYNCED:
            return
        if FACE_OBSERVATIONS_SYNC_THREAD is not None and FACE_OBSERVATIONS_SYNC_THREAD.is_alive():
            return

        def synchronize() -> None:
            try:
                _sync_face_observations()
            except Exception:
                logging.getLogger(__name__).exception("background face observation sync failed")

        FACE_OBSERVATIONS_SYNC_THREAD = threading.Thread(
            target=synchronize,
            name="face-observation-sync",
            daemon=True,
        )
        FACE_OBSERVATIONS_SYNC_THREAD.start()


@app.get("/api/faces/status")
def face_status() -> dict:
    _start_face_observation_sync()
    stats = manager.faces.stats()
    recognition = manager.faces.recognition_status()
    if recognition.get("ready"):
        pending = int(recognition.get("pending") or 0)
        failed = int(recognition.get("failed") or 0)
        recognition_message = (
            f"Recognition ready on {recognition.get('device') or 'OpenVINO'}; "
            f"{recognition.get('embedded', 0)} faces embedded and "
            f"{recognition.get('suggested', 0)} suggestions awaiting review"
            f"; {pending} pending and {failed} unable to process."
        )
    else:
        recognition_message = str(recognition.get("error") or "Configure an OpenVINO face embedding model.")
    return {
        **stats,
        "recognition_ready": bool(recognition.get("ready")),
        "recognition_message": recognition_message,
        "recognition": recognition,
    }


@app.get("/api/faces/people")
def face_people() -> list[dict]:
    _start_face_observation_sync()
    return manager.faces.people()


def _public_face_observation(observation: dict) -> dict:
    payload = dict(observation)
    payload.pop("snapshot_path", None)
    return payload


@app.post("/api/faces/people")
def create_face_person(payload: FacePersonCreate) -> dict:
    try:
        return manager.faces.create_person(payload.name, payload.observation_id, payload.notes)
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.delete("/api/faces/people/{person_id}")
def delete_face_person(person_id: int) -> dict:
    if not manager.faces.delete_person(person_id):
        raise HTTPException(status_code=404, detail="person not found")
    return {"deleted": True, "person_id": person_id}


@app.get("/api/faces/observations")
def face_observations(
    person_id: int | None = None,
    camera_id: str = "",
    status: str = "all",
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    _start_face_observation_sync()
    observations = manager.faces.observations(
        person_id=person_id,
        camera_id=camera_id,
        status=status if status in {"all", "known", "unknown", "suggested"} else "all",
        limit=limit,
        offset=offset,
    )
    return [_public_face_observation(observation) for observation in observations]


@app.get("/api/faces/observations/count")
def face_observation_count(
    person_id: int | None = None,
    camera_id: str = "",
    status: str = "all",
) -> dict:
    _start_face_observation_sync()
    return {
        "total": manager.faces.observation_count(
            person_id=person_id,
            camera_id=camera_id,
            status=status if status in {"all", "known", "unknown", "suggested"} else "all",
        )
    }


@app.get("/api/faces/observations/{observation_id}")
def face_observation(observation_id: int) -> dict:
    observation = manager.faces.observation(observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    return _public_face_observation(observation)


@app.put("/api/faces/observations/{observation_id}")
def assign_face_observation(observation_id: int, payload: FaceAssignment) -> dict:
    try:
        observation = manager.faces.assign(observation_id, payload.person_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if observation is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    return _public_face_observation(observation)


@app.get("/api/faces/observations/{observation_id}/crop.jpg")
def face_crop(observation_id: int, padding: float = 0.2) -> FileResponse:
    if not math.isfinite(padding):
        raise HTTPException(status_code=422, detail="padding must be finite")
    active_manager = manager
    result = active_manager.faces.snapshot_path(observation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="face observation not found")
    snapshot_path, box = result
    pad = max(0.0, min(float(padding), 1.0))
    try:
        stat = snapshot_path.stat()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="snapshot is unavailable") from exc
    box_identity = json.dumps(box, sort_keys=True, separators=(",", ":"))
    identity = f"{observation_id}:{snapshot_path}:{stat.st_size}:{stat.st_mtime_ns}:{box_identity}:{pad:.3f}"

    def build() -> bytes:
        frame = cv2.imread(str(snapshot_path))
        if frame is None:
            raise HTTPException(status_code=404, detail="snapshot is unavailable")
        height, width = frame.shape[:2]
        x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
        x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
        right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
        if right <= left or bottom <= top:
            raise HTTPException(status_code=422, detail="face crop is invalid")
        ok, encoded = cv2.imencode(".jpg", frame[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(status_code=500, detail="failed to encode face crop")
        return encoded.tobytes()

    cached = active_manager.image_cache.get_or_create("faces", identity, build)
    return FileResponse(
        cached,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )


@app.post("/api/events/{event_id}/detect")
def detect_event_snapshot(event_id: int, confidence: float = 0.35) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        event = active_manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    try:
        snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="snapshot not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="snapshot outside storage directory") from None

    if not math.isfinite(confidence):
        raise HTTPException(status_code=422, detail="confidence must be finite")
    safe_confidence = max(0.01, min(0.99, float(confidence)))
    frame = cv2.imread(str(snapshot_path))
    if frame is None:
        raise HTTPException(status_code=422, detail="failed to read snapshot")

    started = time.perf_counter()
    camera = camera_by_id(active_config, str(event.get("camera_id") or ""))
    effective_confidence = detection_threshold(camera, safe_confidence) if camera else safe_confidence
    objects = active_manager.detector.detect(frame, confidence_threshold=effective_confidence)
    detector_error = detection_failure(objects)
    if detector_error:
        raise HTTPException(status_code=503, detail=detector_error)
    if camera:
        apply_detection_zones(
            camera,
            objects,
            int(frame.shape[1]),
            int(frame.shape[0]),
            safe_confidence,
            bool(active_config.detector.require_incident_zone),
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    for detected_object in objects:
        detected_object["frame_source"] = detected_object.get("frame_source") or "manual_snapshot"
        detected_object["detection_source"] = "manual_openvino"
        detected_object["manual_confidence_threshold"] = safe_confidence
    persisted_event = active_manager.events.replace_detected_objects(
        event_id,
        objects_to_json(objects),
    )
    if persisted_event is None:
        raise HTTPException(status_code=404, detail="event not found")
    detected = [
        item for item in objects
        if item.get("label") and item.get("box") and item.get("incident_eligible") is not False
    ]
    if detected:
        active_manager.publish_event("object", {
            "event_id": event_id,
            "camera_id": str(event.get("camera_id") or ""),
            "timestamp": str(event.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "snapshot_path": str(snapshot_path),
            "recording_path": str(event.get("recording_path") or ""),
            "source": "manual_openvino",
            "objects": detected,
        })
    detector_status = active_manager.detector_status()
    return {
        "event_id": event_id,
        "camera_id": event.get("camera_id"),
        "snapshot_path": "available",
        "snapshot_width": int(frame.shape[1]),
        "snapshot_height": int(frame.shape[0]),
        "confidence": safe_confidence,
        "elapsed_ms": elapsed_ms,
        "objects": objects,
        "object_count": len(detected),
        "labels": sorted({str(item.get("label")) for item in detected}),
        "event": _event_row(persisted_event),
        "persisted": True,
        "detector": {
            "enabled": detector_status.get("enabled"),
            "loaded_backend": detector_status.get("loaded_backend"),
            "loaded_device": detector_status.get("loaded_device"),
            "configured_device": detector_status.get("configured_device"),
            "input_shape": detector_status.get("input_shape"),
            "output_format": detector_status.get("output_format"),
        },
    }


def _tracking_comparison_evidence(result: dict) -> dict:
    engines: dict[str, dict] = {}
    for implementation in ("survng_hybrid", "ultralytics_botsort"):
        engine = result.get("engines", {}).get(implementation, {})
        engines[implementation] = {
            key: engine.get(key)
            for key in (
                "track_count",
                "observations",
                "reid_recoveries",
                "fragmentation_proxy",
                "initialization_ms",
                "processing_ms",
                "average_ms_per_frame",
                "labels",
            )
        }
    return {
        key: result.get(key)
        for key in (
            "sample_fps",
            "frames_processed",
            "duration_seconds",
            "frame_width",
            "frame_height",
            "average_frame_decode_ms",
            "average_detection_ms_per_frame",
            "average_appearance_ms_per_frame",
            "appearance_failures",
            "clip_preparation_ms",
            "elapsed_ms",
        )
    } | {"engines": engines}


@app.get("/api/tracking-comparisons")
def tracking_comparison_history(camera_id: str = "", limit: int = 25) -> dict:
    normalized_camera_id = str(camera_id or "").strip()
    return {
        "items": manager.events.tracking_comparison_history(
            camera_id=normalized_camera_id,
            limit=limit,
        ),
        "summary": manager.events.tracking_comparison_summary(
            camera_id=normalized_camera_id,
        ),
    }


@app.put("/api/tracking-comparisons/{comparison_id}/verdict")
def update_tracking_comparison_verdict(
    comparison_id: int,
    payload: TrackingComparisonVerdictRequest,
) -> dict:
    comparison = manager.events.set_tracking_comparison_verdict(
        comparison_id,
        payload.verdict,
    )
    if comparison is None:
        raise HTTPException(status_code=404, detail="tracking comparison not found")
    return {
        "comparison": comparison,
        "summary": manager.events.tracking_comparison_summary(
            camera_id=str(comparison.get("camera_id") or ""),
        ),
    }


@app.post("/api/events/{event_id}/tracking-comparison")
def compare_event_tracking(event_id: int, duration_seconds: float | None = None) -> dict:
    dependency = ultralytics_botsort_dependency_status()
    if not dependency["available"]:
        raise HTTPException(status_code=503, detail=dependency["reason"])
    if duration_seconds is not None and not math.isfinite(duration_seconds):
        raise HTTPException(status_code=422, detail="duration_seconds must be finite")
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        event = active_manager.events.get(event_id)
    duration = max(
        3.0,
        min(
            30.0,
            float(duration_seconds)
            if duration_seconds is not None
            else min(15.0, active_config.detector.tracking.max_session_seconds),
        ),
    )
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    camera = camera_by_id(active_config, str(event.get("camera_id") or ""))
    if camera is None:
        raise HTTPException(status_code=404, detail="event camera is not configured")
    if not TRACKING_COMPARISON_LIMITER.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="another tracking comparison is already running",
            headers={"Retry-After": "3"},
        )
    try:
        request_started = time.perf_counter()
        enriched = _event_row(event)
        clip_started = time.perf_counter()
        comparison_input = _ensure_event_clip(
            enriched,
            before=0.0,
            after=duration,
            source="main",
        )
        clip_ms = (time.perf_counter() - clip_started) * 1000.0
        start_epoch = event_epoch(enriched)
        runner = TrackingComparisonRunner(
            config=active_config.detector.tracking,
            detector=active_manager.detector,
            appearance_encoder=active_manager.person_reidentifier,
        )
        detector_dimensions = [
            int(value)
            for value in (getattr(active_manager.detector, "input_shape", None) or [])
            if isinstance(value, (int, float)) and int(value) > 0
        ]
        comparison_frames = sampled_video_frames(
            comparison_input,
            start_epoch=start_epoch,
            sample_fps=active_config.detector.tracking.sample_fps,
            duration_seconds=duration,
            ffmpeg_path=active_config.ffmpeg_path,
            maximum_width=max([640, *detector_dimensions]),
        )
        result = runner.run(camera, comparison_frames)
        response = {
            "event_id": event_id,
            "camera_id": camera.id,
            "created_at": str(enriched.get("created_at") or ""),
            "requested_duration_seconds": duration,
            "clip_preparation_ms": round(clip_ms, 1),
            "elapsed_ms": round((time.perf_counter() - request_started) * 1000.0, 1),
            **result,
        }
        comparison = active_manager.events.save_tracking_comparison(
            event_id=event_id,
            camera_id=camera.id,
            event_created_at=str(enriched.get("created_at") or ""),
            result=_tracking_comparison_evidence(response),
        )
        response["comparison_id"] = comparison["id"]
        response["verdict"] = comparison["verdict"]
        response["comparison"] = comparison
        response["evidence_summary"] = active_manager.events.tracking_comparison_summary(
            camera_id=camera.id,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger(__name__).exception(
            "tracking comparison failed for event %d",
            event_id,
        )
        raise HTTPException(
            status_code=422,
            detail="tracking comparison failed",
        ) from None
    finally:
        TRACKING_COMPARISON_LIMITER.release()


@app.post("/api/detector/frame")
async def detect_debug_frame(request: Request, confidence: float = 0.35) -> dict:
    maximum_bytes = 2 * 1024 * 1024
    try:
        content_length = int(request.headers.get("content-length") or 0)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid content-length header") from None
    if content_length < 0:
        raise HTTPException(status_code=400, detail="invalid content-length header")
    if content_length > maximum_bytes:
        raise HTTPException(status_code=413, detail="debug frame is too large")
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > maximum_bytes:
            raise HTTPException(status_code=413, detail="debug frame is too large")
        payload.extend(chunk)
    if not payload:
        raise HTTPException(status_code=422, detail="invalid debug frame")
    frame = cv2.imdecode(np.frombuffer(bytes(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="failed to decode debug frame")
    if not math.isfinite(confidence):
        raise HTTPException(status_code=422, detail="confidence must be finite")
    safe_confidence = max(0.01, min(0.99, float(confidence)))
    with MANAGER_RELOAD_LOCK:
        active_detector = manager.detector
    started = time.perf_counter()
    objects = await asyncio.to_thread(
        active_detector.detect,
        frame,
        confidence_threshold=safe_confidence,
    )
    detector_error = detection_failure(objects)
    if detector_error:
        raise HTTPException(status_code=503, detail=detector_error)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    detected = [item for item in objects if item.get("label") and item.get("box")]
    return {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "confidence": safe_confidence,
        "elapsed_ms": elapsed_ms,
        "objects": detected,
    }


@app.get("/api/cameras/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, source: str = "live") -> Response:
    active_manager = manager
    worker = active_manager.workers.get(camera_id)
    camera = active_manager.camera(camera_id)
    if worker is None or camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    if not worker.status().get("running"):
        raise HTTPException(status_code=503, detail="camera is powered off")
    try:
        image = active_manager.go2rtc.snapshot(camera, source)
    except Go2RtcError:
        fallback_source = "live" if source == "main" and camera.source_url("main") == camera.source_url("live") else source
        image = worker.snapshot(fallback_source)
    if image is None:
        raise HTTPException(status_code=503, detail="no frame available")
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/cameras/{camera_id}/zone-snapshot.jpg")
def zone_snapshot(camera_id: str, source: str = "live") -> Response:
    active_manager = manager
    worker = active_manager.workers.get(camera_id)
    camera = active_manager.camera(camera_id)
    if worker is None or camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    image = None
    if worker.status().get("running"):
        try:
            image = active_manager.go2rtc.snapshot(camera, source)
        except Go2RtcError:
            fallback_source = "live" if source == "main" and camera.source_url("main") == camera.source_url("live") else source
            image = worker.snapshot(fallback_source)
    if image is not None:
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    for event in active_manager.events.recent(1000):
        if event.get("camera_id") != camera_id:
            continue
        try:
            snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        media_type = snapshot_media_type(snapshot_path)
        return FileResponse(snapshot_path, media_type=media_type, headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=503, detail="no camera or event snapshot available")


@app.get("/api/cameras/{camera_id}/live-info")
def live_info(camera_id: str, response: Response, source: str = "live") -> dict:
    response.headers["Cache-Control"] = "no-store"
    active_manager = manager
    camera = active_manager.camera(camera_id)
    worker = active_manager.workers.get(camera_id)
    if camera is None or worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    if not worker.status().get("running"):
        raise HTTPException(status_code=503, detail="camera is powered off")
    try:
        return active_manager.go2rtc.stream_info(camera, source)
    except Go2RtcError as exc:
        return {
            "available": False,
            "video_codec": "",
            "video_codecs": [],
            "compatibility": "native",
            "delivery": "native",
            "transcoding": False,
            "error": str(exc)[:160],
        }


@app.get("/api/cameras/{camera_id}/stream.mjpg")
async def stream(
    camera_id: str,
    request: Request,
    source: str = "live",
    fps: float = 4.0,
) -> StreamingResponse:
    active_manager = manager
    worker = active_manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    if not worker.status().get("running"):
        raise HTTPException(status_code=503, detail="camera is powered off")
    frame_interval = 1.0 / max(0.5, min(4.0, fps))

    async def frames():
        while not await request.is_disconnected():
            image = await asyncio.to_thread(worker.snapshot, source)
            if image is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + image
                    + b"\r\n"
                )
            await asyncio.sleep(frame_interval if image is not None else 0.1)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


async def relay_go2rtc_websocket(websocket: WebSocket, camera_id: str, transport: str) -> None:
    """Relay one camera's go2rtc WebSocket without exposing the go2rtc API."""
    try:
        active_manager = manager
        camera = active_manager.camera(camera_id)
        worker = active_manager.workers.get(camera_id)
        if camera is None or worker is None:
            raise Go2RtcError("camera not found")
        if not worker.status().get("running"):
            raise Go2RtcError("camera is powered off")
        upstream_url = await asyncio.to_thread(
            active_manager.go2rtc.websocket_url,
            camera,
            websocket.query_params.get("source", "live"),
        )
    except (Go2RtcError, OSError, RuntimeError):
        await websocket.close(code=1008)
        return
    accepted = False
    tasks: list[asyncio.Task] = []
    try:
        async with websockets.connect(
            upstream_url,
            open_timeout=5,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=10,
            max_size=8 * 1024 * 1024,
            max_queue=4,
            compression=None,
        ) as upstream:
            # Do not report an open browser relay until the upstream go2rtc
            # socket is actually ready. This lets the client advance to its
            # next transport immediately when go2rtc is unavailable.
            await websocket.accept()
            accepted = True

            async def browser_to_go2rtc() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def go2rtc_to_browser() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            tasks = [
                asyncio.create_task(browser_to_go2rtc()),
                asyncio.create_task(go2rtc_to_browser()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:
        logging.getLogger(__name__).warning("%s relay failed for %s: %s", transport, camera_id, exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await websocket.close(code=1000 if accepted else 1013)
        except RuntimeError:
            pass


@app.websocket("/api/cameras/{camera_id}/webrtc")
async def webrtc_signaling(websocket: WebSocket, camera_id: str) -> None:
    """Relay signaling while go2rtc WebRTC media remains direct and shared."""
    await relay_go2rtc_websocket(websocket, camera_id, "WebRTC signaling")


@app.websocket("/api/cameras/{camera_id}/mse")
async def mse_stream(websocket: WebSocket, camera_id: str) -> None:
    """Relay go2rtc fragmented MP4 media over the SurvNG connection."""
    await relay_go2rtc_websocket(websocket, camera_id, "MSE stream")


@app.post("/api/cameras/{camera_id}/camera/start")
def start_camera(camera_id: str) -> dict:
    if not manager.start_camera(camera_id):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True}


@app.post("/api/cameras/{camera_id}/camera/stop")
def stop_camera(camera_id: str) -> dict:
    if not manager.stop_camera(camera_id):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True}

@app.post("/api/cameras/{camera_id}/motion-test")
def motion_test(camera_id: str) -> dict:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    worker.handle_motion_event("manual/test", "manual GUI trigger")
    return {"ok": True}


@app.get("/api/cameras/{camera_id}/motion-debug")
def motion_debug_status(camera_id: str) -> dict:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return worker.motion_debug_status()


@app.put("/api/cameras/{camera_id}/motion-debug")
def set_motion_debug(camera_id: str, state: CameraFeatureState) -> dict:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    worker.set_motion_debug_enabled(state.enabled)
    return worker.motion_debug_status()


@app.get("/api/cameras/{camera_id}/motion-debug/{layer}.jpg")
def motion_debug_image(camera_id: str, layer: str) -> Response:
    worker = manager.workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    image = worker.motion_debug_image(layer)
    if image is None:
        raise HTTPException(status_code=404, detail="motion debug layer not available")
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/cameras/{camera_id}/recording/start")
def start_recording(camera_id: str, source: str = "main") -> dict:
    if not manager.set_recording(camera_id, True):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True, "recording_enabled": True}


@app.post("/api/cameras/{camera_id}/recording/stop")
def stop_recording(camera_id: str, source: str | None = None) -> dict:
    if not manager.set_recording(camera_id, False):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True, "recording_enabled": False}


@app.put("/api/cameras/{camera_id}/recording")
def set_camera_recording(camera_id: str, state: CameraFeatureState) -> dict:
    if not manager.set_recording(camera_id, state.enabled):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True, "recording_enabled": state.enabled}


@app.put("/api/cameras/{camera_id}/detection")
def set_camera_detection(camera_id: str, state: CameraFeatureState) -> dict:
    if not manager.set_detection(camera_id, state.enabled):
        raise HTTPException(status_code=404, detail="camera not found")
    return {"ok": True, "detection_enabled": state.enabled}


@app.get("/api/cameras/{camera_id}/recordings")
def recordings(camera_id: str, limit: int = 200, source: str = "main") -> list[dict]:
    _require_recording_camera(camera_id)
    rows = _recording_rows(
        camera_id,
        limit=max(1, min(limit, RECORDING_LOOKUP_LIMIT)),
        source=recording_source(source),
    )
    return [_public_recording_row(row) for row in rows]


@app.get("/api/cameras/{camera_id}/recordings/events")
def recording_events(camera_id: str, limit: int = 1000, source: str = "main") -> list[dict]:
    _require_recording_camera(camera_id)
    rows = _recording_rows(camera_id, limit=RECORDING_LOOKUP_LIMIT, source=recording_source(source))
    if not rows:
        return []
    start_epoch = rows[0].get("start_epoch")
    end_epoch = rows[-1].get("end_epoch")
    if start_epoch is None or end_epoch is None:
        return []

    events = manager.events.for_camera_range(
        camera_id,
        datetime.fromtimestamp(float(start_epoch), timezone.utc).isoformat(),
        datetime.fromtimestamp(float(end_epoch), timezone.utc).isoformat(),
        limit=max(1, min(limit, 5000)),
    )
    return [_recording_event_row(event, rows) for event in events]


@app.get("/api/cameras/{camera_id}/recordings/day")
def recording_day(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
) -> dict:
    _require_recording_camera(camera_id)
    _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
    selected_source = recording_source(source)
    source_availability = {
        candidate: manager.recorder.recording_availability_between(
            camera_id,
            start_epoch,
            end_epoch,
            candidate,
            discover_missing=False,
        )
        for candidate in ("main", "live")
    }
    availability = source_availability[selected_source]
    available_sources = [
        candidate for candidate in ("main", "live")
        if int(source_availability[candidate]["segment_count"]) > 0
    ]
    events = manager.events.for_camera_range(
        camera_id,
        datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
        datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
        limit=5000,
    )
    public_events = [_event_row(event) for event in events]
    return {
        "camera_id": camera_id,
        "source": selected_source,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "recordings": availability["ranges"],
        "availability": availability["ranges"],
        "recording_count": availability["segment_count"],
        "events": public_events,
        "incidents": [
            _incident_list_payload(incident)
            for incident in _incident_rows(public_events)
        ],
        "available_sources": available_sources,
    }


def _public_media_export(job: dict[str, object]) -> dict[str, object]:
    payload = dict(job)
    for key in ("download_url", "media_url"):
        if payload.get(key):
            payload[key] = public_url(str(payload[key]))
    return payload


@app.post("/api/exports", status_code=202)
def create_media_export(request: MediaExportRequest) -> dict[str, object]:
    _require_recording_camera(request.camera_id)
    maximum = 24 * 60 * 60 if request.kind == "recording" else 7 * 24 * 60 * 60
    _validate_recording_range(
        request.start_epoch,
        request.end_epoch,
        maximum,
        f"invalid {request.kind} export range",
    )
    options: dict[str, object] = {}
    if request.kind == "recording":
        options = {"height": request.height}
    elif request.height > 0:
        options = {
            "sample_interval_seconds": request.sample_interval_seconds,
            "output_fps": request.output_fps,
            "height": request.height,
        }
    else:
        options = {
            "sample_interval_seconds": request.sample_interval_seconds,
            "output_fps": request.output_fps,
            "width": request.width,
        }
    try:
        job = _media_export_manager().create({
            "kind": request.kind,
            "camera_id": request.camera_id,
            "source": recording_source(request.source),
            "start_epoch": request.start_epoch,
            "end_epoch": request.end_epoch,
            "options": options,
            "label": request.label.strip(),
            "origin": "manual",
        })
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return _public_media_export(job)


@app.get("/api/exports")
def list_media_exports(
    limit: int = 50,
    offset: int = 0,
    camera_id: str = "",
    kind: str = "",
    status: str = "",
    protected: bool | None = None,
) -> dict[str, object]:
    if kind and kind not in {"recording", "timelapse"}:
        raise HTTPException(status_code=400, detail="invalid export kind")
    if status and status not in {
        "queued", "running", "cancelling", "completed", "failed", "cancelled",
        "active", "terminal",
    }:
        raise HTTPException(status_code=400, detail="invalid export status")
    export_manager = _media_export_manager()
    filters = {
        "camera_id": camera_id,
        "kind": kind,
        "status": status,
        "protected": protected,
    }
    jobs = export_manager.list(
        max(1, min(limit, 1000)),
        offset=max(0, offset),
        **filters,
    )
    return {
        "exports": [_public_media_export(job) for job in jobs],
        "total": export_manager.count(**filters),
        "offset": max(0, offset),
        "limit": max(1, min(limit, 1000)),
    }


@app.get("/api/exports/summary")
def media_export_summary() -> dict[str, object]:
    return _media_export_manager().summary()


@app.post("/api/exports/batch")
def batch_media_exports(request: MediaExportBatchRequest) -> dict[str, object]:
    return _public_media_export_batch(
        _media_export_manager().batch(request.ids, request.action)
    )


def _public_media_export_batch(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["results"] = [
        _public_media_export(job)
        for job in list(payload.get("results") or [])
    ]
    return result


@app.get("/api/exports/{job_id}")
def get_media_export(job_id: str) -> dict[str, object]:
    job = _media_export_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="export not found")
    return _public_media_export(job)


@app.get("/api/exports/{job_id}/download")
def download_media_export(job_id: str) -> FileResponse:
    try:
        path, name = _media_export_manager().output_path(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="completed export not found") from None
    return FileResponse(
        path,
        filename=name,
        # New recording exports are always MP4. Keep the legacy ZIP media type
        # until any already-completed archive jobs expire from the export store.
        media_type="application/zip" if path.suffix.lower() == ".zip" else "video/mp4",
        headers={"Cache-Control": "private, no-store"},
    )


@app.get("/api/exports/{job_id}/media")
def play_media_export(job_id: str) -> FileResponse:
    try:
        path, _name = _media_export_manager().output_path(job_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="completed export not found") from None
    return FileResponse(
        path,
        media_type="application/zip" if path.suffix.lower() == ".zip" else "video/mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.patch("/api/exports/{job_id}/protection")
def protect_media_export(
    job_id: str,
    request: MediaExportProtectionRequest,
) -> dict[str, object]:
    try:
        return _public_media_export(
            _media_export_manager().set_protected(job_id, request.protected)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="export not found") from None


@app.patch("/api/exports/{job_id}/metadata")
def update_media_export_metadata(
    job_id: str,
    request: MediaExportMetadataRequest,
) -> dict[str, object]:
    try:
        return _public_media_export(
            _media_export_manager().set_label(job_id, request.label)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="export not found") from None


@app.delete("/api/exports/{job_id}", status_code=202)
def delete_media_export(job_id: str, force: bool = False) -> dict[str, object]:
    try:
        return _public_media_export(
            _media_export_manager().cancel_or_delete(job_id, force=force)
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="export not found") from None
    except PermissionError:
        raise HTTPException(
            status_code=409,
            detail="protected exports require explicit forced deletion",
        ) from None


@app.get("/api/cameras/{camera_id}/recordings/window")
def recording_window(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
) -> dict:
    _require_recording_camera(camera_id)
    _validate_recording_range(start_epoch, end_epoch, 3600, "invalid recording window range")
    selected_source = recording_source(source)
    window_start, window_end = _recording_playback_window(start_epoch)
    rows = _recording_day_rows(camera_id, window_start, window_end, selected_source)
    return {
        "camera_id": camera_id,
        "source": selected_source,
        "start_epoch": window_start,
        "end_epoch": window_end,
        "recordings": [_public_recording_row(row) for row in rows],
    }


@app.get("/api/cameras/{camera_id}/recordings/preview.jpg")
def recording_preview(camera_id: str, epoch: float, source: str = "main") -> FileResponse:
    _require_recording_camera(camera_id)
    if not math.isfinite(epoch) or epoch <= 0:
        raise HTTPException(status_code=400, detail="invalid recording preview time")
    selected_source = recording_source(source)
    rows = manager.recorder.recording_rows_between(
        camera_id,
        epoch - 0.001,
        epoch + 0.001,
        selected_source,
        discover_missing=False,
    )
    row = next(
        (
            candidate for candidate in rows
            if float(candidate.get("start_epoch") or 0) <= epoch
            < float(candidate.get("end_epoch") or 0)
        ),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="no recording exists at this time")
    preview_path = _recording_preview_path(row, epoch)
    return FileResponse(
        preview_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/cameras/{camera_id}/recordings/updates")
def recording_updates(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    after_epoch: float,
    source: str = "main",
) -> dict:
    _require_recording_camera(camera_id)
    _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
    if not math.isfinite(after_epoch):
        raise HTTPException(status_code=400, detail="invalid recording update position")
    selected_source = recording_source(source)
    overlap_seconds = max(
        5.0,
        float(config.recording_segment_seconds) * 2,
        float(DEFAULT_INCIDENT_GAP_SECONDS) * 2,
    )
    update_start = max(start_epoch, min(end_epoch, after_epoch) - overlap_seconds)
    # Object analysis and tracking can finish after the recording edge moves.
    # Re-read a wider event window so late-persisted incidents still appear in
    # an already-open recording page without widening the recording-index scan.
    event_update_start = max(
        start_epoch,
        min(end_epoch, after_epoch) - max(overlap_seconds, 5 * 60.0),
    )
    # Keep NFS directory enumeration off the request thread. The index worker
    # services this wake-up immediately and the next lightweight update poll
    # observes newly finalized segments.
    manager.recorder.request_recording_edge_refresh(
        camera_id,
        selected_source,
        after_epoch,
    )
    availability = manager.recorder.recording_availability_between(
        camera_id,
        update_start,
        end_epoch,
        selected_source,
        discover_missing=False,
    )
    events = manager.events.for_camera_range(
        camera_id,
        datetime.fromtimestamp(event_update_start, timezone.utc).isoformat(),
        datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
        limit=1000,
    )
    public_events = [_event_row(event) for event in events]
    return {
        "camera_id": camera_id,
        "source": selected_source,
        "start_epoch": update_start,
        "end_epoch": end_epoch,
        "availability": availability["ranges"],
        "events": public_events,
        "incidents": [
            _incident_list_payload(incident)
            for incident in _incident_rows(public_events)
        ],
    }


def _recording_day_rows(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str,
    *,
    fresh: bool = False,
) -> list[dict]:
    selected_source = recording_source(source)
    cache_key = (camera_id, selected_source, int(start_epoch), int(end_epoch))
    now = time.monotonic()
    if not fresh:
        with RECORDING_DAY_CACHE_LOCK:
            cached = RECORDING_DAY_CACHE.get(cache_key)
            if cached is not None and now - cached[0] < RECORDING_DAY_CACHE_SECONDS:
                manager.recorder.lease_recordings_for_playback(cached[1])
                return cached[1]
    rows = [
        row for row in manager.recorder.recording_rows_between(
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
            discover_missing=False,
        )
        if int(row.get("size_bytes") or 0) > 1024
    ]
    if fresh:
        rows = manager.recorder.discard_missing_recording_rows(rows)
    manager.recorder.lease_recordings_for_playback(rows)
    manager.recorder.queue_stream_fingerprints(rows)
    with RECORDING_DAY_CACHE_LOCK:
        RECORDING_DAY_CACHE[cache_key] = (now, rows)
        expired = [key for key, value in RECORDING_DAY_CACHE.items() if now - value[0] >= RECORDING_DAY_CACHE_SECONDS]
        for key in expired:
            RECORDING_DAY_CACHE.pop(key, None)
    return rows


def _recording_playback_window(epoch: float) -> tuple[float, float]:
    start = math.floor(epoch / RECORDING_PLAYBACK_WINDOW_SECONDS) * RECORDING_PLAYBACK_WINDOW_SECONDS
    return start, start + RECORDING_PLAYBACK_WINDOW_SECONDS


def _recording_cache_metric(origin: str, metric: str, value: float = 1.0) -> None:
    key = f"{origin}_{metric}"
    with RECORDING_CACHE_METRICS_LOCK:
        RECORDING_CACHE_METRICS[key] = float(RECORDING_CACHE_METRICS.get(key, 0.0)) + value


def _signal_recording_prewarm_process(process: subprocess.Popen, sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _run_recording_remux(command: list[str], origin: str) -> subprocess.CompletedProcess:
    global RECORDING_PREWARM_PROCESS
    if origin != "prewarm":
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with RECORDING_PREWARM_PROCESS_LOCK:
        RECORDING_PREWARM_PROCESS = process
    terminate_at: float | None = None
    timeout_at = time.monotonic() + 30.0
    try:
        while True:
            if not RECORDING_PREWARM_STOP.is_set() and time.monotonic() >= timeout_at:
                _signal_recording_prewarm_process(process, signal.SIGTERM)
                try:
                    _stdout, stderr = process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    _signal_recording_prewarm_process(process, signal.SIGKILL)
                    _stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(command, 30, stderr=stderr)
            if RECORDING_PREWARM_STOP.is_set() and process.poll() is None:
                if terminate_at is None:
                    _signal_recording_prewarm_process(process, signal.SIGTERM)
                    terminate_at = time.monotonic() + 3.0
                elif time.monotonic() >= terminate_at:
                    _signal_recording_prewarm_process(process, signal.SIGKILL)
            try:
                _stdout, stderr = process.communicate(timeout=0.25)
                if RECORDING_PREWARM_STOP.is_set():
                    raise RecordingPrewarmCancelled
                return subprocess.CompletedProcess(command, process.returncode, None, stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        with RECORDING_PREWARM_PROCESS_LOCK:
            if RECORDING_PREWARM_PROCESS is process:
                RECORDING_PREWARM_PROCESS = None


def _recording_fmp4_files(
    path: Path,
    duration: float,
    media_offset: float,
    origin: str = "playback",
) -> tuple[Path, Path]:
    stat = path.stat()
    fingerprint = f"v3:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{duration:.3f}:{media_offset:.3f}"
    cache_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    cache_dir = manager.storage_dir / "playback-cache" / "fmp4" / cache_key
    init_path = cache_dir / "init.mp4"
    media_path = cache_dir / "media.m4s"
    if _recording_cache_files_ready(init_path, media_path, touch=True):
        _recording_cache_metric(origin, "hits")
        return init_path, media_path

    with RECORDING_FMP4_LOCKS_GUARD:
        lock = RECORDING_FMP4_LOCKS.setdefault(cache_key, threading.Lock())
    with lock:
        if _recording_cache_files_ready(init_path, media_path, touch=True):
            _recording_cache_metric(origin, "hits")
            return init_path, media_path
        _recording_cache_metric(origin, "misses")
        remux_started = time.monotonic()
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="fmp4-", dir=cache_dir))
        codec = _probe_video_codec(path)
        command = [
            config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-output_ts_offset",
            f"{media_offset:.3f}",
        ]
        if codec in {"hevc", "h265"}:
            command.extend(["-tag:v", "hvc1"])
        command.extend([
            "-f",
            "hls",
            "-hls_time",
            "300",
            "-hls_list_size",
            "0",
            "-hls_segment_type",
            "fmp4",
            "-hls_fmp4_init_filename",
            "init.mp4",
            "-hls_segment_filename",
            str(temp_dir / "media_%d.m4s"),
            str(temp_dir / "index.m3u8"),
        ])
        try:
            result = _run_recording_remux(command, origin)
        except RecordingPrewarmCancelled:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except subprocess.TimeoutExpired as exc:
            _recording_cache_metric(origin, "failures")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=504, detail="recording fragment remux timed out") from exc
        except OSError as exc:
            _recording_cache_metric(origin, "failures")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"recording fragment remux failed: {exc}") from exc
        generated_init = temp_dir / "init.mp4"
        generated_media = temp_dir / "media_0.m4s"
        if result.returncode != 0 or not generated_init.exists() or not generated_media.exists():
            _recording_cache_metric(origin, "failures")
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            shutil.rmtree(temp_dir, ignore_errors=True)
            if time.time() - stat.st_mtime >= float(config.recording_segment_seconds) * 2:
                manager.recorder.schedule_revalidation(path, error or "recording fragment failed")
            with RECORDING_DAY_CACHE_LOCK:
                RECORDING_DAY_CACHE.clear()
            raise HTTPException(status_code=500, detail=f"recording fragment failed: {error[-300:]}")
        try:
            _offset_fmp4_timestamps(generated_init, generated_media, media_offset)
        except Exception as exc:
            _recording_cache_metric(origin, "failures")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"recording fragment timestamp repair failed: {exc}") from exc
        try:
            os.replace(generated_init, init_path)
            os.replace(generated_media, media_path)
        except OSError as exc:
            _recording_cache_metric(origin, "failures")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"recording fragment cache write failed: {exc}") from exc
        remux_ms = (time.monotonic() - remux_started) * 1000
        _recording_cache_metric(origin, "remuxes")
        _recording_cache_metric(origin, "remux_ms", remux_ms)
        with RECORDING_CACHE_METRICS_LOCK:
            RECORDING_CACHE_METRICS[f"{origin}_last_remux_ms"] = remux_ms
        shutil.rmtree(temp_dir, ignore_errors=True)
        _maintain_recording_cache(cache_dir)
        return init_path, media_path


def _recording_cache_files_ready(init_path: Path, media_path: Path, *, touch: bool = False) -> bool:
    try:
        ready = init_path.is_file() and init_path.stat().st_size > 0 and media_path.is_file() and media_path.stat().st_size > 0
        if ready and touch:
            now = time.time()
            os.utime(init_path, (now, now))
            os.utime(media_path, (now, now))
        return ready
    except OSError:
        return False


def _recording_file_response(path: Path, media_type: str) -> FileResponse:
    try:
        now = time.time()
        os.utime(path, (now, now))
    except OSError:
        raise HTTPException(status_code=404, detail="recording fragment cache entry disappeared")
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


def _recording_preview_path(row: dict, epoch: float) -> Path:
    """Return a small cached JPEG near an epoch without mutating playback."""
    source_path = _recording_storage_path(row.get("path"))
    start_epoch = float(row.get("start_epoch") or 0)
    end_epoch = float(row.get("end_epoch") or start_epoch)
    if not start_epoch <= epoch < end_epoch:
        raise HTTPException(status_code=404, detail="no recording exists at this time")
    duration = max(0.05, end_epoch - start_epoch)
    raw_offset = max(0.0, epoch - start_epoch)
    preview_offset = min(
        max(0.0, math.floor(raw_offset / RECORDING_PREVIEW_INTERVAL_SECONDS) * RECORDING_PREVIEW_INTERVAL_SECONDS),
        max(0.0, duration - 0.05),
    )
    try:
        stat = source_path.stat()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="recording file not found") from exc
    fingerprint = (
        f"v1:{source_path}:{stat.st_mtime_ns}:{stat.st_size}:"
        f"{preview_offset:.3f}:480"
    )
    cache_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
    cache_dir = manager.database_dir / "recording-preview-cache"
    preview_path = cache_dir / f"{cache_key}.jpg"
    if _recording_preview_ready(preview_path, touch=True):
        return preview_path

    with RECORDING_PREVIEW_LOCKS_GUARD:
        lock = RECORDING_PREVIEW_LOCKS.setdefault(cache_key, threading.Lock())
    with lock:
        if _recording_preview_ready(preview_path, touch=True):
            return preview_path
        if not RECORDING_PREVIEW_BUILD_LIMITER.acquire(timeout=3.0):
            raise HTTPException(
                status_code=429,
                detail="recording preview generator is busy",
                headers={"Retry-After": "1"},
            )
        temporary = cache_dir / f".{cache_key}.{os.getpid()}.{threading.get_ident()}.tmp.jpg"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            command = [
                config.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{preview_offset:.3f}",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-vf",
                "scale='min(480,iw)':-2",
                "-q:v",
                "5",
                "-y",
                str(temporary),
            ]
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=8,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail="recording preview timed out") from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail="recording preview failed") from exc
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
                error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
                LOGGER.warning(
                    "recording preview failed for %s at %.3f: %s",
                    source_path.name,
                    preview_offset,
                    redact_secret_text(error[-300:]),
                )
                raise HTTPException(status_code=500, detail="recording preview failed")
            os.replace(temporary, preview_path)
        finally:
            temporary.unlink(missing_ok=True)
            RECORDING_PREVIEW_BUILD_LIMITER.release()
        _maintain_recording_preview_cache(preview_path)
        return preview_path


def _recording_preview_ready(path: Path, *, touch: bool = False) -> bool:
    try:
        ready = path.is_file() and path.stat().st_size > 0
        if ready and touch:
            now = time.time()
            os.utime(path, (now, now))
        return ready
    except OSError:
        return False


def _maintain_recording_preview_cache(active_path: Path) -> None:
    global RECORDING_PREVIEW_LAST_MAINTENANCE
    now_monotonic = time.monotonic()
    if now_monotonic - RECORDING_PREVIEW_LAST_MAINTENANCE < 600:
        return
    if not RECORDING_PREVIEW_MAINTENANCE_LOCK.acquire(blocking=False):
        return
    try:
        RECORDING_PREVIEW_LAST_MAINTENANCE = now_monotonic
        cache_dir = active_path.parent
        now_epoch = time.time()
        entries: list[tuple[float, int, Path]] = []
        for path in cache_dir.glob("*.jpg"):
            if path == active_path:
                continue
            try:
                stat = path.stat()
                if now_epoch - stat.st_mtime > RECORDING_PREVIEW_MAX_AGE_SECONDS:
                    path.unlink(missing_ok=True)
                else:
                    entries.append((stat.st_mtime, stat.st_size, path))
            except OSError:
                continue
        try:
            active_size = active_path.stat().st_size
        except OSError:
            active_size = 0
        total_size = sum(size for _, size, _ in entries) + active_size
        for _modified_at, size, path in sorted(entries):
            if total_size <= RECORDING_PREVIEW_MAX_BYTES:
                break
            try:
                path.unlink(missing_ok=True)
                total_size -= size
            except OSError:
                continue
    finally:
        RECORDING_PREVIEW_MAINTENANCE_LOCK.release()


def _maintain_recording_cache(active_dir: Path) -> None:
    global RECORDING_CACHE_LAST_MAINTENANCE
    now_monotonic = time.monotonic()
    if now_monotonic - RECORDING_CACHE_LAST_MAINTENANCE < 600:
        return
    if not RECORDING_CACHE_MAINTENANCE_LOCK.acquire(blocking=False):
        return
    try:
        RECORDING_CACHE_LAST_MAINTENANCE = now_monotonic
        root = manager.storage_dir / "playback-cache" / "fmp4"
        if not root.exists():
            return
        now_epoch = time.time()
        entries: list[tuple[float, int, Path]] = []
        for directory in root.iterdir():
            if not directory.is_dir() or directory == active_dir:
                continue
            with RECORDING_FMP4_LOCKS_GUARD:
                entry_lock = RECORDING_FMP4_LOCKS.setdefault(directory.name, threading.Lock())
            if not entry_lock.acquire(blocking=False):
                continue
            try:
                for child in directory.iterdir():
                    if child.is_dir() and child.name.startswith("fmp4-"):
                        try:
                            if now_epoch - child.stat().st_mtime > 300:
                                shutil.rmtree(child, ignore_errors=True)
                        except OSError:
                            continue
                files = [item for item in directory.iterdir() if item.is_file()]
                modified_at = max((item.stat().st_mtime for item in files), default=directory.stat().st_mtime)
                size = sum(item.stat().st_size for item in files)
                max_age_seconds = int(config.recording_cache_max_days) * 24 * 60 * 60
                if now_epoch - modified_at > max_age_seconds:
                    shutil.rmtree(directory, ignore_errors=True)
                else:
                    entries.append((modified_at, size, directory))
            except OSError:
                continue
            finally:
                entry_lock.release()
        total_size = sum(size for _, size, _ in entries)
        max_bytes = int(float(config.recording_cache_max_gb) * 1024 * 1024 * 1024)
        for modified_at, size, directory in sorted(entries):
            if total_size <= max_bytes:
                break
            if now_epoch - modified_at < 300:
                continue
            with RECORDING_FMP4_LOCKS_GUARD:
                entry_lock = RECORDING_FMP4_LOCKS.setdefault(directory.name, threading.Lock())
            if not entry_lock.acquire(blocking=False):
                continue
            try:
                shutil.rmtree(directory, ignore_errors=True)
                total_size -= size
            finally:
                entry_lock.release()
    finally:
        RECORDING_CACHE_MAINTENANCE_LOCK.release()


def _start_recording_prewarmer() -> None:
    global RECORDING_PREWARM_THREAD
    if RECORDING_PREWARM_THREAD is not None and RECORDING_PREWARM_THREAD.is_alive():
        return
    RECORDING_PREWARM_STOP.clear()
    thread = threading.Thread(
        target=_recording_prewarm_loop,
        name="recording-prewarmer",
        daemon=False,
    )
    RECORDING_PREWARM_THREAD = thread
    try:
        thread.start()
    except BaseException:
        RECORDING_PREWARM_THREAD = None
        RECORDING_PREWARM_STOP.set()
        raise


def _stop_recording_prewarmer() -> None:
    global RECORDING_PREWARM_THREAD
    logger = logging.getLogger(__name__)
    RECORDING_PREWARM_STOP.set()
    with RECORDING_PREWARM_PROCESS_LOCK:
        process = RECORDING_PREWARM_PROCESS
    if process is not None:
        _signal_recording_prewarm_process(process, signal.SIGTERM)
    if RECORDING_PREWARM_THREAD is not None:
        RECORDING_PREWARM_THREAD.join(timeout=5)
        if RECORDING_PREWARM_THREAD.is_alive():
            with RECORDING_PREWARM_PROCESS_LOCK:
                process = RECORDING_PREWARM_PROCESS
            if process is not None:
                _signal_recording_prewarm_process(process, signal.SIGKILL)
            RECORDING_PREWARM_THREAD.join(timeout=3)
        if RECORDING_PREWARM_THREAD.is_alive():
            logger.error("recording prewarmer did not stop after cancellation")
            raise RuntimeError("recording prewarmer did not stop after cancellation")
        else:
            RECORDING_PREWARM_THREAD = None


def _recording_prewarm_loop() -> None:
    while not RECORDING_PREWARM_STOP.wait(5):
        if not config.recording_cache_prewarm:
            continue
        for camera in config.cameras:
            sources = []
            if camera.record:
                sources.append("main")
            if camera.record_sub and camera.live_stream_url:
                sources.append("live")
            for source in sources:
                if RECORDING_PREWARM_STOP.is_set():
                    return
                try:
                    row = manager.recorder.latest_indexed_row(camera.id, source)
                    if row is None:
                        continue
                    path = Path(row["path"])
                    if not path.exists() or time.time() - path.stat().st_mtime < float(config.recording_segment_seconds) * 2:
                        continue
                    window_start, window_end = _recording_playback_window(float(row["start_epoch"]))
                    rows = manager.recorder.recording_rows_between(
                        camera.id,
                        window_start,
                        window_end,
                        source,
                        discover_missing=False,
                    )
                    targets = [rows[0], row] if rows else []
                    warmed: set[str] = set()
                    for target in targets:
                        target_path = str(target["path"])
                        if target_path in warmed:
                            continue
                        index = next((i for i, item in enumerate(rows) if item["path"] == target_path), None)
                        if index is None:
                            continue
                        warmed.add(target_path)
                        media_offset = sum(float(item["duration_seconds"]) for item in rows[:index])
                        _recording_fmp4_files(
                            Path(target_path),
                            float(target["duration_seconds"]),
                            media_offset,
                            origin="prewarm",
                        )
                except RecordingPrewarmCancelled:
                    return
                except Exception:
                    logging.getLogger(__name__).exception("Recording prewarm failed for %s/%s", camera.id, source)


@app.get("/api/cameras/{camera_id}/recordings/day.m3u8")
def recording_day_hls_playlist(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
) -> Response:
    _require_recording_camera(camera_id)
    _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
    selected_source = recording_source(source)
    rows = _recording_day_rows(
        camera_id,
        start_epoch,
        end_epoch,
        selected_source,
        fresh=True,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="no recordings found")
    target_duration = max(1, math.ceil(max(float(row["duration_seconds"]) for row in rows)))
    query = f"start_epoch={start_epoch:.3f}&end_epoch={end_epoch:.3f}&source={selected_source}"
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    media_offset = 0.0
    previous_fingerprint: str | None = None
    fingerprints = resolve_stream_fingerprints([row.get("stream_fingerprint") for row in rows])
    for row, stream_fingerprint in zip(rows, fingerprints):
        row_start = float(row["start_epoch"])
        segment_name = quote(str(row["name"]), safe="")
        segment_query = f"{query}&media_offset={media_offset:.3f}"
        map_lines, previous_fingerprint = hls_map_transition(
            previous_fingerprint,
            stream_fingerprint,
            f"day/segment/{segment_name}/init.mp4?{segment_query}",
        )
        lines.extend(map_lines)
        lines.extend([
            f"#EXT-X-PROGRAM-DATE-TIME:{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
            f"#EXTINF:{float(row['duration_seconds']):.3f},",
            f"day/segment/{segment_name}/media.m4s?{segment_query}",
        ])
        media_offset += float(row["duration_seconds"])
    lines.append("#EXT-X-ENDLIST")
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


def _recording_day_fmp4_paths(
    camera_id: str,
    segment_name: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
    trim_end: bool = False,
) -> tuple[Path, Path]:
    _require_recording_camera(camera_id)
    _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
    if not math.isfinite(media_offset) or media_offset < 0:
        raise HTTPException(status_code=400, detail="invalid recording media offset")
    rows = _recording_day_rows(camera_id, start_epoch, end_epoch, source)
    if not segment_name or Path(segment_name).name != segment_name:
        raise HTTPException(status_code=404, detail="recording segment not found")
    segment_index = next(
        (index for index, row in enumerate(rows) if str(row.get("name") or "") == segment_name),
        None,
    )
    if segment_index is None:
        raise HTTPException(status_code=404, detail="recording segment not found")
    row = rows[segment_index]
    path = _recording_storage_path(row.get("path"))
    segment_duration = playback_segment_duration(
        float(row["start_epoch"]),
        float(row["duration_seconds"]),
        end_epoch,
        trim_end,
    )
    expected_offset = sum(float(row["duration_seconds"]) for row in rows[:segment_index])
    if abs(media_offset - expected_offset) > 0.1:
        media_offset = expected_offset
    return _recording_fmp4_files(path, segment_duration, media_offset)


@app.get("/api/cameras/{camera_id}/recordings/day/segment/{segment_name}/init.mp4")
def recording_day_hls_init(
    camera_id: str,
    segment_name: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
    trim_end: bool = False,
) -> FileResponse:
    init_path, _ = _recording_day_fmp4_paths(
        camera_id, segment_name, start_epoch, end_epoch, source, media_offset, trim_end
    )
    return _recording_file_response(init_path, "video/mp4")


@app.get("/api/cameras/{camera_id}/recordings/day/segment/{segment_name}/media.m4s")
def recording_day_hls_segment(
    camera_id: str,
    segment_name: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
    trim_end: bool = False,
) -> FileResponse:
    _, media_path = _recording_day_fmp4_paths(
        camera_id, segment_name, start_epoch, end_epoch, source, media_offset, trim_end
    )
    return _recording_file_response(media_path, "video/iso.segment")




@app.get("/api/events/{event_id}/clip.mp4")
def event_clip(event_id: int, before: float | None = None, after: float | None = None, source: str = "main") -> FileResponse:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    enriched = _event_row(event)
    before_seconds, after_seconds = _event_clip_window(before, after)
    clip_source = recording_source(source)
    clip_path = _ensure_event_clip(
        enriched,
        before=before_seconds,
        after=after_seconds,
        source=clip_source,
    )
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/events/{event_id}/stream.m3u8")
def event_stream(event_id: int, before: float | None = None, after: float | None = None, source: str = "main") -> Response:
    event = manager.events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    enriched = _event_row(event)
    camera_id = str(enriched.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")
    before_seconds, after_seconds = _event_clip_window(before, after)
    event_created_epoch = event_epoch(enriched)
    window_start = event_created_epoch - before_seconds
    window_end = event_created_epoch + after_seconds
    selected_source = recording_source(source)
    rows = _recording_day_rows(
        camera_id,
        window_start,
        window_end,
        selected_source,
        fresh=True,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="no recording window found")

    first_start = float(rows[0]["start_epoch"])
    start_offset = max(0.0, window_start - first_start)
    clip_durations = [
        playback_segment_duration(
            float(row["start_epoch"]),
            float(row["duration_seconds"]),
            window_end,
            True,
        )
        for row in rows
    ]
    target_duration = max(1, math.ceil(max(clip_durations)))
    query = f"start_epoch={window_start:.3f}&end_epoch={window_end:.3f}&source={selected_source}"
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{target_duration}",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-START:TIME-OFFSET={start_offset:.3f},PRECISE=YES",
    ]
    media_offset = 0.0
    previous_fingerprint: str | None = None
    fingerprints = resolve_stream_fingerprints([row.get("stream_fingerprint") for row in rows])
    for row, stream_fingerprint, clip_duration in zip(rows, fingerprints, clip_durations):
        row_start = float(row["start_epoch"])
        segment_name = quote(str(row["name"]), safe="")
        segment_query = f"{query}&media_offset={media_offset:.3f}&trim_end=true"
        map_uri = public_url(
            f"/api/cameras/{quote(camera_id, safe='')}/recordings/day/segment/{segment_name}/init.mp4?"
            f"{segment_query}"
        )
        map_lines, previous_fingerprint = hls_map_transition(
            previous_fingerprint,
            stream_fingerprint,
            map_uri,
        )
        lines.extend(map_lines)
        lines.extend([
            f"#EXT-X-PROGRAM-DATE-TIME:{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
            f"#EXTINF:{clip_duration:.3f},",
            public_url(f"/api/cameras/{quote(camera_id, safe='')}/recordings/day/segment/{segment_name}/media.m4s?{segment_query}"),
        ])
        media_offset += clip_duration
    lines.append("#EXT-X-ENDLIST")
    return Response(
        "\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "private, max-age=30"},
    )




def _recording_rows(camera_id: str, limit: int, source: str = "main") -> list[dict]:
    return manager.recorder.recording_rows(camera_id, limit=limit, source=recording_source(source))


def _recording_storage_path(value: object) -> Path:
    if not value:
        raise HTTPException(status_code=404, detail="recording file not found")
    try:
        path = Path(str(value)).resolve(strict=True)
        path.relative_to(manager.recorder.recordings_dir.resolve())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="recording file not found") from None
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="recording file is outside storage") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="recording file not found")
    return path


def _public_recording_row(row: dict) -> dict:
    payload = dict(row)
    payload.pop("path", None)
    return payload


def _event_row(row: dict) -> dict:
    event = dict(row)
    event["snapshot_path"] = "available" if event.get("snapshot_path") else ""
    event["recording_path"] = "available" if event.get("recording_path") else ""
    try:
        objects = json.loads(event.pop("objects_json", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        objects = []
    if not isinstance(objects, list):
        objects = []
    objects = [item for item in objects if isinstance(item, dict)]
    qualification_entry = next(
        (
            item.get("motion_qualification")
            for item in reversed(objects)
            if item.get("status") == "motion_qualification"
            and isinstance(item.get("motion_qualification"), dict)
        ),
        None,
    )
    raw_trigger_source = str(
        (qualification_entry or {}).get("trigger_source")
        or event.get("topic")
        or "camera"
    ).lower()
    event["trigger_source"] = (
        "ema"
        if raw_trigger_source in {"adaptive", "visual_backup", "adaptive/visual_backup"}
        else "camera"
    )
    tracking_entry = next(
        (
            item.get("object_tracking")
            for item in reversed(objects)
            if item.get("status") == "object_tracking"
            and isinstance(item.get("object_tracking"), dict)
        ),
        None,
    )
    event["object_tracking"] = tracking_entry

    def positive_confidence(item: dict) -> bool:
        try:
            confidence = float(item.get("confidence") or 0)
            return math.isfinite(confidence) and confidence > 0
        except (TypeError, ValueError):
            return False

    detected_objects = [
        item for item in objects
        if item.get("label")
        and positive_confidence(item)
        and item.get("incident_eligible") is not False
    ]
    tracked_objects = (
        [item for item in tracking_entry.get("tracks", []) if isinstance(item, dict)]
        if isinstance(tracking_entry, dict)
        else []
    )
    event["objects"] = objects
    event["has_objects"] = bool(detected_objects)
    event["labels"] = sorted({
        str(item["label"])
        for item in [*detected_objects, *tracked_objects]
        if item.get("label")
    })
    event["zones"] = sorted({
        str(zone_name)
        for item in [*detected_objects, *tracked_objects]
        for zone_name in (
            item.get("zones", [])
            if isinstance(item.get("zones", []), list)
            else []
        )
        if zone_name
    })
    return event


def _best_incident_event(events: list[dict]) -> dict:
    object_events = [event for event in events if event.get("has_objects")]
    candidates = object_events or events

    def score(event: dict) -> tuple[float, int]:
        objects = event.get("objects") or []
        confidences: list[float] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(confidence):
                confidences.append(confidence)
        best_confidence = max(confidences, default=0.0)
        return (best_confidence, int(event.get("id") or 0))

    return max(candidates, key=score)


def _incident_rows(rows: list[dict], gap_seconds: int = DEFAULT_INCIDENT_GAP_SECONDS) -> list[dict]:
    return [
        _incident_row(camera_id, events)
        for camera_id, events in incident_event_groups(rows, gap_seconds)
    ]


def _recent_incident_summaries(limit: int, gap_seconds: int) -> list[dict]:
    batch_size = max(500, min(5000, limit * 8))
    compact_rows: list[dict] = []
    before_created_at: str | None = None
    before_id: int | None = None

    while True:
        batch = manager.events.recent_compact(batch_size, before_created_at, before_id)
        if not batch:
            return _incident_rows(compact_rows, gap_seconds)[:limit]
        compact_rows.extend(_event_row(row) for row in batch)
        summaries = _incident_rows(compact_rows, gap_seconds)
        if len(summaries) > limit or len(batch) < batch_size:
            return summaries[:limit]
        oldest = batch[-1]
        before_created_at = str(oldest["created_at"])
        before_id = int(oldest["id"])


def _recent_filtered_incident_summaries(
    *,
    limit: int,
    offset: int,
    gap_seconds: int,
    event_type: str,
    camera_id: str = "",
    object_label: str = "",
    zone: str = "",
) -> tuple[list[dict], bool, list[dict]]:
    desired = offset + limit + 1
    compact_rows: list[dict] = []
    before_created_at: str | None = None
    before_id: int | None = None
    # Most first-page filters are satisfied by a few hundred recent rows. A
    # fixed 5,000-row batch made every toggle deserialize and regroup far more
    # history than it displayed. Grow naturally through the cursor loop only
    # when a selective camera/object/zone filter actually needs older events.
    batch_size = max(500, min(5000, desired * 16))

    while True:
        batch = manager.events.recent_compact(batch_size, before_created_at, before_id)
        if not batch:
            summaries = _incident_rows(compact_rows, gap_seconds)
            filtered = _filter_incident_summaries(
                summaries, event_type, camera_id, object_label, zone
            )
            return filtered[offset:offset + limit], False, summaries
        compact_rows.extend(_event_row(row) for row in batch)
        summaries = _incident_rows(compact_rows, gap_seconds)
        filtered = _filter_incident_summaries(
            summaries, event_type, camera_id, object_label, zone
        )
        if len(filtered) >= desired:
            return filtered[offset:offset + limit], True, summaries
        if len(batch) < batch_size:
            return filtered[offset:offset + limit], False, summaries
        oldest = batch[-1]
        before_created_at = str(oldest["created_at"])
        before_id = int(oldest["id"])


def _hydrate_incidents(
    summaries: list[dict],
    active_manager: AppManager | None = None,
) -> list[dict]:
    selected_manager = active_manager or manager
    event_ids = [
        int(event["id"])
        for summary in summaries
        for event in summary.get("events", [])
        if str(event.get("id", "")).isdigit()
    ]
    full_events = {
        int(event["id"]): _event_row(event)
        for event in selected_manager.events.get_many(event_ids)
    }
    observations_by_event: dict[int, list[dict]] = {}
    for audit in selected_manager.events.motion_audits_for_related_events(event_ids):
        related_event_id = int(audit.get("related_event_id") or 0)
        observations_by_event.setdefault(related_event_id, []).append(
            _motion_audit_row(audit, selected_manager.storage_dir)
        )
    for event_id, event in full_events.items():
        event["motion_observations"] = observations_by_event.get(event_id, [])
    hydrated: list[dict] = []
    for summary in summaries:
        events = [
            full_events[int(event["id"])]
            for event in summary.get("events", [])
            if int(event.get("id") or 0) in full_events
        ]
        if events:
            hydrated.append(_incident_row(str(summary.get("camera_id") or ""), events))
    return hydrated


def _incidents_with_faces(
    incidents: list[dict],
    active_manager: AppManager | None = None,
) -> list[dict]:
    selected_manager = active_manager or manager
    event_ids = [
        int(event["id"])
        for incident in incidents
        for event in incident.get("events", [])
        if str(event.get("id", "")).isdigit()
    ]
    observations_by_event: dict[int, list[dict]] = {}
    for observation in selected_manager.faces.for_event_ids(event_ids):
        observations_by_event.setdefault(int(observation["event_id"]), []).append(observation)

    status_rank = {"confirmed": 0, "possible": 1, "unknown": 2}

    def summarize(observations: list[dict]) -> list[dict]:
        summaries: dict[tuple[str, int], dict] = {}
        for observation in observations:
            person_id = observation.get("person_id")
            candidate_id = observation.get("candidate_person_id")
            if person_id is not None:
                status = "confirmed"
                identity_id = int(person_id)
                name = str(observation.get("person_name") or "Unknown")
                confidence = observation.get("match_confidence")
            elif candidate_id is not None:
                status = "possible"
                identity_id = int(candidate_id)
                name = str(observation.get("candidate_person_name") or "Unknown")
                confidence = observation.get("candidate_confidence")
            else:
                status = "unknown"
                identity_id = 0
                name = "Unknown"
                confidence = observation.get("candidate_confidence")
                if confidence is None:
                    confidence = observation.get("confidence")
            try:
                score = float(confidence or 0)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            score = max(0.0, min(1.0, score))
            key = (status, identity_id)
            current = summaries.get(key)
            if current is None or score > current["confidence"]:
                summaries[key] = {
                    "observation_id": int(observation["observation_id"]),
                    "identity_id": identity_id,
                    "name": name,
                    "status": status,
                    "confidence": round(score, 4),
                }
        return sorted(
            summaries.values(),
            key=lambda face: (status_rank[face["status"]], -face["confidence"], face["name"].lower()),
        )

    for incident in incidents:
        incident_observations: list[dict] = []
        for event in incident.get("events", []):
            event_observations = observations_by_event.get(int(event.get("id") or 0), [])
            event["faces"] = summarize(event_observations)
            incident_observations.extend(event_observations)
        incident["faces"] = summarize(incident_observations)
    return incidents


def _incident_event_payload(event: dict) -> dict:
    payload = dict(event)
    payload.pop("topic", None)
    payload.pop("message", None)
    payload["objects"] = [
        {
            key: item[key]
            for key in (
                "label",
                "confidence",
                "box",
                "zones",
                "mask_polygon",
                "incident_eligible",
                "temporal_consensus",
                "temporal_sample_offset_seconds",
                "temporal_observations",
                "temporal_track_observations",
                "temporal_incident_observations",
                "temporal_required_observations",
                "temporal_samples",
                "temporal_peak_confidence",
                "temporal_label_votes",
                "temporal_center_displacement_ratio",
                "temporal_center_path_ratio",
                "temporal_first_observation_offset_seconds",
                "temporal_last_observation_offset_seconds",
                "temporal_newly_appeared",
                "motion_correlated",
                "motion_correlation",
                "motion_correlation_threshold",
                "motion_temporal_evidence_available",
                "track_id",
                "track_state",
                "track_observations",
            )
            if key in item
        }
        for item in payload.get("objects", [])
        if isinstance(item, dict) and item.get("label")
    ]
    payload["object_tracking"] = event.get("object_tracking")
    return payload


def _incident_list_payload(incident: dict) -> dict:
    """Return the media-card data without expensive investigation details."""
    payload = dict(incident)
    representative_id = int(payload.get("representative_event_id") or 0)
    payload.pop("object_tracking", None)
    payload.pop("motion_observations", None)
    payload.pop("faces", None)

    def compact_objects(objects: object) -> list[dict]:
        if not isinstance(objects, list):
            return []
        return [
            {
                key: item[key]
                for key in (
                    "label",
                    "confidence",
                    "box",
                    "zones",
                    "mask_polygon",
                    "incident_eligible",
                    "track_id",
                    "track_state",
                )
                if key in item
            }
            for item in objects
            if isinstance(item, dict) and item.get("label")
        ]

    payload["objects"] = compact_objects(payload.get("objects"))
    compact_events: list[dict] = []
    for event in payload.get("events", []):
        event_id = int(event.get("id") or 0)
        compact_event = {
            key: event[key]
            for key in (
                "id",
                "camera_id",
                "kind",
                "created_at",
                "has_objects",
                "labels",
                "zones",
                "trigger_source",
            )
            if key in event
        }
        compact_event["objects"] = (
            compact_objects(event.get("objects")) if event_id == representative_id else []
        )
        compact_events.append(compact_event)
    payload["events"] = compact_events
    return payload


def _incident_row(camera_id: str, events: list[dict]) -> dict:
    ordered = sorted(events, key=event_epoch)
    first = ordered[0]
    last = ordered[-1]
    representative = _best_incident_event(ordered)
    representative_payload = _incident_event_payload(representative)
    labels = sorted({label for event in ordered for label in event.get("labels", [])})
    zones = sorted({zone for event in ordered for zone in event.get("zones", [])})
    start_epoch = event_epoch(first)
    motion_observations = sorted(
        [
            observation
            for event in ordered
            for observation in event.get("motion_observations", [])
            if isinstance(observation, dict)
        ],
        key=event_epoch,
    )
    tracking_updates = [
        {"created_at": str(tracking.get("updated_at") or "")}
        for event in ordered
        if isinstance((tracking := event.get("object_tracking")), dict)
        and tracking.get("updated_at")
        and tracking.get("tracks")
    ]
    final_item = max([last, *motion_observations, *tracking_updates], key=event_epoch)
    last_epoch = event_epoch(final_item)
    object_count = sum(1 for event in ordered if event.get("has_objects"))
    incident = {
        **representative_payload,
        "id": stable_incident_id(camera_id, first.get("id")),
        "incident_id": stable_incident_key(camera_id, first.get("id")),
        "representative_event_id": representative.get("id"),
        "camera_id": camera_id,
        "kind": "motion",
        "created_at": representative.get("created_at"),
        "start_at": first.get("created_at"),
        "end_at": final_item.get("created_at"),
        "start_epoch": start_epoch,
        "last_epoch": last_epoch,
        "duration_seconds": max(0.0, last_epoch - start_epoch),
        "event_count": len(ordered),
        "motion_observation_count": len(motion_observations),
        "object_event_count": object_count,
        # An incident's trigger is the source that opened it, even when later
        # events from another source are grouped into the same incident.
        "trigger_source": first.get("trigger_source", "camera"),
        "has_objects": bool(labels),
        "labels": labels,
        "zones": zones,
        "events": [_incident_event_payload(event) for event in reversed(ordered)],
        "motion_observations": list(reversed(motion_observations)),
        # Top-level incident media and objects come from the representative
        # event, so its tracking metadata must use the same temporal context.
        "object_tracking": representative.get("object_tracking"),
    }
    return incident


def _recording_event_row(event: dict, recordings: list[dict]) -> dict:
    event = _event_row(event)
    created_epoch = event_epoch(event)
    first_start = float(recordings[0]["start_epoch"])
    event["timeline_offset"] = max(0.0, created_epoch - first_start)
    return event


def _event_clip_window(before: float | None, after: float | None) -> tuple[float, float]:
    try:
        return event_clip_window(
            config.event_clip_before_seconds,
            config.event_clip_after_seconds,
            before,
            after,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _event_clip_path(event: dict, before: float, after: float, source: str = "main") -> Path:
    event_id = int(event.get("id") or 0)
    camera_id = slugify_camera_id(str(event.get("camera_id") or "camera"))
    safe_before = int(max(0.0, min(float(before), 3600.0)) * 1000)
    safe_after = int(max(0.0, min(float(after), 3600.0)) * 1000)
    clip_source = recording_source(source)
    clip_dir = manager.storage_dir / "event_clips" / camera_id / clip_source
    clip_dir.mkdir(parents=True, exist_ok=True)
    accel_mode = _hardware_acceleration_mode()
    return clip_dir / f"{event_id}-{safe_before}-{safe_after}-a3-{accel_mode}.mp4"


def _ensure_event_clip(
    event: dict,
    *,
    before: float,
    after: float,
    source: str = "main",
) -> Path:
    clip_source = recording_source(source)
    clip_path = _event_clip_path(event, before=before, after=after, source=clip_source)
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return clip_path
    cache_key = str(clip_path)
    with EVENT_CLIP_LOCKS_GUARD:
        lock = EVENT_CLIP_LOCKS.setdefault(cache_key, threading.Lock())
    with lock:
        if clip_path.exists() and clip_path.stat().st_size > 0:
            return clip_path
        if not EVENT_CLIP_BUILD_LIMITER.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="too many event clips are already being generated",
                headers={"Retry-After": "3"},
            )
        try:
            _build_event_clip(
                event,
                before=before,
                after=after,
                output_path=clip_path,
                source=clip_source,
            )
        finally:
            EVENT_CLIP_BUILD_LIMITER.release()
    return clip_path


def _build_event_clip(event: dict, before: float, after: float, output_path: Path, source: str = "main") -> None:
    camera_id = str(event.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")

    event_created_epoch = event_epoch(event)
    window_before = max(0.0, min(float(before), 3600.0))
    window_after = max(0.0, min(float(after), 3600.0))
    window_start = event_created_epoch - window_before
    window_end = event_created_epoch + window_after

    rows: list[dict] = []
    for candidate in manager.recorder.recording_rows_between(
            camera_id,
            window_start,
            window_end,
            recording_source(source),
            discover_missing=False,
        ):
        if candidate.get("start_epoch") is None or candidate.get("end_epoch") is None:
            continue
        try:
            candidate = {**candidate, "path": str(_recording_storage_path(candidate.get("path")))}
        except HTTPException:
            continue
        rows.append(candidate)
    rows.sort(key=lambda row: float(row["start_epoch"]))
    selected = [
        row for row in rows
        if float(row["end_epoch"]) > window_start and float(row["start_epoch"]) < window_end
    ]
    if not selected:
        raise HTTPException(status_code=404, detail="no recording window found")

    try:
        local_start, duration = concatenated_clip_timing(selected, window_start, window_end)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    concat_path = _write_concat_file(selected)
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    source_codec = _probe_video_codec(Path(str(selected[0]["path"])))
    commands: list[tuple[str, list[str]]] = []
    if _event_clip_vaapi_enabled(source_codec):
        commands.append(("vaapi", _event_clip_vaapi_command(source_codec, concat_path, local_start, duration, tmp_path)))
    if _event_clip_qsv_enabled(source_codec):
        commands.append(("qsv", _event_clip_qsv_command(source_codec, concat_path, local_start, duration, tmp_path)))
    commands.append(("cpu", _event_clip_cpu_command(concat_path, local_start, duration, tmp_path)))

    try:
        last_error = "event clip generation failed"
        for backend, command in commands:
            tmp_path.unlink(missing_ok=True)
            clip_timeout = max(60.0, min(600.0, duration * 2.0))
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=clip_timeout)
            if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.replace(output_path)
                logging.getLogger(__name__).info(
                    "built event clip %s using %s acceleration (source codec %s)",
                    output_path.name,
                    backend,
                    source_codec or "unknown",
                )
                return
            last_error = (result.stderr or f"event clip generation failed using {backend}").strip()[-500:]
            if backend in {"vaapi", "qsv"}:
                logging.getLogger(__name__).warning(
                    "%s event clip generation failed for %s, falling back to next backend: %s",
                    backend.upper(),
                    output_path.name,
                    last_error,
                )
        logging.getLogger(__name__).error(
            "event clip generation failed for %s: %s",
            output_path.name,
            redact_secret_text(last_error),
        )
        raise HTTPException(status_code=500, detail="event clip generation failed")
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="event clip generation timed out") from exc
    finally:
        concat_path.unlink(missing_ok=True)
        tmp_path.unlink(missing_ok=True)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _write_concat_file(rows: list[dict]) -> Path:
    paths = [str(_recording_storage_path(row.get("path"))) for row in rows]
    if any("\n" in path or "\r" in path for path in paths):
        raise HTTPException(status_code=400, detail="recording path is invalid")
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".ffconcat",
        prefix="survng-recordings-",
        delete=False,
    )
    with handle:
        for path_value in paths:
            escaped = path_value.replace("\\", "\\\\").replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    return Path(handle.name)


def _recording_start_epoch(path: Path) -> float | None:
    return manager.recorder.recording_start_epoch(path)
