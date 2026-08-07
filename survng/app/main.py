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
import signal
import secrets
import platform
import shutil
import time
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
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import cv2
import numpy as np

from .config import (
    AppConfig,
    CameraConfig,
    camera_by_id,
    load_config,
    normalize_config,
    save_config,
    slugify_camera_id,
)
from .config_application import (
    DETECTOR_FACE_ENGINE_FIELDS,
    DETECTOR_HOT_POLICY_FIELDS,
    DETECTOR_OBJECT_ENGINE_FIELDS,
    DETECTOR_OBJECT_TRACKING_RESET_FIELDS,
    DETECTOR_SHARED_ENGINE_FIELDS,
    HOT_CONFIG_FIELDS,
    RECORDER_CONFIG_FIELDS,
    TRACKING_REID_ENGINE_FIELDS,
    TRACKING_SESSION_FIELDS,
    TargetedConfigApplication,
    manager_owned_config,
)
from .config_routes import (
    ConfigProbeRequest,
    ConfigRouteDependencies,
    SECRET_PLACEHOLDER,
    _restore_camera_secrets,
    create_config_router,
    redacted_camera_payload as _redacted_camera_payload,
    redacted_config_payload as _redacted_config_payload,
    restore_config_secrets as _restore_config_secrets,
)
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
from .appearance_routes import AppearanceRouteDependencies, create_appearance_router
from .camera_intelligence import (
    aggregate_camera_intelligence,
    compare_camera_intelligence_results,
    select_balanced_samples,
)
from .camera_routes import match_camera_route
from .calibration import (
    apply_calibration_changes,
    build_calibration_report,
    calibration_configuration_fingerprint,
    calibration_setting_value,
)
from .detector import detection_failure, objects_to_json
from .detection_routes import (
    DetectionRouteDependencies,
    TrackingComparisonVerdictRequest,
    _tracking_comparison_duration,
    create_detection_router,
)
from .face_routes import (
    FaceRouteDependencies,
    _public_face_observation,
    create_face_router,
)
from .manager import (
    AppManager,
    ManagerShutdownIncompleteError,
    validate_motion_pipeline_configuration,
)
from .manager_reload import ManagerGenerationLifecycle, ManagerReloadHooks
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
    snapshot_media_type,
)
from .incident_presenter import (
    _best_incident_event,
    _event_row,
    _incident_event_payload,
    _incident_list_payload,
    _incident_row,
    _incident_rows,
    _recording_event_row,
    _recording_grid_incident_payload,
)
from .incident_queries import (
    IncidentQueryDependencies,
    IncidentQueryService,
    _filter_incident_summaries,
    _filter_incidents_by_event_type,
    _motion_audit_row,
    create_incident_query_router,
)
from .recording_media import (
    concatenated_clip_timing,
    event_clip_window,
    playback_segment_duration,
)
from .recording_routes import (
    MediaExportBatchRequest,
    MediaExportMetadataRequest,
    MediaExportProtectionRequest,
    MediaExportRequest,
    RecordingRouteDependencies,
    _public_recording_row,
    _recording_playback_window,
    _validate_recording_range,
    create_recording_router,
    recording_source,
)
from .object_tracking import ultralytics_deepocsort_dependency_status
from .tracking_comparison import (
    TRACKING_COMPARISON_IMPLEMENTATIONS,
    TrackingComparisonRunner,
    sampled_video_frames,
)
from .system_telemetry import (
    SystemTelemetryDependencies,
    SystemTelemetryService,
    create_system_telemetry_router,
)
from .zones import apply_detection_zones, detection_threshold
from .security import redact_secret_text
from .storage_maintenance import StorageMaintenanceRunner, StorageReconciler

config = load_config()
manager = AppManager(config)
LOGGER = logging.getLogger(__name__)
LOG_LINES: deque[dict] = deque(maxlen=1000)
RECORDING_FMP4_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
RECORDING_FMP4_LOCKS_GUARD = threading.Lock()
EVENT_CLIP_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
EVENT_CLIP_LOCKS_GUARD = threading.Lock()
RECORDING_DAY_CACHE: dict[tuple[str, str, int, int], tuple[float, list[dict]]] = {}
RECORDING_DAY_CACHE_LOCK = threading.Lock()
RECORDING_DAY_CACHE_SECONDS = 30.0
RECORDING_NEAR_LIVE_CACHE_SECONDS = 2.0
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
SYSTEM_TELEMETRY = SystemTelemetryService()
PROCESS_INSTANCE_ID = SYSTEM_TELEMETRY.process_instance_id
INCIDENT_QUERIES = IncidentQueryService()
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


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    object_labels: list[str] = Field(default_factory=list, max_length=100)
    start_at: str = Field(default="", max_length=64)
    end_at: str = Field(default="", max_length=64)
    limit: int = Field(default=50, ge=1, le=500)
    minimum_score: float = Field(default=-1.0, ge=-1.0, le=1.0)


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


def _require_recording_camera(camera_id: str) -> None:
    if manager.camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="camera not found")


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
    """Replace the active manager through the generation lifecycle owner."""
    effective = normalize_config(
        next_config.model_copy(deep=True),
        assign_ids=assign_ids,
    )

    def publish_runtime(next_value: AppConfig, next_manager: AppManager) -> None:
        global config, manager
        config = next_value
        manager = next_manager

    def refresh_runtime_caches() -> None:
        global FACE_OBSERVATIONS_SYNCED
        FACE_OBSERVATIONS_SYNCED = False
        _start_face_observation_sync()
        with RECORDING_DAY_CACHE_LOCK:
            RECORDING_DAY_CACHE.clear()
        _ffmpeg_qsv_info.cache_clear()
        _ffmpeg_vaapi_info.cache_clear()

    lifecycle = ManagerGenerationLifecycle(
        lock=MANAGER_RELOAD_LOCK,
        stopping=APPLICATION_STOPPING,
        manager_factory=AppManager,
        hooks=ManagerReloadHooks(
            active_storage_tasks=_active_storage_tasks,
            active_ai_operations=_active_ai_operations,
            prewarmer_running=lambda: bool(
                RECORDING_PREWARM_THREAD is not None
                and RECORDING_PREWARM_THREAD.is_alive()
            ),
            stop_prewarmer=_stop_recording_prewarmer,
            start_prewarmer=_start_recording_prewarmer,
            save_config=save_config,
            publish_runtime=publish_runtime,
            refresh_runtime_caches=refresh_runtime_caches,
            storage_error=StorageTasksActiveError,
            ai_error=AiOperationsActiveError,
        ),
    )
    lifecycle.reload(config, manager, effective, persist=persist)
    return effective


def _manager_owned_config(config_value: AppConfig) -> dict:
    """Compatibility name for configuration ownership tests."""
    return manager_owned_config(config_value)


def apply_config_update(
    next_config: AppConfig,
    *,
    assign_ids: bool = False,
    persist: bool = True,
) -> tuple[AppConfig, dict[str, object]]:
    """Apply configuration through the dedicated transaction owner."""
    global config
    application = TargetedConfigApplication(
        lock=MANAGER_RELOAD_LOCK,
        save=save_config,
        active_exports=lambda: (
            MEDIA_EXPORTS.active_jobs() if MEDIA_EXPORTS is not None else []
        ),
        storage_error=StorageTasksActiveError,
    )
    effective = application.normalize(next_config, assign_ids=assign_ids)
    if manager_owned_config(config) != manager_owned_config(effective):
        applied = reload_manager(effective, assign_ids=False, persist=persist)
        return applied, {
            "apply_mode": "manager_reload",
            "camera_workers_restarted": True,
            "subsystems_restarted": ["manager"],
        }
    effective, result = application.apply(
        config,
        effective,
        manager,
        persist=persist,
    )
    config = effective
    return effective, result


def _record_process_lifecycle(kind: str) -> None:
    """Record restart evidence without making telemetry a lifecycle dependency."""
    try:
        manager.events.record_lifecycle_event(PROCESS_INSTANCE_ID, kind)
    except Exception:
        logging.getLogger(__name__).exception(
            "could not persist process lifecycle event %s",
            kind,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    APPLICATION_STOPPING.clear()
    loop = asyncio.get_running_loop()
    early_onvif_thread: threading.Thread | None = None
    early_onvif_lock = threading.Lock()
    media_exports: MediaExportManager | None = None
    calibration_monitor_task: asyncio.Task | None = None

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
    _record_process_lifecycle("startup_started")
    manager.start_all()
    try:
        media_exports = _media_export_manager()
        media_exports.start()
        _start_face_observation_sync()
        _start_recording_prewarmer()
        calibration_monitor_task = asyncio.create_task(
            _calibration_followup_monitor(),
            name="survng-calibration-followup-monitor",
        )
        _record_process_lifecycle("startup_ready")
        yield
    finally:
        _record_process_lifecycle("shutdown_requested")
        APPLICATION_STOPPING.set()
        if calibration_monitor_task is not None:
            calibration_monitor_task.cancel()
            try:
                await calibration_monitor_task
            except asyncio.CancelledError:
                pass
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
        _record_process_lifecycle("shutdown_completed")


app = FastAPI(title="SurvNG", lifespan=lifespan)


def _publish_config_runtime(next_config: AppConfig) -> None:
    global config
    config = next_config
    manager.config = next_config


app.include_router(
    create_config_router(
        ConfigRouteDependencies(
            get_config=lambda: config,
            get_manager=lambda: manager,
            publish_config=_publish_config_runtime,
            apply_config=apply_config_update,
            reload_manager=reload_manager,
            save_config=save_config,
            validate_config=validate_motion_pipeline_configuration,
            lock=MANAGER_RELOAD_LOCK,
            probe_limiter=CONFIG_PROBE_LIMITER,
        )
    )
)
app.include_router(
    create_system_telemetry_router(
        SystemTelemetryDependencies(
            get_config=lambda: config,
            get_manager=lambda: manager,
        ),
        SYSTEM_TELEMETRY,
    )
)


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


@app.get("/recordings/search")
def recording_search_page() -> HTMLResponse:
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
                yield _sse_message(
                    "system_state",
                    await asyncio.to_thread(SYSTEM_TELEMETRY.system_status, manager),
                )
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
                        yield _sse_message(
                            "system_state",
                            await asyncio.to_thread(SYSTEM_TELEMETRY.system_status, manager),
                        )
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


@app.get("/api/retention/status")
def recording_retention_status() -> dict:
    return manager.recording.retention_status()


@app.post("/api/retention/run", status_code=202)
def run_recording_retention(request: RecordingRetentionRequest) -> dict:
    return manager.recording.request_retention_run(apply=request.apply)


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
    return {
        "active": "survng_hybrid",
        "implementations": [
            {
                "id": "survng_hybrid",
                "name": "SurvNG Hybrid",
                "available": True,
                "description": "Lightweight geometry tracking with SurvNG appearance recovery.",
            },
        ],
    }


@app.get("/api/detector/models")
def detector_models() -> dict:
    models: list[dict] = []
    search_roots = sorted({
        Path("models/openvino_model"),
        *Path("models").glob("*_openvino_model"),
        # Preserve discovery for installations that have not consolidated yet.
        Path("openvino_model"),
        *Path(".").glob("*_openvino_model"),
    })
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



@app.get("/api/semantic-search/status")
def semantic_search_status() -> dict[str, Any]:
    with MANAGER_RELOAD_LOCK:
        return manager.semantic_search_status()


@app.post("/api/semantic-search")
def semantic_search(request: SemanticSearchRequest) -> dict[str, Any]:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        maximum = min(request.limit, active_manager.config.semantic_search.max_results)
        semantic_service = active_manager.semantic_search
        event_store = active_manager.events
        base_path = active_manager.config.base_path
    # Text inference may take seconds during worker recovery. The semantic
    # service serializes inference against close(), so it need not block the
    # global manager/config lock and unrelated API traffic.
    try:
        hits = semantic_service.search_text(
            request.query,
            camera_ids=request.camera_ids,
            object_labels=request.object_labels,
            start_at=request.start_at,
            end_at=request.end_at,
            limit=maximum * 4,
            minimum_score=request.minimum_score,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    best_by_event: dict[int, Any] = {}
    for hit in hits:
        best_by_event.setdefault(hit.event_id, hit)
        if len(best_by_event) >= maximum:
            break
    event_rows = {
        int(row["id"]): {
            "id": int(row["id"]),
            "camera_id": str(row.get("camera_id") or ""),
            "kind": str(row.get("kind") or ""),
            "created_at": str(row.get("created_at") or ""),
        }
        for row in event_store.get_many(list(best_by_event))
    }
    results = []
    for event_id, hit in best_by_event.items():
        event = event_rows.get(event_id)
        if event is None:
            continue
        results.append({
            "score": round(hit.score, 6),
            "evidence": {
                "source_kind": hit.source_kind,
                "source_key": hit.source_key,
                "object_label": hit.object_label,
                "bbox": hit.bbox,
            },
            "event": event,
            "snapshot_url": f"{base_path}/api/events/{event_id}/snapshot.jpg",
        })
    return {"query": request.query.strip(), "count": len(results), "results": results}




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


class CalibrationRunRequest(BaseModel):
    camera_ids: list[str] = Field(default_factory=list, max_length=128)
    mode: str = Field(default="standard", pattern=r"^(quick|standard|deep)$")
    override_active_evaluation: bool = False


class CalibrationApplyRequest(BaseModel):
    recommendation_ids: list[str] = Field(min_length=1, max_length=256)
    confirmed: bool = False
    configuration_fingerprint: str = Field(min_length=64, max_length=64)
    evaluation_hours: float = Field(default=24.0, ge=24.0, le=168.0)


class CalibrationRollbackRequest(BaseModel):
    change_ids: list[str] = Field(default_factory=list, max_length=256)
    camera_ids: list[str] = Field(default_factory=list, max_length=128)
    confirmed: bool = False
    force_conflicts: bool = False


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
    visual_backup = {
        "grace_seconds": (
            active_config.motion_qualification.visual_backup_grace_seconds
            if override.visual_backup_grace_seconds is None
            else override.visual_backup_grace_seconds
        ),
        "minimum_score": (
            active_config.motion_qualification.visual_backup_min_score
            if override.visual_backup_min_score is None
            else override.visual_backup_min_score
        ),
        "minimum_consecutive": (
            active_config.motion_qualification.visual_backup_min_consecutive
            if override.visual_backup_min_consecutive is None
            else override.visual_backup_min_consecutive
        ),
        "cooldown_seconds": (
            active_config.motion_qualification.visual_backup_cooldown_seconds
            if override.visual_backup_cooldown_seconds is None
            else override.visual_backup_cooldown_seconds
        ),
        "maximum_triggers_5m": (
            active_config.motion_qualification.visual_backup_max_triggers_5m
            if override.visual_backup_max_triggers_5m is None
            else override.visual_backup_max_triggers_5m
        ),
    }
    effective = {
        "mode": effective_mode,
        "sensitivity": active_config.motion_qualification.sensitivity if override.sensitivity == "inherit" else override.sensitivity,
        "stationary_object_tolerance": (
            active_config.motion_qualification.stationary_object_tolerance
            if override.stationary_object_tolerance == "inherit"
            else override.stationary_object_tolerance
        ),
        "illumination_filter_enabled": (
            active_config.motion_qualification.illumination_filter_enabled
            if override.illumination_filter_enabled is None
            else override.illumination_filter_enabled
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
        "visual_backup_grace_seconds": visual_backup["grace_seconds"],
        "visual_backup_min_score": visual_backup["minimum_score"],
        "visual_backup_score_margin": active_config.motion_qualification.visual_backup_score_margin,
        "visual_backup_min_consecutive": visual_backup["minimum_consecutive"],
        "visual_backup_cooldown_seconds": visual_backup["cooldown_seconds"],
        "visual_backup_max_triggers_5m": visual_backup["maximum_triggers_5m"],
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
    incidents = INCIDENT_QUERIES.with_faces(
        active_manager,
        INCIDENT_QUERIES.hydrate(active_manager, incident_summaries),
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


CALIBRATION_MODE_LIMITS = {
    "quick": (24.0, 100, 12),
    "standard": (168.0, 100, 20),
    "deep": (720.0, 100, 40),
}


def _calibration_camera_review(
    camera: CameraConfig,
    *,
    hours: float,
    record_limit: int,
    image_limit: int,
    active_config: AppConfig,
    active_manager: AppManager,
) -> dict[str, Any]:
    samples, records_considered = _camera_intelligence_candidates(
        camera,
        active_manager,
        hours=hours,
        record_limit=record_limit,
        image_limit=image_limit,
    )
    if not samples:
        return {
            "review_type": "camera_intelligence",
            "summary": "No retained incidents or motion decisions were available.",
            "analyzed": 0,
            "failed": 0,
            "recommendations": [],
            "samples": [],
        }
    wait_started = time.monotonic()
    while not AUDIT_AI_LIMITER.acquire(timeout=5):
        if APPLICATION_STOPPING.is_set():
            raise RuntimeError("camera review stopped because SurvNG is shutting down")
        if time.monotonic() - wait_started >= 300:
            raise RuntimeError("timed out waiting for the AI review worker")
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
    _run_camera_intelligence_review(
        int(review["id"]),
        camera.id,
        samples,
        records_considered,
        hours,
        active_config,
        active_manager,
    )
    completed = active_manager.events.get_motion_ai_review(int(review["id"])) or {}
    if completed.get("status") != "completed":
        raise RuntimeError(str(completed.get("error") or "camera review failed"))
    return {
        **(completed.get("result") or {}),
        "source_review_id": int(review["id"]),
    }


def _run_system_calibration(
    run_id: int,
    camera_ids: list[str],
    mode: str,
    active_config: AppConfig,
    active_manager: AppManager,
) -> None:
    hours, record_limit, image_limit = CALIBRATION_MODE_LIMITS[mode]
    reports: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    try:
        active_manager.events.update_calibration_run(run_id, status="running")
        for camera_id in camera_ids:
            if APPLICATION_STOPPING.is_set():
                raise RuntimeError("system calibration stopped because SurvNG is shutting down")
            camera = camera_by_id(active_config, camera_id)
            if camera is None:
                errors[camera_id] = "camera is no longer configured"
                continue
            try:
                reports[camera_id] = _calibration_camera_review(
                    camera,
                    hours=hours,
                    record_limit=record_limit,
                    image_limit=image_limit,
                    active_config=active_config,
                    active_manager=active_manager,
                )
            except Exception as exc:
                LOGGER.warning("calibration review failed for %s", camera_id, exc_info=True)
                errors[camera_id] = redact_secret_text(exc)
            active_manager.events.update_calibration_run(
                run_id,
                status="running",
                result={
                    "progress": {
                        "completed": len(reports) + len(errors),
                        "total": len(camera_ids),
                    },
                    "camera_errors": errors,
                },
            )
        if not reports:
            raise RuntimeError("no selected camera could be analyzed")
        statuses = {str(item.get("id") or ""): item for item in active_manager.statuses()}
        stream_health = {
            camera_id: {
                key: statuses.get(camera_id, {}).get(key)
                for key in (
                    "running", "live_fps", "main_fps", "capture_read_failures",
                    "analysis_frames_dropped", "last_error",
                )
                if key in statuses.get(camera_id, {})
            }
            for camera_id in reports
        }
        result = build_calibration_report(
            active_config,
            reports,
            mode=mode,
            stream_health=stream_health,
        )
        result["camera_errors"] = errors
        result["progress"] = {"completed": len(camera_ids), "total": len(camera_ids)}
        active_manager.events.update_calibration_run(
            run_id,
            status="completed",
            result=result,
        )
    except Exception as exc:
        LOGGER.exception("system calibration run %s failed", run_id)
        active_manager.events.update_calibration_run(
            run_id,
            status="interrupted" if APPLICATION_STOPPING.is_set() else "failed",
            result={"camera_reports": reports, "camera_errors": errors},
            error=redact_secret_text(exc),
        )


@app.post("/api/calibration/runs", status_code=202)
def start_calibration_run(request: CalibrationRunRequest) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        if not active_config.audit_ai.enabled or not ai_provider_configured(active_config.audit_ai):
            raise HTTPException(status_code=400, detail="AI analysis is not configured")
        active_runs = [
            item for item in active_manager.events.calibration_runs(20)
            if item.get("status") in {"queued", "running"}
        ]
        if active_runs:
            raise HTTPException(status_code=409, detail="a system calibration analysis is already running")
        if not request.override_active_evaluation:
            active_sets = [
                item for item in active_manager.events.calibration_change_sets(100)
                if item.get("status") in {"collecting", "reviewing"}
            ]
            if active_sets:
                raise HTTPException(
                    status_code=409,
                    detail="a calibration change set is still being evaluated; explicitly override it to start another run",
                )
        available = {camera.id for camera in active_config.cameras}
        camera_ids = list(dict.fromkeys(request.camera_ids or [camera.id for camera in active_config.cameras]))
        unknown = sorted(set(camera_ids) - available)
        if unknown:
            raise HTTPException(status_code=404, detail=f"unknown cameras: {', '.join(unknown)}")
        if not camera_ids:
            raise HTTPException(status_code=400, detail="no cameras are configured")
        run = active_manager.events.create_calibration_run(
            mode=request.mode,
            camera_ids=camera_ids,
            configuration_fingerprint=calibration_configuration_fingerprint(active_config),
        )
    thread = threading.Thread(
        target=_run_system_calibration,
        args=(int(run["id"]), camera_ids, request.mode, active_config, active_manager),
        name=f"survng-calibration-{run['id']}",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException as exc:
        active_manager.events.update_calibration_run(
            int(run["id"]),
            status="failed",
            error=f"Calibration worker could not start: {redact_secret_text(exc)}",
        )
        raise HTTPException(
            status_code=503,
            detail="calibration worker could not start",
        ) from exc
    return run


@app.get("/api/calibration/runs")
def calibration_runs(limit: int = 20) -> dict:
    return {"runs": manager.events.calibration_runs(limit, include_result=False)}


@app.get("/api/calibration/runs/{run_id}")
def calibration_run(run_id: int) -> dict:
    run = manager.events.get_calibration_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="calibration run not found")
    return run


@app.post("/api/calibration/runs/{run_id}/apply")
def calibration_apply(run_id: int, request: CalibrationApplyRequest) -> dict:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="explicit confirmation is required")
    with MANAGER_RELOAD_LOCK:
        if not config.audit_ai.allow_apply_recommendations:
            raise HTTPException(status_code=403, detail="applying AI recommendations is disabled")
        active_events = manager.events
        run = active_events.get_calibration_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="calibration run not found")
        if run.get("status") != "completed":
            raise HTTPException(status_code=409, detail="calibration analysis is not complete")
        current_fingerprint = calibration_configuration_fingerprint(config)
        if request.configuration_fingerprint != current_fingerprint or run.get("configuration_fingerprint") != current_fingerprint:
            raise HTTPException(
                status_code=409,
                detail="calibration settings changed after analysis; run calibration again",
            )
        recommendations = {
            str(item.get("id") or ""): item
            for item in (run.get("result") or {}).get("recommendations") or []
        }
        recommendation_ids = list(dict.fromkeys(request.recommendation_ids))
        unknown = [item for item in recommendation_ids if item not in recommendations]
        if unknown:
            raise HTTPException(status_code=400, detail="one or more recommendations changed or expired")
        selected = [recommendations[item] for item in recommendation_ids]
        try:
            next_config, changes = apply_calibration_changes(config, selected)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not changes:
            raise HTTPException(status_code=409, detail="selected recommendations no longer change configuration")
        for index, change in enumerate(changes, start=1):
            change["id"] = f"{run_id}:{index}:{change['setting']}:{change.get('camera_id') or 'global'}"
        before_config = config.model_copy(deep=True)
        before_fingerprint = current_fingerprint
        _effective, apply_result = apply_config_update(next_config)
        after_fingerprint = calibration_configuration_fingerprint(config)
        try:
            change_set = active_events.create_calibration_change_set(
                run_id=run_id,
                parent_change_set_id=None,
                action="apply",
                status="collecting",
                evaluation_hours=request.evaluation_hours,
                configuration_fingerprint_before=before_fingerprint,
                configuration_fingerprint_after=after_fingerprint,
                changes=changes,
                apply_result=apply_result,
            )
        except BaseException:
            try:
                apply_config_update(before_config)
            except Exception:
                LOGGER.exception("calibration ledger failure rollback was incomplete")
            raise
    return {"ok": True, "change_set": change_set}


@app.get("/api/calibration/change-sets")
def calibration_change_sets(limit: int = 50) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_events = manager.events
        rows = active_events.calibration_change_sets(limit)
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.get("action") == "apply":
            row["rolled_back_change_ids"] = sorted(
                active_events.calibration_rollback_change_ids(int(row["id"]))
            )
        try:
            created = datetime.fromisoformat(str(row.get("created_at") or "").replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            ready_at = created + timedelta(hours=float(row.get("evaluation_hours") or 24))
            row["ready_at"] = ready_at.isoformat()
            row["seconds_until_ready"] = max(0, round((ready_at - now).total_seconds()))
        except (TypeError, ValueError):
            row["ready_at"] = ""
            row["seconds_until_ready"] = 0
    return {"change_sets": rows}


def _run_calibration_evaluation(
    change_set_id: int,
    active_config: AppConfig,
    active_manager: AppManager,
) -> None:
    change_set = active_manager.events.get_calibration_change_set(change_set_id) or {}
    run = active_manager.events.get_calibration_run(int(change_set.get("run_id") or 0)) or {}
    baseline_reports = (run.get("result") or {}).get("camera_reports") or {}
    global_change = any(
        item.get("scope") == "global" for item in change_set.get("changes") or []
    )
    affected = {
        str(item.get("camera_id") or "")
        for item in change_set.get("changes") or []
        if item.get("camera_id")
    }
    if global_change:
        affected.update(str(item) for item in run.get("camera_ids") or [])
    comparisons: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    try:
        with MANAGER_RELOAD_LOCK:
            current_fingerprint = calibration_configuration_fingerprint(config)
        expected_fingerprint = str(
            change_set.get("configuration_fingerprint_after") or ""
        )
        if expected_fingerprint and current_fingerprint != expected_fingerprint:
            active_manager.events.update_calibration_evaluation(
                change_set_id,
                {
                    "outcome": "inconclusive",
                    "summary": (
                        "Calibration settings changed again before follow-up evidence was "
                        "reviewed, so this change set cannot be evaluated independently."
                    ),
                    "comparison_basis": "configuration_conflict",
                },
                status="evaluated",
            )
            return
        try:
            applied_at = datetime.fromisoformat(
                str(change_set.get("created_at") or "").replace("Z", "+00:00")
            )
            if applied_at.tzinfo is None:
                applied_at = applied_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            applied_at = datetime.now(timezone.utc) - timedelta(hours=24)
        hours = max(1.0, min(720.0, (datetime.now(timezone.utc) - applied_at).total_seconds() / 3600.0))
        mode = str(run.get("mode") or "standard")
        _baseline_hours, record_limit, image_limit = CALIBRATION_MODE_LIMITS.get(
            mode, CALIBRATION_MODE_LIMITS["standard"]
        )
        for camera_id in sorted(affected):
            if APPLICATION_STOPPING.is_set():
                raise RuntimeError("calibration follow-up stopped because SurvNG is shutting down")
            camera = camera_by_id(active_config, camera_id)
            baseline = baseline_reports.get(camera_id)
            if camera is None or not isinstance(baseline, dict):
                errors[camera_id] = "baseline evidence is unavailable"
                continue
            try:
                followup = _calibration_camera_review(
                    camera,
                    hours=hours,
                    record_limit=record_limit,
                    image_limit=image_limit,
                    active_config=active_config,
                    active_manager=active_manager,
                )
                comparisons[camera_id] = {
                    "comparison": compare_camera_intelligence_results(baseline, followup),
                    "followup": followup,
                }
            except Exception as exc:
                LOGGER.warning("calibration follow-up failed for %s", camera_id, exc_info=True)
                errors[camera_id] = redact_secret_text(exc)
        outcomes = Counter(
            str(item.get("comparison", {}).get("outcome") or "inconclusive")
            for item in comparisons.values()
        )
        comparable = outcomes["improved"] + outcomes["worsened"]
        if not comparisons or comparable == 0:
            outcome = "inconclusive"
        elif outcomes["worsened"] and outcomes["improved"]:
            outcome = "mixed"
        elif outcomes["worsened"]:
            outcome = "regressed"
        else:
            outcome = "improved"
        evaluation = {
            "outcome": outcome,
            "camera_comparisons": comparisons,
            "camera_errors": errors,
            "summary": (
                f"{outcomes['improved']} cameras improved, {outcomes['worsened']} regressed, "
                f"and {outcomes['inconclusive']} were inconclusive using matched before/after evidence."
            ),
            "comparison_basis": "category_matched_balanced_samples",
        }
        with MANAGER_RELOAD_LOCK:
            final_fingerprint = calibration_configuration_fingerprint(config)
        if expected_fingerprint and final_fingerprint != expected_fingerprint:
            evaluation = {
                "outcome": "inconclusive",
                "summary": (
                    "Calibration settings changed while follow-up evidence was being "
                    "reviewed, so this result cannot be attributed to one change set."
                ),
                "comparison_basis": "configuration_conflict",
                "camera_errors": errors,
            }
        active_manager.events.update_calibration_evaluation(
            change_set_id,
            evaluation,
            status="evaluated",
        )
    except Exception as exc:
        LOGGER.exception("calibration evaluation %s failed", change_set_id)
        active_manager.events.update_calibration_evaluation(
            change_set_id,
            {"outcome": "failed", "error": redact_secret_text(exc)},
            status="evaluation_failed",
        )


async def _calibration_followup_monitor() -> None:
    """Start due follow-ups automatically without overlapping evaluations."""
    while not APPLICATION_STOPPING.is_set():
        await asyncio.sleep(60)
        if APPLICATION_STOPPING.is_set():
            return
        selected: tuple[int, AppConfig, AppManager] | None = None
        with MANAGER_RELOAD_LOCK:
            active_manager = manager
            change_sets = active_manager.events.calibration_change_sets(100)
            if any(item.get("status") == "reviewing" for item in change_sets):
                continue
            now = datetime.now(timezone.utc)
            for item in reversed(change_sets):
                if item.get("action") != "apply" or item.get("status") != "collecting":
                    continue
                try:
                    created = datetime.fromisoformat(
                        str(item.get("created_at") or "").replace("Z", "+00:00")
                    )
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    ready_at = created + timedelta(
                        hours=float(item.get("evaluation_hours") or 24)
                    )
                except (TypeError, ValueError):
                    continue
                if now < ready_at:
                    continue
                change_set_id = int(item["id"])
                active_manager.events.update_calibration_change_set_status(
                    change_set_id,
                    "reviewing",
                )
                selected = (
                    change_set_id,
                    config.model_copy(deep=True),
                    active_manager,
                )
                break
        if selected is not None:
            change_set_id, active_config, active_manager = selected
            thread = threading.Thread(
                target=_run_calibration_evaluation,
                args=(change_set_id, active_config, active_manager),
                name=f"survng-calibration-evaluation-{change_set_id}",
                daemon=True,
            )
            try:
                thread.start()
            except BaseException as exc:
                LOGGER.exception("calibration evaluation worker could not start")
                active_manager.events.update_calibration_evaluation(
                    change_set_id,
                    {
                        "outcome": "failed",
                        "error": f"Evaluation worker could not start: {redact_secret_text(exc)}",
                    },
                    status="evaluation_failed",
                )


@app.post("/api/calibration/change-sets/{change_set_id}/evaluate", status_code=202)
def start_calibration_evaluation(change_set_id: int) -> dict:
    with MANAGER_RELOAD_LOCK:
        active_manager = manager
        active_config = config.model_copy(deep=True)
        change_set = active_manager.events.get_calibration_change_set(change_set_id)
        if change_set is None:
            raise HTTPException(status_code=404, detail="calibration change set not found")
        if change_set.get("action") != "apply":
            raise HTTPException(status_code=400, detail="rollback entries are not evaluated")
        if change_set.get("status") not in {"collecting", "evaluation_failed"}:
            raise HTTPException(status_code=409, detail="change set is already evaluating or complete")
        try:
            created = datetime.fromisoformat(str(change_set["created_at"]).replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            ready_at = created + timedelta(hours=float(change_set.get("evaluation_hours") or 24))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="evaluation schedule is invalid") from exc
        if datetime.now(timezone.utc) < ready_at:
            raise HTTPException(
                status_code=409,
                detail=f"follow-up evidence is still being collected until {ready_at.isoformat()}",
            )
        active_manager.events.update_calibration_change_set_status(change_set_id, "reviewing")
    thread = threading.Thread(
        target=_run_calibration_evaluation,
        args=(change_set_id, active_config, active_manager),
        name=f"survng-calibration-evaluation-{change_set_id}",
        daemon=True,
    )
    try:
        thread.start()
    except BaseException as exc:
        active_manager.events.update_calibration_evaluation(
            change_set_id,
            {
                "outcome": "failed",
                "error": f"Evaluation worker could not start: {redact_secret_text(exc)}",
            },
            status="evaluation_failed",
        )
        raise HTTPException(
            status_code=503,
            detail="calibration evaluation worker could not start",
        ) from exc
    return active_manager.events.get_calibration_change_set(change_set_id) or {}


@app.post("/api/calibration/change-sets/{change_set_id}/rollback")
def calibration_rollback(change_set_id: int, request: CalibrationRollbackRequest) -> dict:
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="explicit confirmation is required")
    with MANAGER_RELOAD_LOCK:
        active_events = manager.events
        source = active_events.get_calibration_change_set(change_set_id)
        if source is None:
            raise HTTPException(status_code=404, detail="calibration change set not found")
        if source.get("action") != "apply":
            raise HTTPException(status_code=400, detail="only applied calibration changes can be rolled back")
        already_rolled_back = active_events.calibration_rollback_change_ids(change_set_id)
        selected = []
        change_ids = set(request.change_ids)
        camera_ids = set(request.camera_ids)
        for item in source.get("changes") or []:
            if change_ids and str(item.get("id") or "") not in change_ids:
                continue
            if camera_ids and str(item.get("camera_id") or "") not in camera_ids:
                continue
            if str(item.get("id") or "") in already_rolled_back:
                continue
            selected.append(item)
        if not selected:
            raise HTTPException(
                status_code=409 if already_rolled_back else 400,
                detail="the selected calibration changes are already rolled back"
                if already_rolled_back
                else "no matching changes selected for rollback",
            )
        conflicts = []
        inverse = []
        for item in selected:
            current = calibration_setting_value(
                config,
                scope=str(item.get("scope") or ""),
                camera_id=str(item.get("camera_id") or ""),
                setting=str(item.get("setting") or ""),
            )
            if current != item.get("after"):
                conflicts.append({
                    "change_id": item.get("id"),
                    "setting": item.get("setting"),
                    "camera_id": item.get("camera_id"),
                    "expected": item.get("after"),
                    "current": current,
                })
            inverse.append({
                "scope": item.get("scope"),
                "camera_id": item.get("camera_id"),
                "setting": item.get("setting"),
                "proposed": item.get("before"),
                "reason": f"Rollback of calibration change set {change_set_id}",
                "source_change_id": item.get("id"),
            })
        if conflicts and not request.force_conflicts:
            raise HTTPException(
                status_code=409,
                detail={"message": "newer configuration changes conflict with rollback", "conflicts": conflicts},
            )
        try:
            next_config, changes = apply_calibration_changes(config, inverse)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        before_config = config.model_copy(deep=True)
        before_fingerprint = calibration_configuration_fingerprint(config)
        _effective, apply_result = apply_config_update(next_config)
        after_fingerprint = calibration_configuration_fingerprint(config)
        try:
            rollback = active_events.create_calibration_change_set(
                run_id=source.get("run_id"),
                parent_change_set_id=change_set_id,
                action="rollback",
                status="rolled_back",
                evaluation_hours=24,
                configuration_fingerprint_before=before_fingerprint,
                configuration_fingerprint_after=after_fingerprint,
                changes=changes,
                apply_result={**apply_result, "forced_conflicts": conflicts},
            )
        except BaseException:
            try:
                apply_config_update(before_config)
            except Exception:
                LOGGER.exception("calibration rollback ledger failure recovery was incomplete")
            raise
        rolled_count = len(already_rolled_back) + len(selected)
        total_count = len(source.get("changes") or [])
        active_events.update_calibration_change_set_status(
            change_set_id,
            "rolled_back" if rolled_count >= total_count else "partially_rolled_back",
        )
    return {"ok": True, "rollback": rollback, "conflicts": conflicts}



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
    status = SYSTEM_TELEMETRY.system_status(active_manager)
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
    incident = INCIDENT_QUERIES.resolve_event(active_manager, event_id)
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
    incident = INCIDENT_QUERIES.resolve_event(active_manager, event_id)
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
    hydrated = INCIDENT_QUERIES.with_faces(
        active_manager,
        INCIDENT_QUERIES.hydrate(active_manager, candidate_summaries),
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


def _assistant_semantic_search(
    call: AssistantToolCall,
    time_zone: str,
    active_manager: AppManager,
) -> list[AssistantEvidence]:
    query = call.query.strip()
    if not query:
        return []
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
    try:
        hits = active_manager.semantic_search.search_text(
            query,
            camera_ids=[call.camera_id] if call.camera_id else [],
            object_labels=[call.object_label] if call.object_label else [],
            start_at=start.isoformat(), end_at=end.isoformat(),
            limit=min(call.limit * 4, 200), minimum_score=-1.0,
        )
    except RuntimeError as exc:
        return [AssistantEvidence(
            evidence_id="E-semantic-status", kind="semantic_search_status",
            title="Visual search unavailable", summary=str(exc),
            data=active_manager.semantic_search_status(), href="/recordings/search",
        )]
    best: dict[int, Any] = {}
    for hit in hits:
        best.setdefault(hit.event_id, hit)
        if len(best) >= call.limit:
            break
    evidence = [AssistantEvidence(
        evidence_id="E-semantic-search", kind="semantic_search",
        title=f'Visual search · “{query}”',
        summary=f"Found {len(best)} visually similar incident(s) in the indexed evidence.",
        data={
            "query": query, "start_at": start.isoformat(), "end_at": end.isoformat(),
            "camera_id": call.camera_id, "object_label": call.object_label,
            "matches": [
                {"event_id": hit.event_id, "score": round(hit.score, 4),
                 "source_kind": hit.source_kind, "object_label": hit.object_label}
                for hit in best.values()
            ],
            "limitations": [
                "Similarity is not identity proof.",
                "Search currently covers indexed object incidents, not every recording frame.",
            ],
        },
        href="/recordings/search",
    )]
    for event_id in best:
        incident = INCIDENT_QUERIES.resolve_event(active_manager, event_id)
        if incident is not None:
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
        INCIDENT_QUERIES.resolve_event(active_manager, int(event_id))
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

    candidates = INCIDENT_QUERIES.with_faces(
        active_manager,
        INCIDENT_QUERIES.hydrate(active_manager, candidate_summaries),
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
    if call.name == "semantic_search_recordings":
        return _assistant_semantic_search(
            call, request.context.time_zone, active_manager
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


def _recording_day_rows(
    camera_id: str,
    start_epoch: float,
    end_epoch: float,
    source: str,
    *,
    fresh: bool = False,
    active_manager: AppManager | None = None,
) -> list[dict]:
    selected_manager = active_manager or manager
    selected_source = recording_source(source)
    cache_key = (camera_id, selected_source, int(start_epoch), int(end_epoch))
    now = time.monotonic()
    if not fresh:
        with RECORDING_DAY_CACHE_LOCK:
            cached = RECORDING_DAY_CACHE.get(cache_key)
            near_live = end_epoch >= time.time() - max(
                30.0,
                selected_manager.recorder.segment_seconds * 3,
            )
            cache_seconds = (
                RECORDING_NEAR_LIVE_CACHE_SECONDS
                if near_live
                else RECORDING_DAY_CACHE_SECONDS
            )
            if cached is not None and now - cached[0] < cache_seconds:
                selected_manager.recorder.lease_recordings_for_playback(cached[1])
                return cached[1]
    rows = [
        row for row in selected_manager.recorder.recording_rows_between(
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
            discover_missing=False,
        )
        if int(row.get("size_bytes") or 0) > 1024
    ]
    if fresh:
        rows = selected_manager.recorder.discard_missing_recording_rows(rows)
    selected_manager.recorder.lease_recordings_for_playback(rows)
    selected_manager.recorder.queue_stream_fingerprints(rows)
    with RECORDING_DAY_CACHE_LOCK:
        RECORDING_DAY_CACHE[cache_key] = (now, rows)
        expired = [key for key, value in RECORDING_DAY_CACHE.items() if now - value[0] >= RECORDING_DAY_CACHE_SECONDS]
        for key in expired:
            RECORDING_DAY_CACHE.pop(key, None)
    return rows


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


def _recording_preview_path(
    row: dict,
    epoch: float,
    *,
    active_manager: AppManager | None = None,
) -> Path:
    """Return a small cached JPEG near an epoch without mutating playback."""
    selected_manager = active_manager or manager
    selected_config = getattr(selected_manager, "config", config)
    source_path = _recording_storage_path(
        row.get("path"),
        active_manager=selected_manager,
    )
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
    cache_dir = selected_manager.database_dir / "recording-preview-cache"
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
                selected_config.ffmpeg_path,
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


def _recording_day_fmp4_paths(
    camera_id: str,
    segment_name: str,
    start_epoch: float,
    end_epoch: float,
    source: str = "main",
    media_offset: float = 0.0,
    trim_end: bool = False,
    *,
    active_manager: AppManager | None = None,
) -> tuple[Path, Path]:
    selected_manager = active_manager or manager
    if selected_manager.camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="camera not found")
    _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
    if not math.isfinite(media_offset) or media_offset < 0:
        raise HTTPException(status_code=400, detail="invalid recording media offset")
    rows = _recording_day_rows(
        camera_id,
        start_epoch,
        end_epoch,
        source,
        active_manager=selected_manager,
    )
    if not segment_name or Path(segment_name).name != segment_name:
        raise HTTPException(status_code=404, detail="recording segment not found")
    segment_index = next(
        (index for index, row in enumerate(rows) if str(row.get("name") or "") == segment_name),
        None,
    )
    if segment_index is None:
        raise HTTPException(status_code=404, detail="recording segment not found")
    row = rows[segment_index]
    path = _recording_storage_path(
        row.get("path"),
        active_manager=selected_manager,
    )
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


def _recording_rows(
    camera_id: str,
    limit: int,
    source: str = "main",
    *,
    active_manager: AppManager | None = None,
) -> list[dict]:
    selected_manager = active_manager or manager
    return selected_manager.recorder.recording_rows(
        camera_id,
        limit=limit,
        source=recording_source(source),
    )


def _recording_storage_path(
    value: object,
    *,
    active_manager: AppManager | None = None,
) -> Path:
    selected_manager = active_manager or manager
    if not value:
        raise HTTPException(status_code=404, detail="recording file not found")
    try:
        path = Path(str(value)).resolve(strict=True)
        path.relative_to(selected_manager.recorder.recordings_dir.resolve())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="recording file not found") from None
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="recording file is outside storage") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="recording file not found")
    return path










def _event_clip_window(
    before: float | None,
    after: float | None,
    *,
    active_manager: AppManager | None = None,
) -> tuple[float, float]:
    selected_config = (active_manager or manager).config
    try:
        return event_clip_window(
            selected_config.event_clip_before_seconds,
            selected_config.event_clip_after_seconds,
            before,
            after,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _event_clip_path(
    event: dict,
    before: float,
    after: float,
    source: str = "main",
    *,
    active_manager: AppManager | None = None,
) -> Path:
    selected_manager = active_manager or manager
    event_id = int(event.get("id") or 0)
    camera_id = slugify_camera_id(str(event.get("camera_id") or "camera"))
    safe_before = int(max(0.0, min(float(before), 3600.0)) * 1000)
    safe_after = int(max(0.0, min(float(after), 3600.0)) * 1000)
    clip_source = recording_source(source)
    clip_dir = selected_manager.storage_dir / "event_clips" / camera_id / clip_source
    clip_dir.mkdir(parents=True, exist_ok=True)
    accel_mode = _hardware_acceleration_mode()
    return clip_dir / f"{event_id}-{safe_before}-{safe_after}-a3-{accel_mode}.mp4"


def _ensure_event_clip(
    event: dict,
    *,
    before: float,
    after: float,
    source: str = "main",
    active_manager: AppManager | None = None,
) -> Path:
    selected_manager = active_manager or manager
    clip_source = recording_source(source)
    clip_path = _event_clip_path(
        event,
        before=before,
        after=after,
        source=clip_source,
        active_manager=selected_manager,
    )
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
                active_manager=selected_manager,
            )
        finally:
            EVENT_CLIP_BUILD_LIMITER.release()
    return clip_path


def _build_event_clip(
    event: dict,
    before: float,
    after: float,
    output_path: Path,
    source: str = "main",
    *,
    active_manager: AppManager | None = None,
) -> None:
    selected_manager = active_manager or manager
    camera_id = str(event.get("camera_id") or "")
    if not camera_id:
        raise HTTPException(status_code=400, detail="event is missing camera")

    event_created_epoch = event_epoch(event)
    window_before = max(0.0, min(float(before), 3600.0))
    window_after = max(0.0, min(float(after), 3600.0))
    window_start = event_created_epoch - window_before
    window_end = event_created_epoch + window_after

    rows: list[dict] = []
    for candidate in selected_manager.recorder.recording_rows_between(
            camera_id,
            window_start,
            window_end,
            recording_source(source),
            discover_missing=False,
        ):
        if candidate.get("start_epoch") is None or candidate.get("end_epoch") is None:
            continue
        try:
            candidate = {
                **candidate,
                "path": str(
                    _recording_storage_path(
                        candidate.get("path"),
                        active_manager=selected_manager,
                    )
                ),
            }
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

    concat_path = _write_concat_file(selected, active_manager=selected_manager)
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


def _write_concat_file(
    rows: list[dict],
    *,
    active_manager: AppManager | None = None,
) -> Path:
    selected_manager = active_manager or manager
    paths = [
        str(
            _recording_storage_path(
                row.get("path"),
                active_manager=selected_manager,
            )
        )
        for row in rows
    ]
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


# Assemble incident/event queries after all assistant consumers are defined.
_incident_query_bundle = create_incident_query_router(
    IncidentQueryDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
    ),
    INCIDENT_QUERIES,
)
app.include_router(_incident_query_bundle.router)

events = _incident_query_bundle.handlers["events"]
incidents = _incident_query_bundle.handlers["incidents"]
incident_feed = _incident_query_bundle.handlers["incident_feed"]
incident_detail = _incident_query_bundle.handlers["incident_detail"]
incident_search = _incident_query_bundle.handlers["incident_search"]
incident_for_event = _incident_query_bundle.handlers["incident_for_event"]


_detection_route_bundle = create_detection_router(
    DetectionRouteDependencies(
        get_manager=lambda: manager,
        get_config=lambda: config,
        manager_lock=MANAGER_RELOAD_LOCK,
        get_comparison_limiter=lambda: TRACKING_COMPARISON_LIMITER,
        ensure_event_clip=lambda *args, **kwargs: _ensure_event_clip(*args, **kwargs),
        dependency_status=lambda: ultralytics_deepocsort_dependency_status(),
        comparison_runner=lambda *args, **kwargs: TrackingComparisonRunner(
            *args, **kwargs
        ),
        sample_video_frames=lambda *args, **kwargs: sampled_video_frames(
            *args, **kwargs
        ),
    )
)
app.include_router(_detection_route_bundle.router)

detect_event_snapshot = _detection_route_bundle.handlers["detect_event_snapshot"]
tracking_comparison_history = _detection_route_bundle.handlers[
    "tracking_comparison_history"
]
update_tracking_comparison_verdict = _detection_route_bundle.handlers[
    "update_tracking_comparison_verdict"
]
compare_event_tracking = _detection_route_bundle.handlers["compare_event_tracking"]
detect_debug_frame = _detection_route_bundle.handlers["detect_debug_frame"]


_appearance_route_bundle = create_appearance_router(
    AppearanceRouteDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
    )
)
app.include_router(_appearance_route_bundle.router)

event_snapshot = _appearance_route_bundle.handlers["event_snapshot"]
event_thumbnail = _appearance_route_bundle.handlers["event_thumbnail"]
event_appearance_matches = _appearance_route_bundle.handlers["event_appearance_matches"]
event_related_incidents = _appearance_route_bundle.handlers["event_related_incidents"]
appearance_index_status = _appearance_route_bundle.handlers["appearance_index_status"]
queue_appearance_backfill = _appearance_route_bundle.handlers["queue_appearance_backfill"]


_face_route_bundle = create_face_router(
    FaceRouteDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        start_observation_sync=_start_face_observation_sync,
    )
)
app.include_router(_face_route_bundle.router)

face_status = _face_route_bundle.handlers["face_status"]
face_people = _face_route_bundle.handlers["face_people"]
create_face_person = _face_route_bundle.handlers["create_face_person"]
delete_face_person = _face_route_bundle.handlers["delete_face_person"]
face_observations = _face_route_bundle.handlers["face_observations"]
face_observation_count = _face_route_bundle.handlers["face_observation_count"]
face_observation = _face_route_bundle.handlers["face_observation"]
assign_face_observation = _face_route_bundle.handlers["assign_face_observation"]
face_crop = _face_route_bundle.handlers["face_crop"]


# Assemble the recording HTTP boundary after its legacy media helpers exist.
# The explicit aliases preserve the direct-call surface used by internal tools
# and tests while route ownership lives in recording_routes.
_recording_route_bundle = create_recording_router(
    RecordingRouteDependencies(
        get_manager=lambda: manager,
        get_config=lambda: config,
        get_media_exports=lambda: _media_export_manager(),
        public_url=lambda path: public_url(path),
        recording_rows=lambda active_manager, *args, **kwargs: _recording_rows(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
        recording_day_rows=lambda active_manager, *args, **kwargs: _recording_day_rows(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
        recording_preview_path=lambda active_manager, *args, **kwargs: (
            _recording_preview_path(
                *args,
                active_manager=active_manager,
                **kwargs,
            )
        ),
        recording_day_fmp4_paths=lambda active_manager, *args, **kwargs: (
            _recording_day_fmp4_paths(
                *args,
                active_manager=active_manager,
                **kwargs,
            )
        ),
        recording_file_response=lambda *args, **kwargs: _recording_file_response(
            *args,
            **kwargs,
        ),
        event_clip_window=lambda active_manager, before, after: _event_clip_window(
            before,
            after,
            active_manager=active_manager,
        ),
        ensure_event_clip=lambda active_manager, *args, **kwargs: _ensure_event_clip(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
    )
)
app.include_router(_recording_route_bundle.router)

recordings = _recording_route_bundle.handlers["recordings"]
recording_events = _recording_route_bundle.handlers["recording_events"]
recording_day = _recording_route_bundle.handlers["recording_day"]
recording_grid_day = _recording_route_bundle.handlers["recording_grid_day"]
recording_grid_updates = _recording_route_bundle.handlers["recording_grid_updates"]
_public_media_export = _recording_route_bundle.handlers["_public_media_export"]
create_media_export = _recording_route_bundle.handlers["create_media_export"]
list_media_exports = _recording_route_bundle.handlers["list_media_exports"]
media_export_summary = _recording_route_bundle.handlers["media_export_summary"]
batch_media_exports = _recording_route_bundle.handlers["batch_media_exports"]
_public_media_export_batch = _recording_route_bundle.handlers[
    "_public_media_export_batch"
]
get_media_export = _recording_route_bundle.handlers["get_media_export"]
download_media_export = _recording_route_bundle.handlers["download_media_export"]
play_media_export = _recording_route_bundle.handlers["play_media_export"]
protect_media_export = _recording_route_bundle.handlers["protect_media_export"]
update_media_export_metadata = _recording_route_bundle.handlers[
    "update_media_export_metadata"
]
delete_media_export = _recording_route_bundle.handlers["delete_media_export"]
recording_window = _recording_route_bundle.handlers["recording_window"]
recording_preview = _recording_route_bundle.handlers["recording_preview"]
recording_updates = _recording_route_bundle.handlers["recording_updates"]
recording_day_hls_playlist = _recording_route_bundle.handlers[
    "recording_day_hls_playlist"
]
recording_day_hls_init = _recording_route_bundle.handlers["recording_day_hls_init"]
recording_day_hls_segment = _recording_route_bundle.handlers[
    "recording_day_hls_segment"
]
event_clip = _recording_route_bundle.handlers["event_clip"]
event_stream = _recording_route_bundle.handlers["event_stream"]
