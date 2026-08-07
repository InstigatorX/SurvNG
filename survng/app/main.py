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
from .camera_api_routes import (
    CameraApiDependencies,
    CameraFeatureState,
    create_camera_api_router,
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
from .intelligence_routes import (
    AuditAiApplyRequest,
    CalibrationApplyRequest,
    CalibrationRollbackRequest,
    CalibrationRunRequest,
    CameraIntelligenceApplyRequest,
    CameraIntelligenceFollowupRequest,
    IncidentAiApplyRequest,
    IntelligenceDependencies,
    MotionAiReviewRequest,
    create_intelligence_router,
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
from .operations_routes import OperationsRouteDependencies, create_operations_router
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
app.include_router(
    create_operations_router(
        OperationsRouteDependencies(
            get_manager=lambda: manager,
            manager_lock=MANAGER_RELOAD_LOCK,
            log_rows=lambda: tuple(LOG_LINES),
            storage_maintenance=STORAGE_MAINTENANCE,
        )
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


_intelligence_route_bundle = create_intelligence_router(
    IntelligenceDependencies(
        get_config=lambda: config,
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        get_audit_ai_limiter=lambda: AUDIT_AI_LIMITER,
        get_assistant_limiter=lambda: ASSISTANT_LIMITER,
        application_stopping=APPLICATION_STOPPING,
        incident_queries=INCIDENT_QUERIES,
        system_telemetry=SYSTEM_TELEMETRY,
        apply_config_update=lambda *args, **kwargs: apply_config_update(
            *args, **kwargs
        ),
        begin_ai_operation=lambda *args, **kwargs: _begin_ai_operation(*args, **kwargs),
        end_ai_operation=lambda *args, **kwargs: _end_ai_operation(*args, **kwargs),
        media_export_manager=lambda: _media_export_manager(),
    )
)
app.include_router(_intelligence_route_bundle.router)

# Transitional direct-call aliases remain until the final composition cleanup.
for _intelligence_name, _intelligence_handler in (
    _intelligence_route_bundle.handlers.items()
):
    globals()[_intelligence_name] = _intelligence_handler


_camera_api_route_bundle = create_camera_api_router(
    CameraApiDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
    )
)
app.include_router(_camera_api_route_bundle.router)

snapshot = _camera_api_route_bundle.handlers["snapshot"]
zone_snapshot = _camera_api_route_bundle.handlers["zone_snapshot"]
live_info = _camera_api_route_bundle.handlers["live_info"]
stream = _camera_api_route_bundle.handlers["stream"]
relay_go2rtc_websocket = _camera_api_route_bundle.handlers[
    "relay_go2rtc_websocket"
]
webrtc_signaling = _camera_api_route_bundle.handlers["webrtc_signaling"]
mse_stream = _camera_api_route_bundle.handlers["mse_stream"]
start_camera = _camera_api_route_bundle.handlers["start_camera"]
stop_camera = _camera_api_route_bundle.handlers["stop_camera"]
motion_test = _camera_api_route_bundle.handlers["motion_test"]
motion_debug_status = _camera_api_route_bundle.handlers["motion_debug_status"]
set_motion_debug = _camera_api_route_bundle.handlers["set_motion_debug"]
motion_debug_image = _camera_api_route_bundle.handlers["motion_debug_image"]
start_recording = _camera_api_route_bundle.handlers["start_recording"]
stop_recording = _camera_api_route_bundle.handlers["stop_recording"]
set_camera_recording = _camera_api_route_bundle.handlers["set_camera_recording"]
set_camera_detection = _camera_api_route_bundle.handlers["set_camera_detection"]


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
