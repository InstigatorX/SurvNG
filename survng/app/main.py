from __future__ import annotations

import json
import logging
import math
import asyncio
import queue
import signal
import secrets
import platform
import shutil
import time
import threading
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
from .recording_media_runtime import (
    RecordingMediaDependencies,
    RecordingMediaRuntime,
    RecordingPrewarmCancelled,
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
TRACKING_COMPARISON_LIMITER = threading.BoundedSemaphore(1)
SYSTEM_TELEMETRY = SystemTelemetryService()
PROCESS_INSTANCE_ID = SYSTEM_TELEMETRY.process_instance_id
INCIDENT_QUERIES = IncidentQueryService()
STORAGE_MAINTENANCE = StorageMaintenanceRunner()


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
    exports = _recording_media_runtime.active_export_jobs()
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
        _recording_media_runtime.clear_runtime_caches()

    lifecycle = ManagerGenerationLifecycle(
        lock=MANAGER_RELOAD_LOCK,
        stopping=APPLICATION_STOPPING,
        manager_factory=AppManager,
        hooks=ManagerReloadHooks(
            active_storage_tasks=_active_storage_tasks,
            active_ai_operations=_active_ai_operations,
            prewarmer_running=_recording_media_runtime.prewarmer_running,
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
        active_exports=_recording_media_runtime.active_export_jobs,
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
    return _recording_media_runtime.cache_status()



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


_recording_media_runtime = RecordingMediaRuntime(
    RecordingMediaDependencies(
        get_config=lambda: config,
        get_manager=lambda: manager,
        ffprobe_path=_ffprobe_path,
        validate_recording_range=_validate_recording_range,
        recording_playback_window=_recording_playback_window,
    )
)

# Transitional aliases keep direct internal/test callers stable until the
# final composition-root campaign removes the legacy main-module surface.
for _recording_media_name in (
    "_run_ffmpeg_list",
    "_dri_render_devices",
    "_ffmpeg_qsv_info",
    "_ffmpeg_vaapi_info",
    "_hardware_acceleration_mode",
    "_media_export_hardware_backend",
    "_media_export_hardware_device",
    "_media_export_manager",
    "_probe_video_codec",
    "_mp4_boxes",
    "_mp4_track_timescales",
    "_offset_fmp4_timestamps",
    "_event_clip_cache_suffix",
    "_event_clip_vaapi_enabled",
    "_event_clip_qsv_enabled",
    "_event_clip_cpu_command",
    "_event_clip_vaapi_command",
    "_event_clip_qsv_command",
    "_recording_day_rows",
    "_recording_cache_metric",
    "_signal_recording_prewarm_process",
    "_run_recording_remux",
    "_recording_fmp4_files",
    "_recording_cache_files_ready",
    "_recording_file_response",
    "_recording_preview_path",
    "_recording_preview_ready",
    "_maintain_recording_preview_cache",
    "_maintain_recording_cache",
    "_start_recording_prewarmer",
    "_stop_recording_prewarmer",
    "_recording_prewarm_loop",
    "_recording_day_fmp4_paths",
    "_recording_rows",
    "_recording_storage_path",
    "_event_clip_window",
    "_event_clip_path",
    "_ensure_event_clip",
    "_build_event_clip",
    "_write_concat_file",
    "_recording_start_epoch",
):
    globals()[_recording_media_name] = getattr(
        _recording_media_runtime, _recording_media_name
    )





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
