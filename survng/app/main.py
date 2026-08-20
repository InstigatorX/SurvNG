from __future__ import annotations

import json
import logging
import asyncio
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    AppConfig,
    load_config,
    normalize_config,
    save_config,
)
from .config_application import (
    TargetedConfigApplication,
    manager_owned_config,
)
from .config_routes import (
    ConfigRouteDependencies,
    create_config_router,
)
from .appearance_routes import AppearanceRouteDependencies, create_appearance_router
from .camera_api_routes import (
    CameraApiDependencies,
    create_camera_api_router,
)
from .detection_routes import (
    DetectionRouteDependencies,
    create_detection_router,
)
from .face_routes import (
    FaceRouteDependencies,
    create_face_router,
)
from .frontend_routes import FrontendRouteDependencies, create_frontend_router
from .onvif_inspector_routes import create_onvif_inspector_router
from .manager import AppManager, validate_manager_configuration
from .manager_access import ManagerAccessCoordinator
from .manager_reload import ManagerGenerationLifecycle, ManagerReloadHooks
from .media_exports import MediaExportManager
from .model_evaluation import ModelEvaluationRunner
from .incident_queries import (
    IncidentQueryDependencies,
    IncidentQueryService,
    create_incident_query_router,
)
from .intelligence_routes import (
    IntelligenceDependencies,
    create_intelligence_router,
)
from .recording_media_runtime import (
    RecordingMediaDependencies,
    RecordingMediaRuntime,
)
from .recording_routes import (
    RecordingRouteDependencies,
    _recording_playback_window,
    _validate_recording_range,
    create_recording_router,
)
from .semantic_routes import (
    SemanticRouteDependencies,
    create_semantic_router,
)
from .object_tracking import ultralytics_fasttrack_dependency_status
from .operations_routes import OperationsRouteDependencies, create_operations_router
from .tracking_comparison import TrackingComparisonRunner, sampled_video_frames
from .system_telemetry import (
    SystemTelemetryDependencies,
    SystemTelemetryService,
    create_system_telemetry_router,
)
from .system_routes import SystemRouteDependencies, create_system_router
from .training_routes import TrainingRouteDependencies, create_training_router
from .security import authenticate_api_token, redact_secret_text, required_api_scope
from .storage_maintenance import StorageMaintenanceRunner

config = load_config()
manager = AppManager(config)
LOGGER = logging.getLogger(__name__)
LOG_LINES: deque[dict] = deque(maxlen=1000)
FACE_OBSERVATIONS_SYNCED = False
FACE_OBSERVATIONS_SYNC_LOCK = threading.Lock()
FACE_OBSERVATIONS_SYNC_THREAD_LOCK = threading.Lock()
FACE_OBSERVATIONS_SYNC_THREAD: threading.Thread | None = None
MANAGER_RELOAD_LOCK = threading.RLock()
MANAGER_ACCESS = ManagerAccessCoordinator()
APPLICATION_STOPPING = threading.Event()
CONFIG_PROBE_LIMITER = threading.BoundedSemaphore(2)
AUDIT_AI_LIMITER = threading.BoundedSemaphore(1)
ASSISTANT_LIMITER = threading.BoundedSemaphore(2)
AI_ACTIVITY_LOCK = threading.Lock()
AI_ACTIVITY_CONDITION = threading.Condition(AI_ACTIVITY_LOCK)
AI_ACTIVE_OPERATIONS: dict[str, int] = {}
AI_SHUTDOWN_DRAIN_SECONDS = 30.0
TRACKING_COMPARISON_LIMITER = threading.BoundedSemaphore(1)
SYSTEM_TELEMETRY = SystemTelemetryService()
PROCESS_INSTANCE_ID = SYSTEM_TELEMETRY.process_instance_id
INCIDENT_QUERIES = IncidentQueryService()
STORAGE_MAINTENANCE = StorageMaintenanceRunner()
MODEL_EVALUATION = ModelEvaluationRunner(lambda: manager, lambda: config)
SERVER_RESTART_LOCK = threading.Lock()
SERVER_RESTART_SCHEDULED = False


def _begin_ai_operation(kind: str) -> None:
    with AI_ACTIVITY_CONDITION:
        AI_ACTIVE_OPERATIONS[kind] = AI_ACTIVE_OPERATIONS.get(kind, 0) + 1


def _end_ai_operation(kind: str) -> None:
    with AI_ACTIVITY_CONDITION:
        remaining = AI_ACTIVE_OPERATIONS.get(kind, 0) - 1
        if remaining > 0:
            AI_ACTIVE_OPERATIONS[kind] = remaining
        else:
            AI_ACTIVE_OPERATIONS.pop(kind, None)
        AI_ACTIVITY_CONDITION.notify_all()


def _active_ai_operations() -> dict[str, int]:
    with AI_ACTIVITY_LOCK:
        return dict(AI_ACTIVE_OPERATIONS)


def _wait_for_ai_operations(timeout: float) -> dict[str, int]:
    deadline = time.monotonic() + max(0.0, timeout)
    with AI_ACTIVITY_CONDITION:
        while AI_ACTIVE_OPERATIONS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            AI_ACTIVITY_CONDITION.wait(timeout=remaining)
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
        api_path = _api_scope_path(scope)
        if (
            scope_type in {"http", "websocket"}
            and api_path.startswith("/api/")
            and api_path != "/api/health"
            and config.api_auth.enabled
        ):
            authorization = _scope_header(scope, b"authorization")
            principal = authenticate_api_token(authorization, config.api_auth)
            required_scope = required_api_scope(
                str(scope.get("method") or "GET"), api_path
            )
            if principal is None or not principal.permits(required_scope):
                if scope_type == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    status = 401 if principal is None else 403
                    headers = {"Cache-Control": "no-store"}
                    if status == 401:
                        headers["WWW-Authenticate"] = 'Bearer realm="SurvNG"'
                    response = JSONResponse(
                        {
                            "detail": (
                                "valid bearer token required"
                                if status == 401
                                else f"API token requires {required_scope} scope"
                            )
                        },
                        status_code=status,
                        headers=headers,
                    )
                    await response(scope, receive, send)
                return
        if (
            scope_type == "http"
            and APPLICATION_STOPPING.is_set()
            and str(scope.get("method") or "GET").upper() not in {"GET", "HEAD", "OPTIONS"}
            and api_path.startswith("/api/")
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
            and api_path.startswith("/api/")
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


def _request_server_restart() -> dict[str, object]:
    """Queue a supervisor-owned restart after the HTTP response can be sent."""
    global SERVER_RESTART_SCHEDULED
    with SERVER_RESTART_LOCK:
        if SERVER_RESTART_SCHEDULED or APPLICATION_STOPPING.is_set():
            raise RuntimeError("SurvNG is already restarting or shutting down")
        active_tasks = _active_storage_tasks(manager)
        if active_tasks:
            raise RuntimeError(
                "SurvNG cannot restart while storage work is active: "
                f"{', '.join(active_tasks)}. Wait for it to finish or cancel it from Maintenance."
            )
        SERVER_RESTART_SCHEDULED = True
        APPLICATION_STOPPING.set()

    def restart_from_supervisor() -> None:
        global SERVER_RESTART_SCHEDULED
        # Give Uvicorn enough time to flush the accepted response before
        # systemd begins SurvNG's normal graceful-stop sequence.
        time.sleep(0.75)
        try:
            subprocess.run(
                ["systemctl", "--no-block", "restart", "survng.service"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except Exception as error:
            LOGGER.error(
                "server restart request failed: %s",
                redact_secret_text(error),
            )
            with SERVER_RESTART_LOCK:
                SERVER_RESTART_SCHEDULED = False
                APPLICATION_STOPPING.clear()

    threading.Thread(
        target=restart_from_supervisor,
        name="server-restart-request",
        daemon=True,
    ).start()
    return {
        "ok": True,
        "status": "restart_scheduled",
        "instance_id": PROCESS_INSTANCE_ID,
    }


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
        _recording_media_runtime.rebind_media_exports()
        _recording_media_runtime.clear_runtime_caches()

    lifecycle = ManagerGenerationLifecycle(
        lock=MANAGER_RELOAD_LOCK,
        stopping=APPLICATION_STOPPING,
        manager_factory=AppManager,
        hooks=ManagerReloadHooks(
            active_storage_tasks=_active_storage_tasks,
            active_ai_operations=_active_ai_operations,
            prewarmer_running=_recording_media_runtime.prewarmer_running,
            stop_prewarmer=_recording_media_runtime._stop_recording_prewarmer,
            start_prewarmer=_recording_media_runtime._start_recording_prewarmer,
            save_config=save_config,
            publish_runtime=publish_runtime,
            refresh_runtime_caches=refresh_runtime_caches,
            storage_error=StorageTasksActiveError,
            ai_error=AiOperationsActiveError,
            wait_for_manager_idle=lambda active: MANAGER_ACCESS.wait_idle(
                active,
                15.0,
            ),
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
    previous_ffmpeg_path = config.ffmpeg_path
    effective, result = application.apply(
        config,
        effective,
        manager,
        persist=persist,
    )
    config = effective
    if effective.ffmpeg_path != previous_ffmpeg_path:
        _recording_media_runtime.clear_hardware_probe_caches()
    return effective, result


def _record_process_lifecycle(kind: str) -> None:
    """Record restart evidence without making telemetry a lifecycle dependency."""
    try:
        manager.telemetry.record_lifecycle_event(PROCESS_INSTANCE_ID, kind)
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
        media_exports = _recording_media_runtime._media_export_manager()
        media_exports.start()
        _start_face_observation_sync()
        _recording_media_runtime._start_recording_prewarmer()
        calibration_monitor_task = asyncio.create_task(
            _intelligence_route_bundle.service._calibration_followup_monitor(),
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
        remaining_ai = _wait_for_ai_operations(AI_SHUTDOWN_DRAIN_SECONDS)
        if remaining_ai:
            logging.getLogger("uvicorn.error").warning(
                "AI work did not drain before shutdown: %s",
                remaining_ai,
            )
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
                    _recording_media_runtime._stop_recording_prewarmer()
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
            validate_config=validate_manager_configuration,
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
            request_server_restart=_request_server_restart,
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







def _sync_face_observations(limit: int = 5000) -> int:
    global FACE_OBSERVATIONS_SYNCED
    with FACE_OBSERVATIONS_SYNC_LOCK:
        if FACE_OBSERVATIONS_SYNCED:
            return 0
        inserted = 0
        while not APPLICATION_STOPPING.is_set():
            with MANAGER_ACCESS.lease(
                MANAGER_RELOAD_LOCK,
                lambda: manager,
            ) as active_manager:
                inserted += active_manager.faces.ingest_events(
                    active_manager.events.recent(max(1, min(limit, 20000)))
                )
            with MANAGER_RELOAD_LOCK:
                if manager is active_manager:
                    FACE_OBSERVATIONS_SYNCED = True
                    return inserted
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

_frontend_route_bundle = create_frontend_router(
    FrontendRouteDependencies(frontend_response=frontend_response)
)
app.include_router(_frontend_route_bundle.router)
app.include_router(create_onvif_inspector_router())
health = _frontend_route_bundle.handlers["health"]
favicon = _frontend_route_bundle.handlers["favicon"]
apple_touch_icon = _frontend_route_bundle.handlers["apple_touch_icon"]
index = _frontend_route_bundle.handlers["index"]
recordings_page = _frontend_route_bundle.handlers["recordings_page"]
recording_search_page = _frontend_route_bundle.handlers["recording_search_page"]
recording_exports_page = _frontend_route_bundle.handlers["recording_exports_page"]
timeline_page = _frontend_route_bundle.handlers["timeline_page"]
timeline_exports_page = _frontend_route_bundle.handlers["timeline_exports_page"]
exports_page = _frontend_route_bundle.handlers["exports_page"]
search_page = _frontend_route_bundle.handlers["search_page"]
config_page = _frontend_route_bundle.handlers["config_page"]
admin_page = _frontend_route_bundle.handlers["admin_page"]
incidents_page = _frontend_route_bundle.handlers["incidents_page"]
faces_page = _frontend_route_bundle.handlers["faces_page"]
people_page = _frontend_route_bundle.handlers["people_page"]
live_page = _frontend_route_bundle.handlers["live_page"]
onvif_page = _frontend_route_bundle.handlers["onvif_page"]

_semantic_route_bundle = create_semantic_router(
    SemanticRouteDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        manager_access=MANAGER_ACCESS,
    )
)
app.include_router(_semantic_route_bundle.router)
semantic_search_status = _semantic_route_bundle.handlers["semantic_search_status"]
semantic_search = _semantic_route_bundle.handlers["semantic_search"]

_system_route_bundle = create_system_router(
    SystemRouteDependencies(
        get_manager=lambda: manager,
        get_config=lambda: config,
        system_telemetry=SYSTEM_TELEMETRY,
        ffprobe_path=_ffprobe_path,
        ffplay_path=_ffplay_path,
        ffmpeg_qsv_info=_recording_media_runtime._ffmpeg_qsv_info,
        ffmpeg_vaapi_info=_recording_media_runtime._ffmpeg_vaapi_info,
        hardware_acceleration_mode=(
            _recording_media_runtime._hardware_acceleration_mode
        ),
        event_clip_window=lambda before, after: (
            _recording_media_runtime._event_clip_window(before, after)
        ),
        recording_cache_status=_recording_media_runtime.cache_status,
        model_evaluation=MODEL_EVALUATION,
    )
)
app.include_router(_system_route_bundle.router)
cameras = _system_route_bundle.handlers["cameras"]
application_event_stream = _system_route_bundle.handlers["application_event_stream"]
get_motion_pipeline_catalog = _system_route_bundle.handlers[
    "get_motion_pipeline_catalog"
]
accelerator = _system_route_bundle.handlers["accelerator"]
detector_status = _system_route_bundle.handlers["detector_status"]
object_tracking_catalog = _system_route_bundle.handlers["object_tracking_catalog"]
detector_models = _system_route_bundle.handlers["detector_models"]
event_clip_settings = _system_route_bundle.handlers["event_clip_settings"]
recording_cache_status = _system_route_bundle.handlers["recording_cache_status"]


# Assemble incident/event queries after all assistant consumers are defined.
_incident_query_bundle = create_incident_query_router(
    IncidentQueryDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        manager_access=MANAGER_ACCESS,
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
        media_export_manager=lambda: (
            _recording_media_runtime._media_export_manager()
        ),
    )
)
app.include_router(_intelligence_route_bundle.router)


_camera_api_route_bundle = create_camera_api_router(
    CameraApiDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        manager_access=MANAGER_ACCESS,
    )
)
app.include_router(_camera_api_route_bundle.router)

snapshot = _camera_api_route_bundle.handlers["snapshot"]
zone_snapshot = _camera_api_route_bundle.handlers["zone_snapshot"]
live_info = _camera_api_route_bundle.handlers["live_info"]
stream_source = _camera_api_route_bundle.handlers["stream_source"]
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
        ensure_event_clip=lambda *args, **kwargs: (
            _recording_media_runtime._ensure_event_clip(*args, **kwargs)
        ),
        dependency_status=lambda: ultralytics_fasttrack_dependency_status(),
        comparison_runner=lambda *args, **kwargs: TrackingComparisonRunner(
            *args, **kwargs
        ),
        sample_video_frames=lambda *args, **kwargs: sampled_video_frames(
            *args, **kwargs
        ),
        manager_access=MANAGER_ACCESS,
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
        manager_access=MANAGER_ACCESS,
        resolve_incident=lambda active, event_id: INCIDENT_QUERIES.resolve_event(
            active, event_id
        ),
    )
)
app.include_router(_appearance_route_bundle.router)

event_snapshot = _appearance_route_bundle.handlers["event_snapshot"]
event_thumbnail = _appearance_route_bundle.handlers["event_thumbnail"]
event_appearance_matches = _appearance_route_bundle.handlers["event_appearance_matches"]
event_related_incidents = _appearance_route_bundle.handlers["event_related_incidents"]
appearance_index_status = _appearance_route_bundle.handlers["appearance_index_status"]
queue_appearance_backfill = _appearance_route_bundle.handlers["queue_appearance_backfill"]


app.include_router(
    create_training_router(
        TrainingRouteDependencies(
            get_config=lambda: config,
            get_manager=lambda: manager,
            manager_lock=MANAGER_RELOAD_LOCK,
            manager_access=MANAGER_ACCESS,
        )
    )
)


_face_route_bundle = create_face_router(
    FaceRouteDependencies(
        get_manager=lambda: manager,
        manager_lock=MANAGER_RELOAD_LOCK,
        start_observation_sync=_start_face_observation_sync,
        manager_access=MANAGER_ACCESS,
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
        get_media_exports=lambda: _recording_media_runtime._media_export_manager(),
        public_url=lambda path: public_url(path),
        recording_rows=lambda active_manager, *args, **kwargs: _recording_media_runtime._recording_rows(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
        recording_day_rows=lambda active_manager, *args, **kwargs: _recording_media_runtime._recording_day_rows(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
        recording_preview_path=lambda active_manager, *args, **kwargs: (
            _recording_media_runtime._recording_preview_path(
                *args,
                active_manager=active_manager,
                **kwargs,
            )
        ),
        recording_preview_timestamp=(
            _recording_media_runtime._recording_preview_timestamp
        ),
        recording_day_fmp4_paths=lambda active_manager, *args, **kwargs: (
            _recording_media_runtime._recording_day_fmp4_paths(
                *args,
                active_manager=active_manager,
                **kwargs,
            )
        ),
        recording_file_response=lambda *args, **kwargs: _recording_media_runtime._recording_file_response(
            *args,
            **kwargs,
        ),
        event_clip_window=lambda active_manager, before, after: _recording_media_runtime._event_clip_window(
            before,
            after,
            active_manager=active_manager,
        ),
        ensure_event_clip=lambda active_manager, *args, **kwargs: _recording_media_runtime._ensure_event_clip(
            *args,
            active_manager=active_manager,
            **kwargs,
        ),
        manager_lock=MANAGER_RELOAD_LOCK,
        manager_access=MANAGER_ACCESS,
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
