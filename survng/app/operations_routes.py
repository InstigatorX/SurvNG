"""Operator logs, retention, and storage-maintenance HTTP boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .manager import AppManager
from .product_update import ProductUpdateService
from .storage_maintenance import StorageMaintenanceRunner, StorageReconciler


class StorageMaintenanceRequest(BaseModel):
    apply: bool = False
    full: bool = False


class RecordingRetentionRequest(BaseModel):
    apply: bool = False


class ProductUpdateRequest(BaseModel):
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class OperationsRouteDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    log_rows: Callable[[], Sequence[dict[str, Any]]]
    storage_maintenance: StorageMaintenanceRunner
    request_server_restart: Callable[[], dict[str, Any]]
    product_update: ProductUpdateService


def create_operations_router(deps: OperationsRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/logs")
    def logs(limit: int = 300, level: str = "", q: str = "") -> dict[str, Any]:
        safe_limit = max(1, min(limit, 1000))
        wanted_level = level.strip().upper()
        query = q.strip().lower()
        history = list(deps.log_rows())
        rows = history[-safe_limit:]
        if wanted_level:
            levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            try:
                allowed = set(levels[levels.index(wanted_level) :])
                rows = [row for row in rows if row.get("level") in allowed]
            except ValueError:
                rows = [row for row in rows if row.get("level") == wanted_level]
        if query:
            rows = [
                row
                for row in rows
                if query
                in f"{row.get('level', '')} {row.get('logger', '')} {row.get('message', '')}".lower()
            ]
        return {"lines": rows[-safe_limit:], "total": len(history)}

    @router.get("/api/retention/status")
    def recording_retention_status() -> dict[str, Any]:
        with deps.manager_lock:
            return deps.get_manager().recording.retention_status()

    @router.post("/api/retention/run", status_code=202)
    def run_recording_retention(
        request: RecordingRetentionRequest,
    ) -> dict[str, Any]:
        with deps.manager_lock:
            return deps.get_manager().recording.request_retention_run(
                apply=request.apply
            )

    @router.get("/api/maintenance/storage")
    def storage_maintenance_status() -> dict[str, Any]:
        return deps.storage_maintenance.status()

    @router.post("/api/maintenance/storage", status_code=202)
    def start_storage_maintenance(
        request: StorageMaintenanceRequest,
    ) -> dict[str, Any]:
        with deps.manager_lock:
            active_manager = deps.get_manager()
            storage_dir = active_manager.storage_dir
            db_path = active_manager.events.db_path
            database_write_lock = active_manager.database_write_lock
            recorder = active_manager.recorder
            media_storage = active_manager.media_storage
        try:
            return deps.storage_maintenance.start(
                lambda cancel_event, progress: StorageReconciler(
                    storage_dir,
                    db_path,
                    recorder,
                    media_storage=media_storage,
                    cancel_event=cancel_event,
                    progress=progress,
                    database_write_lock=database_write_lock,
                ),
                apply=request.apply,
                full=request.full,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.delete("/api/maintenance/storage", status_code=202)
    def cancel_storage_maintenance() -> dict[str, Any]:
        try:
            return deps.storage_maintenance.cancel()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post("/api/system/restart", status_code=202)
    def restart_server() -> dict[str, Any]:
        try:
            return deps.request_server_restart()
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/api/system/update")
    def product_update_status(
        refresh_remote: bool = False,
        branch: str | None = None,
    ) -> dict[str, Any]:
        return deps.product_update.status(
            refresh_remote=refresh_remote,
            branch=branch,
        )

    @router.post("/api/system/update", status_code=202)
    def start_product_update(
        request: ProductUpdateRequest | None = None,
    ) -> dict[str, Any]:
        try:
            return deps.product_update.start(
                branch=(request.branch if request is not None else None),
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return router
