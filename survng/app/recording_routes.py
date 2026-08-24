"""Recording history, grid, preview, and export HTTP boundary."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .incident_presenter import (
    _event_row,
    _incident_list_payload,
    _incident_rows,
    _recording_event_row,
    _recording_grid_incident_payload,
)
from .incident_utils import DEFAULT_INCIDENT_GAP_SECONDS, event_epoch
from .media_exports import MediaExportManager
from .manager_access import ManagerAccessCoordinator, guard_manager_generation
from .recording_media import (
    playback_segment_duration,
)

RECORDING_LOOKUP_LIMIT = 20_000
RECORDING_PLAYBACK_WINDOW_SECONDS = 15 * 60


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


@dataclass(frozen=True, slots=True)
class RecordingRouteDependencies:
    get_manager: Callable[[], Any]
    get_config: Callable[[], Any]
    get_media_exports: Callable[[], MediaExportManager]
    public_url: Callable[[str], str]
    recording_rows: Callable[..., list[dict]]
    recording_day_rows: Callable[..., list[dict]]
    recording_preview_path: Callable[[dict, float], Path]
    recording_preview_timestamp: Callable[[Path], tuple[float | None, str]]
    recording_day_fmp4_paths: Callable[..., tuple[Path, Path]]
    recording_file_response: Callable[[Path, str], FileResponse]
    event_clip_window: Callable[..., tuple[float, float]]
    ensure_event_clip: Callable[..., Path]
    manager_lock: threading.RLock | None = None
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class RecordingRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def recording_source(source: str = "main") -> str:
    return "live" if source == "live" else "main"


def _require_recording_camera(
    deps: RecordingRouteDependencies,
    camera_id: str,
) -> Any:
    active_manager = deps.get_manager()
    if active_manager.camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return active_manager


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



def _identity_hydrated_recording_incidents(
    active_manager: Any,
    public_events: list[dict],
    *,
    include_identities: bool = True,
) -> list[dict]:
    incidents = _incident_rows(public_events)
    if not include_identities:
        return incidents
    face_store = getattr(active_manager, "faces", None)
    if face_store is None or not hasattr(face_store, "for_event_ids"):
        return incidents
    from .incident_queries import IncidentQueryService
    return IncidentQueryService.with_faces(active_manager, incidents)


def _public_recording_row(row: dict) -> dict:
    payload = dict(row)
    payload.pop("path", None)
    return payload


def _recording_playback_window(epoch: float) -> tuple[float, float]:
    start = (
        math.floor(epoch / RECORDING_PLAYBACK_WINDOW_SECONDS)
        * RECORDING_PLAYBACK_WINDOW_SECONDS
    )
    return start, start + RECORDING_PLAYBACK_WINDOW_SECONDS


def create_recording_router(deps: RecordingRouteDependencies) -> RecordingRouteBundle:
    router = APIRouter()

    @router.get("/api/cameras/{camera_id}/recordings")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recordings(camera_id: str, limit: int = 200, source: str = "main") -> list[dict]:
        active_manager = _require_recording_camera(deps, camera_id)
        rows = deps.recording_rows(
            active_manager,
            camera_id,
            limit=max(1, min(limit, RECORDING_LOOKUP_LIMIT)),
            source=recording_source(source),
        )
        return [_public_recording_row(row) for row in rows]


    @router.get("/api/cameras/{camera_id}/recordings/events")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_events(camera_id: str, limit: int = 1000, source: str = "main") -> list[dict]:
        active_manager = _require_recording_camera(deps, camera_id)
        rows = deps.recording_rows(
            active_manager,
            camera_id,
            limit=RECORDING_LOOKUP_LIMIT,
            source=recording_source(source),
        )
        if not rows:
            return []
        start_epoch = rows[0].get("start_epoch")
        end_epoch = rows[-1].get("end_epoch")
        if start_epoch is None or end_epoch is None:
            return []

        events = active_manager.events.for_camera_range(
            camera_id,
            datetime.fromtimestamp(float(start_epoch), timezone.utc).isoformat(),
            datetime.fromtimestamp(float(end_epoch), timezone.utc).isoformat(),
            limit=max(1, min(limit, 5000)),
        )
        return [_recording_event_row(event, rows) for event in events]


    @router.get("/api/cameras/{camera_id}/recordings/day")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_day(
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        include_identities: bool = True,
    ) -> dict:
        active_manager = _require_recording_camera(deps, camera_id)
        _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
        selected_source = recording_source(source)
        source_availability = {
            candidate: active_manager.recorder.recording_availability_between(
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
        events = active_manager.events.for_camera_range(
            camera_id,
            datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
            datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
            limit=5000,
        )
        public_events = [_event_row(event) for event in events]
        timeline_incidents = _identity_hydrated_recording_incidents(
            active_manager,
            public_events,
            include_identities=include_identities,
        )
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
                for incident in timeline_incidents
            ],
            "available_sources": available_sources,
        }


    @router.get("/api/recordings/grid/day")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_grid_day(
        start_epoch: float,
        end_epoch: float,
        source: str = "live",
        include_identities: bool = True,
    ) -> dict:
        """Return local-index history for the synchronized all-camera recording view."""
        _validate_recording_range(
            start_epoch,
            end_epoch,
            90000,
            "invalid recording grid day range",
        )
        selected_source = recording_source(source)
        active_manager = deps.get_manager()
        active_config = getattr(active_manager, "config", None) or deps.get_config()
        start_at = datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
        end_at = datetime.fromtimestamp(end_epoch, timezone.utc).isoformat()
        camera_payloads: list[dict[str, object]] = []
        aggregate_ranges: list[dict[str, object]] = []
        aggregate_incidents: list[dict] = []
        available_sources: set[str] = set()
        cameras = list(active_config.cameras)
        camera_ids = {camera.id for camera in cameras}
        events_by_camera: dict[str, list[dict]] = {camera_id: [] for camera_id in camera_ids}
        if hasattr(active_manager.events, "between_compact"):
            for event in active_manager.events.between_compact(start_at, end_at):
                camera_id = str(event.get("camera_id") or "")
                if camera_id in events_by_camera:
                    events_by_camera[camera_id].append(event)
        grid_availability = None
        if hasattr(active_manager.recorder, "recording_grid_availability_between"):
            grid_availability = active_manager.recorder.recording_grid_availability_between(
                [camera.id for camera in cameras],
                start_epoch,
                end_epoch,
            )
        for camera in cameras:
            if grid_availability is not None:
                source_availability = grid_availability[camera.id]
            else:
                source_availability = {
                    candidate: active_manager.recorder.recording_availability_between(
                        camera.id,
                        start_epoch,
                        end_epoch,
                        candidate,
                        discover_missing=False,
                    )
                    for candidate in ("main", "live")
                }
            camera_sources = [
                candidate
                for candidate in ("main", "live")
                if int(source_availability[candidate]["segment_count"]) > 0
            ]
            available_sources.update(camera_sources)
            availability = source_availability[selected_source]
            ranges = [dict(item) for item in availability["ranges"]]
            for candidate in (selected_source, "live" if selected_source == "main" else "main"):
                for range_item in source_availability[candidate]["ranges"]:
                    item = dict(range_item)
                    item["camera_id"] = camera.id
                    item["source"] = candidate
                    aggregate_ranges.append(item)
            event_rows = events_by_camera[camera.id]
            if not hasattr(active_manager.events, "between_compact"):
                event_rows = active_manager.events.for_camera_range(
                    camera.id,
                    start_at,
                    end_at,
                    limit=5000,
                )
            public_events = [_event_row(event) for event in event_rows]
            hydrated_incidents = _identity_hydrated_recording_incidents(
                active_manager,
                public_events,
                include_identities=include_identities,
            )
            incidents = [
                _recording_grid_incident_payload(incident)
                for incident in hydrated_incidents
            ]
            aggregate_incidents.extend(incidents)
            camera_payloads.append({
                "camera_id": camera.id,
                "camera_name": camera.name,
                "source": selected_source,
                "recordings": ranges,
                "recording_count": int(availability["segment_count"]),
                "available_sources": camera_sources,
            })
        aggregate_ranges.sort(key=lambda item: float(item.get("start_epoch") or 0))
        aggregate_incidents.sort(
            key=lambda item: float(item.get("start_epoch") or 0)
        )
        return {
            "view": "all_cameras",
            "source": selected_source,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "recordings": aggregate_ranges,
            "availability": aggregate_ranges,
            "incidents": aggregate_incidents,
            "available_sources": sorted(available_sources),
            "cameras": camera_payloads,
        }


    @router.get("/api/recordings/grid/updates")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_grid_updates(
        start_epoch: float,
        end_epoch: float,
        after_epoch: float,
        source: str = "live",
        include_identities: bool = True,
    ) -> dict:
        """Return a bounded near-live delta for the synchronized camera grid."""
        _validate_recording_range(
            start_epoch,
            end_epoch,
            90000,
            "invalid recording grid update range",
        )
        if not math.isfinite(after_epoch):
            raise HTTPException(status_code=400, detail="invalid recording grid update cursor")
        selected_source = recording_source(source)
        active_manager = deps.get_manager()
        active_config = getattr(active_manager, "config", None) or deps.get_config()
        cameras = list(active_config.cameras)
        camera_ids = {camera.id for camera in cameras}
        overlap_seconds = max(
            5.0,
            float(getattr(active_config, "recording_segment_seconds", 10.0)) * 2,
            float(DEFAULT_INCIDENT_GAP_SECONDS) * 2,
        )
        update_start = max(start_epoch, min(end_epoch, after_epoch) - overlap_seconds)
        # The recorder's ten-second index loop already discovers every configured
        # source. Keep this multi-camera request SQLite-only instead of scheduling
        # a second wave of per-camera NFS directory scans.
        availability_by_camera = active_manager.recorder.recording_grid_availability_between(
            [camera.id for camera in cameras],
            update_start,
            end_epoch,
        )
        aggregate_ranges: list[dict[str, object]] = []
        for camera in cameras:
            source_availability = availability_by_camera[camera.id]
            for candidate in (selected_source, "live" if selected_source == "main" else "main"):
                for range_item in source_availability[candidate]["ranges"]:
                    item = dict(range_item)
                    item["camera_id"] = camera.id
                    item["source"] = candidate
                    aggregate_ranges.append(item)
        event_start = max(
            start_epoch,
            min(end_epoch, after_epoch) - max(overlap_seconds, 5 * 60.0),
        )
        event_rows = [
            row
            for row in active_manager.events.between_compact(
                datetime.fromtimestamp(event_start, timezone.utc).isoformat(),
                datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
            )
            if str(row.get("camera_id") or "") in camera_ids
        ]
        public_events = [_event_row(event) for event in event_rows]
        hydrated_incidents = _identity_hydrated_recording_incidents(
            active_manager,
            public_events,
            include_identities=include_identities,
        )
        aggregate_ranges.sort(key=lambda item: float(item.get("start_epoch") or 0))
        return {
            "view": "all_cameras",
            "source": selected_source,
            "start_epoch": update_start,
            "end_epoch": end_epoch,
            "availability": aggregate_ranges,
            "incidents": [
                _recording_grid_incident_payload(incident)
                for incident in hydrated_incidents
            ],
        }


    def _public_media_export(job: dict[str, object]) -> dict[str, object]:
        payload = dict(job)
        for key in ("download_url", "media_url"):
            if payload.get(key):
                payload[key] = deps.public_url(str(payload[key]))
        return payload


    @router.post("/api/exports", status_code=202)
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def create_media_export(request: MediaExportRequest) -> dict[str, object]:
        _require_recording_camera(deps, request.camera_id)
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
            job = deps.get_media_exports().create({
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


    @router.get("/api/exports")
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
        export_manager = deps.get_media_exports()
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


    @router.get("/api/exports/summary")
    def media_export_summary() -> dict[str, object]:
        return deps.get_media_exports().summary()


    @router.post("/api/exports/batch")
    def batch_media_exports(request: MediaExportBatchRequest) -> dict[str, object]:
        return _public_media_export_batch(
            deps.get_media_exports().batch(request.ids, request.action)
        )


    def _public_media_export_batch(payload: dict[str, object]) -> dict[str, object]:
        result = dict(payload)
        result["results"] = [
            _public_media_export(job)
            for job in list(payload.get("results") or [])
        ]
        return result


    @router.get("/api/exports/{job_id}")
    def get_media_export(job_id: str) -> dict[str, object]:
        job = deps.get_media_exports().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="export not found")
        return _public_media_export(job)


    @router.get("/api/exports/{job_id}/download")
    def download_media_export(job_id: str) -> FileResponse:
        try:
            path, name = deps.get_media_exports().output_path(job_id)
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


    @router.get("/api/exports/{job_id}/media")
    def play_media_export(job_id: str) -> FileResponse:
        try:
            path, _name = deps.get_media_exports().output_path(job_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="completed export not found") from None
        return FileResponse(
            path,
            media_type="application/zip" if path.suffix.lower() == ".zip" else "video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )


    @router.patch("/api/exports/{job_id}/protection")
    def protect_media_export(
        job_id: str,
        request: MediaExportProtectionRequest,
    ) -> dict[str, object]:
        try:
            return _public_media_export(
                deps.get_media_exports().set_protected(job_id, request.protected)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="export not found") from None


    @router.patch("/api/exports/{job_id}/metadata")
    def update_media_export_metadata(
        job_id: str,
        request: MediaExportMetadataRequest,
    ) -> dict[str, object]:
        try:
            return _public_media_export(
                deps.get_media_exports().set_label(job_id, request.label)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="export not found") from None


    @router.delete("/api/exports/{job_id}", status_code=202)
    def delete_media_export(job_id: str, force: bool = False) -> dict[str, object]:
        try:
            return _public_media_export(
                deps.get_media_exports().cancel_or_delete(job_id, force=force)
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="export not found") from None
        except PermissionError:
            raise HTTPException(
                status_code=409,
                detail="protected exports require explicit forced deletion",
            ) from None


    @router.get("/api/cameras/{camera_id}/recordings/window")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_window(
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
    ) -> dict:
        active_manager = _require_recording_camera(deps, camera_id)
        _validate_recording_range(start_epoch, end_epoch, 3600, "invalid recording window range")
        selected_source = recording_source(source)
        window_start, window_end = _recording_playback_window(start_epoch)
        rows = deps.recording_day_rows(
            active_manager,
            camera_id,
            window_start,
            window_end,
            selected_source,
        )
        return {
            "camera_id": camera_id,
            "source": selected_source,
            "start_epoch": window_start,
            "end_epoch": window_end,
            "recordings": [_public_recording_row(row) for row in rows],
        }


    @router.get("/api/cameras/{camera_id}/recordings/preview.jpg")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_preview(
        camera_id: str,
        epoch: float,
        source: str = "main",
        width: int = 480,
        exact: bool = False,
    ) -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        if not math.isfinite(epoch) or epoch <= 0:
            raise HTTPException(status_code=400, detail="invalid recording preview time")
        if width < 320 or width > 1920:
            raise HTTPException(status_code=400, detail="invalid recording preview width")
        selected_source = recording_source(source)
        rows = active_manager.recorder.recording_rows_between(
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
        preview_path = deps.recording_preview_path(
            active_manager,
            row,
            epoch,
            width=width,
            exact=exact,
        )
        actual_epoch, timestamp_source = deps.recording_preview_timestamp(
            preview_path
        )
        headers = {
            "Cache-Control": "private, max-age=3600",
            "X-SurvNG-Requested-Timestamp": f"{epoch:.6f}",
            "X-SurvNG-Timestamp-Source": timestamp_source,
        }
        if actual_epoch is not None:
            headers["X-SurvNG-Actual-Timestamp"] = f"{actual_epoch:.6f}"
        return FileResponse(
            preview_path,
            media_type="image/jpeg",
            headers=headers,
        )


    @router.get("/api/cameras/{camera_id}/recordings/updates")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_updates(
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        after_epoch: float,
        source: str = "main",
        include_identities: bool = True,
    ) -> dict:
        active_manager = _require_recording_camera(deps, camera_id)
        _validate_recording_range(start_epoch, end_epoch, 90000, "invalid recording day range")
        if not math.isfinite(after_epoch):
            raise HTTPException(status_code=400, detail="invalid recording update position")
        selected_source = recording_source(source)
        overlap_seconds = max(
            5.0,
            float(
                getattr(
                    getattr(active_manager, "config", None) or deps.get_config(),
                    "recording_segment_seconds",
                    10.0,
                )
            ) * 2,
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
        active_manager.recorder.request_recording_edge_refresh(
            camera_id,
            selected_source,
            after_epoch,
        )
        availability = active_manager.recorder.recording_availability_between(
            camera_id,
            update_start,
            end_epoch,
            selected_source,
            discover_missing=False,
        )
        events = active_manager.events.for_camera_range(
            camera_id,
            datetime.fromtimestamp(event_update_start, timezone.utc).isoformat(),
            datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
            limit=1000,
        )
        public_events = [_event_row(event) for event in events]
        timeline_incidents = _identity_hydrated_recording_incidents(
            active_manager,
            public_events,
            include_identities=include_identities,
        )
        return {
            "camera_id": camera_id,
            "source": selected_source,
            "start_epoch": update_start,
            "end_epoch": end_epoch,
            "availability": availability["ranges"],
            "events": public_events,
            "incidents": [
                _incident_list_payload(incident)
                for incident in timeline_incidents
            ],
        }

    @router.get("/api/cameras/{camera_id}/recordings/day.m3u8")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_day_hls_playlist(
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
    ) -> Response:
        active_manager = _require_recording_camera(deps, camera_id)
        _validate_recording_range(
            start_epoch,
            end_epoch,
            90000,
            "invalid recording day range",
        )
        selected_source = recording_source(source)
        rows = deps.recording_day_rows(
            active_manager,
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
            fresh=True,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="no recordings found")
        target_duration = max(
            1,
            math.ceil(max(float(row["duration_seconds"]) for row in rows)),
        )
        query = (
            f"start_epoch={start_epoch:.3f}&end_epoch={end_epoch:.3f}"
            f"&source={selected_source}"
        )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:VOD",
        ]
        for index, row in enumerate(rows):
            row_start = float(row["start_epoch"])
            segment_name = quote(str(row["name"]), safe="")
            segment_query = query
            # Each recording is remuxed independently.  Its media timestamps
            # therefore restart at zero, even when codec metadata matches the
            # preceding recording.  Make that boundary explicit for native HLS
            # clients instead of synthesizing a continuous decode timeline.
            if index:
                lines.append("#EXT-X-DISCONTINUITY")
            lines.append(
                f'#EXT-X-MAP:URI="day/segment/{segment_name}/init.mp4?{segment_query}"'
            )
            lines.extend(
                [
                    "#EXT-X-PROGRAM-DATE-TIME:"
                    f"{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
                    f"#EXTINF:{float(row['duration_seconds']):.3f},",
                    f"day/segment/{segment_name}/media.m4s?{segment_query}",
                ]
            )
        lines.append("#EXT-X-ENDLIST")
        return Response(
            "\n".join(lines) + "\n",
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-store"},
        )

    @router.get(
        "/api/cameras/{camera_id}/recordings/day/segment/{segment_name}/init.mp4"
    )
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_day_hls_init(
        camera_id: str,
        segment_name: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        trim_end: bool = False,
    ) -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        init_path, _ = deps.recording_day_fmp4_paths(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            trim_end,
        )
        return deps.recording_file_response(init_path, "video/mp4")

    @router.get(
        "/api/cameras/{camera_id}/recordings/day/segment/{segment_name}/media.m4s"
    )
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_day_hls_segment(
        camera_id: str,
        segment_name: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        trim_end: bool = False,
    ) -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        _, media_path = deps.recording_day_fmp4_paths(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            trim_end,
        )
        return deps.recording_file_response(media_path, "video/iso.segment")

    @router.get("/api/events/{event_id}/clip.mp4")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def event_clip(
        event_id: int,
        before: float | None = None,
        after: float | None = None,
        source: str = "main",
    ) -> FileResponse:
        active_manager = deps.get_manager()
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        enriched = _event_row(event)
        before_seconds, after_seconds = deps.event_clip_window(
            active_manager,
            before,
            after,
        )
        clip_path = deps.ensure_event_clip(
            active_manager,
            enriched,
            before=before_seconds,
            after=after_seconds,
            source=recording_source(source),
        )
        return FileResponse(
            clip_path,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @router.get("/api/events/{event_id}/stream.m3u8")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def event_stream(
        event_id: int,
        before: float | None = None,
        after: float | None = None,
        source: str = "main",
    ) -> Response:
        active_manager = deps.get_manager()
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        enriched = _event_row(event)
        camera_id = str(enriched.get("camera_id") or "")
        if not camera_id:
            raise HTTPException(status_code=400, detail="event is missing camera")
        before_seconds, after_seconds = deps.event_clip_window(
            active_manager,
            before,
            after,
        )
        event_created_epoch = event_epoch(enriched)
        window_start = event_created_epoch - before_seconds
        window_end = event_created_epoch + after_seconds
        selected_source = recording_source(source)
        rows = deps.recording_day_rows(
            active_manager,
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
        query = (
            f"start_epoch={window_start:.3f}&end_epoch={window_end:.3f}"
            f"&source={selected_source}"
        )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:VOD",
            f"#EXT-X-START:TIME-OFFSET={start_offset:.3f},PRECISE=YES",
        ]
        for index, (row, clip_duration) in enumerate(zip(rows, clip_durations)):
            row_start = float(row["start_epoch"])
            segment_name = quote(str(row["name"]), safe="")
            segment_query = f"{query}&trim_end=true"
            encoded_camera = quote(camera_id, safe="")
            map_uri = deps.public_url(
                f"/api/cameras/{encoded_camera}/recordings/day/segment/"
                f"{segment_name}/init.mp4?{segment_query}"
            )
            if index:
                lines.append("#EXT-X-DISCONTINUITY")
            lines.append(f'#EXT-X-MAP:URI="{map_uri}"')
            lines.extend(
                [
                    "#EXT-X-PROGRAM-DATE-TIME:"
                    f"{datetime.fromtimestamp(row_start, timezone.utc).isoformat()}",
                    f"#EXTINF:{clip_duration:.3f},",
                    deps.public_url(
                        f"/api/cameras/{encoded_camera}/recordings/day/segment/"
                        f"{segment_name}/media.m4s?{segment_query}"
                    ),
                ]
            )
        lines.append("#EXT-X-ENDLIST")
        return Response(
            "\n".join(lines) + "\n",
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "private, max-age=30"},
        )

    return RecordingRouteBundle(
        router=router,
        handlers={
            handler.__name__: handler
            for handler in (
                recordings,
                recording_events,
                recording_day,
                recording_grid_day,
                recording_grid_updates,
                _public_media_export,
                create_media_export,
                list_media_exports,
                media_export_summary,
                batch_media_exports,
                _public_media_export_batch,
                get_media_export,
                download_media_export,
                play_media_export,
                protect_media_export,
                update_media_export_metadata,
                delete_media_export,
                recording_window,
                recording_preview,
                recording_updates,
                recording_day_hls_playlist,
                recording_day_hls_init,
                recording_day_hls_segment,
                event_clip,
                event_stream,
            )
        },
    )
