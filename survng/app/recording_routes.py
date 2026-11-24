"""Recording history, grid, preview, and export HTTP boundary."""

from __future__ import annotations

import hashlib
import math
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .incident_presenter import (
    _event_row,
    _incident_list_payload,
    _incident_row,
    _recording_event_row,
    _recording_grid_incident_payload,
)
from .incident_utils import DEFAULT_INCIDENT_GAP_SECONDS, event_epoch
from .encoded_fragments import (
    EncodedFragmentSource,
    FragmentSourceError,
    FragmentUnavailable,
    FragmentWindow,
    InvalidFragmentWindow,
)
from .growing_playlists import (
    GROWING_HOLD_SECONDS,
    GROWING_RECORD_LINGER_SECONDS,
    GrowingPlaylistKey,
    GrowingPlaylistRecord,
    GrowingPlaylistRegistry,
    MergeReport,
    RateLimitedLog,
    day_playlist_entry_lines,
    merge_rows,
)
from .media_exports import MediaExportManager
from .manager_access import ManagerAccessCoordinator, guard_manager_generation
from .recording_media import (
    RECORDING_FMP4_VERSION,
    hls_map_transition,
    playback_segment_duration,
    resolve_stream_fingerprints,
)

RECORDING_LOOKUP_LIMIT = 20_000
RECORDING_PLAYBACK_WINDOW_SECONDS = 15 * 60
MOBILE_PLAYBACK_WINDOW_SECONDS = 120
LOGGER = logging.getLogger(__name__)


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
    recording_segment_path: Callable[..., Path]
    recording_day_fmp4_paths: Callable[..., tuple[Path, Path]]
    recording_file_response: Callable[[Path, str], FileResponse]
    event_clip_window: Callable[..., tuple[float, float]]
    ensure_event_clip: Callable[..., Path]
    manager_lock: threading.RLock | None = None
    recording_preview_strip: Callable[..., tuple[Path, dict]] | None = None
    manager_access: ManagerAccessCoordinator | None = None
    encoded_fragment_source: (
        Callable[[Any], EncodedFragmentSource] | None
    ) = None


@dataclass(frozen=True, slots=True)
class RecordingRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]
    growing_playlists: GrowingPlaylistRegistry | None = None


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
    # Resolve whole canonical incidents, then project their camera segments for
    # the recording lanes. The lane keeps the canonical ID for navigation.
    canonical = {}
    seen_events = set()
    requested_cameras = {str(event.get("camera_id") or "") for event in public_events}
    for event in public_events:
        event_id = int(event["id"])
        if event_id in seen_events:
            continue
        incident = active_manager.events.scene_incident(event_id=event_id)
        if incident is None or incident.get("state") == "unconfirmed":
            continue
        canonical[incident["id"]] = incident
        seen_events.update(int(item["id"]) for item in incident.get("events", []))
    incidents = []
    for incident in canonical.values():
        for camera_id in requested_cameras:
            events = [event for event in incident.get("events", []) if event.get("camera_id") == camera_id]
            if not events:
                continue
            local = {**incident, **_incident_row(camera_id, events)}
            local.update(id=incident["id"], incident_id=incident["incident_id"],
                         canonical_start_at=incident["start_at"], canonical_end_at=incident["end_at"])
            episodes = [episode for episode in incident.get("episodes", []) if episode["camera_id"] == camera_id]
            if episodes:
                local["start_at"] = min(episode["start_at"] for episode in episodes)
                local["end_at"] = max(episode["end_at"] for episode in episodes)
                local["start_epoch"] = event_epoch({"created_at": local["start_at"]})
                local["last_epoch"] = event_epoch({"created_at": local["end_at"]})
                local["duration_seconds"] = local["last_epoch"] - local["start_epoch"]
            incidents.append(local)
    incidents.sort(key=lambda item: item["start_epoch"], reverse=True)
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


def _recording_segment_seconds(
    deps: RecordingRouteDependencies,
    active_manager: Any,
) -> float:
    return float(
        getattr(
            getattr(active_manager, "config", None) or deps.get_config(),
            "recording_segment_seconds",
            10.0,
        )
    )


# How far either side of a time with no recording a preview-strip request
# looks for where the recording stops and starts again.
RECORDING_STRIP_GAP_WINDOW_SECONDS = 300.0


def _recording_playback_window(epoch: float) -> tuple[float, float]:
    start = (
        math.floor(epoch / RECORDING_PLAYBACK_WINDOW_SECONDS)
        * RECORDING_PLAYBACK_WINDOW_SECONDS
    )
    return start, start + RECORDING_PLAYBACK_WINDOW_SECONDS


def create_recording_router(deps: RecordingRouteDependencies) -> RecordingRouteBundle:
    router = APIRouter()
    # Entries served by growing day playlists. It belongs to the router, not
    # to a manager, so a manager reload does not reset what players hold.
    growing_playlists = GrowingPlaylistRegistry()
    growing_log = RateLimitedLog(LOGGER)

    def fragment_rows(
        active_manager: Any,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str,
        *,
        trim_end: bool,
        tolerate_unavailable: bool = False,
    ) -> list[dict] | None:
        source_factory = getattr(deps, "encoded_fragment_source", None)
        if source_factory is None:
            return None
        # Only growing playlists ask for a tolerant listing; every other
        # caller keeps the strict one, where any unreadable row fails it.
        options = {"tolerate_unavailable": True} if tolerate_unavailable else {}
        try:
            fragments = list(
                source_factory(active_manager).fragments(
                    FragmentWindow(
                        camera_id=camera_id,
                        source=recording_source(source),
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                    ),
                    trim_end=trim_end,
                    **options,
                )
            )
        except InvalidFragmentWindow as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FragmentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FragmentSourceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return [
            {
                "name": fragment.segment_name,
                "start_epoch": fragment.wall_start_epoch,
                "end_epoch": fragment.wall_end_epoch,
                "duration_seconds": fragment.media_duration_seconds,
                "stream_fingerprint": fragment.stream.fingerprint or "",
                "_encoded_fragment": fragment,
            }
            for fragment in fragments
        ]

    def materialize_fragment(
        active_manager: Any,
        camera_id: str,
        segment_name: str,
        start_epoch: float,
        end_epoch: float,
        source: str,
        media_offset: float,
        trim_end: bool,
        fragment_id: str,
        growing: bool = False,
    ) -> tuple[Any | None, Callable[[], None] | None]:
        if not math.isfinite(media_offset) or media_offset < 0:
            raise HTTPException(
                status_code=400,
                detail="invalid recording media offset",
            )
        source_factory = getattr(deps, "encoded_fragment_source", None)
        if source_factory is None:
            return None, None
        # A growing playlist's segment needs only its own row (and the next,
        # for lookahead): another row of the window that cannot be checked
        # must not fail it.
        rows = fragment_rows(
            active_manager,
            camera_id,
            start_epoch,
            end_epoch,
            source,
            trim_end=trim_end,
            tolerate_unavailable=growing,
        ) or []
        if growing:
            rows = [
                row for row in rows
                if getattr(row.get("_encoded_fragment"), "available", True)
            ]
        selected = next(
            (
                row.get("_encoded_fragment")
                for row in rows
                if row.get("name") == segment_name
            ),
            None,
        )
        if selected is None:
            raise HTTPException(
                status_code=404,
                detail="recording segment not found",
            )
        if fragment_id and fragment_id != selected.source_identity:
            served = (
                growing_playlists.served_entry(
                    (
                        camera_id,
                        recording_source(source),
                        f"{start_epoch:.3f}",
                        f"{end_epoch:.3f}",
                    ),
                    segment_name,
                    fragment_id,
                )
                if growing and not trim_end
                else None
            )
            if served is None:
                raise HTTPException(
                    status_code=409,
                    detail="recording fragment revision changed",
                    headers={"Cache-Control": "no-store"},
                )
            # A growing playlist keeps an entry it has served even when the
            # index later revises the row (for example, a nominal duration
            # replaced by the measured one). Serve the file as that entry
            # described it, so the player's timeline stays as listed.
            growing_log.log(
                logging.WARNING,
                "revised-segment",
                camera_id,
                "Growing playlist segment %s/%s was revised after it was "
                "served; serving it with its served duration %.3f s",
                camera_id,
                segment_name,
                served.duration_seconds,
            )
            selected = replace(
                selected,
                media_duration_seconds=served.duration_seconds,
            )
        # The manifest owns the compact media timeline. Index changes between
        # manifest and fragment requests must not silently move tfdt/sidx.
        selected = replace(
            selected,
            media_start_seconds=media_offset,
        )
        selected_index = next(
            (
                index
                for index, row in enumerate(rows)
                if row.get("_encoded_fragment") is not None
                and row["_encoded_fragment"].segment_name == segment_name
            ),
            None,
        )
        next_fragment = (
            rows[selected_index + 1].get("_encoded_fragment")
            if selected_index is not None and selected_index + 1 < len(rows)
            else None
        )
        if next_fragment is not None:
            next_fragment = replace(
                next_fragment,
                media_start_seconds=(
                    media_offset + selected.media_duration_seconds
                ),
            )
        source = source_factory(active_manager)
        try:
            materialized = source.materialize(selected)
        except FragmentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FragmentSourceError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        def prewarm_next() -> None:
            if next_fragment is None:
                return
            try:
                source.materialize(next_fragment)
            except Exception:
                # The requested fragment has already been served. A lookahead
                # miss must not turn successful playback into an HTTP failure.
                LOGGER.warning(
                    "Recording fragment lookahead failed for %s/%s",
                    camera_id,
                    next_fragment.segment_name,
                    exc_info=True,
                )

        return materialized, prewarm_next if next_fragment is not None else None

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


    @router.get("/api/cameras/{camera_id}/recordings/segment.mp4")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_segment(
        camera_id: str,
        epoch: float,
        source: str = "main",
        mobile: bool = False,
    ) -> FileResponse:
        """Serve one indexed recording segment for native browser playback."""
        active_manager = _require_recording_camera(deps, camera_id)
        if not math.isfinite(epoch) or epoch <= 0:
            raise HTTPException(status_code=400, detail="invalid recording segment time")
        leased = deps.recording_segment_path(
            active_manager,
            camera_id,
            epoch,
            recording_source(source),
        )
        path = getattr(leased, "path", leased)
        lease_token = getattr(leased, "lease_token", None)
        release = getattr(active_manager.recorder, "release_recording_playback", None)
        try:
            if mobile:
                # Transcode exactly this segment, retaining mobile codec compatibility
                # without concatenating across wall-clock gaps.
                rows = active_manager.recorder.recording_rows_between(
                    camera_id, epoch, epoch + 0.001, recording_source(source), discover_missing=False,
                )
                row = next((row for row in rows if Path(str(row["path"])).resolve() == path.resolve()), None)
                if row is None:
                    raise HTTPException(status_code=404, detail="recording segment metadata is unavailable")
                start = float(row["start_epoch"])
                path = deps.ensure_event_clip(
                    active_manager,
                    {"id": int(start * 1000), "camera_id": camera_id,
                     "created_at": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                     "_recording_rows": [row]},
                    before=0.0, after=float(row["end_epoch"]) - start,
                    source=recording_source(source),
                )
            background = BackgroundTask(release, lease_token) if callable(release) and lease_token else None
            # FileResponse provides byte-range requests needed by native MP4 seeking.
            # Do not use the fMP4 cache response helper: this is the source recording.
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "private, max-age=3600"},
                background=background,
            )
        except Exception:
            if callable(release) and lease_token:
                release(lease_token)
            raise

    @router.get("/api/cameras/{camera_id}/recordings/mobile-window.mp4")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_mobile_window(camera_id: str, epoch: float, source: str = "main") -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        if not math.isfinite(epoch) or epoch <= 0:
            raise HTTPException(status_code=400, detail="invalid recording window time")
        window_start = math.floor(epoch / MOBILE_PLAYBACK_WINDOW_SECONDS) * MOBILE_PLAYBACK_WINDOW_SECONDS
        clip_path = deps.ensure_event_clip(
            active_manager,
            {"id": int(window_start), "camera_id": camera_id,
             "created_at": datetime.fromtimestamp(window_start, timezone.utc).isoformat()},
            before=0.0, after=float(MOBILE_PLAYBACK_WINDOW_SECONDS),
            source=recording_source(source),
        )
        return FileResponse(clip_path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


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

    @router.get("/api/cameras/{camera_id}/recordings/preview-strip.jpg")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_preview_strip(
        camera_id: str,
        epoch: float,
        source: str = "main",
        width: int = 320,
    ) -> FileResponse:
        """One segment's frames as a grid, for scrubbing without a request per move.

        The segment is the one holding ``epoch``. A camera's sub stream is
        used when it was recorded then: it shows the same scene and decodes
        several times faster. The headers say which seconds the tiles cover.
        """
        active_manager = _require_recording_camera(deps, camera_id)
        if deps.recording_preview_strip is None:
            raise HTTPException(status_code=404, detail="recording preview strips are unavailable")
        if not math.isfinite(epoch) or epoch <= 0:
            raise HTTPException(status_code=400, detail="invalid recording preview time")
        if width < 160 or width > 640:
            raise HTTPException(status_code=400, detail="invalid recording preview width")
        selected_source = recording_source(source)
        built = None
        unavailable: HTTPException | None = None
        for candidate_source in dict.fromkeys(("live", selected_source)):
            rows = active_manager.recorder.recording_rows_between(
                camera_id,
                epoch - 0.001,
                epoch + 0.001,
                candidate_source,
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
                continue
            try:
                strip_path, layout = deps.recording_preview_strip(
                    active_manager,
                    row,
                    width=width,
                )
            except HTTPException as exc:
                # An indexed file that has gone (or cannot be served) is not
                # the end of it: the other stream may still hold this time.
                if exc.status_code not in {403, 404}:
                    raise
                unavailable = exc
                continue
            built = (row, candidate_source, strip_path, layout)
            break
        if built is None:
            if unavailable is not None:
                raise unavailable
            # Say how far the gap reaches, so the page need not ask about
            # every second of it. Bounded: beyond the window it asks again.
            gap_start = epoch - RECORDING_STRIP_GAP_WINDOW_SECONDS
            # Not into the future: what is being recorded now is "not there"
            # only until it is indexed.
            gap_end = min(
                epoch + RECORDING_STRIP_GAP_WINDOW_SECONDS,
                max(epoch + 1.0, time.time()),
            )
            known = True
            for candidate_source in dict.fromkeys(("live", selected_source)):
                for nearby in active_manager.recorder.recording_rows_between(
                    camera_id,
                    gap_start,
                    gap_end,
                    candidate_source,
                    discover_missing=False,
                ):
                    start = float(nearby.get("start_epoch") or 0)
                    end = float(nearby.get("end_epoch") or 0)
                    if end <= epoch:
                        gap_start = max(gap_start, end)
                    elif start > epoch:
                        gap_end = min(gap_end, start)
                    else:
                        # Indexed since the lookup above: no gap after all,
                        # and nothing is claimed about how far one reaches.
                        known = False
            raise HTTPException(
                status_code=404,
                detail="no recording exists at this time",
                headers={
                    "X-SurvNG-Gap-Start": f"{gap_start:.3f}",
                    "X-SurvNG-Gap-End": f"{gap_end:.3f}",
                } if known else None,
            )
        row, row_source, strip_path, layout = built
        return FileResponse(
            strip_path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-SurvNG-Strip-Start": f"{float(row['start_epoch']):.3f}",
                "X-SurvNG-Strip-End": f"{float(row['end_epoch']):.3f}",
                "X-SurvNG-Strip-Step": f"{float(layout['step_seconds']):g}",
                "X-SurvNG-Strip-Count": str(int(layout["count"])),
                "X-SurvNG-Strip-Columns": str(int(layout["columns"])),
                "X-SurvNG-Strip-Rows": str(int(layout["rows"])),
                "X-SurvNG-Strip-Source": row_source,
            },
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
            _recording_segment_seconds(deps, active_manager) * 2,
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

    def day_playlist_rows(
        active_manager: Any,
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        selected_source: str,
        *,
        tolerate_unavailable: bool = False,
    ) -> list[dict]:
        rows = fragment_rows(
            active_manager,
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
            trim_end=False,
            tolerate_unavailable=tolerate_unavailable,
        )
        if rows is None:
            rows = deps.recording_day_rows(
                active_manager,
                camera_id,
                start_epoch,
                end_epoch,
                selected_source,
                fresh=True,
            )
        return rows

    def day_playlist_header(
        rows: list[dict],
        *,
        still_growing: bool,
        segment_seconds: float,
        requested_start: float,
    ) -> list[str]:
        target_duration = max(
            1,
            math.ceil(max(float(row["duration_seconds"]) for row in rows)),
        )
        if still_growing:
            # A playlist's target duration may not change between re-reads.
            # Segments are cut at the first keyframe after the nominal length,
            # so one a little longer than those listed so far can still come.
            target_duration = max(
                target_duration,
                math.ceil(segment_seconds) + 1,
            )
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            # The recorder's segment muxer stream-copies video and can only cut
            # at keyframes, so every recording file (and so every segment)
            # decodes on its own. Apple's player uses this to start or seek in
            # any segment without fetching the one before it.
            "#EXT-X-INDEPENDENT-SEGMENTS",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            f"#EXT-X-PLAYLIST-TYPE:{'EVENT' if still_growing else 'VOD'}",
        ]
        media_duration = sum(float(row["duration_seconds"]) for row in rows)
        # Native Safari applies EXT-X-START while building seekable ranges.
        # Post-metadata currentTime seeks are raced and intermittent on iPhone.
        # Without a start tag a player opens an unfinished playlist near its
        # end, so a growing playlist always names its start, even at zero.
        if media_duration > 0 and (requested_start > 0 or still_growing):
            start_offset = min(
                requested_start,
                max(0.0, media_duration - 0.01),
            )
            lines.append(
                f"#EXT-X-START:TIME-OFFSET={start_offset:.3f},PRECISE=YES"
            )
        return lines

    def log_growing_merge(
        key: GrowingPlaylistKey,
        report: MergeReport,
    ) -> None:
        scope = key.camera_id
        window = f"{key.camera_id}/{key.source} {key.start_epoch}-{key.end_epoch}"
        for kind, names, level, message in (
            (
                "kept-unreadable", report.kept_unreadable, logging.WARNING,
                "Growing playlist %s kept %d served segment(s) whose file "
                "could not be checked (first: %s)",
            ),
            (
                "kept-missing", report.kept_missing, logging.WARNING,
                "Growing playlist %s kept %d served segment(s) no longer in "
                "the recording index (first: %s)",
            ),
            (
                "kept-changed", report.kept_changed, logging.WARNING,
                "Growing playlist %s kept %d served segment(s) the index has "
                "since revised (first: %s)",
            ),
            (
                "held", report.held, logging.INFO,
                "Growing playlist %s is holding back %d new segment(s) whose "
                "file could not be checked (first: %s)",
            ),
            (
                "skipped", report.skipped, logging.WARNING,
                "Growing playlist %s skipped %d new segment(s) unreadable for "
                f"over {GROWING_HOLD_SECONDS:.0f} s (first: %s)",
            ),
            (
                "out-of-order", report.out_of_order, logging.WARNING,
                "Growing playlist %s cannot insert %d segment(s) indexed "
                "after a later one was served (first: %s)",
            ),
            (
                "over-target", report.over_target, logging.WARNING,
                "Growing playlist %s appended %d segment(s) longer than its "
                "target duration (first: %s)",
            ),
        ):
            if names:
                growing_log.log(
                    level, kind, scope, message, window, len(names), names[0],
                )

    def growing_day_playlist(
        active_manager: Any,
        key: GrowingPlaylistKey,
        *,
        start_epoch: float,
        end_epoch: float,
        still_growing: bool,
        requested_start: float,
        query: str,
    ) -> str | None:
        """Serve a growing playlist from its record, or None if it has none.

        The first read of a playlist (no record) is as strict as a finished
        one: it fails if any row cannot be checked. Later re-reads keep every
        served entry unchanged and only append rows that can be checked.
        """
        with growing_playlists.lock:
            record = growing_playlists.get(key)
        if record is None and not still_growing:
            return None
        segment_seconds = _recording_segment_seconds(deps, active_manager)
        try:
            rows = day_playlist_rows(
                active_manager,
                key.camera_id,
                start_epoch,
                end_epoch,
                key.source,
                tolerate_unavailable=record is not None,
            )
        except HTTPException as exc:
            if record is None or exc.status_code < 404:
                raise
            # The listing itself failed (not one row). Re-serve what was
            # already served rather than fail the re-read.
            growing_log.log(
                logging.WARNING,
                "listing-failed",
                key.camera_id,
                "Growing playlist %s/%s could not list recordings (HTTP %d); "
                "serving its %d served segment(s) unchanged",
                key.camera_id,
                key.source,
                exc.status_code,
                len(record.entries),
            )
            rows = None
        if record is None:
            if not rows:
                raise HTTPException(status_code=404, detail="no recordings found")
            record = GrowingPlaylistRecord(
                key=key,
                header=tuple(
                    day_playlist_header(
                        rows,
                        still_growing=True,
                        segment_seconds=segment_seconds,
                        requested_start=requested_start,
                    )
                ),
                query=query,
                target_duration=0,
                # Kept a little past the time the window can grow, so the
                # re-read that ends the playlist still finds what it served.
                expires_at=(
                    end_epoch
                    + max(30.0, 3.0 * segment_seconds)
                    + GROWING_RECORD_LINGER_SECONDS
                ),
            )
            record.target_duration = int(
                next(
                    line for line in record.header
                    if line.startswith("#EXT-X-TARGETDURATION:")
                ).split(":", 1)[1]
            )
        report: MergeReport | None = None
        evicted: list[GrowingPlaylistKey] = []
        with growing_playlists.lock:
            stored = growing_playlists.get(key)
            if stored is None and record.entries:
                # Evicted or expired since it was read: start again from the
                # rows just read, strictly, as a first read would.
                record = GrowingPlaylistRecord(
                    key=key,
                    header=record.header,
                    query=record.query,
                    target_duration=record.target_duration,
                    expires_at=record.expires_at,
                )
                if rows is None or not all(
                    getattr(row.get("_encoded_fragment"), "available", True)
                    for row in rows
                ):
                    raise HTTPException(
                        status_code=404,
                        detail="recording fragment is unavailable",
                    )
            # A concurrent first read of the same playlist may have stored
            # its record first; extend that one so both stay consistent.
            target = stored or record
            if rows is not None:
                fingerprints = resolve_stream_fingerprints(
                    [row.get("stream_fingerprint") for row in rows]
                )
                report = merge_rows(
                    target,
                    rows,
                    fingerprints,
                    now_monotonic=time.monotonic(),
                )
            if not target.entries:
                raise HTTPException(status_code=404, detail="no recordings found")
            if still_growing:
                if stored is None:
                    evicted = growing_playlists.put(target)
            else:
                # The window can no longer grow: this read ends the playlist
                # and its record is no longer needed.
                growing_playlists.drop(key)
            body = target.body(finished=not still_growing)
        if report is not None:
            log_growing_merge(key, report)
        if evicted:
            growing_log.log(
                logging.WARNING,
                "evicted",
                "registry",
                "Growing playlist records are full; evicted %d (first: %s/%s)",
                len(evicted),
                evicted[0].camera_id,
                evicted[0].source,
            )
        return body

    @router.get("/api/cameras/{camera_id}/recordings/day.m3u8")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def recording_day_hls_playlist(
        camera_id: str,
        start_epoch: float,
        end_epoch: float,
        source: str = "main",
        start: float | None = None,
        growing: bool = False,
        reload: str = "",
    ) -> Response:
        active_manager = _require_recording_camera(deps, camera_id)
        _validate_recording_range(
            start_epoch,
            end_epoch,
            90000,
            "invalid recording day range",
        )
        # A window that can still receive recordings is served, on request, as
        # an append-only playlist: the player re-reads it and plays new
        # recordings as they are indexed. A finished playlist can only grow by
        # being loaded again, which restarts the player at the live edge.
        still_growing = growing and time.time() < end_epoch + max(
            30.0,
            3.0 * _recording_segment_seconds(deps, active_manager),
        )
        selected_source = recording_source(source)
        requested_start = (
            float(start)
            if start is not None and math.isfinite(start) and start > 0
            else 0.0
        )
        query = (
            f"start_epoch={start_epoch:.3f}&end_epoch={end_epoch:.3f}"
            f"&source={selected_source}"
        )
        if growing:
            reload_token = str(reload or "")
            if len(reload_token) > 128:
                reload_token = hashlib.sha256(
                    reload_token.encode("utf-8")
                ).hexdigest()
            # One growing playlist is one URL: the player re-reads exactly
            # the URL it loaded, and a page that loads the window again sends
            # a new reload token (and so gets rows matching its own).
            body = growing_day_playlist(
                active_manager,
                GrowingPlaylistKey(
                    camera_id=camera_id,
                    source=selected_source,
                    start_epoch=f"{start_epoch:.3f}",
                    end_epoch=f"{end_epoch:.3f}",
                    start=f"{requested_start:.3f}",
                    reload=reload_token,
                ),
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                still_growing=still_growing,
                requested_start=requested_start,
                query=query,
            )
            if body is not None:
                return Response(
                    body,
                    media_type="application/vnd.apple.mpegurl",
                    headers={"Cache-Control": "no-store"},
                )
        rows = day_playlist_rows(
            active_manager,
            camera_id,
            start_epoch,
            end_epoch,
            selected_source,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="no recordings found")
        lines = day_playlist_header(
            rows,
            still_growing=False,
            segment_seconds=0.0,  # only a growing header uses it
            requested_start=requested_start,
        )
        media_offset = 0.0
        previous_fingerprint: str | None = None
        fingerprints = resolve_stream_fingerprints(
            [row.get("stream_fingerprint") for row in rows]
        )
        for row, stream_fingerprint in zip(rows, fingerprints):
            entry_lines, previous_fingerprint = day_playlist_entry_lines(
                row,
                stream_fingerprint,
                previous_fingerprint,
                query,
                media_offset,
            )
            lines.extend(entry_lines)
            media_offset += float(row["duration_seconds"])
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
        media_offset: float = 0.0,
        trim_end: bool = False,
        fragment_id: str = "",
        growing: bool = False,
    ) -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        materialized, _prewarm_next = materialize_fragment(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            media_offset,
            trim_end,
            fragment_id,
            growing,
        )
        if materialized is not None:
            if materialized.stream.init_path is None:
                raise HTTPException(
                    status_code=500,
                    detail="recording initialization fragment is unavailable",
                )
            return deps.recording_file_response(
                materialized.stream.init_path,
                "video/mp4",
            )
        init_path, _ = deps.recording_day_fmp4_paths(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            media_offset,
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
        media_offset: float = 0.0,
        trim_end: bool = False,
        fragment_id: str = "",
        growing: bool = False,
    ) -> FileResponse:
        active_manager = _require_recording_camera(deps, camera_id)
        materialized, prewarm_next = materialize_fragment(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            media_offset,
            trim_end,
            fragment_id,
            growing,
        )
        if materialized is not None:
            if materialized.media_path is None:
                raise HTTPException(
                    status_code=500,
                    detail="recording media fragment is unavailable",
                )
            response = deps.recording_file_response(
                materialized.media_path,
                "video/iso.segment",
            )
            if prewarm_next is not None and hasattr(response, "background"):
                response.background = BackgroundTask(prewarm_next)
            return response
        _, media_path = deps.recording_day_fmp4_paths(
            active_manager,
            camera_id,
            segment_name,
            start_epoch,
            end_epoch,
            source,
            media_offset,
            trim_end,
        )
        return deps.recording_file_response(media_path, "video/iso.segment")

    def episode_clip_window(active_manager, enriched, episode_id, start_epoch, end_epoch):
        if episode_id is None and start_epoch is None and end_epoch is None:
            return None
        if not episode_id or start_epoch is None or end_epoch is None:
            raise HTTPException(status_code=400, detail="episode and both clip bounds are required")
        episode = active_manager.events.scene_episode_for_event(int(enriched["id"]), episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="event does not belong to this episode")
        if (not math.isfinite(start_epoch) or not math.isfinite(end_epoch)
                or not 0 < end_epoch - start_epoch <= 3600
                or start_epoch < episode["start_epoch"] - .001
                or end_epoch > max(episode["end_epoch"], episode["start_epoch"] + 1) + .001):
            raise HTTPException(status_code=400, detail="invalid episode clip bounds")
        enriched["created_at"] = datetime.fromtimestamp(start_epoch, timezone.utc).isoformat()
        enriched["scene_clip_start_epoch"] = start_epoch
        return 0.0, end_epoch - start_epoch

    @router.get("/api/events/{event_id}/clip.mp4")
    @guard_manager_generation(deps.manager_access, deps.manager_lock, deps.get_manager)
    def event_clip(
        event_id: int,
        before: float | None = None,
        after: float | None = None,
        source: str = "main",
        episode_id: str | None = None,
        start_epoch: float | None = None,
        end_epoch: float | None = None,
    ) -> FileResponse:
        active_manager = deps.get_manager()
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        enriched = _event_row(event)
        before_seconds, after_seconds = episode_clip_window(
            active_manager, enriched, episode_id, start_epoch, end_epoch
        ) or deps.event_clip_window(
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
        episode_id: str | None = None,
        start_epoch: float | None = None,
        end_epoch: float | None = None,
    ) -> Response:
        active_manager = deps.get_manager()
        event = active_manager.events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        enriched = _event_row(event)
        camera_id = str(enriched.get("camera_id") or "")
        if not camera_id:
            raise HTTPException(status_code=400, detail="event is missing camera")
        before_seconds, after_seconds = episode_clip_window(
            active_manager, enriched, episode_id, start_epoch, end_epoch
        ) or deps.event_clip_window(
            active_manager,
            before,
            after,
        )
        event_created_epoch = event_epoch(enriched)
        window_start = event_created_epoch - before_seconds
        window_end = event_created_epoch + after_seconds
        selected_source = recording_source(source)
        rows = fragment_rows(
            active_manager,
            camera_id,
            window_start,
            window_end,
            selected_source,
            trim_end=True,
        )
        if rows is None:
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
        media_offset = 0.0
        previous_fingerprint: str | None = None
        fingerprints = resolve_stream_fingerprints(
            [row.get("stream_fingerprint") for row in rows]
        )
        for row, stream_fingerprint, clip_duration in zip(
            rows,
            fingerprints,
            clip_durations,
        ):
            row_start = float(row["start_epoch"])
            segment_name = quote(str(row["name"]), safe="")
            encoded_fragment = row.get("_encoded_fragment")
            fragment_id = str(
                getattr(encoded_fragment, "source_identity", "") or ""
            )
            segment_query = (
                f"{query}&media_offset={media_offset:.3f}&trim_end=true"
                f"&fragment_id={quote(fragment_id, safe='')}"
                f"&v={RECORDING_FMP4_VERSION}"
            )
            encoded_camera = quote(camera_id, safe="")
            map_uri = deps.public_url(
                f"/api/cameras/{encoded_camera}/recordings/day/segment/"
                f"{segment_name}/init.mp4?{segment_query}"
            )
            map_lines, previous_fingerprint = hls_map_transition(
                previous_fingerprint,
                stream_fingerprint,
                map_uri,
            )
            lines.extend(map_lines)
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
            media_offset += clip_duration
        lines.append("#EXT-X-ENDLIST")
        return Response(
            "\n".join(lines) + "\n",
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "private, max-age=30"},
        )

    return RecordingRouteBundle(
        router=router,
        growing_playlists=growing_playlists,
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
                recording_segment,
                recording_mobile_window,
                recording_window,
                recording_preview,
                recording_preview_strip,
                recording_updates,
                recording_day_hls_playlist,
                recording_day_hls_init,
                recording_day_hls_segment,
                event_clip,
                event_stream,
            )
        },
    )
