"""Event imagery and cross-camera appearance HTTP boundary."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .camera_routes import match_camera_route
from .incident_utils import event_snapshot_path, snapshot_media_type
from .manager import AppManager


@dataclass(frozen=True, slots=True)
class AppearanceRouteDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock


@dataclass(frozen=True, slots=True)
class AppearanceRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def _jpeg_thumbnail(frame: np.ndarray, width: int, quality: int) -> bytes:
    frame_height, frame_width = frame.shape[:2]
    if frame_width > width:
        target_height = max(1, round(frame_height * width / frame_width))
        frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode thumbnail")
    return encoded.tobytes()


def _appearance_family_labels(event: dict[str, Any], tracking: Any) -> set[str]:
    try:
        objects = json.loads(str(event.get("objects_json") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    if not isinstance(objects, list):
        return set()
    labels = {
        str(item.get("label") or "").strip().lower()
        for item in objects
        if isinstance(item, dict) and item.get("label")
    }
    families: set[str] = set()
    if "person" in labels:
        families.add("person")
    vehicle_labels = set(getattr(tracking, "vehicle_reid_labels", []) or [])
    if labels & vehicle_labels:
        families.add("vehicle")
    return families


def create_appearance_router(deps: AppearanceRouteDependencies) -> AppearanceRouteBundle:
    router = APIRouter()

    def with_manager(operation: Callable[[AppManager], Any]) -> Any:
        with deps.manager_lock:
            return operation(deps.get_manager())

    @router.get("/api/events/{event_id}/snapshot.jpg")
    def event_snapshot(event_id: int, download: bool = False) -> FileResponse:
        def response(active_manager: AppManager) -> FileResponse:
            event = active_manager.events.get(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="event not found")
            try:
                snapshot_path = event_snapshot_path(active_manager.storage_dir, event)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            return FileResponse(
                snapshot_path,
                media_type=snapshot_media_type(snapshot_path),
                filename=snapshot_path.name if download else None,
                headers={"Cache-Control": "private, max-age=3600"},
            )

        return with_manager(response)

    @router.get("/api/events/{event_id}/thumbnail.jpg")
    def event_thumbnail(
        event_id: int, width: int = 640, quality: int = 82
    ) -> FileResponse:
        safe_width = max(160, min(int(width), 1280))
        safe_quality = max(50, min(int(quality), 92))

        def response(active_manager: AppManager) -> FileResponse:
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
            identity = (
                f"{snapshot_path}:{stat.st_size}:{stat.st_mtime_ns}:"
                f"{safe_width}:{safe_quality}"
            )

            def build() -> bytes:
                frame = cv2.imread(str(snapshot_path))
                if frame is None:
                    raise HTTPException(
                        status_code=404, detail="snapshot is unavailable"
                    )
                return _jpeg_thumbnail(frame, safe_width, safe_quality)

            cached = active_manager.image_cache.get_or_create(
                "events", identity, build
            )
            return FileResponse(
                cached,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=86400, immutable"},
            )

        return with_manager(response)

    @router.get("/api/events/{event_id}/appearance-matches")
    def event_appearance_matches(
        event_id: int,
        hours: float = 24.0,
        limit: int = 12,
        cross_camera_only: bool = True,
    ) -> dict[str, Any]:
        bounded_hours = max(0.25, min(float(hours), 24.0 * 30.0))
        bounded_limit = max(1, min(int(limit), 100))

        def matches(active_manager: AppManager) -> dict[str, Any]:
            event = active_manager.events.get(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="event not found")
            try:
                anchor_at = datetime.fromisoformat(
                    str(event.get("created_at") or "").replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail="event timestamp is invalid"
                ) from exc
            if anchor_at.tzinfo is None:
                anchor_at = anchor_at.replace(tzinfo=timezone.utc)
            result = active_manager.appearance_index.matches(
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
                "matches": result,
            }

        return with_manager(matches)

    @router.get("/api/events/{event_id}/related-incidents")
    def event_related_incidents(
        event_id: int,
        hours: float = 24.0,
        sequence_seconds: float | None = None,
        limit: int = 16,
    ) -> dict[str, Any]:
        bounded_hours = max(0.25, min(float(hours), 24.0 * 30.0))
        bounded_limit = max(1, min(int(limit), 100))

        def related(active_manager: AppManager) -> dict[str, Any]:
            event = active_manager.events.get(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="event not found")
            try:
                anchor_at = datetime.fromisoformat(
                    str(event.get("created_at") or "").replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail="event timestamp is invalid"
                ) from exc
            if anchor_at.tzinfo is None:
                anchor_at = anchor_at.replace(tzinfo=timezone.utc)
            tracking = active_manager.config.detector.tracking
            base_window = max(
                1.0,
                min(
                    float(
                        sequence_seconds
                        if sequence_seconds is not None
                        else tracking.related_sequence_window_seconds
                    ),
                    300.0,
                ),
            )
            route_window = max(
                (
                    route.max_seconds
                    for route in tracking.camera_transition_routes
                    if route.enabled
                ),
                default=0.0,
            )
            window = min(300.0, max(base_window, route_window))
            anchor_families = _appearance_family_labels(event, tracking)
            visual_matches = active_manager.appearance_index.matches(
                event_id,
                start_at=(anchor_at - timedelta(hours=bounded_hours)).isoformat(),
                end_at=(anchor_at + timedelta(hours=bounded_hours)).isoformat(),
                cross_camera_only=True,
                limit=100,
            )
            temporal = active_manager.events.between(
                (anchor_at - timedelta(seconds=window)).isoformat(),
                (anchor_at + timedelta(seconds=window, microseconds=1)).isoformat(),
                limit=500,
            )
            visual_by_event = {int(item["event_id"]): item for item in visual_matches}
            combined: dict[int, dict[str, Any]] = {
                int(match["event_id"]): {**match, "relation_type": "appearance"}
                for match in visual_matches
                if match.get("visually_similar")
            }
            for candidate in temporal:
                candidate_id = int(candidate.get("id") or 0)
                if (
                    candidate_id <= 0
                    or candidate_id == event_id
                    or str(candidate.get("camera_id") or "")
                    == str(event.get("camera_id") or "")
                ):
                    continue
                shared_families = sorted(
                    anchor_families & _appearance_family_labels(candidate, tracking)
                )
                if not shared_families:
                    continue
                try:
                    candidate_at = datetime.fromisoformat(
                        str(candidate.get("created_at") or "").replace("Z", "+00:00")
                    )
                    if candidate_at.tzinfo is None:
                        candidate_at = candidate_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                delta = abs((candidate_at - anchor_at).total_seconds())
                if candidate_at <= anchor_at:
                    route_from = str(candidate.get("camera_id") or "")
                    route_to = str(event.get("camera_id") or "")
                else:
                    route_from = str(event.get("camera_id") or "")
                    route_to = str(candidate.get("camera_id") or "")
                route = match_camera_route(
                    tracking.camera_transition_routes, route_from, route_to, delta
                )
                visual = visual_by_event.get(candidate_id)
                similar = bool(visual and visual.get("visually_similar"))
                relation_type = (
                    "appearance_route"
                    if route is not None and similar
                    else "expected_route"
                    if route is not None
                    else "appearance_sequence"
                    if similar
                    else "sequence_candidate"
                )
                combined[candidate_id] = {
                    **(visual or {}),
                    "event_id": candidate_id,
                    "camera_id": str(candidate.get("camera_id") or ""),
                    "created_at": str(candidate.get("created_at") or ""),
                    "model_kind": shared_families[0],
                    "sequence_delta_seconds": round(delta, 3),
                    "relation_type": relation_type,
                    "visually_similar": similar,
                    **(route.as_dict() if route is not None else {}),
                }
            ordered = sorted(
                combined.values(),
                key=lambda item: (
                    0
                    if item.get("relation_type") == "appearance_route"
                    else 1
                    if item.get("relation_type") == "appearance_sequence"
                    else 2
                    if item.get("relation_type") == "expected_route"
                    else 3
                    if item.get("relation_type") == "appearance"
                    else 4,
                    float(item.get("sequence_delta_seconds") or 1e12),
                    -float(item.get("similarity") or 0.0),
                ),
            )[:bounded_limit]
            return {
                "event_id": event_id,
                "hours": bounded_hours,
                "sequence_seconds": window,
                "configured_routes": sum(
                    1 for route in tracking.camera_transition_routes if route.enabled
                ),
                "matches": ordered,
                "identity_notice": (
                    "Time proximity alone is a sequence candidate, not identity proof."
                ),
            }

        return with_manager(related)

    @router.get("/api/appearance-index/status")
    def appearance_index_status() -> dict[str, Any]:
        return with_manager(lambda active: active.appearance_index.status())

    @router.post("/api/appearance-index/backfill")
    def queue_appearance_backfill(start_at: str, end_at: str) -> dict[str, Any]:
        try:
            start = datetime.fromisoformat(str(start_at).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(end_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="backfill timestamps are invalid"
            ) from exc
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start or (end - start) > timedelta(days=1):
            raise HTTPException(
                status_code=422,
                detail="backfill window must be between 0 and 24 hours",
            )

        def queue(active_manager: AppManager) -> dict[str, Any]:
            events = active_manager.events.between(
                start.isoformat(), end.isoformat(), limit=10_000
            )
            queued = [
                int(event["id"])
                for event in events
                if _appearance_family_labels(
                    event, active_manager.config.detector.tracking
                )
                and active_manager.appearance_backfill.enqueue(
                    int(event["id"]),
                    str(event.get("camera_id") or ""),
                    delay_seconds=0,
                )
            ]
            return {
                "queued": len(queued),
                "event_ids": sorted(queued),
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
            }

        return with_manager(queue)

    handlers: dict[str, Callable[..., Any]] = {
        name: value
        for name, value in locals().copy().items()
        if callable(value) and name not in {"with_manager"}
    }
    return AppearanceRouteBundle(router=router, handlers=handlers)
