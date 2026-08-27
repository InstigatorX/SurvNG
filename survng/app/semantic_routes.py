"""Semantic search HTTP boundary."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from .manager_access import ManagerAccessCoordinator, manager_generation_lease
from .identity_projection import apply_event_identity
from .semantic_search import semantic_event_objects


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    object_labels: list[str] = Field(default_factory=list, max_length=100)
    start_at: str = Field(default="", max_length=64)
    end_at: str = Field(default="", max_length=64)
    limit: int = Field(default=50, ge=1, le=500)
    minimum_score: float = Field(default=-1.0, ge=-1.0, le=1.0)


class SemanticVisualSearchRequest(BaseModel):
    event_id: int = Field(ge=1)
    object_index: int = Field(ge=0)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    object_labels: list[str] = Field(default_factory=list, max_length=100)
    start_at: str = Field(default="", max_length=64)
    end_at: str = Field(default="", max_length=64)
    limit: int = Field(default=50, ge=1, le=500)
    minimum_score: float = Field(default=-1.0, ge=-1.0, le=1.0)
    source_kinds: list[str] = Field(default_factory=list, max_length=10)
    exclude_anchor: bool = True


class SemanticVisualFrameSearchRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    epoch: float = Field(gt=0)
    source: str = Field(default="main", pattern=r"^(main|live)$")
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    object_labels: list[str] = Field(default_factory=list, max_length=100)
    start_at: str = Field(default="", max_length=64)
    end_at: str = Field(default="", max_length=64)
    limit: int = Field(default=50, ge=1, le=500)
    minimum_score: float = Field(default=-1.0, ge=-1.0, le=1.0)
    source_kinds: list[str] = Field(default_factory=list, max_length=10)
    exclude_event_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_crop_bounds(self) -> "SemanticVisualFrameSearchRequest":
        epsilon = 1e-6
        values = (self.epoch, self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frame crop values must be finite")
        if self.x + self.width > 1 + epsilon or self.y + self.height > 1 + epsilon:
            raise ValueError("frame crop must stay within the preview image")
        return self


def normalized_crop_bounds(
    image_width: int,
    image_height: int,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[int, int, int, int]:
    """Convert a normalized crop into clamped, non-empty pixel bounds."""
    x1 = max(0, min(image_width, math.floor(x * image_width)))
    y1 = max(0, min(image_height, math.floor(y * image_height)))
    x2 = max(0, min(image_width, math.ceil((x + width) * image_width)))
    y2 = max(0, min(image_height, math.ceil((y + height) * image_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("selected recording frame crop is empty")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class SemanticRouteDependencies:
    get_manager: Callable[[], Any]
    manager_lock: threading.RLock
    recording_preview_path: Callable[..., Path]
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class SemanticRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_semantic_router(deps: SemanticRouteDependencies) -> SemanticRouteBundle:
    router = APIRouter()

    def hydrate_results(
        active_manager: Any,
        hits: list[Any],
        maximum: int,
        *,
        exclude_event_id: int | None = None,
    ) -> list[dict[str, Any]]:
        best_by_event: dict[int, Any] = {}
        for hit in hits:
            if exclude_event_id is not None and hit.event_id == exclude_event_id:
                continue
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
            for row in active_manager.events.get_many(list(best_by_event))
        }
        observations_by_event: dict[int, list[dict[str, Any]]] = {}
        face_store = getattr(active_manager, "faces", None)
        for_event_ids = getattr(face_store, "for_event_ids", None)
        if callable(for_event_ids):
            for observation in for_event_ids(list(best_by_event)):
                observations_by_event.setdefault(
                    int(observation["event_id"]), []
                ).append(observation)

        for event_id, event in event_rows.items():
            observations = observations_by_event.get(event_id, [])
            if not observations:
                continue
            faces: list[dict[str, Any]] = []
            for observation in observations:
                person_id = observation.get("person_id")
                if person_id is None:
                    continue
                faces.append({
                    "observation_id": int(observation.get("observation_id") or 0),
                    "identity_id": int(person_id),
                    "person_id": int(person_id),
                    "name": str(
                        observation.get("person_name")
                        or f"Person {int(person_id)}"
                    ),
                    "status": (
                        "automatic"
                        if bool(observation.get("auto_identified"))
                        or str(observation.get("review_status") or "")
                        == "auto_identified"
                        else "confirmed"
                    ),
                    "review_status": str(
                        observation.get("review_status") or "confirmed"
                    ),
                    "source": (
                        "automatic"
                        if bool(observation.get("auto_identified"))
                        or str(observation.get("review_status") or "")
                        == "auto_identified"
                        else "operator"
                    ),
                    "confidence": float(
                        observation.get("match_confidence") or 0.0
                    ),
                })
            if faces:
                event["faces"] = faces
                apply_event_identity(event)

        base_path = active_manager.config.base_path
        results = []
        for event_id, hit in best_by_event.items():
            event = event_rows.get(event_id)
            if event is None:
                continue
            results.append({
                "score": round(hit.score, 6),
                "rank_score": round(
                    hit.rank_score if hit.rank_score is not None else hit.score,
                    6,
                ),
                "match_strength": hit.match_strength,
                "component_scores": dict(hit.component_scores or {}),
                "evidence": {
                    "source_kind": hit.source_kind,
                    "source_key": hit.source_key,
                    "object_label": hit.object_label,
                    "bbox": hit.bbox,
                },
                "event": event,
                "snapshot_url": f"{base_path}/api/events/{event_id}/snapshot.jpg",
            })
        return results

    @router.get("/api/semantic-search/status")
    def semantic_search_status() -> dict[str, Any]:
        with manager_generation_lease(
            deps.manager_access, deps.manager_lock, deps.get_manager
        ) as active_manager:
            return active_manager.semantic_search_status()

    @router.post("/api/semantic-search")
    def semantic_search(request: SemanticSearchRequest) -> dict[str, Any]:
        with manager_generation_lease(
            deps.manager_access, deps.manager_lock, deps.get_manager
        ) as active_manager:
            maximum = min(request.limit, active_manager.config.semantic_search.max_results)
            semantic_service = active_manager.semantic_search
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
            results = hydrate_results(active_manager, hits, maximum)
            return {"query": request.query.strip(), "count": len(results), "results": results}

    @router.post("/api/semantic-search/visual")
    def semantic_visual_search(
        request: SemanticVisualSearchRequest,
    ) -> dict[str, Any]:
        with manager_generation_lease(
            deps.manager_access, deps.manager_lock, deps.get_manager
        ) as active_manager:
            event = active_manager.events.get(request.event_id)
            if event is None:
                raise HTTPException(status_code=404, detail="event not found")
            maximum = min(
                request.limit,
                active_manager.config.semantic_search.max_results,
            )
            try:
                hits = active_manager.semantic_search.search_event_object(
                    event,
                    request.object_index,
                    camera_ids=request.camera_ids,
                    object_labels=request.object_labels,
                    source_kinds=request.source_kinds,
                    start_at=request.start_at,
                    end_at=request.end_at,
                    limit=maximum * 4,
                    minimum_score=request.minimum_score,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            objects = semantic_event_objects(event)
            object_label = (
                str(objects[request.object_index].get("label") or "")
                if request.object_index < len(objects)
                else ""
            )
            results = hydrate_results(
                active_manager,
                hits,
                maximum,
                exclude_event_id=request.event_id if request.exclude_anchor else None,
            )
            return {
                "query_mode": "visual",
                "event_id": request.event_id,
                "object_index": request.object_index,
                "object_label": object_label,
                "count": len(results),
                "results": results,
            }

    @router.post("/api/semantic-search/visual-frame")
    def semantic_visual_frame_search(
        request: SemanticVisualFrameSearchRequest,
    ) -> dict[str, Any]:
        with manager_generation_lease(
            deps.manager_access, deps.manager_lock, deps.get_manager
        ) as active_manager:
            if active_manager.camera(request.camera_id) is None:
                raise HTTPException(status_code=404, detail="camera not found")
            rows = active_manager.recorder.recording_rows_between(
                request.camera_id,
                request.epoch - 0.001,
                request.epoch + 0.001,
                request.source,
                discover_missing=False,
            )
            row = next(
                (
                    candidate
                    for candidate in rows
                    if float(candidate.get("start_epoch") or 0) <= request.epoch
                    < float(candidate.get("end_epoch") or 0)
                ),
                None,
            )
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail="no recording exists at this time",
                )

            maximum = min(
                request.limit,
                active_manager.config.semantic_search.max_results,
            )
            try:
                preview_path = deps.recording_preview_path(
                    active_manager,
                    row,
                    request.epoch,
                    width=1280,
                    exact=True,
                )
                frame = cv2.imread(str(preview_path))
                if frame is None or frame.size == 0:
                    raise ValueError("recording frame preview is unavailable")
                image_height, image_width = frame.shape[:2]
                x1, y1, x2, y2 = normalized_crop_bounds(
                    image_width,
                    image_height,
                    x=request.x,
                    y=request.y,
                    width=request.width,
                    height=request.height,
                )
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    raise ValueError("selected recording frame crop is empty")
                hits = active_manager.semantic_search.search_image(
                    crop,
                    camera_ids=request.camera_ids,
                    object_labels=request.object_labels,
                    source_kinds=request.source_kinds,
                    start_at=request.start_at,
                    end_at=request.end_at,
                    limit=maximum * 4,
                    minimum_score=request.minimum_score,
                )
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            results = hydrate_results(
                active_manager,
                hits,
                maximum,
                exclude_event_id=request.exclude_event_id,
            )
            return {
                "query_mode": "visual",
                "camera_id": request.camera_id,
                "epoch": request.epoch,
                "crop": {
                    "x": request.x,
                    "y": request.y,
                    "width": request.width,
                    "height": request.height,
                },
                "count": len(results),
                "results": results,
            }

    return SemanticRouteBundle(
        router=router,
        handlers={
            "semantic_search_status": semantic_search_status,
            "semantic_search": semantic_search,
            "semantic_visual_search": semantic_visual_search,
            "semantic_visual_frame_search": semantic_visual_frame_search,
        },
    )
