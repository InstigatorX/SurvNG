"""Semantic search HTTP boundary."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    object_labels: list[str] = Field(default_factory=list, max_length=100)
    start_at: str = Field(default="", max_length=64)
    end_at: str = Field(default="", max_length=64)
    limit: int = Field(default=50, ge=1, le=500)
    minimum_score: float = Field(default=-1.0, ge=-1.0, le=1.0)


@dataclass(frozen=True, slots=True)
class SemanticRouteDependencies:
    get_manager: Callable[[], Any]
    manager_lock: threading.RLock


@dataclass(frozen=True, slots=True)
class SemanticRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_semantic_router(deps: SemanticRouteDependencies) -> SemanticRouteBundle:
    router = APIRouter()

    @router.get("/api/semantic-search/status")
    def semantic_search_status() -> dict[str, Any]:
        with deps.manager_lock:
            return deps.get_manager().semantic_search_status()

    @router.post("/api/semantic-search")
    def semantic_search(request: SemanticSearchRequest) -> dict[str, Any]:
        with deps.manager_lock:
            active_manager = deps.get_manager()
            maximum = min(request.limit, active_manager.config.semantic_search.max_results)
            semantic_service = active_manager.semantic_search
            event_store = active_manager.events
            base_path = active_manager.config.base_path
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

    return SemanticRouteBundle(
        router=router,
        handlers={
            "semantic_search_status": semantic_search_status,
            "semantic_search": semantic_search,
        },
    )
