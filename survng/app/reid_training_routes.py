"""HTTP boundary for optional ReID domain-adaptation training data."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .manager import AppManager
from .manager_access import ManagerAccessCoordinator, manager_generation_lease
from .reid_training.review import ReidTrainingReviewService


class ReidTrainingReviewRequest(BaseModel):
    action: Literal["confirm_same", "mark_different", "unknown", "reject"]
    left_event_id: int | None = Field(default=None, gt=0)
    left_track_id: int | None = Field(default=None, gt=0)
    right_event_id: int | None = Field(default=None, gt=0)
    right_track_id: int | None = Field(default=None, gt=0)
    sample_id: str | None = Field(default=None, min_length=1, max_length=200)
    side: Literal["left", "right", "both"] = "both"
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


@dataclass(frozen=True, slots=True)
class ReidTrainingRouteDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    manager_access: ManagerAccessCoordinator


@dataclass(frozen=True, slots=True)
class ReidTrainingRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_reid_training_router(
    deps: ReidTrainingRouteDependencies,
) -> ReidTrainingRouteBundle:
    router = APIRouter()

    def with_manager(operation: Callable[[AppManager], Any]) -> Any:
        with manager_generation_lease(
            deps.manager_access,
            deps.manager_lock,
            deps.get_manager,
        ) as active_manager:
            return operation(active_manager)

    def _review_service(active: AppManager) -> ReidTrainingReviewService:
        appearance = active.appearance_index

        def matches(
            event_id: int,
            *,
            hours: float = 168.0,
            limit: int = 24,
            cross_camera_only: bool = True,
        ) -> list[dict[str, Any]]:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=max(1.0, float(hours)))
            return appearance.matches(
                int(event_id),
                start_at=start.isoformat(),
                end_at=end.isoformat(),
                cross_camera_only=bool(cross_camera_only),
                limit=int(limit),
            )

        return ReidTrainingReviewService(active.reid_training, matches)

    @router.get("/api/reid-training/status")
    def reid_training_status() -> dict[str, Any]:
        def read(active: AppManager) -> dict[str, Any]:
            tracking = active.config.detector.tracking
            return {
                "collector_enabled": bool(tracking.reid_training_collector_enabled),
                **active.reid_training.status(),
            }

        return with_manager(read)

    @router.get("/api/reid-training/samples")
    def reid_training_samples(
        limit: int = Query(default=50, ge=1, le=500),
        event_id: int | None = Query(default=None, gt=0),
        camera_id: str | None = Query(default=None, max_length=128),
        review_status: str | None = Query(default=None, max_length=32),
        person_id: int | None = Query(default=None, gt=0),
    ) -> dict[str, Any]:
        def read(active: AppManager) -> dict[str, Any]:
            samples = active.reid_training.list_samples(
                limit=limit,
                event_id=event_id,
                camera_id=camera_id,
                review_status=review_status,
                person_id=person_id,
            )
            return {
                "count": len(samples),
                "samples": samples,
            }

        return with_manager(read)

    @router.get("/api/reid-training/samples/{sample_id}/crop.jpg")
    def reid_training_crop(sample_id: str) -> FileResponse:
        def read(active: AppManager) -> FileResponse:
            try:
                path = active.reid_training.resolve_crop_path(sample_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return FileResponse(
                path,
                media_type="image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/webp",
                headers={"Cache-Control": "private, max-age=86400"},
            )

        return with_manager(read)

    @router.get("/api/reid-training/review/queue")
    def reid_training_review_queue(
        limit: int = Query(default=20, ge=1, le=100),
        hours: float = Query(default=168.0, ge=1.0, le=720.0),
    ) -> dict[str, Any]:
        def read(active: AppManager) -> dict[str, Any]:
            return _review_service(active).review_queue(limit=limit, hours=hours)

        return with_manager(read)

    @router.post("/api/reid-training/review")
    def reid_training_review(body: ReidTrainingReviewRequest) -> dict[str, Any]:
        def write(active: AppManager) -> dict[str, Any]:
            try:
                return _review_service(active).apply_review(body.model_dump())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        return with_manager(write)

    return ReidTrainingRouteBundle(
        router=router,
        handlers={
            "reid_training_status": reid_training_status,
            "reid_training_samples": reid_training_samples,
            "reid_training_crop": reid_training_crop,
            "reid_training_review_queue": reid_training_review_queue,
            "reid_training_review": reid_training_review,
        },
    )
