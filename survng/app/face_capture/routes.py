"""Admin routes for face capture: status, and seeding past incidents.

Faces are personal data, so every route here requires the admin scope (see
``security.required_api_scope``).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .store import SeedConflict


@dataclass(frozen=True, slots=True)
class FaceCaptureRouteDependencies:
    get_manager: Callable[[], Any]


class SeedRequest(BaseModel):
    start: str = Field(..., description="ISO 8601 with a timezone")
    end: str = Field(..., description="ISO 8601 with a timezone")
    cameras: list[str] | None = Field(default=None, description="camera ids; empty or missing means all")


def _epoch(value: str, name: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO 8601 timestamp") from None
    if parsed.tzinfo is None:
        raise HTTPException(status_code=400, detail=f"{name} must include a timezone")
    return parsed.timestamp()


def create_face_capture_router(deps: FaceCaptureRouteDependencies) -> APIRouter:
    router = APIRouter()

    def service() -> Any:
        capture = getattr(deps.get_manager(), "face_capture", None)
        if capture is None:
            raise HTTPException(status_code=503, detail="face capture is unavailable")
        return capture

    def cameras() -> list[dict[str, str]]:
        config = deps.get_manager().config
        return [{"id": camera.id, "name": camera.name or camera.id} for camera in config.cameras]

    @router.get("/api/face-capture/status")
    def face_capture_status() -> dict[str, Any]:
        capture = service()
        status = capture.status()
        if not status.get("seeds"):
            status["seeds"] = capture.store.seeds(10) if status.get("enabled") else []
        return {**status, "cameras": cameras()}

    @router.post("/api/face-capture/seeds", status_code=201)
    def create_seed(request: SeedRequest) -> dict[str, Any]:
        capture = service()
        if not capture.enabled():
            raise HTTPException(status_code=409, detail="Turn on face capture and save before seeding.")
        start = _epoch(request.start, "start")
        end = min(_epoch(request.end, "end"), datetime.now(timezone.utc).timestamp())
        if start >= end:
            raise HTTPException(status_code=400, detail="start must be before end (and before now)")
        known = {camera["id"] for camera in cameras()}
        wanted = sorted(set(request.cameras or []))
        unknown = [camera for camera in wanted if camera not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown cameras: {', '.join(unknown)}")
        total = capture.count_seed_events(start, end, wanted or None)
        try:
            seed_id = capture.store.create_seed(start_epoch=start, end_epoch=end, cameras=wanted, total=total)
        except SeedConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        capture.wake()
        return next(seed for seed in capture.store.seeds(10) if seed["id"] == seed_id)

    @router.post("/api/face-capture/seeds/{seed_id}/{action}")
    def change_seed(seed_id: int, action: Literal["pause", "resume", "cancel"]) -> dict[str, Any]:
        capture = service()
        state = {"pause": "paused", "resume": "running", "cancel": "cancelled"}[action]
        try:
            seed = capture.store.set_seed_state(seed_id, state)
        except KeyError:
            raise HTTPException(status_code=404, detail="seed not found") from None
        except SeedConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        capture.wake()
        return seed

    return router
