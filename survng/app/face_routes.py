"""Face-management HTTP boundary."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .manager import AppManager
from .manager_access import ManagerAccessCoordinator


class FacePersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=1000)
    observation_id: int | None = Field(default=None, gt=0)


class FaceAssignment(BaseModel):
    person_id: int | None = Field(default=None, gt=0)


class FaceReferenceUpdate(BaseModel):
    pinned: bool


@dataclass(frozen=True, slots=True)
class FaceRouteDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    start_observation_sync: Callable[[], None]
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class FaceRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def _public_face_observation(observation: dict[str, Any]) -> dict[str, Any]:
    payload = dict(observation)
    payload.pop("snapshot_path", None)
    return payload


def create_face_router(deps: FaceRouteDependencies) -> FaceRouteBundle:
    router = APIRouter()

    def with_manager(operation: Callable[[AppManager], Any]) -> Any:
        if deps.manager_access is None:
            with deps.manager_lock:
                return operation(deps.get_manager())
        with deps.manager_access.lease(deps.manager_lock, deps.get_manager) as active:
            return operation(active)

    @router.get("/api/faces/status")
    def face_status() -> dict[str, Any]:
        deps.start_observation_sync()

        def status(active_manager: AppManager) -> dict[str, Any]:
            stats = active_manager.faces.stats()
            recognition = active_manager.faces.recognition_status()
            if recognition.get("ready"):
                pending = int(recognition.get("pending") or 0)
                failed = int(recognition.get("failed") or 0)
                too_small = int(recognition.get("too_small") or 0)
                message = (
                    f"Recognition ready on {recognition.get('device') or 'OpenVINO'}; "
                    f"{recognition.get('embedded', 0)} faces embedded and "
                    f"{recognition.get('suggested', 0)} suggestions awaiting review"
                    f"; {pending} pending, {too_small} below the recognition size, "
                    f"and {failed} processing failures."
                )
            else:
                message = str(
                    recognition.get("error")
                    or "Configure an OpenVINO face embedding model."
                )
            return {
                **stats,
                "recognition_ready": bool(recognition.get("ready")),
                "recognition_message": message,
                "recognition": recognition,
            }

        return with_manager(status)

    @router.get("/api/faces/people")
    def face_people() -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.people())

    @router.get("/api/faces/calibration")
    def face_calibration() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.calibration())

    @router.post("/api/faces/people")
    def create_face_person(payload: FacePersonCreate) -> dict[str, Any]:
        def create(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.create_person(
                    payload.name, payload.observation_id, payload.notes
                )
            except ValueError as exc:
                status_code = 404 if "not found" in str(exc).lower() else 409
                raise HTTPException(status_code=status_code, detail=str(exc)) from exc

        return with_manager(create)

    @router.delete("/api/faces/people/{person_id}")
    def delete_face_person(person_id: int) -> dict[str, Any]:
        def delete(active_manager: AppManager) -> dict[str, Any]:
            if not active_manager.faces.delete_person(person_id):
                raise HTTPException(status_code=404, detail="person not found")
            return {"deleted": True, "person_id": person_id}

        return with_manager(delete)

    @router.get("/api/faces/observations")
    def face_observations(
        person_id: int | None = None,
        camera_id: str = "",
        status: str = "all",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        resolved_status = (
            status if status in {"all", "known", "unknown", "suggested", "unusable"} else "all"
        )
        return with_manager(
            lambda active: [
                _public_face_observation(observation)
                for observation in active.faces.observations(
                    person_id=person_id,
                    camera_id=camera_id,
                    status=resolved_status,
                    limit=limit,
                    offset=offset,
                )
            ]
        )

    @router.get("/api/faces/observations/count")
    def face_observation_count(
        person_id: int | None = None,
        camera_id: str = "",
        status: str = "all",
    ) -> dict[str, int]:
        deps.start_observation_sync()
        resolved_status = (
            status if status in {"all", "known", "unknown", "suggested", "unusable"} else "all"
        )
        return with_manager(
            lambda active: {
                "total": active.faces.observation_count(
                    person_id=person_id,
                    camera_id=camera_id,
                    status=resolved_status,
                )
            }
        )

    @router.get("/api/faces/observations/{observation_id}")
    def face_observation(observation_id: int) -> dict[str, Any]:
        def get(active_manager: AppManager) -> dict[str, Any]:
            observation = active_manager.faces.observation(observation_id)
            if observation is None:
                raise HTTPException(status_code=404, detail="face observation not found")
            return _public_face_observation(observation)

        return with_manager(get)

    @router.put("/api/faces/observations/{observation_id}")
    def assign_face_observation(
        observation_id: int, payload: FaceAssignment
    ) -> dict[str, Any]:
        def assign(active_manager: AppManager) -> dict[str, Any]:
            try:
                observation = active_manager.faces.assign(
                    observation_id, payload.person_id
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if observation is None:
                raise HTTPException(status_code=404, detail="face observation not found")
            return _public_face_observation(observation)

        return with_manager(assign)

    @router.get("/api/faces/observations/{observation_id}/crop.jpg")
    def face_crop(observation_id: int, padding: float = 0.2) -> FileResponse:
        if not math.isfinite(padding):
            raise HTTPException(status_code=422, detail="padding must be finite")

        def crop(active_manager: AppManager) -> FileResponse:
            result = active_manager.faces.snapshot_path(observation_id)
            if result is None:
                raise HTTPException(status_code=404, detail="face observation not found")
            snapshot_path, box = result
            pad = max(0.0, min(float(padding), 1.0))
            try:
                stat = snapshot_path.stat()
            except OSError as exc:
                raise HTTPException(
                    status_code=404, detail="snapshot is unavailable"
                ) from exc
            box_identity = json.dumps(box, sort_keys=True, separators=(",", ":"))
            identity = (
                f"{observation_id}:{snapshot_path}:{stat.st_size}:{stat.st_mtime_ns}:"
                f"{box_identity}:{pad:.3f}"
            )

            def build() -> bytes:
                frame = cv2.imread(str(snapshot_path))
                if frame is None:
                    raise HTTPException(
                        status_code=404, detail="snapshot is unavailable"
                    )
                height, width = frame.shape[:2]
                x1, y1 = float(box.get("x1", 0)), float(box.get("y1", 0))
                x2, y2 = float(box.get("x2", 0)), float(box.get("y2", 0))
                dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
                left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
                right, bottom = min(width, int(x2 + dx)), min(height, int(y2 + dy))
                if right <= left or bottom <= top:
                    raise HTTPException(status_code=422, detail="face crop is invalid")
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame[top:bottom, left:right],
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )
                if not ok:
                    raise HTTPException(
                        status_code=500, detail="failed to encode face crop"
                    )
                return encoded.tobytes()

            cached = active_manager.image_cache.get_or_create("faces", identity, build)
            return FileResponse(
                cached,
                media_type="image/jpeg",
                headers={"Cache-Control": "private, max-age=86400, immutable"},
            )

        return with_manager(crop)

    @router.put("/api/faces/observations/{observation_id}/reference")
    def update_face_reference(
        observation_id: int,
        payload: FaceReferenceUpdate,
    ) -> dict[str, Any]:
        def update(active_manager: AppManager) -> dict[str, Any]:
            try:
                observation = active_manager.faces.set_reference_pinned(
                    observation_id,
                    payload.pinned,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if observation is None:
                raise HTTPException(status_code=404, detail="face observation not found")
            return _public_face_observation(observation)

        return with_manager(update)

    handlers: dict[str, Callable[..., Any]] = {
        name: value
        for name, value in locals().copy().items()
        if callable(value) and name not in {"with_manager"}
    }
    return FaceRouteBundle(router=router, handlers=handlers)
