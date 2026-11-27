"""Admin routes for face capture: status, and seeding past incidents.

Faces are personal data, so every route here requires the admin scope (see
``security.required_api_scope``).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .groups import GroupNotFound
from .store import SeedConflict

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceCaptureRouteDependencies:
    get_manager: Callable[[], Any]


class SeedRequest(BaseModel):
    start: str = Field(..., description="ISO 8601 with a timezone")
    end: str = Field(..., description="ISO 8601 with a timezone")
    cameras: list[str] | None = Field(default=None, description="camera ids; empty or missing means all")


class NameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class IgnoreRequest(BaseModel):
    ignored: bool = True


class MergeRequest(BaseModel):
    into: int


def _epoch(value: str, name: str) -> float:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be an ISO 8601 timestamp") from None
    if parsed.tzinfo is None:
        raise HTTPException(status_code=400, detail=f"{name} must include a timezone")
    return parsed.timestamp()


def _main_sizes(manager: Any, incident: dict[str, Any] | None = None) -> Callable[[str], tuple[int, int] | None]:
    """Main-stream frame size per camera, from its main recordings around the
    incident (for faces stored before their frame size was). A camera's
    resolution can change, so today's recordings need not match the incident."""
    from ..recording_media import mp4_video_dimensions
    from .binding import _epoch as epoch_of

    cache: dict[str, tuple[int, int] | None] = {}
    start = epoch_of((incident or {}).get("start_at"))
    end = epoch_of((incident or {}).get("end_at")) or start
    if start is None:
        end = datetime.now(timezone.utc).timestamp()
        start = end - 6 * 3600

    def size(camera_id: str) -> tuple[int, int] | None:
        if camera_id not in cache:
            cache[camera_id] = None
            try:
                rows = manager.recorder.recording_rows_between(
                    camera_id, start - 600, end + 600, "main", discover_missing=False)
            except Exception:
                rows = []
            for row in reversed(rows or []):
                found = mp4_video_dimensions(Path(row["path"]))
                if found:
                    cache[camera_id] = found
                    break
        return cache[camera_id]

    return size


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

    # People: groups of captured faces, named once -------------------------

    def groups() -> Any:
        return service().groups

    def edit(action: Callable[[], Any], affected: Callable[[], list[int]] | None = None) -> Any:
        """Run an edit; names in the events it touched reach their incidents.

        ``affected`` is read before the edit (a merge or a forget empties
        what it would look up afterwards).
        """
        try:
            events = affected() if affected is not None else []
            result = action()
        except GroupNotFound:
            raise HTTPException(status_code=404, detail="not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        # The edit is committed: what follows must not turn it into an error.
        try:
            # Again after: faces the capture thread grouped meanwhile too.
            events = sorted(set(events) | set(affected() if affected is not None else []))
            sync = getattr(deps.get_manager(), "sync_face_identities", None)
            if events and callable(sync):
                sync(events)
        except Exception:
            LOGGER.exception("a People page edit was saved but its names could not be queued for incidents")
        return result

    @router.get("/api/face-groups")
    def list_groups(
        status: Literal["unnamed", "named", "ignored"] = "unnamed",
        limit: int = Query(default=60, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        return {**groups().groups(status, limit, offset), "summary": groups().summary(), "cameras": cameras()}

    @router.get("/api/face-groups/{group_id}")
    def get_group(group_id: int) -> dict[str, Any]:
        return edit(lambda: groups().group(group_id))

    @router.post("/api/face-groups/{group_id}/name")
    def name_group(group_id: int, request: NameRequest) -> dict[str, Any]:
        return edit(lambda: groups().name_group(group_id, request.name),
                    lambda: groups().event_ids_of(group_ids=[group_id]))

    @router.post("/api/face-groups/{group_id}/unname")
    def unname_group(group_id: int) -> dict[str, Any]:
        return edit(lambda: groups().unname_group(group_id),
                    lambda: groups().event_ids_of(group_ids=[group_id]))

    @router.post("/api/face-groups/{group_id}/ignore")
    def ignore_group(group_id: int, request: IgnoreRequest) -> dict[str, Any]:
        return edit(lambda: groups().ignore_group(group_id, request.ignored),
                    lambda: groups().event_ids_of(group_ids=[group_id]))

    @router.post("/api/face-groups/{group_id}/merge")
    def merge_group(group_id: int, request: MergeRequest) -> dict[str, Any]:
        return edit(lambda: groups().merge_groups(group_id, request.into),
                    lambda: groups().event_ids_of(group_ids=[group_id, request.into]))

    @router.post("/api/face-captures/{face_id}/remove")
    def remove_face(face_id: int) -> dict[str, Any]:
        return edit(lambda: groups().remove_face(face_id),
                    lambda: groups().event_ids_of(face_ids=[face_id]))

    @router.get("/api/face-captures/by-events")
    def faces_by_events(ids: str = Query(..., max_length=8000, description="comma-separated event ids")) -> dict[str, Any]:
        try:
            event_ids = [int(part) for part in ids.split(",") if part.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="ids must be comma-separated event ids") from None
        if any(not 0 < event_id < 2**63 for event_id in event_ids):
            raise HTTPException(status_code=400, detail="ids must be positive event ids")
        return groups().for_events(event_ids)

    @router.get("/api/face-captures/by-incident/{incident_id}")
    def faces_by_incident(incident_id: str) -> dict[str, Any]:
        manager = deps.get_manager()
        incident = manager.events.scene_incident(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="incident not found")
        return groups().for_incident(incident, _main_sizes(manager, incident))

    @router.get("/api/face-captures/{face_id}/image")
    def face_image(face_id: int) -> FileResponse:
        capture = service()
        path = capture.groups.face_crop_path(face_id)
        if not path:
            raise HTTPException(status_code=404, detail="face not found")
        resolved = Path(path).resolve()
        # Only crops the store wrote are served.
        if not resolved.is_relative_to(capture.store.crop_dir.resolve()) or not resolved.is_file():
            raise HTTPException(status_code=404, detail="face image not found")
        return FileResponse(resolved, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @router.get("/api/face-names")
    def list_names() -> dict[str, Any]:
        """Named people, id and name only: the incident person filter, for every role
        that sees incidents (their names are on the incidents already). From the
        source that names incidents, so a choice always matches."""
        return {"people": deps.get_manager().identity_people()}

    @router.get("/api/face-people")
    def list_people() -> dict[str, Any]:
        return {"people": groups().people(), "summary": groups().summary()}

    @router.get("/api/face-people/{person_id}")
    def get_person(person_id: int) -> dict[str, Any]:
        return edit(lambda: groups().person(person_id))

    @router.post("/api/face-people/{person_id}/rename")
    def rename_person(person_id: int, request: NameRequest) -> dict[str, Any]:
        return edit(lambda: groups().rename_person(person_id, request.name),
                    lambda: groups().event_ids_of(person_ids=[person_id]))

    @router.delete("/api/face-people/{person_id}")
    def forget_person(person_id: int) -> dict[str, Any]:
        edit(lambda: groups().forget_person(person_id),
             lambda: groups().event_ids_of(person_ids=[person_id]))
        return {"forgotten": person_id}

    return router
