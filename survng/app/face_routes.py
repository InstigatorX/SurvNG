"""Face-management HTTP boundary."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .manager import AppManager
from .manager_access import ManagerAccessCoordinator


class FacePersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=1000)
    observation_id: int | None = Field(default=None, gt=0)


class FaceAssignment(BaseModel):
    person_id: int | None = Field(default=None, gt=0)



class FaceBulkReview(BaseModel):
    observation_ids: list[int] = Field(min_length=1, max_length=500)
    action: str = Field(pattern=r"^(assign|unassign)$")
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


def _etag_matches(if_none_match: str, etag: str) -> bool:
    for candidate in if_none_match.split(","):
        candidate = candidate.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        if candidate == etag:
            return True
    return False


def _conditional_json_response(request: Request, payload: Any) -> Response:
    response = JSONResponse(content=jsonable_encoder(payload))
    etag = f'"{hashlib.sha256(response.body).hexdigest()}"'
    cache_headers = {
        "Cache-Control": "private, no-cache",
        "ETag": etag,
    }
    if _etag_matches(request.headers.get("if-none-match", ""), etag):
        return Response(status_code=304, headers=cache_headers)
    response.headers.update(cache_headers)
    return response


def _revision_etag(namespace: str, revision: object) -> str:
    identity = f"{namespace}:{revision}".encode("utf-8")
    return f'"{hashlib.sha256(identity).hexdigest()}"'


def _revisioned_json_response(
    request: Request,
    namespace: str,
    revision_reader: Callable[[], object],
    payload_reader: Callable[[], Any],
) -> Response:
    payload: Any = None
    for _attempt in range(2):
        revision = str(revision_reader())
        etag = _revision_etag(namespace, revision)
        cache_headers = {
            "Cache-Control": "private, no-cache",
            "ETag": etag,
        }
        if _etag_matches(request.headers.get("if-none-match", ""), etag):
            return Response(status_code=304, headers=cache_headers)
        payload = payload_reader()
        if str(revision_reader()) == revision:
            response = JSONResponse(content=jsonable_encoder(payload))
            response.headers.update(cache_headers)
            return response
    # A continuously changing directory must never receive a validator for a
    # different snapshot. Fall back to the exact response-body validator.
    return _conditional_json_response(request, payload)


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

    @router.get("/api/faces/unknown-cluster-health")
    def face_unknown_cluster_health() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.unknown_cluster_health()
        )

    @router.post("/api/faces/unknown-clusters/rebuild")
    def rebuild_unknown_clusters() -> dict[str, Any]:
        deps.start_observation_sync()
        def rebuild(active_manager: AppManager) -> dict[str, Any]:
            active_manager.faces.refresh_unknown_clusters()
            return active_manager.faces.unknown_cluster_health()
        return with_manager(rebuild)

    def face_unknown_clusters_payload() -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.unknown_clusters())

    @router.get("/api/faces/unknown-clusters")
    def face_unknown_clusters(request: Request) -> Response:
        deps.start_observation_sync()
        return with_manager(lambda active: _revisioned_json_response(
            request,
            "unknown-clusters-v1",
            active.faces.unknown_clusters_revision,
            active.faces.unknown_clusters,
        ))


    @router.get("/api/faces/diagnostics/duplicates")
    def face_duplicate_diagnostics() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.duplicate_stats())

    @router.post("/api/faces/maintenance/dedupe")
    def face_dedupe(window_seconds: float = 60.0) -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.dedupe_exact_embeddings(
                window_seconds=window_seconds
            )
        )

    @router.get("/api/faces/diagnostics/confirmed-quality")
    def face_confirmed_quality(limit: int = 100) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: [
                _public_face_observation(item)
                for item in active.faces.confirmed_quality_issues(limit=limit)
            ]
        )

    @router.get("/api/faces/unknown-clusters/{cluster_id}/members")
    def face_unknown_cluster_members(
        cluster_id: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: [
                _public_face_observation(item)
                for item in active.faces.unknown_cluster_members(
                    cluster_id,
                    limit=limit,
                )
            ]
        )


    @router.get("/api/faces/review/queue")
    def face_review_queue(limit: int = 100) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: [
                _public_face_observation(item)
                for item in active.faces.review_queue(limit=limit)
            ]
        )

    @router.get("/api/faces/review/confirmed")
    def face_confirmed_match_diagnostics(
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.confirmed_match_diagnostics(limit=limit)
        )

    @router.post("/api/faces/review/bulk")
    def face_bulk_review(payload: FaceBulkReview) -> dict[str, Any]:
        def update(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.bulk_review(
                    payload.observation_ids,
                    action=payload.action,
                    person_id=payload.person_id,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return with_manager(update)



    @router.get("/api/faces/gallery-optimization")
    def face_gallery_optimization(
        max_references: int = 8,
    ) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.optimize_all_galleries(
                max_references=max_references,
                apply=False,
            )
        )

    @router.post("/api/faces/people/{person_id}/gallery/optimize")
    def face_person_gallery_optimize(
        person_id: int,
        max_references: int = 8,
        apply: bool = False,
    ) -> dict[str, Any]:
        deps.start_observation_sync()
        def optimize(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.optimize_person_gallery(
                    person_id,
                    max_references=max_references,
                    apply=apply,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return with_manager(optimize)

    @router.get("/api/faces/people/representation-health")
    def face_people_representation_health() -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.people_representation_health())

    @router.get("/api/faces/people/{person_id}/representation")
    def face_person_representation(person_id: int) -> dict[str, Any]:
        deps.start_observation_sync()
        def diagnostics(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.person_representation(person_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return with_manager(diagnostics)

    @router.get("/api/faces/people/{person_id}/gallery-candidates")
    def face_person_gallery_candidates(person_id: int, limit: int = 20) -> list[dict[str, Any]]:
        deps.start_observation_sync()
        def candidates(active_manager: AppManager) -> list[dict[str, Any]]:
            try:
                return [
                    _public_face_observation(item)
                    for item in active_manager.faces.gallery_candidates(person_id, limit=limit)
                ]
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return with_manager(candidates)

    @router.post("/api/faces/people/{person_id}/gallery/enrich")
    def face_person_gallery_enrich(person_id: int, target_count: int = 8) -> dict[str, Any]:
        deps.start_observation_sync()
        def enrich(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.enrich_person_gallery(person_id, target_count=target_count)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return with_manager(enrich)

    @router.get("/api/faces/people/{person_id}/history")
    def face_person_history(
        person_id: int,
        limit: int = 100,
    ) -> dict[str, Any]:
        deps.start_observation_sync()
        def history(active_manager: AppManager) -> dict[str, Any]:
            try:
                return active_manager.faces.person_history(
                    person_id,
                    limit=limit,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return with_manager(history)

    def face_people_payload() -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.people())

    @router.get("/api/faces/people")
    def face_people(request: Request) -> Response:
        deps.start_observation_sync()
        return with_manager(lambda active: _revisioned_json_response(
            request,
            "people-directory-v1",
            active.faces.people_directory_revision,
            active.faces.people,
        ))

    @router.get("/api/faces/camera-suitability")
    def face_camera_suitability() -> list[dict[str, Any]]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.camera_suitability())

    @router.get("/api/faces/benchmark/production")
    def face_benchmark_production() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.benchmark_production_matcher()
        )

    @router.get("/api/faces/benchmark/camera-pairs")
    def face_benchmark_camera_pairs() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(
            lambda active: active.faces.benchmark_camera_pairs()
        )

    @router.get("/api/faces/benchmark/by-identity")
    def face_benchmark_by_identity() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.benchmark_by_identity())

    @router.get("/api/faces/benchmark")
    def face_benchmark() -> dict[str, Any]:
        deps.start_observation_sync()
        return with_manager(lambda active: active.faces.benchmark())

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
            status if status in {"all", "known", "unknown", "suggested", "unusable", "pending"} else "all"
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
            status if status in {"all", "known", "unknown", "suggested", "unusable", "pending"} else "all"
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
    # Preserve the direct-call API while the HTTP endpoints add revalidation.
    handlers["face_people"] = face_people_payload
    handlers["face_unknown_clusters"] = face_unknown_clusters_payload
    return FaceRouteBundle(router=router, handlers=handlers)
