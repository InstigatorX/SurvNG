"""Admin-only HomeKit runtime and pairing material; never cache setup secrets.

The installer and encoder endpoints authenticate with their own token. They are
public at the session boundary and still reject a missing or reused credential.
"""
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .homekit_nodes import ffmpeg_archive_iter, media_ingest_root, node_releases, public_api_root, runtime_archive_iter


@dataclass(frozen=True)
class HomeKitRouteDependencies:
    get_manager: Callable[[], Any]


class PairingReset(BaseModel):
    confirm_camera_id: str


class NodeReport(BaseModel):
    name: str = ""
    hardware: str = ""
    ffmpeg: str = ""
    profiles: dict = {}
    backends: dict = {}
    fps: dict = {}

    model_config = {"extra": "ignore"}


def _bearer(request: Request) -> str:
    scheme, _, secret = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secret.strip():
        raise HTTPException(status_code=401, detail="encoder authentication required")
    return secret.strip()


def _node(service, request: Request):
    node = service.nodes.authenticate(_bearer(request))
    if node is None:
        raise HTTPException(status_code=401, detail="encoder authentication required")
    return node


def pairing_qr(uri: str) -> str:
    if not uri.startswith("X-HM://") or len(uri) > 100:
        raise ValueError("invalid HomeKit setup URI")
    matrix = cv2.QRCodeEncoder_create().encode(uri)
    # OpenCV supplies a quiet zone. Add two more modules for reliable scanning.
    side = len(matrix) + 4
    path = "".join(f"M{x + 2} {y + 2}h1v1h-1z" for y, row in enumerate(matrix) for x, value in enumerate(row) if value == 0)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" shape-rendering="crispEdges"><rect width="{side}" height="{side}" fill="white"/><path d="{path}" fill="black"/></svg>'


def create_homekit_router(deps: HomeKitRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/homekit/status")
    def status(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return deps.get_manager().homekit.status()

    @router.post("/api/homekit/cameras/{camera_id}/pairing")
    def pairing(camera_id: str, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        try:
            material = deps.get_manager().homekit.pairing(camera_id)
            return {"setupCode": material["setupCode"], "qrSvg": pairing_qr(material["setupUri"])}
        except (OSError, RuntimeError, ValueError, KeyError, cv2.error):
            raise HTTPException(status_code=409, detail="HomeKit camera is not ready for pairing") from None

    @router.post("/api/homekit/cameras/{camera_id}/reset-pairing")
    def reset_pairing(camera_id: str, request: PairingReset, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        if request.confirm_camera_id != camera_id:
            raise HTTPException(status_code=400, detail="Camera confirmation does not match")
        try:
            deps.get_manager().homekit.reset_pairing(camera_id)
        except (OSError, RuntimeError):
            raise HTTPException(status_code=409, detail="HomeKit pairing could not be reset") from None
        return {"reset": True}

    def installer_root(request: Request) -> str:
        return public_api_root(str(request.base_url), deps.get_manager().config.base_path)

    @router.post("/api/homekit/nodes")
    def begin_enrollment(request: Request, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        try:
            return deps.get_manager().homekit.nodes.begin_enrollment(installer_root(request))
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(status_code=409, detail="HomeKit encoder enrollment is unavailable") from None

    @router.delete("/api/homekit/nodes/{node_id}")
    def remove_node(node_id: str, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        service = deps.get_manager().homekit
        assigned = [camera.id for camera in service.config.cameras if camera.homekit.encoder == node_id]
        try:
            service.nodes.remove(node_id, assigned)
        except KeyError:
            raise HTTPException(status_code=404, detail="HomeKit encoder was not found") from None
        except RuntimeError:
            raise HTTPException(status_code=409, detail="Move this encoder's cameras back to this server first") from None
        return {"removed": True}

    @router.get("/api/homekit/enroll/{token}")
    def enrollment_script(token: str, request: Request) -> Response:
        try:
            script = deps.get_manager().homekit.nodes.enrollment_script(token, installer_root(request))
        except (KeyError, ValueError, OSError, RuntimeError):
            raise HTTPException(status_code=404, detail="Enrollment was not found") from None
        return PlainTextResponse(script, headers={"Cache-Control": "no-store"})

    @router.post("/api/homekit/enroll/{token}")
    def complete_enrollment(token: str) -> Response:
        try:
            enrolled = deps.get_manager().homekit.nodes.complete_enrollment(token, "encoder")
        except (KeyError, ValueError, OSError, RuntimeError):
            raise HTTPException(status_code=404, detail="Enrollment was not found") from None
        return PlainTextResponse(enrolled["environment"], headers={"Cache-Control": "no-store"})

    @router.post("/api/homekit/nodes/sync")
    def node_sync(report: NodeReport, request: Request) -> dict:
        service = deps.get_manager().homekit
        node = _node(service, request)
        try:
            payload = service.nodes.sync(node, report.model_dump(), service.node_assignments(), timeout=10)
        except (KeyError, OSError, RuntimeError):
            raise HTTPException(status_code=401, detail="encoder authentication required") from None
        payload["releases"] = node_releases(service.config.ffmpeg_path)
        try:
            payload["ingest"] = media_ingest_root(request.scope.get("server"), service.config.homekit.bind, service.config.base_path)
        except (TypeError, ValueError):
            pass
        return payload

    @router.post("/api/homekit/nodes/media")
    async def node_media(request: Request) -> Response:
        service = deps.get_manager().homekit
        node = _node(service, request)
        camera_id = request.headers.get("x-survng-camera", "")
        tier = request.headers.get("x-survng-tier", "")
        if not service.nodes.feed_allowed(node, camera_id, tier, service.node_assignments()):
            raise HTTPException(status_code=404, detail="encoder feed was not requested")
        async for chunk in request.stream():
            if chunk:
                service.nodes.write_feed(camera_id, int(tier), chunk)
        return Response(status_code=204)

    @router.post("/api/homekit/nodes/snapshot")
    async def node_snapshot(request: Request) -> dict:
        service = deps.get_manager().homekit
        node = _node(service, request)
        camera_id = request.headers.get("x-survng-camera", "")
        data = await request.body()
        if not service.nodes.accept_snapshot(node, camera_id, data, service.node_assignments()):
            raise HTTPException(status_code=404, detail="snapshot was not requested")
        return {"accepted": True}

    @router.get("/api/homekit/nodes/agent")
    def node_agent(request: Request) -> Response:
        service = deps.get_manager().homekit
        _node(service, request)
        try:
            from .homekit_nodes import agent_release
            payload = agent_release()[1]
        except (OSError, FileNotFoundError, RuntimeError):
            raise HTTPException(status_code=409, detail="HomeKit encoder agent is not built") from None
        return Response(payload, media_type="application/gzip", headers={"Cache-Control": "no-store"})

    @router.get("/api/homekit/nodes/ffmpeg")
    def node_ffmpeg(request: Request) -> StreamingResponse:
        service = deps.get_manager().homekit
        _node(service, request)
        try:
            stream = ffmpeg_archive_iter(service.config.ffmpeg_path)
        except (OSError, ValueError):
            raise HTTPException(status_code=409, detail="HomeKit encoder ffmpeg is unavailable") from None
        return StreamingResponse(stream, media_type="application/gzip", headers={"Cache-Control": "no-store"})

    @router.get("/api/homekit/nodes/runtime")
    def node_runtime(request: Request) -> StreamingResponse:
        service = deps.get_manager().homekit
        _node(service, request)
        try:
            stream = runtime_archive_iter()
        except (OSError, ValueError):
            raise HTTPException(status_code=409, detail="HomeKit encoder runtime is unavailable") from None
        return StreamingResponse(stream, media_type="application/gzip", headers={"Cache-Control": "no-store"})

    return router
