"""Camera media, debugging, and runtime-control HTTP boundary."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import websockets
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .go2rtc import Go2RtcError
from .incident_utils import event_snapshot_path, snapshot_media_type
from .manager import AppManager
from .manager_access import ManagerAccessCoordinator

LOGGER = logging.getLogger(__name__)


class CameraFeatureState(BaseModel):
    enabled: bool


@dataclass(frozen=True, slots=True)
class CameraApiDependencies:
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    manager_access: ManagerAccessCoordinator | None = None


@dataclass(frozen=True, slots=True)
class CameraApiRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_camera_api_router(deps: CameraApiDependencies) -> CameraApiRouteBundle:
    router = APIRouter()

    def with_manager_lease(operation: Callable[[AppManager], Any]) -> Any:
        if deps.manager_access is None:
            with deps.manager_lock:
                return operation(deps.get_manager())
        with deps.manager_access.lease(deps.manager_lock, deps.get_manager) as active:
            return operation(active)

    def with_manager(operation: Callable[[AppManager], Any]) -> Any:
        with deps.manager_lock:
            return operation(deps.get_manager())

    def cached_snapshot(
        worker: Any,
        camera: Any,
        source: str,
        status: dict[str, Any],
    ) -> tuple[bytes | None, bool]:
        normalized = camera.normalized_source(source)
        freshness_key = "main_frame_fresh" if normalized == "main" else "frame_fresh"
        freshness_known = freshness_key in status
        if freshness_known and not status.get(freshness_key):
            return None, True
        if status.get(freshness_key):
            return worker.snapshot(normalized), False
        return None, False

    @router.get("/api/cameras/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str, source: str = "live") -> Response:
        def response(active_manager: AppManager) -> Response:
            worker = active_manager.workers.get(camera_id)
            camera = active_manager.camera(camera_id)
            if worker is None or camera is None:
                raise HTTPException(status_code=404, detail="camera not found")
            status = worker.status()
            if not status.get("running"):
                raise HTTPException(status_code=503, detail="camera is powered off")
            image, stale = cached_snapshot(worker, camera, source, status)
            if image is None and not stale:
                try:
                    image = active_manager.go2rtc.snapshot(camera, source)
                except Go2RtcError:
                    fallback = (
                        "live"
                        if source == "main"
                        and camera.source_url("main") == camera.source_url("live")
                        else source
                    )
                    image = worker.snapshot(fallback)
            if image is None:
                raise HTTPException(status_code=503, detail="no frame available")
            return Response(
                image,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        return with_manager_lease(response)

    @router.get("/api/cameras/{camera_id}/zone-snapshot.jpg")
    def zone_snapshot(camera_id: str, source: str = "live") -> Response:
        def response(active_manager: AppManager) -> Response:
            worker = active_manager.workers.get(camera_id)
            camera = active_manager.camera(camera_id)
            if worker is None or camera is None:
                raise HTTPException(status_code=404, detail="camera not found")
            status = worker.status()
            image, stale = cached_snapshot(worker, camera, source, status)
            if image is None and status.get("running") and not stale:
                try:
                    image = active_manager.go2rtc.snapshot(camera, source)
                except Go2RtcError:
                    fallback = (
                        "live"
                        if source == "main"
                        and camera.source_url("main") == camera.source_url("live")
                        else source
                    )
                    image = worker.snapshot(fallback)
            if image is not None:
                return Response(
                    image,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"},
                )
            for event in active_manager.events.recent(1000):
                if event.get("camera_id") != camera_id:
                    continue
                try:
                    path = event_snapshot_path(active_manager.storage_dir, event)
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                return FileResponse(
                    path,
                    media_type=snapshot_media_type(path),
                    headers={"Cache-Control": "no-store"},
                )
            raise HTTPException(
                status_code=503, detail="no camera or event snapshot available"
            )

        return with_manager_lease(response)

    @router.get("/api/cameras/{camera_id}/live-info")
    def live_info(camera_id: str, response: Response, source: str = "live") -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"

        def info(active_manager: AppManager) -> dict[str, Any]:
            camera = active_manager.camera(camera_id)
            worker = active_manager.workers.get(camera_id)
            if camera is None or worker is None:
                raise HTTPException(status_code=404, detail="camera not found")
            if not worker.status().get("running"):
                raise HTTPException(status_code=503, detail="camera is powered off")
            try:
                return active_manager.go2rtc.stream_info(camera, source)
            except Go2RtcError as exc:
                return {
                    "available": False,
                    "video_codec": "",
                    "video_codecs": [],
                    "compatibility": "native",
                    "delivery": "native",
                    "transcoding": False,
                    "error": str(exc)[:160],
                }

        return with_manager_lease(info)

    @router.get("/api/cameras/{camera_id}/stream.mjpg")
    async def stream(
        camera_id: str,
        request: Request,
        source: str = "live",
        fps: float = 4.0,
    ) -> StreamingResponse:
        with deps.manager_lock:
            worker = deps.get_manager().workers.get(camera_id)
            if worker is None:
                raise HTTPException(status_code=404, detail="camera not found")
            if not worker.status().get("running"):
                raise HTTPException(status_code=503, detail="camera is powered off")
        frame_interval = 1.0 / max(0.5, min(4.0, fps))

        async def frames():
            while not await request.is_disconnected():
                with deps.manager_lock:
                    if deps.get_manager().workers.get(camera_id) is not worker:
                        return
                image = await asyncio.to_thread(worker.snapshot, source)
                if image is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + image
                        + b"\r\n"
                    )
                await asyncio.sleep(frame_interval if image is not None else 0.1)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def relay_go2rtc_websocket(
        websocket: WebSocket, camera_id: str, transport: str
    ) -> None:
        try:
            with deps.manager_lock:
                active_manager = deps.get_manager()
                camera = active_manager.camera(camera_id)
                worker = active_manager.workers.get(camera_id)
                if camera is None or worker is None:
                    raise Go2RtcError("camera not found")
                if not worker.status().get("running"):
                    raise Go2RtcError("camera is powered off")
            upstream_url = await asyncio.to_thread(
                active_manager.go2rtc.websocket_url,
                camera,
                websocket.query_params.get("source", "live"),
            )
        except (Go2RtcError, OSError, RuntimeError):
            await websocket.close(code=1008)
            return
        accepted = False
        tasks: list[asyncio.Task] = []
        try:
            async with websockets.connect(
                upstream_url,
                open_timeout=5,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=10,
                max_size=8 * 1024 * 1024,
                max_queue=4,
                compression=None,
            ) as upstream:
                await websocket.accept()
                accepted = True

                async def browser_to_go2rtc() -> None:
                    while True:
                        message = await websocket.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("text") is not None:
                            await upstream.send(message["text"])
                        elif message.get("bytes") is not None:
                            await upstream.send(message["bytes"])

                async def go2rtc_to_browser() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                tasks = [
                    asyncio.create_task(browser_to_go2rtc()),
                    asyncio.create_task(go2rtc_to_browser()),
                ]
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            pass
        except Exception as exc:
            LOGGER.warning("%s relay failed for %s: %s", transport, camera_id, exc)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await websocket.close(code=1000 if accepted else 1013)
            except (RuntimeError, WebSocketDisconnect):
                pass

    @router.websocket("/api/cameras/{camera_id}/webrtc")
    async def webrtc_signaling(websocket: WebSocket, camera_id: str) -> None:
        await relay_go2rtc_websocket(websocket, camera_id, "WebRTC signaling")

    @router.websocket("/api/cameras/{camera_id}/mse")
    async def mse_stream(websocket: WebSocket, camera_id: str) -> None:
        await relay_go2rtc_websocket(websocket, camera_id, "MSE stream")

    def control(operation: Callable[[AppManager], bool]) -> dict[str, bool]:
        if not with_manager(operation):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"ok": True}

    @router.post("/api/cameras/{camera_id}/camera/start")
    def start_camera(camera_id: str) -> dict[str, bool]:
        return control(lambda active: active.start_camera(camera_id))

    @router.post("/api/cameras/{camera_id}/camera/stop")
    def stop_camera(camera_id: str) -> dict[str, bool]:
        return control(lambda active: active.stop_camera(camera_id))

    @router.post("/api/cameras/{camera_id}/motion-test")
    def motion_test(camera_id: str) -> dict[str, bool]:
        def trigger(active_manager: AppManager) -> bool:
            worker = active_manager.workers.get(camera_id)
            if worker is None:
                return False
            worker.handle_motion_event("manual/test", "manual GUI trigger")
            return True

        return control(trigger)

    @router.get("/api/cameras/{camera_id}/motion-debug")
    def motion_debug_status(camera_id: str) -> dict[str, Any]:
        def status(active_manager: AppManager) -> dict[str, Any]:
            worker = active_manager.workers.get(camera_id)
            if worker is None:
                raise HTTPException(status_code=404, detail="camera not found")
            return worker.motion_debug_status()

        return with_manager(status)

    @router.put("/api/cameras/{camera_id}/motion-debug")
    def set_motion_debug(camera_id: str, state: CameraFeatureState) -> dict[str, Any]:
        def update(active_manager: AppManager) -> dict[str, Any]:
            worker = active_manager.workers.get(camera_id)
            if worker is None:
                raise HTTPException(status_code=404, detail="camera not found")
            worker.set_motion_debug_enabled(state.enabled)
            return worker.motion_debug_status()

        return with_manager(update)

    @router.get("/api/cameras/{camera_id}/motion-debug/{layer}.jpg")
    def motion_debug_image(camera_id: str, layer: str) -> Response:
        def response(active_manager: AppManager) -> Response:
            worker = active_manager.workers.get(camera_id)
            if worker is None:
                raise HTTPException(status_code=404, detail="camera not found")
            image = worker.motion_debug_image(layer)
            if image is None:
                raise HTTPException(
                    status_code=404, detail="motion debug layer not available"
                )
            return Response(
                content=image,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        return with_manager(response)

    @router.post("/api/cameras/{camera_id}/recording/start")
    def start_recording(camera_id: str, source: str = "main") -> dict[str, Any]:
        del source
        if not with_manager(lambda active: active.set_recording(camera_id, True)):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"ok": True, "recording_enabled": True}

    @router.post("/api/cameras/{camera_id}/recording/stop")
    def stop_recording(camera_id: str, source: str | None = None) -> dict[str, Any]:
        del source
        if not with_manager(lambda active: active.set_recording(camera_id, False)):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"ok": True, "recording_enabled": False}

    @router.put("/api/cameras/{camera_id}/recording")
    def set_camera_recording(camera_id: str, state: CameraFeatureState) -> dict[str, Any]:
        if not with_manager(
            lambda active: active.set_recording(camera_id, state.enabled)
        ):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"ok": True, "recording_enabled": state.enabled}

    @router.put("/api/cameras/{camera_id}/detection")
    def set_camera_detection(camera_id: str, state: CameraFeatureState) -> dict[str, Any]:
        if not with_manager(
            lambda active: active.set_detection(camera_id, state.enabled)
        ):
            raise HTTPException(status_code=404, detail="camera not found")
        return {"ok": True, "detection_enabled": state.enabled}

    handlers: dict[str, Callable[..., Any]] = {
        name: value
        for name, value in locals().copy().items()
        if callable(value) and name not in {"with_manager", "control"}
    }
    return CameraApiRouteBundle(router=router, handlers=handlers)
