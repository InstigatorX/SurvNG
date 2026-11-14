"""Admin-only HomeKit runtime and pairing material; never cache setup secrets."""
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel


@dataclass(frozen=True)
class HomeKitRouteDependencies:
    get_manager: Callable[[], Any]


class PairingReset(BaseModel):
    confirm_camera_id: str


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

    return router
