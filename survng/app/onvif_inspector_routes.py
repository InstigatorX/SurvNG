from __future__ import annotations

from fastapi import APIRouter, Query

from .onvif_inspector import ONVIF_INSPECTOR, OnvifInspector


def create_onvif_inspector_router(
    inspector: OnvifInspector = ONVIF_INSPECTOR,
) -> APIRouter:
    router = APIRouter(prefix="/api/onvif-inspector", tags=["onvif-inspector"])

    @router.get("/events")
    def events(
        after: int = Query(0, ge=0),
        camera: str = "",
        recognized_only: bool = False,
        changes_only: bool = False,
        limit: int = Query(250, ge=1, le=1000),
    ):
        return inspector.events_after(
            after,
            camera=camera,
            recognized_only=recognized_only,
            changes_only=changes_only,
            limit=limit,
        )

    @router.get("/state")
    def state():
        return inspector.state_snapshot()

    @router.post("/clear")
    def clear():
        return inspector.clear()

    return router
