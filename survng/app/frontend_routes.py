"""Static frontend page and icon routes."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse, Response

from survng.app.pwa import (
    service_worker_allowed_scope,
    service_worker_script,
    web_app_manifest,
)


@dataclass(frozen=True, slots=True)
class FrontendRouteDependencies:
    frontend_response: Callable[[str], HTMLResponse]
    base_path: Callable[[], str]
    cache_version: Callable[[], str] | None = None


@dataclass(frozen=True, slots=True)
class FrontendRouteBundle:
    router: APIRouter
    handlers: dict[str, Callable[..., Any]]


def create_frontend_router(
    deps: FrontendRouteDependencies,
) -> FrontendRouteBundle:
    router = APIRouter()

    @router.get("/api/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(Path("survng/static/favicon.svg"), media_type="image/svg+xml")

    @router.get("/apple-touch-icon.png", include_in_schema=False)
    @router.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
    def apple_touch_icon() -> FileResponse:
        return FileResponse(Path("survng/static/apple-touch-icon.png"), media_type="image/png")

    @router.get("/manifest.webmanifest", include_in_schema=False)
    def progressive_web_manifest() -> Response:
        payload = web_app_manifest(deps.base_path())
        return Response(
            json.dumps(payload, indent=2) + "\n",
            media_type="application/manifest+json",
            headers={"Cache-Control": "no-cache"},
        )

    @router.get("/sw.js", include_in_schema=False)
    def progressive_web_service_worker() -> Response:
        base_path = deps.base_path()
        cache_version = deps.cache_version() if deps.cache_version is not None else "v1"
        return Response(
            service_worker_script(base_path, cache_version=cache_version),
            media_type="application/javascript; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Service-Worker-Allowed": service_worker_allowed_scope(base_path),
            },
        )

    pages = {
        "index": ("/", "index.html"),
        "recordings_page": ("/recordings", "recordings.html"),
        "recording_search_page": ("/recordings/search", "recordings.html"),
        "recording_exports_page": ("/recordings/exports", "recordings.html"),
        "exports_page": ("/exports", "recordings.html"),
        "timeline_page": ("/timeline", "recordings.html"),
        "timeline_exports_page": ("/timeline/exports", "recordings.html"),
        "search_page": ("/search", "recordings.html"),
        "config_page": ("/config", "config.html"),
        "admin_page": ("/admin", "config.html"),
        "incidents_page": ("/incidents", "index.html"),
        "faces_page": ("/faces", "index.html"),
        "people_page": ("/people", "index.html"),
        "live_page": ("/live", "index.html"),
        "onvif_page": ("/onvif", "onvif.html"),
    }
    handlers: dict[str, Callable[..., Any]] = {
        "health": health,
        "favicon": favicon,
        "apple_touch_icon": apple_touch_icon,
        "progressive_web_manifest": progressive_web_manifest,
        "progressive_web_service_worker": progressive_web_service_worker,
    }
    def page_handler(name: str, filename: str) -> Callable[[], HTMLResponse]:
        def render_page() -> HTMLResponse:
            return deps.frontend_response(filename)

        render_page.__name__ = name
        return render_page

    for name, (path, filename) in pages.items():
        render_page = page_handler(name, filename)
        router.add_api_route(path, render_page, methods=["GET"])
        handlers[name] = render_page

    return FrontendRouteBundle(router=router, handlers=handlers)
