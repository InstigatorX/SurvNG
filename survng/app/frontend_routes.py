"""Static frontend page and icon routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse


@dataclass(frozen=True, slots=True)
class FrontendRouteDependencies:
    frontend_response: Callable[[str], HTMLResponse]


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
