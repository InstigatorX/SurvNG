"""Property bird's-eye image, placement, and recent zone activity."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from .camera_fov import lookup_fov
from .config import (
    AppConfig,
    PropertyCameraPlacement,
    PropertyImageConfig,
    PropertyViewConfig,
    PropertyZoneOverlay,
)
from .property_imagery import (
    ImageryHttp,
    detection_summary,
    fetch_property_image,
    parse_capture_date,
    placement_needs_review,
    property_image_file,
    restore_property_image,
    store_property_image,
)

LOGGER = logging.getLogger(__name__)
_IMAGERY_LOCK = threading.Lock()


class PropertyImageryRequest(BaseModel):
    address: str = Field(default="", max_length=200)
    latitude: float | None = Field(default=None, ge=-85, le=85, allow_inf_nan=False)
    longitude: float | None = Field(default=None, ge=-180, le=180, allow_inf_nan=False)
    width_m: float = Field(default=120, ge=20, le=2000, allow_inf_nan=False)
    height_m: float = Field(default=120, ge=20, le=2000, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_location(self) -> "PropertyImageryRequest":
        has_latitude = self.latitude is not None
        has_longitude = self.longitude is not None
        if has_latitude != has_longitude:
            raise ValueError("latitude and longitude are required together")
        if not (has_latitude and has_longitude) and not self.address.strip():
            raise ValueError("address or latitude and longitude is required")
        return self


class PropertyLayoutRequest(BaseModel):
    cameras: list[PropertyCameraPlacement] = Field(default_factory=list)
    zones: list[PropertyZoneOverlay] = Field(default_factory=list)


class PropertyLocationRequest(BaseModel):
    address: str = Field(default="", max_length=300)
    latitude: float = Field(ge=-85, le=85, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    width_m: float = Field(default=120, ge=20, le=2000, allow_inf_nan=False)
    height_m: float = Field(default=120, ge=20, le=2000, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class PropertyRouteDependencies:
    get_config: Callable[[], AppConfig]
    apply_config: Callable[..., tuple[AppConfig, dict[str, object]]]
    recent_object_events: Callable[[int], list[dict]]
    lock: threading.RLock
    client: ImageryHttp | None = None


def create_property_router(deps: PropertyRouteDependencies) -> APIRouter:
    router = APIRouter()

    @router.get("/api/property/imagery")
    def get_property_imagery() -> FileResponse:
        config = deps.get_config()
        if config.property.image is None:
            raise HTTPException(status_code=404, detail="property image has not been downloaded")
        path = property_image_file(Path(config.storage_dir))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="property image file is missing")
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @router.post("/api/property/imagery")
    def post_property_imagery(body: PropertyImageryRequest) -> dict:
        if not _IMAGERY_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="An imagery search is already running")
        try:
            try:
                fetched = fetch_property_image(
                    address=body.address,
                    latitude=body.latitude,
                    longitude=body.longitude,
                    width_m=body.width_m,
                    height_m=body.height_m,
                    client=deps.client,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            except OSError as error:
                LOGGER.warning("property imagery download failed (%s)", type(error).__name__)
                raise HTTPException(status_code=502, detail="Aerial imagery is unavailable") from error
            current = deps.get_config()
            previous = current.property.image
            warning = placement_needs_review(
                previous_latitude=None if previous is None else previous.latitude,
                previous_longitude=None if previous is None else previous.longitude,
                previous_width_m=None if previous is None else previous.width_m,
                previous_height_m=None if previous is None else previous.height_m,
                has_placements=bool(current.property.cameras or current.property.zones),
                latitude=float(fetched["latitude"]),
                longitude=float(fetched["longitude"]),
                width_m=float(fetched["width_m"]),
                height_m=float(fetched["height_m"]),
            )
            image = PropertyImageConfig.model_validate({
                key: value for key, value in fetched.items() if key not in {"image_bytes", "candidates", "rejected_sources"}
            })
            storage_dir = Path(current.storage_dir)
            previous_bytes = store_property_image(storage_dir, fetched["image_bytes"])
            try:
                with deps.lock:
                    latest = deps.get_config().model_copy(deep=True)
                    latest.property.image = image
                    latest.property.address = image.address
                    latest.property.latitude = image.latitude
                    latest.property.longitude = image.longitude
                    latest.property.width_m = image.width_m
                    effective, result = deps.apply_config(latest, assign_ids=False)
            except Exception:
                restore_property_image(storage_dir, previous_bytes)
                raise
        finally:
            _IMAGERY_LOCK.release()
        return {
            "ok": True,
            "placement_warning": warning,
            "image": effective.property.image.model_dump(mode="json") if effective.property.image else None,
            "candidates": fetched["candidates"],
            "camera_workers_restarted": bool(result.get("camera_workers_restarted")),
            "apply_mode": result.get("apply_mode"),
        }

    @router.put("/api/property/layout")
    def put_property_layout(body: PropertyLayoutRequest) -> dict:
        with deps.lock:
            current = deps.get_config()
            try:
                layout = PropertyViewConfig(
                    image=current.property.image,
                    address=current.property.address,
                    latitude=current.property.latitude,
                    longitude=current.property.longitude,
                    width_m=current.property.width_m,
                    cameras=body.cameras,
                    zones=body.zones,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            next_config = current.model_copy(deep=True)
            next_config.property = layout
            effective, result = deps.apply_config(next_config, assign_ids=False)
        return {
            "ok": True,
            "property": effective.property.model_dump(mode="json"),
            "camera_workers_restarted": bool(result.get("camera_workers_restarted")),
            "apply_mode": result.get("apply_mode"),
        }

    @router.put("/api/property/location")
    def put_property_location(body: PropertyLocationRequest) -> dict:
        with deps.lock:
            current = deps.get_config()
            next_config = current.model_copy(deep=True)
            next_config.property.address = body.address.strip()
            next_config.property.latitude = body.latitude
            next_config.property.longitude = body.longitude
            next_config.property.width_m = body.width_m
            if next_config.property.image is not None:
                next_config.property.image.address = body.address.strip()
                next_config.property.image.latitude = body.latitude
                next_config.property.image.longitude = body.longitude
                next_config.property.image.width_m = body.width_m
                next_config.property.image.height_m = body.height_m
            effective, result = deps.apply_config(next_config, assign_ids=False)
        return {
            "ok": True,
            "latitude": effective.property.latitude,
            "longitude": effective.property.longitude,
            "address": effective.property.address,
            "width_m": effective.property.width_m,
            "camera_workers_restarted": bool(result.get("camera_workers_restarted")),
        }

    @router.get("/api/property/fov")
    def get_property_fov(make: str = "", model: str = "") -> dict:
        return lookup_fov(make[:80], model[:80])

    @router.get("/api/property/activity")
    def get_property_activity(seconds: int = 90) -> dict:
        window = max(1, min(int(seconds), 90))
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)
        events = []
        for row in deps.recent_object_events(400):
            created = parse_capture_date(row.get("created_at"))
            if created is None or created < cutoff:
                continue
            zones, classes = detection_summary(str(row.get("objects_json") or "[]"))
            if not zones:
                continue
            events.append({
                "camera_id": str(row.get("camera_id") or ""),
                "timestamp": created.isoformat(),
                "zones": zones,
                "classes": classes,
            })
        return {"seconds": window, "events": events}

    return router
