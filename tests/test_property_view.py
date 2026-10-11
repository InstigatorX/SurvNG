from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from survng.app.camera_fov import lookup_fov
from survng.app.config import AppConfig, PropertyCameraPlacement, PropertyViewConfig, PropertyZoneOverlay
from survng.app.config_application import (
    TargetedConfigApplication,
    hot_config_changes,
    manager_owned_config,
)
from survng.app.property_imagery import (
    ImageryCandidate,
    fetch_property_image,
    mercator_bbox,
    export_pixels,
    parse_esri_identify,
    parse_geocode,
    parse_naip_query,
    placement_needs_review,
    rank_imagery,
    select_imagery,
    store_property_image,
)
from survng.app.property_routes import (
    PropertyRouteDependencies,
    create_property_router,
)
from survng.app.security import required_api_scope


def _candidate(source: str, resolution: float, captured: str | None) -> ImageryCandidate:
    when = datetime.fromisoformat(captured) if captured else None
    return ImageryCandidate(source, when, resolution, source, f"https://example.invalid/{source}")


def test_ranking_prefers_finer_resolution_then_newer_date_inside_the_band():
    coarse_new = _candidate("esri", 0.60, "2024-06-01T00:00:00+00:00")
    fine_old = _candidate("naip", 0.30, "2020-01-01T00:00:00+00:00")
    assert select_imagery([coarse_new, fine_old]).source == "naip"

    near_old = _candidate("esri", 0.50, "2020-01-01T00:00:00+00:00")
    near_new = _candidate("naip", 0.56, "2024-06-01T00:00:00+00:00")
    assert select_imagery([near_old, near_new]).source == "naip"

    slightly_worse = _candidate("naip", 0.50 * 1.16, "2025-01-01T00:00:00+00:00")
    assert select_imagery([near_old, slightly_worse]).source == "esri"

    undated = _candidate("esri", 0.50, None)
    dated = _candidate("naip", 0.55, "2024-01-01T00:00:00+00:00")
    assert select_imagery([undated, dated]).source == "naip"
    with pytest.raises(ValueError):
        select_imagery([])


def test_geocode_and_provider_metadata_parsing():
    latitude, longitude, label = parse_geocode([{"lat": "40.1", "lon": "-74.2", "display_name": "Home"}])
    assert (latitude, longitude, label) == (40.1, -74.2, "Home")
    for payload in ([], {}, [{"lat": "nope"}], [{"lat": "99", "lon": "0"}]):
        with pytest.raises(ValueError):
            parse_geocode(payload)

    captured, resolution, credit = parse_esri_identify({
        "results": [
            {"attributes": {"SRC_DATE": "2020/01/01", "SRC_RES": "1.0", "NICE_NAME": "Old"}},
            {"attributes": {"SRC_DATE": "2024/05/02", "SRC_RES": "0.3", "NICE_NAME": "Maxar"}},
        ],
    }, 40.0)
    assert resolution == 0.3
    assert captured == datetime(2024, 5, 2, tzinfo=timezone.utc)
    assert credit == "Maxar"
    live_date, live_resolution, live_credit = parse_esri_identify({
        "results": [{
            "attributes": {
                "DATE (YYYYMMDD)": "20250201",
                "RESOLUTION (M)": "0.1524",
                "SRC_DATE2": "2/1/2025",
                "SOURCE": "NC Collaboratory at UNCCH",
            },
        }],
    }, 35.45)
    assert live_resolution == pytest.approx(0.1524)
    assert live_date == datetime(2025, 2, 1, tzinfo=timezone.utc)
    assert live_credit == "NC Collaboratory at UNCCH"
    assert parse_naip_query({"features": []}, 40.0) == (None, None, "")
    naip_date, naip_resolution, _credit = parse_naip_query({
        "features": [
            {"attributes": {"AcquisitionDate": 1577836800000, "LowPS": 0.6}},
            {"attributes": {"AcquisitionDate": "2024-06-01", "LowPS": 0.6}},
        ],
    }, 40.0)
    assert naip_resolution == 0.6
    assert naip_date is not None and naip_date.year == 2024
    usgs_date, usgs_resolution, _credit = parse_naip_query({
        "features": [{
            "attributes": {
                "acquisition_date": 1663200000000,
                "resolution_value": 0.6,
                "resolution_units": "METER",
                "Year": 2022,
            },
        }],
    }, 35.45)
    assert usgs_resolution == 0.6
    assert usgs_date is not None and usgs_date.year == 2022
    foot_date, foot_resolution, _credit = parse_naip_query({
        "features": [{"attributes": {"resolution_value": 2.0, "resolution_units": "FOOT", "Year": 2021}}],
    }, 35.45)
    assert foot_resolution == pytest.approx(0.6096)
    assert foot_date == datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_export_stays_coarser_than_the_source_sample():
    bbox = mercator_bbox(35.451111, -82.560556, 120, 120)
    pixels = export_pixels(bbox, 0.1524)
    span = bbox[2] - bbox[0]
    assert pixels < 1024
    assert span / pixels > 0.1524
    assert export_pixels(bbox, None) >= 64


def test_mercator_box_matches_ground_meters_at_the_equator():
    west, south, east, north = mercator_bbox(0, 0, 120, 80)
    assert east - west == pytest.approx(120)
    assert north - south == pytest.approx(80)
    high_west, _south, high_east, _north = mercator_bbox(40, -74, 120, 120)
    assert high_east - high_west > 120


def test_fov_lookup_accepts_exact_and_model_number_forms():
    exact = lookup_fov("reolink", "rlc-810a")
    assert exact["matched"] is True
    assert exact["hfov_deg"] == 87
    assert exact["match"] == "exact"
    numbered = lookup_fov("", "RLC810A")
    assert numbered["matched"] is True
    assert numbered["model"] == "RLC-810A"
    assert numbered["match"] == "model"
    missing = lookup_fov("Reolink", "not-a-camera")
    assert missing == {"matched": False, "reason": "no catalog entry"}
    assert lookup_fov("Reolink", "")["reason"] == "model is required"


def test_property_config_validates_and_does_not_restart_workers():
    with pytest.raises(ValidationError):
        PropertyCameraPlacement(camera_id="gate", hfov_deg=0, x=0, y=0, heading_deg=10)
    with pytest.raises(ValidationError):
        PropertyViewConfig(cameras=[
            {"camera_id": "gate", "hfov_deg": 80, "x": 0.2, "y": 0.2, "heading_deg": 0},
            {"camera_id": "gate", "hfov_deg": 80, "x": 0.4, "y": 0.4, "heading_deg": 0},
        ])
    placement = PropertyCameraPlacement(
        camera_id="gate", hfov_deg=87, x=0.5, y=0.5, heading_deg=360, range_m=20,
    )
    assert placement.heading_deg == 0

    before = AppConfig()
    after = before.model_copy(deep=True)
    after.property = PropertyViewConfig(cameras=[placement])
    assert manager_owned_config(before) == manager_owned_config(after)
    assert hot_config_changes(before, after) == ["property"]
    runtime = Mock()
    runtime.config = before
    _effective, result = TargetedConfigApplication(
        lock=threading.RLock(),
        save=Mock(),
        active_exports=lambda: [],
    ).apply(before, after, runtime, persist=True)
    assert result["camera_workers_restarted"] is False
    assert result["apply_mode"] == "hot"
    assert result["hot_updated"] == ["property"]
    assert result["subsystems_restarted"] == []


def test_download_picks_the_ranked_jpeg_and_skips_geocoding_for_coordinates(tmp_path: Path):
    jpeg = b"\xff\xd8\xff" + b"image-bytes"
    calls = []

    class Client:
        def get_json(self, url: str):
            calls.append(url)
            if "nominatim" in url:
                raise AssertionError("coordinates should skip geocoding")
            if "identify" in url:
                return {"results": [{"attributes": {"SRC_DATE": "2024/01/01", "SRC_RES": "0.5"}}]}
            if "ImageServer/query" in url:
                return {"features": [{"attributes": {"AcquisitionDate": "2023-01-01", "LowPS": 0.3}}]}
            raise AssertionError(url)

        def get_bytes(self, url: str) -> bytes:
            calls.append(url)
            if "ImageServer/exportImage" in url:
                return b"not-a-jpeg"
            return jpeg

    fetched = fetch_property_image(
        address="kept as label",
        latitude=40.0,
        longitude=-74.0,
        width_m=120,
        height_m=120,
        client=Client(),
        now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc),
    )
    assert fetched["source"] == "esri"
    assert fetched["image_bytes"] == jpeg
    assert fetched["candidates"][0]["source"] == "naip"
    assert fetched["candidates"][0]["selected"] is False
    assert any(item["selected"] and item["source"] == "esri" for item in fetched["candidates"])
    assert store_property_image(tmp_path, jpeg) is None
    assert (tmp_path / "property" / "aerial.jpg").read_bytes() == jpeg
    assert not any("nominatim" in url for url in calls)
    assert placement_needs_review(
        previous_latitude=40,
        previous_longitude=-74,
        previous_width_m=120,
        previous_height_m=120,
        has_placements=True,
        latitude=40.01,
        longitude=-74,
        width_m=120,
        height_m=120,
    )


def test_property_zone_accepts_points_and_legacy_rectangles():
    overlay = PropertyZoneOverlay.model_validate({
        "camera_id": "gate",
        "zone_name": "drive",
        "x": 0.5,
        "y": 0.5,
        "width": 0.2,
        "height": 0.1,
        "rotation_deg": 0,
    })
    assert [(point.x, point.y) for point in overlay.points] == [
        pytest.approx((0.4, 0.45)),
        pytest.approx((0.6, 0.45)),
        pytest.approx((0.6, 0.55)),
        pytest.approx((0.4, 0.55)),
    ]
    direct = PropertyZoneOverlay.model_validate({
        "camera_id": "gate",
        "zone_name": "yard",
        "points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.2}, {"x": 0.2, "y": 0.4}],
    })
    assert len(direct.points) == 3
    with pytest.raises(ValidationError):
        PropertyZoneOverlay.model_validate({
            "camera_id": "gate",
            "zone_name": "yard",
            "points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.2}],
        })


def test_property_routes_save_layout_without_reloading_workers(tmp_path: Path):
    config = AppConfig(storage_dir=str(tmp_path))
    jpeg = b"\xff\xd8\xff" + b"stored"
    store_property_image(tmp_path, jpeg)

    class Client:
        def get_json(self, url: str):
            if "nominatim" in url:
                return [{"lat": "41", "lon": "-72", "display_name": "House"}]
            if "identify" in url:
                return {"results": [{"attributes": {"SRC_DATE": "2024/08/01", "SRC_RES": "0.4", "NICE_NAME": "Maxar"}}]}
            return {"features": []}

        def get_bytes(self, url: str) -> bytes:
            return b"\xff\xd8\xff" + b"downloaded"

    def apply_config(next_config, assign_ids=False):
        assert assign_ids is False
        nonlocal config
        config = next_config
        return next_config, {
            "apply_mode": "hot",
            "camera_workers_restarted": False,
            "subsystems_restarted": [],
            "hot_updated": ["property"],
        }

    app = FastAPI()
    app.include_router(create_property_router(PropertyRouteDependencies(
        get_config=lambda: config,
        apply_config=apply_config,
        recent_object_events=lambda limit: [{
            "camera_id": "gate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "objects_json": '[{"label": "person", "zones": ["drive"]}]',
        }],
        lock=threading.RLock(),
        client=Client(),
    )))
    with TestClient(app) as client:
        denied = client.post("/api/property/imagery", json={})
        assert denied.status_code == 422
        saved = client.post("/api/property/imagery", json={"address": "1 Road"})
        assert saved.status_code == 200
        body = saved.json()
        assert body["camera_workers_restarted"] is False
        assert body["image"]["source"] == "esri"
        assert body["image"]["address"] == "House"
        assert client.get("/api/property/imagery").content.startswith(b"\xff\xd8\xff")
        layout = client.put("/api/property/layout", json={
            "cameras": [{
                "camera_id": "gate",
                "make": "Reolink",
                "model": "RLC-810A",
                "hfov_deg": 87,
                "x": 0.4,
                "y": 0.6,
                "heading_deg": 15,
                "range_m": 18,
            }],
            "zones": [{
                "camera_id": "gate",
                "zone_name": "drive",
                "x": 0.5,
                "y": 0.5,
                "width": 0.2,
                "height": 0.1,
                "rotation_deg": 12,
            }],
        })
        assert layout.status_code == 200
        assert layout.json()["camera_workers_restarted"] is False
        assert layout.json()["property"]["image"]["address"] == "House"
        saved_points = layout.json()["property"]["zones"][0]["points"]
        assert len(saved_points) == 4
        assert all(0 <= point["x"] <= 1 and 0 <= point["y"] <= 1 for point in saved_points)
        assert client.get("/api/property/fov", params={"model": "UVC-G5-Bullet"}).json()["hfov_deg"] == 84.4
        activity = client.get("/api/property/activity").json()
        assert activity["events"][0]["zones"] == ["drive"]
        assert activity["events"][0]["classes"] == ["person"]

    assert required_api_scope("GET", "/api/property/imagery") == "read"
    assert required_api_scope("GET", "/api/property/activity") == "read"
    assert required_api_scope("POST", "/api/property/imagery") == "admin"
    assert required_api_scope("PUT", "/api/property/layout") == "admin"
    assert rank_imagery([]) == []
