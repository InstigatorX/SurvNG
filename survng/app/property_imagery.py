"""Search public aerial imagery and keep the sharpest recent photo of one property.

Nominatim geocodes addresses. Esri World Imagery is global. USGS NAIP is used
when that service has coverage. Callers inject HTTP so tests never reach those
hosts. Downloads are limited to the three hosts this module builds itself.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException as HTTPClientError
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

USER_AGENT = "SurvNG/1.2 (local property imagery)"
RELATIVE_IMAGE = "property/aerial.jpg"
EXPORT_PIXELS = 1024
JSON_LIMIT = 512_000
IMAGE_LIMIT = 6_000_000
_MERCATOR_HALF = 20037508.342789244
_QUALITY_BAND = 1.15
_ALLOWED_HOSTS = frozenset({
    "nominatim.openstreetmap.org",
    "services.arcgisonline.com",
    "imagery.nationalmap.gov",
})
_ESRI_IDENTIFY = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/identify"
_ESRI_EXPORT = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
_NAIP_QUERY = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/query"
_NAIP_EXPORT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
_ESRI_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics, and the GIS User Community"
_NAIP_ATTRIBUTION = "USGS National Agriculture Imagery Program"
_DATE_FIELDS = (
    "DATE (YYYYMMDD)",
    "SRC_DATE",
    "Date",
    "SOURCE_DATE",
    "AcquisitionDate",
    "acquisition_date",
    "SRC_DATE2",
)
_RESOLUTION_FIELDS = (
    "RESOLUTION (M)",
    "SRC_RES",
    "Resolution",
    "RESOLUTION",
    "resolution_value",
    "LowPS",
    "HighPS",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True, slots=True)
class ImageryCandidate:
    source: str
    captured_at: datetime | None
    resolution_m: float
    attribution: str
    download_url: str


def parse_geocode(payload: object) -> tuple[float, float, str]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("address was not found")
    item = payload[0]
    if not isinstance(item, dict):
        raise ValueError("address was not found")
    try:
        latitude = float(item["lat"])
        longitude = float(item["lon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("address was not found") from error
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("address was not found")
    if not -85 <= latitude <= 85 or not -180 <= longitude <= 180:
        raise ValueError("address was not found")
    address = str(item.get("display_name") or "").strip()
    return latitude, longitude, address[:300]


def parse_capture_date(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        seconds = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if "T" in text or text.endswith("Z"):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        sample = text[:8] if fmt == "%Y%m%d" else text[:10]
        try:
            return datetime.strptime(sample, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_resolution_m(value: object, latitude: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0 or number > 50:
        return None
    if number < 0.01:
        return number * 111_320 * max(0.2, math.cos(math.radians(latitude)))
    return number


def select_imagery(candidates: list[ImageryCandidate]) -> ImageryCandidate:
    """Prefer the finest ground resolution. Inside a 15% band, prefer the newer photo."""
    ranked = rank_imagery(candidates)
    if not ranked:
        raise ValueError("no aerial imagery was available")
    return ranked[0]


def rank_imagery(candidates: list[ImageryCandidate]) -> list[ImageryCandidate]:
    remaining = [
        item for item in candidates
        if math.isfinite(item.resolution_m) and item.resolution_m > 0
    ]
    ordered: list[ImageryCandidate] = []
    while remaining:
        best = min(item.resolution_m for item in remaining)
        band = [item for item in remaining if item.resolution_m <= best * _QUALITY_BAND]
        chosen = min(band, key=_rank_key)
        ordered.append(chosen)
        remaining = [item for item in remaining if item is not chosen]
    return ordered


def _rank_key(item: ImageryCandidate) -> tuple[float, float, str]:
    captured = item.captured_at.timestamp() if item.captured_at is not None else float("-inf")
    return (-captured, item.resolution_m, item.source)


def web_mercator(latitude: float, longitude: float) -> tuple[float, float]:
    clamped = min(89.5, max(-89.5, latitude))
    x = longitude * _MERCATOR_HALF / 180.0
    y = math.log(math.tan(math.radians(90 + clamped) / 2.0)) * _MERCATOR_HALF / math.pi
    return x, y


def export_pixels(bbox: tuple[float, float, float, float], resolution_m: float | None) -> int:
    """Pixel count Esri will actually render for this box.

    World Imagery returns an empty image, and then HTTP 500 for ``f=image``,
    when the request is finer than the source sample distance. A 120 m box at
    1024 px is about 0.14 m/px, which is finer than the 0.15 m imagery covering
    much of the US.
    """
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
    if resolution_m is not None and 0.05 <= resolution_m <= 5:
        ground = resolution_m
    else:
        ground = 0.30
    pixels = int(span / (ground * 1.08))
    return max(64, min(EXPORT_PIXELS, pixels))


def mercator_bbox(
    latitude: float,
    longitude: float,
    width_m: float,
    height_m: float,
) -> tuple[float, float, float, float]:
    """North-up box whose ground size is width_m by height_m at this latitude."""
    scale = max(0.2, math.cos(math.radians(latitude)))
    x, y = web_mercator(latitude, longitude)
    half_x = (width_m / scale) / 2
    half_y = (height_m / scale) / 2
    return (x - half_x, y - half_y, x + half_x, y + half_y)


def parse_esri_identify(payload: object, latitude: float) -> tuple[datetime | None, float | None, str]:
    if not isinstance(payload, dict):
        return None, None, ""
    results = payload.get("results")
    if not isinstance(results, list):
        return None, None, ""
    best_date: datetime | None = None
    best_resolution: float | None = None
    credit = ""
    for result in results:
        if not isinstance(result, dict):
            continue
        attributes = result.get("attributes")
        if not isinstance(attributes, dict):
            continue
        captured = _first_date(attributes)
        resolution = _first_resolution(attributes, latitude)
        name = _text_attribute(attributes, ("NICE_NAME", "Source", "SOURCE", "SOURCE_INFO", "DATA_SRC"))
        if resolution is not None and (best_resolution is None or resolution < best_resolution):
            best_resolution = resolution
            if captured is not None:
                best_date = captured
            if name:
                credit = name
        elif best_date is None and captured is not None:
            best_date = captured
            if name and not credit:
                credit = name
    return best_date, best_resolution, credit


def parse_naip_query(payload: object, latitude: float) -> tuple[datetime | None, float | None, str]:
    if not isinstance(payload, dict):
        return None, None, ""
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return None, None, ""
    chosen_date: datetime | None = None
    chosen_resolution: float | None = None
    for feature in features:
        if not isinstance(feature, dict):
            continue
        attributes = feature.get("attributes")
        if not isinstance(attributes, dict):
            continue
        attributes = _naip_attributes(attributes)
        captured = _first_date(attributes)
        resolution = _first_resolution(attributes, latitude)
        if resolution is None:
            continue
        candidate = ImageryCandidate("naip", captured, resolution, "", "")
        current = None if chosen_resolution is None else ImageryCandidate("naip", chosen_date, chosen_resolution, "", "")
        if current is None or select_imagery([current, candidate]) is candidate:
            chosen_date = captured
            chosen_resolution = resolution
    return chosen_date, chosen_resolution, ""


def detection_summary(objects_json: str) -> tuple[list[str], list[str]]:
    try:
        objects = json.loads(objects_json or "[]")
    except json.JSONDecodeError:
        return [], []
    if not isinstance(objects, list):
        return [], []
    zones: set[str] = set()
    classes: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or not item.get("label"):
            continue
        classes.add(str(item["label"]))
        raw_zones = item.get("zones")
        if not isinstance(raw_zones, list):
            continue
        for zone in raw_zones:
            if zone:
                zones.add(str(zone))
    return sorted(zones), sorted(classes)


def property_image_file(storage_dir: Path) -> Path:
    root = Path(storage_dir).resolve()
    path = (root / RELATIVE_IMAGE).resolve()
    path.relative_to(root)
    return path


def store_property_image(storage_dir: Path, data: bytes) -> bytes | None:
    """Install a JPEG and return the previous bytes so a failed config save can restore it."""
    if not data.startswith(b"\xff\xd8\xff"):
        raise ValueError("aerial image was not a JPEG")
    path = property_image_file(storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.is_file() else None
    temporary = path.with_name("aerial.jpg.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return previous


def restore_property_image(storage_dir: Path, previous: bytes | None) -> None:
    path = property_image_file(storage_dir)
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name("aerial.jpg.tmp")
    temporary.write_bytes(previous)
    os.replace(temporary, path)


def placement_needs_review(
    *,
    previous_latitude: float | None,
    previous_longitude: float | None,
    previous_width_m: float | None,
    previous_height_m: float | None,
    has_placements: bool,
    latitude: float,
    longitude: float,
    width_m: float,
    height_m: float,
) -> bool:
    if not has_placements or previous_latitude is None or previous_longitude is None:
        return False
    north = (latitude - previous_latitude) * 111_320
    east = (longitude - previous_longitude) * 111_320 * math.cos(math.radians(latitude))
    moved = math.hypot(north, east) > 1
    resized = (
        previous_width_m is None
        or previous_height_m is None
        or abs(width_m - previous_width_m) > 0.5
        or abs(height_m - previous_height_m) > 0.5
    )
    return moved or resized


class ImageryHttp:
    def get_json(self, url: str) -> object:
        payload = _read(url, JSON_LIMIT, "application/json")
        result = json.loads(payload)
        if not isinstance(result, (dict, list)):
            raise ValueError("invalid imagery response")
        return result

    def get_bytes(self, url: str) -> bytes:
        return _read(url, IMAGE_LIMIT, "image/jpeg,image/*;q=0.8")


def fetch_property_image(
    *,
    address: str,
    latitude: float | None,
    longitude: float | None,
    width_m: float,
    height_m: float,
    client: ImageryHttp | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict:
    http = client or ImageryHttp()
    label = address.strip()
    if latitude is None or longitude is None:
        if not label:
            raise ValueError("address or latitude and longitude is required")
        found_lat, found_lon, display = parse_geocode(http.get_json(_geocode_url(label)))
        latitude, longitude = found_lat, found_lon
        label = display or label
    bbox = mercator_bbox(latitude, longitude, width_m, height_m)
    candidates = _candidates(http, latitude, longitude, bbox)
    errors: list[str] = []
    image: bytes | None = None
    chosen: ImageryCandidate | None = None
    for candidate in rank_imagery(candidates):
        try:
            data = http.get_bytes(candidate.download_url)
        except (OSError, HTTPClientError, ValueError, TimeoutError):
            errors.append(candidate.source)
            continue
        if not data.startswith(b"\xff\xd8\xff"):
            errors.append(candidate.source)
            continue
        image = data
        chosen = candidate
        break
    if image is None or chosen is None:
        raise ValueError("no aerial imagery was available")
    fetched_at = (now or (lambda: datetime.now(timezone.utc)))()
    return {
        "latitude": latitude,
        "longitude": longitude,
        "address": label[:300],
        "width_m": width_m,
        "height_m": height_m,
        "image_path": RELATIVE_IMAGE,
        "source": chosen.source,
        "attribution": chosen.attribution,
        "captured_at": chosen.captured_at.isoformat() if chosen.captured_at else "",
        "resolution_m": chosen.resolution_m,
        "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
        "image_bytes": image,
        "candidates": [
            {
                "source": item.source,
                "captured_at": item.captured_at.isoformat() if item.captured_at else "",
                "resolution_m": item.resolution_m,
                "selected": item is chosen,
            }
            for item in rank_imagery(candidates)
        ],
        "rejected_sources": errors,
    }


def _candidates(client: ImageryHttp, latitude: float, longitude: float, bbox: tuple[float, float, float, float]) -> list[ImageryCandidate]:
    found: list[ImageryCandidate] = []
    esri_payload = _optional_json(client, _esri_identify_url(latitude, longitude))
    if esri_payload is not None:
        captured, resolution, credit = parse_esri_identify(esri_payload, latitude)
        # Without a source sample distance, treat the worldwide low-resolution
        # tier as the quality. An export pixel is not the source resolution.
        if resolution is None:
            resolution = 15.0
        attribution = _ESRI_ATTRIBUTION if not credit else f"{credit}. {_ESRI_ATTRIBUTION}"
        found.append(ImageryCandidate(
            "esri",
            captured,
            resolution,
            attribution[:300],
            _export_url(_ESRI_EXPORT, bbox, None if resolution >= 5 else resolution),
        ))
    naip_payload = _optional_json(client, _naip_query_url(longitude, latitude))
    if naip_payload is not None:
        captured, resolution, _credit = parse_naip_query(naip_payload, latitude)
        if resolution is not None:
            found.append(ImageryCandidate(
                "naip",
                captured,
                resolution,
                _NAIP_ATTRIBUTION,
                _export_url(_NAIP_EXPORT, bbox, resolution),
            ))
    return found


def _optional_json(client: ImageryHttp, url: str) -> object | None:
    try:
        return client.get_json(url)
    except (OSError, HTTPClientError, ValueError, TimeoutError, json.JSONDecodeError):
        return None


def _geocode_url(address: str) -> str:
    return "https://nominatim.openstreetmap.org/search?" + urlencode({
        "q": address,
        "format": "jsonv2",
        "limit": "1",
    })


def _esri_identify_url(latitude: float, longitude: float) -> str:
    span = 0.002
    return _ESRI_IDENTIFY + "?" + urlencode({
        "geometry": json.dumps({"x": longitude, "y": latitude, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "all",
        "tolerance": "1",
        "mapExtent": f"{longitude - span},{latitude - span},{longitude + span},{latitude + span}",
        "imageDisplay": "400,400,96",
        "returnGeometry": "false",
        "f": "json",
    })


def _naip_query_url(longitude: float, latitude: float) -> str:
    return _NAIP_QUERY + "?" + urlencode({
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "acquisition_date,resolution_value,resolution_units,Year,Name",
        "returnGeometry": "false",
        "f": "json",
    })


def _export_url(base: str, bbox: tuple[float, float, float, float], resolution_m: float | None) -> str:
    pixels = export_pixels(bbox, resolution_m)
    return base + "?" + urlencode({
        "bbox": ",".join(f"{value:.3f}" for value in bbox),
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{pixels},{pixels}",
        "format": "jpg",
        "f": "image",
    })


def _naip_attributes(attributes: dict) -> dict:
    normalized = dict(attributes)
    units = str(attributes.get("resolution_units") or "").strip().upper()
    value = attributes.get("resolution_value")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and units in {"FOOT", "FEET", "FT", "USFT"}:
        normalized["resolution_value"] = float(value) * 0.3048
    if attributes.get("acquisition_date") in (None, "") and attributes.get("Year") not in (None, ""):
        normalized["acquisition_date"] = f"{attributes['Year']}-01-01"
    return normalized


def _read(url: str, limit: int, accept: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError("imagery host is not allowed")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with build_opener(_NoRedirect()).open(request, timeout=20) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("imagery response exceeds limit")
    return payload


def _first_date(attributes: dict) -> datetime | None:
    for field in _DATE_FIELDS:
        if field in attributes:
            captured = parse_capture_date(attributes.get(field))
            if captured is not None:
                return captured
    return None


def _first_resolution(attributes: dict, latitude: float) -> float | None:
    for field in _RESOLUTION_FIELDS:
        if field in attributes:
            resolution = parse_resolution_m(attributes.get(field), latitude)
            if resolution is not None:
                return resolution
    return None


def _text_attribute(attributes: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = attributes.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    return ""
