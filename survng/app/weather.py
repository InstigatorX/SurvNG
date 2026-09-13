"""Read-only weather providers, shared by all dashboard clients.

No camera worker or background task is allocated. Sync routes run in FastAPI's
thread pool; each provider has a separate lock and bounded last-good cache.
"""

from __future__ import annotations

import json
from http.client import HTTPException as HTTPClientError
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from fastapi import APIRouter, HTTPException

from .config import WeatherConfig

LOGGER = logging.getLogger(__name__)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "SurvNG weather/1.0", "Accept": "application/json"})
    with build_opener(_NoRedirect()).open(request, timeout=5) as response:
        payload = response.read(262145)
    if len(payload) > 262144:
        raise ValueError("weather response exceeds limit")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("invalid weather response")
    return result


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def conditions(payload: dict) -> dict:
    current = payload.get("current") or {}
    timestamp = _number(current.get("time"))
    if not timestamp or _number(current.get("temperature_2m")) is None:
        raise ValueError("missing current conditions")
    fields = ("temperature_2m", "apparent_temperature", "relative_humidity_2m", "weather_code", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "is_day")
    hourly = payload.get("hourly") or {}
    hours = []
    for index, hour in enumerate((hourly.get("time") or [])[:72]):
        if _number(hour) is None or hour < timestamp:
            continue
        row = {"time": hour}
        for field in ("temperature_2m", "precipitation_probability"):
            values = hourly.get(field) or []
            row[field] = _number(values[index]) if index < len(values) else None
        hours.append(row)
    return {"time": timestamp, **{field: _number(current.get(field)) for field in fields}, "hourly": hours[:12]}


def radar(payload: dict) -> dict:
    if payload.get("host") != "https://tilecache.rainviewer.com":
        raise ValueError("unexpected radar tile host")
    frames = []
    for frame in (payload.get("radar", {}).get("past") or [])[-13:]:
        if not isinstance(frame, dict):
            raise ValueError("invalid radar frame")
        timestamp = _number(frame.get("time"))
        path = frame.get("path")
        if not timestamp or not isinstance(path, str) or not re.fullmatch(r"/v2/radar/[a-fA-F0-9]{1,64}", path):
            raise ValueError("invalid radar frame")
        frames.append({"time": timestamp, "path": path})
    if not frames:
        raise ValueError("no radar frames available")
    return {"frames": sorted(frames, key=lambda frame: frame["time"])}


class ProviderCache:
    def __init__(self, ttl: int, clock: Callable = time.time):
        self.ttl = ttl
        self.clock = clock
        self.lock = threading.Lock()
        self.key = None
        self.data = None
        self.updated_at = None
        self.retry_at = 0
        self.failures = 0

    def get(self, key, load: Callable) -> dict:
        with self.lock:
            now = self.clock()
            if self.key != key:
                self.key, self.data, self.updated_at = key, None, None
                self.retry_at, self.failures = 0, 0
            if now >= self.retry_at:
                try:
                    data = load()
                except (OSError, HTTPClientError, ValueError, TypeError, KeyError, AttributeError) as error:
                    self.failures += 1
                    self.retry_at = now + min(900, 30 * 2 ** min(self.failures - 1, 5))
                    # One diagnostic per upstream attempt, never raw provider errors/URLs.
                    LOGGER.warning("Weather provider %s unavailable (%s)", key[0], type(error).__name__)
                else:
                    self.data, self.updated_at = data, self.clock()
                    self.failures = 0
                    self.retry_at = self.updated_at + self.ttl
            return {"data": self.data, "updated_at": self.updated_at, "stale": bool(self.failures), "unavailable": self.data is None}


def create_weather_router(get_config: Callable[[], WeatherConfig], fetch: Callable = fetch_json) -> APIRouter:
    router = APIRouter(prefix="/api/weather", tags=["weather"])
    current_cache = ProviderCache(600)
    radar_cache = ProviderCache(300)

    def configured() -> WeatherConfig:
        config = get_config()
        if not config.enabled:
            raise HTTPException(status_code=404, detail="Weather is disabled")
        return config

    @router.get("/conditions")
    def get_conditions() -> dict:
        config = configured()
        query = urlencode({
            "latitude": config.latitude, "longitude": config.longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_gusts_10m,wind_direction_10m,is_day",
            "hourly": "temperature_2m,precipitation_probability", "forecast_days": 3,
            "temperature_unit": "fahrenheit" if config.units == "imperial" else "celsius",
            "wind_speed_unit": "mph" if config.units == "imperial" else "kmh",
            "timeformat": "unixtime", "timezone": "auto",
        })
        return current_cache.get(("conditions", config.latitude, config.longitude, config.units), lambda: conditions(fetch("https://api.open-meteo.com/v1/forecast?" + query)))

    @router.get("/radar")
    def get_radar() -> dict:
        configured()
        return radar_cache.get(("radar",), lambda: radar(fetch("https://api.rainviewer.com/public/weather-maps.json")))

    return router
