from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from survng.app.config import AppConfig, WeatherConfig
from survng.app.config_application import hot_config_changes, manager_owned_config
from survng.app.security import required_api_scope
from survng.app.weather import ProviderCache, conditions, create_weather_router, radar


def test_weather_defaults_and_location_validation():
    assert not AppConfig().weather.enabled
    for values in ({"enabled": True}, {"latitude": 91}, {"longitude": float("nan")}, {"radar_zoom": 8}):
        with pytest.raises(ValidationError):
            WeatherConfig(**values)
    assert WeatherConfig(enabled=True, latitude=0, longitude=0).enabled


def test_weather_changes_do_not_restart_camera_manager():
    before = AppConfig()
    after = before.model_copy(deep=True)
    after.weather = WeatherConfig(enabled=True, latitude=40, longitude=-74)
    assert manager_owned_config(before) == manager_owned_config(after)
    assert hot_config_changes(before, after) == ["weather"]


def test_cache_failure_backoff_recovery_and_location_isolation():
    now = [1000]
    cache = ProviderCache(600, lambda: now[0])
    calls = []

    def good():
        calls.append("good")
        return {"temperature": 72}

    def fail():
        calls.append("fail")
        raise OSError("private upstream details")

    assert cache.get(("conditions", 1), good)["data"]["temperature"] == 72
    assert not cache.get(("conditions", 1), fail)["stale"]
    now[0] += 600
    stale = cache.get(("conditions", 1), fail)
    assert stale["stale"] and stale["data"]["temperature"] == 72
    assert "private" not in str(stale)
    cache.get(("conditions", 1), fail)
    assert calls == ["good", "fail"]
    now[0] += 30
    assert not cache.get(("conditions", 1), good)["stale"]
    assert cache.get(("conditions", 2), fail)["data"] is None


def test_concurrent_viewers_share_one_provider_request():
    cache = ProviderCache(600)
    calls = []

    def load():
        calls.append(1)
        return {"temperature": 70}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cache.get(("conditions",), load), range(20)))
    assert len(calls) == 1
    assert all(result["data"]["temperature"] == 70 for result in results)


def test_provider_payload_validation():
    result = conditions({"current": {"time": 1000, "temperature_2m": 0, "wind_speed_10m": float("nan")}, "hourly": {"time": [900, 1100], "temperature_2m": [2, 3]}})
    assert result["temperature_2m"] == 0
    assert result["wind_speed_10m"] is None
    assert result["hourly"] == [{"time": 1100, "temperature_2m": 3, "precipitation_probability": None}]
    with pytest.raises(ValueError):
        conditions({"current": {"time": 1000}})
    assert radar({"host": "https://tilecache.rainviewer.com", "radar": {"past": [{"time": 1000, "path": "/v2/radar/1c3e0f1703e6"}]}})["frames"][0]["path"] == "/v2/radar/1c3e0f1703e6"
    for host, path in (("http://localhost", "/v2/radar/1000"), ("https://tilecache.rainviewer.com", "/../../private")):
        with pytest.raises(ValueError):
            radar({"host": host, "radar": {"past": [{"time": 1000, "path": path}]}})


def test_routes_disabled_units_and_independent_failure():
    config = WeatherConfig()
    urls = []

    def fetch(url):
        urls.append(url)
        if "rainviewer" in url:
            raise OSError("offline")
        return {"current": {"time": 1000, "temperature_2m": 23}}

    app = FastAPI()
    app.include_router(create_weather_router(lambda: config, fetch))
    with TestClient(app) as client:
        assert client.get("/api/weather/conditions").status_code == 404
        assert not urls
        config = WeatherConfig(enabled=True, latitude=0, longitude=0, units="metric")
        assert client.get("/api/weather/conditions").json()["data"]["temperature_2m"] == 23
        query = parse_qs(urlparse(urls[0]).query)
        assert query["temperature_unit"] == ["celsius"]
        assert query["wind_speed_unit"] == ["kmh"]
        assert client.get("/api/weather/radar").json()["unavailable"]
        assert not client.get("/api/weather/conditions").json()["stale"]
        config = config.model_copy(update={"units": "imperial"})
        client.get("/api/weather/conditions")
        assert parse_qs(urlparse(urls[-1]).query)["temperature_unit"] == ["fahrenheit"]
    assert required_api_scope("GET", "/api/weather/radar") == "read"
