from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.config import AppConfig, CameraConfig, DetectionZone
from survng.app.motion_pipeline import build_builtin_motion_registry
from survng.app.system_routes import SystemRouteDependencies, create_system_router


def _route_dependencies(**overrides):
    registry = overrides.pop("motion_pipeline_registry", build_builtin_motion_registry())
    values = dict(
        get_manager=lambda: SimpleNamespace(statuses=lambda: []),
        get_config=lambda: AppConfig(),
        system_telemetry=Mock(),
        ffprobe_path=lambda: "ffprobe",
        ffplay_path=lambda: "ffplay",
        ffmpeg_qsv_info=lambda: {},
        ffmpeg_vaapi_info=lambda: {},
        hardware_acceleration_mode=lambda: "off",
        event_clip_window=lambda _before, _after: (5, 5),
        recording_cache_status=lambda: {},
        model_evaluation=Mock(),
        motion_pipeline_registry=registry,
    )
    values.update(overrides)
    return SystemRouteDependencies(**values)


def test_home_assistant_metadata_is_bounded_and_credential_free() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate", name="Gate", stream_url="rtsp://user:secret@example/gate",
        zones=[DetectionZone(name="Driveway", object_classes=["car"])],
    )])
    config.mqtt.enabled = True
    config.mqtt.discovery_enabled = True
    dependencies = _route_dependencies(get_config=lambda: config)
    payload = create_system_router(dependencies).handlers["home_assistant_metadata"]()

    assert payload["incident_notifications"] == {"enabled": True, "schema_version": 2, "transport": "sse", "mqtt_required": False}
    assert payload["schema_version"] == 1
    assert payload["cameras"] == [{
        "id": "gate", "name": "Gate",
        "zones": [{"name": "Driveway", "object_classes": ["car"], "notifications_enabled": True}],
    }]
    assert "secret" not in str(payload)
    assert "stream" not in str(payload["cameras"])


def test_motion_catalog_route_uses_the_process_registry() -> None:
    registry = build_builtin_motion_registry()
    get_manager = Mock(side_effect=AssertionError("catalog must not touch the manager"))
    dependencies = _route_dependencies(
        get_manager=get_manager,
        motion_pipeline_registry=registry,
    )

    payload = create_system_router(dependencies).handlers["get_motion_pipeline_catalog"]()

    assert payload["schema_version"] == 1
    assert payload["stages"]
    assert payload["presets"]
    get_manager.assert_not_called()
