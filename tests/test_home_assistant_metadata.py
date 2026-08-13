from types import SimpleNamespace
from unittest.mock import Mock

from survng.app.config import AppConfig, CameraConfig, DetectionZone
from survng.app.system_routes import SystemRouteDependencies, create_system_router


def test_home_assistant_metadata_is_bounded_and_credential_free() -> None:
    config = AppConfig(cameras=[CameraConfig(
        id="gate", name="Gate", stream_url="rtsp://user:secret@example/gate",
        zones=[DetectionZone(name="Driveway", object_classes=["car"])],
    )])
    config.mqtt.enabled = True
    config.mqtt.discovery_enabled = True
    dependencies = SystemRouteDependencies(
        get_manager=lambda: SimpleNamespace(statuses=lambda: []),
        get_config=lambda: config,
        system_telemetry=Mock(), ffprobe_path=lambda: "ffprobe",
        ffplay_path=lambda: "ffplay", ffmpeg_qsv_info=lambda: {},
        ffmpeg_vaapi_info=lambda: {}, hardware_acceleration_mode=lambda: "off",
        event_clip_window=lambda _before, _after: (5, 5),
        recording_cache_status=lambda: {},
        model_evaluation=Mock(),
    )
    payload = create_system_router(dependencies).handlers["home_assistant_metadata"]()

    assert payload["schema_version"] == 1
    assert payload["cameras"] == [{
        "id": "gate", "name": "Gate",
        "zones": [{"name": "Driveway", "object_classes": ["car"]}],
    }]
    assert "secret" not in str(payload)
    assert "stream" not in str(payload["cameras"])
