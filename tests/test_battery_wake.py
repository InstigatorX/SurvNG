"""Battery wake sessions stay idle until an external wake and do not reconnect."""

from __future__ import annotations

from pathlib import Path

import pytest

from survng.app.battery_wake import BatteryWakeService
from survng.app.config import AppConfig, CameraConfig, CameraSpeechConfig
from survng.app.speech.lifecycle import SpeechLifecycle


def _camera() -> CameraConfig:
    return CameraConfig(
        id="porch",
        name="Porch",
        stream_url="rtsp://camera/main",
        power="battery",
        wake_max_seconds=30,
        wake_idle_seconds=5,
        wake_open_timeout_seconds=8,
    )


def _service(
    camera: CameraConfig,
    *,
    clock: dict[str, float],
    frames: dict[str, object],
    opens: list[str],
    releases: list[str],
    recordings: list[tuple[str, bool]],
    notices: list[str],
) -> BatteryWakeService:
    def open_capture(camera_id: str) -> bool:
        opens.append(camera_id)
        frames["running"] = True
        return bool(frames.get("open", True))

    def release_capture(camera_id: str) -> None:
        releases.append(camera_id)
        frames["running"] = False

    def set_released(camera_id: str, released: bool) -> None:
        recordings.append((camera_id, released))

    def start_recording(started: CameraConfig) -> None:
        recordings.append((started.id, "start"))

    return BatteryWakeService(
        cameras={camera.id: camera},
        open_capture=open_capture,
        release_capture=release_capture,
        capture_running=lambda _camera_id: bool(frames["running"]),
        frame_monotonic=lambda _camera_id: frames.get("frame"),  # type: ignore[return-value]
        notify_motion=notices.append,
        set_recording_released=set_released,
        start_recording=start_recording,
        monotonic=lambda: clock["now"],
    )


def test_wake_route_requires_camera_control() -> None:
    from survng.app.security import required_api_scope

    assert required_api_scope("POST", "/api/cameras/porch/wake") == "camera:control"


def test_mains_cameras_keep_default_wake_limits() -> None:
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")

    assert camera.power == "mains"
    assert camera.battery_powered() is False
    assert camera.wake_max_seconds == 60
    assert camera.wake_idle_seconds == 4
    assert camera.wake_open_timeout_seconds == 8


def test_wake_limits_must_fit_inside_one_session() -> None:
    with pytest.raises(ValueError, match="wake idle"):
        CameraConfig(
            id="porch",
            name="Porch",
            stream_url="rtsp://camera/main",
            power="battery",
            wake_max_seconds=10,
            wake_idle_seconds=10,
        )


def test_wake_starts_capture_and_recording_once_then_sleeps_without_reconnect() -> None:
    camera = _camera()
    clock = {"now": 0.0}
    frames: dict[str, object] = {"running": False, "frame": None, "open": True}
    opens: list[str] = []
    releases: list[str] = []
    recordings: list[tuple[str, bool]] = []
    notices: list[str] = []
    service = _service(
        camera,
        clock=clock,
        frames=frames,
        opens=opens,
        releases=releases,
        recordings=recordings,
        notices=notices,
    )

    assert service.arm(camera.id)
    assert service.status(camera.id)["phase"] == "connecting"
    assert opens == [camera.id]
    assert recordings == [(camera.id, True), (camera.id, "start")]

    frames["frame"] = 1.0
    clock["now"] = 1.0
    service._poll()
    assert service.status(camera.id)["phase"] == "awake"
    assert notices == [camera.id]

    assert service.arm(camera.id)
    assert opens == [camera.id]
    assert recordings == [(camera.id, True), (camera.id, "start")]
    assert service.status(camera.id)["phase"] == "awake"

    clock["now"] = 7.0
    service._poll()
    assert service.status(camera.id)["phase"] == "asleep"
    assert releases == [camera.id]
    assert recordings[-1] == (camera.id, False)
    assert opens == [camera.id]


def test_stream_end_returns_to_sleep_without_reopening() -> None:
    camera = _camera()
    clock = {"now": 0.0}
    frames: dict[str, object] = {"running": False, "frame": None, "open": True}
    opens: list[str] = []
    releases: list[str] = []
    service = _service(
        camera,
        clock=clock,
        frames=frames,
        opens=opens,
        releases=releases,
        recordings=[],
        notices=[],
    )

    assert service.arm(camera.id)
    frames["frame"] = 1.0
    clock["now"] = 1.0
    service._poll()
    frames["running"] = False
    clock["now"] = 1.5
    service._poll()

    assert service.status(camera.id)["phase"] == "asleep"
    assert releases == [camera.id]
    assert opens == [camera.id]


def test_wake_that_never_opens_is_a_failure() -> None:
    camera = _camera()
    clock = {"now": 0.0}
    frames: dict[str, object] = {"running": False, "frame": None, "open": True}
    service = _service(
        camera,
        clock=clock,
        frames=frames,
        opens=[],
        releases=[],
        recordings=[],
        notices=[],
    )

    assert service.arm(camera.id)
    frames["running"] = False
    clock["now"] = 9.0
    service._poll()

    assert service.status(camera.id)["phase"] == "wake_failed"
    assert service.status(camera.id)["saw_frame"] is False


def test_speech_listening_skips_battery_cameras(tmp_path: Path) -> None:
    battery = CameraConfig(
        id="porch",
        name="Porch",
        stream_url="rtsp://camera/porch",
        power="battery",
        speech=CameraSpeechConfig(enabled=True),
    )
    mains = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://camera/gate",
        speech=CameraSpeechConfig(enabled=True),
    )
    service = SpeechLifecycle(
        AppConfig(cameras=[battery, mains]),
        database_dir=tmp_path,
        camera_enabled=lambda _camera_id: True,
        audio_available=lambda _camera: True,
        publish=lambda *_args, **_kwargs: None,
    )

    assert list(service.cameras) == ["gate"]
