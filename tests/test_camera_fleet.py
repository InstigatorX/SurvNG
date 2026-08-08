from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from survng.app.camera_fleet import CameraFleetLifecycle, CameraFleetOperationError
from survng.app.config import CameraConfig


def _fleet(*, cameras: list[CameraConfig] | None = None):
    camera_list = cameras or [
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/main")
    ]
    workers = {camera.id: Mock() for camera in camera_list}
    for worker in workers.values():
        worker.live_capture_ready.return_value = True
        worker.wait_stopped.return_value = True
        worker.wait_onvif_stopped.return_value = True
        worker.active_workers.return_value = []
    recorder = Mock()
    startup = Mock()
    startup.cancel.return_value = True
    startup.status.return_value = {}
    publisher = Mock()
    fleet = CameraFleetLifecycle(
        cameras=camera_list,
        workers=workers,
        recorder=recorder,
        startup=startup,
        state_publisher=publisher,
    )
    return fleet, workers, recorder, startup, publisher


def test_prepared_generation_uses_an_immutable_preference_snapshot() -> None:
    fleet, workers, recorder, _startup, publisher = _fleet()
    camera_enabled = {"gate": True}
    recording_enabled = {"gate": True}
    detection_enabled = {"gate": False}

    tasks = fleet.prepare_startup(
        camera_enabled=camera_enabled,
        recording_enabled=recording_enabled,
        detection_enabled=detection_enabled,
    )
    camera_enabled["gate"] = False
    recording_enabled["gate"] = False
    detection_enabled["gate"] = True

    assert tasks[0].is_enabled()
    tasks[0].start_recorders()
    tasks[0].publish_state()
    workers["gate"].set_detection_enabled.assert_called_once_with(False)
    recorder.set_camera_enabled.assert_called_once_with("gate", True)
    recorder.start.assert_called_once_with(fleet.cameras[0], "main")
    publisher.publish_camera_state.assert_called_once_with("gate", True)


def test_runtime_power_change_is_visible_to_an_active_admission_task() -> None:
    fleet, _workers, _recorder, _startup, publisher = _fleet()
    task = fleet.prepare_startup(
        camera_enabled={"gate": True},
        recording_enabled={},
        detection_enabled={},
    )[0]

    assert fleet.set_camera_enabled("gate", False)

    assert not task.is_enabled()
    task.publish_state()
    publisher.publish_camera_state.assert_called_once_with("gate", False)


def test_cancelled_admission_prevents_late_recorder_start() -> None:
    fleet, _workers, recorder, startup, _publisher = _fleet()
    task = fleet.prepare_startup(
        camera_enabled={"gate": True},
        recording_enabled={"gate": True},
        detection_enabled={},
    )[0]

    fleet.cancel_admission()
    task.start_recorders()

    startup.cancel.assert_called_once_with()
    recorder.start.assert_not_called()


def test_late_recording_disable_prevents_queued_recorder_start() -> None:
    fleet, _workers, recorder, _startup, _publisher = _fleet()
    recording_enabled = {"gate": True}
    task = fleet.prepare_startup(
        camera_enabled={"gate": True},
        recording_enabled=recording_enabled,
        detection_enabled={},
        recording_is_enabled=lambda camera_id: recording_enabled[camera_id],
    )[0]

    recording_enabled["gate"] = False
    task.start_recorders()

    recorder.start.assert_not_called()


def test_live_capture_starts_when_recording_and_detection_are_disabled() -> None:
    fleet, workers, recorder, _startup, _publisher = _fleet()
    task = fleet.prepare_startup(
        camera_enabled={"gate": True},
        recording_enabled={"gate": False},
        detection_enabled={"gate": False},
    )[0]

    task.start_camera()
    task.start_recorders()

    workers["gate"].set_detection_enabled.assert_called_once_with(False)
    workers["gate"].start.assert_called_once_with()
    recorder.set_camera_enabled.assert_called_once_with("gate", False)
    recorder.start.assert_not_called()


def test_shutdown_attempts_and_closes_every_non_timed_out_camera() -> None:
    cameras = [
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/gate"),
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://camera/yard"),
    ]
    fleet, workers, _recorder, _startup, _publisher = _fleet(cameras=cameras)
    workers["gate"].request_stop.side_effect = RuntimeError("stop failed")

    with pytest.raises(CameraFleetOperationError, match="gate"):
        fleet.stop_workers(timeout=1.0)

    workers["yard"].request_stop.assert_called_once_with()
    workers["gate"].close.assert_called_once_with()
    workers["yard"].close.assert_called_once_with()


def test_shutdown_wait_receives_each_workers_generation_ticket() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    ticket = object()
    workers["gate"].request_stop.return_value = ticket

    fleet.stop_workers(timeout=1.0)

    deadline, forwarded = workers["gate"].wait_stopped.call_args.args
    assert deadline > 0
    assert forwarded is ticket


def test_shutdown_does_not_race_close_against_a_timed_out_stop() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    workers["gate"].wait_stopped.return_value = False
    workers["gate"].active_workers.return_value = ["capture: live"]

    with pytest.raises(CameraFleetOperationError, match="gate"):
        fleet.stop_workers(timeout=0.01)

    workers["gate"].close.assert_not_called()
    assert fleet.status()["shutdown_residual_camera_ids"] == ["gate"]

    workers["gate"].wait_stopped.return_value = True
    workers["gate"].active_workers.return_value = []
    fleet.stop_workers(timeout=1.0)

    workers["gate"].close.assert_called_once_with()
    assert fleet.status()["shutdown_residual_camera_ids"] == []


def test_onvif_release_attempts_every_camera_after_peer_failure() -> None:
    cameras = [
        CameraConfig(id="gate", name="Gate", stream_url="rtsp://camera/gate"),
        CameraConfig(id="yard", name="Yard", stream_url="rtsp://camera/yard"),
    ]
    fleet, workers, _recorder, _startup, _publisher = _fleet(cameras=cameras)
    workers["gate"].request_onvif_stop.side_effect = RuntimeError("release failed")

    with pytest.raises(CameraFleetOperationError, match="gate"):
        fleet.release_onvif(timeout=1.0)

    workers["yard"].request_onvif_stop.assert_called_once_with()


def test_onvif_quiescence_cancels_admission_before_release() -> None:
    fleet, workers, _recorder, startup, _publisher = _fleet()
    order: list[str] = []
    startup.cancel.side_effect = lambda: order.append("cancel") or True
    workers["gate"].request_onvif_stop.side_effect = lambda: order.append("release")

    fleet.quiesce_onvif(timeout=1.0)

    assert order == ["cancel", "release"]
    workers["gate"].request_stop.assert_not_called()


def test_fleet_construction_rejects_camera_worker_mismatch() -> None:
    camera = CameraConfig(
        id="gate",
        name="Gate",
        stream_url="rtsp://camera/gate",
    )

    with pytest.raises(ValueError, match="worker mismatch"):
        CameraFleetLifecycle(
            cameras=[camera],
            workers={},
            recorder=Mock(),
            startup=Mock(),
            state_publisher=Mock(),
        )
