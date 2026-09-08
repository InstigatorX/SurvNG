from __future__ import annotations

import threading
from unittest.mock import Mock, patch

import pytest

from survng.app.camera_control import CameraControlService
from survng.app.camera_startup import CameraStartupCoordinator
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


def test_fleet_shutdown_waits_for_in_progress_direct_camera_start() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    start_entered = threading.Event()
    release_start = threading.Event()
    stop_signaled = threading.Event()

    def start() -> None:
        start_entered.set()
        assert release_start.wait(1.0)

    workers["gate"].start.side_effect = start
    workers["gate"].request_stop.side_effect = stop_signaled.set
    starter = threading.Thread(target=lambda: fleet.start_camera("gate"))
    stopper = threading.Thread(target=lambda: fleet.stop_workers(timeout=1.0))
    starter.start()
    assert start_entered.wait(1.0)
    stopper.start()
    assert not stop_signaled.wait(0.02)

    release_start.set()
    starter.join(1.0)
    stopper.join(1.0)
    assert not starter.is_alive()
    assert not stopper.is_alive()
    workers["gate"].request_stop.assert_called_once_with()


def test_stop_workers_cancels_startup_before_signaling_camera_workers() -> None:
    fleet, workers, _recorder, startup, _publisher = _fleet()
    order: list[str] = []
    startup.cancel.side_effect = lambda: order.append("cancel") or True
    workers["gate"].request_stop.side_effect = lambda: order.append("stop")

    fleet.stop_workers(timeout=1.0)

    assert order[:2] == ["cancel", "stop"]


def test_stopped_fleet_rejects_new_admission_and_direct_camera_start() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    fleet.stop_workers(timeout=1.0)

    assert not fleet.start_camera("gate")
    with pytest.raises(RuntimeError, match="stopping"):
        fleet.start_admission(())
    workers["gate"].start.assert_not_called()


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


def test_onvif_release_waits_for_each_requested_generation_ticket() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    ticket = object()
    workers["gate"].request_onvif_stop.return_value = ticket

    fleet.release_onvif(timeout=1.0)

    deadline, forwarded = workers["gate"].wait_onvif_stopped.call_args.args
    assert deadline > 0
    assert forwarded is ticket

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


def test_queued_admission_restarts_camera_after_successful_power_off(tmp_path):
    camera = CameraConfig(id="gate", name="Gate", stream_url="rtsp://gate")
    worker = Mock()
    running = threading.Event()
    worker.start.side_effect = running.set
    worker.stop.side_effect = running.clear
    worker.live_capture_ready.side_effect = running.is_set
    recorder = Mock()
    publisher = Mock()
    coordinator = CameraStartupCoordinator(recorder_settle_seconds=0)
    fleet = CameraFleetLifecycle(cameras=[camera], workers={camera.id: worker}, recorder=recorder,
                                startup=coordinator, state_publisher=publisher)
    controls = CameraControlService(cameras=[camera], workers={camera.id: worker}, recording=Mock(),
                                    fleet=fleet, mqtt=Mock(), runtime_monitor=Mock(),
                                    state_path=tmp_path / "controls.json")
    tasks = fleet.prepare_startup(camera_enabled={camera.id: True}, recording_enabled={}, detection_enabled={})
    enabled_snapshot_read = threading.Event()
    release_admission = threading.Event()
    original = fleet._admission_enabled

    def paused_initial_check(camera_id):
        value = original(camera_id)
        if not enabled_snapshot_read.is_set():
            enabled_snapshot_read.set()
            assert release_admission.wait(2)
        return value

    with patch.object(fleet, "_admission_enabled", side_effect=paused_initial_check):
        fleet.start_admission(tasks)
        assert enabled_snapshot_read.wait(2)
        assert controls.stop_camera(camera.id)
        assert not running.is_set()
        release_admission.set()
        assert coordinator.wait(2)
    assert controls.camera_enabled(camera.id) is False
    assert fleet.camera_enabled(camera.id) is False
    assert coordinator.status()["cameras"][camera.id]["phase"] == "skipped"
    assert not running.is_set()
    worker.start.assert_not_called()


def test_power_off_waits_for_admitted_start_then_stops_worker() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    task = fleet.prepare_startup(camera_enabled={}, recording_enabled={}, detection_enabled={})[0]
    entered = threading.Event()
    release = threading.Event()
    disabled = threading.Event()
    order = []

    def start():
        entered.set()
        assert release.wait(2)
        order.append("start")

    def power_off():
        assert fleet.set_camera_enabled("gate", False)
        disabled.set()
        assert fleet.stop_camera("gate")

    workers["gate"].start.side_effect = start
    workers["gate"].stop.side_effect = lambda: order.append("stop")
    starter = threading.Thread(target=task.start_camera)
    stopper = threading.Thread(target=power_off)
    starter.start()
    assert entered.wait(2)
    stopper.start()
    try:
        assert not disabled.wait(0.05)
    finally:
        release.set()
        starter.join(2)
        stopper.join(2)
    assert not starter.is_alive() and not stopper.is_alive()
    assert order == ["start", "stop"]
    assert not task.is_enabled()


def test_cancel_admission_does_not_deadlock_queued_camera_start() -> None:
    fleet, workers, _recorder, _startup, _publisher = _fleet()
    fleet.startup = CameraStartupCoordinator(recorder_settle_seconds=0)
    task = fleet.prepare_startup(camera_enabled={}, recording_enabled={}, detection_enabled={})[0]
    checked = threading.Event()
    release = threading.Event()
    original = fleet._admission_enabled

    def paused_check(camera_id):
        enabled = original(camera_id)
        if not checked.is_set():
            checked.set()
            assert release.wait(2)
        return enabled

    with patch.object(fleet, "_admission_enabled", side_effect=paused_check):
        fleet.start_admission([task])
        assert checked.wait(2)
        # Cancellation holds this same lock while it joins admission workers.
        with fleet._operation_lock:
            fleet._stopping.set()
            release.set()
            fleet.cancel_admission()
    assert fleet.wait(1)
    workers["gate"].start.assert_not_called()
