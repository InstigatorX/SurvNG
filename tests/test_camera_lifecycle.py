from __future__ import annotations

import threading
import time
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from survng.app.camera_lifecycle import (
    CameraLifecyclePhase,
    CameraLifecycleService,
    CameraRuntimeState,
)


def _service() -> tuple[CameraLifecycleService, SimpleNamespace]:
    state = CameraRuntimeState()
    capture = Mock()
    capture.start.return_value = True
    capture.threads.return_value = {}
    capture.wait_stopped.return_value = {}
    onvif = Mock()
    onvif.running = False
    onvif.wait_stopped.return_value = True
    tracking = Mock()
    tracking.stop.return_value = True
    tracking.wait_stopped.return_value = True
    tracking.running.return_value = False
    motion_runtime = Mock()
    motion_runtime.active_workers.return_value = []
    motion_runtime.wait_stopped.return_value = True
    tracking_frames = Mock()
    service = CameraLifecycleService(
        camera_id="gate",
        state=state,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        motion_runtime=motion_runtime,
        tracking_frames=tracking_frames,
    )
    return service, SimpleNamespace(
        state=state,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        motion_runtime=motion_runtime,
        tracking_frames=tracking_frames,
    )


def test_detection_work_leases_are_concurrent_and_drain_after_failure() -> None:
    state = CameraRuntimeState()
    concurrent = threading.Event()

    def another_frame():
        with state.detection_work() as admitted:
            if admitted:
                concurrent.set()

    with pytest.raises(ValueError, match="frame failed"):
        with state.detection_work() as admitted:
            assert admitted
            frame = threading.Thread(target=another_frame)
            frame.start()
            try:
                assert concurrent.wait(1.0)
            finally:
                frame.join(1.0)
            assert not state.wait_detection_idle(0)
            with state.detection_work() as second:
                assert second
            state.detection_enabled = False
            with state.detection_work() as rejected:
                assert not rejected
            raise ValueError("frame failed")
    assert state.wait_detection_idle(0)


def test_start_orders_state_cleanup_before_producers() -> None:
    service, owned = _service()
    assert owned.state.enabled is False
    assert owned.state.accepting_motion_events is False
    assert owned.state.phase is CameraLifecyclePhase.STOPPED
    assert owned.state.stop_event.is_set()
    order: list[str] = []
    owned.tracking.sync_accepting.side_effect = lambda: order.append("tracking")
    owned.motion_runtime.start.side_effect = lambda _stop: order.append("motion")
    owned.capture.start.side_effect = lambda: order.append("capture") or True
    service.start()

    assert order == ["tracking", "motion", "capture"]
    assert owned.state.enabled is True
    assert owned.state.accepting_motion_events is True
    assert owned.state.phase is CameraLifecyclePhase.RUNNING
    assert owned.state.generation == 1
    assert not owned.state.stop_event.is_set()
    owned.onvif.start.assert_called_once_with()


def test_start_skips_onvif_when_detection_is_disabled() -> None:
    service, owned = _service()
    service.set_detection_enabled(False)

    service.start()

    assert owned.state.phase is CameraLifecyclePhase.RUNNING
    assert owned.state.detection_enabled is False
    owned.onvif.start.assert_not_called()
    owned.motion_runtime.start.assert_not_called()
    owned.capture.start.assert_called_once_with()


def test_start_failure_rolls_back_runtime_state() -> None:
    service, owned = _service()
    owned.capture.start.side_effect = RuntimeError("capture unavailable")

    with pytest.raises(RuntimeError, match="capture unavailable"):
        service.start()

    assert owned.state.enabled is False
    assert owned.state.accepting_motion_events is False
    assert owned.state.phase is CameraLifecyclePhase.FAILED
    assert "capture unavailable" in owned.state.last_failure
    assert owned.state.stop_event.is_set()
    owned.capture.request_stop.assert_called_once_with()
    owned.onvif.request_stop.assert_called_once_with()


def test_start_failure_redacts_credentials_from_runtime_status() -> None:
    service, owned = _service()
    owned.capture.start.side_effect = RuntimeError(
        "open failed for rtsp://admin:supersecret@192.0.2.10/live"
    )

    with pytest.raises(RuntimeError) as raised:
        service.start()

    status = service.runtime_status()
    assert "supersecret" not in str(raised.value)
    assert "supersecret" not in status["last_failure"]
    assert "rtsp://admin:***@192.0.2.10/live" in status["last_failure"]


def test_stop_attempts_later_cleanup_after_early_failure() -> None:
    service, owned = _service()
    owned.capture.request_stop.side_effect = RuntimeError("capture stop failed")

    with pytest.raises(RuntimeError, match="capture cleanup"):
        service.stop()

    owned.onvif.request_stop.assert_called_once_with()
    owned.tracking.request_stop.assert_called_once_with()
    owned.motion_runtime.request_stop.assert_called_once_with()
    owned.tracking_frames.clear.assert_called_once_with()
    owned.motion_runtime.wait_stopped.assert_called_once_with(
        analysis_timeout=pytest.approx(22.0, abs=0.1),
        decision_timeout=pytest.approx(22.0, abs=0.1),
    )
    assert owned.state.phase is CameraLifecyclePhase.FAILED


def test_stop_failure_does_not_chain_unredacted_credentials() -> None:
    service, owned = _service()
    secret = "rtsp://admin:supersecret@192.0.2.10/live"
    owned.capture.request_stop.side_effect = RuntimeError(f"failed {secret}")

    with pytest.raises(RuntimeError) as raised:
        service.stop()

    formatted = "".join(traceback.format_exception(raised.value))
    assert "supersecret" not in formatted


def test_detection_disable_keeps_admission_closed_when_tracking_sync_fails() -> None:
    service, owned = _service()
    owned.tracking.sync_accepting.side_effect = [RuntimeError("sync failed"), None]

    with pytest.raises(RuntimeError, match="sync failed"):
        service.set_detection_enabled(False)

    assert owned.state.detection_enabled is False
    assert owned.tracking.sync_accepting.call_count == 1
    assert service.runtime_status()["detection_cleanup_required"]


def test_detection_switch_controls_onvif_while_camera_is_running() -> None:
    service, owned = _service()
    service.start()
    owned.onvif.reset_mock()
    motion_stop = owned.motion_runtime.start.call_args.args[0]
    assert motion_stop is not owned.state.stop_event
    owned.motion_runtime.request_stop.side_effect = motion_stop.set

    service.set_detection_enabled(False)

    assert owned.state.detection_enabled is False
    owned.onvif.stop.assert_called_once_with()
    owned.onvif.start.assert_not_called()
    assert motion_stop.is_set()
    assert not owned.state.stop_event.is_set()
    owned.motion_runtime.wait_stopped.assert_called_once()
    owned.tracking_frames.clear.assert_called_once_with(reason="detection_disabled")

    service.set_detection_enabled(True)

    assert owned.state.detection_enabled is True
    owned.onvif.start.assert_called_once_with()
    assert owned.motion_runtime.start.call_count == 2
    assert not owned.motion_runtime.start.call_args.args[0].is_set()
    owned.capture.start.assert_called_once_with()
    owned.capture.request_stop.assert_not_called()


def test_detection_change_rolls_back_when_onvif_start_fails() -> None:
    service, owned = _service()
    service.set_detection_enabled(False)
    service.start()
    owned.motion_runtime.request_stop.reset_mock()
    owned.onvif.start.side_effect = RuntimeError("subscription failed")

    with pytest.raises(RuntimeError, match="subscription failed"):
        service.set_detection_enabled(True)

    assert owned.state.detection_enabled is False
    assert owned.tracking.sync_accepting.call_count == 4
    owned.onvif.stop.assert_called_once_with()
    owned.motion_runtime.request_stop.assert_called_once_with()
    owned.motion_runtime.wait_stopped.assert_called_once()
    assert not owned.state.stop_event.is_set()


def test_detection_disable_timeout_keeps_admission_closed_and_can_retry() -> None:
    service, owned = _service()
    service.start()
    owned.motion_runtime.wait_stopped.side_effect = [False, True, True]

    with pytest.raises(RuntimeError, match="motion runtime did not stop"):
        service.set_detection_enabled(False)

    assert owned.state.detection_enabled is False
    assert service.runtime_status()["detection_cleanup_required"]
    with owned.state.detection_work() as admitted:
        assert not admitted
    assert not owned.state.stop_event.is_set()
    assert owned.motion_runtime.start.call_count == 1
    assert owned.motion_runtime.wait_stopped.call_count == 1
    assert owned.onvif.start.call_count == 1
    owned.capture.start.assert_called_once_with()

    service.set_detection_enabled(False)
    assert not service.runtime_status()["detection_cleanup_required"]
    assert owned.state.detection_enabled is False
    assert owned.motion_runtime.start.call_count == 1


def test_detection_disable_does_not_succeed_with_tracking_still_running() -> None:
    service, owned = _service()
    service.start()
    owned.tracking.running.return_value = True

    with pytest.raises(RuntimeError, match="object tracking did not stop"):
        service.set_detection_enabled(False)

    assert owned.state.detection_enabled is False
    owned.motion_runtime.request_stop.assert_called_once_with()


def test_failed_detection_cleanup_must_finish_before_reenable(caplog) -> None:
    service, owned = _service()
    service.start()
    owned.motion_runtime.wait_stopped.return_value = False
    owned.motion_runtime.active_workers.return_value = ["motion refinement"]

    with pytest.raises(RuntimeError, match="motion runtime did not stop"):
        service.set_detection_enabled(False)

    assert owned.onvif.start.call_count == 1
    assert "detection remains disabled; workers=motion refinement" in caplog.text
    with pytest.raises(RuntimeError, match="motion runtime did not stop"):
        service.set_detection_enabled(True)
    assert owned.state.detection_enabled is False
    assert owned.motion_runtime.start.call_count == 1

    def finish_cleanup(**kwargs):
        with owned.state.detection_work() as admitted:
            assert not admitted
        return True

    owned.motion_runtime.wait_stopped.side_effect = finish_cleanup
    service.set_detection_enabled(True)
    assert owned.state.detection_enabled is True
    assert owned.motion_runtime.start.call_count == 2
    service.set_detection_enabled(True)
    assert owned.motion_runtime.start.call_count == 2


def test_runtime_status_reports_authoritative_state_and_active_workers() -> None:
    service, owned = _service()
    owned.state.enabled = True
    owned.state.phase = CameraLifecyclePhase.RUNNING
    owned.motion_runtime.active_workers.return_value = ["motion analysis"]

    status = service.runtime_status()

    assert status["enabled"] is True
    assert status["phase"] == "running"
    assert status["detection_enabled"] is True
    assert status["active_workers"] == ["motion analysis"]
    assert status["active_worker_count"] == 1


def test_close_attempts_all_owned_resources() -> None:
    service, owned = _service()
    owned.motion_runtime.close.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="failed to close"):
        service.close()

    owned.motion_runtime.close.assert_called_once_with()
    owned.capture.close.assert_called_once_with()
    assert owned.state.phase is CameraLifecyclePhase.FAILED


def test_close_failure_does_not_chain_unredacted_credentials() -> None:
    service, owned = _service()
    secret = "rtsp://admin:supersecret@192.0.2.10/live"
    owned.motion_runtime.close.side_effect = RuntimeError(f"failed {secret}")

    with pytest.raises(RuntimeError) as raised:
        service.close()

    formatted = "".join(traceback.format_exception(raised.value))
    assert "supersecret" not in formatted


def test_starting_camera_does_not_block_runtime_status() -> None:
    service, owned = _service()
    capture_entered = threading.Event()
    release_capture = threading.Event()

    def blocking_capture_start() -> bool:
        capture_entered.set()
        assert release_capture.wait(1.0)
        return True

    owned.capture.start.side_effect = blocking_capture_start
    runner = threading.Thread(target=service.start)

    runner.start()
    assert capture_entered.wait(1.0)
    status = service.runtime_status()
    assert status["phase"] == "starting"
    assert status["enabled"] is True
    release_capture.set()
    runner.join(timeout=1.0)

    assert not runner.is_alive()
    assert service.runtime_status()["phase"] == "running"


def test_stopping_camera_does_not_block_runtime_status() -> None:
    service, owned = _service()
    owned.state.phase = CameraLifecyclePhase.RUNNING
    owned.state.enabled = True
    capture_wait_entered = threading.Event()
    release_capture_wait = threading.Event()

    def blocking_capture_wait(_timeout: float) -> dict:
        capture_wait_entered.set()
        assert release_capture_wait.wait(1.0)
        return {}

    owned.capture.wait_stopped.side_effect = blocking_capture_wait
    runner = threading.Thread(target=service.stop)
    runner.start()
    assert capture_wait_entered.wait(1.0)

    status = service.runtime_status()
    assert status["phase"] == "stopping"
    assert status["enabled"] is False
    release_capture_wait.set()
    runner.join(timeout=1.0)

    assert not runner.is_alive()
    assert service.runtime_status()["phase"] == "stopped"


def test_closed_camera_cannot_be_restarted() -> None:
    service, owned = _service()
    service.close()

    assert owned.state.phase is CameraLifecyclePhase.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        service.start()


def test_repeated_fleet_stop_wait_preserves_closed_phase() -> None:
    service, owned = _service()
    service.close()

    service.request_stop()
    assert service.wait_stopped(time.monotonic() + 1.0)

    assert owned.state.phase is CameraLifecyclePhase.CLOSED


def test_repeated_stop_request_is_idempotent_for_one_generation() -> None:
    service, owned = _service()
    service.start()

    first = service.request_stop()
    second = service.request_stop()

    assert first is not None
    assert second == first
    assert first.generation == 1
    owned.capture.request_stop.assert_called_once_with()
    owned.onvif.request_stop.assert_called_once_with()
    owned.motion_runtime.request_stop.assert_called_once_with()


def test_start_is_rejected_until_requested_stop_is_completed() -> None:
    service, owned = _service()
    service.start()
    ticket = service.request_stop()

    with pytest.raises(RuntimeError, match="phase is stopping"):
        service.start()

    assert ticket is not None
    assert service.wait_stopped(time.monotonic() + 1.0, ticket)
    service.start()
    assert owned.state.generation == 2


def test_stale_stop_ticket_cannot_finalize_new_generation() -> None:
    service, owned = _service()
    service.start()
    first = service.request_stop()
    assert first is not None
    assert service.wait_stopped(time.monotonic() + 1.0, first)
    service.start()

    assert not service.wait_stopped(time.monotonic() + 1.0, first)
    assert owned.state.phase is CameraLifecyclePhase.RUNNING
    assert owned.state.generation == 2


def test_invalid_lifecycle_transition_is_rejected_without_state_change() -> None:
    service, owned = _service()
    owned.state.phase = CameraLifecyclePhase.RUNNING

    with owned.state.lock, pytest.raises(RuntimeError, match="running -> starting"):
        service._transition_locked(CameraLifecyclePhase.STARTING)

    assert owned.state.phase is CameraLifecyclePhase.RUNNING


def test_close_rejects_running_camera_even_without_visible_worker_threads() -> None:
    service, owned = _service()
    owned.state.phase = CameraLifecyclePhase.RUNNING
    owned.state.enabled = True

    with pytest.raises(RuntimeError, match="phase is running"):
        service.close()

    owned.capture.close.assert_not_called()


@pytest.mark.parametrize("residual_kind", ["onvif", "tracking", "capture"])
def test_close_never_marks_camera_closed_with_residual_worker(
    residual_kind: str,
) -> None:
    service, owned = _service()
    owned.state.phase = CameraLifecyclePhase.FAILED
    if residual_kind == "onvif":
        owned.onvif.running = True
    elif residual_kind == "tracking":
        owned.tracking.running.return_value = True
        owned.tracking.wait_stopped.return_value = False
    else:
        capture_thread = Mock()
        capture_thread.is_alive.return_value = True
        owned.capture.threads.return_value = {"live": capture_thread}
        owned.capture.wait_stopped.return_value = {"live": capture_thread}

    with pytest.raises(RuntimeError, match="owned workers remain"):
        service.close()

    assert owned.state.phase is CameraLifecyclePhase.FAILED
    owned.motion_runtime.close.assert_not_called()
    owned.capture.close.assert_not_called()
