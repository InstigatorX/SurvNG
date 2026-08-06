from __future__ import annotations

import threading
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    tracking = Mock()
    tracking.stop.return_value = True
    tracking.running.return_value = False
    motion_analysis = Mock()
    motion_analysis.thread = None
    motion_analysis.wait_stopped.return_value = True
    motion_events = Mock()
    motion_events.thread = None
    tracking_frames = Mock()
    motion_evidence = Mock()
    motion_qualification = Mock()
    pipelines = (("qualification", Mock()), ("fusion", Mock()))
    run_motion_events = Mock()
    service = CameraLifecycleService(
        camera_id="gate",
        state=state,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        motion_analysis=motion_analysis,
        motion_events=motion_events,
        tracking_frames=tracking_frames,
        motion_evidence=motion_evidence,
        motion_qualification=motion_qualification,
        motion_pipelines=pipelines,
        run_motion_events=run_motion_events,
    )
    return service, SimpleNamespace(
        state=state,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        motion_analysis=motion_analysis,
        motion_events=motion_events,
        tracking_frames=tracking_frames,
        motion_evidence=motion_evidence,
        motion_qualification=motion_qualification,
        pipelines=pipelines,
        run_motion_events=run_motion_events,
    )


def test_start_orders_state_cleanup_before_producers() -> None:
    service, owned = _service()
    assert owned.state.enabled is False
    assert owned.state.accepting_motion_events is False
    assert owned.state.phase is CameraLifecyclePhase.STOPPED
    assert owned.state.stop_event.is_set()
    order: list[str] = []
    owned.tracking.sync_accepting.side_effect = lambda: order.append("tracking")
    owned.motion_events.clear.side_effect = lambda: order.append("clear")
    owned.motion_analysis.start.side_effect = lambda _stop: order.append("analysis")
    owned.capture.start.side_effect = lambda: order.append("capture") or True
    motion_thread = Mock()
    motion_thread.is_alive.return_value = False

    with patch(
        "survng.app.camera_lifecycle.threading.Thread",
        return_value=motion_thread,
    ):
        service.start()

    assert order == ["tracking", "clear", "analysis", "capture"]
    assert owned.state.enabled is True
    assert owned.state.accepting_motion_events is True
    assert owned.state.phase is CameraLifecyclePhase.RUNNING
    assert owned.state.generation == 1
    assert not owned.state.stop_event.is_set()
    assert owned.motion_events.thread is motion_thread
    motion_thread.start.assert_called_once_with()
    owned.onvif.start.assert_called_once_with()


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
    owned.onvif.stop.assert_called_once_with()


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

    owned.onvif.stop.assert_called_once_with()
    owned.tracking.stop.assert_called_once_with()
    owned.motion_analysis.request_stop.assert_called_once_with()
    owned.motion_events.signal_stop.assert_called_once_with()
    owned.tracking_frames.clear.assert_called_once_with()
    owned.motion_events.reset.assert_called_once_with()
    assert owned.state.phase is CameraLifecyclePhase.FAILED


def test_stop_failure_does_not_chain_unredacted_credentials() -> None:
    service, owned = _service()
    secret = "rtsp://admin:supersecret@192.0.2.10/live"
    owned.capture.request_stop.side_effect = RuntimeError(f"failed {secret}")

    with pytest.raises(RuntimeError) as raised:
        service.stop()

    formatted = "".join(traceback.format_exception(raised.value))
    assert "supersecret" not in formatted


def test_detection_change_rolls_back_when_tracking_sync_fails() -> None:
    service, owned = _service()
    owned.tracking.sync_accepting.side_effect = [RuntimeError("sync failed"), None]

    with pytest.raises(RuntimeError, match="sync failed"):
        service.set_detection_enabled(False)

    assert owned.state.detection_enabled is True
    assert owned.tracking.sync_accepting.call_count == 2


def test_runtime_status_reports_authoritative_state_and_active_workers() -> None:
    service, owned = _service()
    owned.state.enabled = True
    owned.state.phase = CameraLifecyclePhase.RUNNING
    owned.motion_analysis.thread = Mock()
    owned.motion_analysis.thread.is_alive.return_value = True

    status = service.runtime_status()

    assert status["enabled"] is True
    assert status["phase"] == "running"
    assert status["detection_enabled"] is True
    assert status["active_workers"] == ["motion analysis"]
    assert status["active_worker_count"] == 1


def test_close_attempts_all_owned_resources() -> None:
    service, owned = _service()
    owned.pipelines[0][1].close.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="failed to close"):
        service.close()

    owned.pipelines[1][1].close.assert_called_once_with()
    owned.capture.close.assert_called_once_with()
    assert owned.state.phase is CameraLifecyclePhase.FAILED


def test_close_failure_does_not_chain_unredacted_credentials() -> None:
    service, owned = _service()
    secret = "rtsp://admin:supersecret@192.0.2.10/live"
    owned.pipelines[0][1].close.side_effect = RuntimeError(f"failed {secret}")

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
    motion_thread = Mock()
    motion_thread.is_alive.return_value = False
    runner = threading.Thread(target=service.start)

    with patch(
        "survng.app.camera_lifecycle.threading.Thread",
        return_value=motion_thread,
    ):
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
        owned.tracking.stop.return_value = False
    else:
        capture_thread = Mock()
        capture_thread.is_alive.return_value = True
        owned.capture.threads.return_value = {"live": capture_thread}
        owned.capture.wait_stopped.return_value = {"live": capture_thread}

    with pytest.raises(RuntimeError, match="owned workers remain"):
        service.close()

    assert owned.state.phase is CameraLifecyclePhase.FAILED
    for _label, pipeline in owned.pipelines:
        pipeline.close.assert_not_called()
    owned.capture.close.assert_not_called()
