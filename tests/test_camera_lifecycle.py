from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from survng.app.camera_lifecycle import CameraLifecycleService, CameraRuntimeState


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
    assert owned.state.accepting_motion_events is True
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
    assert owned.state.stop_event.is_set()
    owned.capture.request_stop.assert_called_once_with()
    owned.onvif.stop.assert_called_once_with()


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


def test_detection_change_rolls_back_when_tracking_sync_fails() -> None:
    service, owned = _service()
    owned.tracking.sync_accepting.side_effect = [RuntimeError("sync failed"), None]

    with pytest.raises(RuntimeError, match="sync failed"):
        service.set_detection_enabled(False)

    assert owned.state.detection_enabled is True
    assert owned.tracking.sync_accepting.call_count == 2


def test_close_attempts_all_owned_resources() -> None:
    service, owned = _service()
    owned.pipelines[0][1].close.side_effect = RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="failed to close"):
        service.close()

    owned.pipelines[1][1].close.assert_called_once_with()
    owned.capture.close.assert_called_once_with()
