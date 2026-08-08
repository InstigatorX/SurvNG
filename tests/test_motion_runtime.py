from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from survng.app.camera_lifecycle import CameraRuntimeState
from survng.app.motion import MotionQualificationResult
from survng.app.motion_runtime import CameraMotionState, MotionRuntimeService


def _runtime() -> tuple[MotionRuntimeService, SimpleNamespace]:
    camera_state = CameraRuntimeState()
    state = CameraMotionState(
        camera_id="gate",
        camera_state=camera_state,
        event_callback=None,
    )
    events = Mock()
    events.queue.qsize.return_value = 0
    events.retry_queue_depth.return_value = 0
    events.runtime_status.return_value = {}
    analysis = Mock()
    analysis.running.return_value = False
    analysis.wait_stopped.return_value = True
    decisions = Mock()
    decisions.running.return_value = False
    decisions.wait_stopped.return_value = True
    incidents = Mock()
    incidents.running.return_value = False
    incidents.wait_stopped.return_value = True
    ingress = Mock()
    qualification = Mock()
    evidence = Mock()
    pipelines = (("qualification", Mock()), ("fusion", Mock()))
    runtime = MotionRuntimeService(
        camera_id="gate",
        state=state,
        events=events,
        analysis=analysis,
        decisions=decisions,
        incidents=incidents,
        ingress=ingress,
        qualification=qualification,
        evidence=evidence,
        pipelines=pipelines,
    )
    return runtime, SimpleNamespace(
        camera_state=camera_state,
        state=state,
        events=events,
        analysis=analysis,
        decisions=decisions,
        incidents=incidents,
        ingress=ingress,
        qualification=qualification,
        evidence=evidence,
        pipelines=pipelines,
    )


def test_runtime_starts_both_workers_as_one_generation() -> None:
    runtime, owned = _runtime()
    stop_event = threading.Event()

    runtime.start(stop_event)

    owned.events.clear.assert_called_once_with()
    owned.analysis.start.assert_called_once_with(stop_event)
    owned.decisions.start.assert_called_once_with(stop_event)


def test_partial_start_failure_rolls_back_analysis_worker() -> None:
    runtime, owned = _runtime()
    owned.decisions.start.side_effect = RuntimeError("decision start failed")

    with pytest.raises(RuntimeError, match="decision start failed"):
        runtime.start(threading.Event())

    owned.analysis.request_stop.assert_called_once_with()
    owned.decisions.request_stop.assert_called_once_with()
    owned.decisions.wait_stopped.assert_called_once_with(1.0)
    owned.analysis.wait_stopped.assert_called_once_with(1.0)


def test_successful_stop_resets_owned_generation_state() -> None:
    runtime, owned = _runtime()
    runtime.start(threading.Event())

    runtime.request_stop()
    stopped = runtime.wait_stopped(
        analysis_timeout=3.0,
        decision_timeout=4.0,
    )

    assert stopped
    owned.analysis.request_stop.assert_called_once_with()
    owned.decisions.request_stop.assert_called_once_with()
    decision_timeout = owned.decisions.wait_stopped.call_args.args[0]
    analysis_timeout = owned.analysis.wait_stopped.call_args.args[0]
    assert 3.9 < decision_timeout <= 4.0
    assert 0.0 <= analysis_timeout <= 3.0
    owned.analysis.reset.assert_called_once_with()
    owned.evidence.clear.assert_called_once_with()
    owned.qualification.reset_runtime.assert_called_once_with()
    owned.events.reset.assert_called_once_with()


def test_stuck_worker_preserves_generation_state_for_diagnosis() -> None:
    runtime, owned = _runtime()
    runtime.start(threading.Event())
    owned.decisions.wait_stopped.return_value = False
    owned.decisions.running.return_value = True

    stopped = runtime.wait_stopped(
        analysis_timeout=3.0,
        decision_timeout=4.0,
    )

    assert not stopped
    owned.analysis.reset.assert_not_called()
    owned.evidence.clear.assert_not_called()
    owned.qualification.reset_runtime.assert_not_called()
    owned.events.reset.assert_not_called()
    assert runtime.active_workers() == ["motion events"]


def test_close_attempts_every_pipeline_after_failure() -> None:
    runtime, owned = _runtime()
    owned.pipelines[0][1].close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="failed to close"):
        runtime.close()

    owned.pipelines[1][1].close.assert_called_once_with()


def test_stop_request_signals_both_workers_when_first_signal_fails() -> None:
    runtime, owned = _runtime()
    stop_event = threading.Event()
    runtime.start(stop_event)
    owned.analysis.request_stop.side_effect = RuntimeError("analysis signal failed")

    with pytest.raises(RuntimeError, match="analysis signal failed"):
        runtime.request_stop()

    owned.decisions.request_stop.assert_called_once_with()
    assert stop_event.is_set()


def test_partial_start_failure_sets_shared_stop_event() -> None:
    runtime, owned = _runtime()
    stop_event = threading.Event()
    owned.decisions.start.side_effect = RuntimeError("secret rtsp://admin:pw@host/live")

    with pytest.raises(RuntimeError):
        runtime.start(stop_event)

    assert stop_event.is_set()


def test_runtime_error_does_not_chain_secret_bearing_cause() -> None:
    runtime, owned = _runtime()
    runtime.start(threading.Event())
    owned.analysis.request_stop.side_effect = RuntimeError(
        "rtsp://admin:supersecret@192.0.2.10/live"
    )

    with pytest.raises(RuntimeError) as raised:
        runtime.request_stop()

    assert "supersecret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_stop_wait_joins_both_workers_when_first_join_fails() -> None:
    runtime, owned = _runtime()
    runtime.start(threading.Event())
    owned.decisions.wait_stopped.side_effect = RuntimeError("decision join failed")

    with pytest.raises(RuntimeError, match="decision join failed"):
        runtime.wait_stopped(analysis_timeout=3.0, decision_timeout=4.0)

    owned.analysis.wait_stopped.assert_called_once_with(3.0)
    owned.analysis.reset.assert_not_called()


def test_generation_cleanup_attempts_every_reset_when_one_fails() -> None:
    runtime, owned = _runtime()
    runtime.start(threading.Event())
    owned.analysis.reset.side_effect = RuntimeError("analysis reset failed")

    with pytest.raises(RuntimeError, match="analysis reset failed"):
        runtime.wait_stopped(analysis_timeout=3.0, decision_timeout=4.0)

    owned.evidence.clear.assert_called_once_with()
    owned.qualification.reset_runtime.assert_called_once_with()
    owned.events.reset.assert_called_once_with()
    assert not runtime.runtime_status()["generation_clean"]

    with pytest.raises(RuntimeError, match="incomplete generation cleanup"):
        runtime.start(threading.Event())


def test_motion_state_isolated_callback_failure_and_decision_telemetry() -> None:
    callback = Mock(side_effect=RuntimeError("subscriber failed"))
    state = CameraMotionState(
        camera_id="gate",
        camera_state=CameraRuntimeState(),
        event_callback=callback,
    )
    result = MotionQualificationResult(
        False,
        0.2,
        0.5,
        "low_score",
        3,
        {},
    )

    state.publish_event("motion", {"camera_id": "gate"})
    state.record_decision(
        result=result,
        qualification=result.as_dict(),
        retry_attempt=False,
        priority=False,
        mode="camera",
        borderline_candidate=False,
        suppression_verification_candidate=False,
    )

    snapshot = state.stats_snapshot()
    assert snapshot["event_callback_errors"] == 1
    assert snapshot["bursts"] == 1
    assert snapshot["suppressed"] == 1
    assert snapshot["last_result"]["reason"] == "low_score"

    snapshot["last_result"]["reason"] = "mutated"
    assert state.stats_snapshot()["last_result"]["reason"] == "low_score"


def test_motion_state_reports_bounded_analysis_wait_percentiles() -> None:
    state = CameraMotionState(
        camera_id="gate",
        camera_state=CameraRuntimeState(),
        event_callback=None,
    )

    for wait_ms in range(1, 101):
        state.record_analysis_wait(float(wait_ms))

    snapshot = state.stats_snapshot()
    assert snapshot["analysis_wait_ms_p95"] == 95.0
    assert snapshot["analysis_wait_ms_p99"] == 99.0
