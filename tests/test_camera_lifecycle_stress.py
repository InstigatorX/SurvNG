from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from survng.app.camera_lifecycle import (
    CameraLifecyclePhase,
    CameraLifecycleService,
    CameraRuntimeState,
)
from survng.app.config import MotionQualificationConfig
from survng.app.motion_analysis import FairMotionAnalysisLimiter
from survng.app.motion_analysis_service import MotionAnalysisService
from survng.app.motion_coordinator import VisualBackupCoordinator
from survng.app.motion_decisions import MotionDecisionOrchestrator
from survng.app.motion_events import MotionEventCoordinator
from survng.app.motion_ingress import MotionEventIngressService
from survng.app.motion_pipeline import MotionDebugSnapshotStore
from survng.app.motion_runtime import CameraMotionState, MotionRuntimeService


class _ThreadedProducer:
    """Small controllable producer used to fault-inject lifecycle boundaries."""

    def __init__(self, name: str, callback: Mock | None = None) -> None:
        self.name = name
        self.callback = callback
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.hold_on_stop = threading.Event()
        self.hold_on_stop.set()
        self.fail_start = False

    def start(self) -> bool:
        if self.fail_start:
            raise RuntimeError(f"{self.name} start failed")
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"stress-{self.name}",
            # Production components are non-daemon. Keeping the fault-injection
            # fake daemonized prevents a failed assertion from wedging pytest;
            # the assertions below still require every test-owned thread to exit.
            daemon=True,
        )
        self.thread.start()
        return True

    def _run(self) -> None:
        while not self.stop_event.wait(0.001):
            if self.callback is not None:
                self.callback()
        self.hold_on_stop.wait()

    def request_stop(self) -> None:
        self.stop_event.set()

    def wait_stopped(self, timeout: float) -> bool:
        thread = self.thread
        if thread is None:
            return True
        thread.join(timeout=max(0.0, timeout))
        if thread.is_alive():
            return False
        self.thread = None
        return True

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class _CaptureProducer(_ThreadedProducer):
    def threads(self) -> dict[str, threading.Thread]:
        return {"live": self.thread} if self.running and self.thread is not None else {}

    def wait_stopped(self, timeout: float) -> dict[str, threading.Thread]:
        return {} if super().wait_stopped(timeout) else self.threads()

    def close(self) -> None:
        assert not self.running


class _TrackingProducer(_ThreadedProducer):
    def sync_accepting(self) -> None:
        return

    def running(self) -> bool:
        return super().running


def _stress_runtime() -> tuple[CameraLifecycleService, SimpleNamespace]:
    camera_state = CameraRuntimeState()
    published: list[tuple[str, dict[str, object]]] = []
    state = CameraMotionState(
        camera_id="gate",
        camera_state=camera_state,
        event_callback=lambda kind, payload: published.append((kind, payload)),
    )
    config = MotionQualificationConfig(sample_fps=10.0, burst_quiet_seconds=0.1)
    events = MotionEventCoordinator(queue_size=4, retry_limit=1)
    qualification = Mock()
    qualification.frame_analysis_required.return_value = True
    qualification.settings.return_value = ("adaptive", "balanced", 64)
    qualification.preprocessor_implementation.return_value = "gray_blur"
    qualification.continuous_primary_required.return_value = False
    qualification.continuous_primary_due.return_value = False
    qualification.illumination_filter_enabled.return_value = False
    qualification.trigger_mode.return_value = "adaptive"
    qualification.visual_backup_settings.return_value = {}
    qualification.suppression_verification_rate.return_value = 0.0
    evidence = Mock()
    analysis = MotionAnalysisService(
        camera_id="gate",
        frame_lock=threading.Lock(),
        analysis_lock=threading.Lock(),
        ring_size=4,
        queue_size=1,
        limiter=FairMotionAnalysisLimiter(1),
        events=events,
        evidence=evidence,
        visual_backup=VisualBackupCoordinator(),
        audit_recorder=Mock(),
        debug_store=MotionDebugSnapshotStore(),
        config=config,
        qualification=qualification,
        media=Mock(),
        state=state,
    )
    incidents = Mock()
    decisions = MotionDecisionOrchestrator(
        camera_id="gate",
        events=events,
        audit_recorder=Mock(),
        config=config,
        qualification=qualification,
        incidents=incidents,
        media=Mock(),
        analysis=analysis,
        state=state,
    )
    ingress = MotionEventIngressService(
        camera_id="gate",
        events=events,
        qualification=qualification,
        state=state,
    )
    runtime = MotionRuntimeService(
        camera_id="gate",
        state=state,
        events=events,
        analysis=analysis,
        decisions=decisions,
        ingress=ingress,
        qualification=qualification,
        evidence=evidence,
        pipelines=(),
    )
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    frame.setflags(write=False)
    capture = _CaptureProducer(
        "capture",
        callback=Mock(
            side_effect=lambda: runtime.submit_frame(
                frame,
                time.monotonic(),
                time.time(),
            )
        ),
    )
    onvif = _ThreadedProducer(
        "onvif",
        callback=Mock(side_effect=lambda: runtime.handle_event("onvif/motion")),
    )
    tracking = _TrackingProducer("tracking")
    tracking_frames = Mock()
    lifecycle = CameraLifecycleService(
        camera_id="gate",
        state=camera_state,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        motion_runtime=runtime,
        tracking_frames=tracking_frames,
    )
    return lifecycle, SimpleNamespace(
        state=camera_state,
        runtime=runtime,
        events=events,
        analysis=analysis,
        decisions=decisions,
        capture=capture,
        onvif=onvif,
        tracking=tracking,
        tracking_frames=tracking_frames,
        published=published,
    )


def _force_cleanup(lifecycle: CameraLifecycleService, owned: SimpleNamespace) -> None:
    owned.capture.hold_on_stop.set()
    owned.onvif.hold_on_stop.set()
    owned.tracking.hold_on_stop.set()
    try:
        lifecycle.stop()
    except RuntimeError:
        # The test's assertions report the original lifecycle defect. This
        # fallback only prevents a failed stress assertion from leaking workers.
        owned.state.stop_event.set()
        owned.capture.request_stop()
        owned.onvif.request_stop()
        owned.tracking.request_stop()
        owned.runtime.request_stop()
        owned.capture.wait_stopped(1.0)
        owned.onvif.wait_stopped(1.0)
        owned.tracking.wait_stopped(1.0)
        owned.runtime.wait_stopped(analysis_timeout=1.0, decision_timeout=1.0)


def test_repeated_full_camera_generations_leave_no_workers_or_stale_work() -> None:
    lifecycle, owned = _stress_runtime()
    reader_stop = threading.Event()
    observed_statuses: list[dict[str, object]] = []

    def read_status() -> None:
        while not reader_stop.wait(0.0005):
            observed_statuses.append(lifecycle.runtime_status())

    reader = threading.Thread(target=read_status, name="stress-status-reader")
    reader.start()
    try:
        for expected_generation in range(1, 51):
            lifecycle.start()
            assert owned.analysis.running()
            assert owned.decisions.running()
            assert lifecycle.runtime_status()["generation"] == expected_generation
            time.sleep(0.002)

            ticket = lifecycle.request_stop()
            assert ticket is not None
            assert ticket.generation == expected_generation
            # Producers captured before admission closed may arrive late. The
            # ingress/runtime boundaries must reject them for the stopped run.
            owned.runtime.handle_event("manual/late")
            assert lifecycle.wait_stopped(time.monotonic() + 1.0, ticket)
            assert owned.events.queue.empty()
            assert not owned.analysis.frames
            assert lifecycle.active_workers() == []
            assert lifecycle.runtime_status()["phase"] == "stopped"
    finally:
        reader_stop.set()
        reader.join(timeout=1.0)
        _force_cleanup(lifecycle, owned)

    assert observed_statuses
    assert all(
        status["phase"] in {phase.value for phase in CameraLifecyclePhase}
        and int(status["generation"]) >= 0
        for status in observed_statuses
    )
    assert not any(
        thread.name.startswith(("stress-", "motion-gate", "motion-analysis-gate"))
        and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_stuck_capture_generation_can_be_recovered_without_stale_completion() -> None:
    lifecycle, owned = _stress_runtime()
    try:
        lifecycle.start()
        owned.capture.hold_on_stop.clear()

        ticket = lifecycle.request_stop()
        assert ticket is not None
        with pytest.raises(RuntimeError, match="capture sources: live"):
            lifecycle.wait_stopped(time.monotonic() + 0.01, ticket)
        assert owned.state.phase is CameraLifecyclePhase.FAILED

        owned.capture.hold_on_stop.set()
        recovery = lifecycle.request_stop()
        assert recovery is not None
        assert recovery.generation == ticket.generation
        assert lifecycle.wait_stopped(time.monotonic() + 1.0, recovery)
        assert owned.state.phase is CameraLifecyclePhase.STOPPED
        assert lifecycle.active_workers() == []
    finally:
        _force_cleanup(lifecycle, owned)


def test_startup_rollback_cleans_real_motion_workers_at_each_producer_boundary() -> None:
    for failing_component in ("capture", "onvif"):
        lifecycle, owned = _stress_runtime()
        getattr(owned, failing_component).fail_start = True
        try:
            try:
                lifecycle.start()
            except RuntimeError as error:
                assert "start failed" in str(error)
            else:
                raise AssertionError("fault-injected startup unexpectedly succeeded")

            assert owned.state.phase is CameraLifecyclePhase.FAILED
            assert owned.state.stop_event.is_set()
            assert lifecycle.active_workers() == []
            assert not owned.analysis.running()
            assert not owned.decisions.running()
        finally:
            _force_cleanup(lifecycle, owned)
