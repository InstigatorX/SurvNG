from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from survng.app.camera_capture import (
    CameraCaptureService,
    CaptureHandle,
    CapturedFrame,
)


class FakeHandle:
    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        self.frames = deque(frames or [])
        self.opened = False
        self.closed = False
        self.buffer_size: int | None = None

    def is_opened(self) -> bool:
        return self.opened

    def set_buffer_size(self, size: int) -> None:
        self.buffer_size = size

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.popleft()
        time.sleep(0.005)
        return False, None

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    def __init__(self, frame_batches: list[list[np.ndarray]] | None = None) -> None:
        self.frame_batches = deque(frame_batches or [])
        self.handles: list[FakeHandle] = []
        self.open_calls = 0

    def create_handle(self) -> CaptureHandle:
        frames = self.frame_batches.popleft() if self.frame_batches else []
        handle = FakeHandle(frames)
        self.handles.append(handle)
        return handle

    def open(self, handle, source_url, cancelled) -> bool:
        self.open_calls += 1
        if cancelled():
            return False
        handle.opened = True
        return True


class FailingOpenBackend(FakeBackend):
    def open(self, handle, source_url, cancelled) -> bool:
        raise RuntimeError(f"unable to open {source_url}")


def _service(
    backend: FakeBackend | None = None,
    **kwargs,
) -> CameraCaptureService:
    return CameraCaptureService(
        camera_id="gate",
        source_url=lambda source: f"rtsp://camera/{source}",
        backend=backend or FakeBackend(),
        retry_initial_seconds=0.01,
        retry_max_seconds=0.02,
        **kwargs,
    )


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_live_source_starts_persistently_and_stops_in_two_phases() -> None:
    backend = FakeBackend([[np.ones((10, 20, 3), dtype=np.uint8)]])
    service = _service(backend)

    assert service.start()
    _wait_until(lambda: service.status()["capture_stats"]["live"]["frames_received"] >= 1)
    service.request_stop()
    alive = service.wait_stopped(1.0)

    assert alive == {}
    assert backend.handles
    assert all(handle.closed for handle in backend.handles)
    assert service.latest("live") is None
    service.close()


def test_latest_frame_is_copied_and_rejected_when_stale() -> None:
    now = [100.0]
    service = _service(
        wall_clock=lambda: 1_700_000_000.0,
        monotonic_clock=lambda: now[0],
        stale_seconds=10.0,
    )
    source = np.ones((10, 20, 3), dtype=np.uint8)
    with service._lock:
        service._stop.clear()
    service._publish_frame("live", source)

    first = service.latest("live")
    assert first is not None
    first.image[:] = 7
    second = service.latest("live")
    assert second is not None
    assert int(second.image[0, 0, 0]) == 1
    assert second.captured_at_epoch == 1_700_000_000.0
    now[0] = 111.0
    assert service.latest("live") is None


def test_main_source_is_lazy_and_expires_after_demand_lease() -> None:
    backend = FakeBackend([[np.ones((10, 20, 3), dtype=np.uint8)]])
    service = _service(backend, main_idle_seconds=0.04)
    with service._lock:
        service._stop.clear()

    assert service.status()["main_running"] is False
    service.request_frame("main")
    _wait_until(lambda: service.latest("main") is not None)
    _wait_until(lambda: service.status()["main_running"] is False)

    assert service.latest("main") is None
    service.request_stop()
    assert service.wait_stopped(1.0) == {}


def test_replacing_stopping_source_does_not_let_old_runner_remove_new_one() -> None:
    service = _service(main_idle_seconds=5.0)
    with service._lock:
        service._stop.clear()
        service._last_access["main"] = time.monotonic()
    first = service.ensure_source("main")
    assert first.started
    with service._lock:
        old_stop = service._source_stops["main"]
        old_stop.set()
    second = service.ensure_source("main")

    assert second.started
    _wait_until(lambda: service.status()["main_running"] is True)
    service.request_stop()
    assert service.wait_stopped(1.0) == {}


def test_shutdown_joins_replaced_and_current_source_generations() -> None:
    service = _service(main_idle_seconds=5.0)
    release_old = threading.Event()
    old_stop = threading.Event()
    new_stop = threading.Event()
    old_thread = threading.Thread(target=lambda: release_old.wait(0.5))
    new_thread = threading.Thread(target=lambda: new_stop.wait(0.5))
    old_thread.start()
    new_thread.start()
    with service._lock:
        service._stop.clear()
        service._threads["main"] = new_thread
        service._source_stops["main"] = new_stop
        service._all_threads[old_thread] = ("main", old_stop)
        service._all_threads[new_thread] = ("main", new_stop)

    service.request_stop()
    release_old.set()
    alive = service.wait_stopped(1.0)

    assert alive == {}
    assert old_stop.is_set()
    assert new_stop.is_set()


def test_frame_observer_failure_does_not_reconnect_healthy_source() -> None:
    backend = FakeBackend([[
        np.ones((10, 20, 3), dtype=np.uint8),
        np.full((10, 20, 3), 2, dtype=np.uint8),
    ]])

    def fail(_frame: CapturedFrame) -> None:
        raise RuntimeError("consumer failed")

    service = _service(backend, frame_observer=fail)
    assert service.start()
    _wait_until(
        lambda: service.status()["capture_stats"]["live"]["observer_errors"] >= 2
    )
    status = service.status()
    service.request_stop()
    service.wait_stopped(1.0)

    assert status["capture_stats"]["live"]["frames_received"] >= 2
    assert status["capture_stats"]["live"]["starts"] == 1


def test_source_started_notification_precedes_first_frame() -> None:
    order: list[str] = []
    backend = FakeBackend([[np.ones((10, 20, 3), dtype=np.uint8)]])
    service = _service(
        backend,
        source_started_observer=lambda source: order.append(f"start:{source}"),
        frame_observer=lambda frame: order.append(f"frame:{frame.source}"),
    )

    assert service.start()
    _wait_until(lambda: len(order) >= 2)
    service.request_stop()
    service.wait_stopped(1.0)

    assert order[:2] == ["start:live", "frame:live"]


def test_start_rejects_lingering_thread_from_previous_stop() -> None:
    service = _service()
    lingering = threading.Thread(target=lambda: time.sleep(0.1))
    lingering_stop = threading.Event()
    lingering.start()
    with service._lock:
        service._threads["live"] = lingering
        service._all_threads[lingering] = ("live", lingering_stop)

    try:
        try:
            service.start()
        except RuntimeError as error:
            assert "sources are stopping: live" in str(error)
        else:
            raise AssertionError("capture restart should reject a lingering source")
    finally:
        lingering.join(timeout=1.0)


def test_frame_is_not_published_when_stop_wins_after_native_read() -> None:
    service = _service()
    with service._lock:
        service._stop.clear()
    source_stop = threading.Event()
    source_stop.set()

    stored = service._publish_frame(
        "live",
        np.ones((10, 20, 3), dtype=np.uint8),
        source_stop,
    )

    assert stored is False
    assert service.latest("live") is None


def test_frame_observer_runs_without_capture_lock_held() -> None:
    observed_status: list[dict[str, object]] = []
    service: CameraCaptureService

    def observe(_frame: CapturedFrame) -> None:
        observed_status.append(service.status())

    service = _service(frame_observer=observe)
    with service._lock:
        service._stop.clear()

    assert service._publish_frame(
        "live", np.ones((10, 20, 3), dtype=np.uint8)
    )
    assert observed_status


def test_repeated_start_stop_leaves_no_capture_generations() -> None:
    service = _service(FakeBackend([
        [np.ones((10, 20, 3), dtype=np.uint8)],
        [np.ones((10, 20, 3), dtype=np.uint8)],
    ]))

    for cycle in range(2):
        assert service.start()
        _wait_until(
            lambda: service.status()["capture_stats"]["live"]["starts"]
            >= cycle + 1
        )
        service.request_stop()
        assert service.wait_stopped(1.0) == {}
        with service._lock:
            assert not service._all_threads


def test_backend_error_redacts_credentials_in_status() -> None:
    service = CameraCaptureService(
        camera_id="gate",
        source_url=lambda _source: "rtsp://admin:secret@camera/live",
        backend=FailingOpenBackend(),
        retry_initial_seconds=0.01,
        retry_max_seconds=0.02,
    )

    assert service.start()
    _wait_until(lambda: bool(service.status()["last_error"]))
    error = str(service.status()["last_error"])
    service.request_stop()
    service.wait_stopped(1.0)

    assert "secret" not in error
    assert "rtsp://admin:***@camera/live" in error


def test_close_rejects_active_capture_thread() -> None:
    service = _service()
    assert service.start()

    try:
        try:
            service.close()
        except RuntimeError as error:
            assert "capture sources still running" in str(error)
        else:
            raise AssertionError("active capture close should fail")
    finally:
        service.request_stop()
        service.wait_stopped(1.0)


def test_latest_frame_store_is_bounded_to_one_frame_per_source() -> None:
    service = _service()
    with service._lock:
        service._stop.clear()

    for value in range(20):
        service._publish_frame(
            "live",
            np.full((10, 20, 3), value, dtype=np.uint8),
        )

    with service._lock:
        assert list(service._frames) == ["live"]
    latest = service.latest("live")
    assert latest is not None
    assert int(latest.image[0, 0, 0]) == 19
