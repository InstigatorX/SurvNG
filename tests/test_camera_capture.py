from __future__ import annotations

import subprocess
import threading
import time
from collections import deque

import numpy as np
import pytest

from survng.app.camera_capture import (
    CAPTURE_FRAME_MAX_BYTES,
    CameraCaptureService,
    CaptureHandle,
    CaptureOpenLimiter,
    CapturedFrame,
    FfmpegCaptureOptions,
    FfmpegCaptureBackend,
    FfmpegCaptureHandle,
    _bmp_frame_size,
    _decode_capture_bmp,
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

    def open(self, handle, source_url, cancelled, *, open_timeout_ms=None) -> bool:
        self.open_calls += 1
        if cancelled():
            return False
        handle.opened = True
        return True


class FailingOpenBackend(FakeBackend):
    def open(self, handle, source_url, cancelled, *, open_timeout_ms=None) -> bool:
        raise RuntimeError(f"unable to open {source_url}")


class ClosedHandle(FakeHandle):
    def set_buffer_size(self, size: int) -> None:
        raise AssertionError("buffer configuration must not run on a closed handle")


class ClosedBackend(FakeBackend):
    def create_handle(self) -> CaptureHandle:
        handle = ClosedHandle()
        self.handles.append(handle)
        return handle

    def open(self, handle, source_url, cancelled, *, open_timeout_ms=None) -> bool:
        self.open_calls += 1
        return False


class ScriptedOpenBackend(FakeBackend):
    def __init__(
        self,
        results: list[bool],
        frame_batches: list[list[np.ndarray]],
    ) -> None:
        super().__init__(frame_batches)
        self.results = deque(results)
        self.open_timeouts: list[int | None] = []

    def open(self, handle, source_url, cancelled, *, open_timeout_ms=None) -> bool:
        self.open_calls += 1
        self.open_timeouts.append(open_timeout_ms)
        opened = self.results.popleft() if self.results else True
        handle.opened = opened and not cancelled()
        return handle.opened


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
    status = service.status()["capture_stats"]["live"]
    assert status["frame_copy_count"] == 2
    assert status["frame_copy_bytes"] == source.nbytes * 2
    assert service.frame_ready("live") is True
    now[0] = 111.0
    assert service.latest("live") is None
    assert service.frame_ready("live") is False


def test_latest_copy_does_not_hold_capture_lock_during_image_copy() -> None:
    copy_entered = threading.Event()
    release_copy = threading.Event()

    class SlowCopyArray(np.ndarray):
        def copy(self, *args, **kwargs):
            copy_entered.set()
            assert release_copy.wait(1.0)
            return super().copy(*args, **kwargs)

    service = _service()
    with service._lock:
        service._stop.clear()
    slow = np.ones((20, 30, 3), dtype=np.uint8).view(SlowCopyArray)
    assert service._publish_frame("live", slow)
    reader = threading.Thread(target=service.latest, args=("live",))
    reader.start()
    assert copy_entered.wait(1.0)

    publish_started = time.monotonic()
    assert service._publish_frame(
        "live",
        np.zeros((20, 30, 3), dtype=np.uint8),
    )
    assert time.monotonic() - publish_started < 0.1

    release_copy.set()
    reader.join(timeout=1.0)
    assert not reader.is_alive()
    status = service.status()["capture_stats"]["live"]
    assert status["frame_copy_count"] == 1
    assert status["frame_copy_bytes"] == slow.nbytes


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
        lambda: service.status()["capture_stats"]["live"]["observer_errors"] >= 1
    )
    status = service.status()
    service.request_stop()
    service.wait_stopped(1.0)

    assert status["capture_stats"]["live"]["frames_received"] >= 2
    assert status["capture_stats"]["live"]["starts"] == 1
    assert status["capture_stats"]["live"]["frame_copy_count"] == 0
    assert status["capture_stats"]["live"]["frame_copy_bytes"] == 0
    assert status["capture_stats"]["live"]["frame_transfer_count"] >= 2
    assert status["capture_stats"]["live"]["frame_transfer_bytes"] >= 1200
    assert status["capture_stats"]["live"]["observer_calls"] >= 1
    assert status["capture_stats"]["live"]["observer_submissions"] >= 2
    assert status["capture_stats"]["live"]["observer_p99_ms"] >= 0.0


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
    assert service._observer_dispatch is not None
    service._observer_dispatch.start()

    assert service._publish_frame(
        "live", np.ones((10, 20, 3), dtype=np.uint8)
    )
    _wait_until(lambda: bool(observed_status))
    service.request_stop()
    assert service.wait_stopped(1.0) == {}
    assert observed_status


def test_frame_observer_receives_stable_stored_frame_ownership() -> None:
    observed: list[CapturedFrame] = []
    service = _service(frame_observer=observed.append)
    with service._lock:
        service._stop.clear()
    assert service._observer_dispatch is not None
    service._observer_dispatch.start()
    source = np.ones((10, 20, 3), dtype=np.uint8)

    assert service._publish_frame("live", source)
    _wait_until(lambda: len(observed) == 1)

    assert len(observed) == 1
    assert int(observed[0].image[0, 0, 0]) == 1
    assert observed[0].image is source
    assert not observed[0].image.flags.writeable
    with pytest.raises(ValueError):
        source.fill(9)
    with service._lock:
        assert observed[0].image is service._frames["live"].image
    independent = service.latest("live")
    assert independent is not None
    assert independent.image.flags.writeable
    independent.image.fill(7)
    assert int(observed[0].image[0, 0, 0]) == 1
    service.request_stop()
    assert service.wait_stopped(1.0) == {}


def test_blocked_observer_never_holds_capture_source_thread_open() -> None:
    entered = threading.Event()
    release = threading.Event()

    def observe(_frame: CapturedFrame) -> None:
        entered.set()
        assert release.wait(1.0)

    backend = FakeBackend([[np.ones((10, 20, 3), dtype=np.uint8)]])
    service = _service(backend, frame_observer=observe)
    assert service.start()
    assert entered.wait(1.0)

    service.request_stop()
    alive = service.wait_stopped(0.02)

    assert set(alive) == {"observer"}
    assert service.status()["live_running"] is False

    release.set()
    assert service.wait_stopped(1.0) == {}
    service.close()


def test_observer_mailbox_replaces_backlog_with_latest_frame() -> None:
    observed: list[int] = []
    entered = threading.Event()
    release = threading.Event()

    def observe(frame: CapturedFrame) -> None:
        observed.append(frame.sequence)
        if len(observed) == 1:
            entered.set()
            assert release.wait(1.0)

    service = _service(frame_observer=observe)
    with service._lock:
        service._stop.clear()
    assert service._observer_dispatch is not None
    service._observer_dispatch.start()

    assert service._publish_frame("live", np.full((2, 2, 3), 1, dtype=np.uint8))
    assert entered.wait(1.0)
    assert service._publish_frame("live", np.full((2, 2, 3), 2, dtype=np.uint8))
    assert service._publish_frame("live", np.full((2, 2, 3), 3, dtype=np.uint8))
    assert service.status()["capture_stats"]["live"]["observer_frames_replaced"] == 1

    release.set()
    _wait_until(lambda: len(observed) == 2)
    assert observed == [1, 3]
    service.request_stop()
    assert service.wait_stopped(1.0) == {}


def test_observer_mailbox_preserves_latest_frame_for_each_source() -> None:
    observed: list[tuple[str, int]] = []
    entered = threading.Event()
    release = threading.Event()

    def observe(frame: CapturedFrame) -> None:
        observed.append((frame.source, frame.sequence))
        if len(observed) == 1:
            entered.set()
            assert release.wait(1.0)

    service = _service(frame_observer=observe)
    with service._lock:
        service._stop.clear()
    assert service._observer_dispatch is not None
    service._observer_dispatch.start()

    assert service._publish_frame("live", np.zeros((2, 2, 3), dtype=np.uint8))
    assert entered.wait(1.0)
    assert service._publish_frame("live", np.ones((2, 2, 3), dtype=np.uint8))
    assert service._publish_frame("main", np.ones((2, 2, 3), dtype=np.uint8))

    release.set()
    _wait_until(lambda: len(observed) == 3)
    assert [source for source, _sequence in observed] == ["live", "live", "main"]
    stats = service.status()["capture_stats"]
    assert stats["live"]["observer_frames_replaced"] == 0
    assert stats["main"]["observer_frames_replaced"] == 0
    service.request_stop()
    assert service.wait_stopped(1.0) == {}


def test_observer_mailbox_can_restart_after_clean_stop() -> None:
    observed: list[int] = []
    service = _service(frame_observer=lambda frame: observed.append(frame.sequence))
    assert service._observer_dispatch is not None

    for generation in range(2):
        with service._lock:
            service._stop.clear()
        service._observer_dispatch.start()
        assert service._publish_frame(
            "live",
            np.full((2, 2, 3), generation, dtype=np.uint8),
        )
        _wait_until(lambda: len(observed) == generation + 1)
        service._observer_dispatch.request_stop()
        assert service._observer_dispatch.wait_stopped(1.0)

    assert observed == [1, 2]


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
        _wait_until(lambda: service.latest("live") is not None)
        frame = service.latest("live")
        assert frame is not None
        assert frame.generation == cycle + 1
        service.request_stop()
        assert service.wait_stopped(1.0) == {}
        with service._lock:
            assert not service._all_threads


def test_clean_stop_clears_transient_error_and_fps_window() -> None:
    service = _service()
    with service._lock:
        service._stop.clear()
    service._publish_frame("live", np.ones((10, 20, 3), dtype=np.uint8))
    service._publish_frame("live", np.ones((10, 20, 3), dtype=np.uint8))
    service._set_error("live", "stream read failed")

    service.request_stop()
    assert service.wait_stopped(1.0) == {}
    status = service.status()

    assert status["last_error"] == ""
    assert status["capture_stats"]["live"]["fps"] == 0.0
    assert service.latest("live") is None


def test_unknown_capture_source_is_rejected_instead_of_using_live() -> None:
    service = _service()

    try:
        service.latest("mian")
    except ValueError as error:
        assert "unsupported camera capture source" in str(error)
    else:
        raise AssertionError("misspelled capture source should not select live")


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


def test_failed_open_does_not_configure_closed_native_handle() -> None:
    service = _service(ClosedBackend())

    assert service.start()
    _wait_until(
        lambda: service.status()["capture_stats"]["live"]["open_failures"] >= 1
    )
    service.request_stop()
    service.wait_stopped(1.0)

    assert service.status()["capture_stats"]["live"]["open_failures"] >= 1


def test_live_reconnect_escalates_open_deadline_then_resets_after_frame() -> None:
    frame = np.ones((10, 20, 3), dtype=np.uint8)
    backend = ScriptedOpenBackend(
        [False, True, True],
        [[], [frame], [frame]],
    )
    service = _service(
        backend,
        initial_open_timeout_ms=3000,
        reconnect_open_timeout_ms=10000,
    )

    assert service.start()
    _wait_until(lambda: len(backend.open_timeouts) >= 3)
    status = service.status()["capture_stats"]["live"]
    service.request_stop()
    assert service.wait_stopped(1.0) == {}

    assert backend.open_timeouts[:3] == [3000, 10000, 3000]
    assert status["open_timeout_escalations"] >= 1
    assert status["last_open_timeout_ms"] == 3000


def test_live_reconnect_escalates_when_open_succeeds_without_a_frame() -> None:
    frame = np.ones((10, 20, 3), dtype=np.uint8)
    backend = ScriptedOpenBackend(
        [True, True],
        [[], [frame]],
    )
    service = _service(
        backend,
        initial_open_timeout_ms=3000,
        reconnect_open_timeout_ms=10000,
    )

    assert service.start()
    _wait_until(
        lambda: (
            len(backend.open_timeouts) >= 2
            and service.status()["capture_stats"]["live"]["frames_received"] >= 1
        )
    )
    service.request_stop()
    assert service.wait_stopped(1.0) == {}

    assert backend.open_timeouts[:2] == [3000, 10000]


def test_live_recovers_after_relay_restart_without_a_persistent_consumer() -> None:
    frame = np.ones((10, 20, 3), dtype=np.uint8)
    backend = ScriptedOpenBackend(
        [True, False, True],
        [[frame], [], [frame]],
    )
    service = _service(
        backend,
        initial_open_timeout_ms=3000,
        reconnect_open_timeout_ms=10000,
    )

    assert service.start()
    _wait_until(
        lambda: (
            len(backend.open_timeouts) >= 3
            and service.status()["capture_stats"]["live"]["frames_received"] >= 2
        )
    )
    status = service.status()["capture_stats"]["live"]
    service.request_stop()
    assert service.wait_stopped(1.0) == {}

    assert backend.open_timeouts[:3] == [3000, 3000, 10000]
    assert status["starts"] >= 2
    assert status["reconnects"] >= 1
    assert status["open_failures"] >= 1
    assert status["open_timeout_escalations"] >= 1


def test_ffmpeg_backend_uses_configured_transport_and_policy_rate() -> None:
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(
            ffmpeg_path="custom-ffmpeg",
            rtsp_transport="udp",
            frame_rate=lambda: 5.0,
        ),
    )

    command = backend._command("rtsp://camera/live")

    assert command[:5] == ["custom-ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    assert ["-rtsp_transport", "udp"] == command[command.index("-rtsp_transport") : command.index("-rtsp_transport") + 2]
    assert command[command.index("-threads:v") + 1] == "1"
    assert "bmp" in command
    capture_filter = command[command.index("-vf") + 1]
    assert "prev_selected_t" in capture_filter
    assert "0.200000" in capture_filter
    assert command[command.index("-fps_mode") + 1] == "vfr"
    assert command[-5:] == ["-f", "image2pipe", "-vcodec", "bmp", "pipe:1"]


def test_ffmpeg_backend_warns_once_without_logging_url_credentials(caplog) -> None:
    backend = FfmpegCaptureBackend(CaptureOpenLimiter(1))

    backend._command("rtsp://admin:first-secret@camera/live")
    backend._command("rtsp://admin:second-secret@camera/main")

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "process arguments" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "camera" in warnings[0]
    assert "admin" not in warnings[0]
    assert "secret" not in warnings[0]


def test_ffmpeg_capture_close_waits_for_stderr_drain_before_closing_stream() -> None:
    class Stream:
        def __init__(self) -> None:
            self.eof = threading.Event()
            self.read_started = threading.Event()
            self.reader_finished = threading.Event()
            self.closed = False

        def read(self, _size: int) -> bytes:
            self.read_started.set()
            assert self.eof.wait(1.0)
            self.reader_finished.set()
            return b""

        def close(self) -> None:
            assert self.reader_finished.is_set()
            self.closed = True

    class Process:
        def __init__(self, stderr: Stream) -> None:
            self.stderr = stderr
            self.stdout = Stream()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.stderr.eof.set()
            self.stdout.reader_finished.set()
            return 0

    handle = FfmpegCaptureHandle(read_timeout_ms=1000)
    stderr = Stream()
    process = Process(stderr)
    handle._process = process  # type: ignore[assignment]
    handle._stderr_thread = threading.Thread(target=handle._drain_stderr)
    handle._stderr_thread.start()
    assert stderr.read_started.wait(1.0)

    handle.close()

    assert stderr.closed
    assert process.stdout.closed


def test_ffmpeg_capture_close_is_bounded_when_process_and_stderr_reader_hang(
    monkeypatch, caplog
) -> None:
    class Stream:
        def __init__(self) -> None:
            self.release = threading.Event()
            self.read_started = threading.Event()
            self.closed = False

        def read(self, _size: int) -> bytes:
            self.read_started.set()
            self.release.wait()
            return b""

        def close(self) -> None:
            self.closed = True

    class Process:
        def __init__(self, stderr: Stream) -> None:
            self.stderr = stderr
            self.stdout = Stream()
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            raise subprocess.TimeoutExpired("ffmpeg", 0)

    monkeypatch.setattr("survng.app.camera_capture.CAPTURE_SHUTDOWN_WAIT_SECONDS", 0.01)
    monkeypatch.setattr("survng.app.camera_capture.CAPTURE_STDERR_JOIN_SECONDS", 0.01)
    handle = FfmpegCaptureHandle(read_timeout_ms=1000)
    stderr = Stream()
    process = Process(stderr)
    handle._process = process  # type: ignore[assignment]
    handle._stderr_thread = threading.Thread(target=handle._drain_stderr, daemon=True)
    handle._stderr_thread.start()
    assert stderr.read_started.wait(1.0)

    started = time.monotonic()
    handle.close()

    assert time.monotonic() - started < 0.2
    assert process.wait_calls == 2
    assert process.stdout.closed
    assert not stderr.closed
    messages = [record.getMessage() for record in caplog.records]
    assert any("did not exit after kill" in message for message in messages)
    assert any("stderr reader did not stop" in message for message in messages)

    stderr.release.set()


def test_capture_bmp_limit_accepts_8k_frames_but_remains_bounded() -> None:
    header = bytearray(14)
    header[:2] = b"BM"
    eight_k_frame_size = 7680 * 4320 * 3 + 54
    header[2:6] = eight_k_frame_size.to_bytes(4, "little")

    assert _bmp_frame_size(header) == eight_k_frame_size

    header[2:6] = (CAPTURE_FRAME_MAX_BYTES + 1).to_bytes(4, "little")
    with pytest.raises(RuntimeError, match="invalid BMP size"):
        _bmp_frame_size(header)


def test_capture_bmp_decode_preserves_bgr_pixels() -> None:
    # One bottom-up 1×1 24-bit BMP with BGR payload (20, 40, 200).
    encoded = bytearray(58)
    encoded[:2] = b"BM"
    encoded[2:6] = (58).to_bytes(4, "little")
    encoded[10:14] = (54).to_bytes(4, "little")
    encoded[14:18] = (40).to_bytes(4, "little")
    encoded[18:22] = (1).to_bytes(4, "little", signed=True)
    encoded[22:26] = (1).to_bytes(4, "little", signed=True)
    encoded[26:28] = (1).to_bytes(2, "little")
    encoded[28:30] = (24).to_bytes(2, "little")
    encoded[54:] = bytes((20, 40, 200, 0))
    frame = _decode_capture_bmp(encoded)

    assert frame.shape == (1, 1, 3)
    assert frame[0, 0].tolist() == [20, 40, 200]


def test_capture_failure_includes_redacted_ffmpeg_detail() -> None:
    class FailedHandle(FakeHandle):
        def error_detail(self) -> str:
            return (
                "FFmpeg exited from signal 11: "
                "rtsp://admin:secret@camera/live failed"
            )

    reason = CameraCaptureService._capture_failure_reason(
        "stream read failed",
        FailedHandle(),
    )

    assert "signal 11" in reason
    assert "secret" not in reason
    assert "rtsp://admin:***@camera/live" in reason


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
