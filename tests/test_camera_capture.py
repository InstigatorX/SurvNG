from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from unittest.mock import patch

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
    _parse_showinfo_geometry,
)
from survng.app.ffmpeg_hw import CaptureDecodePlan


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


def test_status_reports_decode_plan_outside_numeric_stats() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    backend = FakeBackend([[frame]])

    def open_with_plan(handle, source_url, cancelled, *, open_timeout_ms=None):
        del source_url, cancelled, open_timeout_ms
        handle.decode_plan = "cpu"
        handle.opened = True
        return True

    backend.open = open_with_plan
    service = _service(backend)

    assert service.start()
    _wait_until(lambda: service.status()["capture_stats"]["live"]["frames_received"] >= 1)
    status = service.status()
    service.request_stop()
    assert service.wait_stopped(1.0) == {}

    assert status["decode_plan"] == {"live": "cpu", "main": ""}
    assert "decode_plan" not in status["capture_stats"]["live"]


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

    assert command[:5] == ["custom-ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info"]
    assert "-nostats" in command
    assert ["-filter_threads", "1"] == command[
        command.index("-filter_threads") : command.index("-filter_threads") + 2
    ]
    assert ["-rtsp_transport", "udp"] == command[command.index("-rtsp_transport") : command.index("-rtsp_transport") + 2]
    assert command[command.index("-threads:v") + 1] == "1"
    assert "bmp" not in command
    capture_filter = command[command.index("-vf") + 1]
    assert "prev_selected_t" in capture_filter
    assert "format=bgr24" in capture_filter
    assert "showinfo@capture=checksum=0" in capture_filter
    assert "0.200000" in capture_filter
    assert command[command.index("-fps_mode") + 1] == "vfr"
    assert command[-3:] == ["-f", "rawvideo", "pipe:1"]
    assert "-hwaccel" not in command
    assert "hwdownload" not in capture_filter


def test_qsv_capture_command_downloads_after_select() -> None:
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(frame_rate=lambda: 5.0),
    )
    plan = CaptureDecodePlan(
        "qsv",
        (
            "-qsv_device",
            "/dev/dri/renderD128",
            "-hwaccel",
            "qsv",
            "-hwaccel_output_format",
            "qsv",
        ),
        ("hwdownload", "format=nv12"),
    )

    command = backend._command("rtsp://camera/live", plan)

    input_at = command.index("-i")
    device_at = command.index("-qsv_device")
    assert command[device_at:input_at] == [
        "-qsv_device",
        "/dev/dri/renderD128",
        "-hwaccel",
        "qsv",
        "-hwaccel_output_format",
        "qsv",
    ]
    video_filter = command[command.index("-vf") + 1]
    assert video_filter.index("select=") < video_filter.index("hwdownload,format=nv12")
    assert video_filter.index("hwdownload,format=nv12") < video_filter.index("format=bgr24")
    assert video_filter.index("format=bgr24") < video_filter.index(
        "showinfo@capture=checksum=0"
    )
    assert command[-3:] == ["-f", "rawvideo", "pipe:1"]
    assert "-hwaccel" not in command[input_at:]


def test_ffmpeg_backend_bmp_transport_remains_available() -> None:
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(frame_transport="bmp", frame_rate=lambda: 5.0),
    )

    command = backend._command("rtsp://camera/live")

    assert command[:5] == ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    assert "showinfo@capture" not in command[command.index("-vf") + 1]
    assert command[-5:] == ["-f", "image2pipe", "-vcodec", "bmp", "pipe:1"]


def test_ffmpeg_backend_rejects_unknown_frame_transport() -> None:
    with pytest.raises(ValueError, match="frame_transport"):
        FfmpegCaptureBackend(
            CaptureOpenLimiter(1),
            FfmpegCaptureOptions(frame_transport="mjpeg"),
        )


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


def _showinfo_line(index: int, width: int, height: int) -> bytes:
    return (
        f"[showinfo@capture @ 0x1] n: {index:3d} pts: 0 pts_time:0 "
        f"fmt:bgr24 s:{width}x{height} i:P iskey:1 type:I \n"
    ).encode()


def test_showinfo_geometry_ignores_unrelated_stderr() -> None:
    assert _parse_showinfo_geometry(_showinfo_line(1, 32, 24)) == (1, 32, 24)
    assert _parse_showinfo_geometry(
        b"[showinfo@capture @ 0x1] color_range:pc color_space:gbr\n"
    ) is None
    assert _parse_showinfo_geometry(
        b"frame=    1 fps=0.0 q=-0.0 size=      18kB time=00:00:00.20\n"
    ) is None


def test_open_failure_names_the_phase_and_skips_showinfo() -> None:
    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")
    handle._stderr.extend(_showinfo_line(10, 3840, 2160))
    handle._stderr.extend(
        b"[showinfo@capture @ 0x1] color_range:pc color_space:gbr\n"
        b"Error opening input: immediate exit\n"
    )

    assert handle.error_detail() == "timed out connecting: Error opening input: immediate exit"

    handle._note_showinfo_line(_showinfo_line(0, 2, 2))
    assert handle.error_detail() == "timed out reading the first frame: Error opening input: immediate exit"
    assert "showinfo@capture" not in handle.error_detail()
    assert "3840x2160" not in handle.error_detail()

    handle._open_phase = "delivered"
    handle._note_showinfo_line(_showinfo_line(1, 2, 2))
    assert handle.error_detail() == "timed out reading a frame: Error opening input: immediate exit"

    handle._process = type("Process", (), {"poll": lambda self: 1})()
    assert handle.error_detail() == "FFmpeg exited with status 1: Error opening input: immediate exit"
    assert "showinfo@capture" not in handle.error_detail()

    handle._transport_failure = "FFmpeg capture geometry queue overflow"
    assert handle.error_detail().startswith("FFmpeg capture geometry queue overflow")


def test_prefetch_reads_the_frame_body_after_the_connect_budget() -> None:
    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")
    read_fd, write_fd = os.pipe()
    stdout = os.fdopen(read_fd, "rb", buffering=0)
    payload = bytes((4, 5, 6)) * 4

    def write_late() -> None:
        time.sleep(0.05)
        os.write(write_fd, payload)
        os.close(write_fd)

    writer = threading.Thread(target=write_late)
    handle._process = type("Process", (), {"stdout": stdout, "poll": lambda self: None})()
    handle._geometries.put((2, 2))
    writer.start()
    try:
        assert handle.prefetch(1, lambda: False, frame_timeout_ms=1000)
        ok, frame = handle.read()
    finally:
        writer.join(timeout=1)
        stdout.close()
        if writer.is_alive():
            os.close(write_fd)

    assert ok and frame is not None
    assert frame.shape == (2, 2, 3)
    assert bytes(frame.reshape(-1)) == payload


def test_raw_capture_reads_owned_frames_across_resolution_change() -> None:
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    stdout = os.fdopen(stdout_read, "rb")
    stderr = os.fdopen(stderr_read, "rb")
    first = bytes((10, 20, 30)) * (2 * 2)
    second = bytes((1, 2, 3))

    def write_frames() -> None:
        os.write(stderr_write, _showinfo_line(0, 2, 2))
        os.write(
            stderr_write,
            b"[showinfo@capture @ 0x1] color_range:pc color_space:gbr\n",
        )
        os.write(stdout_write, first)
        os.write(stderr_write, _showinfo_line(1, 1, 1))
        os.write(stdout_write, second)
        os.close(stdout_write)
        os.close(stderr_write)

    class Process:
        def __init__(self) -> None:
            self.stdout = stdout
            self.stderr = stderr
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")
    handle._process = Process()  # type: ignore[assignment]
    writer = threading.Thread(target=write_frames)
    handle._stderr_thread = threading.Thread(target=handle._drain_stderr)
    handle._stderr_thread.start()
    writer.start()
    try:
        ok, wide = handle.read()
        assert ok and wide is not None
        ok, narrow = handle.read()
        assert ok and narrow is not None
    finally:
        writer.join(1.0)
        handle._stderr_thread.join(1.0)
        stdout.close()
        stderr.close()

    assert wide.shape == (2, 2, 3)
    assert narrow.shape == (1, 1, 3)
    assert wide[0, 0].tolist() == [10, 20, 30]
    assert narrow[0, 0].tolist() == [1, 2, 3]
    wide.setflags(write=False)
    narrow.setflags(write=False)
    assert not np.shares_memory(wide, narrow)


def test_raw_capture_fails_closed_when_showinfo_index_jumps() -> None:
    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")

    handle._note_showinfo_line(_showinfo_line(0, 2, 2))
    handle._note_showinfo_line(_showinfo_line(2, 2, 2))

    assert handle._geometries.qsize() == 1
    assert handle._transport_failed.is_set()
    assert "lost raw frame alignment" in handle.error_detail()


def test_raw_capture_rejects_oversized_geometry_without_allocating() -> None:
    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")
    huge = int((CAPTURE_FRAME_MAX_BYTES // 3) ** 0.5) + 2

    handle._note_showinfo_line(_showinfo_line(0, huge, huge))

    assert handle._geometries.empty()
    assert handle._transport_failed.is_set()


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


def test_ffmpeg_raw_capture_returns_bgr_frames(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    video = tmp_path / "sample.mp4"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=32x24:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=False,
    )
    if encoded.returncode != 0:
        pytest.skip("ffmpeg could not encode the capture fixture")
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(frame_rate=lambda: 5.0, read_timeout_ms=3000),
    )
    handle = backend.create_handle()
    assert isinstance(handle, FfmpegCaptureHandle)
    opened = False
    try:
        opened = backend.open(handle, str(video), lambda: False, open_timeout_ms=5000)
        assert opened
        ok, frame = handle.read()
        assert ok and frame is not None
        assert frame.shape == (24, 32, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous
        frame.setflags(write=False)
        assert int(frame[:, :, 2].mean()) > int(frame[:, :, 0].mean())
    finally:
        if opened:
            handle.close()


def _encode_capture_fixture(path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=32x24:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=False,
    )
    return encoded.returncode == 0


def test_qsv_capture_falls_back_to_cpu_when_device_open_fails(tmp_path, caplog) -> None:
    video = tmp_path / "sample.mp4"
    if not _encode_capture_fixture(video):
        pytest.skip("ffmpeg could not encode the capture fixture")
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(
            frame_rate=lambda: 5.0,
            read_timeout_ms=3000,
            hardware_acceleration="qsv",
        ),
    )
    handle = backend.create_handle()
    assert isinstance(handle, FfmpegCaptureHandle)
    caplog.set_level(logging.WARNING, logger="survng.app.camera_capture")
    try:
        with (
            patch("survng.app.ffmpeg_hw.render_device_available", return_value=True),
            patch(
                "survng.app.ffmpeg_hw.ffmpeg_hwaccels",
                return_value=frozenset({"qsv"}),
            ),
            patch(
                "survng.app.ffmpeg_hw.dri_render_device",
                return_value="/dev/dri/renderD999",
            ),
        ):
            opened = backend.open(
                handle,
                str(video),
                lambda: False,
                open_timeout_ms=8000,
            )
        assert opened
        assert handle.decode_plan == "cpu"
        ok, frame = handle.read()
        assert ok and frame is not None
        assert frame.shape == (24, 32, 3)
        assert frame.flags.c_contiguous
    finally:
        handle.close()

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "decode plan qsv" in record.getMessage()
    ]
    assert warnings
    assert "cpu" in warnings[0]


def test_auto_capture_without_render_node_stays_on_cpu(tmp_path) -> None:
    video = tmp_path / "sample.mp4"
    if not _encode_capture_fixture(video):
        pytest.skip("ffmpeg could not encode the capture fixture")
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(
            frame_rate=lambda: 5.0,
            read_timeout_ms=3000,
            hardware_acceleration="auto",
        ),
    )
    handle = backend.create_handle()
    assert isinstance(handle, FfmpegCaptureHandle)
    try:
        with (
            patch("survng.app.ffmpeg_hw.render_device_available", return_value=False),
            patch("survng.app.ffmpeg_hw.ffmpeg_hwaccels") as probe,
        ):
            opened = backend.open(
                handle,
                str(video),
                lambda: False,
                open_timeout_ms=8000,
            )
            plans = backend._decode_plans()
            command = backend._command(str(video), plans[0])
        probe.assert_not_called()
        assert opened
        assert handle.decode_plan == "cpu"
        assert [plan.name for plan in plans] == ["cpu"]
        assert "-hwaccel" not in command
        assert "hwdownload" not in command[command.index("-vf") + 1]
    finally:
        handle.close()


def test_replaced_decoder_drops_stderr_from_the_previous_process() -> None:
    handle = FfmpegCaptureHandle(read_timeout_ms=1000, frame_transport="rawvideo")
    read_fd, write_fd = os.pipe()
    stderr = os.fdopen(read_fd, "rb", buffering=0)
    handle._process = type(
        "Process",
        (),
        {"stderr": stderr, "stdout": None, "poll": lambda self: None},
    )()
    handle._stream_generation = 7
    reader = threading.Thread(target=handle._drain_stderr)
    reader.start()
    try:
        os.write(write_fd, _showinfo_line(0, 2, 2))
        assert handle._geometries.get(timeout=1) == (2, 2)
        handle._reset_stream_state()
        os.write(write_fd, _showinfo_line(0, 8, 8))
        deadline = time.monotonic() + 1
        while reader.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        os.close(write_fd)
        reader.join(timeout=1)
        stderr.close()

    assert not reader.is_alive()
    assert handle._geometries.empty()
    assert handle._next_showinfo_index == 0
    assert handle._open_phase == "connecting"
    assert handle._stream_generation == 8


def test_capture_start_discards_bytes_from_the_previous_decoder() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    handle = FfmpegCaptureHandle(read_timeout_ms=5000, frame_transport="rawvideo")
    handle._prefetched = np.zeros((8, 8, 3), dtype=np.uint8)
    handle._buffer.extend(b"stale")
    handle._note_showinfo_line(_showinfo_line(4, 8, 8))
    handle._transport_failure = "old failure"
    handle._transport_failed.set()
    handle.start(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "info",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=2x2:rate=5:duration=0.4",
            "-vf",
            "format=bgr24,showinfo@capture=checksum=0",
            "-fps_mode",
            "vfr",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        decode_plan="cpu",
    )
    try:
        ok, frame = handle.read()
        assert ok and frame is not None
        assert frame.shape == (2, 2, 3)
        assert int(frame[:, :, 0].mean()) > int(frame[:, :, 2].mean())
        assert handle.decode_plan == "cpu"
        assert handle._transport_failure == ""
        assert not handle._transport_failed.is_set()
        assert handle._next_showinfo_index >= 1
    finally:
        handle.close()


def test_ffmpeg_raw_capture_reads_when_showinfo_runs_ahead(tmp_path) -> None:
    """A short file can emit more geometries than the queue before prefetch."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    video = tmp_path / "burst.mp4"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=15:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=False,
    )
    if encoded.returncode != 0:
        pytest.skip("ffmpeg could not encode the capture fixture")
    backend = FfmpegCaptureBackend(
        CaptureOpenLimiter(1),
        FfmpegCaptureOptions(frame_rate=lambda: 10.0, read_timeout_ms=3000),
    )
    handle = backend.create_handle()
    assert isinstance(handle, FfmpegCaptureHandle)
    opened = False
    frames = 0
    try:
        opened = backend.open(handle, str(video), lambda: False, open_timeout_ms=5000)
        assert opened
        assert not handle._transport_failed.is_set()
        while True:
            ok, frame = handle.read()
            if not ok or frame is None:
                break
            assert frame.shape == (360, 640, 3)
            frames += 1
    finally:
        if opened:
            handle.close()

    # 640x360 is larger than the stdout pipe. This used to deadlock before
    # the first frame because showinfo sat behind a blocking stderr read.
    assert frames >= 8
    assert "lost raw frame alignment" not in handle.error_detail()


def test_ffmpeg_raw_capture_follows_resolution_change() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    handle = FfmpegCaptureHandle(read_timeout_ms=5000, frame_transport="rawvideo")
    handle.start(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "info",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=0.5",
            "-vf",
            (
                "scale=w='if(lt(n,2),64,32)':h='if(lt(n,2),48,24)':eval=frame,"
                "format=bgr24,showinfo@capture=checksum=0"
            ),
            "-fps_mode",
            "vfr",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    shapes: list[tuple[int, ...]] = []
    try:
        for _ in range(5):
            ok, frame = handle.read()
            assert ok and frame is not None
            shapes.append(frame.shape)
            frame.setflags(write=False)
    finally:
        handle.close()

    # FFmpeg 8.1 evaluates scale `n` so lt(n,2) keeps only the first frame large.
    assert shapes == [(48, 64, 3), (24, 32, 3), (24, 32, 3), (24, 32, 3), (24, 32, 3)]
