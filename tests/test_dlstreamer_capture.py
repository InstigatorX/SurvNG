from __future__ import annotations

import os
import io
import sys
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from survng.app.camera_capture import CameraCaptureService, CaptureOpenLimiter
from survng.app.live_detections import DetectionSnapshot
from survng.app.dlstreamer_capture import (
    DlStreamerCaptureBackend,
    DlStreamerCaptureHandle,
    DlStreamerCaptureOptions,
    adjacent_model_proc,
    _SharedLiveProcess,
    _StreamInbox,
)
from survng.app.dlstreamer_protocol import (
    TYPE_FRAME,
    TYPE_FATAL,
    TYPE_JPEG,
    TYPE_STATUS,
    MessageReader,
    decode_frame_payload,
    decode_json_payload,
    encode_detection_snapshot,
    encode_json,
)
from survng.dlstreamer_live import (
    _SYSTEM_GST_PLUGINS,
    _apply_dlstreamer_env,
    _colon_path,
    _drop_paths,
    _make_live_source,
    _normalize_gva_objects,
    _packed_gray,
    _parser,
    _require_detection_plugin,
    model_instance_id,
)


ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests" / "dlstreamer_live_stub.py"
SUPERVISOR_STUB = ROOT / "tests" / "dlstreamer_supervisor_stub.py"


def test_backend_command_keeps_url_and_uses_configured_policy() -> None:
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            python_executable="/usr/bin/python3",
            rtsp_transport="udp",
            frame_rate=lambda: 5.0,
            decoder="va",
            inference_device="GPU",
        ),
    )

    command = backend.command()

    assert command[:3] == ["/usr/bin/python3", "-m", "survng.dlstreamer_live"]
    assert command[command.index("--rtsp-transport") + 1] == "udp"
    assert command[command.index("--fps") + 1] == "5.000000"
    assert command[command.index("--detect-fps") + 1] == "5.000000"
    assert command[command.index("--main-fps") + 1] == "5.000000"
    assert command[command.index("--decoder") + 1] == "va"
    assert command[command.index("--frame-width") + 1] == "320"
    assert command[command.index("--jpeg-fps") + 1] == "1.000000"
    assert "--supervisor" in command
    assert "--no-detect" in command
    assert "rtsp://" not in " ".join(command)


def test_backend_includes_model_when_detect_enabled() -> None:
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            python_executable=sys.executable,
            detect_enabled=True,
            model_path="/models/yolo.xml",
            inference_device="GPU",
        ),
    )

    command = backend.command()

    assert command[command.index("--model") + 1] == "/models/yolo.xml"
    assert command[command.index("--model-instance-id") + 1] == "survng-yolo-GPU"
    assert "--supervisor" in command
    assert "--no-detect" not in command


def test_live_detection_rate_does_not_throttle_ema_or_main_capture():
    backend = DlStreamerCaptureBackend(CaptureOpenLimiter(1), DlStreamerCaptureOptions(
        frame_rate=lambda: 5, detection_frame_rate=lambda: 2.5, main_frame_rate=lambda: 5,
    ))
    command = backend.command()
    for option, expected in (("--fps", 5), ("--detect-fps", 2.5), ("--main-fps", 5)):
        assert float(command[command.index(option) + 1]) == expected


@pytest.mark.parametrize("detect,configured,expected", [
    (False, 3000, 3000), (True, 3000, 30000), (True, 45000, 45000),
])
def test_inference_startup_budget_agrees_between_parent_and_child(
    monkeypatch, detect, configured, expected,
) -> None:
    backend = DlStreamerCaptureBackend(CaptureOpenLimiter(1), DlStreamerCaptureOptions(
        detect_enabled=detect, open_timeout_ms=configured,
    ))
    observed = []

    def open_shared(handle, url, cancelled, *, timeout_ms):
        observed.append(timeout_ms)
        return True

    monkeypatch.setattr(backend, "_open_shared", open_shared)
    command = backend.command()
    assert backend.startup_timeout_ms == expected
    assert float(command[command.index("--open-timeout") + 1]) == expected / 1000
    assert backend.open(backend.create_handle(), "rtsp://fixture.invalid/live", lambda: False)
    assert observed == [expected]
    # Explicit caller deadlines still take precedence over the default budget.
    assert backend.open(backend.create_handle(), "rtsp://fixture.invalid/live", lambda: False,
                        open_timeout_ms=1000)
    assert observed == [expected, 1000]


def test_backend_passes_labels_and_adjacent_model_proc(tmp_path: Path) -> None:
    model = tmp_path / "yolo.xml"
    model.write_text("<net/>", encoding="utf-8")
    labels = tmp_path / "classes.txt"
    labels.write_text("person\n", encoding="utf-8")
    proc = tmp_path / "yolo.json"
    proc.write_text("{}", encoding="utf-8")
    backend = DlStreamerCaptureBackend(
        CaptureOpenLimiter(1),
        DlStreamerCaptureOptions(
            python_executable=sys.executable,
            detect_enabled=True,
            model_path=str(model),
            labels_path=str(labels),
        ),
    )

    command = backend.command()

    assert command[command.index("--labels") + 1] == str(labels)
    assert command[command.index("--model-proc") + 1] == str(proc)


def test_adjacent_model_proc_finds_json_next_to_ir(tmp_path: Path) -> None:
    model = tmp_path / "yolo.xml"
    model.write_text("<net/>", encoding="utf-8")
    proc = tmp_path / "yolo_proc.json"
    proc.write_text("{}", encoding="utf-8")

    assert adjacent_model_proc(str(model)) == str(proc)


def test_live_parser_accepts_model_proc_and_labels() -> None:
    args = _parser().parse_args(
        ["--model-proc", "/tmp/p.json", "--labels", "/tmp/l.txt", "--no-detect"]
    )

    assert args.model_proc == "/tmp/p.json"
    assert args.labels == "/tmp/l.txt"


def test_live_parser_accepts_qualifier_and_jpeg_rate() -> None:
    args = _parser().parse_args(
        ["--frame-width", "480", "--jpeg-fps", "0", "--detect-fps", "2.5", "--no-detect"]
    )

    assert args.frame_width == 480
    assert args.jpeg_fps == 0.0
    assert args.detect_fps == 2.5


def test_live_parser_accepts_supervisor_and_model_instance_id() -> None:
    args = _parser().parse_args(
        ["--supervisor", "--model-instance-id", "survng-yolo-GPU", "--no-detect"]
    )

    assert args.supervisor is True
    assert args.model_instance_id == "survng-yolo-GPU"


def test_model_instance_id_sanitizes_model_and_device() -> None:
    assert model_instance_id("/models/yolo.xml", "GPU") == "survng-yolo-GPU"
    assert model_instance_id("/models/yolo.xml", "GPU", "custom id!") == "custom-id"
    assert model_instance_id("", "CPU") == "survng-detect-CPU"


def test_backend_warns_once_without_logging_url_credentials(caplog) -> None:
    backend = DlStreamerCaptureBackend(CaptureOpenLimiter(1))

    backend.warn_credentialed_url("rtsp://admin:first-secret@camera/live")
    backend.warn_credentialed_url("rtsp://admin:second-secret@camera/main")

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "credential-free go2rtc" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "camera" in warnings[0]
    assert "admin" not in warnings[0]
    assert "secret" not in warnings[0]


def test_handle_reads_stub_child_frames() -> None:
    class StubBackend(DlStreamerCaptureBackend):
        def command(self) -> list[str]:
            return [sys.executable, str(STUB)]

    backend = StubBackend(CaptureOpenLimiter(1))
    handle = backend.create_handle()
    assert isinstance(handle, DlStreamerCaptureHandle)
    opened = backend.open(
        handle,
        "rtsp://127.0.0.1:8554/porch_sub",
        lambda: False,
        open_timeout_ms=2000,
    )
    try:
        assert opened
        ok, frame = handle.read()
        assert ok
        assert frame is not None
        assert frame.shape == (2, 2, 3)
        assert frame[0, 0].tolist() == [20, 40, 200]
        detections = handle.pop_detections()
        assert detections[0]["label"] == "person"
        assert handle.pop_jpeg() == b"\xff\xd8stub-jpeg\xff\xd9"
        status = handle.pipeline_status()
        assert status["ok"] is True
        assert status["hardware_decoder_selected"] is False
        assert status["preprocess_backend"] == "opencv"
        assert status["source_element"] == "uridecodebin3"
        assert status["decoder_elements"] == ["avdec_h264"]
    finally:
        handle.close()


def test_shared_supervisor_serves_two_handles_from_one_process() -> None:
    class StubBackend(DlStreamerCaptureBackend):
        def command(self) -> list[str]:
            return [sys.executable, str(SUPERVISOR_STUB), "--supervisor"]

    backend = StubBackend(CaptureOpenLimiter(2))
    first = backend.create_handle()
    second = backend.create_handle()
    try:
        assert backend.open(first, "rtsp://127.0.0.1:8554/porch_sub", lambda: False)
        assert backend.open(second, "rtsp://127.0.0.1:8554/drive_sub", lambda: False)
        shared = backend._shared
        assert shared is not None
        assert shared.is_running()
        first_ok, first_frame = first.read()
        second_ok, second_frame = second.read()
        assert first_ok and first_frame is not None
        assert second_ok and second_frame is not None
        assert first.pipeline_status()["model_instance_id"] == "survng-yolo-GPU"
        assert first.pipeline_status()["shared_detect"] is True
        assert first.pop_jpeg() == b"\xff\xd8stub-jpeg\xff\xd9"
        first.close()
        assert second.is_opened()
        assert shared.is_running()
    finally:
        first.close()
        second.close()
        backend.close()
    assert _colon_path("/usr/lib/gstreamer-1.0", "/opt/a:/usr/lib/gstreamer-1.0:/opt/b") == (
        "/usr/lib/gstreamer-1.0:/opt/a:/opt/b"
    )


def test_apply_dlstreamer_env_keeps_ubuntu_playback_plugins(monkeypatch) -> None:
    monkeypatch.setattr(Path, "is_dir", lambda p: str(p) == _SYSTEM_GST_PLUGINS)
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH", raising=False)
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH_1_0", raising=False)
    _apply_dlstreamer_env()
    assert os.environ["GST_PLUGIN_SYSTEM_PATH"].split(":")[0] == _SYSTEM_GST_PLUGINS
    assert os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"].split(":")[0] == _SYSTEM_GST_PLUGINS


@pytest.mark.parametrize("opencv", ["/opt/opencv", "/opt/intel/dlstreamer/opencv/lib"])
def test_intel_environment_includes_bundled_dependencies_and_is_idempotent(monkeypatch, opencv) -> None:
    monkeypatch.setattr(os, "environ", {"LD_LIBRARY_PATH": "/custom/lib"})
    directories = {"/opt/intel/dlstreamer/gstreamer/lib", opencv, "/opt/rdkafka", "/opt/librealsense"}
    monkeypatch.setattr(Path, "is_dir", lambda p: str(p) in directories)
    monkeypatch.setattr(Path, "is_file", lambda p: False)
    _apply_dlstreamer_env()
    expected = ":".join(["/opt/intel/dlstreamer/gstreamer/lib", "/opt/intel/dlstreamer/lib",
                         opencv, "/opt/rdkafka", "/opt/librealsense", "/custom/lib"])
    assert os.environ["LD_LIBRARY_PATH"] == expected
    assert "/opt/intel/dlstreamer/lib/girepository-1.0" in os.environ["GI_TYPELIB_PATH"].split(":")
    _apply_dlstreamer_env()
    assert os.environ["LD_LIBRARY_PATH"] == expected  # prevents re-exec loops


def test_missing_detection_plugin_exposes_native_loader_error(monkeypatch) -> None:
    monkeypatch.setattr(Path, "is_file", lambda p: True)
    def load(_path):
        raise RuntimeError("libopencv_imgproc.so.413: cannot open shared object file")
    gst = SimpleNamespace(ElementFactory=SimpleNamespace(find=lambda name: None),
                          Plugin=SimpleNamespace(load_file=load))
    with pytest.raises(RuntimeError, match="libopencv_imgproc.so.413"):
        _require_detection_plugin(gst)


def test_supervisor_fatal_error_reaches_all_streams_without_credentials(caplog, monkeypatch) -> None:
    monkeypatch.setattr("survng.app.dlstreamer_capture.select.select", lambda *args: ([True], [], []))
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    shared._inboxes = {name: _StreamInbox() for name in ("gate", "downstairs")}
    shared._stderr.extend(b"native detail rtsp://admin:stderr-secret@camera/live")
    shared._process = SimpleNamespace(stdout=io.BytesIO(encode_json(
        TYPE_FATAL, {"ok": False, "error": "libopencv missing; rtsp://admin:secret@camera/live"
                    + "x" * 500 + "; native root cause beyond source prefix"},
    )))
    shared._read_stdout()
    for inbox in shared._inboxes.values():
        assert not inbox.alive
        assert "libopencv missing" in inbox.error
        assert "secret" not in inbox.error
        assert "ProtocolError" not in inbox.error
    assert "secret" not in caplog.text
    assert "native root cause beyond source prefix" in caplog.text
    assert "native detail" in caplog.text


def test_inference_watchdog_distinguishes_empty_results_from_stalled_metadata(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("survng.app.dlstreamer_capture.time.monotonic", lambda: now[0])
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    live, main = _StreamInbox(), _StreamInbox()
    live.inference_started_at = 10.0
    shared._inboxes = {"live": live, "main": main}
    payload = {"schema_version": 1, "source_pts": 1.0, "inference_sequence": 1,
               "width": 320, "height": 240, "objects": []}
    now[0] = 14.0
    live.add_detection_snapshot(payload)
    shared._check_inference_progress(18.0)  # Empty inference is real progress.
    now[0] = 18.0
    live.add_detection_snapshot(payload)  # Repeated PTS cannot hide a stall.
    live.last_frame_at = 18.8  # Video remains continuous during the inference stall.
    now[0] = 19.0
    live.put_frame(np.zeros((2, 2), dtype=np.uint8), 99, 9.0)
    with pytest.raises(RuntimeError, match="inference stalled"):
        shared._check_inference_progress(19.1)
    live.alive = False  # Removed/stopped cameras do not trigger recovery.
    shared._check_inference_progress(100)


def test_inference_watchdog_does_not_restart_other_cameras_for_rtsp_outage():
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    live = _StreamInbox()
    live.inference_started_at = live.last_inference_at = live.last_frame_at = 10.0
    shared._inboxes = {"live": live}
    shared._check_inference_progress(20.0)


def test_stream_error_is_local_but_native_inference_error_fails_shared_reader(monkeypatch):
    monkeypatch.setattr("survng.app.dlstreamer_capture.select.select", lambda *args: ([True], [], []))
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    shared._inboxes = {name: _StreamInbox() for name in ("gate", "downstairs")}
    reader = MessageReader()
    reader.feed(encode_json(TYPE_STATUS, {"ok": False, "error": "RTSP disconnected", "failure_scope": "stream"}, stream_id="gate"))
    shared._dispatch(*reader.pop())
    assert not shared._inboxes["gate"].alive
    assert shared._inboxes["downstairs"].alive
    shared._process = SimpleNamespace(stdout=io.BytesIO(encode_json(
        TYPE_STATUS, {"ok": False, "error": "VA surface failed", "failure_scope": "inference"}, stream_id="downstairs")))
    shared._read_stdout()
    assert shared._failed
    assert all(not inbox.alive for inbox in shared._inboxes.values())


def test_shared_watchdog_does_not_reset_working_pool_for_one_delayed_camera():
    shared = _SharedLiveProcess([], read_timeout_ms=1000)
    delayed, working, main = _StreamInbox(), _StreamInbox(), _StreamInbox()
    for inbox in (delayed, working):
        inbox.inference_started_at = 10.0
        inbox.last_frame_at = 19.9
    delayed.last_inference_at = 12.0
    working.last_inference_at = 19.5
    main.last_frame_at = 19.9
    shared._inboxes = {"delayed": delayed, "working": working, "main": main}
    shared._check_inference_progress(20.0)
    assert all(inbox.alive for inbox in shared._inboxes.values())

    # A real pool-wide stall still fails, even while main/video frames arrive.
    working.last_inference_at = 12.0
    with pytest.raises(RuntimeError, match="shared live inference stalled"):
        shared._check_inference_progress(20.0)

    # Results from a retired stream cannot conceal a stalled active pool.
    working.last_inference_at = 19.5
    working.alive = False
    with pytest.raises(RuntimeError, match="shared live inference stalled"):
        shared._check_inference_progress(20.0)


def test_stalled_shared_process_is_replaced_and_sessions_are_not_reused(monkeypatch):
    monkeypatch.setattr("survng.app.dlstreamer_capture.DLSTREAMER_INFERENCE_STALL_SECONDS", 0.05)

    class StubBackend(DlStreamerCaptureBackend):
        def command(self):
            return [sys.executable, str(SUPERVISOR_STUB), "--supervisor"]

    backend = StubBackend(CaptureOpenLimiter(2))
    first, second = backend.create_handle(), backend.create_handle()
    try:
        assert backend.open(first, "rtsp://fixture.invalid/live", lambda: False)
        old = backend._shared
        old_process = old._process
        old_session = first._inbox.session
        deadline = time.monotonic() + 2
        while old.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not old.is_running()
        assert backend.open(second, "rtsp://fixture.invalid/live", lambda: False)
        assert backend._shared is not old
        assert old_process.poll() is not None
        assert second._inbox.session != old_session
    finally:
        first.close()
        second.close()
        backend.close()


def test_drop_paths_moves_intel_gstreamer_lib_out() -> None:
    assert _drop_paths(
        "/opt/intel/dlstreamer/gstreamer/lib:/usr/lib:/opt/intel/dlstreamer/lib",
        "/opt/intel/dlstreamer/gstreamer/lib",
    ) == "/usr/lib:/opt/intel/dlstreamer/lib"


class _FactoryGst:
    def __init__(self, available: set[str], *, makeable: set[str] | None = None) -> None:
        makeable = available if makeable is None else makeable
        self.ElementFactory = type(
            "ElementFactory",
            (),
            {
                "find": staticmethod(
                    lambda name, _available=available: object() if name in _available else None
                ),
                "make": staticmethod(
                    lambda name, _el, _makeable=makeable: object() if name in _makeable else None
                ),
            },
        )


def test_live_source_prefers_uridecodebin3() -> None:
    source, factory = _make_live_source(
        _FactoryGst({"uridecodebin3", "uridecodebin"}),
        test_source=False,
    )
    assert factory == "uridecodebin3"
    assert source is not None


def test_live_source_falls_back_to_uridecodebin() -> None:
    _source, factory = _make_live_source(_FactoryGst({"uridecodebin"}), test_source=False)
    assert factory == "uridecodebin"


def test_live_source_falls_back_when_uridecodebin3_cannot_instantiate() -> None:
    _source, factory = _make_live_source(
        _FactoryGst({"uridecodebin3", "uridecodebin"}, makeable={"uridecodebin"}),
        test_source=False,
    )
    assert factory == "uridecodebin"


def test_live_source_errors_when_no_uri_decoder_exists() -> None:
    with pytest.raises(RuntimeError, match="uridecodebin3 or uridecodebin"):
        _make_live_source(_FactoryGst({"videotestsrc"}), test_source=False)


def test_capture_close_waits_for_stderr_drain_before_closing_stream() -> None:
    class Stream:
        def __init__(self) -> None:
            self.eof = threading.Event()
            self.read_started = threading.Event()
            self.reader_finished = threading.Event()
            self.closed = False

        def read1(self, _size: int) -> bytes:
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

    handle = DlStreamerCaptureHandle(read_timeout_ms=1000)
    stderr = Stream()
    process = Process(stderr)
    handle._process = process  # type: ignore[assignment]
    handle._stderr_thread = threading.Thread(target=handle._drain_stderr)
    handle._stderr_thread.start()
    assert stderr.read_started.wait(1.0)

    handle.close()

    assert stderr.closed
    assert process.stdout.closed


def test_handle_reshapes_gray_frames() -> None:
    from survng.app.dlstreamer_protocol import encode_frame

    handle = DlStreamerCaptureHandle(read_timeout_ms=1000)
    pixels = bytes((1, 2, 3, 4, 5, 6))
    encoded = encode_frame(width=3, height=2, sequence=1, pts=0.1, pixels=pixels)
    from survng.app.dlstreamer_protocol import MessageReader

    reader = MessageReader()
    reader.feed(encoded)
    message_type, payload = reader.pop() or (0, b"")
    frame = handle._apply_message(message_type, payload)
    assert frame is not None
    assert frame.shape == (2, 3)
    assert frame[0, 0] == 1
    assert frame[1, 2] == 6


def test_handle_preserves_authoritative_empty_detection_snapshot() -> None:
    handle = DlStreamerCaptureHandle(read_timeout_ms=1000)
    reader = MessageReader()
    reader.feed(
        encode_detection_snapshot(
            source_pts=12.5,
            inference_sequence=4,
            width=640,
            height=360,
            objects=[],
        )
    )
    message_type, payload = reader.pop() or (0, b"")
    assert handle._apply_message(message_type, payload) is None
    assert handle.pop_detections() == []
    snapshots = handle.pop_detection_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0].objects == ()
    assert snapshots[0].source_pts == 12.5
    assert snapshots[0].session
    assert handle.pop_detection_snapshots() == []


def test_capture_matches_only_a_recent_prior_snapshot_in_its_generation() -> None:
    capture = CameraCaptureService(
        camera_id="gate",
        source_url=lambda _source: "",
        backend=None,  # Matching is independent of a native backend.
    )
    with capture._lock:
        capture._generation = 1
        history = capture._detection_history["live"]
        history.reset("test")
        history.add(DetectionSnapshot(8.0, 1, 640, 360, ({"label": "car"},), "test"))
        history.add(DetectionSnapshot(8.2, 2, 640, 360, (), "test"))
    # The authoritative empty result clears the old car for later EMA input.
    assert capture.matched_snapshot("live", source_pts=8.3, generation=1, source_session="test").objects == ()
    # A future result and a stale prior result are both rejected.
    assert capture.matched_snapshot("live", source_pts=7.9, generation=1, source_session="test") is None
    assert capture.matched_snapshot("live", source_pts=8.6, generation=1, source_session="test") is None
    assert capture.matched_snapshot("live", source_pts=8.3, generation=2, source_session="test") is None


def test_packed_gray_strips_row_stride() -> None:
    pixels = bytes((1, 2, 9, 9, 3, 4, 9, 9))
    assert _packed_gray(pixels, width=2, height=2) == bytes((1, 2, 3, 4))


def test_normalize_gva_objects_maps_boxes() -> None:
    objects = _normalize_gva_objects(
        {
            "objects": [
                {
                    "detection": {"label": "person", "confidence": 0.91},
                    "x": 10,
                    "y": 20,
                    "w": 30,
                    "h": 40,
                }
            ]
        }
    )
    assert objects == [
        {
            "label": "person",
            "confidence": 0.91,
            "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
        }
    ]


def _gstreamer_live_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        return Gst.ElementFactory.find("videotestsrc") is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _gstreamer_live_available(),
    reason="GStreamer videotestsrc is required for generated live capture",
)
def test_generated_source_emits_frames() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "survng.dlstreamer_live",
            "--test-source",
            "--fps",
            "5",
            "--open-timeout",
            "5",
            "--no-detect",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT),
        env=env,
    )
    assert process.stdout is not None
    try:
        reader = MessageReader()
        deadline = time.monotonic() + 5.0
        saw_frame = False
        saw_jpeg = False
        jpeg_preview = False
        while time.monotonic() < deadline:
            chunk = process.stdout.read1(65536)
            if chunk:
                reader.feed(chunk)
            popped = reader.pop()
            if popped is None:
                if process.poll() is not None:
                    break
                time.sleep(0.02)
                continue
            message_type, payload = popped
            if message_type == TYPE_STATUS:
                status = decode_json_payload(payload)
                assert status["qualifier_format"] == "GRAY8"
                assert status["qualifier_width"] == 320
                jpeg_preview = bool(status.get("jpeg_preview"))
            elif message_type == TYPE_FRAME:
                width, height, _sequence, _pts, pixels = decode_frame_payload(payload)
                assert len(pixels) == width * height
                assert width == 320
                saw_frame = True
            elif message_type == TYPE_JPEG:
                saw_jpeg = True
            if saw_frame and (saw_jpeg or not jpeg_preview):
                break
        assert saw_frame
        if jpeg_preview:
            assert saw_jpeg
    finally:
        process.terminate()
        process.wait(timeout=2.0)


def test_live_child_does_not_import_pydantic_config_stack() -> None:
    live_source = (ROOT / "survng" / "dlstreamer_live.py").read_text(encoding="utf-8")
    redact_source = (ROOT / "survng" / "app" / "redact.py").read_text(encoding="utf-8")

    assert "survng.app.security" not in live_source
    assert "survng.app.redact" in live_source
    assert "pydantic" not in redact_source
    assert "from .config" not in redact_source
    assert "from survng.app.config" not in redact_source


def test_system_python_can_import_live_child_without_pydantic() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            "from survng.dlstreamer_live import main; from survng.app.redact import redact_secret_text",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pydantic" not in result.stderr


@pytest.mark.parametrize("supervisor", [False, True])
def test_system_python_live_main_redacts_errors_without_pydantic(supervisor) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "from survng import dlstreamer_live\n"
                "def boom(argv=None):\n"
                "    raise RuntimeError('rtsp://admin:secret@camera/live failed')\n"
                "dlstreamer_live.run = boom\n"
                f"raise SystemExit(dlstreamer_live.main({['--supervisor'] if supervisor else []!r}))\n"
            ),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=False,
        check=False,
    )
    combined = (result.stdout + result.stderr).decode(errors="replace")
    assert result.returncode == 1
    assert "secret" not in combined
    assert "pydantic" not in combined
    assert "rtsp://admin:***@camera/live" in combined
    reader = MessageReader()
    reader.feed(result.stdout)
    kind, payload = reader.pop()
    assert kind == (TYPE_FATAL if supervisor else TYPE_STATUS)
    assert decode_json_payload(payload)["ok"] is False
