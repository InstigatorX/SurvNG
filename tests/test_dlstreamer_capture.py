from __future__ import annotations

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

import pytest

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.dlstreamer_capture import (
    DlStreamerCaptureBackend,
    DlStreamerCaptureHandle,
    DlStreamerCaptureOptions,
    adjacent_model_proc,
)
from survng.app.dlstreamer_protocol import (
    TYPE_FRAME,
    TYPE_JPEG,
    TYPE_STATUS,
    MessageReader,
    decode_frame_payload,
    decode_json_payload,
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
        ["--frame-width", "480", "--jpeg-fps", "0", "--no-detect"]
    )

    assert args.frame_width == 480
    assert args.jpeg_fps == 0.0


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
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH", raising=False)
    monkeypatch.delenv("GST_PLUGIN_SYSTEM_PATH_1_0", raising=False)
    _apply_dlstreamer_env()
    assert os.environ["GST_PLUGIN_SYSTEM_PATH"].split(":")[0] == _SYSTEM_GST_PLUGINS
    assert os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"].split(":")[0] == _SYSTEM_GST_PLUGINS


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


def test_system_python_live_main_redacts_errors_without_pydantic() -> None:
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
                "raise SystemExit(dlstreamer_live.main([]))\n"
            ),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "secret" not in combined
    assert "pydantic" not in combined
    assert "rtsp://admin:***@camera/live" in combined
