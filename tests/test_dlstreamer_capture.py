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
)
from survng.dlstreamer_live import _normalize_gva_objects


ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests" / "dlstreamer_live_stub.py"


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
    assert "--no-detect" not in command


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
    finally:
        handle.close()


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
    from survng.app.dlstreamer_protocol import TYPE_FRAME, MessageReader

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
            message_type, _payload = popped
            if message_type == TYPE_FRAME:
                saw_frame = True
                break
        assert saw_frame
    finally:
        process.terminate()
        process.wait(timeout=2.0)
