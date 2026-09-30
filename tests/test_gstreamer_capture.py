from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from survng.app.camera_capture import CaptureOpenLimiter
from survng.app.gstreamer_capture import (
    GStreamerCaptureBackend,
    GStreamerCaptureOptions,
)
from survng.app.gstreamer_capture_worker import FRAME_MAGIC


def _worker_available() -> bool:
    python = "/usr/bin/python3"
    if shutil.which("ffmpeg") is None or not Path(python).exists():
        return False
    probe = subprocess.run(
        [
            python,
            "-c",
            "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


class GStreamerCaptureTest(unittest.TestCase):
    def test_options_reject_live_hardware_decode(self) -> None:
        with self.assertRaisesRegex(ValueError, "software decode"):
            GStreamerCaptureOptions(hardware_acceleration="qsv")

    def test_rtsp_command_is_software_and_keeps_the_url_intact(self) -> None:
        backend = GStreamerCaptureBackend(
            CaptureOpenLimiter(),
            GStreamerCaptureOptions(
                rtsp_transport="tcp",
                frame_rate=lambda: 8.2,
            ),
        )
        command = backend._command("rtsp://user:secret@camera/live")
        self.assertEqual(command[0], "/usr/bin/python3")
        self.assertTrue(command[1].endswith("gstreamer_capture_worker.py"))
        self.assertEqual(command[2], "rtsp://user:secret@camera/live")
        self.assertEqual(command[3], "8")
        self.assertEqual(command[4], "tcp")

    def test_file_round_trip_delivers_owned_bgr_frames(self) -> None:
        if not _worker_available():
            self.skipTest("system gstreamer bindings or ffmpeg are unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sample.mp4"
            built = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=32x24:rate=10",
                    "-frames:v",
                    "4",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                capture_output=True,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            backend = GStreamerCaptureBackend(
                CaptureOpenLimiter(),
                GStreamerCaptureOptions(
                    open_timeout_ms=8000,
                    read_timeout_ms=5000,
                    frame_rate=lambda: 10,
                ),
            )
            handle = backend.create_handle()
            try:
                opened = backend.open(handle, str(source), lambda: False)
                self.assertTrue(opened, handle.error_detail())
                ok, first = handle.read()
                self.assertTrue(ok)
                assert first is not None
                self.assertEqual(first.shape, (24, 32, 3))
                self.assertEqual(handle.decode_plan, "gstreamer-cpu")
                ok, second = handle.read()
                self.assertTrue(ok)
                assert second is not None
                self.assertIsNot(first, second)
                self.assertEqual(second.shape, (24, 32, 3))
            finally:
                handle.close()

    def test_header_constant_matches_the_worker_layout(self) -> None:
        packet = FRAME_MAGIC + (32).to_bytes(4, "little") + (24).to_bytes(4, "little")
        self.assertEqual(packet[:4], b"SVNG")
        self.assertEqual(int.from_bytes(packet[4:8], "little"), 32)
        self.assertEqual(int.from_bytes(packet[8:12], "little"), 24)
        self.assertEqual(np.dtype(np.uint8).itemsize, 1)
