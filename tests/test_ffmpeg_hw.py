from __future__ import annotations

import unittest
from unittest.mock import patch

from survng.app.ffmpeg_hw import (
    hardware_mode,
    qsv_decode_args,
    recorded_frame_hw_args,
)


class FfmpegHardwareTest(unittest.TestCase):
    def test_hardware_mode_normalizes_known_values_and_defaults_unknown(self) -> None:
        self.assertEqual(hardware_mode(" QSV "), "qsv")
        self.assertEqual(hardware_mode("vaapi"), "vaapi")
        self.assertEqual(hardware_mode("off"), "off")
        self.assertEqual(hardware_mode("unexpected"), "auto")
        self.assertEqual(hardware_mode(None), "auto")

    def test_qsv_decode_arguments_use_discovered_render_device(self) -> None:
        with patch("survng.app.ffmpeg_hw.dri_render_device", return_value="/dev/dri/renderD129"):
            arguments = qsv_decode_args("qsv")

        self.assertEqual(arguments, [
            "-qsv_device",
            "/dev/dri/renderD129",
            "-hwaccel",
            "qsv",
            "-hwaccel_output_format",
            "qsv",
        ])

    def test_recorded_frame_plans_include_hwdownload_only_for_explicit_hardware(self) -> None:
        with patch("survng.app.ffmpeg_hw.dri_render_device", return_value="/dev/dri/renderD128"):
            vaapi_input, vaapi_filter = recorded_frame_hw_args("vaapi")
            qsv_input, qsv_filter = recorded_frame_hw_args("qsv")

        self.assertIn("vaapi", vaapi_input)
        self.assertEqual(vaapi_filter, ["-vf", "hwdownload,format=nv12"])
        self.assertIn("qsv", qsv_input)
        self.assertEqual(qsv_filter, ["-vf", "hwdownload,format=nv12"])
        self.assertEqual(recorded_frame_hw_args("auto"), ([], []))
        self.assertEqual(recorded_frame_hw_args("off"), ([], []))


if __name__ == "__main__":
    unittest.main()
