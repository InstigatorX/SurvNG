from __future__ import annotations

import unittest
from unittest.mock import patch

from survng.app.ffmpeg_hw import (
    capture_decode_plan_names,
    capture_decode_plans,
    hardware_mode,
    qsv_decode_args,
    recorded_frame_hw_args,
    resolve_capture_decode_plans,
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

    def test_live_capture_plan_order_is_qsv_then_vaapi_then_cpu(self) -> None:
        self.assertEqual(
            capture_decode_plan_names("off", True, True),
            ("cpu",),
        )
        self.assertEqual(
            capture_decode_plan_names("qsv", False, True),
            ("cpu",),
        )
        self.assertEqual(
            capture_decode_plan_names("qsv", True, True),
            ("qsv", "cpu"),
        )
        self.assertEqual(
            capture_decode_plan_names("vaapi", True, True),
            ("vaapi", "cpu"),
        )
        self.assertEqual(
            capture_decode_plan_names("auto", True, True),
            ("qsv", "vaapi", "cpu"),
        )
        self.assertEqual(
            capture_decode_plan_names("auto", False, True),
            ("vaapi", "cpu"),
        )
        self.assertEqual(
            capture_decode_plan_names("auto", False, False),
            ("cpu",),
        )
        plans = capture_decode_plans(
            "auto",
            qsv_ready=True,
            vaapi_ready=True,
            device="/dev/dri/renderD129",
        )
        self.assertEqual(
            plans[0].input_args,
            (
                "-qsv_device",
                "/dev/dri/renderD129",
                "-hwaccel",
                "qsv",
                "-hwaccel_output_format",
                "qsv",
            ),
        )
        self.assertEqual(plans[0].download_filters, ("hwdownload", "format=nv12"))
        self.assertEqual(plans[1].name, "vaapi")
        self.assertEqual(plans[1].download_filters, ("hwdownload", "format=nv12"))
        self.assertEqual(plans[2], capture_decode_plans("off", qsv_ready=True, vaapi_ready=True, device="")[0])

    def test_resolve_skips_the_hwaccel_probe_without_a_render_node(self) -> None:
        with (
            patch("survng.app.ffmpeg_hw.render_device_available", return_value=False),
            patch("survng.app.ffmpeg_hw.ffmpeg_hwaccels") as probe,
        ):
            auto_plans = resolve_capture_decode_plans("auto", "ffmpeg")
            explicit_plans = resolve_capture_decode_plans("qsv", "ffmpeg")

        probe.assert_not_called()
        self.assertEqual([plan.name for plan in auto_plans], ["cpu"])
        self.assertEqual([plan.name for plan in explicit_plans], ["cpu"])

    def test_resolve_off_does_not_probe_when_a_render_node_exists(self) -> None:
        with (
            patch("survng.app.ffmpeg_hw.render_device_available", return_value=True),
            patch("survng.app.ffmpeg_hw.ffmpeg_hwaccels") as probe,
        ):
            plans = resolve_capture_decode_plans("off", "ffmpeg")

        probe.assert_not_called()
        self.assertEqual([plan.name for plan in plans], ["cpu"])

    def test_resolve_auto_uses_only_listed_hwaccels(self) -> None:
        with (
            patch("survng.app.ffmpeg_hw.render_device_available", return_value=True),
            patch(
                "survng.app.ffmpeg_hw.ffmpeg_hwaccels",
                return_value=frozenset({"vaapi"}),
            ),
            patch(
                "survng.app.ffmpeg_hw.dri_render_device",
                return_value="/dev/dri/renderD128",
            ),
        ):
            plans = resolve_capture_decode_plans("auto", "ffmpeg")

        self.assertEqual([plan.name for plan in plans], ["vaapi", "cpu"])
        self.assertEqual(plans[0].input_args[0], "-vaapi_device")
        self.assertEqual(plans[0].input_args[1], "/dev/dri/renderD128")


if __name__ == "__main__":
    unittest.main()
