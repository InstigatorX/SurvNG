from __future__ import annotations

import unittest

from survng.app.config import DetectorConfig
from survng.app.inference_runtime.worker_topology import (
    MAX_OBJECT_WORKERS,
    MIN_OBJECT_WORKERS,
    classify_device_class,
    clamp_object_worker_count,
    object_worker_recommendation_from_status,
    recommend_object_worker_count,
)


class WorkerTopologyTests(unittest.TestCase):
    def test_object_worker_count_defaults_to_protected_tracking_lane(self) -> None:
        self.assertEqual(DetectorConfig().object_worker_count, 2)
        self.assertEqual(
            DetectorConfig(object_worker_count=1).object_worker_count,
            2,
        )
        self.assertEqual(
            DetectorConfig(
                object_worker_count=1,
                tracking={"enabled": False},
            ).effective_object_worker_count(),
            1,
        )
        self.assertEqual(
            DetectorConfig(object_worker_count=3).effective_object_worker_count(),
            3,
        )

    def test_clamp_bounds(self) -> None:
        self.assertEqual(clamp_object_worker_count(0), MIN_OBJECT_WORKERS)
        self.assertEqual(clamp_object_worker_count(99), MAX_OBJECT_WORKERS)
        self.assertEqual(clamp_object_worker_count("3"), 3)
        self.assertEqual(clamp_object_worker_count("bad"), MIN_OBJECT_WORKERS)

    def test_classify_device_class(self) -> None:
        self.assertEqual(classify_device_class("CPU"), "cpu")
        self.assertEqual(classify_device_class("GPU"), "accelerator")
        self.assertEqual(classify_device_class("NPU"), "accelerator")
        self.assertEqual(classify_device_class("AUTO"), "auto")
        self.assertEqual(classify_device_class(""), "auto")

    def test_low_waits_keep_current(self) -> None:
        result = recommend_object_worker_count(
            current=1,
            device="CPU",
            initial_admission_wait_ms_p95=10.0,
            pending_requests=0,
            queue_depth=0,
        )
        self.assertEqual(result["recommended"], 1)
        self.assertEqual(result["current"], 1)
        self.assertTrue(result["reasons"])
        self.assertFalse(result["signals"]["pending_pressure"])

    def test_cpu_high_wait_and_pending_suggests_two(self) -> None:
        result = recommend_object_worker_count(
            current=1,
            device="CPU",
            initial_admission_wait_ms_p95=150.0,
            pending_requests=2,
            queue_depth=2,
        )
        self.assertEqual(result["recommended"], 2)
        self.assertTrue(result["signals"]["pending_pressure"])
        self.assertIn("CPU", result["reasons"][0])

    def test_cpu_extreme_pressure_can_scale_toward_four(self) -> None:
        result = recommend_object_worker_count(
            current=2,
            device="CPU",
            initial_admission_wait_ms_p95=200.0,
            refinement_admission_wait_ms_p95=180.0,
            pending_requests=6,
            queue_depth=6,
            refinement_active=2,
            initial_waiting=3,
        )
        self.assertEqual(result["recommended"], 3)

        result_max = recommend_object_worker_count(
            current=3,
            device="CPU",
            initial_admission_wait_ms_p95=250.0,
            pending_requests=10,
            queue_depth=10,
            initial_waiting=4,
        )
        self.assertEqual(result_max["recommended"], 4)

    def test_gpu_does_not_blindly_scale_to_four(self) -> None:
        result = recommend_object_worker_count(
            current=4,
            device="GPU",
            initial_admission_wait_ms_p95=200.0,
            pending_requests=8,
            queue_depth=8,
        )
        self.assertEqual(result["recommended"], 2)
        self.assertLessEqual(result["recommended"], 2)

    def test_gpu_cautious_scale_from_one_to_two(self) -> None:
        result = recommend_object_worker_count(
            current=1,
            device="GPU",
            initial_admission_wait_ms_p95=160.0,
            pending_requests=2,
        )
        self.assertEqual(result["recommended"], 2)

    def test_never_recommends_outside_bounds(self) -> None:
        result = recommend_object_worker_count(
            current=99,
            device="CPU",
            initial_admission_wait_ms_p95=999.0,
            pending_requests=99,
            initial_waiting=99,
        )
        self.assertGreaterEqual(result["recommended"], MIN_OBJECT_WORKERS)
        self.assertLessEqual(result["recommended"], MAX_OBJECT_WORKERS)
        self.assertEqual(result["current"], MAX_OBJECT_WORKERS)

    def test_security_waiting_defers_scale_up(self) -> None:
        result = recommend_object_worker_count(
            current=1,
            device="CPU",
            initial_admission_wait_ms_p95=200.0,
            pending_requests=3,
            security_waiting=1,
        )
        self.assertEqual(result["recommended"], 1)
        self.assertTrue(
            any("security" in reason for reason in result["reasons"])
        )

    def test_recommendation_from_status_reads_workloads(self) -> None:
        status = {
            "object_worker_count": 1,
            "configured_device": "CPU",
            "loaded_device": "CPU",
            "isolation": {"pending_requests": 3, "configured_workers": 1},
            "runtime": {
                "queue_depth": 3,
                "workloads": {
                    "initial_waiting": 1,
                    "refinement_active": 0,
                    "security_waiting": 0,
                    "classes": {
                        "incident_initial": {"admission_wait_ms_p95": 175.0},
                        "incident_refinement": {"admission_wait_ms_p95": 20.0},
                    },
                },
            },
        }
        result = object_worker_recommendation_from_status(
            status,
            recorded_decode={"waiting": 1},
        )
        self.assertEqual(result["recommended"], 2)
        self.assertEqual(result["signals"]["decode_waiting"], 1)
        self.assertEqual(result["signals"]["device_class"], "cpu")
        self.assertEqual(result["signals"]["initial_admission_wait_ms_p95"], 175.0)


if __name__ == "__main__":
    unittest.main()
