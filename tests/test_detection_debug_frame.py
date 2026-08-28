from __future__ import annotations

import asyncio
import base64
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np

from survng.app.detection_routes import DetectionRouteDependencies, create_detection_router
from survng.app.inference_runtime.types import InferenceWorkload


def _jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    return encoded.tobytes()


class DetectionDebugFrameRouteTests(TestCase):
    def setUp(self) -> None:
        self.detected_objects = [
            {
                "label": "person",
                "confidence": 0.91,
                "box": {"x1": 4, "y1": 6, "x2": 20, "y2": 30},
            }
        ]
        self.enriched_objects = [
            {
                **self.detected_objects[0],
                "depth_stats": {"median_m": 4.2, "min_m": 3.8, "max_m": 4.8},
            }
        ]
        self.heatmap_png = b"\x89PNGdepth-test"
        self.detector = SimpleNamespace(
            detect=Mock(return_value=list(self.detected_objects)),
            depth_status=Mock(
                return_value={
                    "enabled": True,
                    "ready": True,
                    "error": "",
                    "min_distance_m": 0.5,
                    "max_distance_m": 30.0,
                }
            ),
            estimate_depth_for_objects=Mock(
                return_value=(
                    list(self.enriched_objects),
                    {
                        "inference_ms": 42.0,
                        "heatmap_png": self.heatmap_png,
                        "heatmap_range_m": {"min_m": 0.5, "max_m": 30.0},
                    },
                )
            ),
        )
        self.manager = SimpleNamespace(
            detector=self.detector,
            config=SimpleNamespace(),
        )
        dependencies = DetectionRouteDependencies(
            get_manager=lambda: self.manager,
            get_config=lambda: self.manager.config,
            manager_lock=threading.RLock(),
            get_comparison_limiter=Mock(),
            ensure_event_clip=Mock(),
            dependency_status=Mock(return_value={"available": True}),
            comparison_runner=Mock(),
            sample_video_frames=Mock(),
        )
        self.handler = create_detection_router(dependencies).handlers["detect_debug_frame"]

    async def _call(self, *, depth: bool = False, heatmap: bool = False) -> dict:
        body = _jpeg_bytes()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "headers": [(b"content-length", str(len(body)).encode("ascii"))],
            "query_string": b"",
        }
        from starlette.requests import Request

        request = Request(scope, receive)
        return await self.handler(
            request,
            confidence=0.35,
            depth=depth,
            heatmap=heatmap,
        )

    def test_detect_debug_frame_returns_detection_only_by_default(self) -> None:
        payload = asyncio.run(self._call())
        self.assertEqual(payload["objects"], self.detected_objects)
        self.assertEqual(payload["detect_ms"], payload["elapsed_ms"])
        self.assertNotIn("depth_ms", payload)
        self.detector.estimate_depth_for_objects.assert_not_called()

    def test_detect_debug_frame_depth_includes_stats_and_heatmap(self) -> None:
        payload = asyncio.run(self._call(depth=True, heatmap=True))
        self.assertEqual(payload["objects"], self.enriched_objects)
        self.assertEqual(payload["depth_ms"], payload["elapsed_ms"] - payload["detect_ms"])
        self.assertEqual(
            payload["heatmap_png_b64"],
            base64.b64encode(self.heatmap_png).decode("ascii"),
        )
        self.assertEqual(payload["heatmap_range_m"], {"min_m": 0.5, "max_m": 30.0})
        kwargs = self.detector.estimate_depth_for_objects.call_args.kwargs
        self.assertTrue(kwargs["include_heatmap"])
        self.assertEqual(kwargs["workload"], InferenceWorkload.INTERACTIVE)

    def test_detect_debug_frame_reports_depth_configuration_error(self) -> None:
        self.detector.depth_status.return_value = {
            "enabled": False,
            "ready": False,
            "error": "Depth estimation is not configured.",
        }
        payload = asyncio.run(self._call(depth=True))
        self.assertEqual(payload["depth_error"], "Depth estimation is not configured.")
        self.detector.estimate_depth_for_objects.assert_not_called()
