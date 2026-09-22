from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference import InferenceUnavailable, InferenceWorkload
from survng.app.inference_runtime.remote_backend import (
    RoutedInferenceWorkerBackend,
)


class RemoteInferenceBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Mock()
        self.registry.has_ready_worker.return_value = True
        self.registry.status.return_value = {
            "connected": 1,
            "ready": 1,
            "workers": [],
        }
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def _backend(self, mode: str) -> RoutedInferenceWorkerBackend:
        config = DetectorConfig(
            enabled=False,
            inference_mode=mode,
            object_worker_count=1,
            tracking={"enabled": False},
        )
        backend = RoutedInferenceWorkerBackend(
            config,
            "object",
            {"ready": False},
            registry=self.registry,
        )
        backend._remote.start()
        return backend

    def test_remote_mode_routes_without_constructing_local_worker(self) -> None:
        self.registry.request.return_value = [{"label": "person"}]
        backend = self._backend("remote")

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.INCIDENT_INITIAL,
        )

        self.assertEqual(result, [{"label": "person"}])
        self.assertIsNone(backend._local)
        self.registry.request.assert_called_once()

    def test_hybrid_initial_request_falls_back_locally(self) -> None:
        self.registry.request.side_effect = InferenceUnavailable(
            "remote pool unavailable"
        )
        backend = self._backend("hybrid")
        self.assertIsNotNone(backend._local)
        backend._local.request = Mock(return_value=[{"label": "vehicle"}])

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.INCIDENT_INITIAL,
        )

        self.assertEqual(result, [{"label": "vehicle"}])
        backend._local.request.assert_called_once()

    def test_hybrid_optional_request_does_not_use_local_fallback(self) -> None:
        self.registry.request.side_effect = InferenceUnavailable(
            "remote pool unavailable"
        )
        backend = self._backend("hybrid")
        self.assertIsNotNone(backend._local)
        backend._local.request = Mock()

        with self.assertRaises(InferenceUnavailable):
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )

        backend._local.request.assert_not_called()

    def test_hybrid_fallback_can_be_disabled(self) -> None:
        self.registry.request.side_effect = InferenceUnavailable(
            "remote pool unavailable"
        )
        backend = self._backend("hybrid")
        backend.config.remote_incident_fallback = False
        self.assertIsNotNone(backend._local)
        backend._local.request = Mock()

        with self.assertRaises(InferenceUnavailable):
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.INCIDENT_INITIAL,
            )

        backend._local.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
