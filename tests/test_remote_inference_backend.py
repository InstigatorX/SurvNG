from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from survng.app.config import DetectorConfig, detector_routing_payload
from survng.app.inference import InferenceUnavailable, InferenceWorkload
from survng.app.inference import INFERENCE_FAILOVER_SECONDS
from survng.app.inference_runtime.protocol import (
    WorkerHeartbeat,
    WorkerReady,
    WorkerRegistration,
    decode_packet,
    encode_packet,
)
from survng.app.inference_runtime.registry import RemoteInferenceRegistry
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


class _ResultTransport:
    def __init__(self, result: object, *, inference_ms: float | None = None) -> None:
        self.result = result
        self.inference_ms = inference_ms
        self.requests = 0
        self.timeouts: list[float] = []
        self.error: BaseException | None = None

    def request(self, packet: bytes, timeout: float) -> bytes:
        self.timeouts.append(timeout)
        self.requests += 1
        if self.error is not None:
            raise self.error
        message, _frame = decode_packet(packet)
        response = {
            "type": "response",
            "request_id": message["request_id"],
            "connection_generation": message["connection_generation"],
            "ok": True,
            "result": self.result,
        }
        if self.inference_ms is not None:
            response["inference_ms"] = self.inference_ms
        return encode_packet(response)

    def close(self, reason: str) -> None:
        del reason


class WeightedInferenceBalanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def _config(self, **updates: object) -> DetectorConfig:
        config = DetectorConfig(
            enabled=False,
            inference_mode="hybrid",
            inference_balance="weighted",
            inference_primary_weight=1,
            inference_default_worker_weight=1,
            object_worker_count=1,
            tracking={"enabled": False},
        )
        if updates:
            config = config.model_copy(update=updates)
        return config

    def _ready_registry(
        self,
        config: DetectorConfig,
        worker_ids: list[str],
    ) -> tuple[RemoteInferenceRegistry, dict[str, _ResultTransport]]:
        registry = RemoteInferenceRegistry(
            lambda: config,
            lease_seconds=30.0,
            clock=lambda: self.now,
        )
        transports: dict[str, _ResultTransport] = {}
        for worker_id in worker_ids:
            transport = _ResultTransport({"worker": worker_id})
            transports[worker_id] = transport
            welcome = registry.register(
                WorkerRegistration(
                    worker_id=worker_id,
                    name=worker_id,
                    roles=["object"],
                ),
                transport,
            )
            self.assertTrue(registry.mark_ready(WorkerReady(
                worker_id=worker_id,
                connection_generation=welcome["connection_generation"],
                config_generation=welcome["config_generation"],
                statuses={"object": {"ready": True, "enabled": True}},
            )))
        return registry, transports

    def _backend(
        self,
        config: DetectorConfig,
        registry: RemoteInferenceRegistry,
    ) -> RoutedInferenceWorkerBackend:
        backend = RoutedInferenceWorkerBackend(
            config,
            "object",
            {"ready": False},
            registry=registry,
        )
        backend._remote.start()
        assert backend._local is not None
        backend._local.request = Mock(return_value={"worker": "primary"})
        backend._local.pending_requests = Mock(return_value=0)
        return backend

    def test_equal_weights_alternate_primary_and_worker(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        backend = self._backend(config, registry)

        choices = [
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.INCIDENT_INITIAL,
            )["worker"]
            for _index in range(4)
        ]

        self.assertEqual(choices, ["primary", "worker-a", "primary", "worker-a"])
        self.assertEqual(transports["worker-a"].requests, 2)

    def test_heavier_worker_receives_idle_detections(self) -> None:
        config = self._config(
            inference_primary_weight=1,
            inference_worker_weights={"worker-a": 4},
        )
        registry, transports = self._ready_registry(config, ["worker-a"])
        backend = self._backend(config, registry)

        choices = [
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )["worker"]
            for _index in range(3)
        ]

        self.assertEqual(choices, ["worker-a", "worker-a", "worker-a"])
        self.assertEqual(transports["worker-a"].requests, 3)
        backend._local.request.assert_not_called()

    def test_zero_primary_weight_stays_on_workers_until_failure(self) -> None:
        config = self._config(inference_primary_weight=0)
        registry, _transports = self._ready_registry(config, ["worker-a"])
        backend = self._backend(config, registry)
        registry.request = Mock(side_effect=InferenceUnavailable("worker down"))

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.INCIDENT_INITIAL,
        )

        self.assertEqual(result, {"worker": "primary"})
        with self.assertRaises(InferenceUnavailable):
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )

    def test_worker_failure_uses_primary_for_any_workload(self) -> None:
        config = self._config()
        registry, _transports = self._ready_registry(config, ["worker-a"])
        backend = self._backend(config, registry)
        original = registry.request
        calls = {"count": 0}

        def fail_once(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise InferenceUnavailable("worker down")
            return original(*args, **kwargs)

        registry.request = fail_once
        # Equal idle scores start on the primary, so force the worker to win.
        backend._local.pending_requests = Mock(return_value=5)

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(result, {"worker": "primary"})

    def test_recovered_worker_timeout_is_rerouted(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=5)

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(result, {"worker": "primary"})
        worker = registry.status()["workers"][0]
        self.assertEqual(worker["failed_requests"], 0)
        self.assertEqual(worker["rerouted_requests"], 1)
        self.assertEqual(worker["completed_requests"], 0)
        self.assertEqual(worker["last_outcome"], "rerouted")
        self.assertIn("timed out", worker["last_error"])
        self.assertEqual(worker["last_role"], "object")
        self.assertEqual(worker["last_operation"], "detect")
        self.assertEqual(worker["role_attempts"]["object"]["rerouted"], 1)
        self.assertEqual(worker["role_attempts"]["object"]["failed"], 0)

    def test_unrecovered_worker_timeout_stays_failed(self) -> None:
        config = self._config(inference_primary_weight=0)
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)

        with self.assertRaises(InferenceUnavailable):
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )

        worker = registry.status()["workers"][0]
        self.assertEqual(worker["failed_requests"], 1)
        self.assertEqual(worker["rerouted_requests"], 0)
        self.assertEqual(worker["last_outcome"], "failed")
        self.assertEqual(worker["role_attempts"]["object"]["failed"], 1)

    def test_completed_request_records_worker_inference_time(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].inference_ms = 18.2
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=5)

        result = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(result, {"worker": "worker-a"})
        worker = registry.status()["workers"][0]
        self.assertEqual(worker["completed_requests"], 1)
        self.assertEqual(worker["failed_requests"], 0)
        self.assertEqual(worker["last_outcome"], "completed")
        self.assertEqual(worker["last_inference_ms"], 18.2)
        self.assertEqual(worker["average_inference_ms"], 18.2)
        self.assertEqual(worker["last_request_ms"], 0.0)
        self.assertEqual(worker["last_error"], "")
        self.assertEqual(worker["role_attempts"]["object"]["completed"], 1)

    def test_faster_worker_receives_equal_weight_detections(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].inference_ms = 20
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=5)
        backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )
        backend._local.pending_requests = Mock(return_value=0)
        backend._local.cached_status = Mock(return_value={
            "runtime": {"average_inference_ms": 100},
        })

        choices = [
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )["worker"]
            for _index in range(3)
        ]

        self.assertEqual(choices, ["worker-a", "worker-a", "worker-a"])

    def test_unmeasured_primary_uses_the_worker_time(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].inference_ms = 20
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=5)
        backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )
        backend._local.pending_requests = Mock(return_value=0)

        choices = [
            backend.request(
                "detect",
                frame=self.frame,
                workload=InferenceWorkload.TRACKING,
            )["worker"]
            for _index in range(2)
        ]

        self.assertEqual(choices, ["primary", "worker-a"])

    def test_timed_out_worker_is_skipped_while_its_queue_remains(self) -> None:
        config = self._config(inference_worker_weights={"worker-a": 8})
        registry, transports = self._ready_registry(config, ["worker-a"])
        worker = registry.status()["workers"][0]
        self.assertTrue(registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-a",
            connection_generation=worker["connection_generation"],
            pending_requests=3,
        )))
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=10)

        first = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(first, {"worker": "primary"})
        held = registry.status()["workers"][0]
        self.assertTrue(held["routing_hold"])
        self.assertEqual(held["rerouted_requests"], 1)
        backend._local.pending_requests = Mock(return_value=0)
        transports["worker-a"].error = None

        second = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(second, {"worker": "primary"})
        self.assertEqual(transports["worker-a"].requests, 1)
        self.assertTrue(registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-a",
            connection_generation=worker["connection_generation"],
            pending_requests=0,
        )))

        third = backend.request(
            "detect",
            frame=self.frame,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(third, {"worker": "worker-a"})
        self.assertFalse(registry.status()["workers"][0]["routing_hold"])

    def test_worker_attempt_fails_over_before_the_full_timeout(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)
        backend._local.pending_requests = Mock(return_value=5)

        result = backend.request(
            "detect",
            frame=self.frame,
            timeout=15,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(result, {"worker": "primary"})
        self.assertEqual(
            transports["worker-a"].timeouts,
            [INFERENCE_FAILOVER_SECONDS],
        )

    def test_remote_first_tracking_keeps_the_full_timeout(self) -> None:
        config = self._config(inference_balance="remote_first")
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)

        with self.assertRaises(InferenceUnavailable):
            backend.request(
                "detect",
                frame=self.frame,
                timeout=15,
                workload=InferenceWorkload.TRACKING,
            )

        self.assertEqual(transports["worker-a"].timeouts, [15])

    def test_incident_fallback_abandons_a_slow_worker_quickly(self) -> None:
        config = self._config(inference_balance="remote_first")
        registry, transports = self._ready_registry(config, ["worker-a"])
        transports["worker-a"].error = TimeoutError("deadline")
        backend = self._backend(config, registry)

        result = backend.request(
            "detect",
            frame=self.frame,
            timeout=3,
            workload=InferenceWorkload.INCIDENT_INITIAL,
        )

        self.assertEqual(result, {"worker": "primary"})
        self.assertEqual(
            transports["worker-a"].timeouts,
            [INFERENCE_FAILOVER_SECONDS],
        )

    def test_primary_admission_fails_over_without_shortening_execution(self) -> None:
        config = self._config()
        registry, transports = self._ready_registry(config, ["worker-a"])
        backend = self._backend(config, registry)
        seen: dict[str, float | None] = {}

        def local_request(*_args, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            seen["admission_timeout"] = kwargs.get("admission_timeout")
            raise InferenceUnavailable("waiting for admission")

        backend._local.request = local_request

        result = backend.request(
            "detect",
            frame=self.frame,
            timeout=15,
            workload=InferenceWorkload.TRACKING,
        )

        self.assertEqual(result, {"worker": "worker-a"})
        self.assertEqual(seen["timeout"], 15)
        self.assertEqual(seen["admission_timeout"], INFERENCE_FAILOVER_SECONDS)
        self.assertEqual(transports["worker-a"].timeouts, [15])

    def test_weight_edits_do_not_change_worker_generation(self) -> None:
        config = self._config(inference_balance="remote_first")
        changed = config.model_copy(update={
            "inference_balance": "weighted",
            "inference_primary_weight": 5,
            "inference_default_worker_weight": 2,
            "inference_worker_weights": {"worker-a": 9},
        })

        self.assertEqual(
            detector_routing_payload(config)["inference_mode"],
            "hybrid",
        )
        self.assertNotIn("inference_primary_weight", detector_routing_payload(config))
        registry = RemoteInferenceRegistry(lambda: config, clock=lambda: self.now)
        self.assertEqual(
            registry.config_generation(),
            RemoteInferenceRegistry(lambda: changed, clock=lambda: self.now).config_generation(),
        )


if __name__ == "__main__":
    unittest.main()
