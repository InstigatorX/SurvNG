from __future__ import annotations

import threading
import unittest

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference import InferenceUnavailable, InferenceWorkload
from survng.app.inference_runtime.protocol import (
    ProtocolError,
    WorkerHeartbeat,
    WorkerReady,
    WorkerRegistration,
    WorkerUpgradeStatus,
    decode_frame,
    decode_packet,
    encode_binary_packet,
    encode_packet,
)
from survng.app.inference_runtime.registry import (
    RemoteInferenceRegistry,
    detector_config_generation,
)


class _FakeTransport:
    def __init__(self, result: object = None) -> None:
        self.result = result
        self.requests: list[bytes] = []
        self.closed: list[str] = []
        self.control: list[dict] = []

    def send_control(self, message: dict) -> None:
        self.control.append(message)

    def request(self, packet: bytes, timeout: float) -> bytes:
        self.requests.append(packet)
        message, _frame = decode_packet(packet)
        return encode_packet({
            "type": "response",
            "request_id": message["request_id"],
            "connection_generation": message["connection_generation"],
            "ok": True,
            "result": self.result,
        })

    def close(self, reason: str) -> None:
        self.closed.append(reason)


class RemoteInferenceProtocolTests(unittest.TestCase):
    def test_packet_round_trip_preserves_raw_frame_and_binary_results(self) -> None:
        frame = np.arange(3 * 4 * 3, dtype=np.uint8).reshape((3, 4, 3))
        packet = encode_packet(
            {
                "type": "request",
                "payload": {"heatmap_png": b"\x89PNG\r\n"},
            },
            frame=frame,
        )

        message, frame_bytes = decode_packet(packet)
        decoded = decode_frame(message, frame_bytes)

        self.assertEqual(message["payload"]["heatmap_png"], b"\x89PNG\r\n")
        self.assertIsNotNone(decoded)
        np.testing.assert_array_equal(decoded, frame)

    def test_packet_rejects_trailing_bytes_without_frame_metadata(self) -> None:
        packet = encode_packet({"type": "heartbeat"}) + b"unexpected"

        with self.assertRaises(ProtocolError):
            decode_packet(packet)

    def test_packet_rejects_invalid_frames(self) -> None:
        with self.assertRaises(ProtocolError):
            encode_packet(
                {"type": "request"},
                frame=np.zeros((8, 8), dtype=np.uint8),
            )

    def test_binary_packet_round_trip_is_bounded(self) -> None:
        packet = encode_binary_packet(
            {"type": "model_chunk", "digest": "a" * 64},
            b"model-bytes",
        )

        message, payload = decode_packet(packet)

        self.assertEqual(message["type"], "model_chunk")
        self.assertEqual(message["binary"]["byte_count"], len(payload))
        self.assertEqual(payload, b"model-bytes")


class RemoteInferenceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.config = DetectorConfig(
            enabled=False,
            object_worker_count=1,
            tracking={"enabled": False},
        )
        self.registry = RemoteInferenceRegistry(
            lambda: self.config,
            lease_seconds=6.0,
            clock=lambda: self.now,
        )

    def tearDown(self) -> None:
        self.registry.close()

    def _register_ready(
        self,
        worker_id: str = "worker-a",
        *,
        transport: _FakeTransport | None = None,
        roles: list[str] | None = None,
    ) -> tuple[_FakeTransport, dict]:
        active_transport = transport or _FakeTransport(
            [{"label": "person", "confidence": 0.9}]
        )
        registration = WorkerRegistration(
            worker_id=worker_id,
            roles=roles or ["object"],
            devices=["GPU"],
        )
        welcome = self.registry.register(registration, active_transport)
        accepted = self.registry.mark_ready(WorkerReady(
            worker_id=worker_id,
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={"object": {"ready": True}},
        ))
        self.assertTrue(accepted)
        return active_transport, welcome

    def test_registration_routes_matching_requests(self) -> None:
        transport, welcome = self._register_ready()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        result = self.registry.request(
            "object",
            "detect",
            frame=frame,
            workload=InferenceWorkload.INCIDENT_INITIAL,
            timeout=1.0,
            payload={"confidence_threshold": 0.5},
        )

        self.assertEqual(
            result,
            [{"label": "person", "confidence": 0.9}],
        )
        message, frame_bytes = decode_packet(transport.requests[0])
        self.assertEqual(
            message["connection_generation"],
            welcome["connection_generation"],
        )
        self.assertEqual(
            message["workload"],
            int(InferenceWorkload.INCIDENT_INITIAL),
        )
        np.testing.assert_array_equal(
            decode_frame(message, frame_bytes),
            frame,
        )

    def test_invalid_frame_does_not_stick_a_worker_reservation(self) -> None:
        self._register_ready()

        with self.assertRaises(ProtocolError):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((8, 8, 3), dtype=np.float32),
                workload=InferenceWorkload.INCIDENT_INITIAL,
                timeout=1.0,
            )

        worker = self.registry.status()["workers"][0]
        self.assertEqual(worker["pending_requests"], 0)
        self.assertEqual(worker["failed_requests"], 0)

    def test_worker_queue_is_not_added_to_the_primary_reservation(self) -> None:
        class _BlockingTransport(_FakeTransport):
            def __init__(self) -> None:
                super().__init__([])
                self.started = threading.Event()
                self.release = threading.Event()

            def request(self, packet: bytes, timeout: float) -> bytes:
                self.started.set()
                self.release.wait(timeout)
                return super().request(packet, timeout)

            def close(self, reason: str) -> None:
                self.release.set()
                super().close(reason)

        transport = _BlockingTransport()
        _transport, welcome = self._register_ready(transport=transport)
        self.assertTrue(self.registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-a",
            connection_generation=welcome["connection_generation"],
            pending_requests=5,
        )))
        worker = threading.Thread(
            target=self.registry.request,
            args=("object", "detect"),
            kwargs={
                "frame": np.zeros((4, 4, 3), dtype=np.uint8),
                "workload": InferenceWorkload.INCIDENT_INITIAL,
                "timeout": 2.0,
            },
        )
        worker.start()
        self.addCleanup(transport.release.set)
        self.addCleanup(worker.join, 2.0)
        self.assertTrue(transport.started.wait(1.0))

        loads = self.registry.ready_role_loads("object")

        transport.release.set()
        worker.join(2.0)
        self.assertEqual(loads[0]["pending"], 5)

    def test_duplicate_worker_id_fences_previous_transport(self) -> None:
        first, first_welcome = self._register_ready()
        second = _FakeTransport()

        second_welcome = self.registry.register(
            WorkerRegistration(worker_id="worker-a", roles=["object"]),
            second,
        )

        self.assertGreater(
            second_welcome["connection_generation"],
            first_welcome["connection_generation"],
        )
        self.assertEqual(first.closed, ["worker connection was replaced"])
        self.assertFalse(self.registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-a",
            connection_generation=first_welcome["connection_generation"],
        )))

    def test_expired_lease_is_removed_and_closed(self) -> None:
        transport, _welcome = self._register_ready()
        self.now += 7.0

        status = self.registry.status()

        self.assertEqual(status["connected"], 0)
        self.assertEqual(transport.closed, ["worker lease expired"])
        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.INCIDENT_INITIAL,
                timeout=1.0,
            )

    def test_initial_sync_lease_outlives_normal_heartbeat_lease(self) -> None:
        transport = _FakeTransport()
        welcome = self.registry.register(
            WorkerRegistration(worker_id="worker-sync", roles=["object"]),
            transport,
            initial_lease_seconds=30.0,
        )
        self.now += 7.0

        status = self.registry.status()

        self.assertEqual(status["connected"], 1)
        self.assertGreater(
            status["workers"][0]["lease_remaining_seconds"],
            20.0,
        )
        self.assertTrue(self.registry.mark_ready(WorkerReady(
            worker_id="worker-sync",
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={"object": {"ready": True}},
        )))

    def test_loaded_object_engine_without_ready_flag_is_routed(self) -> None:
        transport = _FakeTransport(
            [{"label": "person", "confidence": 0.9}]
        )
        welcome = self.registry.register(
            WorkerRegistration(worker_id="worker-loaded", roles=["object"]),
            transport,
        )

        accepted = self.registry.mark_ready(WorkerReady(
            worker_id="worker-loaded",
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={
                "object": {
                    "enabled": True,
                    "loaded_backend": "openvino",
                    "openvino_loaded": True,
                    "loaded_device": "GPU",
                    "isolation": {
                        "worker_alive": True,
                        "all_workers_alive": True,
                    },
                }
            },
        ))

        self.assertTrue(accepted)
        self.assertEqual(self.registry.status()["ready"], 1)
        result = self.registry.request(
            "object",
            "detect",
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            workload=InferenceWorkload.INCIDENT_INITIAL,
            timeout=1.0,
        )
        self.assertEqual(result, [{"label": "person", "confidence": 0.9}])
        self.assertEqual(len(transport.requests), 1)

    def test_dead_object_engine_without_ready_flag_is_not_routed(self) -> None:
        transport = _FakeTransport()
        welcome = self.registry.register(
            WorkerRegistration(worker_id="worker-dead", roles=["object"]),
            transport,
        )

        accepted = self.registry.mark_ready(WorkerReady(
            worker_id="worker-dead",
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={
                "object": {
                    "enabled": True,
                    "loaded_backend": "openvino",
                    "openvino_loaded": True,
                    "isolation": {
                        "worker_alive": False,
                        "all_workers_alive": False,
                    },
                }
            },
        ))

        self.assertTrue(accepted)
        self.assertEqual(self.registry.status()["ready"], 0)
        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.INCIDENT_INITIAL,
                timeout=1.0,
            )

    def test_unready_role_is_not_routed(self) -> None:
        transport = _FakeTransport()
        welcome = self.registry.register(
            WorkerRegistration(worker_id="worker-unready", roles=["object"]),
            transport,
        )

        accepted = self.registry.mark_ready(WorkerReady(
            worker_id="worker-unready",
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={"object": {"ready": False}},
        ))

        self.assertTrue(accepted)
        self.assertEqual(self.registry.status()["ready"], 0)
        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.INCIDENT_INITIAL,
                timeout=1.0,
            )

    def test_stale_config_generation_is_not_routed(self) -> None:
        self._register_ready()
        self.config = self.config.model_copy(
            update={"device": "GPU"},
            deep=True,
        )

        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.INCIDENT_INITIAL,
                timeout=1.0,
            )
        self.assertNotEqual(
            detector_config_generation(self.config),
            self.registry.status()["workers"][0]["config_generation"],
        )

    def test_upgrade_sends_the_primary_commit(self) -> None:
        target = "ab" * 20
        registration_version = "cd" * 20
        transport = _FakeTransport()
        welcome = self.registry.register(
            WorkerRegistration(
                worker_id="worker-a",
                roles=["object"],
                software_version=registration_version,
            ),
            transport,
        )
        with self.assertRaises(InferenceUnavailable) as unready:
            self.registry.request_upgrade("worker-a", target)
        self.assertIn("not ready", str(unready.exception))
        self.assertEqual(transport.control, [])
        self.assertTrue(self.registry.mark_ready(WorkerReady(
            worker_id="worker-a",
            connection_generation=welcome["connection_generation"],
            config_generation=welcome["config_generation"],
            statuses={"object": {"ready": True}},
        )))

        with self.assertRaises(InferenceUnavailable) as unavailable:
            self.registry.request_upgrade("missing", target)
        self.assertIn("not connected", str(unavailable.exception))

        payload = self.registry.request_upgrade("worker-a", target)

        self.assertEqual(payload["upgrade_phase"], "requested")
        self.assertEqual(payload["software_version"], registration_version)
        self.assertEqual(transport.control, [{
            "type": "upgrade",
            "worker_id": "worker-a",
            "connection_generation": welcome["connection_generation"],
            "target_sha": target,
        }])

        self.registry.note_upgrade(WorkerUpgradeStatus(
            worker_id="worker-a",
            connection_generation=welcome["connection_generation"],
            phase="accepted",
            detail="upgrade requested",
            target_sha=target,
        ))
        self.registry.note_upgrade(WorkerUpgradeStatus(
            worker_id="worker-a",
            connection_generation=welcome["connection_generation"] + 1,
            phase="failed",
            detail="stale",
            target_sha=target,
        ))

        worker = self.registry.status()["workers"][0]
        self.assertEqual(worker["upgrade_phase"], "accepted")
        self.assertEqual(worker["upgrade_detail"], "upgrade requested")
        self.assertEqual(worker["upgrade_target_sha"], target)

    def test_upgrade_records_a_send_failure(self) -> None:
        class _BrokenTransport(_FakeTransport):
            def send_control(self, message: dict) -> None:
                raise ConnectionError("closed")

        target = "ab" * 20
        self._register_ready(transport=_BrokenTransport())

        with self.assertRaises(InferenceUnavailable):
            self.registry.request_upgrade("worker-a", target)

        worker = self.registry.status()["workers"][0]
        self.assertEqual(worker["upgrade_phase"], "failed")
        self.assertEqual(worker["upgrade_detail"], "upgrade request was not sent")

        with self.assertRaises(InferenceUnavailable) as unavailable:
            self.registry.request_upgrade("worker-a", "abc")
        self.assertIn("full git commit", str(unavailable.exception))

    def test_timeout_prefers_another_worker_until_the_queue_drains(self) -> None:
        class _TimeoutTransport(_FakeTransport):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def request(self, packet: bytes, timeout: float) -> bytes:
                del packet, timeout
                self.calls += 1
                raise TimeoutError("deadline")

        slow = _TimeoutTransport()
        _transport, welcome = self._register_ready(
            "worker-slow",
            transport=slow,
        )
        fast, _fast_welcome = self._register_ready("worker-fast")
        self.assertTrue(self.registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-slow",
            connection_generation=welcome["connection_generation"],
            pending_requests=2,
        )))
        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.TRACKING,
                timeout=1.0,
                worker_weights={"worker-slow": 100, "worker-fast": 1},
            )

        held = {
            item["worker_id"]: item
            for item in self.registry.status()["workers"]
        }
        self.assertTrue(held["worker-slow"]["routing_hold"])
        self.assertEqual(slow.calls, 1)
        self.registry.request(
            "object",
            "detect",
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            workload=InferenceWorkload.TRACKING,
            timeout=1.0,
            worker_weights={"worker-slow": 4, "worker-fast": 1},
        )
        self.assertEqual(len(fast.requests), 1)
        self.assertEqual(slow.calls, 1)

    def test_completed_request_clears_a_routing_hold(self) -> None:
        class _OnceTimeout(_FakeTransport):
            def __init__(self) -> None:
                super().__init__(result={"ok": True})
                self.failed = False

            def request(self, packet: bytes, timeout: float) -> bytes:
                if not self.failed:
                    self.failed = True
                    raise TimeoutError("deadline")
                return super().request(packet, timeout)

        transport, welcome = self._register_ready(transport=_OnceTimeout())
        self.assertTrue(self.registry.heartbeat(WorkerHeartbeat(
            worker_id="worker-a",
            connection_generation=welcome["connection_generation"],
            pending_requests=2,
        )))
        with self.assertRaises(InferenceUnavailable):
            self.registry.request(
                "object",
                "detect",
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                workload=InferenceWorkload.TRACKING,
                timeout=1.0,
            )
        self.assertTrue(self.registry.status()["workers"][0]["routing_hold"])

        self.registry.request(
            "object",
            "detect",
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            workload=InferenceWorkload.TRACKING,
            timeout=1.0,
        )

        self.assertFalse(self.registry.status()["workers"][0]["routing_hold"])
        self.assertEqual(len(transport.requests), 1)


if __name__ == "__main__":
    unittest.main()
