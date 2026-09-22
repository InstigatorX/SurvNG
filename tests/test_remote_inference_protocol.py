from __future__ import annotations

import unittest

import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference import InferenceUnavailable, InferenceWorkload
from survng.app.inference_runtime.protocol import (
    ProtocolError,
    WorkerHeartbeat,
    WorkerReady,
    WorkerRegistration,
    decode_frame,
    decode_packet,
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


if __name__ == "__main__":
    unittest.main()
