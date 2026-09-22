from __future__ import annotations

import asyncio
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import Mock

import numpy as np

from survng.app.inference import InferenceWorkload
from survng.app.inference_runtime.protocol import decode_packet, encode_packet
from survng.app.inference_worker_routes import WebSocketRegistryTransport
from survng.inference_worker import (
    InferenceWorkerClient,
    load_or_create_worker_id,
    worker_websocket_url,
)


class InferenceWorkerClientTests(unittest.TestCase):
    def test_worker_url_preserves_base_path(self) -> None:
        self.assertEqual(
            worker_websocket_url("https://survng.example/survng"),
            (
                "wss://survng.example/survng/"
                "api/inference/workers/connect"
            ),
        )
        self.assertEqual(
            worker_websocket_url(
                "ws://10.0.0.2:8088/api/inference/workers/connect"
            ),
            "ws://10.0.0.2:8088/api/inference/workers/connect",
        )

    def test_worker_id_is_stable_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker-id"

            first = load_or_create_worker_id(path)
            second = load_or_create_worker_id(path)

            self.assertEqual(first, second)
            self.assertTrue(first.startswith("worker-"))
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode),
                0o600,
            )

    def test_request_dispatches_to_matching_workload(self) -> None:
        client = InferenceWorkerClient(
            server_url="http://survng.internal:8088",
            token="secret",
            worker_id="worker-a",
            name="worker",
            roles=["object"],
        )
        supervisor = Mock()
        supervisor.detect_initial.return_value = [
            {"label": "person", "confidence": 0.9}
        ]
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        packet = encode_packet(
            {
                "type": "request",
                "request_id": "request-1",
                "connection_generation": 4,
                "config_generation": "config-a",
                "role": "object",
                "operation": "detect",
                "workload": int(
                    InferenceWorkload.INCIDENT_INITIAL
                ),
                "deadline_unix_ms": int(
                    (time.time() + 5.0) * 1000
                ),
                "payload": {"confidence_threshold": 0.4},
            },
            frame=frame,
        )

        response_packet = client.handle_packet(
            supervisor,
            packet,
            connection_generation=4,
            config_generation="config-a",
        )
        response, trailing = decode_packet(response_packet)

        self.assertFalse(trailing)
        self.assertTrue(response["ok"])
        self.assertEqual(
            response["result"],
            [{"label": "person", "confidence": 0.9}],
        )
        called_frame, called_threshold = (
            supervisor.detect_initial.call_args.args
        )
        np.testing.assert_array_equal(called_frame, frame)
        self.assertEqual(called_threshold, 0.4)

    def test_request_rejects_stale_connection_generation(self) -> None:
        client = InferenceWorkerClient(
            server_url="http://survng.internal:8088",
            token="secret",
            worker_id="worker-a",
            name="worker",
            roles=["object"],
        )
        supervisor = Mock()
        packet = encode_packet({
            "type": "request",
            "request_id": "request-1",
            "connection_generation": 2,
            "config_generation": "config-a",
            "role": "object",
            "operation": "detect",
            "workload": int(InferenceWorkload.INCIDENT_INITIAL),
            "deadline_unix_ms": int((time.time() + 5.0) * 1000),
            "payload": {},
        })

        response_packet = client.handle_packet(
            supervisor,
            packet,
            connection_generation=3,
            config_generation="config-a",
        )
        response, _trailing = decode_packet(response_packet)

        self.assertFalse(response["ok"])
        supervisor.detect_initial.assert_not_called()


class WebSocketRegistryTransportTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_transport_correlates_response_with_waiting_thread(
        self,
    ) -> None:
        transport = WebSocketRegistryTransport(
            asyncio.get_running_loop()
        )
        request = encode_packet({
            "type": "request",
            "request_id": "request-1",
        })
        waiting = asyncio.create_task(
            asyncio.to_thread(transport.request, request, 1.0)
        )
        outbound = await asyncio.wait_for(
            transport._outbound.get(),
            timeout=1.0,
        )
        self.assertEqual(outbound, request)
        response = encode_packet({
            "type": "response",
            "request_id": "request-1",
            "ok": True,
        })

        self.assertTrue(transport.deliver(response))
        self.assertEqual(await waiting, response)


if __name__ == "__main__":
    unittest.main()
