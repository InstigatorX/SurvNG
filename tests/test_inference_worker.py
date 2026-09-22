from __future__ import annotations

import asyncio
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import Mock

from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient
import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference import InferenceWorkload
from survng.app.inference_runtime.protocol import (
    WorkerRegistration,
    decode_packet,
    encode_packet,
)
from survng.app.inference_runtime.registry import RemoteInferenceRegistry
from survng.app.inference_worker_routes import (
    InferenceWorkerRouteDependencies,
    WebSocketRegistryTransport,
    create_inference_worker_router,
)
from survng.app.security import (
    authenticate_inference_worker,
    hash_api_token,
)
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


class InferenceWorkerRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DetectorConfig(
            enabled=False,
            object_worker_count=1,
            tracking={"enabled": False},
        )
        self.registry = RemoteInferenceRegistry(lambda: self.config)
        token_hash = hash_api_token("worker-secret")
        app = FastAPI()
        app.include_router(create_inference_worker_router(
            InferenceWorkerRouteDependencies(
                registry=self.registry,
                authenticate=lambda value: authenticate_inference_worker(
                    value,
                    token_hash,
                ),
            )
        ))
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.registry.close()

    def test_authenticated_worker_registers_and_renews_lease(self) -> None:
        with self.client.websocket_connect(
            "/api/inference/workers/connect",
            headers={"Authorization": "Bearer worker-secret"},
        ) as websocket:
            websocket.send_text(WorkerRegistration(
                worker_id="worker-a",
                roles=["object"],
            ).model_dump_json())
            welcome = websocket.receive_json()
            websocket.send_json({
                "type": "ready",
                "worker_id": "worker-a",
                "connection_generation": (
                    welcome["connection_generation"]
                ),
                "config_generation": welcome["config_generation"],
                "statuses": {"object": {"ready": True}},
            })
            websocket.send_json({
                "type": "heartbeat",
                "worker_id": "worker-a",
                "connection_generation": (
                    welcome["connection_generation"]
                ),
                "pending_requests": 0,
            })
            heartbeat_ack = websocket.receive_json()

            self.assertTrue(heartbeat_ack["accepted"])
            self.assertEqual(self.registry.status()["ready"], 1)

    def test_invalid_worker_credential_is_rejected(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.client.websocket_connect(
                "/api/inference/workers/connect",
                headers={"Authorization": "Bearer wrong-secret"},
            ):
                pass

        self.assertEqual(raised.exception.code, 1008)


if __name__ == "__main__":
    unittest.main()
