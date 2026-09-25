from __future__ import annotations

import asyncio
import json
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
    encode_binary_packet,
    encode_packet,
)
from survng.app.inference_runtime.model_sync import (
    ModelBundleCatalog,
    WorkerModelCache,
    worker_config_for_roles,
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

    def test_worker_role_config_does_not_start_unrequested_engines(self) -> None:
        config = DetectorConfig(
            enabled=True,
            model_path="/models/object.onnx",
            face_recognition_enabled=True,
            face_embedding_model_path="/models/face.xml",
            face_landmark_model_path="/models/landmark.xml",
            tracking={
                "reid_enabled": True,
                "reid_model_path": "/models/reid.xml",
            },
            depth={
                "enabled": True,
                "model_path": "/models/depth.xml",
            },
        )

        worker_config = worker_config_for_roles(config, ["object"])

        self.assertTrue(worker_config.enabled)
        self.assertEqual(worker_config.object_worker_count, 1)
        self.assertFalse(worker_config.face_recognition_enabled)
        self.assertFalse(worker_config.tracking.reid_enabled)
        self.assertFalse(worker_config.depth.enabled)
        self.assertEqual(worker_config.inference_mode, "local")
        supervisor = InferenceSupervisor(
            worker_config,
            enabled_roles=["object"],
        )
        self.addCleanup(supervisor.stop)
        self.assertTrue(supervisor._object.start_enabled)
        self.assertFalse(supervisor._face.start_enabled)
        self.assertFalse(supervisor._reid.start_enabled)
        self.assertFalse(supervisor._depth.start_enabled)


class ModelSynchronizationTests(unittest.TestCase):
    def test_catalog_tracks_model_content_and_materializes_worker_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "object.onnx"
            source.write_bytes(b"first-model")
            (root / "metadata.yaml").write_text(
                "task: detect\n",
                encoding="utf-8",
            )
            config = DetectorConfig(
                enabled=True,
                model_path=str(source),
                object_worker_count=2,
                tracking={"enabled": False},
            )
            catalog = ModelBundleCatalog()
            first = catalog.prepare(config, ["object"])
            source.write_bytes(b"updated-model")
            second = catalog.prepare(config, ["object"])

            self.assertNotEqual(
                first.config_generation,
                second.config_generation,
            )
            binding = second.manifest.bindings["model_path"]
            first_model = next(
                item for item in first.manifest.files
                if item.path == binding
            )
            second_model = next(
                item for item in second.manifest.files
                if item.path == binding
            )
            self.assertNotEqual(first_model.digest, second_model.digest)
            self.assertEqual(first.config.object_worker_count, 1)
            self.assertTrue(any(
                item.path.endswith("/metadata.yaml")
                for item in second.manifest.files
            ))

            cache = WorkerModelCache(root / "cache")
            websocket = Mock()
            websocket.recv.side_effect = [
                *[
                    encode_binary_packet(
                        {
                            "type": "model_chunk",
                            "digest": model_file.digest,
                            "offset": 0,
                            "total_size": model_file.size,
                        },
                        second.sources[model_file.digest].read_bytes(),
                    )
                    for model_file in second.manifest.files
                ],
                json.dumps({
                    "type": "model_sync_complete",
                    "generation": second.manifest.generation,
                }),
            ]

            cache.receive(websocket, second.manifest)
            worker_config = cache.materialize(
                second.config,
                second.manifest,
            )

            worker_path = Path(worker_config.model_path)
            self.assertEqual(worker_path.read_bytes(), b"updated-model")
            self.assertEqual(
                (worker_path.parent / "metadata.yaml").read_text(
                    encoding="utf-8"
                ),
                "task: detect\n",
            )
            self.assertTrue(
                worker_path.is_relative_to(root / "cache" / "bundles")
            )

    def test_cached_model_digests_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = WorkerModelCache(Path(temporary))
            cache.blob_dir.mkdir(parents=True)
            invalid = cache.blob_dir / ("a" * 64)
            invalid.write_bytes(b"not-the-advertised-content")

            self.assertEqual(cache.available_digests(), [])
            self.assertFalse(invalid.exists())


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
        self.catalog = ModelBundleCatalog()
        self.registry = RemoteInferenceRegistry(
            lambda: self.config,
            generation_provider=self.catalog.config_generation,
        )
        token_hash = hash_api_token("worker-secret")
        app = FastAPI()
        app.include_router(create_inference_worker_router(
            InferenceWorkerRouteDependencies(
                registry=self.registry,
                model_catalog=self.catalog,
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
            model_sync = websocket.receive_json()
            self.assertEqual(model_sync["type"], "model_sync_complete")
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

    def test_primary_pushes_missing_model_before_worker_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "object.onnx"
            model_path.write_bytes(b"remote-object-model")
            self.config = DetectorConfig(
                enabled=True,
                model_path=str(model_path),
                object_worker_count=1,
                tracking={"enabled": False},
            )
            with self.client.websocket_connect(
                "/api/inference/workers/connect",
                headers={"Authorization": "Bearer worker-secret"},
            ) as websocket:
                websocket.send_text(WorkerRegistration(
                    worker_id="worker-model",
                    roles=["object"],
                ).model_dump_json())
                welcome = websocket.receive_json()
                packet = websocket.receive_bytes()
                complete = websocket.receive_json()

            message, payload = decode_packet(packet)
            manifest_file = welcome["model_manifest"]["files"][0]
            self.assertEqual(message["type"], "model_chunk")
            self.assertEqual(message["digest"], manifest_file["digest"])
            self.assertEqual(payload, b"remote-object-model")
            self.assertEqual(
                complete,
                {
                    "type": "model_sync_complete",
                    "generation": welcome["model_manifest"]["generation"],
                },
            )
            with self.client.websocket_connect(
                "/api/inference/workers/connect",
                headers={"Authorization": "Bearer worker-secret"},
            ) as websocket:
                websocket.send_text(WorkerRegistration(
                    worker_id="worker-model",
                    roles=["object"],
                    cached_model_digests=[manifest_file["digest"]],
                ).model_dump_json())
                cached_welcome = websocket.receive_json()
                cached_complete = websocket.receive_json()

            self.assertEqual(
                cached_complete["generation"],
                cached_welcome["model_manifest"]["generation"],
            )


if __name__ == "__main__":
    unittest.main()
