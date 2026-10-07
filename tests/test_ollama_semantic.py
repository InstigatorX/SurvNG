from __future__ import annotations

import base64
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from survng.app.config import AppConfig, SemanticSearchConfig
from survng.app.semantic_search import (
    OLLAMA_IMAGE_BATCH_SIZE,
    OLLAMA_SEARCH_QUERY_PREFIX,
    SemanticIndex,
    SemanticSearchService,
    UnavailableSemanticSearch,
    build_semantic_search,
    ollama_semantic_contract,
)


class _EmbedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        self.server.requests.append((self.path, body))
        mode = self.server.mode
        count = len(body["input"]) if isinstance(body["input"], list) else 1
        if mode == "http_error":
            raw = json.dumps({"error": "model not found"}).encode()
            status = 404
        elif mode == "mismatch":
            raw = json.dumps({"embeddings": [[1.0, 0.0] for _ in range(count)]}).encode()
            status = 200
        elif mode == "short":
            raw = json.dumps({"embeddings": []}).encode()
            status = 200
        else:
            dimensions = int(body["dimensions"])
            raw = json.dumps({
                "embeddings": [[0.0] * (dimensions - 1) + [1.0] for _ in range(count)],
            }).encode()
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        return


class _EmbedServer:
    def __init__(self, mode: str = "ok") -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
        self.server.requests = []
        self.server.mode = mode
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _config(**overrides: object) -> SemanticSearchConfig:
    values = {
        "enabled": True,
        "implementation": "ollama",
        "ollama_dimensions": 128,
    }
    values.update(overrides)
    return SemanticSearchConfig(**values)


class OllamaSemanticConfigTest(unittest.TestCase):
    def test_existing_config_keeps_openvino_and_ollama_defaults(self) -> None:
        config = AppConfig.model_validate({
            "semantic_search": {
                "enabled": False,
                "implementation": "mobileclip2_openvino",
                "model_dir": "",
            },
        })

        self.assertEqual(config.semantic_search.implementation, "mobileclip2_openvino")
        self.assertEqual(config.semantic_search.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(config.semantic_search.ollama_model, "embeddinggemma-2")
        self.assertEqual(config.semantic_search.ollama_dimensions, 768)

    def test_ollama_url_model_and_dimensions_are_validated(self) -> None:
        self.assertEqual(
            SemanticSearchConfig(ollama_base_url="").ollama_base_url,
            "http://127.0.0.1:11434",
        )
        self.assertEqual(SemanticSearchConfig(ollama_dimensions="256").ollama_dimensions, 256)
        with self.assertRaises(ValidationError):
            SemanticSearchConfig(ollama_base_url="http://user:secret@127.0.0.1:11434")
        with self.assertRaises(ValidationError):
            SemanticSearchConfig(ollama_model=" ")
        with self.assertRaises(ValidationError):
            SemanticSearchConfig(ollama_dimensions=100)

    def test_enabled_ollama_does_not_require_a_model_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = SemanticIndex(Path(temporary) / "survng.sqlite3")
            service = build_semantic_search(_config(model_dir=""), index)
            packaged = build_semantic_search(SemanticSearchConfig(enabled=True), index)

            self.assertIsInstance(service, SemanticSearchService)
            self.assertEqual(service.status()["implementation"], "ollama")
            self.assertIsInstance(packaged, UnavailableSemanticSearch)
            self.assertIn("model directory", packaged.reason)


class OllamaSemanticEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _EmbedServer()
        self.temporary = tempfile.TemporaryDirectory()
        self.index = SemanticIndex(Path(self.temporary.name) / "survng.sqlite3")

    def tearDown(self) -> None:
        self.server.close()
        self.temporary.cleanup()

    def _encoder(self, config: SemanticSearchConfig | None = None):
        from survng.app.semantic_search import OllamaEmbeddingEncoder

        selected = config or _config(ollama_base_url=self.server.url)
        identity = self.index.resolve_ollama_identity(selected)
        return OllamaEmbeddingEncoder(selected, identity), selected

    def test_text_request_uses_the_search_prefix_and_dimensions(self) -> None:
        encoder, _config_value = self._encoder()

        vectors = encoder.encode_text(["red truck", "a truck"])

        path, body = self.server.server.requests[-1]
        self.assertEqual(path, "/api/embed")
        self.assertEqual(body["model"], "embeddinggemma-2")
        self.assertEqual(body["dimensions"], 128)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["input"], [
            f"{OLLAMA_SEARCH_QUERY_PREFIX}red truck",
            f"{OLLAMA_SEARCH_QUERY_PREFIX}a truck",
        ])
        self.assertEqual(vectors.shape, (2, 128))
        encoder.close()

    def test_images_are_jpeg_objects_without_a_task_prefix(self) -> None:
        encoder, _config_value = self._encoder()
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[0, 0] = (10, 20, 30)
        count = OLLAMA_IMAGE_BATCH_SIZE + 1

        vectors = encoder.encode_images([image] * count)

        self.assertEqual(len(self.server.server.requests), 2)
        self.assertEqual(vectors.shape, (count, 128))
        sizes = [len(body["input"]) for _path, body in self.server.server.requests]
        self.assertEqual(sizes, [OLLAMA_IMAGE_BATCH_SIZE, 1])
        for _path, body in self.server.server.requests:
            self.assertEqual(body["dimensions"], 128)
            self.assertTrue(all(set(item) == {"image"} for item in body["input"]))
            raw = base64.b64decode(body["input"][0]["image"])
            self.assertTrue(raw.startswith(b"\xff\xd8"))
            self.assertNotIn(OLLAMA_SEARCH_QUERY_PREFIX, json.dumps(body["input"]))
        encoder.close()

    def test_dimension_mismatch_and_http_failure_are_encoder_errors(self) -> None:
        encoder, _config_value = self._encoder()
        self.server.server.mode = "mismatch"
        with self.assertRaisesRegex(RuntimeError, "returned 2 dimensions"):
            encoder.encode_text(["gate"])

        self.server.server.mode = "short"
        with self.assertRaisesRegex(RuntimeError, "did not match the request"):
            encoder.encode_text(["gate"])

        self.server.server.mode = "http_error"
        with self.assertRaisesRegex(RuntimeError, "HTTP 404: model not found"):
            encoder.encode_text(["gate"])
        encoder.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            encoder.encode_text(["gate"])

    def test_model_or_dimension_change_starts_a_new_generation(self) -> None:
        base = _config(ollama_base_url=self.server.url)
        moved = base.model_copy(update={"ollama_base_url": "http://10.1.1.8:11434"})
        renamed = base.model_copy(update={"ollama_model": "embeddinggemma-2:440m"})
        resized = base.model_copy(update={"ollama_dimensions": 256})

        base_identity, _contract = ollama_semantic_contract(base)
        moved_identity, _contract = ollama_semantic_contract(moved)
        renamed_identity, _contract = ollama_semantic_contract(renamed)
        resized_identity, _contract = ollama_semantic_contract(resized)
        stored = self.index.resolve_ollama_identity(base)
        repeated = self.index.resolve_ollama_identity(base)
        resized_stored = self.index.resolve_ollama_identity(resized)

        self.assertEqual(base_identity, moved_identity)
        self.assertEqual(stored, repeated)
        self.assertEqual(stored.generation, base_identity.generation)
        self.assertNotEqual(renamed_identity.generation, base_identity.generation)
        self.assertNotEqual(resized_identity.generation, base_identity.generation)
        self.assertEqual(resized_stored.dimensions, 256)
        self.assertNotEqual(resized_stored.generation, stored.generation)

    def test_startup_probes_ollama_and_does_not_fall_back_to_cpu(self) -> None:
        class EventStore:
            def recent_compact(self, *_args):
                return []

        service = SemanticSearchService(
            _config(ollama_base_url=self.server.url),
            self.index,
            Path(),
            {},
        )

        def unexpected_openvino(*_args, **_kwargs):
            raise AssertionError("OpenVINO encoder must not start for Ollama")

        with patch(
            "survng.app.semantic_search.IsolatedOpenVinoManifestEncoder",
            unexpected_openvino,
        ):
            service.start(EventStore(), Path(self.temporary.name))
            deadline = time.monotonic() + 2.0
            while service.status()["state"] != "ready" and time.monotonic() < deadline:
                time.sleep(0.01)

        status = service.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["device"], "ollama")
        self.assertEqual(status["model"], "embeddinggemma-2")
        self.assertFalse(status["fallback_active"])
        self.assertTrue(self.server.server.requests)
        self.assertTrue(
            self.server.server.requests[0][1]["input"][0].startswith(OLLAMA_SEARCH_QUERY_PREFIX)
        )
        service.close()
        self.assertEqual(service.status()["state"], "stopped")
