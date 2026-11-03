"""SigLIP 2 stays optional, downloads a pinned package, and keeps its own generation."""
from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from survng.app.config import SemanticSearchConfig
from survng.app.semantic_search import (
    OpenVinoManifestEncoder,
    SemanticIndex,
    SemanticSearchService,
    build_semantic_search,
)
from survng.app.siglip2_search import (
    SIGLIP2_TEXT_LENGTH,
    Siglip2Tokenizer,
    ensure_siglip2_package,
    siglip2_manifest,
    siglip2_package_dir,
    siglip2_semantic_contract,
)


class _FileHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        name = self.path.rsplit("/", 1)[-1]
        payload = self.server.files.get(name)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        self.server.requests.append(name)
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FileServer:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FileHandler)
        self.server.files = files
        self.server.requests = []
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


def _catalog() -> tuple[tuple[str, str, int], ...]:
    body = b"siglip2-weights"
    digest = hashlib.sha256(body).hexdigest()
    return (("vision.onnx", digest, len(body)),)


class Siglip2PackageTest(unittest.TestCase):
    def test_download_checks_the_digest_and_skips_a_complete_file(self) -> None:
        catalog = _catalog()
        name, _digest, _size = catalog[0]
        server = _FileServer({name: b"siglip2-weights"})
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = ensure_siglip2_package(
                directory, files=catalog, base_url=server.url,
            )
            second = ensure_siglip2_package(
                directory, files=catalog, base_url=server.url,
            )

            self.assertEqual(first, second)
            self.assertEqual(server.server.requests, [name])
            manifest = json.loads((directory / "semantic_model.json").read_text())
            self.assertEqual(manifest["implementation"], "siglip2")
            self.assertEqual(manifest["image"]["size"], siglip2_manifest()["image"]["size"])

    def test_a_short_download_is_rejected(self) -> None:
        catalog = (("vision.onnx", hashlib.sha256(b"siglip2-weights").hexdigest(), 99),)
        server = _FileServer({"vision.onnx": b"siglip2-weights"})
        self.addCleanup(server.close)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "did not match"):
                ensure_siglip2_package(
                    Path(temporary), files=catalog, base_url=server.url,
                )
            self.assertFalse((Path(temporary) / "vision.onnx").exists())
            self.assertFalse((Path(temporary) / "vision.onnx.partial").exists())

    def test_cache_path_defaults_under_storage_and_can_be_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            self.assertEqual(
                siglip2_package_dir("", storage),
                storage / "models" / "siglip2-base-patch16-224",
            )
            self.assertEqual(
                siglip2_package_dir("cache/siglip", storage),
                storage / "cache" / "siglip",
            )
            self.assertEqual(
                siglip2_package_dir(str(storage / "absolute"), storage),
                storage / "absolute",
            )


class Siglip2ContractTest(unittest.TestCase):
    def test_existing_config_stays_on_openvino_and_siglip2_needs_no_package(self) -> None:
        config = SemanticSearchConfig()
        self.assertEqual(config.implementation, "mobileclip2_openvino")
        self.assertEqual(config.siglip2_model_dir, "")
        with tempfile.TemporaryDirectory() as temporary:
            index = SemanticIndex(Path(temporary) / "survng.sqlite3")
            service = build_semantic_search(
                SemanticSearchConfig(enabled=True, implementation="siglip2"),
                index,
            )
            self.assertIsInstance(service, SemanticSearchService)
            self.assertEqual(service.status()["implementation"], "siglip2")
            self.assertEqual(service.status()["model"], "siglip2-base-patch16-224")

    def test_directory_does_not_change_the_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = SemanticIndex(Path(temporary) / "survng.sqlite3")
            first_identity, first_contract = siglip2_semantic_contract()
            second_identity, second_contract = siglip2_semantic_contract()
            stored = index.resolve_siglip2_identity()
            repeated = index.resolve_siglip2_identity()

            self.assertEqual(first_contract, second_contract)
            self.assertEqual(first_identity.generation, second_identity.generation)
            self.assertEqual(stored, repeated)
            self.assertEqual(stored.dimensions, 768)
            self.assertEqual(stored.implementation, "siglip2")
            ollama = index.resolve_ollama_identity(
                SemanticSearchConfig(implementation="ollama")
            )
            self.assertNotEqual(ollama.generation, stored.generation)

    def test_images_are_square_224_and_scaled_to_the_siglip_range(self) -> None:
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :] = (0, 0, 255)
        prepared = OpenVinoManifestEncoder.prepare_images([image], siglip2_manifest()["image"])

        self.assertEqual(prepared.shape, (1, 3, 224, 224))
        self.assertAlmostEqual(float(prepared[0, 0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(prepared[0, 2].mean()), -1.0, places=5)

    def test_tokenizer_lowercases_and_pads_to_64(self) -> None:
        tokenizer = Tokenizer(WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "red": 2, "truck": 3},
            unk_token="[UNK]",
        ))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.enable_padding(length=SIGLIP2_TEXT_LENGTH, pad_id=0, pad_token="[PAD]")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.json"
            tokenizer.save(str(path))
            encoded = Siglip2Tokenizer(path)(["RED truck", "red truck"])

        self.assertEqual(encoded["input_ids"].shape, (2, SIGLIP2_TEXT_LENGTH))
        self.assertTrue(np.array_equal(encoded["input_ids"][0], encoded["input_ids"][1]))
        self.assertEqual(int(encoded["input_ids"][0, 0]), 2)
        self.assertEqual(int(encoded["input_ids"][0, -1]), 0)


class Siglip2StartupTest(unittest.TestCase):
    def test_startup_downloads_then_loads_the_openvino_worker(self) -> None:
        class EventStore:
            def recent_compact(self, *_args):
                return []

        with tempfile.TemporaryDirectory() as temporary:
            index = SemanticIndex(Path(temporary) / "survng.sqlite3")
            service = SemanticSearchService(
                SemanticSearchConfig(enabled=True, implementation="siglip2"),
                index,
                Path(),
                {},
            )
            created = []

            def ensure(directory, should_stop=None, files=None, base_url=None):
                del should_stop, files, base_url
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "semantic_model.json").write_text(
                    json.dumps(siglip2_manifest()),
                    encoding="utf-8",
                )
                return directory

            class Worker:
                def __init__(self, model_dir, manifest, device, identity=None):
                    created.append((Path(model_dir), manifest, device, identity))
                    self.identity = identity

                def close(self) -> None:
                    return None

            with patch("survng.app.siglip2_search.ensure_siglip2_package", ensure), patch(
                "survng.app.semantic_search.IsolatedOpenVinoManifestEncoder",
                Worker,
            ):
                service.start(EventStore(), Path(temporary))
                deadline = time.monotonic() + 2.0
                while service.status()["state"] != "ready" and time.monotonic() < deadline:
                    time.sleep(0.01)

            status = service.status()
            self.assertEqual(status["state"], "ready")
            self.assertEqual(status["device"], "GPU")
            self.assertEqual(status["model"], "siglip2-base-patch16-224")
            self.assertEqual(created[0][0], Path(temporary) / "models" / "siglip2-base-patch16-224")
            self.assertEqual(created[0][2], "GPU")
            self.assertEqual(created[0][3].implementation, "siglip2")
            service.close()
            self.assertEqual(service.status()["state"], "stopped")
