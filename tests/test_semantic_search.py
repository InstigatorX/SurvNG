from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from survng.app.semantic_search import (
    HuggingFaceJsonTokenizer,
    OpenVinoManifestEncoder,
    SemanticEvidence,
    SemanticIndex,
    SemanticModelIdentity,
    SemanticSearchService,
    _semantic_encoder_worker_main,
    fingerprint_model_package,
    normalized_matrix,
    _semantic_text_inputs,
    _prepare_siglip2_images,
    _prepare_fixed_pil_images,
    validate_semantic_runtime_manifest,
)


class SemanticIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "events.db"
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("create table events (id integer primary key)")
            connection.executemany("insert into events(id) values (?)", [(1,), (2,)])
        self.index = SemanticIndex(self.database_path)
        self.identity = SemanticModelIdentity("test", "model-a", "prep-a", 3)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_huggingface_json_tokenizer_emits_ids_and_attention_mask(self) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "white": 4, "truck": 5}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", 2), ("<eos>", 3)],
        )
        path = Path(self.temporary.name) / "tokenizer.json"
        tokenizer.save(str(path))

        runtime = HuggingFaceJsonTokenizer(path, {
            "max_length": 6,
            "pad_token_id": 0,
            "pad_token": "<pad>",
            "padding_side": "right",
        })

        encoded = runtime(["white truck"])
        np.testing.assert_array_equal(encoded["input_ids"], [[2, 4, 5, 3, 0, 0]])
        np.testing.assert_array_equal(encoded["attention_mask"], [[1, 1, 1, 1, 0, 0]])

    def test_siglip2_runtime_requires_cross_modal_validation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing cross-modal"):
            validate_semantic_runtime_manifest({"implementation": "siglip2_openvino"})

        with self.assertRaisesRegex(RuntimeError, "failed cross-modal"):
            validate_semantic_runtime_manifest({
                "implementation": "siglip2_openvino",
                "validation": {"cross_modal": {"maximum_cosine_error": 0.001}},
            })

        validate_semantic_runtime_manifest({
            "implementation": "siglip2_openvino",
            "validation": {"cross_modal": {"maximum_cosine_error": 0.00005}},
        })

    def test_semantic_text_inputs_maps_multiple_manifest_inputs(self) -> None:
        tokens = {
            "input_ids": np.asarray([[1, 2]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
        }
        mapped = _semantic_text_inputs({
            "inputs": {
                "input_ids": "tokens",
                "attention_mask": "mask",
            }
        }, tokens, object())

        self.assertEqual(set(mapped), {"tokens", "mask"})
        np.testing.assert_array_equal(mapped["tokens"], tokens["input_ids"])
        np.testing.assert_array_equal(mapped["mask"], tokens["attention_mask"])

    def test_fixed_image_preprocessing_preserves_manifest_shape(self) -> None:
        image = np.zeros((80, 240, 3), dtype=np.uint8)
        prepared = OpenVinoManifestEncoder.prepare_images([image], {
            "size": 224,
            "resize_mode": "fixed",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        })

        self.assertEqual(prepared.shape, (1, 3, 224, 224))

    def test_siglip2_preprocessing_packs_aspect_aware_patches(self) -> None:
        prepared = _prepare_siglip2_images(
            [np.zeros((80, 240, 3), dtype=np.uint8)],
            {
                "patch_size": 16,
                "max_num_patches": 256,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
        )

        self.assertEqual(prepared["pixel_values"].shape, (1, 256, 768))
        self.assertEqual(prepared["pixel_attention_mask"].shape, (1, 256))
        self.assertEqual(prepared["spatial_shapes"].shape, (1, 2))
        rows, columns = prepared["spatial_shapes"][0]
        self.assertEqual(int(prepared["pixel_attention_mask"].sum()), rows * columns)
        self.assertGreater(columns, rows)

    def test_fixed_pil_preprocessing_uses_manifest_normalization(self) -> None:
        prepared = _prepare_fixed_pil_images(
            [np.zeros((80, 240, 3), dtype=np.uint8)],
            {"size": 224, "mean": [0.5] * 3, "std": [0.5] * 3},
        )

        self.assertEqual(prepared.shape, (1, 3, 224, 224))
        np.testing.assert_array_equal(prepared, -1.0)

    def test_normalization_rejects_invalid_embeddings(self) -> None:
        with self.assertRaises(ValueError):
            normalized_matrix([0.0, 0.0])
        with self.assertRaises(ValueError):
            normalized_matrix([1.0, float("nan")])

    def test_search_ranks_cosine_similarity_and_preserves_evidence(self) -> None:
        evidence = [
            SemanticEvidence(1, "gate", "2026-08-03T12:00:00+00:00", "full_frame", "frame", "one.webp"),
            SemanticEvidence(2, "driveway", "2026-08-03T12:01:00+00:00", "object_crop", "car:0", "two.webp", "car", (1, 2, 3, 4)),
        ]
        self.assertEqual(
            self.index.upsert(evidence, [[1, 0, 0], [0.8, 0.2, 0]], self.identity),
            2,
        )

        hits = self.index.search([1, 0, 0], self.identity, limit=2)

        self.assertEqual([hit.event_id for hit in hits], [1, 2])
        self.assertEqual(hits[1].bbox, (1, 2, 3, 4))
        self.assertGreater(hits[0].score, hits[1].score)

    def test_generations_are_isolated(self) -> None:
        evidence = [SemanticEvidence(1, "gate", "now", "full_frame", "frame", "one.webp")]
        self.index.upsert(evidence, [[1, 0, 0]], self.identity)
        other = SemanticModelIdentity("test", "model-b", "prep-a", 3)
        self.index.upsert(evidence, [[0, 1, 0]], other)

        self.assertEqual(self.index.coverage(self.identity)["event_count"], 1)
        self.assertEqual(self.index.coverage(other)["event_count"], 1)
        self.assertEqual(self.index.search([1, 0, 0], self.identity)[0].score, 1.0)

    def test_model_fingerprint_detects_same_size_change_beyond_first_chunk(self) -> None:
        model_dir = Path(self.temporary.name) / "model"
        model_dir.mkdir()
        model_file = model_dir / "weights.bin"
        model_file.write_bytes(b"a" * (1024 * 1024 + 16))
        before = fingerprint_model_package(model_dir)
        with model_file.open("r+b") as handle:
            handle.seek(1024 * 1024 + 8)
            handle.write(b"b")

        self.assertNotEqual(before, fingerprint_model_package(model_dir))

    def test_upsert_is_idempotent_per_generation_and_source(self) -> None:
        evidence = [SemanticEvidence(1, "gate", "now", "full_frame", "frame", "one.webp")]
        self.index.upsert(evidence, [[1, 0, 0]], self.identity)
        self.index.upsert(evidence, [[0, 1, 0]], self.identity)

        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 1, "event_count": 1})
        self.assertAlmostEqual(
            self.index.search([0, 1, 0], self.identity)[0].score,
            1.0,
            places=3,
        )

    def test_motion_metadata_is_not_semantic_evidence(self) -> None:
        from survng.app.config import SemanticSearchConfig

        self.index.upsert(
            [SemanticEvidence(1, "boiler", "now", "full_frame", "frame", "old.webp")],
            [[1, 0, 0]],
            self.identity,
        )
        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()
        motion_only = {
            "id": 1,
            "camera_id": "boiler",
            "snapshot_path": "snapshot.webp",
            "objects_json": '[{"status":"motion_qualification","motion_qualification":{"accepted":true}}]',
        }

        self.assertFalse(service.queue_event(motion_only))
        service._index_event(motion_only)
        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 0, "event_count": 0})

    def test_service_indexes_full_frame_and_object_crop(self) -> None:
        from survng.app.config import SemanticSearchConfig

        image_path = Path(self.temporary.name) / "snapshot.jpg"
        cv2.imwrite(str(image_path), np.full((100, 200, 3), 127, dtype=np.uint8))

        class FakeEncoder:
            identity = self.identity

            def encode_images(self, images):
                self.calls = getattr(self, "calls", [])
                self.calls.append([image.shape for image in images])
                vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                return np.asarray(vectors[:len(images)], dtype=np.float32)

            def encode_text(self, texts):
                return np.asarray([[1, 0, 0]], dtype=np.float32)

            def close(self):
                return

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        encoder = FakeEncoder()
        service.encoder = encoder
        service._storage_dir = Path(self.temporary.name)
        written = service.index_event({
            "id": 1, "camera_id": "gate", "created_at": "now",
            "snapshot_path": "snapshot.jpg",
            "objects_json": '[{"label":"car","box":{"x1":10,"y1":20,"x2":110,"y2":80}}]',
        })

        self.assertEqual(written, 2)
        self.assertEqual(encoder.calls, [[(100, 200, 3), (60, 100, 3)]])
        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 2, "event_count": 1})

        # A second pass for the same model generation performs no inference.
        written = service.index_event({
            "id": 1, "camera_id": "gate", "created_at": "now",
            "snapshot_path": "snapshot.jpg",
            "objects_json": '[{"label":"car","box":{"x1":10,"y1":20,"x2":110,"y2":80}}]',
        })
        self.assertEqual(written, 0)
        self.assertEqual(len(encoder.calls), 1)

        # If only crops are missing, repair them without re-encoding the frame.
        self.index.upsert(
            [SemanticEvidence(2, "gate", "now", "full_frame", "frame", "snapshot.jpg")],
            [[1, 0, 0]],
            self.identity,
        )
        service._index_event({
            "id": 2, "camera_id": "gate", "created_at": "now",
            "snapshot_path": "snapshot.jpg",
            "objects_json": '[{"label":"car","box":{"x1":10,"y1":20,"x2":110,"y2":80}}]',
        })
        self.assertEqual(encoder.calls[1], [(60, 100, 3)])
        self.assertEqual(self.index.coverage(self.identity), {"evidence_count": 4, "event_count": 2})

    def test_clone_image_generation_preserves_source_and_is_idempotent(self) -> None:
        target = SemanticModelIdentity("siglip2", "model-new", "preprocess-new", 3)
        self.index.upsert(
            [SemanticEvidence(1, "gate", "now", "full_frame", "frame", "x.webp")],
            [[1, 0, 0]],
            self.identity,
        )

        self.assertEqual(self.index.clone_image_generation(self.identity, target), 1)
        self.assertEqual(self.index.clone_image_generation(self.identity, target), 0)
        self.assertEqual(self.index.coverage(self.identity)["evidence_count"], 1)
        self.assertEqual(self.index.coverage(target)["evidence_count"], 1)

        incompatible = SemanticModelIdentity("siglip2", "other", "other-pre", 4)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.index.clone_image_generation(self.identity, incompatible)

    def test_object_crops_are_scaled_ranked_and_bounded(self) -> None:
        from survng.app.config import SemanticSearchConfig

        image_path = Path(self.temporary.name) / "snapshot.jpg"
        cv2.imwrite(str(image_path), np.full((100, 200, 3), 127, dtype=np.uint8))

        class FakeEncoder:
            identity = self.identity

            def encode_images(self, images):
                self.calls = getattr(self, "calls", [])
                self.calls.append([image.shape for image in images])
                if getattr(self, "fail", False):
                    raise RuntimeError("temporary inference failure")
                return np.asarray([[1, 0, 0]] * len(images), dtype=np.float32)

        service = SemanticSearchService(
            SemanticSearchConfig(
                enabled=True,
                index_full_frame=False,
                max_object_crops_per_event=1,
            ),
            self.index,
            Path(self.temporary.name),
            {},
        )
        encoder = FakeEncoder()
        service.encoder = encoder
        service._storage_dir = Path(self.temporary.name)
        event = {
            "id": 1,
            "camera_id": "gate",
            "created_at": "now",
            "snapshot_path": "snapshot.jpg",
            "objects": [
                {"label": "person", "confidence": 0.4, "box": [0, 0, 20, 20]},
                {
                    "label": "car",
                    "confidence": 0.9,
                    "box": [10, 10, 60, 40],
                    "detection_frame_width": 100,
                    "detection_frame_height": 50,
                },
            ],
        }
        service._index_event(event)

        self.assertEqual(encoder.calls, [[(60, 100, 3)]])
        hit = self.index.search([1, 0, 0], self.identity)[0]
        self.assertTrue(hit.source_key.startswith("car:1:"))
        self.assertEqual(hit.bbox, (20, 20, 120, 80))

        # Raising the cap adds only the newly eligible crop.
        service.config.max_object_crops_per_event = 2
        service._index_event(event)
        self.assertEqual(encoder.calls[1], [(20, 20, 3)])
        self.assertEqual(self.index.coverage(self.identity)["evidence_count"], 2)

        # A replacement detection at the same list position replaces stale evidence.
        event["objects"][0]["box"] = [0, 0, 30, 30]
        previous_keys = self.index.event_source_keys(1, self.identity, "object_crop")
        encoder.fail = True
        with self.assertRaisesRegex(RuntimeError, "temporary inference failure"):
            service._index_event(event)
        self.assertEqual(
            self.index.event_source_keys(1, self.identity, "object_crop"),
            previous_keys,
        )
        encoder.fail = False
        service._index_event(event)
        self.assertEqual(encoder.calls[3], [(30, 30, 3)])
        self.assertEqual(self.index.coverage(self.identity)["evidence_count"], 2)

    def test_backfill_skips_noop_motion_cleanup_and_boxless_crop_retry(self) -> None:
        from survng.app.config import SemanticSearchConfig

        class EventStore:
            calls = 0

            def recent_compact(inner_self, *_args):
                inner_self.calls += 1
                if inner_self.calls > 1:
                    return []
                return [
                    {"id": 2, "created_at": "old", "objects_json": "[]"},
                    {
                        "id": 1,
                        "created_at": "new",
                        "objects_json": '[{"label":"person"}]',
                    },
                ]

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True, index_full_frame=False),
            self.index,
            Path(self.temporary.name),
            {},
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()
        with patch.object(self.index, "delete_event", wraps=self.index.delete_event) as delete:
            service._backfill(EventStore())

        delete.assert_not_called()
        self.assertTrue(service._queue.empty())

    def test_close_during_initialization_cannot_publish_workers(self) -> None:
        from survng.app.config import SemanticSearchConfig

        constructing = threading.Event()
        release = threading.Event()
        closed = threading.Event()

        class FakeEncoder:
            def __init__(inner_self, *_args):
                constructing.set()
                release.wait(2.0)

            def close(inner_self):
                closed.set()

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        with patch("survng.app.semantic_search.IsolatedOpenVinoManifestEncoder", FakeEncoder):
            service.start(object(), Path(self.temporary.name))
            self.assertTrue(constructing.wait(1.0))
            close_thread = threading.Thread(target=service.close)
            close_thread.start()
            release.set()
            close_thread.join(3.0)

        self.assertFalse(close_thread.is_alive())
        self.assertTrue(closed.is_set())
        self.assertEqual(service.status()["state"], "stopped")
        self.assertIsNone(service.encoder)
        self.assertIsNone(service._thread)
        self.assertIsNone(service._backfill_thread)

    def test_initialization_retries_isolated_worker_crash_and_recovers(self) -> None:
        from survng.app.config import SemanticSearchConfig

        class EventStore:
            def recent_compact(self, *_args):
                return []

        class FakeEncoder:
            identity = self.identity
            worker_pid = 123

            def close(self):
                return

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        with (
            patch(
                "survng.app.semantic_search.IsolatedOpenVinoManifestEncoder",
                side_effect=[RuntimeError(""), FakeEncoder()],
            ) as encoder,
            patch("survng.app.semantic_search.SEMANTIC_WORKER_RETRY_INITIAL_SECONDS", 0.001),
        ):
            service.start(EventStore(), Path(self.temporary.name))
            deadline = time.monotonic() + 2.0
            while service.status()["state"] != "ready" and time.monotonic() < deadline:
                time.sleep(0.005)

        status = service.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["initialization_attempts"], 1)
        self.assertEqual(status["error"], "")
        self.assertEqual(status["retry_in_seconds"], 0.0)
        self.assertEqual(encoder.call_count, 2)
        service.close()

    def test_close_interrupts_worker_startup_retry_delay(self) -> None:
        from survng.app.config import SemanticSearchConfig

        failed = threading.Event()

        def fail(*_args):
            failed.set()
            raise RuntimeError("native worker exited")

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        with (
            patch(
                "survng.app.semantic_search.IsolatedOpenVinoManifestEncoder",
                side_effect=fail,
            ),
            patch("survng.app.semantic_search.SEMANTIC_WORKER_RETRY_INITIAL_SECONDS", 30.0),
        ):
            service.start(object(), Path(self.temporary.name))
            self.assertTrue(failed.wait(1.0))
            deadline = time.monotonic() + 1.0
            while service.status()["state"] != "recovering" and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(service.status()["state"], "recovering")
            started = time.monotonic()
            service.close()

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(service.status()["state"], "stopped")

    def test_initialization_falls_back_to_cpu_after_configured_device_crashes(self) -> None:
        from survng.app.config import SemanticSearchConfig

        class EventStore:
            def recent_compact(self, *_args):
                return []

        class FakeEncoder:
            identity = self.identity
            worker_pid = 456

            def close(self):
                return

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True, device="GPU"),
            self.index,
            Path(self.temporary.name),
            {},
        )
        with (
            patch(
                "survng.app.semantic_search.IsolatedOpenVinoManifestEncoder",
                side_effect=[
                    RuntimeError("GPU compiler crash"),
                    RuntimeError("GPU compiler crash"),
                    FakeEncoder(),
                ],
            ) as encoder,
            patch("survng.app.semantic_search.SEMANTIC_WORKER_RETRY_INITIAL_SECONDS", 0.001),
            patch("survng.app.semantic_search.SEMANTIC_WORKER_FALLBACK_DELAY_SECONDS", 0.001),
        ):
            service.start(EventStore(), Path(self.temporary.name))
            deadline = time.monotonic() + 2.0
            while service.status()["state"] != "ready" and time.monotonic() < deadline:
                time.sleep(0.005)

        status = service.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["configured_device"], "GPU")
        self.assertEqual(status["device"], "CPU")
        self.assertTrue(status["fallback_active"])
        self.assertEqual(status["initialization_attempts"], 2)
        self.assertEqual([call.args[2] for call in encoder.call_args_list], ["GPU", "GPU", "CPU"])
        service.close()

    def test_backfill_supervisor_retries_transient_failures(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        with (
            patch.object(
                service,
                "_backfill",
                side_effect=[sqlite3.OperationalError("busy"), None],
            ) as backfill,
            patch("survng.app.semantic_search.SEMANTIC_BACKFILL_RETRY_SECONDS", 0.001),
        ):
            service._run_backfill(object())

        self.assertEqual(backfill.call_count, 2)

    def test_missing_retained_snapshot_is_counted_without_faulting_service(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()
        service._storage_dir = Path(self.temporary.name)

        service._index_event({
            "id": 1,
            "snapshot_path": "expired.webp",
            "objects_json": '[{"label":"person"}]',
        })

        self.assertEqual(service.status()["skipped_missing_since_start"], 1)
        self.assertEqual(service.status()["error"], "")

    def test_live_events_have_priority_over_historical_backfill(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True), self.index, Path(self.temporary.name), {}
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()
        service._queue.put_nowait((1, next(service._queue_sequence), {"id": 1}))

        self.assertTrue(service.queue_event({
            "id": 2,
            "snapshot_path": "live.webp",
            "objects_json": '[{"label":"person"}]',
        }))

        priority, _sequence, event = service._queue.get_nowait()
        self.assertEqual(priority, 0)
        self.assertEqual(event["id"], 2)

    def test_historical_backfill_reserves_capacity_for_live_events(self) -> None:
        from survng.app.config import SemanticSearchConfig

        service = SemanticSearchService(
            SemanticSearchConfig(enabled=True, worker_queue_size=16),
            self.index,
            Path(self.temporary.name),
            {},
        )
        service.encoder = type("Encoder", (), {"identity": self.identity})()

        self.assertEqual(service._live_queue_reserve, 4)
        for event_id in range(12):
            service._queue.put_nowait(
                (1, next(service._queue_sequence), {"id": event_id + 1})
            )
        self.assertFalse(service._history_queue_has_capacity())
        self.assertTrue(service.queue_event({
            "id": 99,
            "snapshot_path": "live.webp",
            "objects_json": '[{"label":"person"}]',
        }))

    def test_semantic_worker_uses_distinct_process_name(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.sent = []

            def send(self, value) -> None:
                self.sent.append(value)

            def recv(self):
                return {"id": 1, "op": "shutdown"}

            def close(self) -> None:
                return

        class FakeEncoder:
            def __init__(self, *_args, **_kwargs) -> None:
                return

            def close(self) -> None:
                return

        connection = FakeConnection()
        with (
            patch("survng.app.semantic_search.OpenVinoManifestEncoder", FakeEncoder),
            patch("survng.app.inference._set_worker_process_name") as set_name,
            patch("survng.app.inference._disable_worker_core_dumps"),
        ):
            _semantic_encoder_worker_main(connection, "/models", {}, "GPU")

        set_name.assert_called_once_with("semantic")
        self.assertEqual(connection.sent[0]["type"], "ready")
        self.assertEqual(connection.sent[-1], {"id": 1, "type": "stopped"})


if __name__ == "__main__":
    unittest.main()
