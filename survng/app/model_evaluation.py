"""Bounded offline comparison of two object-detector model generations."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import cv2

from .config import DetectorConfig
from .detector import OpenVinoDetector, detection_failure
from .incident_utils import event_snapshot_path

LOGGER = logging.getLogger(__name__)


def _labels(objects: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("label") or "").strip()
        for item in objects
        if (
            isinstance(item, dict)
            and item.get("label")
            and item.get("snapshot_visible") is not False
        )
    }


def _reference_labels(event: dict[str, Any]) -> set[str]:
    try:
        objects = json.loads(event.get("objects_json") or "[]")
    except (TypeError, ValueError):
        return set()
    return _labels(objects if isinstance(objects, list) else [])


class ModelEvaluationRunner:
    """Own one cancellable evaluation job without touching production inference."""

    def __init__(self, get_manager: Callable[[], Any], get_config: Callable[[], Any]) -> None:
        self._get_manager = get_manager
        self._get_config = get_config
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._status: dict[str, Any] = {"status": "idle"}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status, default=str))

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._status.get("status") not in {"queued", "running"}:
                raise RuntimeError("no model evaluation is running")
            self._cancel.set()
            self._status["status"] = "cancelling"
            return dict(self._status)

    @staticmethod
    def _validated_model_path(value: str) -> str:
        path = Path(value).expanduser().resolve()
        model_roots = []
        for root in (Path("models"), Path("/models")):
            try:
                model_roots.append(root.resolve())
            except OSError:
                continue
        if path.suffix.lower() not in {".xml", ".onnx"} or not path.is_file():
            raise ValueError("model must be an existing OpenVINO XML or ONNX file")
        if not any(path == root or root in path.parents for root in model_roots):
            raise ValueError("model must be stored under the SurvNG models directory")
        if path.suffix.lower() == ".xml" and not path.with_suffix(".bin").is_file():
            raise ValueError("OpenVINO model is missing its BIN weights file")
        return str(path)

    def start(
        self,
        *,
        baseline_path: str,
        candidate_path: str,
        sample_count: int,
        confidence: float,
    ) -> dict[str, Any]:
        baseline = self._validated_model_path(baseline_path)
        candidate = self._validated_model_path(candidate_path)
        if baseline == candidate:
            raise ValueError("choose two different model generations")
        bounded_count = max(10, min(int(sample_count), 500))
        bounded_confidence = max(0.01, min(float(confidence), 0.99))
        if not math.isfinite(bounded_confidence):
            raise ValueError("confidence must be finite")
        with self._lock:
            if self._status.get("status") in {"queued", "running", "cancelling"}:
                raise RuntimeError("a model evaluation is already running")
            self._cancel = threading.Event()
            self._status = {
                "status": "queued",
                "baseline_path": baseline,
                "candidate_path": candidate,
                "sample_count_requested": bounded_count,
                "confidence": bounded_confidence,
                "progress": {"completed": 0, "total": bounded_count * 2},
                "started_at": "",
                "completed_at": "",
                "error": "",
            }
            evaluation_config = self._get_config().detector.model_copy(deep=True)
        threading.Thread(
            target=self._run,
            args=(baseline, candidate, bounded_count, bounded_confidence, evaluation_config),
            name="model-evaluation",
            daemon=True,
        ).start()
        return self.status()

    def _corpus(self, sample_count: int) -> list[tuple[dict[str, Any], Path]]:
        manager = self._get_manager()
        candidates: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
        for event in manager.events.recent(min(10000, max(1000, sample_count * 20))):
            if not event.get("snapshot_path"):
                continue
            try:
                path = event_snapshot_path(
                    manager.storage_dir,
                    event,
                    getattr(manager, "media_storage", None),
                )
            except (FileNotFoundError, PermissionError):
                continue
            sample = dict(event)
            sample["source_kind"] = "incident"
            sample["source_id"] = int(event.get("id") or 0)
            camera_id = str(event.get("camera_id") or "unknown")
            candidates[(camera_id, "incident")].append((sample, path))
        negative_limit = min(500, max(100, sample_count * 2))
        for offset in range(0, negative_limit, 100):
            audits, _total = manager.events.motion_audits(
                limit=min(100, negative_limit - offset),
                offset=offset,
                outcome="clear",
            )
            if not audits:
                break
            for audit in audits:
                if not audit.get("snapshot_path"):
                    continue
                try:
                    path = event_snapshot_path(
                        manager.storage_dir,
                        audit,
                        getattr(manager, "media_storage", None),
                    )
                except (FileNotFoundError, PermissionError):
                    continue
                sample = dict(audit)
                sample["objects_json"] = "[]"
                sample["source_kind"] = "motion_audit"
                sample["source_id"] = int(audit.get("id") or 0)
                camera_id = str(audit.get("camera_id") or "unknown")
                candidates[(camera_id, "motion_audit")].append((sample, path))
        corpus: list[tuple[dict[str, Any], Path]] = []
        sources = sorted(candidates)
        offset = 0
        while len(corpus) < sample_count and sources:
            remaining = []
            for source in sources:
                rows = candidates[source]
                if offset < len(rows):
                    corpus.append(rows[offset])
                    remaining.append(source)
                    if len(corpus) >= sample_count:
                        break
            sources = remaining
            offset += 1
        return corpus

    def _evaluate_model(
        self,
        path: str,
        corpus: list[tuple[dict[str, Any], Path]],
        confidence: float,
        completed_offset: int,
        evaluation_config: DetectorConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        config = evaluation_config.model_copy(deep=True)
        config.backend = "openvino"
        config.model_path = path
        config.model_xml = ""
        config.labels_path = ""
        config.enabled = True
        manager = self._get_manager()
        production_detector = getattr(manager, "detector", None)
        lease_factory = getattr(production_detector, "offline_device_lease", None)
        load_lease = lease_factory() if callable(lease_factory) else nullcontext()
        with load_lease:
            detector = OpenVinoDetector(config)
            status = detector.status()
        if not status.get("openvino_loaded") and not status.get("opencv_loaded"):
            raise RuntimeError(f"could not load model {path}")
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        counts: Counter[str] = Counter()
        for index, (event, image_path) in enumerate(corpus):
            if self._cancel.is_set():
                raise InterruptedError("model evaluation cancelled")
            frame = cv2.imread(str(image_path))
            if frame is None:
                objects: list[dict[str, Any]] = []
                elapsed_ms = 0.0
                error = "image_unreadable"
            else:
                started = time.perf_counter()
                lease = lease_factory() if callable(lease_factory) else nullcontext()
                with lease:
                    objects = detector.detect(frame, confidence_threshold=confidence)
                elapsed_ms = (time.perf_counter() - started) * 1000
                error = detection_failure(objects)
            labels = sorted(_labels(objects))
            counts.update(labels)
            if elapsed_ms:
                latencies.append(elapsed_ms)
            rows.append({"labels": labels, "objects": objects, "elapsed_ms": round(elapsed_ms, 2), "error": error})
            with self._lock:
                self._status["status"] = "running"
                self._status["progress"] = {
                    "completed": completed_offset + index + 1,
                    "total": len(corpus) * 2,
                }
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0
        return rows, {
            "path": path,
            "device": status.get("loaded_device") or status.get("configured_device"),
            "backend": status.get("loaded_backend"),
            "input_shape": status.get("input_shape"),
            "model_load_ms": status.get("model_load_ms"),
            "average_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95_ms": round(p95, 2),
            "frames_with_objects": sum(bool(row["labels"]) for row in rows),
            "label_counts": dict(sorted(counts.items())),
            "errors": sum(bool(row["error"]) for row in rows),
        }

    def _run(
        self,
        baseline: str,
        candidate: str,
        sample_count: int,
        confidence: float,
        evaluation_config: DetectorConfig | None = None,
    ) -> None:
        try:
            frozen_config = evaluation_config or self._get_config().detector.model_copy(deep=True)
            with self._lock:
                self._status["status"] = "running"
                self._status["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            corpus = self._corpus(sample_count)
            if len(corpus) < 10:
                raise RuntimeError("fewer than 10 readable incident snapshots are available")
            baseline_rows, baseline_summary = self._evaluate_model(
                baseline, corpus, confidence, 0, frozen_config
            )
            candidate_rows, candidate_summary = self._evaluate_model(
                candidate, corpus, confidence, len(corpus), frozen_config
            )
            comparisons = []
            agreements = 0
            stored_reference_hits = {"baseline": 0, "candidate": 0, "total": 0}
            for (event, _path), left, right in zip(corpus, baseline_rows, candidate_rows):
                reference = _reference_labels(event)
                left_labels, right_labels = set(left["labels"]), set(right["labels"])
                agreements += left_labels == right_labels
                stored_reference_hits["total"] += len(reference)
                stored_reference_hits["baseline"] += len(reference & left_labels)
                stored_reference_hits["candidate"] += len(reference & right_labels)
                if left_labels != right_labels:
                    comparisons.append({
                        "event_id": int(event.get("id") or 0) if event.get("source_kind") == "incident" else 0,
                        "source_kind": str(event.get("source_kind") or "incident"),
                        "source_id": int(event.get("source_id") or event.get("id") or 0),
                        "camera_id": str(event.get("camera_id") or ""),
                        "created_at": str(event.get("created_at") or ""),
                        "image_url": (
                            f"/api/motion-audit/{int(event.get('source_id') or 0)}/snapshot.jpg"
                            if event.get("source_kind") == "motion_audit"
                            else f"/api/events/{int(event.get('id') or 0)}/snapshot.jpg"
                        ),
                        "reference_labels": sorted(reference),
                        "baseline_labels": sorted(left_labels),
                        "candidate_labels": sorted(right_labels),
                        "candidate_only": sorted(right_labels - left_labels),
                        "baseline_only": sorted(left_labels - right_labels),
                    })
            total_reference = stored_reference_hits["total"]
            result = {
                "sample_count": len(corpus),
                "camera_count": len({str(event.get("camera_id") or "") for event, _ in corpus}),
                "source_counts": dict(sorted(Counter(
                    str(event.get("source_kind") or "incident")
                    for event, _ in corpus
                ).items())),
                "agreement_frames": agreements,
                "disagreement_frames": len(comparisons),
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "stored_evidence_recall": {
                    "baseline": round(stored_reference_hits["baseline"] / total_reference, 3) if total_reference else None,
                    "candidate": round(stored_reference_hits["candidate"] / total_reference, 3) if total_reference else None,
                    "reference_labels": total_reference,
                },
                "disagreements": comparisons[:200],
            }
            with self._lock:
                self._status.update({
                    "status": "completed",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "progress": {"completed": len(corpus) * 2, "total": len(corpus) * 2},
                    "result": result,
                })
        except InterruptedError:
            with self._lock:
                self._status.update({"status": "cancelled", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        except Exception as error:
            LOGGER.exception("model evaluation failed")
            with self._lock:
                self._status.update({"status": "failed", "error": str(error), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
