"""Bounded, low-priority recorded stills for isolated review findings.

This adapter deliberately has no event store or publisher. In particular it
does not reuse recorded incident refinement, which can establish live alerts.
Timestamps identify requested sampling positions, not verified source PTS.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import detection_failure
from .evidence_work import (
    EvidenceWorkPreempted,
    cancellable_evidence_work,
    run_evidence_process,
)
from .ffmpeg_hw import RECORDED_FRAME_INPUT_THREAD_ARGS, RECORDED_FRAME_OUTPUT_THREAD_ARGS
from .recording_process.index import RECORDING_FINALIZE_GRACE_SECONDS


SAMPLE_INTERVAL_SECONDS = 5.0
MAX_SAMPLES = 12
MAX_SEGMENTS = 128
MAX_FRAME_DIMENSION = 640
MAX_ENCODED_FRAME_BYTES = 4 * 1024 * 1024
MAX_RUN_SECONDS = 60.0
DECODE_TIMEOUT_SECONDS = 5.0
ADMISSION_WAIT_SECONDS = 0.05
POLICY_VERSION = "recording-review-stills-v1"


class _SampleUnavailable(Exception):
    """A safe, fixed reason for an incomplete sample, without private paths."""


class RecordingReviewRuntime:
    def __init__(
        self,
        *,
        recorder: Any,
        detector_provider: Callable[[], Any],
        decode_budget: Any,
        detector_config_provider: Callable[[], Any] | None = None,
    ) -> None:
        self.recorder = recorder
        self.detector_provider = detector_provider
        self.decode_budget = decode_budget
        self.detector_config_provider = detector_config_provider
        self._identity_lock = threading.Lock()
        self._loaded_files: dict[int, tuple[tuple[Any, int], dict[str, Any]]] = {}
        # The manager constructs this adapter after starting inference. Pin
        # loaded model files now, rather than at the first later review request.
        if callable(getattr(self.detector_provider(), "isolation_status", None)):
            self.analysis_identity()

    def manifest(self, camera_id: str, source: str, start: float, end: float) -> list[dict]:
        """Read only the existing index; never discover/backfill the archive."""
        if source not in {"main", "live"}:
            raise ValueError("unsupported recording source")
        if not all(math.isfinite(value) for value in (start, end)) or not 0 < end - start <= 60:
            raise ValueError("recording review requires a bounded minute")
        rows = self.recorder.recording_rows_between(
            camera_id, start, end, source=source, discover_missing=False,
        )
        if len(rows) > MAX_SEGMENTS:
            raise ValueError("recording window has too many segments")
        manifest: list[dict] = []
        now = time.time()
        for row in rows:
            row_start = float(row.get("start_epoch") or 0)
            row_end = float(row.get("end_epoch") or 0)
            modified = float(row.get("modified_at") or 0)
            size = int(row.get("size_bytes") or 0)
            if not all(math.isfinite(value) for value in (row_start, row_end, modified)):
                continue
            if row_end <= start or row_start >= end or row_end <= row_start:
                continue
            if max(row_end, modified) + RECORDING_FINALIZE_GRACE_SECONDS > now:
                # Closed minutes can still overlap the recorder's active tail.
                # A later request obtains a changed manifest once it finalizes.
                continue
            path = str(row.get("path") or "")
            if not path or size <= 0:
                continue
            manifest.append({
                "path": path,
                "start_epoch": row_start,
                "end_epoch": row_end,
                "source": source,
                "size_bytes": size,
                "modified_at": modified,
                "stream_fingerprint": str(row.get("stream_fingerprint") or ""),
            })
        return sorted(manifest, key=lambda row: (row["start_epoch"], row["path"]))

    def analysis_identity(self) -> str:
        """Identify loaded models, not newer files an old worker has not read."""
        with self._identity_lock:
            detector = self.detector_provider()
            config = (
                self.detector_config_provider()
                if self.detector_config_provider is not None
                else getattr(detector, "config", None)
            )
            if hasattr(config, "model_dump"):
                policy = config.model_dump(mode="json")
            elif isinstance(config, dict):
                policy = dict(config)
            else:
                raise RuntimeError("recording review detector policy is unavailable")
            isolation_status = getattr(detector, "isolation_status", None)
            if not callable(isolation_status):
                # Non-isolated adapters cannot attest a loaded generation.
                files: Any = self._model_file_signatures(policy)
            else:
                status = isolation_status()
                instances = status.get("instances") or [
                    {"index": index + 1, "worker_pid": pid, "generation": status.get("generation", 0)}
                    for index, pid in enumerate(status.get("worker_pids") or [status.get("worker_pid")])
                ]
                files = []
                current_indexes = set()
                for position, instance in enumerate(instances):
                    index = int(instance.get("index", position + 1))
                    current_indexes.add(index)
                    generation = (instance.get("worker_pid"), int(instance.get("generation") or 0))
                    prior = self._loaded_files.get(index)
                    # A dead worker did not load new weights. Preserve its
                    # signature until its replacement has actually started.
                    if prior is None or (generation[0] is not None and prior[0] != generation):
                        prior = (generation, self._model_file_signatures(policy))
                        self._loaded_files[index] = prior
                    files.append((index, prior[1]))
                self._loaded_files = {key: value for key, value in self._loaded_files.items() if key in current_indexes}
                files.sort(key=lambda item: item[0])
            # Process identity detects reloads, but is not part of cache
            # identity: unchanged models can reuse results across restarts.
            payload = {"version": POLICY_VERSION, "policy": policy, "files": files}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _model_file_signatures(policy: dict) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for key in ("model_path", "model_xml", "coreml_model_path", "labels_path"):
            raw_path = str(policy.get(key) or "")
            if not raw_path:
                continue
            path = Path(raw_path)
            paths = [path]
            if path.suffix.lower() == ".xml":
                paths.append(path.with_suffix(".bin"))
            # Core ML packages keep their weights below the package directory.
            # Known files are bounded and do not scan model/cache trees.
            if path.suffix.lower() == ".mlpackage":
                paths.extend((
                    path / "Manifest.json",
                    path / "Data/com.apple.CoreML/model.mlmodel",
                    path / "Data/com.apple.CoreML/weights/weight.bin",
                ))
            for candidate in paths:
                try:
                    stat = candidate.stat()
                    files[str(candidate)] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
                except OSError:
                    files[str(candidate)] = None
        return files

    def analyze(self, job: dict, stop_event: threading.Event) -> Iterator[dict]:
        """At most twelve individually admitted stills; no resources cross yield."""
        start = float(job["start_epoch"])
        end = float(job["end_epoch"])
        if not all(math.isfinite(value) for value in (start, end)) or not 0 < end - start <= 60:
            raise ValueError("recording review requires a bounded minute")
        rows = list(job["manifest"])
        if len(rows) > MAX_SEGMENTS:
            raise ValueError("recording window has too many segments")
        deadline = time.monotonic() + MAX_RUN_SECONDS
        for index in range(MAX_SAMPLES):
            target = start + SAMPLE_INTERVAL_SECONDS * (index + 0.5)
            if target >= end or stop_event.is_set():
                return
            result = {
                "timestamp": target,
                "sample_time_kind": "requested",
                "objects": [],
                "status": "unavailable",
            }
            if time.monotonic() >= deadline:
                yield {**result, "status": "deferred", "reason": "review_time_budget"}
                continue
            row = next((row for row in rows if row["start_epoch"] <= target < row["end_epoch"]), None)
            if row is None:
                yield {**result, "reason": "recording_unavailable"}
                continue
            try:
                with cancellable_evidence_work(
                    lambda: stop_event.is_set() or time.monotonic() >= deadline,
                ):
                    objects = self._sample(job, row, target, deadline, stop_event)
                if stop_event.is_set():
                    return
                if any(isinstance(item, dict) and item.get("status") == "inference_deferred" for item in objects):
                    result.update(status="deferred", reason="inference_busy")
                elif detection_failure(objects):
                    result.update(status="failed", reason="detector_unavailable")
                else:
                    result.update(status="sampled", objects=objects)
            except EvidenceWorkPreempted:
                if stop_event.is_set():
                    return
                result.update(status="deferred", reason="review_time_budget" if time.monotonic() >= deadline else "inference_busy")
            except _SampleUnavailable as error:
                reason = str(error)
                result.update(status="deferred" if reason == "decoder_busy" else "unavailable", reason=reason)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, cv2.error):
                result.update(status="failed", reason="sample_failed")
            yield result

    def _sample(self, job: dict, row: dict, target: float, deadline: float, stop_event: threading.Event) -> list[dict]:
        # Do not fall back to raw detect(): that would bypass optional inference
        # admission and could delay immediate security confirmation.
        detector = self.detector_provider()
        detect = getattr(detector, "detect_enrichment", None)
        if not callable(detect):
            return [{"status": "detector_unavailable"}]
        cancelled = lambda: stop_event.is_set() or time.monotonic() >= deadline
        # Historical wall-clock times would unfairly jump ahead of current
        # incident decoders in this pool's oldest-first queue.
        workflow = self.decode_budget.reserve_workflow(
            maximum_frames=1, camera_id=str(job["camera_id"]), incident_epoch=float("inf"),
            deadline=min(deadline, time.monotonic() + ADMISSION_WAIT_SECONDS), cancelled=cancelled,
        )
        if workflow is None:
            raise _SampleUnavailable("decoder_busy")
        with workflow:
            token = self.recorder.acquire_recording_for_playback(row, ttl_seconds=MAX_RUN_SECONDS + 30)
            if token is None:
                raise _SampleUnavailable("recording_unavailable")
            try:
                path = Path(str(row["path"]))
                try:
                    stat = path.stat()
                except OSError:
                    raise _SampleUnavailable("recording_unavailable") from None
                if stat.st_size != row["size_bytes"] or abs(stat.st_mtime - row["modified_at"]) > 0.000001:
                    raise _SampleUnavailable("recording_changed")
                process_lease = self.decode_budget.acquire_process(
                    incident_epoch=float("inf"),
                    deadline=min(deadline, time.monotonic() + ADMISSION_WAIT_SECONDS), cancelled=cancelled,
                )
                if process_lease is None:
                    raise _SampleUnavailable("decoder_busy")
                with process_lease:
                    frame = self._decode(path, target - float(row["start_epoch"]), deadline)
                objects = list(detect(frame) or [])
                if detection_failure(objects) or any(
                    isinstance(item, dict) and item.get("status") == "inference_deferred" for item in objects
                ):
                    return objects
                height, width = frame.shape[:2]
                return _safe_objects(objects, width, height)
            finally:
                self.recorder.release_recording_playback(token)

    def _decode(self, path: Path, offset: float, deadline: float) -> np.ndarray:
        # Accurate input seeking decodes from the preceding keyframe, not the
        # entire minute. Each tiny request can be cancelled and reaped safely.
        command = [
            self.recorder.ffmpeg_path, "-nostdin", "-v", "error",
            *RECORDED_FRAME_INPUT_THREAD_ARGS,
            "-ss", f"{max(0.0, offset):.6f}", "-i", str(path),
            "-frames:v", "1", "-an", "-sn", "-dn",
            "-vf", f"scale={MAX_FRAME_DIMENSION}:{MAX_FRAME_DIMENSION}:force_original_aspect_ratio=decrease",
            "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3",
            *RECORDED_FRAME_OUTPUT_THREAD_ARGS, "pipe:1",
        ]
        result = run_evidence_process(command, timeout=min(DECODE_TIMEOUT_SECONDS, max(0.001, deadline - time.monotonic())))
        success = result.returncode == 0 and bool(result.stdout) and not result.stderr.strip()
        self.decode_budget.record_ffmpeg("cpu", success=success)
        # Corruption can produce a concealed image and exit successfully; do
        # not turn decoder errors into a trustworthy clean-negative sample.
        if not success or len(result.stdout) > MAX_ENCODED_FRAME_BYTES:
            raise _SampleUnavailable("decode_failed")
        frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > MAX_FRAME_DIMENSION:
            raise _SampleUnavailable("decode_failed")
        return frame


def _safe_objects(objects: list[dict], width: int, height: int) -> list[dict]:
    result = []
    for item in objects[:200]:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("box")
        if not isinstance(label, str) or not label or not isinstance(box, dict):
            continue
        try:
            confidence = float(item["confidence"])
            values = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in [confidence, *values]) or not 0 <= confidence <= 1:
            continue
        normalized = [max(0.0, min(1.0, value / dimension)) for value, dimension in zip(values, (width, height, width, height))]
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        result.append({
            "label": label[:128], "confidence": confidence,
            "box": dict(zip(("x1", "y1", "x2", "y2"), normalized)),
        })
    return result
