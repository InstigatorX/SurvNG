#!/usr/bin/env python3
"""Isolated, bounded replay of existing supervisor policy; never connects to cameras."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np

from survng.app.config import DetectorConfig
from survng.app.inference_runtime.supervisor import InferenceSupervisor
from survng.app.inference_runtime.worker_topology import apply_inference_thread_limits
from survng.app.local_observability import _process_gpu, _process_memory


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def fixtures(paths: list[str]) -> list[np.ndarray]:
    if paths:
        frames = [cv2.imread(path) for path in paths]
        if any(frame is None for frame in frames):
            raise ValueError("could not read a fixture")
        return frames
    # Stable portrait/landscape inputs exercise resize/padding and transfer costs.
    rng = np.random.default_rng(308)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            for h, w in ((1080, 1920), (1920, 1080), (480, 640))]


def percentiles(values: list[float]) -> dict:
    return {f"p{p}": round(float(np.percentile(values, p)), 3)
            for p in (50, 95, 99)} if values else {}


def parent_preprocess_probe(worker, frames, repeats=5, requests=100):
    """Separate no-IPC resize timing; it must not be inferred from IPC latency."""
    results = []
    for frame in frames:
        worker._prepare_object_frame(frame)
        for repeat in range(repeats):
            samples = []
            for _ in range(requests):
                started = time.perf_counter()
                prepared, _scale = worker._prepare_object_frame(frame)
                samples.append((time.perf_counter() - started) * 1000)
            results.append({"shape": list(frame.shape), "repeat": repeat,
                            "prepared_bytes": prepared.nbytes, "ms": samples})
    return results


def run(supervisor, frames, scenario, cameras, requests):
    methods = {
        "tracking": [supervisor.detect_tracking],
        "initial": [supervisor.detect_initial],
        "refinement": [supervisor.detect_refinement],
        "mixed": [supervisor.detect_tracking, supervisor.detect_enrichment,
                  supervisor.detect_initial, supervisor.detect_refinement],
    }[scenario]
    # Fixed-size waves bound both queued references and active callers. A barrier
    # produces simultaneous camera equivalents without an unbounded executor FIFO.
    samples = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cameras) as executor:
        for offset in range(0, requests, cameras):
            count = min(cameras, requests - offset)
            barrier = threading.Barrier(count)

            def call(index):
                method = methods[index % len(methods)]
                barrier.wait(timeout=30)
                begin = time.perf_counter()
                result = method(frames[index % len(frames)], (.25, .7)[index % 2])
                elapsed = (time.perf_counter() - begin) * 1000
                statuses = sorted({item["status"] for item in result if "status" in item})
                return {"request": index, "workload": method.__name__, "ms": elapsed,
                        "outcome": ",".join(statuses) or "completed",
                        "digest": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()}

            futures = [executor.submit(call, i) for i in range(offset, offset + count)]
            samples.extend(future.result() for future in futures)
    elapsed = time.perf_counter() - started
    statuses = [worker.status() for worker in supervisor._object_workers]
    completed = [sample for sample in samples if sample["outcome"] == "completed"]
    return {
        "scenario": scenario, "cameras": cameras, "seconds": elapsed,
        "completed_per_second": len(completed) / elapsed,
        "outcomes": dict(Counter(sample["outcome"] for sample in samples)),
        "latency_ms": {name: percentiles([s["ms"] for s in samples
                                          if s["workload"] == name and s["outcome"] == "completed"])
                       for name in sorted({s["workload"] for s in samples})},
        "samples": samples,
        # Existing telemetry windows are trailing/cumulative, not per-run deltas.
        "workers": statuses,
        "device_gate": supervisor.workload_status(),
        "memory_at_end": [{**_process_memory(status["isolation"]["worker_pid"]),
                           "gpu": _process_gpu(status["isolation"]["worker_pid"])}
                          for status in statuses],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--frames", nargs="*", default=[])
    parser.add_argument("--cameras", type=positive, nargs="+", default=[1, 4, 8, 32])
    parser.add_argument("--runs", type=positive, default=5)
    parser.add_argument("--requests", type=positive, default=100)
    parser.add_argument("--soak-seconds", type=positive, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = fixtures(args.frames)
    config = DetectorConfig(enabled=True, model_path=args.model, device=args.device,
                            object_worker_count=args.workers, cache_enabled=False,
                            warmup_enabled=True, tracking={"enabled": False})
    supervisor = InferenceSupervisor(config)
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "openvino")},
        "opencv": cv2.__version__, "device": args.device,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "weights_sha256": (hashlib.sha256(Path(args.model).with_suffix(".bin").read_bytes()).hexdigest()
                           if Path(args.model).suffix == ".xml" else None),
        "synthetic": not args.frames, "fixture_shapes": [list(f.shape) for f in frames],
        "fixture_sha256": [hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
        "cache_enabled": False, "warmup_enabled": True,
        "max_concurrent_refinements": config.max_concurrent_refinements,
        "thread_policy": apply_inference_thread_limits(
            object_worker_count=args.workers, environ=dict(os.environ),
        ),
        "runs": [],
    }
    try:
        started = time.perf_counter()
        if not supervisor.start():
            raise RuntimeError("benchmark workers failed to start")
        report["cold_start_seconds"] = time.perf_counter() - started
        report["model"] = supervisor.inspect_model(args.model)
        report["initial_workers"] = [w.status() for w in supervisor._object_workers]
        if any(not s.get("openvino_loaded") for s in report["initial_workers"]):
            raise RuntimeError("expected OpenVINO backend; refusing to benchmark fallback")
        report["parent_preprocess_probe"] = parent_preprocess_probe(
            supervisor._object_workers[0], [*frames, cv2.resize(frames[0], (3840, 2160))],
        )
        for cameras in args.cameras:
            for scenario in ("tracking", "initial", "refinement", "mixed"):
                for repeat in range(args.runs):
                    result = run(supervisor, frames, scenario, cameras, args.requests)
                    result["repeat"] = repeat
                    report["runs"].append(result)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"completed {scenario} / {cameras} camera equivalents", flush=True)
        soak_deadline = time.monotonic() + args.soak_seconds
        while time.monotonic() < soak_deadline:
            result = run(supervisor, frames, "mixed", max(args.cameras), args.requests)
            result["soak"] = True
            report["runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("completed mixed soak wave", flush=True)
    finally:
        supervisor.stop()
        supervisor.stop_resource_tracker()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
