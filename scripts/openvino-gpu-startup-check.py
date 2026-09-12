"""Repeat cold-cache GPU worker startup without cameras or application state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survng.app.config import DetectorConfig
from survng.app.detector import detection_failure
from survng.app.inference import InferenceSupervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not args.model.is_file() or args.iterations < 1:
        parser.error("a local model file and positive iteration count are required")
    for iteration in range(args.iterations):
        with tempfile.TemporaryDirectory(prefix="survng-cold-gpu-") as cache:
            supervisor = InferenceSupervisor(DetectorConfig(
                enabled=True, model_path=str(args.model.resolve()), device="GPU",
                object_worker_count=2, cache_enabled=True, cache_dir=cache,
            ))
            started = time.monotonic()
            try:
                assert supervisor.start(), "GPU worker startup failed"
                # Inspect each pool member, not only the aggregate/primary.
                for worker in supervisor._object_workers:
                    assert worker.cached_status()["loaded_device"].upper().startswith("GPU")
                for _ in range(4):
                    result = supervisor.detect_refinement(np.zeros((360, 640, 3), np.uint8))
                    assert not detection_failure(result), "refinement inference failed"
                isolation = supervisor.isolation_status()
                assert isolation["all_workers_alive"] and isolation["restart_count"] == 0
                assert isolation["crash_count"] == 0 and not isolation["fallback_active"]
                print(json.dumps({"iteration": iteration + 1, "gpu_workers": 2,
                                  "inferences": 4, "crashes": 0,
                                  "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
            finally:
                supervisor.stop()
                assert supervisor.stop_resource_tracker(), "resource tracker did not stop"


if __name__ == "__main__":
    main()
