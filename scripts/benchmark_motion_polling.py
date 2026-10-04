#!/usr/bin/env python3
"""Isolated before/after CPU benchmark; never opens production databases.

Run with .venv/bin/python scripts/benchmark_motion_polling.py. The baseline is
the playback-fix commit before these optimizations; override --baseline as needed.
Reports component costs, not an estimate of whole-system CPU savings.
"""
from __future__ import annotations

import argparse
import ast
import gc
import importlib
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from survng.app.database_polling import database_polling_session
from survng.app.event_store import EventStore
from survng.app.motion_pipeline.adaptive_stages import _background_statistics, _difference_statistics


def original_function(ref, module_name, name):
    path = module_name.replace(".", "/") + ".py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(vars(importlib.import_module(module_name)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def measure(label, before, after, iterations):
    samples = [[], []]
    for work in (before, after):
        work()
    for repeat in range(7):
        # Alternate order so a changing live workload does not always penalize
        # the same version. Process CPU excludes other processes' CPU time.
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            gc.collect()
            cpu, wall = time.process_time(), time.perf_counter()
            for _ in range(iterations):
                (before, after)[index]()
            samples[index].append(((time.process_time() - cpu) * 1000 / iterations,
                                   (time.perf_counter() - wall) * 1000 / iterations))
    values = [(statistics.median(x[0] for x in group), statistics.median(x[1] for x in group)) for group in samples]
    print(f"{label}: CPU {values[0][0]:.3f} -> {values[1][0]:.3f} ms "
          f"({100 * (1 - values[1][0] / values[0][0]):.1f}% less); "
          f"wall {values[0][1]:.3f} -> {values[1][1]:.3f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="0c119dd")
    args = parser.parse_args()
    baseline = original_function(args.baseline, "survng.app.motion_pipeline.adaptive_stages", "_difference_statistics")
    rng = np.random.default_rng(725)
    for name, frame in [
        ("quiet", rng.integers(0, 12, (360, 640), dtype=np.uint8)),
        ("busy", rng.integers(0, 256, (360, 640), dtype=np.uint8)),
        ("flat", np.full((360, 640), 7, dtype=np.uint8)),
    ]:
        assert baseline(frame) == _difference_statistics(frame)
        measure(f"640x360 threshold statistics ({name})", lambda: baseline(frame), lambda: _difference_statistics(frame), 40)

    delta = rng.uniform(0, 255, (360, 640)).astype(np.float32)
    def background_before():
        median = float(np.median(delta))
        return median, float(np.median(np.abs(delta - median)))
    assert background_before() == _background_statistics(delta)
    measure("640x360 background median/MAD", background_before, lambda: _background_statistics(delta), 40)

    methods = [original_function(args.baseline, "survng.app.event_store." + module, name)
               for module, name in [("jobs", "claim_detection_job"), ("scene_acquisition", "claim_scene_candidate"),
                                    ("evidence", "claim_cover_requirement"), ("scene_tracking", "claim_scene_tracking")]]
    with tempfile.TemporaryDirectory(prefix="survng-poll-benchmark-") as directory:
        store = EventStore(Path(directory))
        def before():
            for camera in range(10):
                for method in methods:
                    assert method(store, str(camera), lease_owner="benchmark") is None
        def after():
            for camera in range(10):
                for method in methods:
                    assert getattr(store, method.__name__)(str(camera), lease_owner="benchmark") is None
        with database_polling_session():
            measure("10-camera idle queue sweep", before, after, 20)


if __name__ == "__main__":
    main()
