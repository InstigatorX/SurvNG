#!/usr/bin/env python3
"""Benchmark YOLO26-depth OpenVINO export latency on sample frames."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def _load_frames(paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    return frames


def _synthetic_frames(count: int, width: int, height: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(count):
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        base = np.tile(gradient, (height, 1))
        frame = cv2.merge([base, np.roll(base, index * 3, axis=1), 255 - base])
        frames.append(frame)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="OpenVINO depth model .xml path")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--frames", nargs="*", default=[], help="Image files to benchmark")
    parser.add_argument("--count", type=int, default=10, help="Synthetic frame count when no images")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=768)
    args = parser.parse_args()

    from survng.app.config import DepthConfig, DetectorConfig
    from survng.app.depth_estimation import OpenVinoDepthEstimator

    model_path = args.model
    if not model_path:
        default = Path("models/yolo26n-depth_openvino_model/yolo26n-depth.xml")
        if default.is_file():
            model_path = str(default)
    if not model_path:
        print("No depth model path provided and default model was not found.")
        return 2

    config = DetectorConfig(
        depth=DepthConfig(
            enabled=True,
            model_path=model_path,
            device=args.device,
            input_size=args.input_size,
        )
    )
    estimator = OpenVinoDepthEstimator(config)
    if not estimator.ready:
        print(f"Depth estimator failed to load: {estimator.error}")
        return 1

    image_paths = [Path(path) for path in args.frames]
    frames = _load_frames(image_paths)
    if not frames:
        frames = _synthetic_frames(args.count, 1280, 720)

    for _ in range(max(0, args.warmup)):
        estimator.estimate_depth_map(frames[0])

    timings: list[float] = []
    for frame in frames:
        started = time.perf_counter()
        depth_map = estimator.estimate_depth_map(frame)
        timings.append((time.perf_counter() - started) * 1000.0)
        finite = depth_map[np.isfinite(depth_map)]
        print(
            f"frame {frame.shape[1]}x{frame.shape[0]} "
            f"depth {depth_map.shape[1]}x{depth_map.shape[0]} "
            f"median={float(np.median(finite)):.2f}m "
            f"latency={timings[-1]:.1f}ms"
        )

    print(
        "summary:",
        f"frames={len(timings)}",
        f"p50={statistics.median(timings):.1f}ms",
        f"mean={statistics.mean(timings):.1f}ms",
        f"device={estimator.loaded_device}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
