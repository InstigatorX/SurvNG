#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2

from survng.app.config import load_config
from survng.app.detector import OpenVinoDetector


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vectorized and scalar YOLO parsing on retained incidents."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--database", default="runtime/database/survng.sqlite3")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--camera", default="")
    parser.add_argument("--device", default="CPU")
    return parser.parse_args()


def _balanced_snapshots(
    database: Path,
    storage_dir: Path,
    *,
    limit: int,
    camera_id: str,
) -> list[tuple[int, str, Path]]:
    query = (
        "select id, camera_id, snapshot_path from events "
        "where snapshot_path != '' and kind in ('object', 'motion')"
    )
    parameters: list[Any] = []
    if camera_id:
        query += " and camera_id = ?"
        parameters.append(camera_id)
    query += " order by created_at desc, id desc limit ?"
    parameters.append(max(limit * 8, limit))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(query, parameters).fetchall()
    finally:
        connection.close()

    by_camera: dict[str, deque[tuple[int, str, Path]]] = defaultdict(deque)
    seen: set[str] = set()
    for event_id, row_camera_id, raw_path in rows:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = storage_dir / path
        resolved = path.resolve(strict=False)
        key = str(resolved)
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        by_camera[str(row_camera_id)].append((int(event_id), str(row_camera_id), resolved))

    selected: list[tuple[int, str, Path]] = []
    camera_ids = sorted(by_camera)
    while len(selected) < limit:
        added = False
        for selected_camera_id in camera_ids:
            if by_camera[selected_camera_id] and len(selected) < limit:
                selected.append(by_camera[selected_camera_id].popleft())
                added = True
        if not added:
            break
    return selected


def main() -> int:
    args = _arguments()
    active_config = load_config(args.config)
    detector_config = active_config.detector.model_copy(
        update={"device": str(args.device), "warmup_enabled": False}
    )
    detector = OpenVinoDetector(detector_config)
    if detector.compiled_model is None:
        print("OpenVINO detector did not load", file=sys.stderr)
        return 2
    if detector.output_format != "yolo":
        print(
            f"Replay requires raw YOLO output, found {detector.output_format}",
            file=sys.stderr,
        )
        return 2

    snapshots = _balanced_snapshots(
        Path(args.database),
        Path(active_config.storage_dir),
        limit=max(1, int(args.limit)),
        camera_id=str(args.camera).strip(),
    )
    if not snapshots:
        print("No retained incident snapshots were available", file=sys.stderr)
        return 2

    compared = 0
    unreadable = 0
    mismatches: list[dict[str, Any]] = []
    vectorized_ms = 0.0
    scalar_ms = 0.0
    candidate_totals = defaultdict(int)
    detections = 0
    by_camera = defaultdict(int)
    for event_id, camera_id, path in snapshots:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            unreadable += 1
            continue
        tensor, metadata = detector._preprocess(frame)
        inference = detector.infer_request.infer([tensor])
        output = inference[detector.output_layer]

        metrics: dict[str, float | int] = {}
        started = time.perf_counter()
        vectorized = detector._parse_yolo_output(output, metadata, metrics=metrics)
        vectorized_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        scalar = detector._parse_yolo_output_scalar_reference(output, metadata)
        scalar_ms += (time.perf_counter() - started) * 1000
        compared += 1
        by_camera[camera_id] += 1
        detections += len(vectorized)
        for key in ("raw", "confidence", "valid_boxes", "selected"):
            candidate_totals[key] += int(metrics.get(key) or 0)
        if vectorized != scalar:
            mismatches.append({
                "event_id": event_id,
                "camera_id": camera_id,
                "path": str(path),
                "vectorized": vectorized,
                "scalar": scalar,
            })
            if len(mismatches) >= 10:
                break

    summary = {
        "model": active_config.detector.resolved_model_path(),
        "device": detector.loaded_device,
        "output_format": detector.output_format,
        "input_shape": list(detector.input_shape),
        "compared_images": compared,
        "unreadable_images": unreadable,
        "mismatches": len(mismatches),
        "detections": detections,
        "images_by_camera": dict(sorted(by_camera.items())),
        "average_candidates": {
            key: round(value / max(1, compared), 2)
            for key, value in candidate_totals.items()
        },
        "average_parser_ms": {
            "vectorized": round(vectorized_ms / max(1, compared), 3),
            "scalar": round(scalar_ms / max(1, compared), 3),
            "speedup": round(scalar_ms / max(vectorized_ms, 0.000001), 2),
        },
    }
    print(json.dumps(summary, indent=2))
    if mismatches:
        print(json.dumps({"first_mismatches": mismatches}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
