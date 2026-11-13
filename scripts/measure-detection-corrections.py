#!/usr/bin/env python3
"""Offline phase-0 measurements; never opens the application store for writing.

See docs/detection-correction-memory-phase0.md for inputs and limitations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CATEGORIES = {"same_object", "different_object_same_label", "different_label"}


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("non-finite measurement")
    return {"count": len(values), **{
        name: float(value) for name, value in zip(
            ("min", "p05", "p50", "p95", "p99", "max"),
            np.quantile(values, (0, .05, .5, .95, .99, 1)),
        )}}


def vector(row):
    value = np.frombuffer(row["embedding_blob"], dtype=np.float16).astype(np.float32)
    if len(value) != row["embedding_size"] or not np.isfinite(value).all():
        raise ValueError(f"invalid embedding {row['id']}")
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise ValueError(f"zero embedding {row['id']}")
    return value / norm


def identity(row):
    return (row["implementation"], row["model_fingerprint"],
            row["preprocessing_fingerprint"], row["embedding_size"])


def cosine(left, right):
    if identity(left) != identity(right):
        raise ValueError("cannot compare encoder generations")
    return float(np.clip(vector(left) @ vector(right), -1, 1))


def reviewed_measurements(connection, cases):
    """Explicit row IDs prevent silently guessing physical identity from similarity."""
    cache = {}

    def get(key):
        if key not in cache:
            row = connection.execute(
                "select * from semantic_embeddings where id=?", (int(key),)
            ).fetchone()
            if row is None or row["source_kind"] != "object_crop":
                raise ValueError(f"missing object crop embedding {key}")
            cache[key] = dict(row)
        return cache[key]

    groups = {}
    for case in cases.get("pairs", []):
        category = case["category"]
        if category not in CATEGORIES or not str(case.get("reviewer", "")).strip():
            raise ValueError("pair needs a known category and reviewer")
        a, b = get(case["a"]), get(case["b"])
        if a["event_id"] == b["event_id"] or a["camera_id"] != b["camera_id"]:
            raise ValueError("reviewed pairs must cross events on the same camera")
        score = cosine(a, b)
        key = json.dumps([*identity(a), a["camera_id"]])
        groups.setdefault(key, {}).setdefault(category, []).append(score)
    margins = {}
    for case in cases.get("triplets", []):
        if not str(case.get("reviewer", "")).strip():
            raise ValueError("triplet needs a reviewer")
        query, positive, opposing = (get(case[k]) for k in ("query", "positive", "opposing"))
        if len({r["camera_id"] for r in (query, positive, opposing)}) != 1:
            raise ValueError("triplets must use one camera")
        if query["event_id"] in {positive["event_id"], opposing["event_id"]}:
            raise ValueError("triplet query must come from a held-out event")
        key = json.dumps([*identity(query), query["camera_id"]])
        margins.setdefault(key, []).append(cosine(query, positive) - cosine(query, opposing))
    return {
        "pairs": {key: {cat: distribution(values) for cat, values in group.items()}
                  for key, group in groups.items()},
        "winning_minus_opposing": {key: distribution(values) for key, values in margins.items()},
        "threshold_recommendation": None,
        "reason": "Require held-out reviewed identities/verdicts and an agreed false-suppression budget; pair quantiles alone do not establish safe thresholds.",
    }


def index_measurements(database, cases, limit=1000, pair_attempts=5000, seed=0):
    # Do not construct SemanticIndex: its constructor initializes/migrates schema.
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("pragma query_only=on")
        conn.execute("begin")
        generations = [dict(row) for row in conn.execute(
            "select implementation, model_fingerprint, preprocessing_fingerprint, "
            "embedding_size, count(*) as count from semantic_embeddings "
            "where source_kind='object_crop' group by 1,2,3,4"
        )]
        reports = []
        rng = np.random.default_rng(seed)
        for generation in generations:
            rows = [dict(row) for row in conn.execute(
                "select * from semantic_embeddings where source_kind='object_crop' "
                "and implementation=? and model_fingerprint=? and preprocessing_fingerprint=? "
                "and embedding_size=? order by id desc limit ?",
                (*[generation[k] for k in ("implementation", "model_fingerprint", "preprocessing_fingerprint", "embedding_size")], limit),
            )]
            vectors = [vector(row) for row in rows]
            proxies = {}
            seen = set()
            for _ in range(pair_attempts if len(rows) > 1 else 0):
                i, j = sorted(int(x) for x in rng.choice(len(rows), 2, replace=False))
                a, b = rows[i], rows[j]
                if (i, j) in seen or a["event_id"] == b["event_id"] or a["camera_id"] != b["camera_id"]:
                    continue
                seen.add((i, j))
                kind = "same_detector_label_identity_unknown" if a["object_label"] == b["object_label"] else "different_detector_label_unreviewed"
                key = json.dumps([a["camera_id"], kind])
                proxies.setdefault(key, []).append(float(np.clip(vectors[i] @ vectors[j], -1, 1)))
            reports.append({"identity": generation, "sampled_recent_rows": len(rows),
                            "row_id_range": [min(r['id'] for r in rows), max(r['id'] for r in rows)] if rows else [],
                            "unreviewed_proxies": {key: distribution(values) for key, values in proxies.items()}})
        return {"generations": reports, "reviewed": reviewed_measurements(conn, cases),
                "sampling": {"recent_rows_per_generation": limit, "pair_attempts": pair_attempts, "seed": seed}}


def crop(frame, box, plane):
    if len(box) != 4 or len(plane) != 2:
        raise ValueError("crop needs xyxy and width,height")
    box, plane = np.asarray(box, dtype=float), np.asarray(plane, dtype=float)
    if not np.isfinite(box).all() or not np.isfinite(plane).all() or (plane <= 0).any():
        raise ValueError("invalid crop geometry")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box * np.array([width, height, width, height]) / np.tile(plane, 2)).astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty crop")
    return frame[y1:y2, x1:x2].copy()


def benchmark_encoder(encoder, images, repeats, batch_size, wait_ms):
    """Controlled local contention, NOT a measurement of the running service lock."""
    lock = threading.RLock()
    batch = [images[i % len(images)] for i in range(batch_size)]
    cold = time.perf_counter()
    encoder.encode_images(batch)
    cold_ms = (time.perf_counter() - cold) * 1000
    latencies, waits = [], []
    timeouts = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(repeats):
            started = time.perf_counter()
            with lock:
                encoder.encode_images(batch)
            latencies.append((time.perf_counter() - started) * 1000)
            acquired = threading.Event()

            def backfill_batch():
                with lock:
                    acquired.set()
                    encoder.encode_images(batch)

            future = pool.submit(backfill_batch)
            if not acquired.wait(5):
                raise RuntimeError("benchmark worker did not acquire the lock")
            started = time.perf_counter()
            granted = lock.acquire(timeout=wait_ms / 1000)
            waits.append((time.perf_counter() - started) * 1000)
            if granted:
                lock.release()
            else:
                timeouts += 1
            future.result()  # Surface worker failures; never report them as a latency.
    return {"kind": "offline_controlled_single_backfill_batch", "batch_size": batch_size,
            "cold_encode_ms": cold_ms, "warm_encode_ms": distribution(latencies),
            "lock_wait_ms_censored_at_timeout": distribution(waits), "timeouts": timeouts,
            "lock_timeout_ms": wait_ms, "live_service_lock_wait": "not_measured"}


def image_measurements(model_dir, cases, device, repeats, batch_size, wait_ms):
    import cv2
    from survng.app.semantic_search import IsolatedOpenVinoManifestEncoder, load_semantic_manifest

    images, pairs = [], []
    for case in cases.get("crops", []):
        frame = cv2.imread(str(case["image"]))
        if frame is None:
            raise ValueError("measurement image unavailable")
        detected = crop(frame, case["detector_box"], case["coordinate_plane"])
        images.append(detected)
        if "motion_box" in case:
            if case.get("alignment_verified") is not True or not case.get("event_id"):
                raise ValueError("motion comparison requires an event and verified alignment")
            # motion_box must ALREADY be transformed to this image's normalized plane.
            motion = crop(frame, case["motion_box"], [1, 1])
            db = np.asarray(case["detector_box"]) / np.tile(case["coordinate_plane"], 2)
            mb = np.asarray(case["motion_box"])
            if min(db[2], mb[2]) <= max(db[0], mb[0]) or min(db[3], mb[3]) <= max(db[1], mb[1]):
                raise ValueError("motion region does not overlap detection")
            pairs.append((detected, motion))
    if not images:
        raise ValueError("--model-dir needs crops in --cases")
    manifest = load_semantic_manifest(model_dir)
    encoder_identity = None
    if manifest.get("implementation") == "siglip2":
        from survng.app.siglip2_search import (
            SIGLIP2_FILES, _file_matches, siglip2_manifest, siglip2_semantic_contract,
        )
        if manifest != siglip2_manifest() or not all(
            _file_matches(model_dir / name, digest, size, None)
            for name, digest, size in SIGLIP2_FILES
        ):
            raise ValueError("SigLIP 2 package does not match the pinned contract")
        encoder_identity, _ = siglip2_semantic_contract()
    encoder = IsolatedOpenVinoManifestEncoder(model_dir, manifest, device, encoder_identity)
    try:
        latency = benchmark_encoder(encoder, images, repeats, batch_size, wait_ms)
        similarities = []
        for detected, motion in pairs:
            values = np.asarray(encoder.encode_images([detected, motion]), dtype=float)
            norms = np.linalg.norm(values, axis=1)
            if not np.isfinite(values).all() or (norms <= 0).any():
                raise ValueError("invalid encoder output")
            values /= norms[:, None]
            similarities.append(float(np.clip(values[0] @ values[1], -1, 1)))
        return {"identity": asdict(encoder.identity), "device": device, "latency": latency,
                "motion_crop_cosine": distribution(similarities),
                "motion_crop_loss_from_identical_crop": distribution([1 - v for v in similarities]),
                "motion_matching_recommendation": "pending reviewed cross-event motion-to-exemplar retrieval; same-frame agreement is insufficient"}
    finally:
        encoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pair-attempts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lock-timeout-ms", type=float, default=50)
    args = parser.parse_args()
    if not (1 <= args.limit <= 10000 and 1 <= args.pair_attempts <= 100000
            and 1 <= args.repeats <= 100 and 1 <= args.batch_size <= 64
            and 0 < args.lock_timeout_ms <= 10000):
        parser.error("measurement bounds exceeded")
    if args.output and args.output.resolve() in {args.database.resolve(), args.cases.resolve() if args.cases else None}:
        parser.error("output must not overwrite an input")
    cases = json.loads(args.cases.read_text()) if args.cases else {}
    result = {"schema_version": 1, "index": index_measurements(args.database, cases, args.limit, args.pair_attempts, args.seed)}
    if args.model_dir:
        result["encoder"] = image_measurements(args.model_dir, cases, args.device, args.repeats, args.batch_size, args.lock_timeout_ms)
    else:
        result["encoder"] = {"status": "not_measured", "reason": "Supply --model-dir and reviewed crop cases"}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        # Exclusive creation also protects existing databases, images and reports.
        with args.output.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
