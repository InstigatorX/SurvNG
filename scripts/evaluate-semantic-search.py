#!/usr/bin/env python3
"""Measure Smart Search ranking against a reviewed incident benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SearchFunction = Callable[[str, int], tuple[list[dict[str, Any]], float]]


def _http_search(
    base_url: str, token: str, query: str, limit: int
) -> tuple[list[dict[str, Any]], float]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/semantic-search",
        data=json.dumps({"query": query, "limit": limit}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Smart Search returned HTTP {exc.code}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return list(payload.get("results") or []), elapsed_ms


def _event_id(result: dict[str, Any]) -> int:
    return int(dict(result.get("event") or {}).get("id") or 0)


def _metrics(ids: list[int], relevant: set[int], cutoff: int) -> dict[str, float | int]:
    selected = ids[:cutoff]
    matches = sum(event_id in relevant for event_id in selected)
    return {
        "matches": matches,
        "precision": round(matches / max(1, len(selected)), 4),
        "recall": round(matches / max(1, len(relevant)), 4),
    }


def evaluate(
    benchmark: dict[str, Any],
    *,
    search: SearchFunction,
    source: dict[str, Any],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for case in benchmark.get("queries") or []:
        query = str(case["query"])
        limit = max(10, int(case.get("limit") or 10))
        results, latency_ms = search(query, limit)
        ids = [_event_id(result) for result in results]
        relevant = {int(value) for value in case.get("relevant_event_ids") or []}
        judged = bool(case.get("judged", relevant))
        report: dict[str, Any] = {
            "id": str(case.get("id") or query),
            "query": query,
            "latency_ms": round(latency_ms, 2),
            "result_event_ids": ids,
            "results": [
                {
                    "event_id": _event_id(result),
                    "score": result.get("score"),
                    "snapshot_url": result.get("snapshot_url"),
                }
                for result in results
            ],
            "judged": judged,
        }
        if judged:
            report["relevant_event_ids"] = sorted(relevant)
            report["precision_at_5"] = _metrics(ids, relevant, 5)
            report["precision_at_10"] = _metrics(ids, relevant, 10)
            ranks = [index + 1 for index, event_id in enumerate(ids) if event_id in relevant]
            report["reciprocal_rank"] = round(1 / ranks[0], 4) if ranks else 0.0
        reports.append(report)
    latencies = [float(report["latency_ms"]) for report in reports]
    return {
        "schema_version": 1,
        "benchmark": benchmark.get("name") or "semantic-search",
        "base_url": str(source.get("base_url") or ""),
        "source": source,
        "queries": reports,
        "summary": {
            "query_count": len(reports),
            "judged_query_count": sum(bool(report["judged"]) for report in reports),
            "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
            "maximum_latency_ms": round(max(latencies), 2) if latencies else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/survng")
    parser.add_argument("--token", default="")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    encoder: Any = None
    if args.model_dir:
        if not args.database:
            parser.error("--database is required with --model-dir")
        from survng.app.semantic_search import (
            IsolatedOpenVinoManifestEncoder,
            SemanticIndex,
            load_semantic_manifest,
        )

        manifest = load_semantic_manifest(args.model_dir)
        encoder = IsolatedOpenVinoManifestEncoder(
            args.model_dir, manifest, args.device
        )
        index = SemanticIndex(args.database)

        def search(query: str, limit: int) -> tuple[list[dict[str, Any]], float]:
            started = time.perf_counter()
            embedding = encoder.encode_text([query])
            hits = index.search(embedding, encoder.identity, limit=limit * 4)
            best_by_event: dict[int, Any] = {}
            for hit in hits:
                best_by_event.setdefault(hit.event_id, hit)
                if len(best_by_event) >= limit:
                    break
            elapsed_ms = (time.perf_counter() - started) * 1000
            return [
                {
                    "score": round(hit.score, 6),
                    "event": {
                        "id": event_id,
                        "camera_id": hit.camera_id,
                        "created_at": hit.captured_at,
                    },
                    "snapshot_url": hit.image_path,
                }
                for event_id, hit in best_by_event.items()
            ], elapsed_ms

        source = {
            "kind": "local_generation",
            "model_dir": str(args.model_dir.resolve()),
            "generation": encoder.identity.generation,
            "device": args.device,
        }
    else:
        search = lambda query, limit: _http_search(
            args.base_url, args.token, query, limit
        )
        source = {"kind": "api", "base_url": args.base_url}
    try:
        report = evaluate(benchmark, search=search, source=source)
    finally:
        if encoder is not None:
            encoder.close()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
