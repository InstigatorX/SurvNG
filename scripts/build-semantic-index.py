#!/usr/bin/env python3
"""Build or resume one semantic model generation without activating it."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import resource
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from survng.app.config import SemanticSearchConfig
from survng.app.events import EventStore
from survng.app.semantic_search import (
    IsolatedOpenVinoManifestEncoder,
    SemanticIndex,
    SemanticSearchService,
    load_semantic_manifest,
    semantic_event_objects,
)

SQLITE_LOCK_RETRY_ATTEMPTS = 8
SQLITE_LOCK_RETRY_INITIAL_SECONDS = 0.25


def _process_memory_mb(pid: int) -> tuple[float, float]:
    """Return current and high-water RSS without adding a process dependency."""
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0.0, 0.0
    values: dict[str, float] = {}
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            try:
                values[key] = float(value.strip().split()[0]) / 1024
            except (IndexError, ValueError):
                continue
    return values.get("VmRSS", 0.0), values.get("VmHWM", 0.0)


def _event_pages(database: Path, page_size: int) -> Iterator[list[dict[str, Any]]]:
    uri = f"file:{database.resolve()}?mode=ro"
    before_created_at: str | None = None
    before_id: int | None = None
    while True:
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("pragma busy_timeout = 30000")
            if before_created_at is None or before_id is None:
                rows = connection.execute(
                    f"""
                    select {EventStore.COMPACT_COLUMNS} from events
                    order by created_at desc, id desc limit ?
                    """,
                    (page_size,),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    select {EventStore.COMPACT_COLUMNS} from events
                    where created_at < ? or (created_at = ? and id < ?)
                    order by created_at desc, id desc limit ?
                    """,
                    (before_created_at, before_created_at, before_id, page_size),
                ).fetchall()
        if not rows:
            return
        page = [dict(row) for row in rows]
        yield page
        last = page[-1]
        before_created_at = str(last.get("created_at") or "")
        before_id = int(last.get("id") or 0)
        if len(page) < page_size:
            return


def _index_event_with_retry(
    service: SemanticSearchService,
    event: dict[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, int]:
    """Retry transient SQLite writer contention without hiding other failures."""
    retries = 0
    delay = SQLITE_LOCK_RETRY_INITIAL_SECONDS
    while True:
        try:
            return service.index_event(event), retries
        except sqlite3.OperationalError as exc:
            if (
                "locked" not in str(exc).lower()
                or retries >= SQLITE_LOCK_RETRY_ATTEMPTS
            ):
                raise
            retries += 1
            sleep(delay)
            delay = min(5.0, delay * 2.0)


def build(
    *,
    database: Path,
    storage_dir: Path,
    model_dir: Path,
    device: str,
    page_size: int,
    pause_seconds: float,
    limit: int,
    progress_every: int,
) -> dict[str, Any]:
    manifest = load_semantic_manifest(model_dir)
    index = SemanticIndex(database)
    config = SemanticSearchConfig(
        enabled=True,
        implementation="openvino_manifest",
        model_dir=str(model_dir),
        device=device,
        index_full_frame=True,
        index_object_crops=True,
        max_object_crops_per_event=24,
    )
    service = SemanticSearchService(config, index, model_dir, manifest)
    encoder = IsolatedOpenVinoManifestEncoder(model_dir, manifest, device)
    service.encoder = encoder
    service._storage_dir = storage_dir
    before = index.coverage(encoder.identity)
    started = time.monotonic()
    eligible = 0
    attempted = 0
    encoded = 0
    written_evidence = 0
    maximum_combined_rss_mb = 0.0
    maximum_worker_rss_mb = 0.0
    sqlite_lock_retries = 0
    try:
        for page in _event_pages(database, page_size):
            for event in page:
                if not semantic_event_objects(event):
                    continue
                if limit and attempted >= limit:
                    break
                eligible += 1
                written, event_lock_retries = _index_event_with_retry(service, event)
                sqlite_lock_retries += event_lock_retries
                attempted += 1
                if written:
                    encoded += 1
                    written_evidence += written
                parent_rss, parent_hwm = _process_memory_mb(os.getpid())
                worker_pid = encoder.worker_pid
                worker_rss, worker_hwm = (
                    _process_memory_mb(worker_pid) if worker_pid else (0.0, 0.0)
                )
                maximum_worker_rss_mb = max(maximum_worker_rss_mb, worker_hwm, worker_rss)
                maximum_combined_rss_mb = max(
                    maximum_combined_rss_mb,
                    parent_hwm,
                    parent_rss + worker_rss,
                )
                if progress_every and attempted % progress_every == 0:
                    coverage = index.coverage(encoder.identity)
                    elapsed = max(0.001, time.monotonic() - started)
                    print(json.dumps({
                        "state": "running",
                        "attempted_events": attempted,
                        "encoded_events": encoded,
                        "written_evidence": written_evidence,
                        "indexed_events": coverage["event_count"],
                        "evidence_count": coverage["evidence_count"],
                        "events_per_second": round(attempted / elapsed, 3),
                        "encoded_events_per_second": round(encoded / elapsed, 3),
                        "missing_snapshots": service._skipped_missing,
                        "sqlite_lock_retries": sqlite_lock_retries,
                    }), flush=True)
                if pause_seconds:
                    time.sleep(pause_seconds)
            if limit and attempted >= limit:
                break
    finally:
        encoder.close()
        service.encoder = None
    elapsed = max(0.001, time.monotonic() - started)
    after = index.coverage(encoder.identity)
    return {
        "state": "complete",
        "generation": encoder.identity.generation,
        "implementation": encoder.identity.implementation,
        "device": device,
        "attempted_events": attempted,
        "encoded_events": encoded,
        "written_evidence": written_evidence,
        "eligible_events_seen": eligible,
        "new_events": after["event_count"] - before["event_count"],
        "new_evidence": after["evidence_count"] - before["evidence_count"],
        "event_count": after["event_count"],
        "evidence_count": after["evidence_count"],
        "missing_snapshots": service._skipped_missing,
        "sqlite_lock_retries": sqlite_lock_retries,
        "elapsed_seconds": round(elapsed, 3),
        "events_per_second": round(attempted / elapsed, 3),
        "encoded_events_per_second": round(encoded / elapsed, 3),
        "parent_maximum_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "worker_maximum_rss_mb": round(maximum_worker_rss_mb, 1),
        "combined_maximum_rss_mb": round(maximum_combined_rss_mb, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--storage-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--pause-seconds", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 10000:
        parser.error("--page-size must be between 1 and 10000")
    if args.pause_seconds < 0 or args.pause_seconds > 10:
        parser.error("--pause-seconds must be between 0 and 10")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    lock_path = args.database.with_suffix(args.database.suffix + ".semantic-build.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another semantic comparison build is already running") from exc
        report = build(
            database=args.database,
            storage_dir=args.storage_dir,
            model_dir=args.model_dir,
            device=args.device,
            page_size=args.page_size,
            pause_seconds=args.pause_seconds,
            limit=args.limit,
            progress_every=max(0, args.progress_every),
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
