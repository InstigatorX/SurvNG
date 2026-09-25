#!/usr/bin/env python3
"""Read-only, owner-local runtime and Linux process-tree sampling (no credentials)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survng.ctl import default_socket_path, request_runtime_status


def process_sample(pid: int, proc: Path = Path("/proc")) -> dict:
    """Read counters only; never read process arguments, environment or FD targets."""
    root = proc / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    try:
        descriptors = sum(1 for _ in (root / "fd").iterdir())
    except OSError:
        descriptors = None
    return {
        "pid": pid, "ppid": int(fields[1]), "state": fields[0],
        "start_ticks": int(fields[19]),
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        "threads": int(fields[17]), "fds": descriptors,
    }


def process_tree(pid: int, proc: Path = Path("/proc")) -> list[dict]:
    samples = {}
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            row = process_sample(int(directory.name), proc)
        except (OSError, ValueError, IndexError):
            continue  # Processes may exit while sampled.
        samples[row["pid"]] = row
    selected = {pid} if pid in samples else set()
    while True:
        children = {key for key, row in samples.items() if row["ppid"] in selected}
        if children <= selected:
            break
        selected |= children
    return [samples[key] for key in sorted(selected)]


def summarize(path: Path) -> dict:
    """Keep scalar samples, not large runtime payloads, while reading the log."""
    generations = {}
    errors = 0
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            errors += int("snapshot_error" in record)
            runtime = record.get("runtime", {})
            owner = runtime.get("process", {})
            identity = owner.get("instance_id")
            if identity is None:
                continue
            group = generations.setdefault(identity, {"samples": 0, "metrics": {}})
            group["samples"] += 1
            if float(owner.get("uptime_seconds", 0)) < 3600:
                continue  # Model/camera warmup is not a leak trend.
            main = next((p for p in record["processes"] if p["pid"] == owner["pid"]), {})
            detector = runtime.get("detector", {}).get("runtime", {})
            cameras = [c for c in runtime.get("cameras", []) if c.get("enabled")]
            values = {
                "main_rss_bytes": main.get("rss_bytes"),
                "main_threads": main.get("threads"), "main_fds": main.get("fds"),
                "tree_rss_bytes": record["tree_totals"]["rss_bytes"],
                "tree_processes": record["tree_totals"]["process_count"],
                "asyncio_tasks": runtime.get("event_loop", {}).get("tasks"),
                "loop_probe_lag_ms": runtime.get("event_loop", {}).get("probe_lag_ms"),
                "snapshot_ms": record["snapshot_ms"],
                "inference_ms": detector.get("last_inference_ms"),
                "inference_queue": detector.get("queue_depth"),
                "max_frame_age_seconds": max((c.get("last_frame_age_seconds") or 0 for c in cameras), default=0),
                "max_analysis_queue": max(((c.get("motion") or {}).get("analysis_queue_depth") or 0 for c in cameras), default=0),
                "unhealthy_cameras": sum(not c.get("connected") for c in cameras),
            }
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    group["metrics"].setdefault(name, []).append(value)
    for group in generations.values():
        group["metrics"] = {
            name: {"first_10_median": statistics.median(values[:10]),
                   "last_10_median": statistics.median(values[-10:]), "max": max(values)}
            for name, values in group["metrics"].items()
        }
    return {"snapshot_errors": errors, "generations": generations,
            "note": "Metrics exclude the first hour of each process lifetime; compare equivalent workloads."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize is not None:
        print(json.dumps(summarize(args.summarize), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required when recording samples")
    if not 0 < args.hours <= 168 or not 0.1 <= args.interval <= 3600:
        parser.error("hours must be in (0, 168]; interval must be in [0.1, 3600]")
    deadline = time.monotonic() + args.hours * 3600
    previous = {}
    previous_at = None
    last_owner = None
    # Refuse overwrites and make the output owner-only from creation.
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        while time.monotonic() < deadline:
            started = time.monotonic()
            row = {"sampled_at": datetime.now(timezone.utc).isoformat()}
            try:
                runtime = request_runtime_status(args.socket, timeout=5.0)
                runtime.pop("recent_logs", None)
                row["runtime"] = runtime
                pid = int(runtime["process"]["pid"])
                owner = process_sample(pid)
                last_owner = (pid, owner["start_ticks"])
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                # Type only: diagnostics must never leak arbitrary provider text.
                row["snapshot_error"] = type(error).__name__
            row["snapshot_ms"] = round((time.monotonic() - started) * 1000, 3)
            tree = []
            if last_owner is not None:
                try:
                    owner = process_sample(last_owner[0])
                    if owner["start_ticks"] == last_owner[1]:
                        tree = process_tree(last_owner[0])
                except (OSError, ValueError, IndexError):
                    pass
            current = {}
            for process in tree:
                key = (process["pid"], process["start_ticks"])
                current[key] = process["cpu_seconds"]
                if key in previous and previous_at is not None:
                    process["cpu_percent"] = round(
                        100 * max(0, process["cpu_seconds"] - previous[key])
                        / max(.001, started - previous_at), 2,
                    )
            previous, previous_at = current, started
            row["processes"] = tree
            row["tree_totals"] = {
                field: sum(process.get(field) or 0 for process in tree)
                for field in ("rss_bytes", "threads", "fds", "cpu_percent")
            }
            row["tree_totals"]["process_count"] = len(tree)
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
            time.sleep(max(0, min(deadline - time.monotonic(), args.interval - (time.monotonic() - started))))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
