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


# Conservative review thresholds for post-warmup samples; override with
# --threshold NAME=VALUE. Gauge thresholds compare the largest sample; the
# *_increase thresholds compare the growth of a since-start counter.
DEFAULT_THRESHOLDS = {
    "loop_lag_recent_p99_ms": 100.0,
    "loop_lag_recent_max_ms": 1000.0,
    "db_transaction_recent_p99_ms": 250.0,
    "db_transaction_recent_max_ms": 5000.0,
    "main_write_lock_wait_recent_max_ms": 1000.0,
    "main_write_lock_hold_recent_max_ms": 2000.0,
    "db_busy_errors_increase": 0.0,
    "db_transactions_failed_increase": 0.0,
}


def _path(source: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(source, dict):
            return None
        source = source.get(key)
    return source


def runtime_metric_values(runtime: dict) -> dict:
    """Scalar continuous loop-lag and per-database metrics; absent fields are None."""
    lag = _path(runtime, "event_loop", "lag")
    values = {
        "loop_lag_recent_p99_ms": _path(lag, "recent", "p99_ms"),
        "loop_lag_recent_max_ms": _path(lag, "recent", "max_ms"),
        "loop_lag_max_ms": _path(lag, "max_ms"),
    }
    lock = _path(runtime, "databases", "main_write_lock")
    for kind in ("wait", "hold"):
        for field in ("p99_ms", "max_ms"):
            values[f"main_write_lock_{kind}_recent_{field}"] = _path(
                lock, f"{kind}_ms", "recent", field
            )
    domains = _path(runtime, "databases", "domains")
    for name, domain in (domains.items() if isinstance(domains, dict) else ()):
        values[f"db_{name}_transaction_recent_p99_ms"] = _path(
            domain, "transaction_ms", "recent", "p99_ms"
        )
        values[f"db_{name}_transaction_recent_max_ms"] = _path(
            domain, "transaction_ms", "recent", "max_ms"
        )
        values[f"db_{name}_busy_errors"] = _path(domain, "busy_errors")
        values[f"db_{name}_transactions_failed"] = _path(domain, "transactions_failed")
    return values


def _threshold_for(metric: str, thresholds: dict) -> tuple[str, float] | None:
    if metric in thresholds:
        return metric, thresholds[metric]
    if metric.startswith("db_"):
        for suffix in ("transaction_recent_p99_ms", "transaction_recent_max_ms"):
            if metric.endswith("_" + suffix):
                return f"db_{suffix}", thresholds[f"db_{suffix}"]
        for suffix in ("busy_errors", "transactions_failed"):
            if metric.endswith("_" + suffix):
                return f"db_{suffix}_increase", thresholds[f"db_{suffix}_increase"]
    return None


def threshold_breaches(metrics: dict, thresholds: dict) -> list[dict]:
    breaches = []
    for metric, values in sorted(metrics.items()):
        rule = _threshold_for(metric, thresholds)
        if rule is None or not values:
            continue
        name, limit = rule
        observed = values[-1] - values[0] if name.endswith("_increase") else max(values)
        if observed > limit:
            breaches.append({"metric": metric, "threshold": name, "limit": limit,
                             "observed": observed,
                             "samples_over": sum(v > limit for v in values)
                             if not name.endswith("_increase") else None})
    return breaches


def summarize(path: Path, thresholds: dict | None = None) -> dict:
    """Keep scalar samples, not large runtime payloads, while reading the log."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
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
                **runtime_metric_values(runtime),
            }
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    group["metrics"].setdefault(name, []).append(value)
    for group in generations.values():
        series = group["metrics"]
        group["threshold_breaches"] = threshold_breaches(series, thresholds)
        seen = set(series)
        group["unavailable_metrics"] = sorted(
            name for name in ("loop_lag_recent_p99_ms", "loop_lag_recent_max_ms",
                              "main_write_lock_wait_recent_max_ms",
                              "main_write_lock_hold_recent_max_ms")
            if name not in seen
        ) + ([] if any(name.startswith("db_") for name in seen) else ["db_*"])
        if group["samples"] and not series:
            group["unavailable_metrics"] = ["all (no post-warmup samples)"]
        group["metrics"] = {
            name: {"first_10_median": statistics.median(values[:10]),
                   "last_10_median": statistics.median(values[-10:]), "max": max(values)}
            for name, values in series.items()
        }
    return {"snapshot_errors": errors, "generations": generations,
            "thresholds": thresholds,
            "threshold_breach_count": sum(
                len(group["threshold_breaches"]) for group in generations.values()
            ),
            "note": "Metrics exclude the first hour of each process lifetime; compare equivalent workloads. "
                    "Unavailable metrics were not reported by the server (older build) and were not evaluated."}


def _threshold_argument(text: str) -> tuple[str, float]:
    name, separator, value = text.partition("=")
    if not separator or name not in DEFAULT_THRESHOLDS:
        raise argparse.ArgumentTypeError(
            f"expected NAME=VALUE with NAME in: {', '.join(sorted(DEFAULT_THRESHOLDS))}"
        )
    try:
        return name, float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid threshold value: {value!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--threshold", type=_threshold_argument, action="append", default=[],
                        metavar="NAME=VALUE", help="override a summary review threshold")
    args = parser.parse_args()
    if args.summarize is not None:
        print(json.dumps(summarize(args.summarize, dict(args.threshold)), indent=2))
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
