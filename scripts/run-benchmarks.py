#!/usr/bin/env python3
"""Run the unattended CPU benchmarks with fixed inputs and record one JSON result.

Record only: there are no pass/fail thresholds. Each benchmark script is run
as a subprocess with ``--json`` and fixed parameters, ``--repetitions`` times;
the runner aggregates median/min/max per metric across those runs. Inputs are
synthetic and generated at run time (see docs/performance/nightly-benchmarks.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPOSITORY = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Benchmark:
    name: str
    command: tuple[str, ...]
    # Maps one run's parsed --json output to {metric: (value, unit)}.
    extract: Callable[[dict], dict[str, tuple[float, str]]]
    requires: tuple[str, ...] = ()
    timeout_seconds: float = 300.0


def _capture_metrics(result: dict) -> dict[str, tuple[float, str]]:
    metrics: dict[str, tuple[float, str]] = {}
    for transport, row in sorted(result["transports"].items()):
        metrics[f"{transport}_ffmpeg_cpu_per_frame"] = (row["ffmpeg_ms_per_frame_median"], "ms")
        metrics[f"{transport}_python_cpu_per_frame"] = (row["python_ms_per_frame_median"], "ms")
        runs = row.get("runs") or []
        if runs:
            metrics[f"{transport}_read_p95"] = (
                statistics.median(float(run["read_ms_p95"]) for run in runs), "ms",
            )
            metrics[f"{transport}_elapsed"] = (
                statistics.median(float(run["elapsed_s"]) for run in runs), "s",
            )
        metrics[f"{transport}_frames"] = (row["frames"], "frames")
        metrics[f"{transport}_ffmpeg_rss"] = (row["ffmpeg_rss_bytes"] / 1048576, "MiB")
    return metrics


def _color_reuse_metrics(result: dict) -> dict[str, tuple[float, str]]:
    return {name: (value, "ms/frame") for name, value in sorted(result["ms_per_frame"].items())}


def _capture(width: int, height: int) -> Benchmark:
    # 20 s keeps ~200 delivered frames per run, so 10 ms /proc CPU ticks
    # quantize per-frame CPU far less than the script's 4 s default.
    return Benchmark(
        name=f"capture_transport_{width}x{height}",
        command=(
            sys.executable, "scripts/benchmark_capture_transport.py", "--json",
            "--width", str(width), "--height", str(height), "--fps", "20",
            "--duration", "20", "--frame-rate", "10", "--repeats", "3",
        ),
        extract=_capture_metrics,
        requires=("ffmpeg",),
    )


DEFAULT_BENCHMARKS: tuple[Benchmark, ...] = (
    _capture(640, 360),
    _capture(1280, 720),
    Benchmark(
        name="motion_color_reuse",
        command=(sys.executable, "scripts/benchmark_motion_color_reuse.py", "--json", "--seed", "725"),
        extract=_color_reuse_metrics,
    ),
)


def aggregate(runs: list[dict[str, tuple[float, str]]]) -> dict[str, dict[str, Any]]:
    """Per metric: unit, median, min, max and the number of runs that reported it."""
    metrics: dict[str, dict[str, Any]] = {}
    for run in runs:
        for name, (value, unit) in run.items():
            entry = metrics.setdefault(name, {"unit": unit, "values": []})
            entry["values"].append(float(value))
    return {
        name: {
            "unit": entry["unit"],
            "median": round(statistics.median(entry["values"]), 4),
            "min": round(min(entry["values"]), 4),
            "max": round(max(entry["values"]), 4),
            "repetitions": len(entry["values"]),
            "values": [round(value, 4) for value in entry["values"]],
        }
        for name, entry in sorted(metrics.items())
    }


def run_benchmark(benchmark: Benchmark, repetitions: int, cwd: Path = REPOSITORY) -> dict[str, Any]:
    missing = [tool for tool in benchmark.requires if shutil.which(tool) is None]
    if missing:
        return {"status": "skipped", "reason": f"missing: {', '.join(missing)}", "metrics": {}}
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        part for part in (str(cwd), os.environ.get("PYTHONPATH", "")) if part
    )}
    runs = []
    started = time.monotonic()
    for _ in range(repetitions):
        try:
            completed = subprocess.run(
                benchmark.command, cwd=cwd, env=environment, capture_output=True,
                text=True, timeout=benchmark.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": "timeout", "metrics": aggregate(runs),
                    "wall_seconds": round(time.monotonic() - started, 2)}
        if completed.returncode != 0:
            return {"status": "failed", "reason": f"exit {completed.returncode}",
                    "stderr_tail": completed.stderr[-2000:], "metrics": aggregate(runs),
                    "wall_seconds": round(time.monotonic() - started, 2)}
        try:
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            runs.append(benchmark.extract(json.loads(lines[-1])))
        except (IndexError, KeyError, TypeError, ValueError) as error:
            return {"status": "failed", "reason": f"unparseable output ({type(error).__name__})",
                    "metrics": aggregate(runs),
                    "wall_seconds": round(time.monotonic() - started, 2)}
    return {"status": "ok", "metrics": aggregate(runs),
            "wall_seconds": round(time.monotonic() - started, 2)}


def _first_line(command: list[str]) -> str | None:
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (output.stdout or output.stderr).strip()
    return text.splitlines()[0] if text else None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def environment_info(cwd: Path = REPOSITORY) -> dict[str, Any]:
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {
        "commit": os.environ.get("GITHUB_SHA") or _first_line(["git", "-C", str(cwd), "rev-parse", "HEAD"]),
        "python": platform.python_version(),
        "opencv": opencv,
        "numpy": numpy_version,
        "ffmpeg": (
            (_first_line(["ffmpeg", "-hide_banner", "-version"]) or "").split(" Copyright")[0] or None
            if shutil.which("ffmpeg") else None
        ),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "runner": os.environ.get("RUNNER_NAME"),
    }


def build_report(results: dict[str, dict], environment: dict, repetitions: int,
                 wall_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repetitions": repetitions,
        "wall_seconds": round(wall_seconds, 2),
        "environment": environment,
        "benchmarks": results,
    }


def summary_markdown(report: dict[str, Any]) -> str:
    environment = report["environment"]
    lines = [
        "## Nightly benchmarks (record only)",
        "",
        f"Commit `{environment.get('commit')}` · {environment.get('cpu_model')} × "
        f"{environment.get('cpu_count')} · Python {environment.get('python')} · "
        f"OpenCV {environment.get('opencv')} · {environment.get('ffmpeg')} · "
        f"{report['repetitions']} repetitions · {report['wall_seconds']} s",
        "",
        "| Benchmark | Metric | Median | Min | Max | Unit |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, result in report["benchmarks"].items():
        if result["status"] != "ok":
            lines.append(f"| {name} | **{result['status']}**: {result.get('reason')} | | | | |")
        for metric, value in result["metrics"].items():
            lines.append(f"| {name} | {metric} | {value['median']} | {value['min']} | "
                         f"{value['max']} | {value['unit']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--only", action="append", default=[], help="run only this benchmark name")
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= 20:
        parser.error("repetitions must be in [1, 20]")
    selected = [b for b in DEFAULT_BENCHMARKS if not args.only or b.name in args.only]
    started = time.monotonic()
    results = {}
    for benchmark in selected:
        print(f"running {benchmark.name} x{args.repetitions}", flush=True)
        results[benchmark.name] = run_benchmark(benchmark, args.repetitions)
    report = build_report(results, environment_info(), args.repetitions, time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = summary_markdown(report)
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    # A benchmark that breaks should be visible; skipped ones are not failures.
    return 1 if any(result["status"] == "failed" for result in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
