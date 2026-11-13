from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-benchmarks.py"
_SPEC = importlib.util.spec_from_file_location("run_benchmarks", _SCRIPT)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner  # dataclasses resolve the module by name
_SPEC.loader.exec_module(runner)

_STUB = (
    "import json, pathlib, sys;"
    "counter = pathlib.Path(sys.argv[1]);"
    "n = int(counter.read_text()) if counter.exists() else 0;"
    "counter.write_text(str(n + 1));"
    "print('progress text');"
    "print(json.dumps({'ms_per_frame': {'case': [3.0, 1.0, 2.0][n % 3]}}))"
)


def _stub(tmp_path: Path, **kwargs) -> object:
    return runner.Benchmark(
        name="stub",
        command=(sys.executable, "-c", _STUB, str(tmp_path / "count")),
        extract=lambda result: {"case": (result["ms_per_frame"]["case"], "ms")},
        **kwargs,
    )


def test_aggregate_reports_median_min_max_and_repetitions() -> None:
    result = runner.aggregate([
        {"a": (3.0, "ms"), "b": (10, "frames")},
        {"a": (1.0, "ms")},
        {"a": (2.0, "ms"), "b": (12, "frames")},
    ])
    assert result["a"] == {"unit": "ms", "median": 2.0, "min": 1.0, "max": 3.0,
                           "repetitions": 3, "values": [3.0, 1.0, 2.0]}
    assert result["b"]["median"] == 11.0 and result["b"]["repetitions"] == 2


def test_stubbed_benchmark_run_and_report_schema(tmp_path: Path) -> None:
    result = runner.run_benchmark(_stub(tmp_path), 3, cwd=tmp_path)
    assert result["status"] == "ok"
    assert result["metrics"]["case"]["median"] == 2.0
    assert result["metrics"]["case"]["min"] == 1.0
    assert result["metrics"]["case"]["max"] == 3.0

    report = runner.build_report({"stub": result}, {"commit": "abc", "cpu_count": 4}, 3, 1.5)
    encoded = json.loads(json.dumps(report, allow_nan=False))
    assert set(encoded) == {"schema_version", "generated_at", "repetitions", "wall_seconds",
                            "environment", "benchmarks"}
    assert encoded["benchmarks"]["stub"]["metrics"]["case"]["unit"] == "ms"
    assert "| stub | case | 2.0 | 1.0 | 3.0 | ms |" in runner.summary_markdown(report)


def test_missing_tool_is_skipped_and_failures_are_reported(tmp_path: Path) -> None:
    skipped = runner.run_benchmark(_stub(tmp_path, requires=("no-such-tool-xyz",)), 2, cwd=tmp_path)
    assert skipped["status"] == "skipped" and skipped["metrics"] == {}

    failing = runner.Benchmark(
        name="broken", command=(sys.executable, "-c", "raise SystemExit(3)"),
        extract=lambda result: {},
    )
    failed = runner.run_benchmark(failing, 2, cwd=tmp_path)
    assert failed["status"] == "failed" and failed["reason"] == "exit 3"


def test_environment_info_has_required_fields() -> None:
    info = runner.environment_info()
    for key in ("commit", "python", "opencv", "ffmpeg", "cpu_model", "cpu_count"):
        assert key in info
