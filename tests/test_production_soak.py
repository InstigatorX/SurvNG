from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "production-soak.py"
_SPEC = importlib.util.spec_from_file_location("production_soak", _SCRIPT)
soak = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(soak)


def _histogram(p99: float, maximum: float) -> dict:
    return {"max_ms": maximum, "recent": {"p99_ms": p99, "max_ms": maximum}}


def _row(uptime: float, *, new_fields: bool, lag_p99: float = 1.0, busy: int = 0) -> dict:
    runtime = {
        "process": {"instance_id": "one", "pid": 10, "uptime_seconds": uptime},
        "event_loop": {"tasks": 6, "probe_lag_ms": 0.2},
        "cameras": [],
    }
    if new_fields:
        runtime["event_loop"]["lag"] = _histogram(lag_p99, lag_p99 * 2)
        runtime["databases"] = {
            "domains": {
                "main": {"busy_errors": busy, "transactions_failed": 0,
                         "transaction_ms": _histogram(3.0, 20.0)},
                "jobs": {"busy_errors": 0, "transactions_failed": 0,
                         "transaction_ms": _histogram(2.0, 10.0)},
            },
            "main_write_lock": {"available": True, "wait_ms": _histogram(1.0, 2.0),
                                "hold_ms": _histogram(4.0, 8.0)},
        }
    return {
        "runtime": runtime,
        "processes": [{"pid": 10, "rss_bytes": 1, "threads": 2, "fds": 3}],
        "tree_totals": {"rss_bytes": 1, "process_count": 1},
        "snapshot_ms": 12.0,
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_summary_flags_loop_lag_and_busy_error_breaches(tmp_path: Path) -> None:
    rows = [_row(3600 + i, new_fields=True, lag_p99=1.0, busy=0) for i in range(3)]
    rows.append(_row(3700, new_fields=True, lag_p99=400.0, busy=2))
    summary = soak.summarize(_write(tmp_path / "s.jsonl", rows))
    group = summary["generations"]["one"]
    breached = {item["metric"]: item for item in group["threshold_breaches"]}
    assert set(breached) == {"loop_lag_recent_p99_ms", "db_main_busy_errors"}
    assert breached["loop_lag_recent_p99_ms"]["samples_over"] == 1
    assert breached["db_main_busy_errors"]["observed"] == 2
    assert group["unavailable_metrics"] == []
    assert group["metrics"]["db_jobs_transaction_recent_max_ms"]["max"] == 10.0
    assert summary["threshold_breach_count"] == 2

    relaxed = soak.summarize(
        tmp_path / "s.jsonl",
        {"loop_lag_recent_p99_ms": 500.0, "db_busy_errors_increase": 5.0},
    )
    assert relaxed["generations"]["one"]["threshold_breaches"] == []


def test_summary_reports_missing_fields_from_older_builds_as_unavailable(tmp_path: Path) -> None:
    rows = [_row(100, new_fields=False)] + [_row(3600 + i, new_fields=False) for i in range(2)]
    summary = soak.summarize(_write(tmp_path / "old.jsonl", rows))
    group = summary["generations"]["one"]
    assert group["threshold_breaches"] == []
    assert "loop_lag_recent_p99_ms" in group["unavailable_metrics"]
    assert "db_*" in group["unavailable_metrics"]
    assert group["metrics"]["loop_probe_lag_ms"]["max"] == 0.2


def test_threshold_argument_rejects_unknown_names() -> None:
    assert soak._threshold_argument("loop_lag_recent_max_ms=250") == ("loop_lag_recent_max_ms", 250.0)
    with pytest.raises(Exception):
        soak._threshold_argument("rss=1")
