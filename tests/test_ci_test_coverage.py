"""Contract for scripts/check-ci-test-coverage.py (PR CI test assignment)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check-ci-test-coverage.py"
SPEC = importlib.util.spec_from_file_location("survng_check_ci_test_coverage", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COVERAGE = importlib.util.module_from_spec(SPEC)
# Register before executing so dataclasses can resolve the module namespace.
sys.modules[SPEC.name] = COVERAGE
SPEC.loader.exec_module(COVERAGE)

WORKFLOW = """\
jobs:
  tests:
    steps:
      - name: Single line
        run: scripts/run-tests.sh tests/test_alpha.py tests/test_beta.py
      - name: Continued
        run: |
          scripts/run-tests.sh \\
            tests/test_gamma*.py \\
            tests/test_delta.py
      - name: Unrelated
        run: echo tests/test_not_a_runner_argument.py
"""


def _layout(tmp_path: Path, tests: list[str], workflow: str = WORKFLOW, allowlist: str = "") -> Path:
    (tmp_path / "tests").mkdir()
    for name in tests:
        (tmp_path / "tests" / name).write_text("", encoding="utf-8")
    (tmp_path / "ci.yml").write_text(workflow, encoding="utf-8")
    (tmp_path / "nightly.txt").write_text(allowlist, encoding="utf-8")
    return tmp_path


def _check(root: Path):
    return COVERAGE.check(root, root / "ci.yml", root / "nightly.txt")


BASE_TESTS = [
    "test_alpha.py",
    "test_beta.py",
    "test_gamma.py",
    "test_gamma_extra.py",
    "test_delta.py",
]


def test_runner_patterns_handle_single_line_and_continuations():
    assert COVERAGE.runner_patterns(WORKFLOW) == [
        "tests/test_alpha.py",
        "tests/test_beta.py",
        "tests/test_gamma*.py",
        "tests/test_delta.py",
    ]


def test_consistent_layout_passes_and_counts(tmp_path):
    root = _layout(
        tmp_path,
        [*BASE_TESTS, "test_slow.py"],
        allowlist="# nightly only\ntests/test_slow.py  # heavy\n",
    )
    report = _check(root)
    assert report.ok, COVERAGE.format_report(report, "ci.yml", "nightly.txt")
    assert len(report.pr_ci) == 5
    assert report.nightly_only == {"tests/test_slow.py"}


def test_unassigned_test_file_fails(tmp_path):
    root = _layout(tmp_path, [*BASE_TESTS, "test_new.py"])
    report = _check(root)
    assert not report.ok
    assert report.unassigned == ["tests/test_new.py"]
    assert "tests/test_new.py" in COVERAGE.format_report(report, "ci.yml", "nightly.txt")


def test_missing_allowlist_entry_fails(tmp_path):
    root = _layout(tmp_path, BASE_TESTS, allowlist="tests/test_gone.py\n")
    report = _check(root)
    assert report.missing_allowlisted == ["tests/test_gone.py"]
    assert not report.ok


def test_allowlist_entry_also_in_pr_ci_fails(tmp_path):
    root = _layout(tmp_path, BASE_TESTS, allowlist="tests/test_gamma_extra.py\n")
    report = _check(root)
    assert report.allowlisted_in_pr_ci == ["tests/test_gamma_extra.py"]
    assert not report.ok


def test_pattern_matching_nothing_fails(tmp_path):
    root = _layout(tmp_path, ["test_alpha.py", "test_beta.py", "test_delta.py"])
    report = _check(root)
    assert report.empty_patterns == ["tests/test_gamma*.py"]
    assert not report.ok


def test_unsorted_or_duplicate_allowlist_fails(tmp_path):
    root = _layout(
        tmp_path,
        [*BASE_TESTS, "test_a_slow.py", "test_z_slow.py"],
        allowlist="tests/test_z_slow.py\ntests/test_a_slow.py\ntests/test_z_slow.py\n",
    )
    report = _check(root)
    assert report.unsorted_allowlist
    assert report.duplicate_allowlisted == ["tests/test_z_slow.py"]
    assert not report.ok


def test_main_exit_status(tmp_path, capsys):
    root = _layout(tmp_path, [*BASE_TESTS, "test_new.py"])
    args = ["--root", str(root), "--workflow", "ci.yml", "--allowlist", "nightly.txt"]
    assert COVERAGE.main(args) == 1
    assert "tests/test_new.py" in capsys.readouterr().err
    (root / "nightly.txt").write_text("tests/test_new.py\n", encoding="utf-8")
    assert COVERAGE.main(args) == 0


@pytest.mark.parametrize("missing", ["ci.yml", "nightly.txt"])
def test_main_reports_missing_inputs(tmp_path, capsys, missing):
    root = _layout(tmp_path, BASE_TESTS)
    (root / missing).unlink()
    args = ["--root", str(root), "--workflow", "ci.yml", "--allowlist", "nightly.txt"]
    assert COVERAGE.main(args) == 1
    assert missing in capsys.readouterr().err


def test_repository_assignment_is_consistent():
    root = SCRIPT_PATH.parents[1]
    assert COVERAGE.main(["--root", str(root)]) == 0
