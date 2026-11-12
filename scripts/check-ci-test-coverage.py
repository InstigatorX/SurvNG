#!/usr/bin/env python3
"""Fail when a tests/test_*.py file is neither run by PR CI nor nightly-only.

PR CI (.github/workflows/ci.yml) runs hand-maintained lists and globs of test
files through scripts/run-tests.sh. This check parses those arguments with
plain text processing (no YAML dependency), expands the globs against the
repository, and compares the result with the actual tests/test_*.py files and
the explicit nightly-only allowlist (tests/ci-nightly-only.txt).

Usage: scripts/check-ci-test-coverage.py [--root DIR] [--workflow PATH]
                                         [--allowlist PATH]
Exit status is 0 when consistent and 1 when any problem is found.
"""
from __future__ import annotations

import argparse
import glob
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNNER = "scripts/run-tests.sh"
TEST_GLOB = "tests/test_*.py"
DEFAULT_WORKFLOW = ".github/workflows/ci.yml"
DEFAULT_ALLOWLIST = "tests/ci-nightly-only.txt"


def _logical_lines(text: str) -> list[str]:
    """Join shell backslash continuations into single logical lines."""
    lines: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            current += stripped[:-1] + " "
            continue
        lines.append(current + stripped)
        current = ""
    if current:
        lines.append(current)
    return lines


def runner_patterns(workflow_text: str) -> list[str]:
    """Return every tests/... argument passed to scripts/run-tests.sh."""
    patterns: list[str] = []
    for line in _logical_lines(workflow_text):
        if RUNNER not in line:
            continue
        # Drop a leading YAML key such as "run:" or "- run:" before the command.
        command = line[line.index(RUNNER) + len(RUNNER):]
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError as exc:
            raise ValueError(f"cannot parse {RUNNER} arguments: {line.strip()!r}: {exc}") from exc
        for token in tokens:
            if token in {"&&", "||", ";", "|"}:
                break
            if token.startswith("tests/"):
                patterns.append(token)
    return patterns


def read_allowlist(path: Path) -> list[str]:
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    return entries


@dataclass
class Report:
    unassigned: list[str] = field(default_factory=list)
    missing_allowlisted: list[str] = field(default_factory=list)
    allowlisted_in_pr_ci: list[str] = field(default_factory=list)
    empty_patterns: list[str] = field(default_factory=list)
    duplicate_allowlisted: list[str] = field(default_factory=list)
    unsorted_allowlist: bool = False
    pr_ci: set[str] = field(default_factory=set)
    nightly_only: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not (
            self.unassigned
            or self.missing_allowlisted
            or self.allowlisted_in_pr_ci
            or self.empty_patterns
            or self.duplicate_allowlisted
            or self.unsorted_allowlist
        )


def _expand(root: Path, pattern: str) -> set[str]:
    matches = glob.glob(pattern, root_dir=root)
    return {Path(match).as_posix() for match in matches if (root / match).is_file()}


def check(root: Path, workflow: Path, allowlist: Path) -> Report:
    report = Report()
    patterns = runner_patterns(workflow.read_text(encoding="utf-8"))
    for pattern in patterns:
        matched = _expand(root, pattern)
        if not matched:
            report.empty_patterns.append(pattern)
        report.pr_ci |= matched

    entries = read_allowlist(allowlist)
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            report.duplicate_allowlisted.append(entry)
        seen.add(entry)
    report.unsorted_allowlist = entries != sorted(entries)
    report.nightly_only = seen
    report.missing_allowlisted = sorted(e for e in seen if not (root / e).is_file())
    report.allowlisted_in_pr_ci = sorted(seen & report.pr_ci)

    actual = _expand(root, TEST_GLOB)
    report.unassigned = sorted(actual - report.pr_ci - report.nightly_only)
    return report


def format_report(report: Report, workflow: str, allowlist: str) -> str:
    sections = [
        (
            report.unassigned,
            f"Test files neither run by {workflow} nor listed in {allowlist} "
            "(add them to a PR CI step, or to the nightly-only list if too slow):",
        ),
        (report.missing_allowlisted, f"Entries in {allowlist} that do not exist:"),
        (
            report.allowlisted_in_pr_ci,
            f"Entries in {allowlist} that are also run by {workflow} (remove them from the list):",
        ),
        (report.empty_patterns, f"{RUNNER} arguments in {workflow} that match no files:"),
        (report.duplicate_allowlisted, f"Duplicate entries in {allowlist}:"),
    ]
    out: list[str] = []
    for items, title in sections:
        if items:
            out.append(title)
            out.extend(f"  {item}" for item in items)
    if report.unsorted_allowlist:
        out.append(f"Entries in {allowlist} are not sorted.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--allowlist", default=DEFAULT_ALLOWLIST)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        report = check(root, root / args.workflow, root / args.allowlist)
    except (OSError, ValueError) as exc:
        print(f"check-ci-test-coverage: {exc}", file=sys.stderr)
        return 1
    if not report.ok:
        print(format_report(report, args.workflow, args.allowlist), file=sys.stderr)
        return 1
    print(
        f"CI test assignment OK: {len(report.pr_ci)} test files in PR CI, "
        f"{len(report.nightly_only)} nightly-only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
