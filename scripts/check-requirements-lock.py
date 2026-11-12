#!/usr/bin/env python3
"""Offline check that requirements.lock is not stale relative to requirements.txt.

Every requirement in requirements.txt must have a pin in requirements.lock that
satisfies its version specifier, and every package the lock attributes directly
to requirements.txt ("# via -r requirements.txt") must still be listed there.
This does not resolve dependencies or contact an index; regenerate the lock
with scripts/lock-requirements.sh when it fails.

Requires `packaging` (installed from requirements.lock; pip also vendors it).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:  # pragma: no cover - fallback for a bare interpreter
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def read_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError(f"{path.name}:{number}: unsupported option line {line!r}")
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
    return requirements


def read_lock(path: Path, source_name: str) -> tuple[dict[str, Version], set[str]]:
    """Return {canonical name: pinned version} and names attributed to source_name."""
    pins: dict[str, Version] = {}
    direct: set[str] = set()
    current: str | None = None
    marker = f"-r {source_name}"
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = PIN_RE.match(raw)
        if match:
            current = canonicalize_name(match.group(1))
            pins[current] = Version(match.group(2))
            continue
        comment = raw.strip()
        if current and comment.startswith("#") and comment.lstrip("# ").removeprefix("via ").strip() == marker:
            direct.add(current)
    return pins, direct


def check(requirements_path: Path, lock_path: Path) -> list[str]:
    problems: list[str] = []
    requirements = read_requirements(requirements_path)
    pins, direct = read_lock(lock_path, requirements_path.name)
    wanted = set()
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        wanted.add(name)
        pinned = pins.get(name)
        if pinned is None:
            problems.append(f"{requirement} has no pin in {lock_path.name}")
        elif not requirement.specifier.contains(pinned, prereleases=True):
            problems.append(f"{requirement} is not satisfied by locked {name}=={pinned}")
    for name in sorted(direct - wanted):
        problems.append(f"{lock_path.name} still pins {name} for {requirements_path.name}, which no longer lists it")
    return problems


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requirements", type=Path, default=root / "requirements.txt")
    parser.add_argument("--lock", type=Path, default=root / "requirements.lock")
    args = parser.parse_args(argv)
    try:
        problems = check(args.requirements, args.lock)
    except (OSError, ValueError) as exc:
        print(f"check-requirements-lock: {exc}", file=sys.stderr)
        return 1
    if problems:
        print(f"{args.lock.name} is stale; run scripts/lock-requirements.sh:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"{args.lock.name} satisfies {args.requirements.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
