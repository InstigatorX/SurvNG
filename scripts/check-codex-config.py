#!/usr/bin/env python3
"""Check SurvNG's repository-level Codex skill metadata and references."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SKILLS = {
    "survng-architecture",
    "survng-orchestration",
    "survng-research",
    "survng-verification",
}


def main() -> int:
    """Report missing skills, invalid headers, or stale guidance references."""
    guidance = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    references = set(re.findall(r"`(survng-[a-z0-9-]+)`", guidance))
    skills = {
        path.parent.name: path
        for path in (ROOT / ".agents" / "skills").glob("survng-*/SKILL.md")
    }
    errors = [
        *(f"Missing required skill: {name}" for name in sorted(REQUIRED_SKILLS - skills.keys())),
        *(
            f"Missing skill referenced by AGENTS.md: {name}"
            for name in sorted(references - skills.keys() - REQUIRED_SKILLS)
        ),
        *(f"Skill missing from AGENTS.md: {name}" for name in sorted(skills.keys() - references)),
    ]
    for name, path in sorted(skills.items()):
        contents = path.read_text(encoding="utf-8")
        # These repository skills use plain, single-line name and description fields.
        opening, separator, remainder = contents.partition("\n---\n")
        fields = opening.splitlines()
        if (
            not separator
            or fields[:2] != ["---", f"name: {name}"]
            or len(fields) != 3
            or not re.fullmatch(r"description: [A-Za-z][^\t]*", fields[2])
            or re.search(r":(?:\s|$)|\s#", fields[2].removeprefix("description: "))
            or not remainder.strip()
        ):
            errors.append(f"Invalid skill header or empty body: {path.relative_to(ROOT)}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Codex configuration OK ({len(skills)} skills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
