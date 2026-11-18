---
name: survng-research
description: Investigate SurvNG read-only, trace how a subsystem works, or recover why a design changed from Git history, project records, and upstream documentation.
---

# Research SurvNG

1. Frame a question and record the revision and any relevant working-tree
   changes. Keep the checkout and live services untouched during read-only
   work. Use an independent read-only evidence check when a consequential
   claim would benefit from it and delegation is available.
2. Start at the normal entry point. Follow configuration, registration,
   dispatch, state changes, and observable output. Search focused symbols,
   their callers, tests, and CI configuration.
3. For design rationale, follow `git log -- <path>`, `git log -S`, or blame
   to the relevant change and inspect its diff and available PR discussion.
   For external contracts, prefer primary docs or source for the relevant
   version. Mark missing source material and inferred intent.
4. Resolve conflicts between docs, current code, and tests by identifying
   what each actually establishes. Distinguish static behavior, assertions
   exercised by a test, and live observations. For authorized host-local
   runtime inspection, follow the observability rules in AGENTS.md.
5. Return an answer with file and line citations, source revisions, and gaps.
   Describe the smallest check that could resolve each important uncertainty.
