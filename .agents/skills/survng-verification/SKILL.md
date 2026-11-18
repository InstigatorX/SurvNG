---
name: survng-verification
description: Establish baseline evidence, prevent SurvNG regressions, run targeted checks, and independently review a candidate change against its acceptance criteria.
---

# Verify SurvNG changes

1. Define a pass condition in observable terms and identify the normal entry
   point. For a bug, reproduce the failure before changing production code.
   When a focused regression test is practical and coverage is missing,
   confirm it fails for the intended reason before the fix. Record a safe
   substitute and the remaining gap when reproduction is impractical.
2. Use existing checks and fixtures first. Inspect PR and nightly CI scope
   separately. Use `scripts/run-tests.sh` with explicit test paths for
   targeted Python checks after dependencies are installed. Match the
   affected boundary with appropriate lint, type checks, or integration
   checks. Use isolated test state for subprocesses, storage, and ports.
3. Verify expected output and relevant side effects through the owning
   boundary. Check retry, rollback, cancellation, reload, and existing data
   where they can affect the observed behavior. Preserve existing assertions
   unless the requested contract changes.
4. Give a reviewer who did not implement the change the intended behavior,
   base, candidate diff, and acceptance criteria. Ask for actionable findings
   with severity, file and line, impact, and a reproduction or concrete
   code-path evidence. Mark unproven risks and allow a clean review.
5. Check findings against the candidate, fix within scope, and recheck the
   impacted tests and review after changes. Record the tested snapshot,
   commands and outcomes, environmental gaps, and final diff scope.
