# SurvNG Codex Working Agreement

## Objective

Work efficiently and safely on SurvNG. Prioritize correctness, architectural
judgment, and the user's selected model. Complete the requested task with
proportionate investigation, focused changes, and meaningful validation.

## Execution and model policy

The primary agent owns the outcome, chooses the workflow, verifies delegated
evidence, and integrates the result. Handle small, tightly coupled tasks directly.
For substantial work, delegate bounded research, implementation, testing, and
independent review where supported and useful. Use the smallest useful team.
If delegation is unavailable or restricted, run the same stages locally and
report the limits of review independence.

- Honor the user's model and reasoning choices. Inherit settings where possible
  and report actual settings only when verifiable. Surface unavailable
  settings before using a fallback. Avoid silent model or effort changes.
- Give each delegate a question or deliverable, an exact baseline, scope and
  permissions, owned paths, expected checks, and a reporting format.
- Keep research and review read-only. Run independent work concurrently only
  when inputs and outputs are independent. Keep one writer per path.
- Prefer isolated worktrees for concurrent writes. If isolation is unavailable,
  serialize writes in the current checkout and preserve existing user changes.
- The primary agent reconciles findings, inspects changes and verifies the
  integrated snapshot. Request independent review from someone who did not
  implement the change. Repeat affected checks after substantive edits.
- Subagents may delegate only when the user authorizes that structure.
- Obtain explicit user approval before merging.

### Workflow skills

Use the repository skills under `.agents/skills/` as appropriate:

- `survng-orchestration` for bounded delegation and integration.
- `survng-research` for read-only investigations and design rationale.
- `survng-architecture` for ownership, interfaces, lifecycle and design choices.
- `survng-verification` for baseline evidence, regression checks and independent review.

## Context efficiency

- Search narrowly first; read the relevant function or range before whole files.
- Do not recursively inspect unrelated directories or repeat commands without
  new evidence or a specific validation need.
- Summarize large logs and retain the excerpts that support the diagnosis.
- Prefer existing project conventions, tools, and tests.
- Exclude `.git/`, `.venv/`, `venv/`, `node_modules/`, `dist/`, `build/`,
  `coverage/`, generated artifacts, model/cache directories, recordings, and
  snapshots from broad searches unless directly relevant.
- Reduce duplicate work and unnecessary context before compromising correctness.

## Change discipline

- Do not modify unrelated code or reformat whole files for a small change.
- Do not change public behavior unless requested or required by the fix.
- Preserve backward compatibility where practical.
- Do not add dependencies when the standard library or an existing dependency
  cleanly solves the problem.
- Never silently swallow errors merely to make logs quiet.
- Diagnostics must be rate-limited or scoped when they can run continuously.
- Mark temporary debug instrumentation and remove it after diagnosis unless
  the user requests a permanent diagnostic.

## SurvNG protocol rules

### Host-local runtime observability

For questions about the currently running SurvNG process, prefer the owner-only
Unix-socket snapshot over saved configuration files, databases, or
unauthenticated HTTP requests:

```bash
./survngctl status
```

- The default socket is `/run/survng/observability.sock`; it is created after
  SurvNG starts and is accessible only to the service owner or root.
- It returns allowlisted in-memory state: effective tracking settings and
  capacity, camera health, detector state, and storage status.
- Treat it as read-only operational evidence. It intentionally excludes
  passwords, API tokens, private keys, stream URLs, raw errors, and all
  mutation commands.
- Do not work around missing access by extracting browser cookies, asking for
  secrets in chat, or disabling API security.
- If the socket is absent or inaccessible, report the observed failure.
  Distinguish a stopped service, unavailable feature, different socket path,
  and insufficient permissions when evidence permits. Use safe persisted
  evidence when appropriate, clearly identifying it as persisted state.

### ONVIF

When debugging ONVIF:

- Distinguish raw SOAP/XML from Zeep-parsed objects.
- Preserve the exact Topic dialect and topic string when relevant.
- Distinguish subscription creation, PullMessages, renew, unsubscribe, and
  reconnect lifecycle failures.
- Do not assume all cameras implement the ONVIF event spec identically.
- Treat vendor-specific topics as evidence, not universal behavior.
- Prefer capability discovery and observed event payloads over hard-coded guesses.

For event bugs, capture the camera/vendor, subscription type, raw Topic,
raw Message payload when needed, parsed representation, and state transition
produced by SurvNG. Redact credentials and other secrets from evidence.

### FFmpeg

For FFmpeg problems:

- Capture the command line, redacting credentials or tokens before reporting it.
- Identify input and output codecs and containers.
- Distinguish decoder errors from timestamp/muxer errors.
- Distinguish camera-stream corruption from SurvNG lifecycle bugs.
- Inspect process start/stop/restart behavior.
- Do not suppress warnings without understanding whether they indicate corrupt
  or incomplete output.

### Recording and state

For recording bugs:

- Trace state across event input, in-memory state, recorder process state,
  database/index state, and filesystem output.
- For lifecycle or race issues, establish ownership, transitions, and
  consistency requirements across these boundaries before changing code.
- Avoid fixes that merely mask inconsistent state.

## Frontend rules

- For small visual changes, keep edits local to the owning component/style.
- Do not redesign neighboring UI without being asked.
- For state/API integration, verify loading, error, empty, and reconnect states
  when relevant.

## Testing

Define the observable pass condition before editing. Establish a baseline
when practical, then use the narrowest meaningful validation first:

1. Run directly affected tests.
2. Run relevant lint/type checks if the changed area uses them.
3. Broaden testing when the impact or project requirements warrant it.

For a behavior-changing bug fix without suitable coverage, add a focused
regression test when practical. Avoid tests that merely duplicate the
implementation or add little protection for a cosmetic change.

Use `scripts/verify-architecture.sh static` for the Codex guidance, CI test
assignment, and dependency lock. Use its `lifecycle`, `recording`, or `media`
suite when investigating those boundaries. Run the smallest individual test
through `scripts/run-tests.sh` when it gives sufficient coverage.

Once appropriate checks pass, do not repeat or expand them without new changes,
failures, or unresolved concerns.

Report tests run, pass/fail results, evidence of pre-existing failures, and
validation that could not be performed. Never claim verification for a test or
runtime check that was not run.

## Git

Before committing:

- Inspect `git status` and the final diff.
- Ensure no unrelated generated files, secrets, logs, recordings, model
  artifacts, or debug dumps are included.
- Do not rewrite history or force-push unless explicitly requested.
- Do not discard uncommitted user changes.

A request to "commit this" does not require a deep repository investigation.
Review the scoped diff, run proportionate validation, and commit.

## Dead-code and cleanup work

Cleanup must be evidence-driven. Before deleting code:

- Confirm references and callers.
- Account for dynamic imports, routes, configuration, reflection, callbacks,
  registration tables, and frontend references.
- Distinguish apparently unused code from proven dead code.

Large cleanup audits should be report-first unless implementation was explicitly
requested.

## Working style

For a normal task:

1. Identify the requested outcome.
2. Inspect enough code to locate the real execution path.
3. Make the smallest coherent change, delegating bounded work when appropriate.
4. Validate proportionately.
5. Report the result, supporting evidence, and remaining limitations concisely.

For difficult debugging:

1. Reproduce the failure or establish reliable evidence.
2. Identify competing hypotheses and run diagnostics that distinguish them.
3. Establish root cause and the relevant ownership and system invariants.
4. Implement the narrowest durable fix and add regression protection.
5. Remove temporary debug code and report remaining uncertainty.

If fixes repeatedly fail, revisit assumptions and broaden the investigation
where evidence points. Do not automatically create more agents or pass the
problem through model tiers.

Do not turn a small request into an architecture review.
Do not turn an architecture problem into a sequence of speculative one-line fixes.
