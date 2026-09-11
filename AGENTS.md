# SurvNG Codex Working Agreement

## Objective

Work efficiently and safely on SurvNG. Prioritize correctness, architectural
judgment, and the user's selected model. Complete the requested task with
proportionate investigation, focused changes, and meaningful validation.

## Execution and model policy

### One agent by default

- The primary agent owns the task end to end: investigation, design,
  implementation, testing, and final review.
- Use the user's selected model for all of that work, including routine
  searches, mechanical edits, Git inspection, and known test runs.
- Do not spawn subagents or delegate work unless the user explicitly requests
  delegation or parallel agent work for the current task.
- Do not ask about delegation for routine work. Continue directly.
- Task size, complexity, or the availability of cheaper models does not by
  itself authorize delegation.
- Do not use a fixed orchestrator/worker topology, model-role matrix, or
  automatic model-escalation ladder.
- Independent tool calls may run concurrently when useful and safe. This
  does not require additional agents.

### Respect the user's settings

- Honor explicit model and reasoning-effort instructions. Do not silently
  substitute another model, reduce the selected model to a coordinator, or
  change settings to pursue speculative cost savings.
- This file defines working behavior; it does not select a model, change
  reasoning settings, install agents, or override platform permissions.
- Use only capabilities actually available in the current environment.
  Never claim a model or reasoning setting was used unless verifiable.
- If a specifically requested model is unavailable, disclose that limitation
  and continue useful preparation within the user's authorized scope.

### When the user requests delegation

- Keep delegation limited to the requested scope and the smallest useful
  number of agents. Parallel work should have independent, bounded outcomes.
- Before spawning, briefly state each subtask and why delegation helps.
  Honor any model, effort, or agent-count limits the user specified.
- Give each agent only the context, evidence, ownership boundaries, and
  completion criteria needed for its subtask.
- Do not delegate the same investigation to multiple agents unless the user
  requested independent assessments. Do not let agents recursively delegate
  unless the user explicitly authorized that structure.
- Do not run parallel write agents against the same files.
- The primary agent remains responsible for integration and must inspect
  consequential diffs and supporting evidence. A summary alone is insufficient.
- Report the delegated scope and models actually used when verifiable.

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

Use the narrowest meaningful validation first:

1. Run directly affected tests.
2. Run relevant lint/type checks if the changed area uses them.
3. Broaden testing when the impact or project requirements warrant it.

For a behavior-changing bug fix without suitable coverage, add a focused
regression test when practical. Avoid tests that merely duplicate the
implementation or add little protection for a cosmetic change.

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
3. Make the smallest coherent change using the selected model directly.
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
