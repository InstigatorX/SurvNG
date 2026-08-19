# SurvNG Codex Working Agreement

## Objective

Work efficiently and safely on SurvNG. Use the least expensive model that can
reliably complete the task. Avoid wasting high-capability reasoning on
repository navigation, mechanical edits, routine Git work, or known test runs.

The project uses three custom agents:

- `luna`: bounded, mechanical, read-heavy, repetitive, or clearly specified work
- `terra`: normal software engineering and routine debugging
- `sol`: difficult root-cause analysis, architecture, protocol behavior, or high-risk work

## Model-routing policy

### Do not delegate reflexively

Subagents have overhead. The primary agent may complete a trivial action itself
when spawning an agent would cost more than doing the work.

Delegate when the subtask is substantive enough to benefit from a specialized
model, when it keeps noisy exploration out of the main context, or when the
requested implementation clearly belongs to another tier.

Do not create parallel write agents against the same files.

### Route to Luna

Use `luna` for low-risk work with a clear definition of done:

- find files, symbols, references, definitions, routes, handlers, and call sites
- inspect repository layout
- targeted code search
- inspect `git status`, `git diff`, `git log`, branches, or blame
- summarize bounded logs, tracebacks, or test failures
- run an already-known test/lint/build command and report the result
- simple one-file edits
- small CSS/layout/text changes
- mechanical renames
- comments, docs, formatting, or boilerplate
- repetitive changes with an established pattern
- collect evidence for another agent

Examples:
- "Where is PullMessages parsed?"
- "Find every reference to RuleEngine/MyRuleDetector/VehicleDetect."
- "Move this button beside the camera selector."
- "Run the ONVIF unit tests and summarize failures."
- "Show me what changed since the last commit."

### Route to Terra

Use `terra` for normal implementation and ordinary debugging:

- routine bug fixes
- normal features
- Python backend changes
- React/frontend changes
- API changes
- multi-file implementation
- configuration support
- tests
- moderate refactors
- known ONVIF mappings
- ordinary camera integrations
- well-understood recording/snapshot changes

Examples:
- "Add support for this known Reolink ONVIF topic."
- "Add an endpoint and UI control for this setting."
- "Implement the sniffer panel using this defined interface."
- "Fix this reproducible settings-save bug."
- "Add tests for this recorder-state transition."

Terra is the normal implementation tier.

### Route to Sol

Use `sol` only when deeper reasoning is justified:

- difficult or ambiguous root-cause analysis
- architecture or subsystem design
- ONVIF/SOAP/Zeep discrepancies
- vendor protocol interoperability
- FFmpeg timestamp, muxing, decoding, or process-lifecycle problems
- concurrency, race conditions, async lifecycle, or state corruption
- event/state-machine design
- recording/index/storage consistency
- large refactors with meaningful regression risk
- security-sensitive behavior
- repeated failed fixes
- problems spanning several subsystems with an unclear root cause

Examples:
- raw SOAP contains the motion topic but Zeep does not preserve it
- PullPoint renewals fail intermittently and the failure path is unclear
- FFmpeg non-monotonic DTS correlates with corrupt/incomplete recordings
- recorder process state and UI/database state diverge
- a motion pipeline needs redesign rather than another conditional

### Escalation

Prefer:

`Luna -> Terra -> Sol`

Do not escalate merely because a task is large. Escalate because it is
ambiguous, reasoning-intensive, high-risk, or the lower tier has failed.

Escalate from Luna to Terra when:
- implementation requires nontrivial judgment
- changes span components
- behavior is ambiguous
- tests reveal a real defect requiring investigation

Escalate from Terra to Sol when:
- root cause remains uncertain after targeted investigation
- two reasonable fixes fail
- raw protocol data contradicts library/parser behavior
- concurrency, lifecycle, timestamp, or state corruption is suspected
- architectural boundaries must change
- regression risk is high

When escalating, pass a concise evidence packet instead of making the next
agent rediscover everything:
- observed behavior
- exact errors/log excerpts
- relevant files/symbols
- commands already run
- hypotheses tested
- what failed
- remaining question

## Context-efficiency rules

1. Search narrowly first.
2. Do not read entire large files when a relevant function or range is enough.
3. Do not recursively inspect unrelated directories.
4. Do not repeatedly run the same command unless new evidence justifies it.
5. Summarize large logs instead of pasting them into the main context.
6. Prefer existing tests and project conventions over inventing new harnesses.
7. Do not investigate generated artifacts, virtualenvs, `node_modules`, build
   output, model weights, recordings, or snapshots unless directly relevant.
8. Exclude these from broad searches unless explicitly needed:
   - `.git/`
   - `.venv/`
   - `venv/`
   - `node_modules/`
   - `dist/`
   - `build/`
   - `coverage/`
   - model/cache directories
   - recordings
   - snapshots

## Change discipline

- Do not modify unrelated code.
- Do not reformat whole files for a small functional change.
- Do not change public behavior unless requested or required by the fix.
- Preserve backward compatibility where practical.
- Do not add dependencies when the standard library or an existing dependency
  cleanly solves the problem.
- Never silently swallow errors merely to make logs quiet.
- Diagnostics must be rate-limited or scoped when they can run continuously.
- Temporary debug instrumentation must be clearly marked and removed once the
  diagnosis is complete unless the user requests a permanent diagnostic.

## SurvNG protocol rules

### ONVIF

When debugging ONVIF:
- distinguish raw SOAP/XML from Zeep-parsed objects
- preserve the exact Topic dialect and topic string when relevant
- distinguish subscription creation, PullMessages, renew, unsubscribe, and
  reconnect lifecycle failures
- do not assume all cameras implement the ONVIF event spec identically
- treat vendor-specific topics as evidence, not universal behavior
- prefer capability discovery and observed event payloads over hard-coded guesses

For event bugs, capture:
- camera/vendor
- subscription type
- raw Topic
- raw Message payload when needed
- parsed representation
- state transition produced by SurvNG

### FFmpeg

For FFmpeg problems:
- capture the exact command line
- identify input codec/container and output codec/container
- distinguish decoder errors from timestamp/muxer errors
- distinguish camera-stream corruption from SurvNG lifecycle bugs
- inspect process start/stop/restart behavior
- do not suppress FFmpeg warnings without understanding whether they indicate
  corrupt or incomplete output

### Recording and state

For recording bugs:
- trace state across event input, in-memory state, recorder process state,
  database/index state, and filesystem output
- treat lifecycle/race issues as Sol-class when state can diverge
- avoid fixes that merely mask inconsistent state

## Frontend rules

For small visual changes:
- use Luna when behavior is already understood
- keep edits local to the owning component/style
- do not redesign neighboring UI without being asked

For state/API integration:
- use Terra
- verify loading, error, empty, and reconnect states when relevant

## Testing

Use the narrowest meaningful validation first.

After a change:
1. run directly affected tests
2. run relevant lint/type checks if the changed area uses them
3. broaden testing only when the impact warrants it

If no suitable test exists for a behavior-changing bug fix, add a focused
regression test when practical.

Report:
- tests run
- pass/fail result
- failures that appear pre-existing
- validation that could not be performed

Never claim a fix is verified when the relevant test or runtime check was not run.

## Git

Routine Git inspection belongs to Luna or the parent.

Before committing:
- inspect `git status`
- inspect the final diff
- ensure no unrelated generated files, secrets, logs, recordings, model
  artifacts, or debug dumps are included
- do not rewrite history unless explicitly requested
- do not force-push unless explicitly requested
- do not discard uncommitted user changes

A request to "commit this" does not require a deep repository investigation.
Review the scoped diff, run proportionate validation, and commit.

## Dead-code and cleanup work

Cleanup must be evidence-driven.

Before deleting code:
- confirm references/callers
- account for dynamic imports, routes, configuration, reflection, callbacks,
  registration tables, and frontend references
- separate "apparently unused" from "proven dead"

Large cleanup audits should be report-first unless implementation was explicitly requested.

## Working style

For a normal task:
1. identify the requested outcome
2. inspect only enough code to locate the real execution path
3. classify the work as Luna, Terra, or Sol
4. delegate only when useful
5. make the smallest coherent change
6. validate proportionately
7. report the result concisely

For difficult debugging:
1. reproduce or establish evidence
2. identify competing hypotheses
3. run discriminating diagnostics
4. establish root cause
5. implement the narrowest durable fix
6. add regression protection
7. remove temporary debug code
8. report remaining uncertainty

Do not turn a small request into an architecture review.
Do not turn an architecture problem into a sequence of speculative one-line fixes.
