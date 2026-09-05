# SurvNG Codex Working Agreement

## Objective

Work efficiently and safely on SurvNG. Prioritize correctness, architectural
judgment, and the user's model choice. Optimize cost only after those needs
are met. Avoid unnecessary delegation for repository navigation, mechanical
edits, routine Git work, or known test runs.

The project uses four model roles, when available in the current environment:

- `luna`: bounded, mechanical, read-heavy, repetitive, or clearly specified work
- `terra`: normal software engineering and routine debugging
- `sol`: difficult but bounded debugging and implementation within established boundaries
- `astra`: architecture, subsystem audits, cross-subsystem root-cause analysis,
  high-risk design, and final integration review

## Model-routing policy

### Respect the user's selected model

- An explicit model instruction takes precedence over the default routing below.
- When the user selects or requests Astra, keep Astra responsible for the core
  investigation, architectural decisions, risk assessment, and final review.
  Do not turn an Astra-led task into Sol/Terra work merely to reduce cost.
- Bounded evidence collection, routine implementation against an agreed design,
  and known test runs may be delegated when useful, unless the user requests
  Astra-only work or prohibits delegation. Astra must inspect the consequential
  diffs and evidence before accepting the result; a subagent summary is not
  a substitute for review.
- State the model roles and delegated scope briefly before substantive
  delegation. Report the models actually used, not just the intended roles.
- Use model IDs or agent aliases actually exposed by the current environment.
  This file defines policy; it does not install an agent or change the session's
  selected model. Never claim Astra performed work if that cannot be verified.
- If Astra is required but unavailable, disclose that limitation. Continue
  useful bounded preparation, but do not silently substitute another model
  for the required architectural judgment or final approval.

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

Use `sol` for deeper reasoning within a bounded, established design:

- difficult root-cause analysis confined to one component
- ONVIF/SOAP/Zeep or FFmpeg debugging with a defined scope and contract
- implementation of a complex fix whose ownership and invariants are agreed
- localized concurrency fixes with understood lifecycle boundaries
- targeted technical review supporting an Astra-led investigation

Escalate to Astra when the investigation reveals conflicting ownership,
uncertain system invariants, or consequences spanning multiple subsystems.

### Route to Astra

Use `astra` directly when the task requires system-level judgment:

- whole-repository or subsystem architecture and production-readiness audits
- design or simplification of boundaries, contracts, and ownership
- ambiguous failures spanning capture, motion, inference, persistence, or UI
- races involving admission, retries, leases, shutdown, or durable completion
- event/state-machine redesign and recording/index/storage consistency
- protocol interoperability that changes shared event semantics or policy
- security-sensitive design and changes with broad regression risk
- repeated failed fixes that suggest the underlying model is wrong
- performance tradeoffs involving detection quality, latency, and resource use
- staging related PRs and reviewing their combined behavior

Examples:
- raw SOAP contains the motion topic but Zeep does not preserve it
- PullPoint renewals fail intermittently and the failure path is unclear
- FFmpeg non-monotonic DTS correlates with corrupt/incomplete recordings
- recorder process state and UI/database state diverge
- a motion pipeline needs redesign rather than another conditional

### Escalation

Default escalation path, subject to the user's model instruction:

`Luna -> Terra -> Sol -> Astra`

Start with Astra immediately for Astra-class work. The path is not a requirement
to exhaust cheaper models first, and task size alone does not determine the role.

Do not escalate merely because a task is large. Escalate because it is
ambiguous, reasoning-intensive, high-risk, or the lower tier has failed.

Escalate from Luna to Terra when:
- implementation requires nontrivial judgment
- changes span components
- behavior is ambiguous
- tests reveal a real defect requiring investigation

Escalate from Terra to Sol when:
- root cause remains uncertain after targeted investigation
- raw protocol data contradicts library/parser behavior
- a bounded implementation requires deeper algorithmic or protocol reasoning

Escalate from any role directly to Astra when:
- architectural boundaries or system invariants must change
- concurrency, lifecycle, or durable-state ownership spans components
- two reasonable fixes fail or competing explanations remain unresolved
- regression risk or performance/correctness tradeoffs extend beyond one component

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

### Host-local runtime observability

For questions about the **currently running** SurvNG process, prefer the
owner-only Unix-socket snapshot over saved configuration files, databases, or
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
  secrets in chat, or disabling API security. If the socket is absent, report
  that the running service has not loaded this feature (usually it needs a
  restart) and use safe persisted evidence only when appropriate.

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
- use Astra for lifecycle/race issues when state can diverge across these boundaries
  and Sol for bounded implementation under the established contract
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
3. honor the user's model instruction, then classify the work as Luna, Terra, Sol, or Astra
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
