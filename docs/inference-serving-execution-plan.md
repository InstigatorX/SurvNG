# Inference-serving architecture: execution plan

Status: proposed; implementation has not started. This planning PR changes documentation only.

## Objective and scope

Improve the throughput, memory efficiency, resource ownership, and failure behavior of SurvNG's existing inference workloads using selected serving patterns from Tamserve. Do not add surveillance functionality, change detection models, adopt SAM/EdgeTAM, add distributed deployment, or replace capture/recording infrastructure.

The architecture review examined SurvNG commit 414d4c66e59bf3a366e5a383f0a57462d76daecc and Tamserve commit 975272ec278085bc46b8cbaa5f341154e9ae94e6. The implementation branch starts from v1.2, which may contain newer changes. Reconcile this plan with current code before editing; references below describe the reviewed snapshot, not a guarantee of current line numbers.

No speedup or memory saving has been measured for SurvNG. All proposed optimization benefits are hypotheses until validated on its actual backend and hardware.

## Current execution path and evidence

1. InferenceSupervisor builds independent object workers, routes by pending-request pressure, and protects worker zero from lower-priority tracking work.
2. _InferenceWorker.request uses priority admission, one active request, a shared RawArray input buffer, a synchronous pipe request, response-ID checks, and timeouts.
3. _inference_worker_main creates one OpenVinoDetector per object process and maps the shared input buffer as a NumPy view.
4. OpenVinoDetector owns one compiled model, one infer request, reusable preprocessing buffers, and a detection lock. It temporarily changes the configured confidence threshold during a call.
5. Tracking sessions consume timestamped frames through the same detector service. Motion analysis already has latest-frame replacement and bounded histories.
6. Worker termination, crash accounting, CPU fallback, and inference lifecycle rollback already exist.

Source map, pinned to the review:

- [Supervisor: worker construction and routing](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_runtime/supervisor.py#L82-L115)
- [Supervisor: workload admission and failover](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_runtime/supervisor.py#L742-L806)
- [Worker: shared input and bounded image dimensions](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_runtime/worker.py#L389-L456)
- [Worker: request admission and IPC](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_runtime/worker.py#L494-L629)
- [Child: role ownership and request dispatch](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_runtime/process.py#L136-L244)
- [Detector: model compilation and infer-request creation](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/detector.py#L175-L275)
- [Detector: serialized detection and temporary threshold mutation](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/detector.py#L322-L405)
- [Detector: CPU resizing, letterboxing, buffer reuse](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/detector.py#L868-L904)
- [Motion: frame ownership and latest-frame replacement](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/motion_analysis_service.py#L383-L455)
- [Tracking: existing detection consumer](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/object_track/session.py#L877-L931)
- [Lifecycle: role replacement and rollback](https://github.com/InstigatorX/SurvNG/blob/414d4c66e59bf3a366e5a383f0a57462d76daecc/survng/app/inference_lifecycle.py#L339-L400)

## Patterns to borrow, and limits

Tamserve separates a shared model from per-camera/object state, gathers ready cameras without waiting for slow cameras, groups compatible tensor work, reuses upload buffers, and separates command sending from result collection.

- [Engine model and track ownership](https://github.com/aryamandarda/tamserve/blob/975272ec278085bc46b8cbaa5f341154e9ae94e6/src/tamserve/engine.py#L118-L181)
- [Preparation, staging, and batched encoding](https://github.com/aryamandarda/tamserve/blob/975272ec278085bc46b8cbaa5f341154e9ae94e6/src/tamserve/engine.py#L318-L414)
- [Compatible-state batching](https://github.com/aryamandarda/tamserve/blob/975272ec278085bc46b8cbaa5f341154e9ae94e6/src/tamserve/engine.py#L537-L572)
- [Ready-camera round and batch failure boundary](https://github.com/aryamandarda/tamserve/blob/975272ec278085bc46b8cbaa5f341154e9ae94e6/src/tamserve/worker.py#L342-L383)
- [Dedicated command sender](https://github.com/aryamandarda/tamserve/blob/975272ec278085bc46b8cbaa5f341154e9ae94e6/src/tamserve/router.py#L56-L83)

Do not copy Tamserve's runtime wholesale. Its command and result queues are unbounded; an engine-step exception affects all cameras in that round. Its CUDA/PyTorch optimizations are backend-specific. SurvNG already has shared-memory IPC, bounded motion histories, model warmup/cache, embedded OpenVINO color conversion/normalization, and stronger application-specific priority rules. These are constraints to preserve, not features to reimplement.

## Required invariants

- Preserve capture, recording, incident creation, zone filtering, timestamps, class thresholds, result schemas, and tracking semantics.
- Preserve separate object, face, and ReID processes. Preserve the protected initial-incident lane during the first rollout.
- Model execution concurrency must never share mutable request inputs, thresholds, preprocessing metadata, or outputs unsafely.
- Every in-flight request has a unique ID, lifecycle generation, workload, absolute deadline, and owned input slot.
- A queued cancellation can release a slot immediately; an executing cancellation cannot recycle memory while native inference may still read it.
- Release slots exactly once after completion or confirmed worker termination. Reject late results from retired generations.
- Bound request count, queued input bytes, active slots, batch size, and batch waiting time. A bounded queue of references alone does not bound retained image memory.
- Overload policy is workload-specific: replace/drop stale optional live samples; preserve required refinement semantics or return an explicit failure/defer result.
- One malformed item must not silently invalidate unrelated work. Native process failure remains a process-level failure, with existing failover behavior.
- No silent fallback that hides failures, changes thresholds, or drops required evidence.

## Target shape

Retain InferenceSupervisor as policy owner and process isolation as the failure boundary. Inside an object worker, separate:

- Model owner: compiled model and immutable model metadata.
- Request context: effective threshold, original dimensions, scale/padding, IDs, generation, deadline, and workload.
- Execution slot: infer request, input/preprocessing storage, and output lifetime.
- Scheduler: bounded admission, priority, compatible work grouping, and deadline handling.

These are conceptual responsibilities, not a requirement to introduce four new classes. Follow existing repository conventions and prefer the smallest coherent change.

Start with multiple infer requests sharing one compiled model inside a process. Treat asynchronous request concurrency and true tensor batching as separate experiments. Keep serialized execution for unsupported backends/models.

## Execution phases

### Phase 0 — Reconcile and measure

- [ ] Read current AGENTS.md and applicable nested instructions; inspect branch status and relevant diffs.
- [ ] Re-trace the current paths above and identify changes since the reviewed commit.
- [ ] Locate existing inference, detector, lifecycle, motion, and tracking tests and reuse their fixtures.
- [ ] Record backend/runtime version, model/input/output shapes, device, worker count, warmup/cache state, and existing admission limits.
- [ ] Build a repeatable fixture-based workload covering steady tracking, simultaneous camera incidents, refinement bursts, and mixed optional/initial work.
- [ ] Capture queue/admission time, preprocessing, IPC copy bytes/time, inference, postprocessing, end-to-end latency by workload, throughput, worker RSS, and accelerator memory where measurable.
- [ ] Produce the baseline and identify the dominant cost before selecting optimization settings.

Deliverable: benchmark procedure and baseline table. No runtime default changes.

### Phase 1 — Request-local resource ownership

Primary files: detector.py; inference_runtime/types.py, process.py, worker.py as needed.

- [ ] Pass the effective confidence threshold through detection/postprocessing rather than mutating shared config for each request.
- [ ] Make original dimensions and resize/padding metadata request-local.
- [ ] Identify every reused input/output array and ensure its lifetime is explicit.
- [ ] Introduce a slot abstraction only where needed; initially retain a single slot and serialized execution.
- [ ] Ensure outputs are copied or consumed before the infer request's storage is reused.
- [ ] Keep all supported output formats and current backend fallback paths working.

Acceptance: equivalent results to the baseline for identical inputs; interleaved requests with different thresholds and dimensions cannot contaminate each other; existing tests pass.

Suggested commit: refactor inference request state without changing scheduling.

### Phase 2 — Shared-model execution capacity

Primary files: detector.py; inference_runtime/process.py, worker.py, supervisor.py.

- [ ] Determine backend/model capability using the installed runtime and model; do not assume async behavior or arbitrary batch support.
- [ ] Prototype a small bounded infer-request pool against one compiled model inside an object process.
- [ ] Keep the current public synchronous detection contract; correlate internal completion by request ID and generation.
- [ ] Extend the single-buffer IPC design only as necessary to support multiple owned slots.
- [ ] Specify ownership transitions: free -> admitted -> executing -> completed/failed -> released; distinguish queued and executing cancellation.
- [ ] Apply backpressure before copying or retaining large frames.
- [ ] Keep role isolation and the initial-incident lane. Do not automatically lower object_worker_count.
- [ ] Compare the baseline topology against shared-model execution under equal workload and hardware conditions.

Acceptance: bounded memory under saturation; deterministic request/result correlation; no unsafe reuse on timeout; measurable benefit without incident-latency regression. If unsupported or unhelpful, retain serialized execution and record the evidence.

Suggested commit: optional bounded request pool with compatibility fallback.

### Phase 3 — Bounded opportunistic batching

Depends on Phase 1; enable only when Phase 2 measurements justify it and the model supports it.

- [ ] Batch only compatible model/backend/input-shape requests; preserve per-item threshold and geometric metadata.
- [ ] Start with refinement/tracking requests already admitted by existing policy.
- [ ] Dispatch immediately when no compatible work is ready; never wait for every camera.
- [ ] Set explicit maximum batch size and a short maximum collection delay, capped by the earliest request deadline.
- [ ] Preserve priorities and per-camera fairness; optional batches must not occupy the protected incident lane.
- [ ] Parse outputs per batch item for each supported format; unsupported formats stay on the serialized path.
- [ ] Validate items before dispatch. Define bounded recovery for recoverable batch errors; never replay indefinitely or beyond deadlines.
- [ ] Preserve ordered tracking updates per camera even if inference finishes out of order.

Acceptance: equivalence to individual inference within documented numerical tolerances; no starvation; predictable bounds under heterogeneous camera rates; no degradation of initial-incident tail latency.

Suggested commit: opt-in compatible-request batching.

### Phase 4 — Preprocessing and transfer efficiency

Can be evaluated after Phase 1; combine with pooling only after separate measurements.

- [ ] Measure the parent-side size reduction, IPC copy, child resize/letterbox, and device transfer independently.
- [ ] Evaluate removing redundant resizing versus its effect on IPC bytes and detection quality.
- [ ] Consider backend-native resize/letterbox only if it preserves geometry and improves measured end-to-end cost.
- [ ] Reuse bounded storage per supported shape/slot; cap cached shapes and bytes.
- [ ] Preserve interpolation, padding, color ordering, coordinate restoration, and existing backend-specific paths.
- [ ] Use completion-aware buffer release; do not copy CUDA pinned-memory assumptions into OpenVINO or Core ML.

Acceptance: representative boxes/classes/confidences meet agreed tolerances; original-coordinate mapping remains correct; no unbounded shape-cache growth; measured improvement exceeds benchmark noise.

Suggested commit: targeted preprocessing optimization with equivalence evidence.

### Phase 5 — Transport, failure, and lifecycle hardening

Required before enabling asynchronous execution by default.

- [ ] Audit blocking send/receive paths against absolute request and shutdown deadlines.
- [ ] If transport becomes multi-request, separate bounded command sending from response draining so neither blocks the other indefinitely.
- [ ] Do not mistake a live heartbeat thread for successful inference progress; retain per-request progress deadlines.
- [ ] Preserve terminate/kill escalation, alternate-worker routing, crash accounting, and CPU fallback.
- [ ] On stop/reconfigure, close admission, settle queued requests, fence old results, and release buffers only after consumers are finished.
- [ ] Verify rollback to previous configuration and that camera capture/recording remain running.
- [ ] Ensure recoverable per-item errors remain scoped to that item; document the unavoidable process-wide effect of native crashes.

Acceptance: no orphan processes, stuck callers, slot leaks, double releases, or acceptance of old-generation results under injected failures.

Suggested commit: transport/lifecycle protections needed by the new execution model.

### Phase 6 — Rollout and decision

- [ ] Keep new execution modes opt-in until backend-specific validation passes.
- [ ] Use existing configuration and telemetry conventions; avoid adding UI controls merely to expose implementation details.
- [ ] Record configuration defaults, supported combinations, limits, and fallback reasons.
- [ ] Compare each optimization independently before testing combinations.
- [ ] Document accepted/rejected experiments and the exact configuration to revert.
- [ ] Change defaults or worker topology only in a separate evidence-backed decision.

## Test and benchmark matrix

Use scripts/run-tests.sh according to its current interface and repository CI. Confirm commands in the implementation checkout; do not invent or bypass the project's test runner.

Existing suites identified during planning include tests/test_inference.py, tests/test_inference_lifecycle.py, tests/test_object_tracking.py, tests/test_camera_worker.py, tests/test_config_reload.py, and tests/test_manager_lifecycle.py. Discover detector-specific tests before changing parsers/preprocessing.

| Scenario | Required assertion |
| --- | --- |
| Interleaved thresholds and frame dimensions | No shared-state contamination; correct coordinate restoration |
| One request, batch of one, partial/full batch | Equivalent output schema and detection semantics |
| Out-of-order completion | Correct caller/result mapping; camera tracking order preserved |
| Saturated optional work plus new incident | Protected lane and incident priority preserved |
| Slow/missing camera | Other ready cameras continue; no all-camera barrier |
| Queue saturation | Explicit shedding/defer behavior; bounded retained bytes |
| Cancel before dispatch / timeout during execution | Exactly-once release; executing storage not reused early |
| Malformed item / recoverable model exception | Scoped error, bounded retry, unrelated requests continue |
| Worker crash, native hang, broken pipe | Existing termination, failover, fallback and caller completion |
| Reconfigure during in-flight work | Old generation fenced; rollback works; no orphaned resources |
| Repeated stop/start and shape churn | Memory stabilizes; no slot/process/cache leak |
| Unsupported backend/model/output format | Existing serialized behavior remains available |

Performance protocol: identical fixtures and hardware, separate cold-start and warmed runs, at least five repeated measured runs after warmup, and enough initial requests to report meaningful p95/p99 latency. Suggested load points are 1, 4, and 8 active camera equivalents plus an overload case; these are test scenarios, not product capacity claims. Include a sustained mixed-load run of at least 30 minutes and repeated lifecycle cycles.

Proposed decision gates, to be confirmed against baseline variance before optimization:
- Correctness, priority, resource-lifetime, and failure tests are mandatory.
- Initial-incident p95/p99 must remain within the existing latency budget; reject a >5% repeatable regression against baseline.
- Retain an optimization only if it delivers a repeatable material benefit (target >=10% throughput improvement or >=15% relevant memory reduction) without violating latency/correctness gates.
- If timing variance prevents a conclusion, report inconclusive rather than claiming a win.
- Report raw measurements and hardware/backend details; do not transfer Tamserve benchmark numbers to SurvNG.

## Rollback

Retain the prior serialized path and defaults until rollout gates pass. Turning off pooling/batching must close admission to the affected generation and transition through existing lifecycle management, not recycle active buffers in place. Preserve existing worker-count configuration. Revert each optimization independently; avoid schema changes that make rollback irreversible.

## VS Code Codex handoff

Start with Phase 0, then implement Phase 1 as a behavior-preserving change. Read repository instructions and inspect current code before applying this plan. Use the selected model and one agent unless the user explicitly requests delegation. Work in small reviewable commits and update the checkboxes with evidence.

Do not implement every optimization speculatively. Advance through measurement gates; record unsupported or unsuccessful experiments. Hardware-dependent work that cannot be validated must remain opt-in and clearly identified as unverified. Keep changes limited to existing inference workloads and necessary tests/documentation.

For each implementation checkpoint report: files changed, invariants preserved, tests run and results, measured performance if applicable, outstanding limitations, and the rollback method.

## Planning-PR validation

This PR contains only this execution plan. Source references and the current root working agreement were reviewed. No runtime implementation, test execution, hardware benchmark, or claimed performance improvement is included.
