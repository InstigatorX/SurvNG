# Inference ownership: Phase 0 and Phase 1 checkpoint

Implementation of [PR #308](https://github.com/InstigatorX/SurvNG/pull/308), based on
v1.2 `f8291c9f4b0a31f05b44977bcd59df9212fd79fe`. Runtime defaults and scheduling
are unchanged. Pooling, batching, and transfer optimizations remain gated.

## Reconciliation with the reviewed source

The plan reviewed `414d4c6`. The seven relevant detector/runtime/test files have
945 additions and 182 deletions since that revision. The current execution path
is still supervisor device admission → worker priority admission → parent resize
and shared-memory copy → synchronous child detection → Python results over the
pipe. Important changes since the reviewed source:

- Supervisor admission now separately accounts for initials, capped refinements,
  interactive work, offline work, and optional enrichment sources. Optional work
  yields to security work. The protected initial lane still excludes tracking.
- Parent object-frame reduction moved before worker admission, overlapping another
  inference. A late shape-discovery path can prepare the original frame after
  worker startup. Its allocation is caller-owned, not a shared preprocessing slot.
- Cancellation checks and bounded waits participate in admission and worker locking.
  Role reconfiguration includes recovery fencing and rollback. Depth is another
  separately owned process. None of these mechanisms needs changing for Phase 1.
- Detector contracts now validate layout, precision, batch-one static input,
  output formats, graph NMS, fallback compatibility, and backend geometry.
  Model metadata supplies labels. Final YOLO outputs, raw YOLO, segmentation,
  SSD, and Core ML coordinates/confidence need separate threshold propagation.
- Geometry was already returned in a newly allocated dictionary by both
  preprocessing paths. No detector-wide original dimensions or padding state
  needs moving. Existing stage, admission, copy, and resource telemetry is reused.

Fixtures and suites were located in `test_detector*.py`, `test_inference.py`,
`test_inference_lifecycle.py`, `test_object_tracking.py`, `test_camera_worker.py`,
`test_config_reload.py`, `test_manager_lifecycle.py`, and
`test_detection_path_replay.py`. Current root AGENTS.md applies; no nested
AGENTS.md files were found.

## Reproduction and measurement boundaries

Run from the repository root, with its installed runtime dependencies:

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark-inference-serving.py \
  --model models/20261005-174505_openvino_model/best.xml --device CPU \
  --workers 2 --cameras 1 4 8 32 --runs 5 --requests 100 \
  --output /tmp/inference-serving-baseline.json
```

For production qualification, repeat on the intended device with representative
local fixtures (`--frames image1.jpg image2.jpg`) and `--soak-seconds 1800`.
Use a maintenance window or an equivalent isolated host for saturated GPU tests.
Run the same harness against both revisions, holding fixtures, configuration,
hardware, and background load constant. Hashes identify the model XML, weights,
and decoded fixtures; do not commit private images or their paths. The source
commit in the JSON identifies runtime code; the harness can be copied into an
older checkout for comparison.

The harness creates its own supervisor and workers and never opens camera streams,
loads private service configuration, or reconfigures the running application.
Auxiliary face, ReID, and depth roles are disabled in this object-only benchmark.
Its fixed-size waves and executor limit bound generated callers and retained
frame references. These bounds belong to the harness, not to production admission.
Tracking-only, simultaneous initials, refinement bursts, and mixed
tracking/enrichment/initial/refinement waves each run five times at 1, 4, 8, and
32 simultaneous camera equivalents. The 32-camera case is synthetic overload,
not a capacity claim. Warmup and worker startup are reported separately.

Per-request end-to-end samples and result digests are retained in raw output.
Latency percentiles exclude deferred/unavailable results; outcome counts always
include them. Throughput counts only completed requests. Worker admission and IPC
counters are cumulative and include diagnostic status requests. Detector stages
are each worker's trailing 100 detections, so they may span scenario boundaries
or remain stale on a worker which did no work. They must not be treated as exact
per-workload stage breakdowns. End-of-wave RSS/DRM snapshots are observations,
not peak-memory measurements. The separate parent preparation probe measures
resize/no-op costs without IPC, including a generated 3840×2160 input; it does
not measure queueing or device execution.

The default seed-308 random images exercise geometry, resizing and transfers.
They do not represent surveillance scenes or candidate-heavy postprocessing.
The real-image equivalence check below complements them for correctness, not
performance. Background service activity and test execution shared this host;
these are exploratory baseline measurements, not an optimization comparison.

## Baseline environment and results

Measured on 2026-11-13, with Intel Core i9-12900H, Linux x86-64, Python 3.12.3,
OpenVINO 2026.3.1 (runtime build `22476-759c5a6ab8c-releases/2026/3`),
OpenCV 5.0.0. Two independently compiled CPU object workers, LATENCY hint,
one stream each; model cache disabled and warmup enabled. The reviewed local
model has FP32 NCHW input `[1,3,960,960]` and final YOLO output `[1,300,6]`.
Startup including worker creation was 2.195 s; model loads 710.3/585.7 ms and
warmups 79.4/83.1 ms. The accompanying JSON records model/fixture hashes.

The [measurement data](inference-serving-phase0-1.json) contains 80 measured runs
and 8,000 request outcomes (five repeats of 100 at each scenario/load point).
Throughput below is the median of five completed-request rates; p95/p99 pool
successful end-to-end samples across repeats. Failures and deferrals are shown
separately so fast rejected requests cannot improve the reported latency.

| Cameras | Scenario | Completed / deferred / unavailable | Completed/s | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- |
| 1 | tracking | 500 / 0 / 0 | 11.51 | 123.9 | 157.4 |
| 1 | initial | 500 / 0 / 0 | 11.30 | 100.3 | 109.6 |
| 1 | refinement | 500 / 0 / 0 | 10.75 | 100.0 | 105.6 |
| 1 | mixed | 500 / 0 / 0 | 8.74 | 168.0 | 191.9 |
| 4 | tracking | 500 / 0 / 0 | 10.67 | 384.3 | 431.1 |
| 4 | initial | 500 / 0 / 0 | 11.45 | 400.8 | 433.9 |
| 4 | refinement | 500 / 0 / 0 | 11.42 | 421.1 | 450.6 |
| 4 | mixed | 250 / 250 / 0 | 11.38 | 190.4 | 198.2 |
| 8 | tracking | 500 / 0 / 0 | 11.51 | 712.4 | 907.3 |
| 8 | initial | 500 / 0 / 0 | 11.11 | 743.4 | 786.1 |
| 8 | refinement | 500 / 0 / 0 | 11.40 | 721.4 | 748.5 |
| 8 | mixed | 250 / 250 / 0 | 11.27 | 366.3 | 406.5 |
| 32 | tracking | 500 / 0 / 0 | 11.61 | 2656.0 | 2779.1 |
| 32 | initial | 289 / 0 / 211 | 11.27 | 1574.7 | 1601.6 |
| 32 | refinement | 500 / 0 / 0 | 11.42 | 2751.8 | 2835.6 |
| 32 | mixed | 250 / 250 / 0 | 11.27 | 1388.2 | 1463.7 |

Final mixed-load worker snapshots (each stage is its own trailing-100 window):

| Worker | Preprocess avg ms | Native avg ms | Postprocess avg ms | Copy p95 ms | Copied MiB (cumulative) | RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.83 | 162.91 | 0.27 | 1.562 | 10163.2 | 210.4 |
| 2 | 0.78 | 170.92 | 0.29 | 1.7 | 19806.2 | 212.0 |

The separate 3840×2160 parent preparation probe measured median 0.860 ms,
p95 2.431 ms and p99 4.351 ms across 500 calls; smaller default fixtures
required no parent reduction (median about 0.001 ms). CPU workers had no
DRM accelerator allocation. These observations do not establish peak memory
or a production GPU latency budget.

The overload initial lane returned explicit unavailable outcomes when its
existing admission/failover budgets expired. Mixed optional work was deferred
by existing policy. Required initial work was never counted as a successful
completion merely because it returned an error. The baseline exposes these
limits; this ownership change does not attempt to alter overload policy.

Existing limits: one active request and one 64 MiB shared input allocation per
worker; object inputs reduce their longest side to at most twice the model side;
resize cache holds at most eight shapes. Worker request timeout is 15 s.
Multiworker initial requests use a 3 s worker timeout and 0.75 s admission timeout
before alternate-worker routing. Device refinement concurrency is capped at four in this benchmark
(the configuration default). Existing admission heaps do not impose a global queued-byte or
request-count cap; this is a Phase 2/5 concern, not addressed by this serialized
ownership refactor. The benchmark cannot establish unbounded-load safety.

A read-only `./survngctl status` snapshot separately confirmed that the live
service was healthy and used two OpenVINO GPU object workers. It reported average
native inference 28.4 ms over 6,755 requests, zero inference failures, initial
admission p95 0.012 ms, 54 shed enrichment requests and 12 shed tracking requests.
Object RSS was 155.4/157.3 MiB and DRM resident memory 102.6 MiB each in the first
snapshot. These cumulative production observations are not controlled CPU replay
results and do not describe initial-request end-to-end p95/p99.

## Ownership implementation

`detect()` snapshots the effective threshold under its existing execution lock,
validates/clamps overrides exactly as before, and passes it through each backend
and parser including confidence filtering and NMS. Configuration is never
modified during a request. Parser entry points retain an optional keyword default
for existing direct callers and the scalar offline reference parser.

| Resource | Owner and release boundary |
| --- | --- |
| Caller frame / parent resized array | Synchronous caller; never retained in a new scheduler or cache |
| Worker RawArray and child NumPy view | Existing worker admission and IPC lock; child finishes and sends its response before reuse; failure termination path remains unchanged |
| Infer request, canvas, resize cache | Existing detector execution lock through all postprocessing; exclusive startup/warmup before serving |
| Embedded OpenVINO input tensor | Borrowed canvas view, read synchronously before the next preprocessing call |
| OpenCV/Core ML array input | Request-local RGB/tensor arrays; synchronous forward/predict consumption |
| Geometry dictionary and effective threshold | Request-local values, propagated to the relevant parser |
| Native output arrays and segmentation coefficient/prototype views | Consumed under the same lock before another native inference can overwrite storage |
| Returned detections and polygons | Python dictionaries, lists and scalar values; no native tensor views escape |

The existing single slot is sufficient. A new slot class or copying every native
output would add machinery or cost without improving ownership while execution
remains serialized. Comments make the current borrowing contract explicit.
Future concurrency must allocate separate native requests and preprocessing slots.

## Correctness and validation

- Before changes: 131 tests and 14 subtests passed across detector, backend
  contracts, inference and inference lifecycle.
- Added 14 cases: interleaved concurrent thresholds/dimensions across OpenVINO,
  OpenCV and Core ML; all supported output families; Core ML image inputs;
  configuration stability during native execution; retained results after output
  storage reuse; geometry surviving preprocessing reuse and bounded shape churn.
- Real native CPU equivalence: eight retained incident snapshots, source dimensions
  from 2560×1440 to 4512×2512, each replayed at default, .2, .7, .01 and .99
  thresholds against baseline and changed detectors. All 40 comparisons matched
  exactly, including 16 returned detections. Private images, paths and detections
  were not published; the JSON contains only fixture hashes and dimensions.
- Broader validation: 485 tests and 33 subtests passed; six notification-related
  manager lifecycle tests failed. All six also failed in an untouched detached
  `f8291c9` worktree (58 passed, seven failed there; the additional worktree-only
  failure was `test_manager_with_camera_starts_below_media_reserve`). These are
  outside the changed inference path. No notification changes were made.
- The scoped run excluding manager lifecycle passed: 426 tests and 31 subtests.
- Benchmark CLI and native one-worker smoke run passed, including preparation
  probes and the optional soak loop (one second, not the 30-minute gate).
- Ruff on the changed Python files passed. Detector/runtime modules are outside
  the repository's scoped mypy file list.

## Remaining gates and rollback

No optimization was selected. CPU native inference dominates the sampled stage
costs; controlled production-backend measurements must precede choosing pooling
or batching settings. GPU/Core ML hardware replay, candidate-heavy performance,
30-minute saturation/peak-memory measurement, and repeated lifecycle performance
cycles remain unverified. Existing lifecycle tests provide functional coverage,
not those sustained hardware gates. No throughput or memory improvement is claimed.

Revert the detector/test refactor commit to restore prior request handling.
The benchmark and documentation can remain independently. Runtime configuration,
worker topology, schemas, capture, recording and lifecycle behavior are unchanged.
