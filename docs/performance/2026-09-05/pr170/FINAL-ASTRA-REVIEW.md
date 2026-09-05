> Completed measurement of PR #170, published after execution.
> Raw captures, recordings, decoded arrays, worker logs and full host-local provenance
> remain under `/root/survng-measurements/pr170-20260905T180430Z/`.
> Statements about no GitHub writes refer to the measurement, before this documentation PR.

# Final Astra review

Recommendation: **performance benefit is inconclusive**. The completed comparison supports correctness on these recorded sequences. It does not justify a claim of bundled or fleet CPU savings, or establish that a component-only variant would be better. No deployment or runtime change is recommended from this evidence alone.

The 20-minute live capture finished before isolated replay began at 18:25:14 UTC. Replay completed at 18:28:18 UTC in 183.9 seconds, within its 900-second global budget. All three cameras completed three alternating before/after pairs, identical initialization and 20-second preroll, and two separate diagnostic passes. The existing focused suite passed all seven tests in 0.510 seconds. No worker/decoder warnings or errors were emitted.

| Camera | Measured pipeline calls/run | CPU changes across three pairs | Median elapsed change | Exact-zero morphology masks | Exact-zero component masks |
|---|---:|---|---:|---:|---:|
| Gate | 117 | +14.51%, −3.32%, −4.71% | −2.18% | 0/351 | 0/351 |
| Lower garage | 50 | −1.18%, −2.61%, −3.22% | +3.40% | 0/150 | 39/150 |
| Back left | 116 | −4.65%, −5.53%, +5.59% | −4.26% | 0/348 | 0/348 |

Across all three cameras, every meaningful output and complete semantic runtime-state fingerprint matched for all 384 compared diagnostic invocations, including preroll. The comparison covers background arrays, noise/threshold accumulators, persistent change age, historical statistics, zone state, blob/tracker state, scoring state, masks and decisions. Locks and timing metrics are excluded. This establishes tested-sequence equivalence; it is not a recall or recording-continuity evaluation.

Exact mask counting found **0/849 morphology-empty opportunities**. The morphology fast path was never taken in the measured footage; its stage elapsed time increased in all nine pairs. Component extraction had 39/849 empty-mask opportunities, all in lower garage: 26% of that camera's mask observations, with nine wholly empty and eight mixed-history invocations. Component stage elapsed time fell in all three lower-garage pairs. These stage measurements are elapsed time, not independently measured stage CPU. A component-only ablation was not run, so this result is suggestive rather than sufficient to choose that variant.

Gate and back-left total CPU changes changed sign across repetitions. Lower-garage CPU decreased modestly while elapsed latency increased in every pair. Alongside the missing morphology-empty cases, this variation prevents a persuasive full-pipeline performance conclusion. Existing timed observations were additionally grouped by exact component-input regimes after execution; those empty/nonempty/mixed/observed-mixture CPU and elapsed tables are saved in `REPORT.md` and `replay-results.json`. That analysis ran no additional workload.

The principal fidelity limit is **main-recording fallback**: no qualifying continuous validated live-source span was available, so all three clips used main recordings. Gate inputs resized to 640×480; lower garage and back left to 640×360. Main-stream codec, field of view and geometry may differ from original live/substream analysis. All selected frames fell within the live capture timestamps; each segment was decoded once, without timestamp fallbacks, then identical immutable frames, cached derivatives and timestamps were reused by both versions. Maximum sampled gaps were 0.250, 0.260 and 0.233 seconds respectively, with no material truncation. The original live frame/admission timestamps and learned state were unavailable.

Source provenance identifies actual pre-change `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c` and deployed checkout `ab4d66dd71e2ab98f91696d1134d54e27df21a77`; their delta is the two intended motion files and focused tests. Replay used the service virtual environment: Python 3.12.3, OpenCV 5.0.0, NumPy 2.4.6, OpenCV thread count 16 and CPU affinity 0–15. Live `/proc` native CV2/NumPy mapping inodes matched the installed files, and every worker checked matching dependencies/settings. The socket does not expose an in-process CV thread query or all options/zone fields. Loaded source is inferred from clean checkout plus pre-start reflog, and omitted effective fields come from unchanged persisted configuration; neither is a complete live-memory attestation.

The reviewed live result is 3.988 service CPU cores over 1199.1 measured seconds, including 2.543 main-process cores. These are post-deployment observations, not a before/after savings estimate. Resource sampling retained 1200 samples at approximately one-second cadence; application sampling retained 14 successful snapshots with adaptive gaps up to 120.17 seconds and a 26.44-second final tail. No sampled disconnection or expected-but-false recording state, new inference failure, or new recorded-decode timeout was observed. This sparse application coverage cannot prove uninterrupted operation.

The live summarizer correctly rejected two camera policy boundaries: foyer and downstairs changed `detection_enabled` and `recording_enabled` from false to true between 18:10:41 and 18:12:41 UTC. This task made no such changes. All requested owner/generation, finite-value and clock-tick checks were reviewed. Stage elapsed totals remain separate from CPU and from overlapping qualification/worker totals.

All work stayed in the authorized measurement directory. There was no service restart, deployment, setting/debug change, extra stream, inference process, package installation, GitHub write or production code edit. The useful next evidence would be representative recorded live/substream footage that actually exercises exact-zero morphology masks and, if deciding between implementations, a separately authorized component-only ablation. Neither was fabricated here.

Published artifacts: `REPORT.md`, `live-summary.json`, and `replay-results.json`. Per-camera timed/diagnostic/decode evidence, source and native-library provenance, and the focused test log remain host-local.
