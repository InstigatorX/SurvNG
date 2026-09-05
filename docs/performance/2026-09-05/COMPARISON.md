# Post-deployment comparison

Supporting raw captures, JSON evidence, collector sources, and metric definitions referenced below are retained on the measurement host under `/root/survng-measurements/`; they are not included in this documentation-only change.

Generated 2026-09-05T03:35:10.334831+00:00. Post capture: finished.

Earlier raw measurements, collector source, definitions and scenario checklist are preserved under `pre-deployment/`. Hashes are recorded in `preservation.json`. See [CONFIGURATION.md](CONFIGURATION.md) for deployment/configuration evidence.

## Startup and comparison boundaries

Post capture began at process age 112 seconds: the first ~112 seconds of startup were not captured. Samples below age 300 seconds or with unready detector/workers/cameras are separated. The five-minute cutoff is a predeclared conservative analysis rule, not a measured warmup completion. Model/cache/background learning can persist beyond it. The earlier capture began nearly eight hours after startup; its cold-start behavior is unavailable.

Post startup/unready: 185.0 resource seconds, 37 CPU intervals, mean 297.481% of one core; 13 application snapshots. These are excluded below.

## Unmatched steady-state candidates — descriptive, not causal

| Measure | Earlier | Post |
|---|---:|---:|
| Resource seconds / CPU interval count | 1795.0 / 359 | 1606.4 / 321 |
| Mean CPU, 100%=one core | 243.932% | 316.027% |
| Peak sampled interval CPU | 547.930% | 851.516% |
| Mean cgroup-accounted memory, GiB | 10.285 | 10.743 |
| Inference completions / covered seconds | 597 / 1789.8 | 481 / 1580.7 |
| Inference completions/minute | 20.014 | 18.257 |
| Failed inference completions | 0 | 0 |
| Application snapshots / max queue depth | 74 / 0 | 92 / 0 |
| Unhealthy enabled-camera samples | 0 | 0 |
| Enabled-recording false samples | 0 | 0 |

Measured changes are reported symmetrically: higher CPU/memory or failures are adverse observations; lower values are favorable observations. Neither proves a code regression/improvement without equivalent scene activity and readiness. All queue maxima are sampled gauges and can miss bursts; recorder flags do not validate media continuity.

Post CPU is 29.6% higher in these unmatched windows. This is an adverse resource-use signal to investigate, not an established code regression.

## Per-camera workload and preparation

Preparation fps is estimated from nominal one-minute buckets, not completed qualification cadence. Counts are not differenced again. Durations differ; compare counts/minutes, not raw totals. Episode counters reflect changed admission semantics, so they cannot independently establish equal real-world activity.

| Camera | Minutes pre/post | Preparation fps pre/post | Admissions pre/post | Superseded frames pre/post |
|---|---:|---:|---:|---:|
| back-left | 28/24 | 3.985/4.385 | 9/6 | 0/0 |
| back-middle | 28/24 | 3.735/3.390 | 4/2 | 0/0 |
| back-right | 28/24 | 4.072/4.332 | 5/2 | 0/0 |
| boiler | 28/24 | 3.975/4.024 | 8/4 | 0/0 |
| downstairs | 28/24 | 3.832/3.992 | 2/3 | 0/0 |
| foyer | 28/24 | 3.789/3.929 | 2/2 | 0/0 |
| front-door | 28/24 | 4.013/4.033 | 0/0 | 0/0 |
| front-side | 28/24 | 3.974/4.376 | 0/0 | 0/0 |
| gate | 28/24 | 3.996/4.032 | 2/3 | 0/0 |
| lower-garage | 28/24 | 3.004/2.897 | 0/1 | 0/0 |
| sherry-garage | 28/24 | 4.383/4.393 | 0/0 | 0/0 |
| steve-garage | 28/24 | 4.121/4.322 | 0/0 | 0/0 |
| upper-garage | 28/24 | 3.561/3.290 | 1/2 | 0/0 |

## Evidence and confidence

- High confidence: process/cgroup accounting and recorded snapshot values for their specified windows. Per-process CPU can miss short-lived children; cgroup totals include them. GPU engine and process-name breakdowns are in comparison.json.
- Moderate confidence: configuration equivalence where confirmed by identical persisted-file fingerprint, unchanged schema defaults, and exported runtime flags. Exact loaded model identity and unexported runtime overrides remain unverified.
- No causal confidence yet: the two windows occur at different times with different process age/cache/scene-learning history and unmarked scene activity. PRs #157–#159 also changed behavior; this is a combined deployment comparison, not an isolated six-PR experiment.
- No verified equivalent ground-truth periods exist in the earlier capture. Quiet/normal/busy classification, detection recall and observed misses are unavailable. A lower inference rate could be reduced redundant work, less activity, or missed triggers; it is not automatically an improvement.
- Inference compute, limiter wait, durable refinement queue age, attempts/retry reason and end-to-end persisted-event latency remain distinct. Missing stages are unavailable, not estimated from completion counters. No camera/fleet percentiles are averaged.

## Next steps

1. Inspect adverse as well as favorable steady-window changes alongside per-camera preparation/admission rates and process CPU breakdown. Compare only aligned measurement coverage; do not compare startup peaks with old steady-state.
2. Run the unchanged live scenario checklist with explicit timestamps. Record scene truth separately from detections. The old unmarked capture cannot supply retrospective missed-event ground truth.
3. If attribution remains necessary, authorize only the previously proposed socket allowlist additions for existing numeric analysis/limiter/refinement/recording telemetry. No instrumentation was changed.
4. Investigate disabled-camera preparation and any newly observed backlog/failures according to measured resource or quality impact. Do not infer savings from call counts alone.

Collector sampling policy is unchanged: process ~5 s, application initially ~15 s with identical >250 ms backoff, DRM ~15 s, existing indexed telemetry ~60 s. Actual application intervals may differ because request costs differ. No inference/replay, configuration edits, service restart, package installation or recording interruption was performed.
