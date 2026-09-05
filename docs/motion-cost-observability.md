# Measuring motion processing cost

`./survngctl status --compact` includes `cameras[].motion`, a bounded projection
of measurements already collected by the motion runtime. This supports the
CPU investigation in PRs #167 and #168. Loading this change requires deployment
and a service restart; creating or merging the PR does not change a running
process. It changes reporting only, not detection or learning policy.

The socket remains owner-only. This addition reads the camera status already
built for the request. It adds no frame acquisition, image encoding, inference,
per-frame timer, database scan, or diagnostic lease. It excludes result payloads,
stage options, debug images and errors. Stage identifiers and implementation IDs
are bounded to 128 identifier characters. Each of the qualification, observation
and fusion pipelines exports at most 64 stage rows, with an explicit `truncated`
flag. Missing or nonfinite numbers and unavailable booleans are `null`, not zero;
an unavailable motion or pipeline snapshot is also `null`.

## Fields and interpretation

| Path under `cameras[].motion` | Meaning |
|---|---|
| `analysis.preprocess_count/total_ms` | Successful preparation calls and elapsed time for resize/gray/optional blur, before ring updates. |
| `analysis.qualification_count/total_ms` | Continuous-analysis invocations and elapsed time after acquiring an analysis slot, including result/policy bookkeeping. Early returns and failures count too. |
| `analysis.capture_to_analysis_*` | Capture-to-worker handoff delay before preparation. |
| `analysis.analysis_cycle_*` | Worker cycle including preparation and inline qualification; excludes qualification later executed on a deferred wakeup. |
| `continuous_frames` / `continuous_candidates` | Completed continuous results / accepted continuous results, not physical frames or distinct incidents. |
| `pipelines.<name>.stages[]` | Stage ID/implementation, calls, failures, raw cumulative `total_ms`, rounded last/max elapsed time. Includes all invocations of that live pipeline, not only continuous qualification. |
| `limiter.analysis_wait_ms_total/max/p95/p99` | Successful analysis-admission waits, including zero waits. No lifetime wait count exists, so no exact interval mean is implied. |
| `analysis.analysis_slot_deferrals` | Deferred request episodes, not every acquisition attempt. |
| `demand` | Current policy requirements for adaptive analysis, continuous qualification, frame observers, and combined frame analysis. Independent of `detection_enabled`, worker activity and pending-work ownership. |
| `debug.enabled/expires_in_seconds` | Actual debug-lease state at snapshot time, not a request to start diagnostics. |
| `mode`, `illumination_filter_enabled`, sampling settings | Effective reported policy/settings. No assertion of loaded model identity or scene-learning convergence. |
| `refinement` | Existing queue/pending gauges and oldest job age, reused from incident status. Age starts at original job creation; it is not pure queue wait or inference time. |

All durations are **wall time, not CPU time**. Qualification overlaps its stage
durations; analysis cycles overlap inline qualification; parallel stages overlap
each other. Do not sum these as independent CPU costs. Rolling p95/p99 values
cover the most recent 600 samples: do not difference or average camera percentiles.
Preparation/qualification totals retain the existing three-decimal precision;
stage totals expose the accumulator directly, avoiding reconstruction from a
rounded lifetime average.

## Counter lifetimes and collection

Use the root `process.instance_id` and camera ID for every series, plus:

- `motion.metrics_instance_id` for motion-state counters and limiter waits;
- `motion.analysis.metrics_started_monotonic` for preparation/qualification
  telemetry, scoped to the analysis-service object;
- `motion.pipelines.<name>.metrics_instance_id` for each pipeline's stage totals.

These identifiers change on replacement of the corresponding owner.
`camera.lifecycle_generation` and each pipeline's `runtime_generation` identify
behavioral state transitions, but are not counter epochs. Scene resets leave
these cumulative metrics intact. Treat changed generations/settings as comparison
boundaries even when counters continue. Snapshots use existing component locks;
the fleet is not captured atomically. Reject missing values, counter decreases,
or owner changes when calculating interval deltas.

For one unchanged owner, interval average stage latency is
`delta(total_ms) / delta(calls)` when `delta(calls) > 0`. Retain the count,
elapsed capture duration, failures and concurrent process CPU alongside it.
Summaries should rank stages by cumulative elapsed time and call rate, while
keeping CPU attribution explicitly separate.

After deployment, adapt the existing collector to retain these fields rather
than start a second collector. Record actual request start/end, sample gaps,
settings and counter epochs. Start with the existing 15-second application
sampling policy and retain its >250 ms adaptive backoff. The prior capture
scheduled up to 120 seconds and ended with an 84.5-second state-observation gap;
never assume uniform application coverage. Continue independent process CPU
sampling and mark actual camera activity separately from detection results.

Measure payload size and request cost on the live host before interpreting a
new capture. Additional serialization cost and full service-side snapshot cost
are unmeasured; collector self CPU alone does not establish overhead. Do not
change motion policy during the attribution capture. These measurements can
identify expensive operations but do not establish recall, recording continuity,
or the CPU savings of a future optimization.
