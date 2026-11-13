# GPU replay and live restart validation

Follow-up to [the Phase 0–1 checkpoint](inference-serving-phase0-1.md) and
[PR #309](https://github.com/InstigatorX/SurvNG/pull/309), performed with explicit
authorization on 2026-11-13. The service now runs the PR's runtime revision
`33e1639`. This validation changes no runtime code or configuration.

## Procedure

1. Observe the existing `f8291c9` service through its owner-only socket every
   15 seconds for 10 minutes. Save 41 snapshots and recording-index evidence.
2. Decode and privately retain eight fixed incident snapshots. Run the same
   object-only replay against a separate baseline checkout and the PR checkout,
   using the same frozen detector settings, fixture hashes and thresholds.
3. Restart `survng.service` with the clean PR checkout. Verify new process
   identity, retirement of old cgroup processes, camera connections, recording,
   tracking settings and ONVIF connection states.
4. After full readiness, observe another 10 minutes and repeat the replay at
   approximately the same point in the observation window.
5. Probe finalized MP4 files before and after restart and estimate restart gaps
   from adjacent indexed segments. Inspect the initial-latency tail in an
   additional alternating baseline/candidate experiment when it crosses 5%.

The fixed replay used two OpenVINO object processes on Intel Iris Xe integrated
GPU hardware, the current FP32 NCHW 960×960 model with final YOLO detections,
cache disabled and warmup enabled. Auxiliary roles were disabled only in the
standalone benchmark. The production face/ReID processes remained configured as
before; both production ReID models reported ready on GPU after restart.

There were five repeats of 100 requests for each of tracking, initial incidents,
refinement and mixed work on each revision. Four caller threads offered bounded
waves at at most 20 requests/s. Thresholds were default, .2, .7, .01 and .99.
The source inputs span 2560×1440 to 4512×2512 and exercise parent reduction,
letterboxing and coordinate restoration. The standalone replay creates no
production incidents or recordings and does not submit images to an HTTP API.

## Correctness and latency

Both principal replays completed all 2,000 requests, without deferrals or
unavailable results. All 2,000 candidate results matched their baseline case
exactly, including 1,160 returned detection objects across repeated requests.
All 40 unique image/threshold cases were deterministic in the baseline.

| Workload | Baseline p95 ms | PR p95 ms | Baseline p99 ms | PR p99 ms |
| --- | --- | --- | --- | --- |
| Tracking | 115.142 | 118.738 | 130.880 | 131.628 |
| Initial | 127.679 | 131.160 | 133.890 | 142.245 |
| Refinement | 126.875 | 127.214 | 134.699 | 138.781 |
| Mixed | 127.688 | 127.004 | 138.745 | 135.012 |

Initial p99 was 6.24% higher in the first comparison, triggering a repeatability
check rather than a passing performance claim. Five alternating pairs then ran
100 initial requests per revision, reversing which revision went first on every
other pair. Both revisions ran against the same post-restart production context.

| Alternating initial check | Baseline | PR | Change |
| --- | --- | --- | --- |
| Completed requests | 500 | 500 | — |
| p95 ms | 115.102 | 115.120 | +0.016% |
| p99 ms | 129.027 | 125.756 | −2.535% |
| Detection mismatches | 0 | 0 | — |

The original >5% tail increase did not reproduce in this follow-up. Across both
experiments, all 5,000 requests completed and all 2,500 candidate results matched
the expected detections. This supports correctness and shows no repeatable
initial-latency regression in this limited workload. It establishes no speedup
or saturated-capacity improvement. Cold startup differed substantially between
experiments; runtime/driver cache state was uncontrolled, so startup durations
must not be interpreted as an ownership-refactor benefit.

## Live operation and restart

41 snapshots per window, sampled every 15 seconds. Natural traffic was quiet:
neither window had admitted initial or tracking requests. The figures below
therefore complement the controlled replay rather than replace it.

| Observation | Before | After |
| --- | --- | --- |
| Live inferences during window | 188 | 126 |
| Inference failures | 0 | 0 |
| Socket errors | 0 | 0 |
| Connected cameras at every sample | 13 | 13 |
| Configured recorders active at every sample | 9 | 9 |
| Warnings/errors in sampled socket recent logs | 0 | 0 |

The service restart was requested at 14:32:22 EST. Systemd returned after 6.43 s;
all 13 camera connections, nine configured recorders and the detector were ready
67.60 s after the request. None of the old service cgroup processes remained.
Tracking settings and per-camera ONVIF connection states matched the baseline.

The recording index estimates per-camera restart gaps of 9–23 s across nine
recorders. These are real restart interruptions, not continuous recording through
the restart. Nine finalized pre-restart and nine finalized post-restart MP4 files
were readable by FFprobe without reported errors, with H.264/HEVC video and AAC
audio. Header/container probing does not certify every decoded video frame.

## Memory and remaining limits

Production object RSS was 154.7/156.7 MiB in the old long-running processes and
193.0/209.2 MiB after restart. Each new worker's RSS varied by less than 0.5 MiB
throughout the post-restart observation. These observations do not explain the higher fresh
process RSS or establish a memory saving. The alternating experiment with fresh
workers on both revisions produced broadly comparable worker RSS near 237 MiB
and 102.6 MiB GPU residency per worker. Driver/compiler allocation and process
age are confounders; sustained memory qualification remains open.

Live traffic in these windows was mainly optional enrichment, with insufficient
natural initial/tracking traffic to establish production incident tail latency.
The controlled replay supplies inference-workload coverage, not a complete
capture-to-incident latency test. Live GPU resource counters and model readiness
complement the replay; the socket's device field is configured device, whereas
the replay explicitly verifies the worker's loaded GPU backend and rejects
fallback. Additional alternating replay load occurred during the second live
window, so aggregate live timings are contextual observations, not a matched
performance experiment.

Other GPU models/output contracts, Core ML hardware, 30-minute saturation,
peak-memory checks and repeated lifecycle performance cycles remain unverified.
Private images, detector configuration, raw detections, camera names and recording
paths are retained only in owner-private local artifacts. Generated measurement
dumps, hashes and latency samples are archived locally outside version control;
this report retains the measured results, procedure and limitations.

## Later normal-load observation

At 16:57 EST, about 2 hours 25 minutes after restart, the same service process
remained active with no automatic restarts. All 13 cameras were connected and
all nine configured recorders were active. The snapshot reported 4,518
inferences with zero inference failures, 28.7 ms average inference, and no
pending requests. All 95 initial and 956 refinement admissions had completed
without admission timeouts. Object RSS had decreased to 173.5/182.0 MiB, with
GPU residency about 102.6 MiB per worker; this is ordinary-load evidence, not
a sustained saturation or peak-memory test.

Later logs included an optional face-refinement error and one camera's
H.264/timestamp recovery error burst at 15:54–15:58 EST. That camera's latest
main/live MP4 outputs were fresh and readable by FFprobe. Their cause was not
investigated in this ownership PR; no connection to the refactor was established.
