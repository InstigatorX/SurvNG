> Completed measurement of PR #170, published after execution.
> Raw captures, recordings, decoded arrays, worker logs and full host-local provenance
> remain under `/root/survng-measurements/pr170-20260905T180430Z/`.
> Statements about no GitHub writes refer to the measurement, before this documentation PR.

# PR #170 unattended performance measurement

Updated 2026-09-05T18:31:47.988806+00:00. Live capture complete.

Steady-state service CPU: 3.988 cores over 1199.1 seconds (1199 intervals). Startup excluded: 0.0 seconds; unknown age/owner: 0.0 seconds.

Resource samples: 1200; application snapshots: 14; errors: 0. Actual sampling, ownership boundaries, health and per-camera settings are in live-summary.json.

## Process CPU (one core = 1.0)

| Group | Cores |
|---|---:|
| main | 2.5427 |
| other_children | 0.0001 |
| survng-object | 0.0190 |
| survng-face | 0.0004 |
| survng-reid | 0.0009 |
| survng-recorder | 0.0784 |
| survng-semantic | 0.0100 |
| survng-capture | 0.5708 |
| ffmpeg | 0.5063 |
| Unattributed cgroup residual | 0.2597 |

The cgroup total includes its children; process groups partition observed CPU and are not added to that total. Short-lived processes can remain in the residual. RSS/cgroup memory are distinct.

## Stage wall timings

| Camera | Pipeline/stage | Calls | Failures | Interval mean ms |
|---|---|---:|---:|---:|
| front-door | qualification/background | 2194 | 0 | 26.552 |
| gate | qualification/background | 2180 | 0 | 25.706 |
| sherry-garage | qualification/background | 2218 | 0 | 22.799 |
| boiler | qualification/background | 2108 | 0 | 23.176 |
| downstairs | qualification/background | 1863 | 0 | 25.480 |
| steve-garage | qualification/background | 2053 | 0 | 23.055 |
| front-side | qualification/background | 2047 | 0 | 22.939 |
| back-left | qualification/background | 2072 | 0 | 22.379 |
| back-right | qualification/background | 1980 | 0 | 23.326 |
| foyer | qualification/background | 1828 | 0 | 24.730 |
| back-middle | qualification/background | 1766 | 0 | 23.668 |
| lower-garage | qualification/background | 1831 | 0 | 22.322 |
| upper-garage | qualification/background | 1770 | 0 | 19.135 |
| gate | qualification/threshold | 2180 | 0 | 12.101 |
| front-door | qualification/threshold | 2194 | 0 | 12.000 |
| front-side | qualification/threshold | 2047 | 0 | 12.400 |
| back-left | qualification/threshold | 2072 | 0 | 11.803 |
| steve-garage | qualification/threshold | 2053 | 0 | 11.638 |
| back-right | qualification/threshold | 1980 | 0 | 10.706 |
| sherry-garage | qualification/threshold | 2218 | 0 | 9.291 |
| downstairs | qualification/threshold | 1863 | 0 | 10.517 |
| foyer | qualification/threshold | 1828 | 0 | 10.466 |
| boiler | qualification/threshold | 2108 | 0 | 8.871 |
| lower-garage | qualification/threshold | 1831 | 0 | 9.594 |
| upper-garage | qualification/threshold | 1770 | 0 | 8.749 |
| back-middle | qualification/threshold | 1766 | 0 | 8.515 |
| gate | qualification/blob_extract | 2180 | 0 | 3.916 |
| front-door | qualification/blob_extract | 2194 | 0 | 3.386 |
| back-left | qualification/blob_extract | 2072 | 0 | 2.948 |
| sherry-garage | qualification/blob_extract | 2218 | 0 | 2.558 |

Ranked by cumulative elapsed time; these are not CPU shares. Full per-camera stage rows, accepted coverage and counter rejection reasons are in live-summary.json.

## Replay and final interpretation

## Operational observations

Detector counter deltas: {'total_inferences': 403.0, 'failed_inferences': 0.0, 'recorded_decode_process_timeouts': 0.0, 'recorded_decode_memory_timeouts': 0.0}. Sampled inference queue: {'n': 14, 'min': 0, 'median': 0.0, 'max': 0}; recorded-decode waiting: {'n': 14, 'min': 0, 'median': 0.0, 'max': 1}.
Application request cost (ms): {'n': 14, 'min': 60.91528292745352, 'median': 81.48837101180106, 'max': 1143.628529040143}; actual intervals (seconds): {'n': 13, 'min': 30.407480571069755, 'median': 120.0969138899818, 'max': 120.17347005999181}; final application tail: 26.4 seconds.

| Camera | Disconnected samples | Recording expected but false | Deferrals | Mailbox replacements | Max refinement age ms |
|---|---:|---:|---:|---:|---:|
| back-left | 0 | 0 | 525.0 | 3.0 | 7146.505 |
| back-middle | 0 | 0 | 482.0 | 2.0 | 0.0 |
| back-right | 0 | 0 | 521.0 | 3.0 | 32261.482 |
| boiler | 0 | 0 | 561.0 | 6.0 | 0.0 |
| downstairs | 0 | 0 | 383.0 | 3.0 | 0.0 |
| foyer | 0 | 0 | 392.0 | 4.0 | 0.0 |
| front-door | 0 | 0 | 604.0 | 8.0 | 3562.641 |
| front-side | 0 | 0 | 489.0 | 5.0 | 21733.257 |
| gate | 0 | 0 | 521.0 | 7.0 | 67337.137 |
| lower-garage | 0 | 0 | 483.0 | 1.0 | 68430.144 |
| sherry-garage | 0 | 0 | 530.0 | 6.0 | 0.0 |
| steve-garage | 0 | 0 | 508.0 | 4.0 | 0.0 |
| upper-garage | 0 | 0 | 497.0 | 0.0 | 13670.07 |

No disconnected/recording-false samples is not proof of gap-free decodable recordings. Absolute timeout counters at capture start are not new failures; reported counter deltas exclude that history.

## Same-footage comparison

# Isolated same-footage PR170 qualification comparison

Status: complete. Duration: 183.9 seconds; global budget 900 seconds.

Before `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c`; after `ab4d66dd71e2ab98f91696d1134d54e27df21a77`.

Replay protocol: identical decoded frames, timestamps, cached derivatives, camera configuration, fresh initialization, 20-second preroll, and continuous learning/tracking. Timed runs use alternating before/after, after/before, before/after order. Exact-mask and detailed state comparisons are separate untimed runs. Completed runs and any gaps are identified below.

| Camera | Analyzed frames/run | CPU before/after ms/frame (3 pairs) | Median CPU change | Median elapsed change | Exact zero premorph masks | State/output match |
|---|---:|---|---:|---:|---:|---|
| gate | 117 | 68.260/78.162; 78.937/76.316; 76.081/72.500 | -3.32% | -2.18% | 0/351 | True |
| lower-garage | 50 | 51.962/51.347; 46.059/44.859; 48.764/47.194 | -2.61% | +3.40% | 0/150 | True |
| back-left | 116 | 56.133/53.523; 56.179/53.072; 52.399/55.328 | -4.65% | -4.26% | 0/348 | True |

Negative change means lower cost. Three pairs describe observed variation; they do not establish statistical significance or a fleet-wide CPU saving.

## gate

Recorded source `main`, 2026-09-05T18:09:44+00:00 through 2026-09-05T18:11:14+00:00; status complete.

premorph: 0 exact-zero and 351 nonzero mask observations; invocations all-empty 0, all-nonempty 117, mixed 0. Counts include repeated observations in overlapping windows, not unique physical frames.
precomponents: 0 exact-zero and 351 nonzero mask observations; invocations all-empty 0, all-nonempty 117, mixed 0. Counts include repeated observations in overlapping windows, not unique physical frames.

all_empty: no observed invocations; no timing or saving inferred.
all_nonempty (117 analyzed frames/run): before/after ms/frame for pairs 1–3: 68.260/78.162 CPU, 39.249/46.437 elapsed; 78.937/76.316 CPU, 46.130/45.122 elapsed; 76.081/72.500 CPU, 44.217/42.135 elapsed.
mixed_history: no observed invocations; no timing or saving inferred.
observed_mixture (117 analyzed frames/run): before/after ms/frame for pairs 1–3: 68.260/78.162 CPU, 39.249/46.437 elapsed; 78.937/76.316 CPU, 46.130/45.122 elapsed; 76.081/72.500 CPU, 44.217/42.135 elapsed.

Pair 1 (before/after): elapsed before/after 39.249/46.437 ms/frame; CPU change +14.51%, elapsed change +18.31%.
Pair 2 (after/before): elapsed before/after 46.130/45.122 ms/frame; CPU change -3.32%, elapsed change -2.18%.
Pair 3 (before/after): elapsed before/after 44.217/42.135 ms/frame; CPU change -4.71%, elapsed change -4.71%.

Equivalence: {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}

## lower-garage

Recorded source `main`, 2026-09-05T18:09:24+00:00 through 2026-09-05T18:10:14+00:00; status complete.

premorph: 0 exact-zero and 150 nonzero mask observations; invocations all-empty 0, all-nonempty 50, mixed 0. Counts include repeated observations in overlapping windows, not unique physical frames.
precomponents: 39 exact-zero and 111 nonzero mask observations; invocations all-empty 9, all-nonempty 33, mixed 8. Counts include repeated observations in overlapping windows, not unique physical frames.

all_empty: no observed invocations; no timing or saving inferred.
all_nonempty (50 analyzed frames/run): before/after ms/frame for pairs 1–3: 51.962/51.347 CPU, 29.425/30.424 elapsed; 46.059/44.859 CPU, 25.817/26.635 elapsed; 48.764/47.194 CPU, 27.028/28.205 elapsed.
mixed_history: no observed invocations; no timing or saving inferred.
observed_mixture (50 analyzed frames/run): before/after ms/frame for pairs 1–3: 51.962/51.347 CPU, 29.425/30.424 elapsed; 46.059/44.859 CPU, 25.817/26.635 elapsed; 48.764/47.194 CPU, 27.028/28.205 elapsed.

Pair 1 (before/after): elapsed before/after 29.425/30.424 ms/frame; CPU change -1.18%, elapsed change +3.40%.
Pair 2 (after/before): elapsed before/after 25.817/26.635 ms/frame; CPU change -2.61%, elapsed change +3.17%.
Pair 3 (before/after): elapsed before/after 27.028/28.205 ms/frame; CPU change -3.22%, elapsed change +4.35%.

Equivalence: {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 84, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}

## back-left

Recorded source `main`, 2026-09-05T18:09:32+00:00 through 2026-09-05T18:11:02+00:00; status complete.

premorph: 0 exact-zero and 348 nonzero mask observations; invocations all-empty 0, all-nonempty 116, mixed 0. Counts include repeated observations in overlapping windows, not unique physical frames.
precomponents: 0 exact-zero and 348 nonzero mask observations; invocations all-empty 0, all-nonempty 116, mixed 0. Counts include repeated observations in overlapping windows, not unique physical frames.

all_empty: no observed invocations; no timing or saving inferred.
all_nonempty (116 analyzed frames/run): before/after ms/frame for pairs 1–3: 56.133/53.523 CPU, 30.316/29.026 elapsed; 56.179/53.072 CPU, 30.326/28.888 elapsed; 52.399/55.328 CPU, 28.548/29.658 elapsed.
mixed_history: no observed invocations; no timing or saving inferred.
observed_mixture (116 analyzed frames/run): before/after ms/frame for pairs 1–3: 56.133/53.523 CPU, 30.316/29.026 elapsed; 56.179/53.072 CPU, 30.326/28.888 elapsed; 52.399/55.328 CPU, 28.548/29.658 elapsed.

Pair 1 (before/after): elapsed before/after 30.316/29.026 ms/frame; CPU change -4.65%, elapsed change -4.26%.
Pair 2 (after/before): elapsed before/after 30.326/28.888 ms/frame; CPU change -5.53%, elapsed change -4.74%.
Pair 3 (before/after): elapsed before/after 28.548/29.658 ms/frame; CPU change +5.59%, elapsed change +3.89%.

Equivalence: {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}

## Validation and limits

passed: existing tests/test_motion_empty_masks.py

- All selected clips used validated MAIN-recording fallback because no qualifying continuous validated live-source span was available. Main-stream codec, field of view and resize geometry may differ from the original live/substream capture. Indexed start epochs plus decoder PTS are not original capture timestamps.
- Live mailbox replacements, frame drops, analysis admission, ONVIF triggers, inference, event dispatch, recording continuity and UI are outside this isolated qualification replay.
- Runtime background/tracker state cannot be exported safely. Both versions start fresh and receive identical 20-second preroll; this does not establish convergence to live scene state.
- Socket verifies available scalar settings and stage identity. Sensitivity, stationary tolerance, stage options and zone geometry come from the unchanged persisted configuration and are not fully runtime-attested.
- No in-process OpenCV thread query is available. Replay uses the service interpreter/dependencies, affinity and allowlisted environment; absent runtime setNumThreads calls plus clean source support, but do not prove, matching live CV threads.
- CPU and elapsed latency measure full qualification pipeline.process calls only; decoding, cached resize/gray/blur, imports, output hashing, inference and production bookkeeping are excluded.
- The live service continues competing for host resources during replay; alternating pairs reduce order bias but are not an unloaded laboratory benchmark.
- Loaded deployed source is inferred from clean checkout and pre-start reflog, not a runtime build identifier.

Gap: {"camera": "front-door", "selection_gap": "No >=45-second indexed continuous clip within capture"}

Gap: {"camera": "sherry-garage", "selection_gap": "No >=45-second indexed continuous clip within capture"}

Gap: {"camera": "back-right", "selection_gap": "No >=45-second indexed continuous clip within capture"}

Artifacts: `replay-results.json`, `selection.json`, `source-provenance.json`, `source-diff.patch`, `environment.json`, and per-camera decode/timed/diagnostic JSON and logs. No production mutation, extra stream, inference process, deployment, or package installation was performed.

## Final Astra interpretation

Recommendation: **inconclusive for performance benefit**. The replay supports correctness on the tested sequences. It does not establish a bundled or fleet CPU saving, or show that a component-only variant is better.

Morphology saw 0/849 exact-zero mask observations and its stage elapsed time increased in all nine pairs. Component extraction saw 39/849 exact-zero observations, all in lower-garage (26% there); its stage elapsed time decreased in all three lower-garage pairs. These are stage wall measurements, not isolated stage CPU measurements.

Gate and back-left CPU changes changed sign across pairs. Lower-garage CPU decreased 1.18–3.22% while elapsed latency increased 3.17–4.35%. The variation and main-recording fallback limit attribution. No morphology-empty or mixed morphology-input timing is available; no saving is inferred for those absent cases.

All 384 diagnostic invocation pairs matched complete meaningful outputs and semantic runtime state. Seven existing focused tests passed. No recall or recording-continuity conclusion follows from this comparison.

The following classification uses exact component-input masks from the separate diagnostic pass to group the already saved full-pipeline timings. It adds no new replay or timing workload. Earlier all_empty/all_nonempty/mixed_history tables classify morphology inputs.

| Camera | Component-input regime | Analyzed frames/run | Full-pipeline CPU before/after ms/frame, pairs 1–3 | Full-pipeline elapsed before/after ms/frame, pairs 1–3 |
|---|---|---:|---|---|
| gate | all_empty | 0 | unavailable | unavailable |
| gate | all_nonempty | 117 | 68.260/78.162; 78.937/76.316; 76.081/72.500 | 39.249/46.437; 46.130/45.122; 44.217/42.135 |
| gate | mixed_history | 0 | unavailable | unavailable |
| gate | observed_mixture | 117 | 68.260/78.162; 78.937/76.316; 76.081/72.500 | 39.249/46.437; 46.130/45.122; 44.217/42.135 |
| lower-garage | all_empty | 9 | 45.892/32.834; 41.277/30.756; 45.605/32.389 | 27.518/27.142; 24.893/25.765; 26.602/26.462 |
| lower-garage | all_nonempty | 33 | 54.677/58.706; 48.356/50.239; 49.565/53.106 | 30.481/32.085; 26.417/27.311; 26.937/29.152 |
| lower-garage | mixed_history | 8 | 47.587/41.819; 41.966/38.532; 49.014/39.462 | 27.216/27.269; 24.385/24.828; 27.887/26.259 |
| lower-garage | observed_mixture | 50 | 51.962/51.347; 46.059/44.859; 48.764/47.194 | 29.425/30.424; 25.817/26.635; 27.028/28.205 |
| back-left | all_empty | 0 | unavailable | unavailable |
| back-left | all_nonempty | 116 | 56.133/53.523; 56.179/53.072; 52.399/55.328 | 30.316/29.026; 30.326/28.888; 28.548/29.658 |
| back-left | mixed_history | 0 | unavailable | unavailable |
| back-left | observed_mixture | 116 | 56.133/53.523; 56.179/53.072; 52.399/55.328 | 30.316/29.026; 30.326/28.888; 28.548/29.658 |

The nine all-empty and eight mixed component-input invocations in lower-garage are small observed subsets, not independent synthetic benchmarks. Exact mask groups include repeated historical masks within four-frame windows.

Full final integration review: [Final Astra review](FINAL-ASTRA-REVIEW.md).


## Limits

- Stage elapsed totals are wall time, not CPU; overlapping stage/qualification/worker totals must not be summed.
- Sampling cannot observe short failures/queues between snapshots. Missing values and owner changes are not zero.
- Unlabelled recordings do not establish recall or actual quiet scenes.
- Loaded code inferred from clean checkout plus pre-start reflog; runtime exposes no build identifier.
