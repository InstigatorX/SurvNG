> Completed comparison; published after execution at the user's request.
> Statements about no GitHub writes describe the measurement, before this docs PR.
> Raw inputs, scripts, worker outputs and full host-local provenance remain in
> `/root/survng-measurements/pr172-173-20260905T190835Z/` and the original PR #171 capture directory.

# PR172 + PR173 three-variant saved-frame comparison

Assessment: **inconclusive**. Outputs/state matched, but candidate full-pipeline improvement was not consistently larger than observed round variation across cameras and both baselines.

Execution status complete; 171.7s of 900s budget. 2026-09-05T19:12:02.341778+00:00 through 2026-09-05T19:14:54.049158+00:00.

| Variant | Actual commit |
|---|---|
| A_pre170 | `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c` |
| B_pr170 | `ab4d66dd71e2ab98f91696d1134d54e27df21a77` |
| C_pr172_173 | `8177935464ed3c398cad01082c7dfab47b2d2e05` |

The candidate combines both PR172 and PR173. Individual PR effects are not identified. The original PR171 worker is reused byte-for-byte; only probe, timed and diagnostic modes run. No media is decoded again. Original frame/config/timestamp inputs remain read-only.

Each camera uses three rounds in ABC, BCA, CAB order. Fresh pipeline initialization and identical 20-second preroll occur for every run, with continuously advancing learning/tracking thereafter. Detailed state verification is separate from timed passes.

## Every timed round

| Camera | Round | Variant | Analyzed frames | CPU ms/frame | Elapsed ms/frame | Morphology elapsed ms/frame | Background elapsed ms/frame |
|---|---:|---|---:|---:|---:|---:|---:|
| gate | 1 | A_pre170 | 117 | 60.600 | 34.486 | 0.3949 | 16.3289 |
| gate | 1 | B_pr170 | 117 | 61.089 | 34.986 | 0.4029 | 16.6276 |
| gate | 1 | C_pr172_173 | 117 | 61.346 | 34.750 | 0.3811 | 16.1638 |
| gate | 2 | B_pr170 | 117 | 70.285 | 40.387 | 0.5030 | 18.8982 |
| gate | 2 | C_pr172_173 | 117 | 64.925 | 37.307 | 0.4319 | 17.4659 |
| gate | 2 | A_pre170 | 117 | 66.761 | 38.663 | 0.4264 | 18.4221 |
| gate | 3 | C_pr172_173 | 117 | 64.129 | 36.553 | 0.4074 | 17.0766 |
| gate | 3 | A_pre170 | 117 | 64.731 | 36.856 | 0.4186 | 17.1857 |
| gate | 3 | B_pr170 | 117 | 68.164 | 39.564 | 0.4851 | 18.8003 |
| lower-garage | 1 | A_pre170 | 50 | 47.362 | 26.575 | 0.3285 | 14.7229 |
| lower-garage | 1 | B_pr170 | 50 | 43.460 | 25.904 | 0.3361 | 14.4302 |
| lower-garage | 1 | C_pr172_173 | 50 | 44.281 | 26.387 | 0.3276 | 14.7812 |
| lower-garage | 2 | B_pr170 | 50 | 45.250 | 27.015 | 0.3731 | 14.9830 |
| lower-garage | 2 | C_pr172_173 | 50 | 46.134 | 27.304 | 0.3390 | 15.2135 |
| lower-garage | 2 | A_pre170 | 50 | 47.480 | 26.376 | 0.3119 | 14.6689 |
| lower-garage | 3 | C_pr172_173 | 50 | 43.507 | 26.027 | 0.3534 | 14.3526 |
| lower-garage | 3 | A_pre170 | 50 | 48.777 | 27.124 | 0.3136 | 15.0518 |
| lower-garage | 3 | B_pr170 | 50 | 46.893 | 27.896 | 0.3599 | 15.5393 |
| back-left | 1 | A_pre170 | 116 | 56.729 | 30.888 | 0.3347 | 14.9591 |
| back-left | 1 | B_pr170 | 116 | 53.869 | 29.130 | 0.3610 | 14.1443 |
| back-left | 1 | C_pr172_173 | 116 | 53.680 | 28.743 | 0.3250 | 13.8814 |
| back-left | 2 | B_pr170 | 116 | 56.017 | 30.312 | 0.3756 | 14.8197 |
| back-left | 2 | C_pr172_173 | 116 | 53.621 | 28.994 | 0.3318 | 13.9784 |
| back-left | 2 | A_pre170 | 116 | 54.621 | 29.428 | 0.3251 | 14.2395 |
| back-left | 3 | C_pr172_173 | 116 | 53.613 | 28.729 | 0.3214 | 13.8528 |
| back-left | 3 | A_pre170 | 116 | 55.252 | 30.217 | 0.3319 | 14.6651 |
| back-left | 3 | B_pr170 | 116 | 57.301 | 30.841 | 0.3695 | 14.8314 |

## Candidate versus each baseline

For each camera/baseline/metric, require all three paired changes to share a sign and the absolute median paired change to exceed the MAXIMUM of (1) full paired-change percentage-point range, (2) baseline raw-round range divided by its median, expressed as percent, and (3) candidate raw-round range divided by its median, expressed as percent. Otherwise classify inconclusive. The raw-round guard was agreed while execution was running, before inspecting timing results. This descriptive rule is conservative, not a statistical significance test.

Negative changes mean lower cost. The three values correspond to rounds 1–3; ranges describe variation, not confidence intervals.

| Camera | Baseline | Metric | Round changes % | Median change % | Full round spread, pp | Assessment |
|---|---|---|---|---:|---:|---|
| gate | A_pre170 | background_elapsed | -1.01, -5.19, -0.64 | -1.01 | 4.56 | inconclusive |
| gate | A_pre170 | cpu_ms_per_analyzed_frame | +1.23, -2.75, -0.93 | -0.93 | 3.98 | inconclusive |
| gate | A_pre170 | morphology_elapsed | -3.48, +1.30, -2.68 | -2.68 | 4.77 | inconclusive |
| gate | A_pre170 | wall_ms_per_analyzed_frame | +0.77, -3.51, -0.82 | -0.82 | 4.28 | inconclusive |
| gate | B_pr170 | background_elapsed | -2.79, -7.58, -9.17 | -7.58 | 6.38 | inconclusive |
| gate | B_pr170 | cpu_ms_per_analyzed_frame | +0.42, -7.63, -5.92 | -5.92 | 8.05 | inconclusive |
| gate | B_pr170 | morphology_elapsed | -5.40, -14.14, -16.02 | -14.14 | 10.62 | inconclusive |
| gate | B_pr170 | wall_ms_per_analyzed_frame | -0.67, -7.63, -7.61 | -7.61 | 6.95 | inconclusive |
| lower-garage | A_pre170 | background_elapsed | +0.40, +3.71, -4.65 | +0.40 | 8.36 | inconclusive |
| lower-garage | A_pre170 | cpu_ms_per_analyzed_frame | -6.50, -2.83, -10.80 | -6.50 | 7.97 | inconclusive |
| lower-garage | A_pre170 | morphology_elapsed | -0.26, +8.68, +12.69 | +8.68 | 12.95 | inconclusive |
| lower-garage | A_pre170 | wall_ms_per_analyzed_frame | -0.71, +3.52, -4.04 | -0.71 | 7.56 | inconclusive |
| lower-garage | B_pr170 | background_elapsed | +2.43, +1.54, -7.64 | +1.54 | 10.07 | inconclusive |
| lower-garage | B_pr170 | cpu_ms_per_analyzed_frame | +1.89, +1.96, -7.22 | +1.89 | 9.18 | inconclusive |
| lower-garage | B_pr170 | morphology_elapsed | -2.53, -9.13, -1.83 | -2.53 | 7.30 | inconclusive |
| lower-garage | B_pr170 | wall_ms_per_analyzed_frame | +1.86, +1.07, -6.70 | +1.07 | 8.56 | inconclusive |
| back-left | A_pre170 | background_elapsed | -7.20, -1.83, -5.54 | -5.54 | 5.37 | supported decrease |
| back-left | A_pre170 | cpu_ms_per_analyzed_frame | -5.37, -1.83, -2.97 | -2.97 | 3.54 | inconclusive |
| back-left | A_pre170 | morphology_elapsed | -2.89, +2.08, -3.17 | -2.89 | 5.25 | inconclusive |
| back-left | A_pre170 | wall_ms_per_analyzed_frame | -6.95, -1.48, -4.93 | -4.93 | 5.47 | inconclusive |
| back-left | B_pr170 | background_elapsed | -1.86, -5.68, -6.60 | -5.68 | 4.74 | supported decrease |
| back-left | B_pr170 | cpu_ms_per_analyzed_frame | -0.35, -4.28, -6.43 | -4.28 | 6.08 | inconclusive |
| back-left | B_pr170 | morphology_elapsed | -9.98, -11.66, -13.01 | -11.66 | 3.03 | supported decrease |
| back-left | B_pr170 | wall_ms_per_analyzed_frame | -1.33, -4.35, -6.85 | -4.35 | 5.52 | inconclusive |

## Correctness and coverage

gate: complete; timed score/final-state equality True. Diagnostic comparisons: `{"A_pre170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}, "B_pr170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}}`.

Exact measured masks: morphology `{'exact_zero': 0, 'nonempty': 351, 'total': 351}`; components `{'exact_zero': 0, 'nonempty': 351, 'total': 351}`.

lower-garage: complete; timed score/final-state equality True. Diagnostic comparisons: `{"A_pre170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 84, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}, "B_pr170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 84, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}}`.

Exact measured masks: morphology `{'exact_zero': 0, 'nonempty': 150, 'total': 150}`; components `{'exact_zero': 39, 'nonempty': 111, 'total': 150}`.

back-left: complete; timed score/final-state equality True. Diagnostic comparisons: `{"A_pre170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}, "B_pr170": {"all_output_and_runtime_fingerprints_equal": true, "compared_pipeline_invocations": 150, "final_runtime_equal": true, "precomponents_mask_counts_equal": true, "premorph_mask_counts_equal": true, "score_sequence_equal": true}}`.

Exact measured masks: morphology `{'exact_zero': 0, 'nonempty': 348, 'total': 348}`; components `{'exact_zero': 0, 'nonempty': 348, 'total': 348}`.

Existing candidate regression tests: "No additional test workload; requested timed and full-state comparisons only."

## Limits

- Same saved main-recording frames as PR171: gate 640x480, lower-garage/back-left 640x360. Main-stream codec/FOV/geometry may differ from original live analysis.
- Identical saved settings, four-frame windows, due cadence, initialization and 20-second preroll; original live learned state and admission/capture timing are not reconstructed.
- CPU and wall metrics cover full pipeline.process calls, excluding already-cached preparation, imports, array loading, hash verification and production bookkeeping.
- Stage measurements are elapsed wall time, not stage CPU. Do not sum them with overlapping qualification/worker totals.
- Live service remains running and competes for resources. Rotating order balances position; three rounds do not establish statistical significance.
- Candidate includes BOTH PR172 and PR173. Comparison cannot attribute combined effects to either individual PR.
- These results do not measure recall, event admission, inference, recording continuity, UI or fleet CPU savings.

Complete numeric results: `results.json`. Per-camera timed/diagnostic worker outputs and logs, exact specs, unchanged worker, extracted source snapshots and environment probe remain in the host-local measurement directory, not this documentation PR. Published `verification-before.json` / `verification-after.json` establish input/service preservation separately.

## Raw timing variation and final guard

Each cell below is min / median / max across the same three timed rounds, in ms per analyzed frame. These are observed ranges, not confidence intervals.

| Camera | Variant | CPU min/median/max | Elapsed min/median/max | Morphology elapsed min/median/max | Background elapsed min/median/max |
|---|---|---|---|---|---|
| gate | A_pre170 | 60.5997 / 64.7310 / 66.7611 | 34.4856 / 36.8564 / 38.6629 | 0.3949 / 0.4186 / 0.4264 | 16.3289 / 17.1857 / 18.4221 |
| gate | B_pr170 | 61.0893 / 68.1643 / 70.2852 | 34.9863 / 39.5635 / 40.3871 | 0.4029 / 0.4851 / 0.5030 | 16.6276 / 18.8003 / 18.8982 |
| gate | C_pr172_173 | 61.3460 / 64.1285 / 64.9252 | 34.7505 / 36.5532 / 37.3066 | 0.3811 / 0.4074 / 0.4319 | 16.1638 / 17.0766 / 17.4659 |
| lower-garage | A_pre170 | 47.3620 / 47.4804 / 48.7770 | 26.3762 / 26.5749 / 27.1236 | 0.3119 / 0.3136 / 0.3285 | 14.6689 / 14.7229 / 15.0518 |
| lower-garage | B_pr170 | 43.4604 / 45.2498 / 46.8933 | 25.9041 / 27.0154 / 27.8956 | 0.3361 / 0.3599 / 0.3731 | 14.4302 / 14.9830 / 15.5393 |
| lower-garage | C_pr172_173 | 43.5069 / 44.2812 / 46.1345 | 26.0274 / 26.3872 / 27.3044 | 0.3276 / 0.3390 / 0.3534 | 14.3526 / 14.7812 / 15.2135 |
| back-left | A_pre170 | 54.6212 / 55.2518 / 56.7290 | 29.4279 / 30.2174 / 30.8883 | 0.3251 / 0.3319 / 0.3347 | 14.2395 / 14.6651 / 14.9591 |
| back-left | B_pr170 | 53.8695 / 56.0170 / 57.3005 | 29.1296 / 30.3123 / 30.8405 | 0.3610 / 0.3695 / 0.3756 | 14.1443 / 14.8197 / 14.8314 |
| back-left | C_pr172_173 | 53.6134 / 53.6211 / 53.6805 | 28.7285 / 28.7425 / 28.9936 | 0.3214 / 0.3250 / 0.3318 | 13.8528 / 13.8814 / 13.9784 |

The earlier comparison table reports the paired-change range; the following table shows the larger effective guard actually used, including both variants’ raw timing variation.

| Camera | Baseline | Metric | Absolute median paired change % | Effective variation guard, pp | Final assessment |
|---|---|---|---:|---:|---|
| gate | A_pre170 | background_elapsed | 1.01 | 12.18 | inconclusive |
| gate | A_pre170 | cpu_ms_per_analyzed_frame | 0.93 | 9.52 | inconclusive |
| gate | A_pre170 | morphology_elapsed | 2.68 | 12.46 | inconclusive |
| gate | A_pre170 | wall_ms_per_analyzed_frame | 0.82 | 11.33 | inconclusive |
| gate | B_pr170 | background_elapsed | 7.58 | 12.08 | inconclusive |
| gate | B_pr170 | cpu_ms_per_analyzed_frame | 5.92 | 13.49 | inconclusive |
| gate | B_pr170 | morphology_elapsed | 14.14 | 20.64 | inconclusive |
| gate | B_pr170 | wall_ms_per_analyzed_frame | 7.61 | 13.65 | inconclusive |
| lower-garage | A_pre170 | background_elapsed | 0.40 | 8.36 | inconclusive |
| lower-garage | A_pre170 | cpu_ms_per_analyzed_frame | 6.50 | 7.97 | inconclusive |
| lower-garage | A_pre170 | morphology_elapsed | 8.68 | 12.95 | inconclusive |
| lower-garage | A_pre170 | wall_ms_per_analyzed_frame | 0.71 | 7.56 | inconclusive |
| lower-garage | B_pr170 | background_elapsed | 1.54 | 10.07 | inconclusive |
| lower-garage | B_pr170 | cpu_ms_per_analyzed_frame | 1.89 | 9.18 | inconclusive |
| lower-garage | B_pr170 | morphology_elapsed | 2.53 | 10.28 | inconclusive |
| lower-garage | B_pr170 | wall_ms_per_analyzed_frame | 1.07 | 8.56 | inconclusive |
| back-left | A_pre170 | background_elapsed | 5.54 | 5.37 | supported decrease |
| back-left | A_pre170 | cpu_ms_per_analyzed_frame | 2.97 | 3.81 | inconclusive |
| back-left | A_pre170 | morphology_elapsed | 2.89 | 5.25 | inconclusive |
| back-left | A_pre170 | wall_ms_per_analyzed_frame | 4.93 | 5.47 | inconclusive |
| back-left | B_pr170 | background_elapsed | 5.68 | 4.74 | supported decrease |
| back-left | B_pr170 | cpu_ms_per_analyzed_frame | 4.28 | 6.13 | inconclusive |
| back-left | B_pr170 | morphology_elapsed | 11.66 | 3.95 | supported decrease |
| back-left | B_pr170 | wall_ms_per_analyzed_frame | 4.35 | 5.64 | inconclusive |

Final assessment: **inconclusive**. Outputs/state matched, but candidate full-pipeline improvement was not consistently larger than observed round variation across cameras and both baselines.

Final architectural review is saved in `FINAL-ASTRA-REVIEW.md`. Parent before/after verification reports separately establish that original evidence, service and settings were preserved.
