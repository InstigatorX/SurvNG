> Completed comparison; published after execution at the user's request.
> Statements about no GitHub writes describe the measurement, before this docs PR.
> Raw inputs, scripts, worker outputs and full host-local provenance remain in
> `/root/survng-measurements/pr172-173-20260905T190835Z/` and the original PR #171 capture directory.

# Final Astra review: PR172 + PR173

**Assessment: inconclusive for full-pipeline performance.** The combined candidate matched both baselines’ meaningful outputs and complete learning/tracking state. Its full-pipeline CPU and elapsed changes did not exceed observed timing variation under the declared conservative rule. Some back-left stage elapsed reductions meet that descriptive rule; they do not establish a full-pipeline or fleet CPU saving.

Execution completed at 19:14:54 UTC in 171.708 seconds, within the single 900-second budget. All three cameras completed three rotating rounds (ABC, BCA, CAB), covering 27 timed workers and nine separate diagnostic workers. There were no execution gaps or worker warnings/errors. No extra unit-test workloads were run.

The actual variants were A, pre-PR170 `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c`; B, PR170 `ab4d66dd71e2ab98f91696d1134d54e27df21a77`; and C, the combination of both PR172 and PR173, `8177935464ed3c398cad01082c7dfab47b2d2e05`. Independent parent verification confirmed both PR heads are ancestors of C and its two changed motion-file blobs match the respective PR implementations. This experiment measures the combined candidate and does not attribute performance to either individual PR.

| Camera | C vs A median CPU change | C vs A median elapsed change | C vs B median CPU change | C vs B median elapsed change | Full-pipeline assessment |
|---|---:|---:|---:|---:|---|
| Gate | −0.93% | −0.82% | −5.92% | −7.61% | Inconclusive |
| Lower garage | −6.50% | −0.71% | +1.89% | +1.07% | Inconclusive |
| Back left | −2.97% | −4.93% | −4.28% | −4.35% | Inconclusive |

The rule requires all three round changes to have the same sign and the absolute median paired change to exceed the maximum of: paired-change range; baseline raw-round range divided by its median; and candidate raw-round range divided by its median. The raw-round guard was added while execution was running, before timing results were inspected. Every full-pipeline camera/baseline comparison fails at least one condition. For example, back-left C versus B CPU has a −4.28% median change and a 6.13-percentage-point variability guard. Lower-garage C versus A CPU decreases in all three rounds, but its −6.50% median remains below the 7.97-point guard. Observed ranges are not confidence intervals, and three rounds do not establish statistical significance.

Back-left background elapsed time met the rule versus both A (−5.54% median, 5.37-point guard) and B (−5.68%, 4.74-point guard). Back-left morphology elapsed also met it versus B (−11.66%, 3.95-point guard). These are narrowly scoped elapsed-time observations. The background result versus A clears the guard only slightly. All other morphology/background comparisons are inconclusive, and none of these stage values is an independently measured stage CPU cost.

All 384 candidate diagnostic invocations, including preroll, matched each baseline at every saved meaningful-output and semantic-runtime-state fingerprint: 768 candidate/baseline invocation comparisons. Scoring sequences and final runtime state also matched across all nine timed runs per camera. The checks include background arrays, noise and threshold accumulators, persistent-change age, historical statistics, zone state, blobs, tracks, masks, scoring and decisions. Locks and timing metrics are excluded. The combined source changes alter redundant computation without adding a new learning-state field or changing the replay lifecycle; the exact state equality supports that expectation on these sequences.

The original immutable inputs were reused without decoding: gate 450 frames at 640×480, lower garage 250 at 640×360, and back left 450 at 640×360. All variants used the same cached derivatives, timestamps, saved camera configuration, four-frame windows, due cadence, fresh initialization and identical 20-second preroll, followed by continuous learning/tracking. Measured calls per run were 117, 50 and 116 respectively. The copied worker is byte-for-byte identical to the original and was invoked only in probe, timed and diagnostic modes. It checked the same Python 3.12.3, OpenCV 5.0.0, NumPy 2.4.6, OpenCV thread count 16, optimized/OpenCL settings and affinity 0–15. No OpenCV thread adjustment was made.

The original dataset still contains 0/849 exact-zero morphology-mask observations and 39/849 exact-zero component-mask observations, all 39 in lower garage. These are overlapping-window mask observations, not unique physical frames. The same main-recording fallback limitation remains: codec, field of view and geometry may differ from the original live/substream capture. Original live learned state and admission timing cannot be reconstructed. There was no recall, inference, recording-continuity, UI or fleet-CPU experiment.

Parent [before](verification-before.json) and [after](verification-after.json) verification found all 19 original input/harness file hashes unchanged and all nine array hashes matching the original decode manifests. The service PID 651962, process start, configuration checksum, checkout commit and clean status were unchanged. This comparison performed no service/configuration changes, redecoding, new streams, inference, package installation or GitHub writes.

The [complete report](REPORT.md) contains all 27 timed rows, each variant’s CPU and elapsed latency plus morphology/background stage elapsed values, every candidate-versus-baseline round change, raw min/median/max values, and the final variability guards. [results.json](results.json) retains the unrounded numerical results and equivalence outcomes; per-camera worker outputs preserve detailed evidence. The executed runner and worker were not changed during execution. Final classification used saved-result arithmetic only.
