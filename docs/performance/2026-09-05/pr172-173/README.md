# Completed PR #172 + #173 performance comparison

**Conclusion: inconclusive for full-pipeline performance.** The candidate's
CPU and elapsed-latency changes did not exceed observed timing variation under
the conservative comparison rule. Some Back Left stage elapsed reductions met
that rule; they are not independently measured stage CPU or fleet savings.

The comparison reused PR #171's saved inputs and byte-identical worker. It ran
three isolated variants, with both candidate changes verified rather than
assumed from a branch name:

| Variant | Commit |
|---|---|
| Before #170 | `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c` |
| With #170 | `ab4d66dd71e2ab98f91696d1134d54e27df21a77` |
| Both #172 and #173 | `8177935464ed3c398cad01082c7dfab47b2d2e05` |

Execution completed on 2026-09-05, 19:12:02–19:14:54 UTC, in 171.708 seconds
within a 900-second budget. Gate, Lower Garage and Back Left each completed
three rounds in ABC, BCA, CAB order: 27 timed runs and nine separate diagnostic
runs. Every run used the same saved frames, timestamps, dependencies, OpenCV
thread settings, camera configuration, initialization and 20-second preroll,
with continuous learning/tracking state thereafter.

All 768 candidate-versus-baseline diagnostic invocation comparisons matched,
including preroll. Timed-run scores and final state matched too. No new footage
was decoded, no streams or inference were started, and no extra unit tests were
run. This establishes equivalence on the tested sequences, not detection recall.

## Complete results

| File | Contents |
|---|---|
| [Report](REPORT.md) | Every timed round, both baseline comparisons, CPU and elapsed variation, morphology/background elapsed measurements and classification guards. |
| [Final Astra review](FINAL-ASTRA-REVIEW.md) | Interpretation, supported stage observations, correctness and fidelity limits. |
| [Numeric results](results.json) | All unrounded timing, comparison, variation and output/state-equivalence results. |
| [Before verification](verification-before.json) | Candidate ancestry/blob checks, original input hashes and service identity. |
| [After verification](verification-after.json) | All 19 original input/harness files and the service/configuration/checkout unchanged. |

Stage durations are wall time, not CPU; overlapping totals are not independent
costs. The candidate contains both changes, so combined effects cannot be
assigned to either PR individually. Main-recording fallback, missing original
live learned state/admission timestamps, competing service load and only three
rounds limit generalization. The report retains unfavorable rounds as well as
favorable ones and marks benefits below variation as inconclusive.

Original evidence remains under
`/root/survng-measurements/pr172-173-20260905T190835Z/`, with reused inputs under
`/root/survng-measurements/pr170-20260905T180430Z/replay/`. Recordings, arrays,
scripts, detailed worker outputs/logs, full camera options and host-specific
paths are intentionally not published. Numeric summaries preserve all result
rows while omitting those details. Verification input keys are relative paths.

This PR only publishes documentation and results. The comparison did not change
the service, settings or production code. GitHub publication was requested
separately after the local measurement completed; no benchmark was rerun for it.
