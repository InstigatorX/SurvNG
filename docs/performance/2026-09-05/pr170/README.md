# PR #170: completed unattended performance measurement

The performance benefit is **inconclusive**. Tested outputs and learning/tracking
state match, but these recordings do not exercise the morphology empty-mask
shortcut. Component-only performance was not independently measured.

- Live capture: 2026-09-05 18:05:08–18:25:09 UTC, 20 minutes, no startup exclusion
  needed. Service CPU averaged 3.988 cores over 1199.1 measured seconds.
- Replay: 18:25:14–18:28:18 UTC, 183.9 seconds within a 900-second budget.
  Three cameras, three alternating before/after pairs each.
- Exact-zero opportunities: 0/849 before morphology; 39/849 before components.
  These are mask observations in overlapping windows, not unique frames.
- All 384 diagnostic output/state comparisons matched, including preroll.
  Seven existing focused regression tests passed.
- All replay clips used main recordings because suitable continuous validated
  live-source recordings were unavailable. This limits live-stream fidelity.

## Results and method

| File | Contents |
|---|---|
| [Final Astra review](FINAL-ASTRA-REVIEW.md) | Recommendation, confidence, regressions, equivalence and limitations. |
| [Complete report](REPORT.md) | Live CPU/health, stage timings, all paired replay results, and both morphology-input and component-input workload groups. |
| [Method review](METHOD-REVIEW.md) | Isolation, sampling, state preservation, ownership, recording selection and budget rules. |
| [Live numeric summary](live-summary.json) | All per-camera stage rows and settings, counter deltas/rejections, resource coverage and sampled health; not just the report's top-stage table. |
| [Replay numeric results](replay-results.json) | All paired/regime timings, exact-mask counts, output/state comparisons, environment versions and execution results. |

Actual pre-change source is `6e2540800b51d3ae4f172f69cdde58bcd4a1a68c`;
deployed checkout is `ab4d66dd71e2ab98f91696d1134d54e27df21a77`. Their source
delta is the two optimization files and focused tests. Loaded source is inferred
from the clean checkout and pre-start deployment history, not a runtime build ID.

Stage elapsed time is wall time, not CPU. Overlapping stages, qualification and
worker totals must not be added as separate costs. CPU differences in earlier
unmatched captures are not attributed to this optimization. Unlabelled footage
does not establish detection recall or gap-free recording.

The original evidence remains in
`/root/survng-measurements/pr170-20260905T180430Z/`. Raw snapshots, recordings,
decoded arrays, per-invocation fingerprints, scripts, logs, full options/zones
and host-specific provenance are intentionally not published. Numeric summaries
omit those details while retaining all measured result rows. Original files
are preserved unchanged by publication.

This is documentation only. The measurement did not change services, settings,
debug streams, packages or production code. This later PR publication was
separately requested; no additional replay or inference was run to prepare it.
