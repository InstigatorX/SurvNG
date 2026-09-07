# TrackTrack-inspired association experiment (PR 2)

This experiment adds directional consistency, confidence projection,
height-modulated overlap, and bounded per-cue diagnostics. It is a SurvNG-native
association candidate, not upstream TrackTrack and not a production promotion.
Track-Aware Initialization (TAI), duplicate-birth suppression, and depth-based
association are not part of this tranche.

## Run on saved Compare inputs

Download **replay inputs** from an incident's existing Compare view, then run:

```sh
.venv/bin/python -m survng.app.tracking_evaluation incident-replay.json \
  --association-cues --profile fixed_2fps --output association-2fps.json
.venv/bin/python -m survng.app.tracking_evaluation incident-replay.json \
  --association-cues --profile fixed_075fps --output association-slow.json
.venv/bin/python -m survng.app.tracking_evaluation incident-replay.json \
  --association-cues --profile sparse_gaps --output association-gaps.json
```

The flag evaluates exactly two independent trackers on the identical saved
observations: `survng_hybrid` and `survng_hybrid_multicue`. It does not rerun YOLO,
decoding, or embedding inference and needs no optional Ultralytics/PyTorch
tracking runtime. The normal SurvNG Python dependencies are still required.
Add `--labels incident-labels.json` for the existing identity metrics. Without
labels, outputs and proxies are diagnostic evidence, not ground-truth accuracy.

The default Compare UI and default replay still evaluate Hybrid, TrackTrack,
and BoT-SORT. This first association-cue experiment is replay-only; it does not
add a fourth default card or a persisted verdict type. The new identifier is
registered only on the CLI's fresh offline registry. Production and the historic
`survng_hybrid_candidate` identifier are not repurposed. `--label-template` and
`--association-cues` are mutually exclusive.

## Association policy

The corrected production schedule from PR #180 is inherited unchanged:
strict high-confidence geometry, uncontested relaxed high geometry,
selective high-confidence appearance recovery, strict low-confidence geometry,
remaining relaxed geometry, then low-confidence appearance recovery.

Only geometry-admissible pairs with competing tracks or detections receive cue
adjustments. An isolated one-track/one-detection edge keeps its exact baseline
score. Class, elapsed-age, scale-jump and existing geometry gates are unchanged.
The candidate does not widen those gates or add a match-count bonus.

For an ambiguous pair:

```text
score = max(0, G - 0.25 * (IoU - HMIoU) - 0.10 * C - 0.05 * A)
```

`G` is the original Hybrid geometry score, not a normalized probability. The
three penalties are experimental starting weights, not fitted parameters or
claims of optimal tracking quality. Constructor-only weights support ablation;
they must be finite, nonnegative and sum to at most 0.5.

**Overlap:** HMIoU is IoU multiplied by the vertical intersection/enclosing span.
Replacing a fraction of IoU with HMIoU distinguishes equal-IoU matches with
different height alignment while preserving SurvNG's zero-IoU center and
containment fallbacks. It is not an extra neural-network output.

**Confidence:** `C` is absolute difference from the track's projected confidence,
not a blanket preference for the highest-confidence detection. Projection uses
the two latest actual observations, their elapsed-time slope, a two-second
freshness bound, at most one observed interval of extrapolation, and clipping to
[0,1]. Missing history, equal/backwards timestamps and stale intervals contribute
no confidence penalty. Missing detections do not become confidence observations.

**Direction:** `A` is the angle/pi between smoothed center velocity and candidate
center displacement. Three recent observations must support stable motion, with
each history interval at most two seconds. Stationary/jitter-scale motion,
recent sharp turns, short history, and long/equal/backwards timestamp gaps are
neutral. There is no frame-counter motion model and no hard direction rejection.
This adapts the direction concept, not TrackTrack's corner-angle implementation.

All penalties enter the existing maximum-weight assignment. No new encoder,
Kalman filter, inference worker, global-motion compensation, or background task
is introduced. Track creation thresholds, confirmation, capacity, expiry,
completed-track recovery and persistence rules are inherited, not overridden.
Changed associations can of course change which detections remain unmatched;
that is different from changing the new-track initialization policy.

## Read the diagnostics

Inspect:

```text
engines.survng_hybrid_multicue.reid_diagnostics.association_cues
```

It includes the cue version, weights, valid/ambiguous/adjusted pair counts,
availability counts, summed score contributions, and the last 64 evaluated
ambiguous pairs. Each sample records the timestamp, track ID, detection index,
raw IoU/HMIoU, projected confidence, cue distances or neutral reasons, all three
weighted penalties, and the final score. No embeddings, crops or URLs are copied
into the diagnostic samples. A truncation count reports omitted earlier pairs.
These are evaluated candidate pairs, not a claim that every pair was selected;
use `frame_observations` for actual assignments. Totals cover evaluated valid
pairs and are not identity-accuracy metrics.

## Validation boundary

Regression tests cover a crossing at 2 FPS and 0.75 FPS, confidence/overlap ties,
neutral direction cases, timestamps, disabled-cue parity, bounded diagnostics,
unchanged birth policy, and the identity regressions from PR #180. The crossing
is synthetic evidence that the cue can change the intended decision, not proof
of improvement on a camera corpus. Reject a candidate that achieves fewer IDs
by merging different people. Do not enable live use based only on these tests.

Reference concepts: [Ultralytics TrackTrack API](https://docs.ultralytics.com/reference/trackers/track_tracker/).
