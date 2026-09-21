# Tracking identity corpus (Phase 0)

Use this checklist before changing production Hybrid. Collect evidence with the
incident Compare action and the offline evaluation CLI described in
[tracking-evaluation.md](tracking-evaluation.md).

## Collect

Choose 20–30 incidents across cameras. Include:

- crossings and close walks
- partial occlusions and doorway merges/splits
- 0.5 / 1 / 3 / 5 / 10 second returns
- small or distant people, night scenes, box-size jitter
- uncomplicated single-person controls

Stratify top-down versus oblique cameras. Reserve about one third as held-out;
do not tune thresholds against that set. Keep camera, day/night, scenario, and
train/held-out labels in a corpus manifest outside the replay JSON.

For each incident: open Compare, run a sampling profile, **Download replay
inputs**. Prefer captures with person ReID enabled so appearance experiments are
meaningful (`appearance_input_count` > 0).

## Baseline

```sh
python -m survng.app.tracking_evaluation gate-replay.json --labels gate-labels.json --profile fixed_2fps --output gate-scored.json
python -m survng.app.tracking_evaluation gate-replay.json --labels gate-labels.json --profile sparse_gaps --output gate-gaps.json
```

Record Hybrid and Sparse Identity `identity_metrics`:

- IDF1, ID switches, fragmentations, false merges
- `recovery_by_gap` precision at 0.5s / 1s / 3s / 5s / 10s

Fleet telemetry to snapshot alongside: `reid_enabled`, `reid_recoveries`,
`association_counts`, `reid_avoided_geometry_matches`.

## Promotion

`survng_sparse_identity` remains offline-only until a held-out promotion gate
passes. Use `promotion_gate(baseline_metrics, candidate_metrics)` from
`survng.app.tracking_evaluation` (or equivalent manual comparison): promote only
when fragmentations and/or ID switches and/or IDF1 improve and false merges do
not rise.
