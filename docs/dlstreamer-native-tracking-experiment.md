# DL Streamer native tracking experiment

This experiment keeps SurvNG Hybrid as the authoritative event tracker while
allowing the live/substream GStreamer branch to run `gvadetect` and optionally
`gvatrack` before metadata crosses into SurvNG.

The experiment is **off by default**. It does not replace Hybrid IDs, recorded
main-stream refinement, ReID, incident persistence, or zone policy.

## Exact-frame reuse

Fresh native results, including completed empty results, are matched by capture
generation, source session, and exact PTS before Hybrid consumes them. They
replace duplicate live object inference. Missing or untrusted results use the
existing demand-driven detector. Fresh ROIs are captured before `gvatrack` so
propagated predictions cannot become detector confirmations.

Prediction-only frames still run fallback inference. Their positions may assist
Hybrid association for one update, without creating tracks or adding observations.

See [the implementation and validation report](native-live-detection-reuse.md)
for execution paths, provenance, counters, and the complete A/B configuration.

## Why test it

Intel DL Streamer `short-term-imageless` tracking can extrapolate ROI positions
on frames where object detection is skipped. That makes it possible to lower
live detector cadence with `inference-interval` while keeping a denser stream
of object boxes. `gvatrack` itself is CPU work; the expected trade is lower GPU
inference load for some additional CPU tracking work. This does not guarantee
lower total inference load: fallback still runs on selected skipped frames.

## Configuration

Under `detector`, start with:

```json
{
  "enabled": true,
  "live_pipeline_inference_enabled": true,
  "live_pipeline_inference_interval": 1,
  "live_pipeline_tracking": "off"
}
```

`live_pipeline_inference_interval` is intentionally bounded to 1-5, matching
the supported detector interval for DL Streamer's short-term imageless tracker.

Baselines worth comparing on the same cameras:

1. Current/default path: `live_pipeline_inference_enabled=false`.
2. Native detection only: enabled, interval `1`, tracking `off`.
3. Native tracking assist: enabled, interval `2` or `3`, tracking
   `short-term-imageless`.

## What remains authoritative

`native_track_id` is carried through when DL Streamer exposes an object ID, but
it is diagnostic metadata only. SurvNG Hybrid still performs event-scoped
association, high/low-confidence handling, appearance recovery, persisted track
IDs, cover selection, and recorded catch-up.

## What to measure

Camera pipeline status reports:

- `detect`
- `detect_fps`
- `inference_interval`
- `effective_inference_fps`
- `native_tracking`
- `native_tracking_authoritative` (always `false` in this experiment)
- `native_tracking_memory`

Compare Intel GPU utilization, CPU utilization, detector latency, incident recall,
duplicate/fragmented Hybrid tracks, and small/distant-object misses. Do not judge
the experiment on FPS alone.

## Rollback

Set `live_pipeline_inference_enabled` to `false` and restart. The default runtime
then returns to the existing frames-only live capture path.
