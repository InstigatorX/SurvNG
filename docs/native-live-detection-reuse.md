# PR #206: exact-frame native detection reuse

Implemented on `experiment/dlstreamer-native-tracking`, starting at `9ce4431`.
The implementation is published on PR #206. The separate worktree is
`/root/SurvNG-native-reuse`; existing work in `/root/SurvNG` was preserved.
No PR merge, deployment, or database migration.

## Verified root cause and execution paths

The base branch constructs live capture with detection disabled. PR #206 makes
`AppManager` opt into `gvadetect`, but leaves both tracking routes demand-driven:

1. `CameraWorker` creates `CameraFrameTimeline(requires_inference=True)`.
   `remember_capture()` retains live frames with that flag, and
   `_hydrate_live_results()` immediately returns. Recorded catch-up can therefore
   bridge through live pixels without consuming native results.
2. `CameraWorker._get_tracking_capture()` prefers main frames; when eligible live
   geometry is used, it independently returns `TrackingFrame(..., requires_inference=True)`.
3. `ObjectTrackingSession._run()` calls `_tracking_detections_for_frame()`, which
   calls `_detect_tracking_objects()` → `detector.detect_tracking()` for these frames.
4. Initial live admission also lacks a native provider and calls `detect_initial()`.

Simply removing the hydration guard would be unsafe. `DetectionHistory.match()`
previously accepted an earlier result within a detector-cadence window, and the
metadata appsink was downstream of `gvatrack`. Tracker ROIs could consequently
be treated as fresh observations, including on frames containing real detections.

New live path:

```text
hardware decode / VAMemory → gvadetect
  → pre-tracker probe records fresh detector ROIs and frame provenance
  → output queue → optional gvatrack → gvametaconvert → metadata appsink
  → DetectionSnapshot / session-qualified DetectionHistory
  → capture generation + source session + exact PTS check
  → TrackingFrame → ObjectTrackingSession → existing Hybrid
```

Both direct live tracking and delayed timeline hydration call
`CameraFrameTimeline.tracking_frame()`. A fresh exact snapshot, including an
empty snapshot, bypasses `detect_tracking()` at `_tracking_detections_for_frame()`.
Initial admission gets the same strict capture matcher and bypasses
`detect_initial()` only with fresh exact evidence.

Missing, malformed, unknown-provenance, wrong-session, wrong-generation, or
wrong-PTS evidence uses the existing demand-driven detector. PTS matching allows
only 1 ns for floating-point transport rounding, not a nearby frame. The older
cadence matcher remains available for display. Retained timeline results remain
bound to the historical frame they originally matched.

Main-frame preference, geometry trust checks, recorded catch-up, main recording
refinement, zone policy, high/low association, ReID, appearance recovery, event
confirmation, security checks, and cover selection keep their existing paths.
Native object confidence uses the existing low candidate floor configured by
`live_detection_threshold()`.

## Fresh inference and prediction contract

The provenance field is carried on `DetectionSnapshot` and copied to objects as
`detection_provenance`:

- `native_fresh_detection`: actual native inference for these pixels.
- `native_tracked_prediction`: skipped detector frame with optional tracker output.
- `fallback_live_inference`: demand-driven inference on live pixels.
- `recorded_refinement`: existing inference on main/recorded evidence.
- `unknown`: no trustworthy native inference contract; never authorizes reuse.

The probe runs on **gvadetect's source pad, before the leaky queue and gvatrack**.
It reads ROI metadata without mapping pixels. It counts detector output buffers,
not appsink arrivals, and saves the unmodified detector ROIs. The bounded ledger
holds at most 128 entries; missing entries fail closed. A backward/reused PTS
invalidates native identity until pipeline recreation, preventing collisions with
metadata already queued downstream. Malformed metadata increments
`native_evidence_invalid` and leaves video available for fallback.

The scheduling contract is full-frame inference, `no-block=false`, and fixed
`inference-interval`: first buffer inferred, then every Nth buffer. This was
verified against DL Streamer source and exercised on the installed 2026.2 runtime:

- [Inference scheduling and ordered buffer delivery](https://github.com/open-edge-platform/dlstreamer/blob/7647126a50358df12792bf5aecec7249f342e906/src/monolithic/gst/inference_elements/base/inference_impl.cpp)
- [First-frame scheduling initialization](https://github.com/open-edge-platform/dlstreamer/blob/7647126a50358df12792bf5aecec7249f342e906/src/monolithic/gst/inference_elements/base/gva_base_inference.cpp)
- [Tracker appending unassociated predicted ROIs](https://github.com/open-edge-platform/dlstreamer/blob/7647126a50358df12792bf5aecec7249f342e906/src/monolithic/gst/elements/gvatrack/tracker.cpp)

Fresh snapshots contain only the ROIs captured before the tracker, including when
gvatrack subsequently adds predictions to that same buffer. Empty inference is
therefore distinguishable from an empty skipped frame. Skipping without gvatrack
is `unknown`, not a tracker prediction.

Prediction-only frames **still run fallback inference**. For an exact prediction
snapshot, Hybrid can use an unambiguous predicted position as a one-update
association hint for an existing track. Applying the hint does not change hits,
confidence, last-seen time, history, appearance, or identity, and cannot create a
track. Only the actual fallback detection adds an observation. Hints expire after
that update. Native IDs do not select or replace Hybrid/event/persisted IDs.

This implements conservative assistance; interval 3 does not promise that all
SurvNG object inference runs at 1.67 FPS. At a 5 FPS native branch rate, native
model inference is approximately 1.67 FPS, but selected skipped frames still need
fallback inference to preserve recall.

## Counters

Read the existing owner-only socket with `./survngctl status` in the deployed
checkout. Per-camera status exposes:

| Location | Fields | Meaning |
| --- | --- | --- |
| `live_detection_matching` | `native_detection_snapshots` | Accepted fresh native snapshots, including empty results |
| `live_detection_matching` | `native_tracker_predictions` | Accepted prediction snapshots; not detector confirmations |
| `live_detection_matching` | `native_detection_stale` | Exact-match lookups rejected for an earlier, different PTS |
| `tracking` | `native_detection_hits` | Tracking frames actually served by native inference |
| `tracking` | `native_detection_empty_hits` | Subset of hits containing zero native objects |
| `tracking` | `native_detection_misses` | Live tracking attempts requiring fallback |
| `tracking` | `fallback_detector_calls` | Demand-driven live tracking detector requests, including deferred requests |
| `live_pipeline` | `detect_fps`, `inference_interval`, `effective_inference_fps`, `native_evidence_invalid` | Configured native cadence and metadata failure count |

Tracking counters survive incident changes for the lifetime of the session
service object. Capture counters survive history resets for the capture service
lifetime. Restarting/recreating services resets counters. Use deltas over equal
A/B windows. Matching counters include hydration retries, so stale lookups are
not unique frames. Tracking hit/call counters measure consumption rather than
lookups; they exclude initial admission and main refinement. Global totals can
be computed by summing camera deltas. No high-frequency logging was added.

## First live A/B test

Merge these keys into the existing configuration; keep model, thresholds, zones,
ReID, and recording/refinement policy unchanged. Use the same frame rates in both
arms and restart after changing the native switch:

```json
{
  "motion_qualification": {"sample_fps": 5.0},
  "detector": {
    "enabled": true,
    "live_sample_fps": 5.0,
    "live_pipeline_inference_enabled": true,
    "live_pipeline_inference_interval": 1,
    "live_pipeline_tracking": "off",
    "tracking": {
      "enabled": true,
      "implementation": "survng_hybrid",
      "sample_fps": 5.0
    }
  }
}
```

- **A / rollback:** set `live_pipeline_inference_enabled=false`.
- **B / first experiment:** use the settings above: interval 1, tracking off.
- **Later assistance comparison:** after validating B, set interval 3 and
  `live_pipeline_tracking="short-term-imageless"`.

Compare native hits and fallback calls alongside GPU/CPU load, incident recall,
small/distant-object results, and fragmentation. Verify main-stream refinement
still occurs for incidents. Live bridging requires the existing identical-FOV
trust check; warmed main frames still take precedence, so some cameras/sessions
may produce few live hits. Late results or unmatched branch PTS legitimately
fall back. Continuous native inference also runs outside tracking sessions;
net resource savings must be measured, not inferred solely from fewer duplicates.

## Validation

Full branch and combined PR/base regression suites were run with:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
/root/SurvNG/.venv/bin/python -m pytest -q
```

- Branch suite: **2,857 passed, 1 skipped, 279 subtests passed**.
- Combined tree with base `c4f8b76`: **2,961 passed, 1 skipped, 279 subtests passed**.
  This was an exported integration tree; no branch or PR was merged.
- After final duplicate-PTS capture fencing, the focused capture/provenance/Hybrid
  suite passed **124 tests**, with one skipped.
- The skipped generated-source test requires GI unavailable to that virtual
  environment. Native execution was separately exercised with system Python.

Older tests were updated to supply explicit fresh provenance and exact PTS,
and to require fallback for stale/missing metadata. Main-stream cancellation,
recorded evidence, geometry trust, and deferred-inference status contracts remain
covered. The initial full run also exposed missing frontend build artifacts;
the suites above ran after the normal frontend build.

Native validation uses system Python with GI/OpenVINO/DL Streamer available:

```bash
python3 scripts/gstreamer-smoke.py
```

The synthetic CPU suite exercises positive/empty detections, interval 3 plus
short-term tracking, frames-only rollback, main frames, shared multi-stream
inference, NMS, and existing sparse-metadata startup checks. Strengthened checks
assert exact fresh-frame matches and explicit provenance. Observed detection-only
runs had 8 fresh snapshots and 8 exact frame matches, including empty detections;
the interval-3 run had 3 fresh snapshots and 5 prediction snapshots, with 3 exact
fresh matches. The native suite passes; plugin scanning emits an existing
`_dma_fmt_to_dma_drm_fmts` assertion diagnostic without failing the suite.

`git diff --check` passed. No production camera stream, live GPU performance,
recognition accuracy, or production incident A/B run was tested. Synthetic native
execution proves the metadata contract and plumbing, not real-world recall.
Revalidate scheduling/provenance when upgrading DL Streamer or changing its
inference scheduling properties.

## Changed files

- Runtime: `survng/dlstreamer_live.py`, `survng/app/live_detections.py`,
  `survng/app/camera_capture.py`, `survng/app/dlstreamer_capture.py`,
  `survng/app/camera.py`,
  `survng/app/tracking_frames.py`, `survng/app/object_track/types.py`,
  `survng/app/object_track/session.py`, `survng/app/object_track/hybrid.py`,
  `survng/app/motion_pipeline/object_detection.py`, `survng/app/local_observability.py`.
- Validation: `tests/test_native_detection_reuse.py`,
  `tests/test_live_detection_evidence.py`, `tests/test_object_tracking.py`,
  `tests/test_dlstreamer_memory_graph.py`, `tests/test_gstreamer_campaign.py`,
  `tests/test_gstreamer_review_contracts.py`, `tests/test_qualified_inference.py`,
  `tests/test_recorded_object_detection.py`, `scripts/gstreamer-smoke.py`.
- Documentation: `docs/native-live-detection-reuse.md`,
  `docs/dlstreamer-native-tracking-experiment.md`.
