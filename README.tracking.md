# SurvNG Object Tracking

SurvNG uses one production tracker and an offline evaluation workflow:

- **SurvNG Hybrid** (`survng_hybrid`) is the default.
- **Hybrid candidate**, **TrackTrack**, and **BoT-SORT** are offline comparison
  engines. The candidate contains bounded ID-stability repairs; it is not live.
- Existing FastTrack and Deep OC-SORT results remain readable.

See [the evaluation guide](docs/tracking-evaluation.md) for saved-input replay,
annotation, scoring, sampling profiles, and the promotion criteria.

All four comparison engines receive the same saved detections and available
SurvNG person and vehicle embeddings. Missing optional engines are reported
individually; Hybrid evaluation and input capture remain available. Compare never changes the production
tracker. Historical verdicts remain readable. None of the comparison alternatives is
selectable for production.

## Why SurvNG Hybrid is the default

“Better” here means better suited to SurvNG's normal operating constraints. It
does not mean Hybrid will draw the best path in every recording.

### It is designed around SurvNG's sampled event pipeline

SurvNG tracks eligible objects from bounded samples of the main recording; it
does not run a detector on every source frame. Hybrid uses elapsed wall-clock
time, predicted box movement, overlap, center distance, object class, and
configurable lost-track grace to associate those sparse samples. Its lifecycle
therefore matches SurvNG's delayed samples, segment boundaries, and temporary
detection gaps directly.

Production therefore remains on Hybrid. Upstream trackers are exercised only
inside bounded offline comparisons and never participate in camera-event
lifecycle decisions.

### Appearance recovery is selective and shared with SurvNG

Hybrid first uses inexpensive geometry. It requests a person or vehicle ReID
embedding only when a track is created, periodically refreshed, or cannot be
recovered confidently from geometry. A strong compatible appearance match can
reconnect an object after occlusion or a larger movement without running ReID
on every ordinary match.

The resulting appearance signatures also feed SurvNG's durable appearance
index, related-incident suggestions, and cross-camera investigation features.
This keeps live tracking and later incident intelligence on the same identity
evidence and model thresholds.

The alternative trackers do not replace or improve the ReID models. Comparison
uses the same supplied embeddings, with each engine's association behavior and
appearance gates recorded in its diagnostics.

### It has a smaller and more predictable runtime footprint

Hybrid is implemented with the NumPy/OpenCV stack SurvNG already uses. It does
not require PyTorch, Ultralytics, LAP, CUDA Python packages, or a second model
runtime. This matters on an NVR that is simultaneously recording many streams,
decoding event windows, running OpenVINO inference, and serving playback.

The optional Ultralytics installation is substantially larger and has more
upstream dependencies. SurvNG loads it lazily only when an offline comparison
starts; routine status checks and production tracking do not load it.

`detector.tracking.sample_fps` is a memory setting as much as an accuracy one.
Each camera keeps twelve seconds of catch-up history for its main and live
streams so an unfinalized segment tail can still be walked, so the retained
frame count scales directly with the sampling rate. At the default 2.0 FPS that
history is roughly 35 MB per camera for a 1080p stream; at the maximum 5.0 FPS
it is roughly 85 MB, which raises total per-camera frame memory from around
55 MB to around 100 MB. The change applies on config reload rather than at
restart, and it multiplies by camera count, so raise it deliberately on a fleet
that is already close to its memory limit.

### Its behavior is directly configurable and observable

Hybrid's association, confirmation, lost-track, appearance-refresh, and
per-class ReID settings map directly to SurvNG configuration. Its decisions are
reported through track histories, recovery evidence, per-camera telemetry, and
incident replay. SurvNG can bound concurrent sessions and appearance work using
the same capacity controls used elsewhere in the application.

The implementation is maintained and regression-tested with SurvNG's event,
recording, zone, replay, and persistence behavior. It does not depend on private
tracker internals from another package.

## How Compare works

The incident viewer's **Compare** action evaluates a bounded 30-second window.
Detection and available appearance extraction run once. It retains their exact
outputs in a checksummed replay bundle, then runs current Hybrid, repaired Hybrid,
TrackTrack and BoT-SORT independently. Download the replay inputs before leaving
the viewer to preserve the full input data; history stores compact results only.

Select a 2 FPS target, a 0.75 FPS target, or deliberate sample gaps. These profiles
select saved source frames and preserve timestamps. They do not simulate the
production adaptive policy or full session/capacity lifecycle. An explicit
lost-track cutoff diagnostic shows when the backend's live-track predicate first
becomes false, while offline replay continues to observe possible recovery.

The viewer shows paths, timing, supplied appearance count and a fragmentation
proxy. Actual identity metrics require human-labeled ground truth and the offline
CLI described in the evaluation guide. Neither proxy counts nor visual verdicts
change the live tracker. Only one comparison can run at a time.

## Production and optional comparison runtime

Use **SurvNG Hybrid** when you want the normal recommended configuration:

- lowest dependency and memory overhead;
- behavior tuned for sparse, timestamped SurvNG samples;
- selective person and vehicle appearance recovery;
- direct integration with stored tracks and cross-camera intelligence; and
- stable configuration that SurvNG owns and tests end to end.

Production always uses Hybrid. Install the optional comparison
runtime with:

```bash
.venv/bin/pip install -r requirements-ultralytics-tracking.txt
```

TrackTrack and BoT-SORT require the tested Ultralytics 8.4.129 API. These
adapters reuse supplied embeddings and do not download another model.

## Practical conclusion

Hybrid is purpose-built for how SurvNG obtains, timestamps, stores, and reviews
detections. The alternative engines are offline evaluation candidates. Any
production change requires evidence from representative recordings. Accumulated side-by-side evidence helps expose difficult
camera scenes without allowing a comparison to alter runtime behavior.
