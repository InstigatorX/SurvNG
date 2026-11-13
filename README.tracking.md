# SurvNG Object Tracking

SurvNG uses one production tracker and an offline evaluation workflow:

- **SurvNG Sparse Identity** (`survng_sparse_identity`) is the production default
  for identity-retention testing. It adds cascaded association, buffered IoU,
  quality-gated ReID galleries, and entity relink.
- **SurvNG Hybrid** (`survng_hybrid`) remains selectable as the prior production
  engine.
## SurvNG Hybrid behavior

“Better” here means better suited to SurvNG's normal operating constraints. It
does not mean Hybrid will draw the best path in every recording.

### It is designed around SurvNG's sampled event pipeline

SurvNG tracks eligible objects from bounded samples of the main recording; it
does not run a detector on every source frame. Hybrid uses elapsed wall-clock
time, predicted box movement, overlap, center distance, object class, and
configurable lost-track grace to associate those sparse samples. Its lifecycle
therefore matches SurvNG's delayed samples, segment boundaries, and temporary
detection gaps directly.

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

### It has a smaller and more predictable runtime footprint

Hybrid is implemented with the NumPy/OpenCV stack SurvNG already uses. It does
not require PyTorch, Ultralytics, LAP, CUDA Python packages, or a second model
runtime. This matters on an NVR that is simultaneously recording many streams,
decoding event windows, running OpenVINO inference, and serving playback.

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

## Production runtime

Sparse Identity is the default; Hybrid remains selectable. Both use the
existing NumPy/OpenCV stack. Configure the implementation in Admin and inspect
stored tracks and runtime diagnostics to assess behavior.

The comparison UI, comparison APIs, offline replay command, and optional
Ultralytics backends have been removed. Identity metrics remain available to
production tracker regression tests.
