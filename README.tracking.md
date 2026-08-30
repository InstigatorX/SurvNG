# SurvNG Object Tracking

SurvNG uses one production tracker and one offline comparison engine:

- **SurvNG Hybrid** (`survng_hybrid`) is the default.
- **FastTrack** (`ultralytics_fasttrack`) is used only by the incident
  Compare workflow.

Both engines receive the same OpenVINO detections. Hybrid also receives the
available SurvNG person and vehicle embeddings; FastTrack intentionally tests a
lightweight motion-and-occlusion strategy. Compare never changes the production
tracker. Historic BoT-SORT and Deep OC-SORT verdicts remain readable, but those
engines are no longer offered for production or new comparisons.

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

FastTrack does not replace or improve the ReID models. Hybrid provides the
SurvNG-specific appearance recovery and indexing behavior; FastTrack is a
deliberately independent comparison of motion continuity and occlusion handling.

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

## Why Compare uses FastTrack

FastTrack supplies a meaningfully different benchmark. It extends a ByteTrack-
style association path with explicit occlusion detection, bounded motion-state
rollback, temporary search-box enlargement, and reappearance handling. The
adapter keeps classes isolated, gives every comparison independent IDs, and
converts its frame-based retention windows to SurvNG's configured sample rate.

## How Compare works

The incident viewer's **Compare** action is an offline diagnostic:

1. SurvNG decodes a bounded 30-second recording window beginning at the event.
2. OpenVINO object detection and appearance extraction run once.
3. The identical timestamped detections are sent to Hybrid and FastTrack;
   Hybrid additionally receives SurvNG's available appearance embeddings.
4. SurvNG renders both paths over the same recording and reports their tracker
   time, track count, observations, and extra-ID fragmentation signal.
5. You review the videos and record **Hybrid**, **FastTrack**, or **No clear
   winner**.

An extra track ID is only a fragmentation warning; without hand-labeled ground
truth, SurvNG cannot automatically declare an ID switch or a winner. Compare
does not change the configured live tracker. Only one comparison can run at a
time, so the longer window cannot multiply into concurrent detector jobs.

Test several difficult incidents from each important camera: partial
occlusions, distant people, vehicles crossing the full frame, objects entering
near an edge, night video, and temporary missed detections. Prefer the engine
that is consistently better across those cases, not the winner of one clip.

## Production and optional comparison runtime

Use **SurvNG Hybrid** when you want the normal recommended configuration:

- lowest dependency and memory overhead;
- behavior tuned for sparse, timestamped SurvNG samples;
- selective person and vehicle appearance recovery;
- direct integration with stored tracks and cross-camera intelligence; and
- stable configuration that SurvNG owns and tests end to end.

Production always uses Hybrid. Install the optional FastTrack comparison
runtime with:

```bash
.venv/bin/pip install -r requirements-ultralytics-tracking.txt
```

SurvNG pins a tested version for reproducible installs and accepts compatible
patch releases on the reviewed 8.4.x tracker API line.

## Practical conclusion

Hybrid is purpose-built for how SurvNG obtains, timestamps, stores, and reviews
detections. FastTrack is an offline diagnostic benchmark, not an automatic
production upgrade. Accumulated side-by-side evidence helps expose difficult
camera scenes without allowing a comparison to alter runtime behavior.
