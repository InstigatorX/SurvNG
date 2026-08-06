# SurvNG Object Tracking

SurvNG uses one production tracker and one offline comparison engine:

- **SurvNG Hybrid** (`survng_hybrid`) is the default.
- **Deep OC-SORT** (`ultralytics_deepocsort`) is used only by the incident
  Compare workflow.

Both engines receive the same OpenVINO detections and available SurvNG person
and vehicle embeddings. Compare never changes the production tracker. Historic
BoT-SORT comparison verdicts remain readable, but BoT-SORT is no longer offered
for production or new comparisons.

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

Deep OC-SORT can consume the same embeddings during Compare, but installing it
does not improve the ReID models themselves. Hybrid already provides the
SurvNG-specific appearance recovery and indexing behavior without Ultralytics.

### It has a smaller and more predictable runtime footprint

Hybrid is implemented with the NumPy/OpenCV stack SurvNG already uses. It does
not require PyTorch, Ultralytics, LAP, CUDA Python packages, or a second model
runtime. This matters on an NVR that is simultaneously recording many streams,
decoding event windows, running OpenVINO inference, and serving playback.

The optional Ultralytics installation is substantially larger and has more
upstream dependencies. SurvNG loads it lazily only when an offline comparison
starts; routine status checks and production tracking do not load it.

### Its behavior is directly configurable and observable

Hybrid's association, confirmation, lost-track, appearance-refresh, and
per-class ReID settings map directly to SurvNG configuration. Its decisions are
reported through track histories, recovery evidence, per-camera telemetry, and
incident replay. SurvNG can bound concurrent sessions and appearance work using
the same capacity controls used elsewhere in the application.

The implementation is maintained and regression-tested with SurvNG's event,
recording, zone, replay, and persistence behavior. It does not depend on private
tracker internals from another package.

## Why Compare uses Deep OC-SORT

Deep OC-SORT supplies a meaningfully different benchmark. Its
observation-centric recovery repairs motion state after missed observations and
occlusions, while adaptive appearance association can consume SurvNG's existing
person and vehicle embeddings. The adapter keeps classes isolated, gives every
comparison independent IDs, converts retention to the configured sample rate,
and disables camera-motion compensation for fixed cameras.

## How Compare works

The incident viewer's **Compare** action is an offline diagnostic:

1. SurvNG decodes a bounded 30-second recording window beginning at the event.
2. OpenVINO object detection and appearance extraction run once.
3. The identical timestamped detections and embeddings are sent to Hybrid and
   Deep OC-SORT.
4. SurvNG renders both paths over the same recording and reports their tracker
   time, track count, observations, and extra-ID fragmentation signal.
5. You review the videos and record **Hybrid**, **Deep OC-SORT**, or **No clear
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

Production always uses Hybrid. Install the optional Deep OC-SORT comparison
runtime with:

```bash
.venv/bin/pip install -r requirements-ultralytics-tracking.txt
```

SurvNG pins a tested version for reproducible installs and accepts compatible
patch releases on the reviewed 8.4.x tracker API line.

## Practical conclusion

Hybrid is purpose-built for how SurvNG obtains, timestamps, stores, and reviews
detections. Deep OC-SORT is an offline diagnostic benchmark, not an automatic
production upgrade. Accumulated side-by-side evidence helps expose difficult
camera scenes without allowing a comparison to alter runtime behavior.
