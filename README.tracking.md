# SurvNG Object Tracking

SurvNG supports two object-tracking engines:

- **SurvNG Hybrid** (`survng_hybrid`) is the default.
- **Ultralytics BoT-SORT** (`ultralytics_botsort`) is an optional alternative
  and offline comparison engine.

Both engines receive the same OpenVINO detections. Neither engine performs
object classification, and selecting a tracker does not change the object
detector or its confidence scores.

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

BoT-SORT originated in a conventional consecutive-frame tracking pipeline. The
SurvNG adapter makes it work with sampled detections, but its Kalman state and
frame-oriented thresholds can still be less intuitive when observations are
several hundred milliseconds apart or a recording has a coverage gap.

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

BoT-SORT can consume the same embeddings during a tracking run, but installing
it does not improve the ReID models themselves. Hybrid already provides the
SurvNG-specific appearance recovery and indexing behavior without requiring
Ultralytics.

### It has a smaller and more predictable runtime footprint

Hybrid is implemented with the NumPy/OpenCV stack SurvNG already uses. It does
not require PyTorch, Ultralytics, LAP, CUDA Python packages, or a second model
runtime. This matters on an NVR that is simultaneously recording many streams,
decoding event windows, running OpenVINO inference, and serving playback.

The optional Ultralytics installation is substantially larger and has more
upstream dependencies. SurvNG loads it lazily only when BoT-SORT is selected or
an offline comparison is started; routine status checks and Hybrid tracking do
not load it.

### Its behavior is directly configurable and observable

Hybrid's association, confirmation, lost-track, appearance-refresh, and
per-class ReID settings map directly to SurvNG configuration. Its decisions are
reported through track histories, recovery evidence, per-camera telemetry, and
incident replay. SurvNG can bound concurrent sessions and appearance work using
the same capacity controls used elsewhere in the application.

The implementation is maintained and regression-tested with SurvNG's event,
recording, zone, replay, and persistence behavior. It does not depend on private
tracker internals from another package.

## Where Ultralytics BoT-SORT may be better

BoT-SORT remains useful. On a smooth, densely sampled sequence with stable
detections, its Kalman motion model and multi-stage association can produce a
cleaner trajectory or retain an ID through a geometry pattern that Hybrid
handles less well. The result depends on camera angle, object speed, detector
quality, sample rate, occlusion, and keyframe availability.

Use BoT-SORT when representative comparisons consistently look better—not
because it is a larger dependency or has a familiar upstream name. Its use in
SurvNG remains class-aware, disables global camera-motion compensation for
fixed cameras, and uses the same detections and available appearance features
as Hybrid.

## How Compare works

The incident viewer's **Compare** action is an offline diagnostic:

1. SurvNG decodes a bounded 30-second recording window beginning at the event.
2. OpenVINO object detection and appearance extraction run once.
3. The identical timestamped detections are sent to Hybrid and BoT-SORT.
4. SurvNG renders both paths over the same recording and reports their tracker
   time, track count, observations, and extra-ID fragmentation signal.
5. You review the videos and record **Hybrid**, **BoT-SORT**, or **No clear
   winner**.

An extra track ID is only a fragmentation warning; without hand-labeled ground
truth, SurvNG cannot automatically declare an ID switch or a winner. Compare
does not change the configured live tracker. Only one comparison can run at a
time, so the longer window cannot multiply into concurrent detector jobs.

Test several difficult incidents from each important camera: partial
occlusions, distant people, vehicles crossing the full frame, objects entering
near an edge, night video, and temporary missed detections. Prefer the engine
that is consistently better across those cases, not the winner of one clip.

## Choosing an engine

Use **SurvNG Hybrid** when you want the normal recommended configuration:

- lowest dependency and memory overhead;
- behavior tuned for sparse, timestamped SurvNG samples;
- selective person and vehicle appearance recovery;
- direct integration with stored tracks and cross-camera intelligence; and
- stable configuration that SurvNG owns and tests end to end.

Consider **Ultralytics BoT-SORT** when repeated offline comparisons on your own
cameras demonstrate a meaningful improvement that justifies its larger runtime
stack. Install it with:

```bash
.venv/bin/pip install -r requirements-ultralytics-tracking.txt
```

Then select it under **Admin → Object Detection → Tracking**. SurvNG pins a
tested version for reproducible installs and accepts compatible patch releases
on the reviewed 8.4.x tracker API line.

## Practical conclusion

Hybrid is the safer system-level default because it is purpose-built for how
SurvNG obtains, timestamps, stores, and reviews detections. BoT-SORT is a useful
alternative algorithm and benchmark, not an automatic accuracy upgrade. The
best choice for a particular installation should be based on side-by-side
evidence from that installation's cameras.
