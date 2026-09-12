# GStreamer inference and replay fixes

This supersedes the initial runtime/cadence choices in the September 11 audit.
Changes are confined to `gstreamer`; the v1.2 checkout is unchanged.

## Failures and remedies

1. **YOLO26 GPU accuracy:** the application's OpenVINO Python wheel was 2026.3,
   but DL Streamer linked native OpenVINO 2026.1. The latter produced NaN scores
   for the actual YOLO26s model in FP16, which appeared as empty detections.
   Updating the Python wheel alone cannot fix that. The Intel image now pins
   DL Streamer 2026.2.0, including its matching native OpenVINO dependencies;
   CI and the isolated smoke Compose use the corresponding digest-pinned image.
   No model replacement, forced FP32, or silent CPU fallback is used.
2. **Late metadata consumption:** capture harvested completed inference only
   after another video read. During a video gap, admission/display queried old
   history while current metadata was already waiting in the inbox. Queries now
   harvest the active handle's completed snapshots before matching. Session,
   generation, PTS, ordered-history and missing-versus-empty checks still apply.
   There is no widened tolerance and no reuse of arbitrary previous boxes.
3. **Shared inference failure containment:** a native `gvadetect` error can
   poison the shared model/request pool while other cameras still emit video.
   Typed inference failures invalidate the shared supervisor. A five-second
   no-inference-progress watchdog catches silent stalls while video continues.
   Empty results count as progress; duplicate timestamps do not. Ordinary RTSP
   outages retain camera-local recovery. Supervisor replacement creates fresh
   evidence sessions. Stderr is drained promptly, not in blocking 64 KiB batches.
4. **Independent cadence:** `detector.live_sample_fps` defaults to 5 for existing
   installations. For the N100 test host, 2.5 live detection FPS is tested with
   5 FPS EMA. Recorded-main capture retains its previous EMA/tracking-derived
   rate through a separate internal `--main-fps` argument. Changing live cadence
   uses the existing transactional manager reload; it does not change EMA,
   tracking sample settings, recording cadence or zone geometry.
5. **Observability:** the owner-only socket includes
   `cameras[].live_detection_matching.lag_seconds`. Missing/current-session
   evidence is distinguished from a large video-to-detection PTS gap.

GPU capture uses VA surface sharing, batch size 1, one inference request and
one VA preprocessing worker. GPU execution requests latency mode with one
stream. Shared model scheduling uses `throughput`: separate RTSP pipelines
have independent PTS origins, so comparing their raw PTS for latency scheduling
unfairly prioritizes some cameras. The bounded detection input remains before
preprocessing, with `inference-interval=1`; interval skipping is unnecessary to
implement a lower input cadence. CPU inference retains the supported VA-download
preprocessor rather than requesting GPU surface sharing.

## Measurements on the test N100

The original Downstairs walk-through was preserved and replayed via loopback
RTSP into a full application instance with separate configuration, database and
media. Replay notifications were disabled. Main/live clips were trimmed with
stream copy to align their starts; original recordings/models were not modified.

- Identical person frame, native OpenVINO 2026.1 FP16: NaN scores. Native 2026.2
  FP16: person confidence 0.9302, approximately 93 ms/inference. FP32: confidence
  0.9314, approximately 171 ms/inference. The newer app runtime agreed with 2026.2.
- Native 2026.2 recognition check on the original live recording: 81 frames,
  19 person-positive frames, 36 frames in the specified empty window and zero
  person detections in that window; playback reached EOS.
- Three-camera 5 FPS overload test: continuous inference for three minutes,
  but late matching demonstrated that decoded FPS is not an accuracy check.
- Three-camera 2.5 FPS test: continuous inference without the previously
  reproduced VA error during the 150-second test. This is bounded evidence,
  not a guarantee against all future driver failures.
- Full-application replay after the metadata-consumption fix: consecutive person
  events became visible through the live fast path at 0.579 and 0.572 seconds
  after the EMA trigger. Recorded-main refinement followed at 4.401 and 8.461
  seconds. Snapshots and referenced recording files existed. Refinement delay
  still depends on finalizing the configured recording segments.
- The reduced RTSP latency experiment did not explain/fix the metadata ownership
  problem and was removed. Temporary timing instrumentation was also removed.

The local full suite passed 2,145 tests and 216 subtests (one skipped), with one
failure in the optional legacy Ultralytics DeepOCSort far-person ReID test.
That same failure was reproduced in a detached, unchanged `2fe0c1e` checkout;
it is not introduced by this change. Focused capture/configuration/packaging
and observability tests passed. Clean-image deployment and recovery results
should be checked separately from these source-level tests.

## Repeat recognition checks

The synthetic smoke test verifies plumbing, not recognition. The Intel image
also includes `scripts/gstreamer-model-check.py` for a local recording containing
a known object, with an optional empty window after it leaves:

```bash
/usr/bin/python3 /app/scripts/gstreamer-model-check.py \
  --video /replay/live.mp4 \
  --model /models/yolo26s_openvino_model/yolo26s.xml \
  --labels /models/yolo26s_openvino_model/classes.txt \
  --expect-label person --negative-after 45
```

Use a read-only recording/model mount and GPU device access. Choose the negative
window for the actual clip, not an arbitrary timestamp. The command fails for
empty-only output, incomplete playback, or detections in the specified empty
window. It does not replace full-app replay or multi-camera/recovery testing.
