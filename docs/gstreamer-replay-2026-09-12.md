# GStreamer inference and replay fixes

This supersedes the initial runtime/cadence choices in the September 11 audit.
Changes are confined to `gstreamer`; the v1.2 checkout is unchanged.

## Later stability findings and v1.3 integration

The bounded tests below do not establish long-running GPU stability. Later
observation on the N100 found seven shared-supervisor VA-API image-conversion
failures between 03:48 and 04:11 UTC on September 12, including failures with
the newly exported end-to-end model. Recovery restored live capture, but each
failure interrupted live inference and EMA input across cameras. Independent
recording continued. Fifteen recordings covering investigated failure windows
decoded cleanly with both software and VA-API hardware decoding.

The test service subsequently migrated to an i9-12900H / Iris Xe host with the
same image and Intel userspace packages. Early checks found no recurrence of
the original VA-API error, but startup recording-prewarm errors and a supervisor
output-ended warning occurred. A configuration save at 04:19:55 UTC restarted
capture; its uninterrupted baseline must be measured from that restart.

The `v1.3-gstreamer` integration preserves the existing GStreamer implementation
and v1.3's newer recorded-tracking continuity, detector-contract, storage, and
UI fixes. It does **not** implement the proposed supervisor-owned shared VA
context or teardown acknowledgement; those remain a separate stability task.
The inference path remains GPU-resident; no system-memory inference fallback
is introduced. Merge validation is not a substitute for a hardware soak and
recorded-scene accuracy checks on the combined application.

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

## Published-image and deployment validation

Implementation commit: `8731deb98a175ae9a6c707340e585a37d55c5920`.
Test-server image: `ghcr.io/instigatorx/survng:sha-8731deb-intel`, manifest digest
`sha256:7710b9b32b82a33b66954ca72db8d2dc19562750a11b6f2985da5c5b4933e0b7`.
The installed native packages were verified as DL Streamer 2026.2.0 and
OpenVINO 2026.2.0.21903. No diagnostic code or source-file overrides are deployed.

- [CI](https://github.com/InstigatorX/SurvNG/actions/runs/34670188521) passed:
  2,134 Python tests, 216 subtests, 14 skips; all 52 frontend test files; native
  Intel metadata and multistream smoke checks. The optional legacy Ultralytics
  dependency responsible for the baseline local failure is absent in CI.
- [Docker publication](https://github.com/InstigatorX/SurvNG/actions/runs/34670188507)
  passed and reused unaffected dependency layers.
- An isolated three-camera hardware harness discarded only detection messages
  for an eight-second window while video continued. The five-second watchdog
  fired, terminated the old supervisor, and all three cameras recovered with
  new evidence sessions and advancing inference. The harness asserted recovery
  at the end of its 95-second run; fault injection is not application code.
- The unmodified published image ran a five-minute full-application test with
  real Gate/Foyer streams and the aligned Downstairs recording replay. Observed
  consecutive live person admissions were 0.590 and 0.693 seconds after the EMA
  trigger; recorded-main confirmation followed at 7.615 and 12.843 seconds.
  All three cameras maintained inference progress without session resets or
  inference failures. Recording references and thumbnails existed.
- The normal three-camera test service was restored on the same image and
  passed a separate five-minute observation, including a normal event
  refinement. It remained container-healthy with zero restarts and zero failed
  inferences. Finalized main/live recordings for all cameras contained readable
  H.264 video and AAC audio. Memory was approximately 2.4 GiB at a sampled check.
- Six prewarm errors on deployment corresponded to old, incomplete recording
  tails from before deployment, with missing MP4 `moov` atoms. Existing index
  revalidation marked them unplayable; no recordings were deleted. Separately,
  all six final recording tails from the clean-image replay shutdown validated.
- Only live inference cadence changed in the test configuration: 2.5 FPS,
  while EMA remains 5 FPS and configured tracking remains 2 FPS. Previous
  configuration and Compose files have `.before-8731deb` backups on the test
  host. Replay containers are stopped; original footage/models are preserved.

These are bounded regression and hardware checks, not an exhaustive accuracy
benchmark or an overnight stability guarantee. Recorded-main confirmation still
depends on segment finalization; timestamp-mismatched live evidence is rejected.
