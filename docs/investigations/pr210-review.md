# PR 210 review and replay investigation

## Recorded evidence

Downstairs event 71305 (2026-09-15 21:31:11 EDT) has roughly 21 seconds
of saved native history and four continuous recording segments. The production
clip builder produced 32.599675 seconds of MP4. The production HLS playlist and
fragment remux also play past 30 seconds in local Chromium and Linux WebKit.
Three main/substream frame pairs passed the existing ORB alignment estimator
with identity geometry. The owner confirmed matching FOVs for every camera.
The previous stored false compatibility flag was unverified geometry, not a
measured mismatch.

The reported failure is specifically Tracks replay on desktop macOS Safari;
Clean plays 32.6 seconds. No local Safari connector is available, so the exact
macOS failure was not reproduced. Linux WebKit is not macOS Safari.

## Review findings and fixes

1. Historical native records retained unverified replay geometry after camera
   confirmation. Apply current confirmation when projecting API responses,
   retaining archived track data. Confirmed cameras use the same main recording
   for Clean and Tracks.
2. A destroyed Shaka player's pending attach rejection could call the current
   error callback. Ignore that rejection after disposal; a browser regression
   explicitly closes the player before rejecting the attachment.
3. Growing active histories could replace a playing video. Retain its window
   while active, extend on completion, and preserve the current epoch when
   loading a replacement window. Test both incident surfaces.
4. HLS-to-MP4 fallback restarted playback. Preserve the playback epoch across
   transport origins. Safari substream playback uses the complete MP4 path.
5. Explicit batching could be configured to exceed freshness limits even at
   nominal cadence when other cameras disconnect. Backend and Admin reject
   those combinations. Real FPS below target still requires monitoring.
6. Correct stale native-guide language about offline high-resolution covers.

Reviewed native admission/freshness, tracking history, stationary state,
evidence queue completion, metadata reconnects, shared batch configuration,
recording fragment/clip generation, replay lifecycle, API projections, and
telemetry semantics. Follow-up review checked the fixes and their regression
coverage. No remaining actionable findings were identified in these paths;
this is not a guarantee of defect-free behavior.

## Validation

- Focused Python regression campaign: 357 passed, one environment-dependent
  capture smoke skipped, four subtests passed.
- Frontend: 68 unit test files passed.
- Browser tests use three independently encoded recording files through the
  production HLS/remux and MP4 builders. They cover segment transitions,
  boxes at 22 seconds, late history, repeated openings, active updates,
  completion, fallback position, and confirmed main-recording geometry.
- Linux WebKit's missing macOS native-HLS capability is simulated only for
  transport selection; actual MP4 decoding runs in WebKit. This does not
  validate the user's exact Safari installation.

## Latency interpretation

The selected INT8 model's isolated GPU probe previously verified DP4A in
102 executed convolution kernels. This supports an INT8 benefit, but there is
no controlled FP16/INT8 comparison explaining the entire historical drop.
The last batch-2 restart increased sampled means from approximately 23–38 ms
to 40–49 ms, without a demonstrated FPS gain. The current metric measures
entry-to-exit time at gvadetect, including preprocessing and scheduling;
it is not end-to-end camera or notification latency. Recent batching did not
introduce a new timing definition.
