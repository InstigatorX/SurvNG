# Native runtime review campaign — September 12, 2026

Scope: `ceb931c` and the native installer/preflight it extends from `7272182`.
Reviewed and implemented directly by one agent. No changes to camera URLs,
models, inference cadence, thresholds, recording policy, or driver versions.

## Findings fixed

| Priority | Finding and evidence | Correction |
| --- | --- | --- |
| High | A clean Ubuntu 24.04 installation failed: DL Streamer requires `libva2 >= 2.21`, while stock Noble offered `2.20.0-2ubuntu0.2`. The graphics PPA was only configured with `--intel-gpu`. | Always configure the graphics repository for the required runtime library. Keep the pinned compute/media driver installation conditional on `--intel-gpu`. |
| High | Starting or resuming cameras could repeatedly mask an established shared-pool stall. Their grace periods returned early even when another continuously active stream had exhausted its budget. Two regressions reproduced this. | Only actual inference completions establish pool progress. Startup/resume grace applies to that stream's eligibility, not to other stalled streams. |
| High | The rolling stderr buffer could begin or end inside a credential-bearing URL. Redacting that fragment cannot recognize the missing URL prefix or terminator. A synthetic long-password regression exposed the fragment in the new supervisor log. | Render only complete stderr lines, discard potentially truncated boundary fragments, redact before limiting displayed text, and apply the same policy to capture error details. No real credential exposure was established. |
| Medium | Recent inference was ignored when that camera's video had paused or when its first status had not arrived. Metadata can precede initial status. Both cases could incorrectly reset a working shared pool. | Count valid recent inference independently of video freshness and initial status. Retired inboxes still cannot establish progress; video-only streams cannot mask a stall. |
| Medium | The native preflight reported success without exercising the new `GstVa` introspection/display path, even when used after a GPU install. | Add `check-native-runtime.py --intel-gpu`, propagate it from the installer, verify display creation and render-device access, and document checking as the service user. |
| Medium | The decoder-selection fallback assumed missing factories return `None`. The clean no-GPU container instead raised `No such element: vah264dec`; a missing H.264 factory could prevent checking H.265. | Discover each factory before constructing it. Cover both available decoder choices and display-init failure cleanup. |

The five-second shared-pool timeout is unchanged. Per-camera snapshot freshness,
native fatal-error handling, and shared VA display ownership remain intact.
Partial stderr without a complete line is deliberately omitted rather than
exposing an unredactable fragment; structured native status errors remain
available independently.

## Validation

- Baseline affected suites: **72 passed, one skipped**.
- New watchdog, credential-boundary and decoder-fallback regressions failed
  against the prior implementation, then passed after the fixes.
- Final full Python suite: **2,675 passed; 272 subtests passed; one skipped** in
  85.70 seconds. One FastAPI/Starlette `httpx` deprecation warning remains.
  Run: `SURVNG_TEST_TIMEOUT_SECONDS=600 scripts/run-tests.sh -q`.
- Real Intel GPU context check passed twice. VA buffers from two independent
  pipelines referenced the same display; live PTS advanced through three main
  pipeline teardown/recreation cycles. The fixture uses small synthetic frames,
  no camera connections and no model inference. Run:
  `timeout --signal=TERM --kill-after=5s 30s /usr/bin/python3 scripts/gstreamer-va-context-check.py`.
- GPU preflight passed on the local Intel render device.
- Fresh native package installation and immediate repeat installation passed
  in a clean `ubuntu:24.04` container after the dependency fix. This exercised
  the native installer, not SurvNG's application Docker image. The repeat
  installed/upgraded no packages.
- Basic preflight passed as an unprivileged user in that installed environment.
  GPU preflight correctly exited 1 without a VA decoder/render device, with
  actionable setup guidance. Only a minimal public source bundle was mounted;
  no application configuration or camera credentials were supplied.
- The host's nested LXC environment could not load Docker's default AppArmor
  profile. The disposable installer container used `apparmor=unconfined`, with
  no privileged mode or host devices. The test container, derived image, and
  staging tree were removed after verification.
- Shell syntax and `git diff --check` passed.
- The reviewed fixes were loaded by restarting the local service. From
  16:33:09 to 16:36:11 UTC, all 13 live cameras retained their initial sessions,
  with sampled frame ages below one second and no shared-supervisor failure.
  Every camera produced new detections (161–187 snapshots each). A single
  `back-right/main` RTSP internal-stream error was logged during startup at
  16:32:06; it did not repeat during this observation. Its underlying cause
  was not established by this review.

## Limits

The clean environment test covers package provisioning and import/plugin
readiness, not an end-to-end installation with models, storage and real cameras.
The hardware fixture verifies VA context sharing and teardown, not model
accuracy or indefinite GPU stability. The broader application architecture and
new per-camera inference-recovery policies are outside this commit review.
