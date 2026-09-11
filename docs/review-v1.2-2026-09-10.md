# v1.2 seven-day review campaign

Review baseline: `99ae17f..76b8891`, selected from the seven-day Git history at campaign start. The change set spans 223 files. This was a risk-based subsystem review with regression testing, not a claim that every changed line or hardware combination is defect-free.

## Scope

Reviewed changed execution paths in recording/index/playback, storage retention and exports, face identity and snapshot ownership, motion admission and durable refinement, tracking and inference, deployment/access control, and camera/Timeline/incident UI. The complete Python suite and frontend unit suite provided broader regression coverage.

## Findings fixed

| Priority | Finding | Resolution and evidence |
| --- | --- | --- |
| P1 | Motion admission helpers ignored `FOLLOWUP_RESERVED`. After an EMA follow-up failed admission, a replacement camera request or retained fallback could remain reserved without being queued. | Accept both reservation kinds in camera ingress and fallback helpers. Regression coverage includes an actual controller sequence and both replacement-camera decision kinds. The replacement test failed before the fix. |
| P2 | Native MP4 seek completion accepted a delayed event from an earlier scrub on the same video. | Require the video to finish seeking and reach the current target before clearing pending state. The new regression failed before the fix and passes afterward. |
| P2 | Tracking tests imported an installed third-party `tests` package, preventing Python collection. | Make the repository tests a package and qualify the face-test helper imports consistently. Full collection and execution now pass. |
| P2 | Lifespan tests attempted to bind the running service's observability socket. | Give each test a temporary private socket directory while exercising the real server lifecycle. Both previously failing lifespan tests pass. |
| P2 | Timeline interaction checks treated unchanged hourly label text as evidence that a half-hour pan failed. | Include tick positions in fixture snapshots. The full Timeline browser fixture passes. |
| P2 | The automated MP4 fixture ignored Range requests, unlike the production FileResponse endpoint; Chromium reset a paused seek to zero. | Return byte-range responses from the fixture. The existing paused-seek and playback-handoff assertions pass. |

Concurrent edits to `survng/app/recording_retention.py` and `tests/test_recording_retention.py` were preserved. They move the deletion budget after candidate selection; they were included in the passing final suite, but were not authored by this campaign.

## Final validation

- Full Python suite: **2,393 passed, 18 skipped**. The skips require optional `ultralytics==8.4.129` and LAP. One existing Starlette/httpx deprecation warning remains.
- Frontend unit suite: **65 test files passed**.
- HLS browser fixture: passed using its portable H.264 fixture.
- Native MP4 browser fixture: passed, including 4× handoff, retained frames, paused seeking, superseded sources, stale events, errors, and standby audio.
- Full Timeline browser fixture: passed, including mobile date/camera controls, scale changes, panning and interaction checks.
- Frontend production build: passed; bundle-size advisory remains.
- Final diff whitespace check: passed.

The general browser suite stopped at `assistant-apply-modal.mjs` because the expected camera tile was not visible on its configured application. Remaining tests in that general runner were not executed. Physical iOS authorization, hardware-specific inference, HEVC/mixed-codec browser playback, and deployment execution were not established by these checks.

The validation results above were captured before the user-requested commit and service restart.
