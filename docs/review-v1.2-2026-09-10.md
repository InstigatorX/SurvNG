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

## Timeline usability follow-up (iPhone/Safari target)

- Start at least one second inside the latest available recording, and reconcile the overview target with the finalized window's actual media bounds.
- Pause outgoing footage as soon as a seek starts. Keep camera/day transitions pending through index loading so outgoing time updates and end events cannot overwrite the selected time or complete the replacement seek.
- Preserve the HLS video element across camera changes and carry playback intent into the selected child camera. Empty native HLS sources release the old media without requesting the page as a video.
- Reuse up to eight recent recording windows (30-second archive lifetime, two seconds near live); Retry clears the cache. This avoids repeated metadata requests when revisiting footage; no production page-load timing improvement is claimed.
- Add a visible mobile playhead handle: dragging near it seeks at one second per pixel; swiping elsewhere pans the range. Center the mobile native date field and remove its extra internal spacing.
- Clear the seeking indicator when window loading fails.

Validation: 66 frontend unit test files passed. Chromium and WebKit full Timeline fixtures passed, including delayed window loading, outgoing end events, cached revisits, and linked-camera autoplay at the retained time. Chromium additionally exercised real touch dragging of the fine scrubber; WebKit mobile layout was visually checked. Native MP4 and HLS lifecycle coverage was rerun. These automated engines do not establish physical iPhone autoplay policy or camera-specific codec behavior.

## Camera-switch latency: independent expert review and implementation

An independent specialist reproduced 9.768 seconds of MP4-header reads across 89 Upper Garage segments; the primary investigation measured 17 seconds on another cold window versus 23 ms for its index query. Exact video duration differs from the index's availability estimate, so skipping header resolution without retaining that duration would corrupt HLS offsets.

The implementation adds one nullable `playback_duration_seconds` column to the existing index. The existing metadata worker receives only recent finalized recordings; legacy windows populate metadata on demand. A stable size/mtime identity is checked before reuse and before conditional persistence. Replaced files invalidate derived metadata. No archive-wide backfill, new service, or new worker pool is introduced.

An isolated test using the actual implementation and 90 production recording paths, with a temporary index, reopened the index and resolved identical metadata in 1.4 ms with zero header reads. This measures metadata preparation, not end-to-end iPhone playback latency. A first request for legacy metadata can still require header reads.

The new WebKit test failed before the frontend fix: a camera switch emitted a new-camera playlist before its own window loaded. Scoping HLS and native playback details to camera/source/day removes this premature request. The independent final diff review found no remaining blockers.

Final validation: 2,407 Python tests and 233 subtests passed; 18 optional-dependency tests skipped, with the existing Starlette/httpx deprecation warning. All 66 frontend unit files, Chromium/WebKit Timeline fixtures, production frontend build, and diff whitespace checks passed. Independent review used one GPT-5.6-Sol specialist for theory validation and read-only diff review; implementation and integration remained with the primary agent.
