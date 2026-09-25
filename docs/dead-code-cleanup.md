# Dead-code cleanup campaign

This implements the five-phase campaign following [the original audit](dead-code-audit.md). The audit's line numbers and inventories describe the pre-cleanup source. Application changes were made sequentially, with focused tests and a service restart after each phase.

## Changes and phase validation

| Phase | Change | Validation |
| --- | --- | --- |
| 1: obvious dead code | Removed the orphaned 382-line identity UI, three unused local UI handlers, unused constants/calculations/imports, and unused destructured bindings. Preserved the state setter that forces admin baseline rerenders, exception handling, and side effects in test setup. Made `load_detector_labels` an explicit re-export. | 219 Python tests and 8 subtests; 68 frontend unit-test files; production build; authenticated desktop/mobile workspace smoke checks. |
| 2: unreferenced backend APIs | Removed the eight uncalled methods/properties/wrappers identified for this phase, the unused cancellation timestamp, and the write-only media-session ownership flag. Kept the underlying cancellation, depth sampling, cover-claim, and diagnostics-stream implementations. | 229 Python tests and 2 subtests; restart; authenticated desktop/mobile checks. |
| 3: obsolete test-only implementations | Retired the old recording-grid layout algorithm and its dedicated test/script, unused frontend helper APIs, obsolete depth-evidence/manual-merge helpers, and an unused fake provider. Kept tests for live shared helpers; zone geometry tests now exercise `insertZonePointWithIndex` directly, and action validation uses a representative server-authored payload. | 40 Python tests; all 67 remaining frontend unit-test files; build; Timeline browser interaction fixture; restart and live workspace checks. |
| 4: ownership and stale wiring | Removed the unused route-watch consumption callback chain and its manager wrapper. Retained durable target-admission consumption and tested that EMA admission alone does not consume watches. Removed the obsolete semantic reconciliation method; active reconciliation remains inside revision-checked `upsert`. | 315 Python tests and 2 subtests covering semantic refresh, motion admission, watch ownership, camera workers and manager lifecycle. |
| 4: discovered legacy-schema defect | Real browsing exposed an existing People review-queue HTTP 500: upgraded databases retained `body_embedding_blob`, and `SELECT o.*` exposed binary data to JSON serialization. The observation projection now removes this private legacy field alongside `embedding_blob`. No database column or stored data was deleted. | New real-SQLite/FastAPI regression failed with `PydanticSerializationError` before the fix; all 61 related face tests passed afterward. Authenticated live queue/workspace checks passed after restart. |
| 5: CSS | Removed 48 selectors for `.camera-command-area`, `.telemetry-section-tabs`, and `.recordings-v2-selected-event`/its image class across six stylesheets. Preserved other members of grouped selectors. Retired the source-text assertion for the removed card and updated the old browser expectation to current evidence selection. | 67 frontend unit-test files; production build; before/after browser CSS checks at 1440, 900 and 390px on Timeline, camera administration and telemetry; Incident Detail and Timeline browser fixtures. |

The real-browser smoke checks visit Live, Incidents, Timeline, People, and Admin at desktop and phone widths. Credentials/session state were held outside the repository. Startup snapshot HTTP 503s were observed while cameras reconnected; validation was repeated after warmup. Restarts also cancelled outstanding streaming requests at the configured graceful-shutdown timeout. These observations were not hidden by changing API security or error handling.

The CSS comparison removes candidate rules within a browser page, then compares computed properties for every DOM element synchronously. All nine viewport/workspace combinations had zero matching candidate elements and zero computed-style changes. The post-build check verifies that those selectors are absent. This does not claim coverage of every product state, so the other heuristic CSS candidates were retained.

## Deliberately retained candidates

Absence of a production caller alone was not sufficient to retire tested operational APIs or test seams:

- `HybridCandidateObjectTracker`, `bind_for_compatibility`, and `_index_event` preserve explicit compatibility contracts.
- `replay_ema_signal`, `qualify_motion`, `remember_frame`, `visual_backup_readiness`, `_best_match`, and `_is_motion_event` provide tested replay, reference, or focused entrypoints into live logic.
- `source_is_idle`, `wait_for_camera_startup`, `active_leases`, `accepting_events`, `episode_snapshot`, and `evidence_attempts` support lifecycle/diagnostic assertions.
- `cancel_admission`, `set_phase`, `reset_stages`, `settle_cover_requirement`, `update_objects`, and `record_operational_event` are tested lifecycle/storage interfaces. Their removal would retire contracts rather than simply remove unreferenced code.
- `recent_files`, `_reconcile_recording_source`, `latest_with_fallback`, and `recorded_frames` retain tested recording/decoding behavior. Whole-path retirement would require a separate compatibility decision.
- Intentional object-rest omissions (`_cameras`, `_zones`, `replay`), ignored catch bindings, serialized fields, framework callbacks, dynamic dispatch and route registrations remain intact.

## Final validation

- Full backend suite: **2,671 passed, 247 subtests passed**, in 102.07 seconds. One existing Starlette/httpx deprecation warning.
- Frontend unit suite: **67 test files passed**. Targeted assistant-message and workspace-lazy checks also passed after final import/comment cleanup.
- Production Vite build: passed. The existing large-chunk warning for the bundled video player remains.
- Python unused imports/redefinitions/locals (`F401,F811,F841`): passed. A separate undefined-name scan found 15 pre-existing Python annotation diagnostics, reproduced against the baseline source; none were introduced by this campaign.
- Frontend undefined-name scan: passed. The unused-binding scan reports only the eight intentional omission/catch bindings described above.
- Real-browser Incident Detail fixture: passed desktop/mobile rendering, playback controls, reconnect/completion, missing-image and expired-link checks.
- Real-browser Timeline fixture: passed playback, camera selection, pan/zoom/seek and related interaction checks.
- Authenticated live smoke: passed on desktop/mobile after each phase, with the identified legacy People defect resolved in phase 4. Final checks observed no JavaScript exceptions or HTTP 5xx responses.
- CSS browser checks: all nine combinations passed before and after deployment, with zero computed-style changes; all 48 retired selectors were absent afterward.
- Final owner-only runtime snapshot: all 13 cameras connected, detector ready. No generated assets, test artifacts, credentials or local configuration are included in the commit.

The entire browser test collection was not run: selected isolated fixtures and authenticated live checks covered the affected surfaces. The CSS harness used keyboard activation for the Timeline Nearby control because the existing assistant launcher intercepted a pointer click at the tested viewport. This pre-existing interaction is outside the deleted selector families. No claim is made that every historical CSS candidate or tested compatibility interface is dead.

## Post-commit review

The cleanup was committed as `2eca7be`, then reviewed against its parent. The review rechecked removed backend definitions and the Vite module graph: no unexpected residual callers or orphaned frontend modules were found.

The review found that the revised historical release browser test could select evidence while Nearby was closed. The follow-up opens the panel explicitly, scopes the button to Nearby evidence, and waits for selection to be persisted in the URL. Its exact updated interaction block passed in an authenticated real browser against a real incident, starting with Nearby closed. JavaScript syntax and whitespace checks passed. The complete historical release script was not run. This test/documentation-only follow-up does not change the deployed application.

That focused browser check also confirmed a pre-existing Timeline limitation: live incident IDs can be nonnumeric, while selected-event lookup/highlighting coerces them with `Number`. The selection is recorded in the URL, but the pressed highlight can remain absent. The numeric comparisons were verified in the parent commit; this is retained as a separate functional defect, not attributed to the cleanup. The follow-up diff was reviewed again with no further cleanup-introduced findings.
