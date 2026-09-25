# Dead-code audit

Audited commit: `f29de45`. This is the original report-only audit, preserved as a historical inventory. See [the cleanup campaign](dead-code-cleanup.md) for subsequent removals, retained candidates, and validation.

## Scope and method

- Inventoried all 698 tracked files. Parsed all 386 Python files and all 180 frontend JS/JSX/MJS files; scanned all 18 frontend stylesheets.
- Checked Python imports, definitions, symbol references, unused bindings, and unreachable-code candidates with AST analysis, Ruff (`F401,F811,F841`), and Vulture. References included tests, scripts, documentation, configuration, and deployment files.
- Checked frontend imports, dynamic imports, exports, local bindings, and unreachable statements using an AST import graph and ESLint (`no-unused-vars`, `no-unreachable`, React JSX usage rules).
- Built all four Vite entrypoints with `build.write=false` and inspected the resulting module graph. The production build passed: 1,976 modules transformed; 110 source modules represented, including stylesheets. Of 93 JS/JSX/MJS source files, only `identityWorkflow.js` was absent.
- Reviewed framework registration, string-based dispatch, compatibility shims, test-only callers, and selected CSS families before classifying findings.
- Dependencies and temporary audit scripts were installed/written under `/tmp`, outside the repository. Existing production assets were not rewritten.

Generated assets, virtual environments, dependencies, caches, model files, recordings, local configuration/secrets, and crash dumps were excluded from source analysis. Standalone command-line scripts are entrypoints, not dead simply because application code does not import them. No clearly orphaned backend implementation module was established; the exceptional import-graph results were entrypoints, package initializers, and a tested compatibility alias.

Static absence of callers establishes non-use within this checkout, not whether an external consumer imports a public Python name. No running service was modified or exercised. Unit/browser suites were not run because this audit changed no behavior. CSS coverage across runtime states was not collected. These limitations prevent treating the candidate lists as an automatic deletion manifest.

## Highest-confidence findings

| Location | Finding and evidence | Suggested cleanup |
| --- | --- | --- |
| `frontend/src/identityWorkflow.js:1` | Entire 382-line legacy people interface is orphaned. No tracked import, HTML script tag, server injection, or build entrypoint references it. Vite excludes it. It contains its own DOM renderer, CSS injection, listeners, and interval. The active people page is the lazy-loaded React `FacesPage` in `App.jsx:42`. | Remove the whole file. Its interval is not currently running; this is source cleanup, not a demonstrated runtime performance fix. |
| `frontend/src/admin/ConfigPage.jsx:199` | Local `updateSettings` handler has no references. | Remove the handler. |
| `frontend/src/live/LivePage.jsx:1501` | Local `toggleIncident` handler has no references. | Remove the handler; keep surrounding incident state and overlay callbacks, which have other users. |
| `frontend/src/timeline/TimelinePages.jsx:1884` | Local `selectTrailHit` handler has no references. Its roughly 60-line fetch/navigation path cannot be entered through the current component. | Remove the handler, or explicitly restore its intended UI connection if this represents missing functionality. Review dependencies exposed by its removal. |
| `survng/app/main.py:626` | `normalize_source` occurs only at its definition. | Remove the unused helper. |
| `survng/app/audit_ai.py:182` | `gemini_advice_schema` occurs only at its definition. Active provider paths use `ADVICE_SCHEMA` directly. | Remove the wrapper; retain the schema and other users of `copy`. |
| `survng/app/semantic_search.py:572` | `weakest_required` is computed here and assigned again at line 594 but never read. | Remove both assignments, including the unnecessary NumPy reduction. |
| `survng/app/inference_lifecycle.py:279` | A deep copy of `config.tracking` is assigned to `tracking` but never read. Limiter/factory construction uses `config`. | Remove the unused copy. |
| `survng/app/face_store/benchmarks.py:78` | `per_identity` is initialized but never read. | Remove the unused local. |

## Unreferenced backend methods and properties

These have no discovered callers, including exact string references and tests. They are strong internal cleanup candidates, but method removal changes the exposed Python surface.

| Location | Symbol | Evidence / boundary |
| --- | --- | --- |
| `survng/app/depth_estimation.py:320` | `estimate_object_depth_stats` | Definition only. Keep the shared depth sampling functions and the separate active estimation methods. |
| `survng/app/event_store/evidence.py:155` | `pending_cover_requirements` | Definition only. Refinement uses `claim_cover_requirement`, including a dynamic lookup in `motion_incidents.py`. |
| `survng/app/media_sessions.py:110` | `MediaCancellation.cancelled_at` | Property has no readers. Review its backing timestamp separately before removing related writes. |
| `survng/app/media_sessions.py:294` | `cancel_all` | Definition only. Do not remove cancellation primitives used by other manager methods. |
| `survng/app/motion_analysis_service.py:860` | `visual_backup_stable_samples` | Property has no readers. The neighboring readiness property remains used. |
| `survng/app/motion_events.py:458` | `fail_deliveries` | Definition only. `fail_motion_trigger` is still called by the retry path and must remain. |
| `survng/app/semantic_search.py:761` | `reconcile_event_source_keys` | Definition only. It describes stale-evidence cleanup; determine whether the intended behavior should be connected rather than deleting it as merely obsolete. |
| `survng/app/runtime_monitor.py:422` | `RuntimeMonitor.export_diagnostics` | No callers of this method. The identically named HTTP handler is live and calls `export_diagnostics_stream` instead. |
| `survng/app/main.py:783` | `_manager_owned_config` | Compatibility wrapper with no current callers, despite its test-compatibility docstring. Tests import `config_application.manager_owned_config` directly. Retire deliberately if the old import path is no longer supported. |

## Dead constants and write-only state

| Location | Candidate | Evidence / qualification |
| --- | --- | --- |
| `survng/app/product_update.py:22` | `UPDATE_TIMEOUT_SECONDS` | Definition only; operation-specific timeout constants are used instead. |
| `survng/app/motion_pipeline/object_detection.py:46` | `RECORDED_EVENT_FRAME_OFFSETS` | Definition only. Keep the stage tuples used by actual sampling. |
| `survng/app/object_activity.py:144` | `STABLE_DISPLACEMENT_RATIO`, `STABLE_PATH_RATIO`, `CONTEXT_MEMORY_MIN_IOU`, `CONTEXT_MEMORY_MIN_PRIOR_SIGHTINGS` | Each occurs only at its definition, at lines 144, 145, 148, and 149. Other context-memory constants remain active. |
| `survng/app/manager.py:275` | `_owns_media_sessions` | Assigned once, never read. Its name implies ownership policy; removing the assignment does not establish whether existing shutdown behavior is correct. |
| `survng/app/motion_analysis_service.py:208` | `_route_watch_consumer` | Initialized and assigned at line 320 but never read. The supplied `consume_route_watch` callback is therefore unused by this service. Review intended watch-consumption semantics before deleting the public parameter/camera wiring. |
| `frontend/src/browserAppearance.mjs:3` | `BROWSER_APPEARANCE_THEME_KEY` | No readers/importers. Other appearance helpers remain active. |
| `frontend/src/shared/constants.js:35` | `STREAM_MODES`, `STREAM_LABELS` | No readers/importers. |
| `frontend/src/detectionOccupancy.mjs:27` | `asPercent`, `isEmaTriggerSource` | No callers/importers. `EMA_TRIGGER_SOURCES` at line 8 is used only by the latter, making this a removable three-symbol cluster. |

## Code exercised only by tests

These are not unexecuted code: tests use them. They have no discovered production caller and warrant an explicit keep/remove decision. Delete associated tests only if retiring the tested surface, not merely to make an unused-code tool quiet.

The largest frontend example is `frontend/src/recordingGrid.mjs:120`: `recordingGridLayout` is imported only by `recording-grid.mjs` and `live-grid-layout.mjs` tests. Its private helpers (`partitionRows`, `portraitLayoutScale`, `rectanglesOverlap`, `packSpanningGrid`) form the same production-unused cluster, approximately lines 11–184. Keep `recordingCameraAspect` and `recordingGridBestEpoch`, which the timeline imports.

Other frontend functions with only test callers and no same-module callers:

| File | Symbols |
| --- | --- |
| `frontend/src/assistant/assistantTuneLoop.mjs:32` | `buildStartCameraReviewAction` |
| `frontend/src/assistantMessage.mjs:13` | `splitAssistantCitations`, `assistantEvidenceLabel` (line 28); `CITATION_PATTERN` is used only by the former |
| `frontend/src/incidentNavigation.mjs:247` | `incidentThumbnailObjectFocusEnabled`, `incidentObjectFocusThumbnailWidth` (line 359) |
| `frontend/src/peopleWorkspace.mjs:45` | `peopleFilterLabel`, `peopleModeLabel` |
| `frontend/src/shared/mediaUrls.js:99` | `recordingMobileSegmentUrl`, `recordingMobileWindowUrl` |
| `frontend/src/timelineWorkspace.mjs:45` | `timelineCompanionGrid`, `timelineViewportPage` (line 167) |
| `frontend/src/visualSearchTrail.mjs:23` | `parseTrailEventIds`, `trailPosition` |
| `frontend/src/workspaceNavigation.mjs:109` | `systemHealthState` |
| `frontend/src/zoneGeometry.mjs:48` | `insertZonePoint`; production uses `insertZonePointWithIndex` |

`resetExportPollingCacheForTests` is deliberately test-only and is not a cleanup finding. Likewise, `HybridCandidateObjectTracker` is a documented, tested compatibility alias and `replay_ema_signal` is a test replay utility; neither should be classified as accidentally dead.

## CSS candidates requiring visual verification

The lexical selector scan found 283 rule blocks containing at least one class not literally present in frontend executable source/HTML. This is a noisy upper bound, not 283 confirmed dead rules. Dynamic class construction such as page/workspace/mode names creates false positives, and grouped selectors can contain both live and obsolete members.

Concrete families with no matching producer found include:

- `.telemetry-section-tabs` in `frontend/src/admin/admin.css:276` and responsive rules in `admin/workspace.css`.
- `.camera-command-area` in `frontend/src/admin/admin.css:82` and `admin/workspace.css`; the current component uses `.camera-command-bar`.
- `.recordings-v2-selected-event` and its descendants in `frontend/src/timeline/timeline.css:1050`, `shell/mobile.css`, and `shell/responsive.css`.

Verify desktop/mobile layouts, error/loading/empty states, and dynamic class producers before removing these. Do not delete entire comma-separated rules just because one selector is obsolete.

## False positives explicitly excluded

- `survng/app/inference_runtime/process.py:14` imports `load_detector_labels` without using it locally, but **other production modules and tests import it from this module**. Ruff reports F401; deleting this import would break those imports. Keep or make the re-export explicit.
- FastAPI decorators and handler dictionaries register functions that need no direct call by Python name. Vulture's route findings are not evidence that the endpoints are dead.
- Pydantic validators and serialized model/dataclass fields can be consumed through framework behavior or API payloads.
- `JsonGZipResponder.send_with_compression`, urllib's `redirect_request`, and Ultralytics tracker overrides are framework hooks.
- `InferenceSupervisor.detect_enrichment` is selected by string in `object_track/geometry.py:50`.
- ONVIF timestamps and retry counters are read through the attribute-name list in `camera_status.py:255`, despite Vulture reporting writes without reads.
- Callback/protocol signature arguments, context-manager exception arguments, ignored catch bindings, and `_cameras`/`_zones` destructuring used to omit fields are not dead implementations.
- An export without external importers can still be used inside its own file. Examples include `shouldSkipWebRtc`, `streamUrlDefaults`, `RecordingFallback`, and `incidentIndexForEvent`.

## Validation and cleanup order

The audit tools completed successfully. Ruff reported 22 findings, including the live re-export above. ESLint reported 46 unused-binding findings and no unreachable-statement findings. Vulture's high-confidence output consisted of unused signature arguments, not proven dead blocks. The production Vite build passed with writes disabled.

Recommended first change: remove the orphaned identity UI, the three unreachable local UI handlers, unused constants, and obvious unused calculations/import bindings. Then separately decide which test-only and compatibility surfaces to retire. Investigate the unused route-watch callback and semantic reconciliation method as possible missing wiring before removing them. CSS cleanup should be a separate change with browser validation.

No application tests or runtime checks were run, and no behavior-preservation claim is made for changes that have not yet been implemented.

## Appendix: unused Python bindings

Locations below are Ruff findings, not automatic deletion instructions. Remove unused bindings without removing side effects from their expressions. Exception aliases can be removed while preserving exception handling.

| Location | Finding | Disposition |
| --- | --- | --- |
| `survng/app/activity_events.py:7` | `time` imported but unused | Unused binding in this scope. |
| `survng/app/camera.py:49` | `.video_frames.DecodedVideoFrame` imported but unused | Unused binding in this scope. |
| `survng/app/event_store/store.py:13` | `..durable_payload.durable_json_dumps` imported but unused | Unused binding in this scope. |
| `survng/app/face_store/benchmarks.py:78` | Local variable `per_identity` is assigned to but never used | Unused binding in this scope. |
| `survng/app/face_store/unknown.py:14` | `.quality.LOGGER` imported but unused | Unused binding in this scope. |
| `survng/app/inference_lifecycle.py:279` | Local variable `tracking` is assigned to but never used | Unused binding in this scope. |
| `survng/app/inference_runtime/adapters.py:9` | `.types.InferenceUnavailable` imported but unused | Unused binding in this scope. |
| `survng/app/inference_runtime/process.py:14` | `..detector_labels.load_detector_labels` imported but unused | Keep: live re-export, described above. |
| `survng/app/inference_runtime/worker.py:317` | Local variable `exc` is assigned to but never used | Unused binding in this scope. |
| `survng/app/inference_runtime/worker.py:659` | Local variable `exc` is assigned to but never used | Unused binding in this scope. |
| `survng/app/media_storage.py:17` | `.config.MediaStorageLocationConfig` imported but unused | Unused binding in this scope. |
| `survng/app/motion_ingress.py:12` | `.motion_decisions.priority_motion_topic` imported but unused | Unused binding in this scope. |
| `survng/app/object_track/bytetrack.py:11` | `.types.ObjectTrackerBackend` imported but unused | Unused binding in this scope. |
| `survng/app/semantic_search.py:594` | Local variable `weakest_required` is assigned to but never used | Unused binding in this scope. |
| `survng/app/tracking_comparison.py:13` | `collections.defaultdict` imported but unused | Unused binding in this scope. |
| `tests/test_adaptive_motion.py:990` | Local variable `first` is assigned to but never used | Unused binding in this scope. |
| `tests/test_camera_worker.py:15` | `cv2` imported but unused | Unused binding in this scope. |
| `tests/test_cover_recovery_pipeline.py:5` | `types.SimpleNamespace` imported but unused | Unused binding in this scope. |
| `tests/test_event_store.py:1583` | Local variable `object_event` is assigned to but never used | Unused binding in this scope. |
| `tests/test_face_reference_retention.py:4` | `pathlib.Path` imported but unused | Unused binding in this scope. |
| `tests/test_recorded_frame_integration.py:2` | `pathlib.Path` imported but unused | Unused binding in this scope. |
| `tests/test_semantic_refresh.py:4` | `pathlib.Path` imported but unused | Unused binding in this scope. |

## Appendix: frontend unused bindings

ESLint findings after excluding five intentionally ignored catch bindings and two object-rest omission bindings. The `person` finding belongs to the entirely orphaned identity module. Unused React state destructuring does not imply the hook or underlying state is unused.

| Location | Finding |
| --- | --- |
| `frontend/src/admin/ConfigPage.jsx:31` | 'Pause' is defined but never used. |
| `frontend/src/admin/ConfigPage.jsx:199` | 'updateSettings' is defined but never used. |
| `frontend/src/admin/ConfigPage.jsx:1138` | 'baselineRevision' is assigned a value but never used. |
| `frontend/src/admin/ConfigPage.jsx:1668` | 'runtimeStatusById' is assigned a value but never used. |
| `frontend/src/admin/ConfigPage.jsx:3847` | 'coremlLabel' is assigned a value but never used. |
| `frontend/src/admin/ConfigPage.jsx:3852` | 'gpuLabel' is assigned a value but never used. |
| `frontend/src/admin/ConfigPage.jsx:3867` | 'vaapiLabel' is assigned a value but never used. |
| `frontend/src/admin/ConfigPage.jsx:3872` | 'qsvLabel' is assigned a value but never used. |
| `frontend/src/admin/cameraEditors.jsx:3` | 'Camera' is defined but never used. |
| `frontend/src/admin/cameraEditors.jsx:4` | 'Copy' is defined but never used. |
| `frontend/src/detectionOccupancy.mjs:643` | 'restartCount' is assigned a value but never used. |
| `frontend/src/identityWorkflow.js:218` | 'person' is assigned a value but never used. |
| `frontend/src/incidents/IncidentCard.jsx:44` | 'eventObjects' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:3` | 'ArrowLeft' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:4` | 'ArrowRight' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:5` | 'Camera' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:21` | 'formatDateTime' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:27` | 'incidentLabels' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:27` | 'IncidentObjectBadges' is defined but never used. |
| `frontend/src/incidents/IncidentsPage.jsx:66` | 'incidentSelectionRequestRef' is assigned a value but never used. |
| `frontend/src/live/LivePage.jsx:6` | 'Activity' is defined but never used. |
| `frontend/src/live/LivePage.jsx:7` | 'ArrowLeft' is defined but never used. |
| `frontend/src/live/LivePage.jsx:8` | 'ArrowRight' is defined but never used. |
| `frontend/src/live/LivePage.jsx:845` | 'setSelectedEvent' is assigned a value but never used. |
| `frontend/src/live/LivePage.jsx:1501` | 'toggleIncident' is defined but never used. |
| `frontend/src/shared/evidence.jsx:5` | 'ArrowLeft' is defined but never used. |
| `frontend/src/shared/evidence.jsx:6` | 'ArrowRight' is defined but never used. |
| `frontend/src/shared/evidence.jsx:26` | 'Video' is defined but never used. |
| `frontend/src/timeline/TimelinePages.jsx:3` | 'ArrowLeft' is defined but never used. |
| `frontend/src/timeline/TimelinePages.jsx:4` | 'ArrowRight' is defined but never used. |
| `frontend/src/timeline/TimelinePages.jsx:16` | 'Grid2X2' is defined but never used. |
| `frontend/src/timeline/TimelinePages.jsx:32` | 'Video' is defined but never used. |
| `frontend/src/timeline/TimelinePages.jsx:840` | 'daySeconds' is assigned a value but never used. |
| `frontend/src/timeline/TimelinePages.jsx:947` | 'frameSearchTrailIds' is assigned a value but never used. |
| `frontend/src/timeline/TimelinePages.jsx:1124` | 'selectedEventDuration' is assigned a value but never used. |
| `frontend/src/timeline/TimelinePages.jsx:1127` | 'selectedEventConfidence' is assigned a value but never used. |
| `frontend/src/timeline/TimelinePages.jsx:1884` | 'selectTrailHit' is defined but never used. |
| `frontend/src/trackingComparison.mjs:36` | 'replay' is assigned a value but never used. |
| `frontend/tests/timeline-interactions.mjs:3` | 'readFileSync' is defined but never used. |

## Appendix: backend test-only candidate inventory

Each name below appears once in non-test tracked text (its definition), with additional test references. This is a conservative exact-name scan, not a whole-program reachability proof. Test utilities, observability/test seams, compatibility APIs, and wrappers can be intentional; review before deletion. Shared lower-level implementations must remain when other production callers use them.

| Location | Symbol |
| --- | --- |
| `survng/app/camera_capture.py:948` | `source_is_idle` |
| `survng/app/camera_fleet.py:269` | `cancel_admission` |
| `survng/app/depth_estimation.py:134` | `depth_motion_evidence_values` |
| `survng/app/detector.py:84` | `merge_manual_detection_objects` |
| `survng/app/ema_v2.py:1138` | `replay_ema_signal` |
| `survng/app/event_store/evidence.py:231` | `settle_cover_requirement` |
| `survng/app/event_store/evidence.py:269` | `evidence_attempts` |
| `survng/app/event_store/store.py:804` | `update_objects` |
| `survng/app/face_store/recognition.py:844` | `_best_match` |
| `survng/app/manager.py:929` | `wait_for_camera_startup` |
| `survng/app/manager_access.py:65` | `active_leases` |
| `survng/app/media_sessions.py:178` | `set_phase` |
| `survng/app/motion.py:306` | `qualify_motion` |
| `survng/app/motion_analysis_service.py:456` | `remember_frame` |
| `survng/app/motion_analysis_service.py:840` | `visual_backup_readiness` |
| `survng/app/motion_events.py:726` | `episode_snapshot` |
| `survng/app/motion_pipeline/runtime.py:36` | `reset_stages` |
| `survng/app/motion_runtime.py:121` | `accepting_events` |
| `survng/app/object_track/candidate.py:8` | `HybridCandidateObjectTracker` |
| `survng/app/object_tracking_lifecycle.py:61` | `bind_for_compatibility` |
| `survng/app/onvif_events.py:1117` | `_is_motion_event` |
| `survng/app/recording_process/index.py:340` | `recent_files` |
| `survng/app/recording_process/index.py:992` | `_reconcile_recording_source` |
| `survng/app/semantic_search.py:1835` | `_index_event` |
| `survng/app/telemetry_store.py:579` | `record_operational_event` |
| `survng/app/tracking_frames.py:158` | `latest_with_fallback` |
| `survng/app/tracking_frames.py:386` | `recorded_frames` |

Vulture also identified `tests/test_cover_recovery_pipeline.py:108` (`FakeProvider`) as an unused test helper. This is a lower-priority candidate, separate from production cleanup.
