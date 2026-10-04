# Dead-code and duplication audit — 2026-10-04

Audited commit: `99cc078d38227b6c346924ecf2e11d48f4ac7dd5` (`v1.2`).
The plan was prepared at `9f0e02e`; the branch advanced by one cleanly merged
face-recognition fix before execution, so the six changed files and three new
tests in that merge are included here. The September reports
[`dead-code-audit.md`](dead-code-audit.md) and
[`dead-code-cleanup.md`](dead-code-cleanup.md) remain historical records and
were not rewritten.

## Executive summary

The repository contains 786 tracked files: 216 production Python files, 219
Python test files, 122 frontend source files, and 95 frontend test/fixture files.
The audit parsed or scanned all tracked source, configuration, documentation,
deployment, and operational files. Dependencies, virtual environments, caches,
generated static output, models, recordings, and runtime databases were excluded
as cleanup targets, but tracked code that reads or generates them was included
in reference tracing.

The current tree has no orphaned production Python module. All frontend source
modules and styles are in the four-entry Vite graph except
`frontend/src/cameraSemantics.mjs`, a tested but production-unreferenced module.
Static checks found a small, conclusive batch of unused imports and locals. It
also found four uncalled public helpers, one test-only frontend module, stale-looking
CSS families, and several unused component parameters; those are retained because
external, intended-future, compatibility, or visual-state use cannot be ruled out.

Clone detection found 48 exact blocks (785 lines, 0.50% of the scanned source).
Review found no safe consolidation: the blocks either express small local UI
behavior or sit on different lifecycle, model, transaction, storage, protocol,
or ownership boundaries. Introducing shared abstractions would add coupling or
erase intentional differences.

| Classification | Findings | Estimated reduction | Disposition |
| --- | ---: | ---: | --- |
| A. Proven safe removal | 16 unused bindings | 16 source lines | One low-risk batch selected |
| B. Likely dead, requires confirmation | 5 code surfaces, 4 prop groups, CSS upper bound | about 50 code lines plus unverified CSS | Retain |
| C. Safe duplicate consolidation | 0 | 0 | None justified |
| D. Similar but intentionally distinct | 48 clone blocks in 19 families | 0 | Retain |
| E. Obsolete or suspicious, but blocked | 4 compatibility/migration families | potentially large | Requires policy or migration decision |
| F. False positive | Framework, dynamic, test-seam, asset, and operational families | 0 | Retain |

## Baseline and tools

The worktree was clean before the report and no service was restarted or deployed.

| Check | Pre-change result |
| --- | --- |
| `scripts/run-tests.sh` | **3,214 passed**, one existing Starlette/httpx deprecation warning, 125.20s |
| `node frontend/tests/run-tests.mjs unit` | **70 test files passed** |
| Vite `build({build:{write:false}})` | **Passed**, 1,987 modules transformed |
| `python -m survng.app --help` | Passed; server CLI and observability option discovered |
| `./survngctl --help` | Passed; `status` command discovered |
| Git status | Clean at baseline |

Audit-only tools were installed under `/tmp`, not added to project dependencies:
Ruff 0.16.10, Vulture 2.16, ESLint 9.39.5 with `eslint-plugin-react`
7.37.5, and jscpd 5.4.0. Ruff checked `F401`, `F811`, and `F841`.
Vulture was run at 80% and 60% confidence. ESLint checked unused bindings,
unreachable statements, and JSX references. jscpd used 10 lines/70 tokens.
Custom AST scans checked Python imports, module reachability, module constants,
definitions, config fields, and exact identifier references. Vite's in-memory
Rollup output supplied the production frontend module graph.

## Architecture and entry points

- The server entry points are `python -m survng.app` and
  `uvicorn survng.app.main:app`. `survng.app.main` is the composition root and
  assembles 20 focused router factories containing 188 decorated FastAPI handlers.
- `survngctl` / `python -m survng` is the owner-only operational CLI. Its current
  public command is the read-only Unix-socket `status` snapshot.
- The Vite build has four HTML entries: the main, recordings, and config shells
  all load `App.jsx`; ONVIF loads its own `onvif/main.jsx`. React workspace pages
  are lazy imports selected by `App.jsx` and workspace navigation.
- Long-running owners include the manager generation, camera fleet, capture,
  recording, inference, motion/refinement, MQTT, appearance/semantic backfills,
  exports, retention/storage maintenance, telemetry, and product-update workers.
- Dynamic selection is material: motion stages and object trackers use registries;
  detector roles and optional store methods use string/getattr dispatch; FastAPI,
  Pydantic validators, MQTT/ONVIF callbacks, codec adapters, and tracker overrides
  are framework-called.
- SQLite schemas and in-place migrations are owned by the event, face, semantic,
  appearance, telemetry, visit, calibration, tracking, training, and operational
  stores. Serialized configuration and retained event JSON are compatibility
  boundaries, not internal implementation details.
- Operational entry points include 16 standalone scripts, Compose variants,
  Docker entrypoints/health checks/model installation, the systemd unit, three CI
  workflows, and the Node-RED incident-notification flow. A lack of Python imports
  is not evidence that these files are unused.

## A. Proven safe removal

All searches below covered production, tests, scripts, deployment files,
configuration, docs, and Git history. None of these names is registered by
FastAPI, a registry, configuration, reflection, serialization, or an operational
entry point. Import removals have no import-time side effect because the containing
modules still import and execute their real dependencies. Assignment removals
discard only already-available values and remove no call.

| Location | What it does and evidence | Hidden-consumer review | Risk / confidence | Coverage and exact action |
| --- | --- | --- | --- | --- |
| `survng/app/event_store/scene_admission.py:9` | `json` import; Ruff F401 and exact-name scan find no use in the module. | Not a re-export; module is imported through the event-store mixin. | Low / high | Scene acquisition/admission suites; remove import. |
| `survng/app/event_store/scene_admission.py:12` | `_iso` import; only `_epoch`, `_json`, and `_objects` are read. | Private sibling helper, no export obligation. | Low / high | Scene suites; remove imported name. |
| `survng/app/event_store/scene_admission.py:132` | `camera_notice` receives a pure boolean comparison and is never read. | No callback, attribute write, or expression side effect. | Low / high | Scene establishment/incidents; remove assignment. |
| `survng/app/event_store/scene_review.py:4` | `json` import; no module use. | Not re-exported. | Low / high | Observation-review tests; remove import. |
| `survng/app/event_store/scenes.py:14` | `typing.Any` import; no annotation or runtime use. | Not re-exported. | Low / high | Scene and incident suites; remove import. |
| `survng/app/incident_queries.py:8` | `re` import; no regex use in the module. | Not re-exported. | Low / high | Incident query/API suites; remove import. |
| `survng/app/incident_queries.py:27` | Private `_incident_row` import; `_event_row` and `_incident_list_payload` are used, this name is not. | Private presenter helper and not a compatibility export from this module. | Low / high | Incident query/API suites; remove imported name. |
| `survng/app/motion_pipeline/object_detection.py:25` | `detection_failure` import; exact search finds no read. | Not re-exported and detector module remains imported elsewhere. | Low / high | Recorded detection and motion pipeline suites; remove import. |
| `survng/app/recording_routes.py:857` | `source_path = path`; `source_path` occurs only here. The route continues to use `path` and the lease token. | Local inside a registered route; removal leaves lease/release and mobile transcode behavior intact. | Low / high | Recording routes/API/fMP4 suites; remove assignment. |
| `tests/test_incident_queries.py:9` | Unused `datetime`, `timedelta`, and `timezone` imports. | Test-local standard-library names. | Low / high | Run file; remove names. |
| `tests/test_scene_activity.py:12` | Unused `resolve_recorded_refinement_plan` import. | Test does not rely on importing it for registration; object-detection is imported elsewhere in the same test surface. | Low / high | Run file; remove import. |
| `tests/test_scene_establishment.py:4` | Unused `SimpleNamespace` import. | Test-local standard-library name. | Low / high | Run file; remove import. |
| `frontend/src/admin/ConfigPage.jsx:7` | Unused `ArrowLeft` icon import; ESLint and exact JSX search agree. | Static ES import from icon package, not dynamically addressed. | Low / high | Frontend unit suite and Vite build; remove imported name. |
| `frontend/src/incidents/IncidentCard.jsx:14` | Unused `Search` icon import; ESLint and exact JSX search agree. | Static ES import, not a component registry entry. | Low / high | Incident frontend tests and Vite build; remove imported name. |

These 16 binding removals form the one selected cleanup batch. No new
characterization test is needed because the values cannot affect behavior and the
owning route/component/subsystem suites already exercise the containing code.

## B. Likely dead, requires confirmation

These items have no discovered production caller, but their public/tested shape or
intended product behavior prevents automatic removal.

| Location | Purpose and traces performed | Possible hidden consumer / reason retained | Risk / confidence | Recommended confirmation and action |
| --- | --- | --- | --- | --- |
| `frontend/src/cameraSemantics.mjs:1-43` | Normalizes vendor-neutral camera reports. No source import, HTML tag, server injection, or Vite module; only `frontend/tests/camera-semantics.mjs` imports it. Vite includes the other 121 frontend source files and excludes this one. History ties it to camera-report UI work; matching `.incident-camera-report*` CSS also has no literal producer. | May be an unfinished or temporarily disconnected product surface; deleting it would retire tested semantics. | Medium / high that it is production-unused | Product decision: reconnect it to incident presentation or remove module, test, and verified-unused styles together. |
| `survng/app/scene_context_memory.py:158-160` | `SceneContextSubject.stable_sightings` returns the length of `stable_event_keys`; definition-only across code, tests, docs, and strings. | Public property on a domain dataclass could be consumed by external diagnostics. | Low / medium | Confirm no external Python consumers, then remove the property (3 lines). |
| `survng/app/object_tracking_lifecycle.py:227-230` | `close_out_scene_work` forwards to durable store closure. Exact search finds only the definition; current shutdown uses `abort_scene_work`, which first stops compute. History shows the wrapper predates the safer abort path. | A direct external caller could depend on closure without abort; semantics are lifecycle-sensitive. | Medium / high that it is internally unused | Confirm no integration caller and decide whether direct terminalization remains supported; then remove with lifecycle regression tests. |
| `survng/app/semantic_search.py:1280-1286` | `observation_indexed` performs a one-row lookup. Definition-only; active callers use bulk `indexed_observation_keys`. History shows it predates the bulk path. | Public store method could be used by maintenance tooling outside this checkout. | Medium / high that it is internally superseded | Confirm external tooling, then remove and run semantic/backfill suites. |
| `survng/app/encoded_fragments.py:83-94,131-141` | `FragmentSourceInfo` and protocol/implementation `describe` have no call. Fragment streaming uses `fragments` and `materialize`; module and routes are otherwise live. | This is a typed protocol surface and may be reserved for future transport negotiation or external consumers. | Medium / high that it is internally unused | Interface decision: retain as planned capability or remove the dataclass plus both declarations with recording/fMP4 coverage. |
| `frontend/src/admin/ConfigPage.jsx:3558`, `Shell.jsx:54`, `TimelinePages.jsx:724,1468,3274` | ESLint reports unused `runtimeStatus`, advisor props, `theme`, `onAskAssistant`, `fastSeek`, `selectedEventId`, and `onEventSelect`. Call sites still supply several values and components are exported/tested. | These are component interfaces and may support fixtures, pending UI restoration, or downstream imports. | Low–medium / medium | Review each component contract in a dedicated frontend cleanup; do not silently narrow props in the binding batch. |
| `frontend/src/detectionOccupancy.mjs:614` | `restarts` is accepted but not used in occupancy presentation. | The input payload contract may intentionally accept the runtime counter for forward compatibility. | Low / medium | Decide whether restart count should be displayed or remove only the local destructuring, preserving caller payloads. |
| CSS lexical candidates | 1,174 class selectors were scanned; 126 names have no literal producer. Concrete old families include calibration/tune-up, legacy event cards, camera reports, and older Timeline inspector controls. Dynamic `page-*`, mode, health, track-color, and state classes demonstrate why lexical absence is insufficient. | Conditional, responsive, error, loading, and dynamically composed classes; no complete browser-state coverage was collected. | Medium / low as a group | Audit one UI family at a time with desktop/mobile live DOM and computed-style comparison before deletion. Estimated opportunity: hundreds of lines, not yet safely quantifiable. |

The ESLint `name`/`index` and object-rest `restarts`/`replay` reports were also
inspected. `name` and `index` participate only in omission/destructuring patterns;
`replay` deliberately strips a large replay payload. They are not deletion findings.

## C. Safe duplicate consolidation

None. No clone was both behaviorally equivalent and owned by a compatible
boundary strongly enough to justify coupling in this audit.

## D. Similar but intentionally distinct

jscpd reported the exact blocks below. Each block was inspected at both sites.
The grouped decision applies to every raw candidate; line ranges are the audited
pre-cleanup ranges.

| Family and exact locations | Why distinct / action |
| --- | --- |
| UI sizing, pan, range, and menu mechanics: `ConfigPage.jsx:2850-2863` ↔ `shared/evidence.jsx:236-249`; `IncidentCard.jsx:257-267` ↔ `shared/evidence.jsx:1008-1018`; `IncidentRecordingPlayer.jsx:64-74` ↔ `TimelinePages.jsx:122-132`; `TimelinePages.jsx:343-354` ↔ `442-453`; `TimelinePages.jsx:2575-2594` ↔ `2608-2627` | Similar mechanics live in components with different state, accessibility, playback, and cleanup ownership. Small local duplication is clearer than a cross-component abstraction. |
| Theme/responsive CSS: `shell.css:316-336` ↔ `826-846`; `shell.css:682-694` ↔ `729-741`; `styles.css:67-102` ↔ `105-140` | Selector/media-query contexts intentionally differ; merging declarations would change cascade or media behavior. |
| SQLite connection setup: `appearance_backfill.py:60-74` ↔ `semantic_search.py:357-371`; `appearance_backfill.py:61-74` ↔ `appearance_index.py:27-40`; `appearance_index.py:29-44` ↔ `semantic_search.py:360-375` | Separate store/worker ownership and lock policy; all already delegate the low-level connection primitive to `connect_main_database`. |
| Router construction and local endpoint closures: `appearance_routes.py:318-328` ↔ `face_routes.py:134-144`; `appearance_routes.py:458-471` ↔ `502-515` | Framework boilerplate and two different endpoint contracts; consolidation would obscure route ownership. |
| AI provider transport: `assistant.py:739-756` ↔ `audit_ai.py:619-636` | Different schemas, limits, and error semantics at different product boundaries. Shared lower-level transport already exists. |
| Model inference timing/preprocessing: `depth_estimation.py:166-181` ↔ `face_detection.py:44-59`, `face_recognition.py:69-82`, and `person_reidentification.py:69-83`; `depth_estimation.py:228-246` ↔ `detector.py:775-793`; `detector.py:771-793` ↔ `1059-1081`; `detector.py:1215-1225` ↔ `1340-1353`; `detector.py:1304-1321` ↔ `1408-1423` | Models intentionally differ in tensor layouts, failure handling, timing metrics, and post-processing. A shared abstraction would hide performance-sensitive differences. |
| Event persistence signatures/projection: `motion_intelligence.py:17-33` ↔ `decision_handler.py:447-463` and `1269-1282`; `event_store/store.py:299-312` ↔ `decision_handler.py:430-443`; `store.py:1050-1061` ↔ `1282-1293` | Store API, injected callback protocol, and two projection passes have different transaction and mutation ownership. |
| Face benchmark setup/query loops: `face_store/benchmarks.py:345-359` ↔ `488-502`, `703-715`, and `837-848`; `368-387` ↔ `510-529`; `373-383` ↔ `727-737`; `704-717` ↔ `838-849` | Each benchmark has different cohort selection and result semantics. Shared extraction would couple diagnostic experiments without reducing business logic. |
| Face query/projection: `face_store/people.py:902-912` ↔ `queries.py:346-356`; `people.py:1141-1159` ↔ `1312-1330` | Similar joins/identity formatting serve different mutation and read transactions. |
| Incident query filters: `incident_queries.py:377-387` ↔ `630-640` | List and notification queries intentionally expose parallel filters but return different projections. |
| Worker request loop: `inference_runtime/worker.py:196-211` ↔ `semantic_search.py:2109-2124` | Separate subprocess protocols, timeouts, and recovery ownership. |
| Assistant incident searches: `intelligence_routes.py:1919-1930` ↔ `1985-1996` | Similar timezone validation precedes different tools and evidence shapes. |
| Observability socket defaults: `local_observability.py:49-60` ↔ `ctl.py:19-30` | Server and standalone client intentionally remain independently startable; sharing would pull application imports into the control CLI. |
| Event/face semantics: `manager.py:1653-1685` ↔ `semantic_routes.py:169-201` | One enriches persisted event processing; the other formats request results under a manager-generation lease. |
| Media export builders: `media_exports.py:825-835` ↔ `1035-1045` | Recording and timelapse jobs use different FFmpeg graphs, progress, and fallback paths. |
| Storage containment: `media_storage.py:302-314` ↔ `317-329` | One answers boolean membership; the other resolves an owning location. Keeping both loops avoids changing exception/return contracts. |
| Motion rescue maps: `object_detection.py:640-652` ↔ `743-755`; `655-681` ↔ `756-782` | The two passes operate on different evidence phases and deliberately recompute against phase-local tracks. |
| Geometry/tracker materialization: `object_activity.py:645-657` ↔ `scene_context_memory.py:135-148`; `object_track/bytetrack.py:226-241` ↔ `ultralytics_tracking.py:304-319` | Identical math/data filling is embedded in independently owned attribution and tracker implementations with different accepted inputs. |
| Scene-context memory implementations: `scene_context_memory.py:218-233` ↔ `382-397`; `240-271` ↔ `405-436` | In-memory and SQLite implementations mirror one protocol but differ in lock, transaction, persistence, and ordering behavior. |
| Semantic embedding iteration: `semantic_search.py:674-685` ↔ `1010-1021` | Migration and normal query paths have different transaction/cache effects. |
| Tracking frame snapshots: `tracking_frames.py:336-367` ↔ `452-483` | Live and recorded snapshot paths use separate buffers and fallback rules. |

## E. Obsolete or suspicious, but blocked

| Surface | Evidence and blocker | Required decision |
| --- | --- | --- |
| Compatibility aliases and tested seams (`HybridCandidateObjectTracker`, `bind_for_compatibility`, `main.py` route aliases, `_index_event`, replay/reference helpers) | Several have only tests or dynamic callers, but docs and tests explicitly describe compatibility behavior. | Define a supported Python API/deprecation window before removal. |
| Legacy configuration normalization (`bytetrack` alias, retired MOG2 rejection, ignored historical camera keys, secret placeholders) | Validators and tests consume old persisted/user-supplied shapes. Removing them changes supported configuration and startup behavior. | Publish a config-version migration and minimum supported version. |
| SQLite and retained-JSON migrations | Schema probes, `ALTER TABLE` paths, legacy body embeddings, scene/outbox conversion, semantic identities, and portable-media migration remain exercised by tests. | Establish a minimum database schema/data age and an offline migration before deleting code or columns. |
| Old frontend routes and CSS families | `/recordings` and other legacy routes are explicitly canonicalized; old-looking styles overlap dynamic classes and historical browser states. | Product/deprecation decision plus browser coverage; do not remove from lexical evidence alone. |

No database field, migration, compatibility alias, environment variable,
configuration key, or serialized field is approved for removal by this report.

## F. False positives and explicitly retained code

- All 188 decorated route handlers reported by Vulture are live through FastAPI
  registration, even when their local Python names have no caller. Router bundle
  aliases installed in `main.py` are also deliberate compatibility exports.
- Pydantic validators and model fields are consumed by model construction and
  serialization. The low textual count for `adaptive_sampling_enabled` is valid:
  `object_track/session.py` reads it and config-application classification includes it.
- `JsonGZipResponder.send_with_compression`, urllib `redirect_request`, MQTT
  callback parameters, Ultralytics `init_track`/`reset_id`, and similar names are
  framework overrides. Removing unused signature parameters would violate callback
  contracts.
- ONVIF counters and timestamps flagged as write-only are read by the string field
  list in `camera_status.py` and the inspector's dynamic `getattr` projection.
- Store delay hints are called by string from the motion worker. Inference role
  methods are selected dynamically. Registry implementations are configuration
  values, not ordinary imports.
- Test-only lifecycle/diagnostic APIs—including `wait_for_camera_startup`,
  `active_leases`, `accepting_events`, `set_phase`, `episode_snapshot`,
  `latest_with_fallback`, `recorded_frames`, acquisition-ledger queries, and
  telemetry export—remain intentional tested seams. Their absence from production
  call graphs is not proof of dead implementation.
- The Python import graph reaches every production module; only
  `survng/__main__.py` lacks an importer because it is an executable entry point.
- Every tracked public icon is referenced by HTML, frontend code, the PWA manifest,
  or server routes. Standalone scripts, Docker files, systemd, CI, and Node-RED are
  operational entry points and were not judged by import graphs.
- No unreachable JavaScript statement was reported. Intentional object-rest
  omissions remove fields from returned payloads and are not unused logic.

## Validation matrix and cleanup sequence

| Change family | Focused validation | Full validation |
| --- | --- | --- |
| Selected A binding batch | Scene admission/review/establishment/incidents; incident queries; recorded object detection/motion pipeline; recording routes/API/fMP4; the three directly edited test files; affected frontend unit tests | Ruff/ESLint scans, 3,214-test backend suite, 70-file frontend suite, Vite production build, CLI discovery |
| B semantic/test-only helpers | Owning store/lifecycle tests plus external-consumer confirmation | Full backend/frontend suites as applicable |
| B CSS/UI surfaces | Targeted desktop/mobile browser states and synchronous computed-style comparison | Frontend unit suite and production build |
| E compatibility/migrations | Old-config and upgraded-real-SQLite fixtures, documented deprecation/migration | Full suite plus startup/upgrade smoke |

Recommended future batches, each requiring separate approval:

1. Confirm or reconnect the camera-semantics UI as a product decision.
2. Decide the supported external Python surface, then retire confirmed superseded
   public helpers one subsystem at a time.
3. Audit one stale CSS family with real browser coverage; never bulk-delete the
   126-name lexical upper bound.
4. Consider compatibility or migration removal only after an explicit version and
   deprecation policy.

The selected first batch is limited to category A. No category B–F item is part
of the source patch, and no commit, deployment, or service restart is authorized.

## Implemented first batch and post-change validation

The selected category-A batch removed all 16 unused bindings from 11 files. The
source/test diff deletes 15 lines and rewrites two import-list lines, for a net
reduction of 13 lines. No behavior, interface, route, configuration, schema,
serialization, persistence, or visual code changed.

| Check | Post-change result |
| --- | --- |
| Focused backend suite | **245 passed** in 13.62s |
| Focused frontend tests | Admin workspace, incident card, and incident navigation passed |
| Ruff `F401,F811,F841` | **Passed**, no findings across production, tests, scripts, or Docker helpers |
| ESLint audit | Selected findings reduced from 14 to 12; the 12 remaining component-contract/omission findings are documented in B/F and predate the batch |
| Vulture at 80% | Only 12 protocol/callback signature variables remain; all are framework or protocol false positives documented in F |
| Full backend suite | **3,214 passed**, one unchanged Starlette/httpx deprecation warning, 119.64s |
| Full frontend suite | **70 test files passed** |
| Production Vite build | **Passed**, 1,987 modules transformed; unchanged large video-player chunk warning |
| Server and control CLI help | Passed |
| `git diff --check` | Passed |

The production build wrote only ignored `survng/static/` output. Final review found
no tracked generated artifacts, no changes to the September reports, and no
unrelated source edits. The running service was not queried, restarted, or
deployed because this batch changes no runtime behavior and validation did not
require touching external state.
