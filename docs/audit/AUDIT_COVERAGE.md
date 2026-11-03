# SurvNG Audit Coverage Ledger

**Repository:** https://github.com/InstigatorX/SurvNG  
**Branch audited:** `v1.2` (base); audit docs authored on `cursor/comprehensive-codebase-audit-88c6`  
**Commit SHA:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1`  
**Audit date:** 2026-10-20  
**Method:** Static code review of execution paths, specialist deep-dives, focused pytest execution. No production code was modified. No live cameras, 24h soak, or destructive fault injection were performed.

---

## 1. Repository inventory (reviewed)

| Area | Path(s) | Technique | Status |
|------|---------|-----------|--------|
| Package entry | `survng/__main__.py`, `survng/app/__main__.py`, `survngctl`, `survng/ctl.py` | Read | Complete |
| FastAPI app / lifespan | `survng/app/main.py` | Read + prior hardening audit cross-check | Complete |
| Config | `survng/app/config.py`, `config.example.json`, `.env.example` | Read | Complete |
| Manager / reload | `manager.py`, `manager_access.py`, `manager_reload.py` | Inventory + hardening audit | Complete |
| Security / auth | `security.py`, `auth_routes.py`, `proxy.py`, middleware in `main.py` | Deep dive + tests | Complete |
| Camera fleet / worker | `camera.py`, `camera_fleet.py`, `camera_lifecycle.py`, `camera_capture.py`, `camera_startup.py` | Deep dive | Complete |
| ONVIF | `onvif_events.py`, `onvif_inspector.py`, `motion_topics.py` | Deep dive + addressing tests | Complete |
| Motion pipeline | `motion_*.py`, `ema_v2.py`, `motion_pipeline/*` | Deep dive + tests | Complete |
| Detector / AI | `detector.py`, `detector_model_settings.py`, `inference_runtime/*`, `person_reidentification.py`, `face_*.py` | Deep dive + detector tests | Complete |
| Tracking | `object_track/*`, `ultralytics_tracking.py`, `object_tracking_lifecycle.py` | Deep dive | Complete |
| Recording | `recording_process/*`, `recording_lifecycle.py`, `recording_media_runtime.py`, `recording_retention.py`, `ffmpeg_*.py` | Deep dive | Complete |
| Event store / scenes | `event_store/*`, `incident_*.py`, `evidence_*.py` | Deep dive + tests | Complete |
| Face / appearance / semantic | `face_store/*`, `appearance_*.py`, `semantic_search.py` | Inventory + selective read | Complete |
| Telemetry / ops | `telemetry_*.py`, `runtime_monitor.py`, `operations_routes.py`, `local_observability.py` | Inventory + selective read | Complete |
| Frontend SPA | `frontend/src/**` (49 TS/JS/JSX modules) | Inventory + targeted UX audit | Complete |
| Tests | `tests/` (~223 files), `frontend/tests/` (~90) | Inventory + focused execution | Partial (see §4) |
| CI / deploy | `.github/workflows/*`, `Dockerfile*`, `compose*.yaml`, `deploy/`, `docker/` | Read | Complete |
| Docs | `docs/**`, `README*.md`, `VIDEO_PIPELINE.md` | Inventory + selective | Complete |
| Models | `models/` (packaged IR trees) | Listed only; weights not executed | Partial |
| Node-RED | `node-red/` | Listed only | Not deep-reviewed |

---

## 2. Execution paths traced

1. **ONVIF → motion → detection → scene incident**  
   `OnvifEventListener._run` → `CameraWorker.handle_motion_event` → `MotionRuntimeService` → `MotionEventIngressService` → `MotionEventCoordinator` → `MotionDecisionOrchestrator` → `MotionIncidentService` / pipeline → `event_store` scene admission → `CanonicalIncidentLifecycle` → SSE/MQTT.

2. **Capture → analysis → EMA**  
   `FfmpegCaptureBackend` → `MotionAnalysisService` → EMA / qualification → admission decisions.

3. **Inference**  
   `InferenceLifecycle` → spawn workers → `OpenVinoDetector.detect*` → postprocess/NMS → tracking/ReID adapters.

4. **Recording**  
   `Recorder` FFmpeg subprocess → `recordings.sqlite3` index → retention planner → protected paths from EventStore.

5. **Auth boundary**  
   `SecurityBoundaryMiddleware` → session/API token → scopes → same-origin / CSRF heuristics.

6. **Frontend incident gallery**  
   SSE `/api/events/stream` + polling → `IncidentsPage` selection / related preview → API hydrate.

---

## 3. Audit techniques applied

- Recursive repository discovery (file counts, module map)
- Line-cited static analysis of critical modules
- Cross-module path tracing (motion, security, AI, DB)
- Comparison with existing `docs/production-hardening-audit.md` (baseline `347d6d2`; current HEAD `6fc78ac8`)
- Focused pytest execution (see §4)
- Frontend static review (no Playwright browser suite in this environment)
- Dependency inventory from `requirements*.txt` / `package.json`
- CI workflow mapping for coverage gaps

---

## 4. Tests executed (this audit)

Environment: fresh `/workspace/.venv` from `requirements.txt` + `pytest`/`anyio`/`httpx`; frontend built once so `survng/static` exists; FFmpeg present; no camera hardware; no Intel GPU assumed.

| Suite | Result |
|-------|--------|
| `test_api_security.py`, `test_web_auth.py`, `test_proxy.py` | Collected/ran after static build; included in 164-pass batch |
| `test_detector_model_settings.py` | Passed (includes e2e duplicate-box contract) |
| `test_database_connection_ownership.py` | Passed |
| `test_onvif_addressing.py` | Passed |
| `test_ema_v2.py`, `test_motion_ingress.py`, `test_scene_alert_decisions.py` | Passed |
| Combined focused batch | **164 passed**, 3 subtests, 1 Starlette/httpx deprecation warning, ~5.2s |
| Follow-up batch: `test_motion_events`, `test_motion_decisions`, `test_scene_incidents`, `test_evidence_projection`, `test_recording_retention`, `test_runtime_monitor` | **136 passed**, ~5.8s |
| **Combined focused total** | **300 passed** |

**Not executed in this audit:** full `tests/` suite (~223 files), frontend Playwright browser tests, Ultralytics optional suite, `scripts/production-soak.py`, live camera reconnects, disk-full injection, dependency CVE scanner (no `pip-audit`/`npm audit` run claimed).

---

## 5. Tools used

- `git`, `find`, `rg`/`Grep`, `Read`, Vite/`npm run build`
- `pytest` in isolated venv
- Cursor explore/generalPurpose specialist agents (architecture, security, AI, ONVIF/motion, DB, frontend/perf)
- No production DB, no external camera network access

---

## 6. Areas not validated / evidence gaps

| Gap | Why |
|-----|-----|
| 24/7 soak / multi-day backlog growth | Requires live deployment; soak script not run |
| Real camera firmware ONVIF quirks | No hardware; offline addressing tests only |
| GPU/OpenVINO device utilization under load | No sustained inference benchmark this run |
| Actual credential leakage via `ps`/`journal` | Static confirmation of argv logging; not demonstrated on host |
| Cross-site CSRF with modern browsers | Design review only |
| Full CI matrix | Self-hosted runner not available here |
| npm/pip CVE database freshness | Not scanned |
| Node-RED flows | Out of primary product path |
| “Storyline” product feature | **Does not exist by that name**; reviewed related-incidents / scene / timeline instead |
| End-to-end YOLO duplicate impact on incidents | Unit contract proven; production incident impact not measured |

---

## 7. Follow-up investigations required

1. Instrument episode merge-after-complete (AUD-003) under continuous ONVIF starts with EMA quiet.
2. Measure `protected_recording_paths` + snapshot retention CTE latency on multi-month DBs.
3. Confirm whether `trusted_proxies: ["*"]` appears in any customer configs.
4. Browser Playwright suite for Live hold-preview unmount and Incidents pagination selection.
5. Run `scripts/production-soak.py` for 12–24h after Wave 1 remediations.
6. Optional: `pip-audit` / `npm audit` with pinned advisory DB date.

---

## 8. Coverage completeness gate

| Phase | Status |
|-------|--------|
| 1 Discovery | Complete |
| 2 Correctness | Complete (major paths; not every of 228 modules line-by-line) |
| 3 Video/camera | Complete |
| 4 AI pipeline | Complete |
| 5 Incidents/evidence | Complete (storyline → related/scene terminology) |
| 6 Database | Complete |
| 7 Security | Complete |
| 8 Performance | Complete (mostly structural; limited measured) |
| 9 Frontend | Complete (static + selective) |
| 10 Config/deploy | Complete |
| 11 Tests | Complete inventory; partial execution |
| 12 Tech debt | Complete |
| 13 Cross-system FMEA | Complete |
| 14 Findings register | Complete in `FINDINGS.md` |
| 15 Deliverables | This directory |

Every confirmed finding in `FINDINGS.md` cites file/function/line evidence from HEAD `6fc78ac8`.
