# SurvNG Findings Register

**Commit:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1` (`v1.2`)  
**Legend:** Classification = Confirmed defect / Supported risk / Hypothesis / Improvement

Findings ordered by priority, then severity.

---

## AUD-001 — Authentication disabled by default while binding all interfaces

| Field | Value |
|-------|-------|
| Severity | Critical |
| Confidence | High |
| Classification | Confirmed defect (insecure default) |
| Subsystem | Security / deployment |
| Priority | P0 |
| Effort | Small |

**Evidence**

- `survng/app/config.py:40–42` — `ApiAuthConfig.enabled: bool = False`
- `survng/app/config.py:83–84` — `WebAuthConfig.enabled: bool = False`
- `survng/app/main.py:358–361` — auth enforced only when either flag is enabled
- `survng/app/__main__.py:17–18` — `--host` default `0.0.0.0`, port `8088`

**Root cause:** Auth is opt-in; network bind is open by default.  
**Failure scenario:** Fresh install on a shared LAN/VPN without enabling auth.  
**Impact:** Unauthenticated read/alter of recordings, incidents, config, camera control, restart.  
**Reproduction:** Start with example config; `curl` `/api/config` and `/api/cameras` without credentials.  
**Recommended fix:** Default `web_auth.enabled=True` with forced bootstrap; or refuse non-loopback bind when both auth modes are off; document as blocking install step.  
**Regression risk:** Breaks unauthenticated lab workflows — provide explicit `SURVNG_ALLOW_OPEN_BIND=1` escape.  
**Validation:** Unit test: open bind rejected; integration: unauthenticated `/api/*` → 401 when defaults apply.

---

## AUD-002 — First-admin bootstrap trustable from any private or empty client IP

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Auth |
| Priority | P0 |
| Effort | Small |

**Evidence**

- `survng/app/proxy.py:63–74` — `ip_is_local` treats `""`, private, link-local as local
- `survng/app/auth_routes.py:384–404` — bootstrap requires setup token only if `not ip_is_local(client_ip)`

**Root cause:** “Local” over-broad for enrollment threat model.  
**Failure scenario:** Fresh install, no users; any RFC1918 host POSTs `/api/auth/bootstrap` without token.  
**Impact:** Attacker becomes sole admin; enables auth and locks out operator.  
**Reproduction:** Empty `web_auth.users`; POST bootstrap from another LAN IP without token.  
**Recommended fix:** Require bootstrap token for all non-loopback peers; never treat empty IP as local; optional time-boxed setup mode.  
**Regression risk:** Headless LAN installs need token file (already produced for remote).  
**Validation:** Extend `tests/test_web_auth.py` for private-IP-without-token → 403.

---

## AUD-003 — After detection completes, camera-only ONVIF motion is merged without new work

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Motion / incidents |
| Priority | P1 |
| Effort | Medium |

**Evidence**

- `survng/app/ema_v2.py:541–552` — if status `COMPLETED`/`FAILED` and `ema is None`, return `MERGED_WITH_REQUEST` (no reserve)
- Episode gap refresh on observe keeps continuous starts from rolling over (`ema_v2` episode gap logic ~988–1016)

**Root cause:** Follow-up reservation gated on EMA; camera-only path treated as merge after terminal status.  
**Failure scenario:** Detection finishes; second person arrives within ~30s while EMA quiet and ONVIF keeps sending starts.  
**Impact:** Missed incidents / missed detections under continuous motion notifications.  
**Reproduction:** Unit-drive `MotionEpisodeController` — complete request, observe camera source without EMA, assert no new reserve (today) then expect reserve after fix.  
**Recommended fix:** Allow camera-only re-reserve after COMPLETED/FAILED, or close episode on motion-end / gap without refreshing on non-actionable observes.  
**Regression risk:** Duplicate incidents if too aggressive — pair with occurrence dedupe already in jobs.  
**Validation:** New regression in `tests/test_ema_v2.py`; motion path replay.

---

## AUD-004 — Scene incident state forced to `active` without prior-state predicate

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Incident lifecycle / DB |
| Priority | P1 |
| Effort | Small |

**Evidence**

- `survng/app/event_store/scenes.py:891–892` — `UPDATE scene_incidents SET state='active' ... WHERE id=?` when recent activity
- Contrast: restore only from `unconfirmed` → `complete` with predicate (`event_store/scene_admission.py:367–370`)
- Demotion to `unconfirmed` at `scene_admission.py:360`

**Root cause:** Reactivation write omits `AND state IN (...)`.  
**Failure scenario:** Incident demoted to `unconfirmed`; subsequent activity within 45s rewrites to `active` and re-notifies.  
**Impact:** Incorrect operator alerts; policy bypass for unconfirmed discoveries.  
**Reproduction:** Create incident → force `unconfirmed` → project activity → observe state `active`.  
**Recommended fix:** Restrict update to `state IN ('complete','active')` or exclude `unconfirmed`/`merged`.  
**Regression risk:** Intentional reopen of `complete` within 45s must remain.  
**Validation:** Scene alert / establishment tests.

---

## AUD-005 — Camera credentials exposed in FFmpeg process argv

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Security / capture / recording |
| Priority | P0 |
| Effort | Medium |

**Evidence**

- `survng/app/ffmpeg_input.py:8–18` — `-i` + `camera.source_url(source)` (includes userinfo)
- Used by recorder (`recording_process/recorder.py` ~153+) and capture
- Capture warns but still proceeds (`camera_capture.py` `_warn_credentialed_process_url` ~843–860)

**Root cause:** RTSP URLs with embedded credentials passed on command line.  
**Impact:** Local users / process listing / some crash dumps obtain camera passwords.  
**Recommended fix:** Prefer go2rtc credential-free restream; or FFmpeg auth via env/file not argv; strip userinfo from logged commands.  
**Regression risk:** Cameras that only accept URL userinfo need adapter path.  
**Validation:** Assert argv redaction helper; integration with go2rtc restream config.

---

## AUD-006 — Secrets persisted plaintext in `config.json`

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Security / config |
| Priority | P1 |
| Effort | Large (encryption) / Small (chmod hardening) |

**Evidence**

- `config.py` fields: MQTT/ONVIF `password`, AI `api_key` (~127–132, 188–193, 241–242)
- `save_config` writes `model_dump` JSON (`config.py` ~1426–1427), preserving existing mode

**Impact:** Backup/leak of config file exposes camera and cloud credentials.  
**Recommended fix (phased):** Force `0600`; separate secrets file or OS keyring; encrypt at rest.  
**Validation:** File mode tests; API still masks secrets (`redacted_config_payload`).

---

## AUD-007 — ONVIF motion-end never enters the motion path

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | ONVIF / motion |
| Priority | P1 |
| Effort | Medium |

**Evidence**

- `survng/app/onvif_events.py:491–507` — only `motion_state is True` calls `on_motion`; `False` increments `inactive_motion_events`

**Impact:** Episodes rely on gap timers; continuous start-only streams prolong episodes; compounds AUD-003.  
**Recommended fix:** Plumb inactive into ingress as motion-end observation (freeze gap / end episode).  
**Validation:** ONVIF fixture with start then end; episode closes promptly.

---

## AUD-008 — PullPoint resubscribe window drops buffered camera events

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | ONVIF |
| Priority | P1 |
| Effort | Medium |

**Evidence**

- `onvif_events.py:558–567` — unsubscribe/close then backoff retry (2→60s)
- Renew failure forces resubscribe path (~720–728)

**Impact:** Motion events during gap are lost; no durable camera-side buffer after new PullPoint.  
**Recommended fix:** Prefer renew recovery; overlap create-before-destroy where cameras allow; metric/alert on resubscribe rate.  
**Validation:** Inject renew failure; assert metric + minimized gap.

---

## AUD-009 — Retention planning scans full incident history for protected recordings

| Field | Value |
|-------|-------|
| Severity | High |
| Confidence | High |
| Classification | Supported risk (structural) |
| Subsystem | Database / retention |
| Priority | P1 |
| Effort | Medium |

**Evidence**

- `event_store/store.py:122–145` — `protected_recording_paths` UNION over all non-empty `recording_path` (no index on `events.recording_path`)
- Snapshot retention CTEs with window functions over all snapshots (`store.py` ~1517–1562)

**Impact:** Retention/plan latency and memory grow with history; disk-pressure feedback at 50–100 cameras.  
**Recommended fix:** Partial index on `recording_path`; maintain protected-set table; incremental candidates.  
**Validation:** Synthetic N-row DB microbench before/after.

---

## AUD-010 — End-to-end YOLO near-duplicate policy can retain overlapping boxes

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Confirmed behavior / Supported risk for incidents |
| Subsystem | AI pipeline |
| Priority | P2 |
| Effort | Small–Medium |

**Evidence**

- `detector.py:1200–1205` — `_suppress_e2e_near_duplicates` only after e2e parse
- `tests/test_detector_model_settings.py:63–98` — IoU 0.80 duplicates both kept by design
- `yolo-seg-e2e` applies neither full NMS nor near-dupe (`test_final_segmentation_does_not_apply_nms_twice`)
- Temporal association IoU min **0.05** (`motion_pipeline/object_detection.py:70–71`)

**Impact:** Duplicate boxes → looser temporal merges → possible false multi-frame confirmation.  
**Recommended fix:** Configurable e2e secondary NMS; add near-dupe to seg-e2e; tighten temporal IoU when e2e format.  
**Validation:** Existing detector tests + recorded object detection crowded scene fixture.

---

## AUD-011 — `trusted_proxies: ["*"]` enables IP/scheme spoofing

| Field | Value |
|-------|-------|
| Severity | High (when configured) |
| Confidence | High |
| Classification | Confirmed defect (dangerous option) |
| Subsystem | Security / proxy |
| Priority | P1 |
| Effort | Small |

**Evidence:** `proxy.py:24–26`, `51–53`, `110–112`  
**Impact:** Spoof private IP → bootstrap without token (AUD-002); spoof HTTPS flags.  
**Fix:** Remove `*` or require explicit danger flag + warning; never recommend in docs.

---

## AUD-012 — Recorder FFmpeg stderr logged without secret redaction

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Confirmed defect |
| Subsystem | Security / logging |
| Priority | P1 |
| Effort | Small |

**Evidence:** `recording_process/recorder.py:321` — logs raw line  
**Impact:** `rtsp://user:pass@...` in journal/stdout (memory log handler may redact API view, not systemd).  
**Fix:** `redact_secret_text(line)` at emit site (also capture `_log_decode_fallback` ~754–765).

---

## AUD-013 — Ambiguous six-column tensors inferred as `yolo-e2e` skip SurvNG NMS

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | AI |
| Priority | P2 |
| Effort | Small |

**Evidence:** `detector_model_settings.py:100–108` — `channels == 6 and anchors <= 1000` → e2e  
**Impact:** Raw 2-class YOLO misclassified → many duplicates.  
**Fix:** Prefer safer default `yolo` when metadata absent; strengthen warning in Admin UI.

---

## AUD-014 — Soft foreign keys leave orphan face/audit rows

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Database |
| Priority | P2 |
| Effort | Medium |

**Evidence:** `face_store/store.py` face_observations without FK to events; `motion_audits` / `tracking_comparisons` without FK (`event_store/jobs.py`)  
**Impact:** Integrity relies on app prune paths; deletes can leave orphans.  
**Fix:** Add FKs where delete policy is clear; periodic orphan repair job.

---

## AUD-015 — Single process-wide main DB write lock

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Performance / DB |
| Priority | P2 |
| Effort | Large |

**Evidence:** `main_database.py:40–56`; shared lock wired in `manager.py`  
**Impact:** Motion storms + face + semantic serialize; API latency spikes.  
**Fix:** Batch writes; move non-critical indexes off main DB; measure before split.

---

## AUD-016 — `recordings.sqlite3` connections omit `busy_timeout` pragma

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Recording index |
| Priority | P2 |
| Effort | Small |

**Evidence:** `recording_process/index.py:33–40` — connect timeout only  
**Fix:** `PRAGMA busy_timeout=10000` on each connection (match EventStore).

---

## AUD-017 — Reolink/generic topics default missing state attributes to motion start

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | ONVIF |
| Priority | P2 |
| Effort | Small |

**Evidence:** `onvif_events.py:1036–1053`, `_motion_event_state` → `return True` ~1115  
**Impact:** Ambiguous notifications become starts (false triggers).  
**Fix:** Treat unknown as unrecognized; require explicit state when dialect supports it.

---

## AUD-018 — Live hold-preview unmount may leak `selectionTimer` interval

| Field | Value |
|-------|-------|
| Severity | Low |
| Confidence | Moderate |
| Classification | Supported risk |
| Subsystem | Frontend |
| Priority | P2 |
| Effort | Small |

**Evidence:** `frontend/src/live/LivePage.jsx:129–132` clears timeouts but not `selectionTimer`; arm at 164–180; clear path at 138–143  
**Impact:** 50ms interval continues after unmount if gesture mid-flight.  
**Fix:** Call `clearHoldPreviewArm()` in unmount effect.

---

## AUD-019 — CSRF relies on Origin/Sec-Fetch-Site heuristics without token

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | Moderate |
| Classification | Supported risk |
| Subsystem | Security |
| Priority | P2 |
| Effort | Medium |

**Evidence:** `main.py:_same_origin_request` ~314–318; cookie `SameSite=Lax` (`auth_routes.py` ~187–194)  
**Impact:** Weaker against nonstandard clients / missing headers.  
**Fix:** Synchronizer token for cookie-authenticated mutations.

---

## AUD-020 — CI executes minority of Python tests; no Playwright in CI

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Confirmed gap |
| Subsystem | QA |
| Priority | P2 |
| Effort | Medium |

**Evidence:** `.github/workflows/ci.yml` focused globs; ~92/223 tests matched; frontend `unit` only  
**Impact:** Scene/auth/semantic/recording regressions can land unnoticed.  
**Fix:** Expand CI batches; nightly full suite; optional browser job.

---

## AUD-021 — Person-visit query uses JSON LIKE + N+1

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Performance / DB |
| Priority | P3 |
| Effort | Medium |

**Evidence:** `person_visits.py:205–224`  
**Fix:** Structured person flags/index; batch appearance/face fetch.

---

## AUD-022 — Orphan media files reported but not auto-deleted

| Field | Value |
|-------|-------|
| Severity | Low |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Storage |
| Priority | P3 |
| Effort | Medium |

**Evidence:** `storage_maintenance.py` orphan reporting without delete (~327–369)  
**Impact:** Disk growth after crash between write and DB commit.  
**Fix:** Age-gated orphan reclaim with dry-run.

---

## AUD-023 — Detection request status transitions lack predecessor guards

| Field | Value |
|-------|-------|
| Severity | Low |
| Confidence | High |
| Classification | Supported risk |
| Subsystem | Motion |
| Priority | P3 |
| Effort | Small |

**Evidence:** `ema_v2.py:852–896` `mark_running`/`complete`/`fail`/`abort` set status without legal-predecessor checks  
**Impact:** Illegal transitions if callers race; partially mitigated by intent checks.  
**Fix:** Explicit transition table.

---

## AUD-024 — OpenCV-DNN fallback skips graph NMS inspection

| Field | Value |
|-------|-------|
| Severity | Medium |
| Confidence | Moderate |
| Classification | Supported risk |
| Subsystem | AI |
| Priority | P2 |
| Effort | Medium |

**Evidence:** `detector.py:396–411` fallback sets `output_format="unknown"`; `_model_has_nms` only on OpenVINO path ~319  
**Impact:** Wrong NMS count after OpenVINO failure.  
**Fix:** Re-run format resolution + conservative NMS on fallback.

---

## AUD-025 — No skip-to-content / incomplete a11y baseline

| Field | Value |
|-------|-------|
| Severity | Low |
| Confidence | High |
| Classification | Improvement |
| Subsystem | Frontend |
| Priority | P3 |
| Effort | Small |

**Evidence:** No skip link under `frontend/src`; blueprint cites WCAG AA (`docs/product-ui-blueprint.md`)  
**Fix:** Skip link; aria-labels on pagers; contrast audit.

---

## Strengths (non-findings)

- Path traversal containment on media/export/recording serve paths
- No `shell=True` in app subprocesses
- Durable outboxes for incidents/evidence
- Generation-fenced camera/ONVIF stop
- `ClosingDatabaseConnection` closes on context exit (`main_database.py:25–37`)
- Prior P1 lifecycle defects documented as fixed in `docs/production-hardening-audit.md`

---

## Severity calibration notes

- AUD-001 is Critical because default deployment exposes sensitive evidence APIs, not because auth code is broken when enabled.
- AUD-010 is Medium: duplicate boxes are intentional for e2e; incident impact is Supported risk pending production metrics.
- No Critical severity assigned to purely theoretical concurrency without a concrete illegal transition (AUD-004/003 are High with code-path proof).
