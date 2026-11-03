# SurvNG Remediation Roadmap

**Commit audited:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1`  
**Principle:** Small, reviewable PRs. No production code was changed in this audit. Do not begin remediation until this plan is accepted.

---

## Wave 1 — Critical correctness, security, data integrity

**Goal:** Safe defaults and stop silent incident corruption / open exposure.

| Finding | Modules | Sequence |
|---------|---------|----------|
| AUD-001 | `config.py`, `__main__.py`, docs/install | 1. Bind/auth policy |
| AUD-002 | `proxy.py`, `auth_routes.py`, tests | 2. Bootstrap locality |
| AUD-011 | `proxy.py`, config schema | 3. Remove/gate `trusted_proxies: *` |
| AUD-004 | `event_store/scenes.py`, scene tests | 4. State predicate on active write |
| AUD-005 | `ffmpeg_input.py`, capture/recorder, go2rtc docs | 5. Credential-free argv path |
| AUD-012 | `recording_process/recorder.py`, `camera_capture.py` | 6. Redact logs |

**Dependencies:** AUD-002 before relying on proxy trust; AUD-011 with AUD-002.  
**Regression risks:** Lab workflows needing open API; cameras requiring URL userinfo; intentional complete→active reopen.  
**Required tests:** web_auth bootstrap; api_security defaults; scene state; argv/log redaction.  
**Acceptance criteria:** Fresh install cannot be silently open on LAN; unconfirmed incidents cannot become active via unguarded UPDATE; no secrets in new log lines; documented bind policy.  
**Effort:** Small–Medium (~1 focused PR per finding preferred).

---

## Wave 2 — Reliability and failure recovery

**Goal:** Motion/ONVIF semantics that do not miss second intrusions.

| Finding | Modules | Sequence |
|---------|---------|----------|
| AUD-007 | `onvif_events.py`, `motion_ingress.py`, `ema_v2.py` | 1. Plumb motion-end |
| AUD-003 | `ema_v2.py`, motion decision tests | 2. Re-reserve after COMPLETED/FAILED |
| AUD-008 | `onvif_events.py` | 3. Renew-first / overlap subscribe |
| AUD-017 | `onvif_events.py` | 4. Unknown state ≠ start |
| AUD-016 | `recording_process/index.py` | 5. busy_timeout |
| AUD-023 | `ema_v2.py` | 6. Transition guards (optional same PR as AUD-003) |

**Dependencies:** AUD-007 makes AUD-003 less likely; still fix AUD-003 explicitly.  
**Regression risks:** Duplicate incidents if re-reserve too eager — lean on durable occurrence dedupe.  
**Required tests:** ema_v2 post-complete camera observe; ONVIF inactive; renew failure metrics.  
**Acceptance criteria:** Second motion after completed detection produces new work when policy expects; inactive ends episode gap; resubscribe rate visible.  
**Effort:** Medium.

---

## Wave 3 — Performance and resource optimization

**Goal:** Predictable latency as history and cameras grow.

| Finding | Modules | Sequence |
|---------|---------|----------|
| AUD-009 | `event_store/store.py`, retention | 1. Index + incremental protected set |
| AUD-015 | write batching / shed paths | 2. Measure then batch (no speculative DB split) |
| Thread pinning | `inference_runtime`, `worker_topology` | 3. OMP/OpenVINO env policy |
| Recording day cache | `recording_media_runtime.py` | 4. Hard cap |
| AUD-021 | `person_visits.py` | 5. Query reshape |

**Dependencies:** Microbench AUD-009 before claiming win.  
**Regression risks:** Retention deleting protected media — keep claim protocol.  
**Required tests:** retention unit + synthetic N-row timing harness.  
**Acceptance criteria:** Retention plan time grows sub-linearly with protected-set maintenance; no unprotected deletes.  
**Effort:** Medium–Large.

---

## Wave 4 — Architecture and maintainability

**Goal:** Reduce coupling without rewrite.

| Item | Modules | Notes |
|------|---------|-------|
| AUD-014 FKs / orphan repair | face_store, jobs | After delete-policy clarity |
| AUD-020 CI expansion | `.github/workflows/ci.yml` | Nightly full pytest; browser job |
| AUD-006 secrets at rest | config save/load | Keyring/encrypted secrets file |
| AUD-019 CSRF tokens | main.py, auth, frontend | Cookie mutation paths |
| Module seams | camera/motion/event_store | Split only where churn is high |
| AUD-018 / AUD-025 | LivePage, Shell | Frontend hygiene |

**Acceptance criteria:** CI catches scene/auth regressions; secrets not plaintext in default deploy path.  
**Effort:** Medium–Large (secrets Large).

---

## Wave 5 — Advanced capabilities and scalability

**Goal:** Only after Waves 1–3.

- Live HW decode where measured safe
- Main DB write-domain split (faces/semantic) if Wave 3 batching insufficient
- Stronger e2e NMS policy / temporal IoU when format is e2e (AUD-010, AUD-013, AUD-024)
- Cross-camera identity only with evaluation evidence
- Multi-host scaling — **not recommended** until single-host cliffs are measured and remaining

**Do not** add ReID complexity or microservices without KPI proof.

---

## Suggested PR backlog (reviewable units)

1. `fix(security): require auth or loopback bind by default` (AUD-001)  
2. `fix(auth): require bootstrap token for non-loopback` (AUD-002)  
3. `fix(proxy): disallow trusted_proxies wildcard` (AUD-011)  
4. `fix(scenes): guard incident reactivation states` (AUD-004)  
5. `fix(logging): redact FFmpeg and capture URLs` (AUD-012)  
6. `fix(media): avoid credentials in FFmpeg argv` (AUD-005)  
7. `fix(onvif): plumb motion inactive to episode end` (AUD-007)  
8. `fix(motion): re-reserve after completed episode for camera notices` (AUD-003)  
9. `fix(onvif): prefer renew recovery; metric resubscribes` (AUD-008)  
10. `perf(db): index recording_path + protected-set maintenance` (AUD-009)  
11. `fix(sqlite): busy_timeout on recordings index` (AUD-016)  
12. `ci: expand pytest batches + nightly full suite` (AUD-020)  
13. `fix(detector): safer default for ambiguous 6-col; seg-e2e near-dupe` (AUD-010/013)  
14. `fix(ui): clear hold-preview interval on unmount` (AUD-018)

---

## Wave effort summary

| Wave | Effort | Blocks production? |
|------|--------|--------------------|
| 1 | Small–Medium | **Yes** for unattended/exposed |
| 2 | Medium | **Yes** for evidence-critical motion reliability |
| 3 | Medium–Large | Scale / disk pressure |
| 4 | Medium–Large | Hardening / maintainability |
| 5 | Large / optional | Only with measurements |
