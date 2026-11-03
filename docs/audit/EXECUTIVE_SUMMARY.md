# SurvNG Engineering Audit — Executive Summary

**Audited commit:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1` (`v1.2`)  
**Date:** 2026-10-20  
**Scope:** Full-repository evidence-based review; production code unchanged.

---

## Overall assessment

SurvNG is a **mature, thoughtfully engineered single-host NVR + AI analytics stack** with strong durability patterns (WAL SQLite, durable motion/evidence outboxes, generation-fenced camera workers, isolated inference processes) and an unusually large regression suite. It is **not yet “set-and-forget 24/7 ready” for unattended multi-camera deployments without mandatory hardening**, primarily because authentication is **disabled by default** while the server binds `0.0.0.0`, and because several motion/incident state-machine gaps can silently miss or resurrect security events under continuous camera chatter.

**Production-readiness recommendation:** **Conditional go / no-go.**

| Deployment profile | Recommendation |
|--------------------|----------------|
| Homelab / trusted LAN, operator enables web auth immediately, ≤~10 cameras | Acceptable with Wave 1 hardening checklist |
| Unattended / internet-adjacent / multi-tenant / evidence-critical | **Not ready** until Wave 1 (auth defaults, bootstrap, credential exposure, incident state guards) is done |
| 50–100 camera fleet on one process | Architectural capacity risk; plan write-path and retention query work (Wave 3–4) before scaling claims |

---

## Major strengths

1. **Clear process ownership:** FastAPI + `AppManager`, per-camera threads, spawn-isolated inference, independent FFmpeg recorders.
2. **Durability:** Motion trigger jobs, detection jobs, scene notification outbox, evidence outbox with crash-safe ack semantics.
3. **Security boundary design (when enabled):** Scopes, session epochs, same-origin checks, WS auth, media path containment, secret redaction in many paths.
4. **Motion architecture sophistication:** EMA v2 episode controller, qualification stages, recorded decode budgets, generation fencing.
5. **Test culture:** ~223 backend + ~90 frontend tests; prior production-hardening audit closed multiple P1 lifecycle bugs with regressions.
6. **Operational tooling:** `survngctl`, telemetry SQLite, support bundles, soak script, Docker/systemd install docs.

---

## Major weaknesses

1. **Insecure defaults:** `api_auth.enabled=False`, `web_auth.enabled=False`, bind `0.0.0.0:8088`.
2. **Bootstrap trust model** treats all private IPs (and empty IP) as “local,” enabling LAN race to first admin.
3. **Secrets in config and process argv** (camera/MQTT/AI keys plaintext; FFmpeg `-i rtsp://user:pass@...`).
4. **ONVIF motion-end ignored;** post-complete camera-only motions can be swallowed inside episode gap → missed second incidents.
5. **Scene incident state can jump `unconfirmed → active`** without a prior-state predicate.
6. **Main SQLite writer serialization + history-wide retention scans** as the primary scale cliff.
7. **CI runs a minority of Python tests;** browser Playwright suite not in CI.
8. **No named Storyline feature** — documentation of “storylines” maps to scene incidents + related-incident UI; operators may over-trust related associations.

---

## Top 10 risks

| Rank | ID | Title | Priority |
|------|-----|-------|----------|
| 1 | AUD-001 | Auth disabled by default on all-interfaces bind | P0 |
| 2 | AUD-002 | First-admin bootstrap from any private/empty client IP | P0 |
| 3 | AUD-003 | Post-completion camera-only ONVIF motion swallowed (missed detections) | P1 |
| 4 | AUD-004 | Scene incident `unconfirmed`/`complete` → `active` without state guard | P1 |
| 5 | AUD-005 | Camera credentials in FFmpeg process arguments | P0 |
| 6 | AUD-006 | Plaintext secrets in `config.json` | P1 |
| 7 | AUD-007 | ONVIF motion-end never enters motion path | P1 |
| 8 | AUD-008 | PullPoint resubscribe drops buffered events | P1 |
| 9 | AUD-009 | History-wide `protected_recording_paths` / retention CTEs | P1 |
| 10 | AUD-010 | e2e YOLO near-duplicate policy can feed tracking merges | P2 |

Full register: [`FINDINGS.md`](./FINDINGS.md).

---

## Security posture

**When auth is enabled and trusted proxies are loopback-only:** solid for a single-tenant appliance (scopes, CSRF heuristics + SameSite, path containment, hashed API tokens, scrypt passwords).

**As shipped:** Critical insecure-default exposure. Credential hygiene (disk + process argv + some log lines) is the next tier. See [`SECURITY_AUDIT.md`](./SECURITY_AUDIT.md).

---

## Reliability assessment

Camera isolation is generally good (per-worker threads, generation checks, bounded motion queues). Cross-camera coupling exists via global capture-open limiter (2) and shared main DB writer. Highest silent failure modes are **missed second motions after detection completes** and **incident state resurrection**. Prior hardening closed several lifecycle P1s; remaining risks are state-machine and ONVIF semantics. See [`RELIABILITY_AUDIT.md`](./RELIABILITY_AUDIT.md).

---

## Performance assessment

No new sustained-load benchmarks were run in this audit. Structural evidence points to: live software decode CPU cost, main-DB write contention under multi-camera motion storms, retention planning scanning full incident history, and inference thread oversubscription risk (OpenVINO/OMP not pinned relative to worker counts). Documented prior measurement: detector-replay event-loop stall (fixed). See [`PERFORMANCE_AUDIT.md`](./PERFORMANCE_AUDIT.md).

---

## Maintainability assessment

Backend is large (~228 Python modules) but mostly domain-partitioned with intentional shims. Coupling centers on `AppManager` / `EventStore` / motion orchestration. Frontend is React 19 without a global store (context + modules + SSE) — workable but race-prone at edges. Dead-code audits already exist; no wholesale rewrite is justified.

---

## Important unknowns

- Multi-day durable-outbox growth under dependency outage (not soaked)
- Real multi-vendor ONVIF end-signal behavior
- Production impact of e2e duplicate boxes on false incidents
- Exact camera count at which retention/writer latency becomes operator-visible
- Whether any deployments use `trusted_proxies: ["*"]`

---

## Decision

Treat SurvNG as **production-capable only after Wave 1 remediation and an explicit auth-on / secrets / incident-state checklist**. Architecture is sound enough for iterative hardening via focused PRs; do **not** rewrite into microservices. Follow [`REMEDIATION_ROADMAP.md`](./REMEDIATION_ROADMAP.md).
