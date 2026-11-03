# SurvNG Test Coverage Audit

**Commit:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1`

---

## 1. Inventory

| Suite | Location | Scale |
|-------|----------|-------|
| Backend pytest | `/workspace/tests` | ~223 `test_*.py` |
| Frontend unit | `frontend/tests/*.mjs` via `run-tests.mjs unit` | ~71 |
| Frontend browser | same runner `browser` (Playwright) | ~19 |
| Helpers | `conftest.py`, `_isolated_runtime.py`, fixtures | — |

Backend coverage themes (non-exhaustive): motion/EMA, detector/NMS, camera lifecycle stress, event store/scenes, faces/identity, recording/fMP4, semantic search, auth/security, inference lifecycle, retention, telemetry, MQTT, assistant.

---

## 2. CI coverage gap

`.github/workflows/ci.yml` runs **focused globs** on a self-hosted runner (event store, detector, faces, motion, tracking, ultralytics optional, frontend **unit** only).

Approximate: **~92 / 223** Python files match CI patterns. **Not in CI** (examples): full scene suite, recording API, web auth, semantic search, assistant, MQTT, telemetry suite, property, many integration files.

**Browser Playwright tests are not in CI.**

---

## 3. Validation executed this audit

Environment: `/workspace/.venv` + frontend build for `survng/static`; no cameras.

| Batch | Result |
|-------|--------|
| `test_api_security`, `test_web_auth`, `test_proxy`, `test_detector_model_settings`, `test_database_connection_ownership`, `test_onvif_addressing`, `test_ema_v2`, `test_motion_ingress`, `test_scene_alert_decisions` | **164 passed**, 3 subtests, ~5.2s |
| `test_motion_events`, `test_motion_decisions`, `test_scene_incidents`, `test_evidence_projection`, `test_recording_retention`, `test_runtime_monitor` | **136 passed**, ~5.8s |

**Total this audit:** **300 passed** focused tests. Full suite not run.

Collection initially failed without `survng/static` (`RuntimeError: Directory 'survng/static' does not exist`) — document for contributors.

---

## 4. High-risk paths with weak automated coverage

| Path | Gap |
|------|-----|
| AUD-003 camera-only post-complete merge | Needs explicit EMA controller regression |
| AUD-004 unconfirmed→active SQL | Needs scene projection state predicate test |
| AUD-007 motion-end plumbing | No end-to-end inactive→episode close test |
| Auth defaults / bind policy | Defaults not treated as security tests |
| Bootstrap private-IP without token | Partial web_auth tests; extend for AUD-002 |
| Credential argv redaction | Warning exists; no assert on argv contents |
| Retention plan on large synthetic DB | Unit tests exist; no scale microbench in CI |
| Live hold-preview unmount | No browser test for interval clear |
| 24h continuous operation | Soak script exists; not CI |

Passing tests ≠ correctness for uncovered transitions.

---

## 5. Required failure scenarios — status

| # | Scenario | Status |
|---|----------|--------|
| 1 | Camera disconnect mid-inference | Partial (lifecycle/stress mocks); not hardware |
| 2 | ONVIF subscription expiration | Addressing/offline tests; renew-fail gap AUD-008 untested as soak |
| 3 | Two simultaneous motion events | Motion/durable job tests exist |
| 4 | Multiple objects enter/leave | Tracking/motion tests; crowded e2e duplicates weak |
| 5 | Inference longer than motion | Episode merge/followup logic; AUD-003 related |
| 6 | DB temporarily unavailable | Some export/face refill injection in prior hardening; not re-run fully |
| 7 | Evidence storage full | Partial retention pressure; not injected here |
| 8 | Restart during incident creation | Outbox/crash-safe design + cover recovery tests |
| 9 | Unexpected tensor dimensions | Detector settings errors |
| 10 | e2e overlapping detections | **Covered** by `test_detector_model_settings` (keeps duplicates) |
| 11 | Tracker loss on occlusion | Evaluation metrics exist; not CI-gated fully |
| 12 | Frontend out-of-order API | Sequence refs in runtimeState; browser gaps |
| 13 | Storyline multi-page selection | No Storyline; related-incident + Incidents pagination unit tests |
| 14 | Corrupt/missing evidence | Reconciler / cover recovery tests |
| 15 | Long-duration operation | Soak script; **not executed** |

---

## 6. Recommended regression tests (before fixing)

1. **AUD-003:** `MotionEpisodeController` complete → camera observe without EMA → expect new reserve (after fix) / document current merge.
2. **AUD-004:** `_scene_project` with `state='unconfirmed'` + activity → state remains `unconfirmed`.
3. **AUD-002:** bootstrap from `10.0.0.5` without token → 403.
4. **AUD-001:** config default + non-loopback bind policy.
5. **AUD-007:** inactive motion notification ends or freezes episode.
6. **AUD-012:** recorder log line with `rtsp://user:pass@host` redacted.
7. Frontend: unmount mid-hold-preview → no remaining intervals.

---

## 7. Static analysis / lint

Not run as separate tools this audit (`ruff`/`mypy`/`pip-audit`/`npm audit` not claimed). CI performs packaging/test batches only.
