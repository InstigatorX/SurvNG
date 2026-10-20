# SurvNG Security Audit

**Commit:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1`  
**Method:** Threat-model review + static confirmation. No destructive exploitation. Focused auth/proxy tests executed (see `AUDIT_COVERAGE.md`).

---

## 1. Threat model (appliance)

| Asset | Sensitivity |
|-------|-------------|
| Live/recorded video | High (privacy, physical security) |
| Incident evidence images | High |
| Camera ONVIF/RTSP credentials | High |
| MQTT / AI API keys | High |
| Admin session / API tokens | High |
| Config / zone geometry | Medium |
| Telemetry / logs | Medium (may embed URLs) |

**Actors:** LAN peer, malicious browser tab, compromised client, local host user, internet client if port forwarded, supply-chain dependency.

**Trust boundary:** Single host; SurvNG process can see all camera credentials and media. Network clients must not.

---

## 2. Attack surfaces

| Surface | Control |
|---------|---------|
| HTTP `/api/*` | `SecurityBoundaryMiddleware` when auth enabled |
| WebSocket `/webrtc`, `/mse` | Same auth + same-origin |
| SSE `/api/events/stream` | Auth when enabled |
| Static SPA `/static` | Public assets |
| Bootstrap `/api/auth/bootstrap` | Local IP or setup token |
| Config file on disk | File permissions (variable) |
| FFmpeg argv / logs | Weak |
| Unix observability socket | Owner-only |
| Trusted proxy headers | Config-dependent |

---

## 3. Verified vulnerabilities / confirmed defects

| ID | CWE | Title | CVSS-ish (qualitative) |
|----|-----|-------|------------------------|
| AUD-001 | CWE-306 | Auth off by default + `0.0.0.0` | Critical in exposed deploy |
| AUD-002 | CWE-862 | Bootstrap from private/empty IP without token | High |
| AUD-005 | CWE-214 | Credentials in process arguments | High |
| AUD-006 | CWE-312 | Secrets plaintext in config | High |
| AUD-011 | CWE-290 | `trusted_proxies: *` spoofing | High if enabled |
| AUD-012 | CWE-532 | Unredacted FFmpeg logs | Medium |

Details and line evidence: [`FINDINGS.md`](./FINDINGS.md).

---

## 4. Supported risks (not fully weaponized here)

- **AUD-019 CSRF:** Origin/`Sec-Fetch-Site` + `SameSite=Lax` mitigate classic cookie CSRF; missing Origin + missing Sec-Fetch-Site allowed.
- Session cookie theft still powerful if XSS exists — SPA XSS not exhaustively hunted; React escaping helps; `dangerouslySetInnerHTML` not systematically enumerated this pass.
- Dependency CVEs: versions pinned with ranges in `requirements.txt`; **no `pip-audit`/`npm audit` executed**.

---

## 5. Areas confirmed healthy (when auth enabled)

| Control | Evidence |
|---------|----------|
| Scope map admin vs viewer | `security.py` `required_api_scope` |
| API tokens stored as SHA-256 digests | `authenticate_api_token` |
| Passwords scrypt + compare_digest | `security.py` |
| WS denied without principal | `main.py` middleware |
| Media path containment | `relative_to` / `media_storage.contains` patterns |
| Uvicorn `proxy_headers=False` | `__main__.py:40–41` |
| Config API secret masking | `redacted_config_payload` |
| No `shell=True` in app | repo search |
| Login throttling | auth routes |

**Unauthenticated access when auth enabled:** Public paths limited to health/login/logout/bootstrap/session (`security.py` `is_public_api_path`). Sensitive media/config require principal — confirmed by design review; `test_api_security` / `test_web_auth` exercised in this audit’s pytest batch.

---

## 6. Evidence access question

> Could an unauthenticated or low-privilege user access, alter, or delete recordings, incidents, configurations, or credentials?

| Condition | Answer |
|-----------|--------|
| Auth disabled (default) | **Yes — full API** |
| Auth enabled, viewer role | Read-oriented scopes; should not mutate admin config (scope map) — not every route re-audited beyond middleware scope helper |
| Auth enabled, no credentials | **No** for `/api/*` non-public |
| Local host user | Can read config file + FFmpeg argv (AUD-005/006) regardless of web auth |

---

## 7. Remediation guidance (security-first)

1. **P0:** Force auth or loopback-only bind (AUD-001); harden bootstrap (AUD-002); remove credentials from argv (AUD-005).
2. **P1:** Kill `trusted_proxies: *` (AUD-011); redact logs (AUD-012); force config `0600` / secrets separation (AUD-006).
3. **P2:** CSRF tokens (AUD-019); dependency scanning in CI.

Do not claim CVSS numeric scores without a formal scoring worksheet; qualitative ratings above are for prioritization.
