# SurvNG Performance Audit

**Commit:** `6fc78ac8266a2334b4ec0ce62c8d252866f722a1`  
**Note:** This audit did **not** run sustained multi-camera load benches. Distinguish **measured** (prior docs / unit contracts) from **structural** suspicions.

---

## 1. Measurable KPIs (recommended instrumentation)

| KPI | Existing signal | Gap |
|-----|-----------------|-----|
| Per-camera CPU | Host tools / telemetry | Not broken out per thread |
| System CPU | soak / OS | — |
| Memory over time | `process_memory`, soak RSS | Need 12–24h soak |
| Inference p50/p95/p99 | detector metrics (`nms_ms`, etc.) | Aggregate histograms incomplete |
| Motion→detection latency | motion stats / audits | Operator dashboard optional |
| Detection→incident latency | outbox timing | Partial |
| Dropped-frame / mailbox replace | capture/analysis telemetry | Use under pressure |
| Queue depth | motion queue, refinement | Exposed in status paths |
| Camera reconnect frequency | camera status | — |
| DB query latency | busy/retry counters incomplete | Harden metrics |
| Evidence write latency | projection worker | — |
| API p95 | not continuous | Add if soak shows pain |

Prior hardening audit measured detector-replay heartbeat delay **1.0006s** under lock hold (fixed). Soak short trials showed loop delay ~0.1–1.0ms — **not** leak freedom.

---

## 2. Structural bottlenecks (highest impact)

### P1 — Main DB writer serialization (AUD-015)

All event/face/semantic/appearance writes share one RLock + SQLite writer (`main_database.py`). Under multi-camera motion storms this is the primary capacity cliff.

**Capacity model (not measurements):**

| Cameras | Expectation |
|---------|-------------|
| 5 | Comfortable |
| 20 | Busy peaks OK |
| 50 | Sustained queueing likely |
| 100 | Primary scaling cliff |

### P1 — Retention history scans (AUD-009)

`protected_recording_paths` materializes all recording refs; snapshot retention uses full-history window functions. Cost ∝ retained history, amplified by camera count × days.

### P2 — Live software decode CPU

`docs/performance/video-pipeline-optimization.md`: live capture stays SW decode; QSV for recorded evidence. Largest CPU opportunity if HW decode validated per host.

### P2 — Inference / BLAS thread oversubscription

Defaults: `object_worker_count=2`, refinements=4, analysis=2, recorded decode=2, plus per-camera motion `ThreadPoolExecutor`. No in-repo pinning of OpenVINO/OMP/BLAS threads → suspected oversubscription.

### P2 — Evidence write amplification

Per decision: cover snapshot + up to 2 gallery JPEGs + face candidate JPEGs + outbox rows (`decision_handler` / `evidence_gallery`). Structural multi-write, not IOPS-measured.

---

## 3. Bounded vs weakly bounded resources

| Resource | Bound | Notes |
|----------|-------|-------|
| Motion trigger queue | 32 | Evict / durable defer |
| Analysis mailbox | 1 | Drop oldest |
| Image cache | 5000 entries | Soft maintain interval |
| Semantic vector cache | 512MiB / 4096 | Harder bounds |
| Recording day list cache | TTL only | Suspected key growth |
| Evidence projection maps | `batch_size * 4` | OK |
| Media export queue | 100 | Documented |

---

## 4. Frontend performance notes

- SSE preferred over polls with sequence refs (`runtimeState.jsx`) — good.
- Live refresh clears all `incidentDetails` then reloads — harsher than Incidents page.
- Snapshot polling 2s desktop — expected for live tiles.
- Large Config/Timeline bundles (Vite warning >500kB for shaka) — load-time concern, not runtime leak.

---

## 5. Prioritized optimizations

| Rank | Optimization | Evidence type | Expected benefit |
|------|--------------|---------------|------------------|
| 1 | Index + incremental protected recording set | Structural AUD-009 | Retention plan CPU/memory |
| 2 | Batch / shed main-DB writers under storm | Structural AUD-015 | API + admit latency |
| 3 | Pin OpenVINO/OMP threads to topology | Suspected | Stable inference latency |
| 4 | Live HW decode where safe | Docs / host-specific | Per-camera CPU |
| 5 | Hard-cap recording day cache | Suspected | Long-uptime memory |
| 6 | Reduce face-candidate JPEG fanout | Structural | Disk/IOPS |

---

## 6. What was not claimed

- No invented FPS/throughput numbers.
- No claim that SurvNG “supports N cameras” as a product rating.
- Full-suite performance profiling not run in this environment.
