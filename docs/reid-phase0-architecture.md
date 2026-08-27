# SurvNG ReID Fine-Tuning — Phase 0 Architecture Report

**Status:** Discovery complete. No production code changed.  
**Scope:** Person appearance ReID as implemented today, plus adjacent identity systems that constrain future work.  
**Date:** 2026-08-27  

---

## Implementation status (post Phase 0)

Started on `v1.2` (environment-adaptation MVP):

- Isolated store: `survng/app/reid_training/` → `reid-training.sqlite3` + `storage/reid_training/`
- Config-gated collector on tracking complete (`reid_training_collector_enabled`, default off)
- Same-track samples share an anonymous `person_NNNNNN` identity (`assignment_source=track`)
- Review APIs + Admin **ReID Training** workspace (`/admin?section=reid`):
  - hard cross-camera pairs from appearance matches joined to training crops
  - actions: same / different / unknown / reject
- No dataset export / train / OpenVINO promote yet

Enable in detector tracking config:

```json
"reid_training_collector_enabled": true
```

---

## A. Existing Architecture

SurvNG already has **three separate embedding systems**. Only the first is the production person ReID path this project must fine-tune.

### 1. Whole-object appearance ReID (person + vehicle)

Purpose:

- reconnect tracks after geometry association fails (occlusion / large movement)
- store durable appearance signatures for related-incident / forensic visual similarity

It does **not** assign named people. README and forensic-search docs explicitly treat strong matches as **visual similarity, not confirmed identity**.

Implementation center:

| Concern | Location |
| --- | --- |
| Encoder | `survng/app/person_reidentification.py` |
| Crop + annotate | `survng/app/object_track/session.py` |
| Track association | `survng/app/object_track/bytetrack.py` |
| Durable index | `survng/app/appearance_index.py` |
| Deferred recovery | `survng/app/appearance_backfill.py` |
| Isolated inference | `survng/app/inference_runtime/*` |
| Config | `survng/app/config.py` → `ObjectTrackingConfig` |
| HTTP | `survng/app/appearance_routes.py` |

Default person model (installer):

```text
/models/person_reid_model/person-reidentification-retail-0286.xml
/models/person_reid_model/person-reidentification-retail-0286.bin
```

Source: Intel Open Model Zoo FP16 OpenVINO IR (Apache-2.0). Architecture is **OSNet + Linear Context Transform (LCT)**, originally PyTorch. Official OMZ contract:

- input `1×3×256×128` (NCHW), **BGR**, height 256 / width 128
- output `1×256` embedding (`reid_embedding`)
- comparison intended via cosine distance on descriptors

SurvNG discovers layout/dims at load time and L2-normalizes embeddings, then compares with **cosine similarity** (`dot` of unit vectors). Higher is more similar. Default person threshold: **0.70**.

Person ReID is **disabled by default** (`reid_enabled=false`) until a model path is configured.

### 2. Face recognition (named identity)

Separate ArcFace pipeline (`face_recognition.py`, `face_store/*`, `/people` UI). This is the existing **named person / unknown cluster / review queue** system. Do not replace it with ReID identities.

### 3. Semantic / Smart Search (MobileCLIP2)

Separate CLIP-style embeddings for text/image search. Not person ReID.

### Production tracker

**SurvNG Hybrid** (`survng_hybrid`) is the production tracker: ByteTrack-style two-pass geometry association, then selective appearance recovery. FastTrack / Deep OC-SORT are offline comparison only (`README.tracking.md`).

### Runtime isolation

ReID runs in a dedicated inference worker role (`role == "reid"`) via `InferenceSupervisor` / `IsolatedPersonReidentifier`. Production runtime depends on **OpenVINO + OpenCV + NumPy only** — no PyTorch.

---

## B. Code Map

```text
survng/app/
  person_reidentification.py
    OpenVinoPersonReidentifier          # person or vehicle engine
      _load() / _image_input() / _image_tensor() / _fingerprint()
      embed()
      status()
    OpenVinoAppearanceReidentifier      # label router
      supports_label() / embed_for_label() / model_identity_for_label()

  object_track/
    types.py
      AppearanceEncoder / AppearanceIndexWriter protocols
    geometry.py
      _encode_appearance() / _appearance() / _ensure_detection_appearance()
    bytetrack.py
      ObjectTrack                       # appearance EMA, reid_recovery_history
      ByteTrackObjectTracker
        _associate_appearance()         # cosine recovery
        appearance_records()            # index export
    session.py
      ObjectTrackingSession
        _annotate_appearances()         # bbox crop → embed (eager/lazy)
        _encode_with_telemetry()
        _index_track_appearances()
    registry.py                         # survng_hybrid, bytetrack, ultralytics_*

  appearance_index.py
    AppearanceIndex                     # SQLite appearance_embeddings
      replace_event() / matches() / status()

  appearance_backfill.py
    DeferredAppearanceBackfill          # snapshot crop when multi-frame tracking missed

  appearance_routes.py
    GET /api/events/{id}/appearance-matches
    GET /api/appearance-index/status
    POST /api/appearance-index/backfill
    GET /api/events/{id}/related-incidents

  inference_runtime/
    process.py                          # embed_person / embed_reid ops
    adapters.py                         # IsolatedPersonReidentifier
    supervisor.py                       # reid worker lifecycle / timeouts

  inference_lifecycle.py                # wires encoder + indexer into tracking
  object_tracking_lifecycle.py          # per-camera tracking sessions
  config.py                             # ObjectTrackingConfig ReID fields
  config_application.py                 # TRACKING_REID_ENGINE_FIELDS reload

  # Adjacent — do not conflate with person ReID
  face_recognition.py / face_store/* / face_routes.py / identity_projection.py
  semantic_search.py
  training_routes.py                    # detector-box training samples API (not ReID)
  model_evaluation.py                   # detector A/B evaluation (not ReID)

scripts/
  install-docker-models.sh              # downloads OMZ person-0286 + vehicle OSNet ONNX
  export-mobileclip2-openvino.py        # pattern for optional Torch→OpenVINO export

models/
  person_reid_model/                    # expected install path (binaries not in git)
  vehicle_reid_model/README.md

frontend/src/
  workspaceNavigation.mjs               # /admin, /people workspaces
  admin/ConfigPage.jsx                  # Tracking & ReID settings
  people/FacesPage.jsx                  # named face review patterns to reuse
  visualSearch.mjs                      # appearance-matches client
  objectTrackReplay.mjs                 # reid_recovery_history replay

tests/
  test_person_reidentification.py
  test_appearance_index.py
  test_appearance_backfill.py
  test_object_tracking.py
  test_inference*.py
  test_cross_camera_trace.py
```

---

## C. Current Data Flow

```text
OpenVinoDetector.detect()
      ↓  person / vehicle boxes (xyxy on frame)
ObjectTrackingLifecycle.start_incident()
      ↓
ObjectTrackingSession (sampled frames from recording)
      ↓
_annotate_appearances(frame, objects, lazy=hybrid)
      │  clamp box → tight crop frame[y1:y2, x1:x2]
      │  reject if h<16 or w<8
      │  limit top-N by confidence (reid_max_embeddings_per_frame, default 8)
      ↓
eager: _encode_with_telemetry → embed_for_label
  or
lazy:  _tracking_embedding_provider (Hybrid only; infer when association needs it)
      ↓
IsolatedPersonReidentifier / OpenVinoAppearanceReidentifier
      ↓
OpenVinoPersonReidentifier.embed(crop)
      │  resize INTER_AREA to (W,H)=(128,256)
      │  float32, BGR (person), NCHW/NHWC as discovered
      │  OpenVINO infer → float32 vector
      │  L2 normalize
      ↓
detection["_tracking_embedding"]
      ↓
ByteTrackObjectTracker.update()
      │  1) high/low confidence geometry association (IoU / center)
      │  2) _associate_appearance: cosine = dot(track.appearance, embedding)
      │     resume lost/completed tracks if score ≥ label threshold
      │  3) ObjectTrack.observe(): EMA update
      │        appearance = L2(0.8 * old + 0.2 * new)
      ↓
track summary → events.objects_json (track_id, trajectory, reid_matches, …)
      ↓
appearance_records() → AppearanceIndex.replace_event()
      ↓
appearance_embeddings (SQLite blob, model_fingerprint gated)
      ↓
AppearanceIndex.matches() / related-incidents API
      ↓
operator sees visual similarity (not a named ReID identity)
```

Face path (parallel, not this subsystem):

```text
face candidates → padded crop → ArcFace embed → face_observations
      → gallery match / unknown clusters → /people identity
```

---

## D. Current Model Contract

### Person ReID (production target)

| Field | Value |
| --- | --- |
| Model | `person-reidentification-retail-0286` |
| Source | Intel OMZ FP16 IR (`.xml` + `.bin`) |
| Backbone | OSNet + LCT (PyTorch origin) |
| Inference | OpenVINO (`Core.read_model` / `compile_model`) |
| Device | config `reid_device` (default `AUTO`, CPU fallback) |
| Batching | **single-image** infer request; no batch API |
| Input shape | `(1, 3, 256, 128)` NCHW after load (SurvNG stores `(W,H)=(128,256)`) |
| Color | **BGR** (OpenCV crop as-is) |
| Resize | `cv2.INTER_AREA` to exact size (no letterbox, no aspect preserve) |
| Pixel scaling | `astype(np.float32)` only — **no `/255`, no mean/std in SurvNG** |
| Dtype in | `float32` |
| Output | `embedding_size` discovered at load; OMZ documents **256** |
| Dtype out | `float32` |
| Normalization | **L2** in `embed()`; re-normalized in tracker EMA and `AppearanceIndex` |
| Comparison | **cosine similarity** = `np.dot(a, b)` on unit vectors |
| Threshold | `reid_match_threshold` default **0.70** (person); vehicle **0.80** |
| Crop expansion | **none** (tight detector box) |
| Min crop | height ≥ 16, width ≥ 8 (session); embed requires min side ≥ 8 |

### Vehicle ReID (related, out of initial person fine-tune scope)

| Field | Value |
| --- | --- |
| Model | `vehicle-reid-0001` ONNX (`osnet_ain_x1_0_vehicle_reid.onnx`) |
| Input | `(208, 208)`, **RGB** (BGR→RGB in `_image_tensor`) |
| Threshold | 0.80 |

### Important contract notes for training/export

1. SurvNG preprocessing must be reproduced exactly: tight crop → INTER_AREA → float32 BGR → (layout) → OpenVINO.
2. Do not assume IR-internal preprocessing; SurvNG does not apply ImageNet mean/std.
3. Production compares **similarity**, not distance. Threshold tooling must use the same polarity.
4. Model generations are gated by **SHA-256 fingerprint** (`model_fingerprint`, 24 hex chars). New exports must not silently mix with old index rows.

---

## E. Persistence Assessment

### Reusable today

| Asset | Reuse for ReID training? |
| --- | --- |
| `events` + `objects_json` | Yes — event/track/box/trajectory/reid evidence |
| `appearance_embeddings` | Yes — track-level embeddings + model fingerprint + quality |
| Event snapshots / recordings | Yes — crop source frames |
| `appearance_backfill_jobs` | Pattern for deferred crop work |
| `GET /api/training/samples` | Detector pseudo-labels only; **not** person-ID labels |
| `face_people` / observations | Named identity UX patterns; optional *link*, not storage for body ReID samples |
| Face review / unknown-cluster UI | Best frontend template for cluster review |

### Gaps (need new / isolated ReID-training persistence)

Person ReID today has **no**:

- person crop archive for training
- sample-level metadata (review_status, assignment_source, hard pos/neg)
- versioned labeled datasets
- ReID model registry / promotion records
- training-run metrics
- named ReID identity distinct from face (and should not overload `face_people` without an explicit link design)

**Recommendation:** isolate ReID-training tables (or a sidecar SQLite DB under storage), e.g.:

- `reid_samples`
- `reid_identities` (stable internal ID + optional display name + optional `face_person_id` link)
- `reid_assignments` (manual authoritative)
- `reid_review_queue` / cluster jobs
- `reid_dataset_versions` (immutable metadata pointers)
- `reid_model_registry`

Do **not** stuff training crops into `appearance_embeddings`. That table is a production similarity index with cascade-delete on events and fingerprint gating.

---

## F. Training Integration Recommendation

### Framework

**Primary recommendation: Torchreid (or a thin native PyTorch trainer inspired by it), not FastReID as the default.**

Rationale:

- Production backbone is **OSNet** (OMZ 0286). Torchreid is the natural OSNet ecosystem.
- FastReID is heavier and more opinionated; acceptable later if experiments justify it.
- SurvNG already uses optional Torch only for **offline export** (MobileCLIP → OpenVINO), never as a production dependency.

### Integration strategy (mirrors MobileCLIP)

1. **Production remains OpenVINO-only** (`requirements.txt` unchanged for ReID training).
2. Add optional `requirements-reid-train.txt` (torch, torchvision, optionally torchreid).
3. Put training/eval/export under `tools/reid/` or `scripts/reid_*` + a non-imported package tree that production never imports.
4. Export path: trained weights → **OpenVINO IR FP16** (`.xml`/`.bin`) matching SurvNG’s `read_model` loader.
5. Validate export parity: Torch embedding vs OpenVINO embedding on a fixed crop set within an explicit tolerance; then re-check cosine scores.
6. Promote only via explicit registry entry + config `reid_model_path` change (existing reload already restarts the `reid` inference role).

### Loss / sampling (initial experiment, configurable)

- Start from pretrained OSNet-compatible weights compatible with 0286 lineage when available; otherwise train a same-architecture OSNet and validate against current 0286 embeddings before promotion.
- `L = L_identity + λ L_triplet` with PK sampling (start P=8, K=4).
- Augmentations: hflip, brightness/contrast, mild jitter, random erasing, mild blur/compression — no vertical flip / large rotation.

---

## G. Proposed Module Layout

Align with SurvNG conventions (focused routers, optional tools, models outside git):

```text
survng/app/
  reid_training/                 # optional; import-guarded; unused unless enabled
    __init__.py                  # raises/clears if torch missing
    schema.py                    # isolated tables
    collector.py                 # track sample selection (3–8 / track)
    review_service.py
    dataset_builder.py
    metrics.py                   # Rank-1/5, mAP, TPR@FPR, threshold curves
  reid_training_routes.py        # /api/reid-training/... (admin only)
  # keep person_reidentification.py as the sole production encoder

tools/reid/                      # or scripts/reid/
  train.py
  evaluate.py
  export_openvino.py
  validate_export.py
  build_dataset.py

models/reid/
  production.json                # pointer to active version
  reid-v001/
    model.xml / model.bin
    training_config.yaml
    metrics.json
    dataset.json
    model_metadata.json

datasets/reid/                   # under storage_dir, not git
  survng-reid-v001/
  reid-benchmark-v1/             # immutable once published

frontend/src/
  admin/reid/                    # or admin subsection
    ReidAdminPage.jsx            # cluster review, datasets, models
  # Prefer /admin?section=reid  OR dedicated /admin/reid under admin workspace
  # Do NOT invent a top-level workspace until needed
```

**UI route preference:** extend the existing **Admin** workspace (`/admin`) with a ReID subsection, or add `/admin/reid` under the same admin shell. Reuse **People** (`/people`) patterns for gallery/crop review, but keep ReID labeling separate from face confirmation so operators do not confuse ArcFace identity with body appearance labels.

Image serving: follow `GET /api/faces/observations/{id}/crop.jpg` + `image_cache` pattern for ReID sample crops.

---

## H. Risks / Conflicts

1. **ReID ≠ named identity today.** Spec §10–12 wants person IDs and review. Production ReID is anonymous visual similarity. Extending it into identity must not undermine face recognition or claim body-ReID matches as legal identity.
2. **Face system already owns “people.”** Creating a parallel `Steve` namespace without linking to `face_people` will confuse operators. Prefer stable `reid_identity_id` with optional link to `face_person_id`.
3. **No crop padding today.** Spec suggests 5–10% expansion. Changing inference crops would invalidate the current model contract and indexed embeddings. Training/export must start with **tight boxes**; expansion is a controlled experiment requiring re-fingerprint + re-index.
4. **No mean/std /255 in SurvNG.** Many Torchreid pipelines normalize differently. Export validation must lock SurvNG’s preprocessing, not Torchreid defaults.
5. **Track EMA vs raw embeddings.** Indexed vectors are EMA-smoothed track signatures, not single-crop embeddings. Training samples should store **per-crop** embeddings computed consistently; do not treat index blobs as crop labels.
6. **Tracks are per-camera / per-event.** Cross-camera identity is investigative (`appearance-matches`, transition routes), not a unified multi-camera track ID. Dataset building must merge identities via human review, not assume track_id equality across cameras.
7. **Fingerprint gating.** Fine-tuned models are a new generation; old `appearance_embeddings` will not match until backfill/reindex.
8. **Training deps must stay optional.** Putting torch in main `requirements.txt` violates SurvNG’s NVR footprint principles (`README.tracking.md`).
9. **Existing `/api/training/samples` is detector-oriented.** Do not overload it for ReID person-ID datasets; add a dedicated API.
10. **Spec cosine assumption.** Correct here (SurvNG uses cosine similarity), but thresholds are similarity thresholds — distance-based Torch metrics must be converted carefully.
11. **Person ReID disabled by default.** Collector should no-op cleanly when ReID is off.
12. **Vehicle ReID shares the encoder class.** Keep person fine-tune scoped; do not break vehicle path.

---

## I. Implementation Plan

Small, independently testable commits. **Do not implement until this plan is reviewed.**

### Commit 1 — Contract freeze + fixtures

- Document the model contract (this report) in-repo.
- Add a golden-crop fixture test that asserts SurvNG preprocess tensor shape/dtype/color order against a synthetic crop (no weights required).
- Record expected OMZ dims (256) when a model is present; skip soft if absent.

### Commit 2 — Isolated ReID-training schema

- Sidecar tables for samples / identities / assignments / review_status.
- No changes to `appearance_embeddings` semantics.
- Migration tests.

### Commit 3 — Optional collector (config-gated)

- Hook after track completion / indexing.
- Select 3–8 samples/track (start/mid/end, largest, highest conf, embedding-diverse).
- Quality gates: min pixels, confidence, blur/clip heuristics.
- Store crop paths + metadata; never overwrite manual assignments.

### Commit 4 — Crop + embedding API

- Admin APIs to list samples, serve `crop.jpg`, read embedding/model fingerprint.
- Unit tests with fake encoder.

### Commit 5 — Review queue API (no UI yet)

- Cluster/track review actions: confirm, assign, create identity, merge, split, unknown, reject.
- Hard-positive / hard-negative flags from disagreement with model suggestion.
- Manual truth precedence tests.

### Commit 6 — Minimal Admin ReID UI

- `/admin` subsection or `/admin/reid`.
- Cluster gallery using People/face crop patterns.
- High-throughput confirm/assign/reject.

### Commit 7 — Dataset builder + versioning

- Export immutable `survng-reid-vNNN/` with `(image_path, person_id, camera_id)` plus SurvNG metadata sidecar.
- Split by **track** (never adjacent-frame random split).
- Manifest: counts, camera/identity distribution, git commit, source DB generation.

### Commit 8 — Benchmark package

- Freeze `reid-benchmark-v1` with hard cases; read-only once published.
- Dual eval modes: held-out identities vs known-identity different tracks/days.

### Commit 9 — Optional training environment

- `requirements-reid-train.txt` + `tools/reid/train.py`.
- OSNet init, PK sampler, identity + triplet loss, surveillance-safe augments.
- Does not import into production app path.

### Commit 10 — Evaluation harness

- Rank-1, Rank-5, mAP.
- TPR @ FPR 1% / 0.1% / 0.01%.
- Same/different similarity percentile tables.
- Threshold sweep table.
- Per camera-pair breakdown.

### Commit 11 — Export + parity validation

- Torch → OpenVINO FP16 IR.
- Numerical embedding parity + cosine consistency checks.
- Refuse registry publish on parity failure.

### Commit 12 — Model registry + explicit promotion

- `models/reid/production.json` + version dirs.
- Admin action to set `reid_model_path` / promote.
- Fingerprint change triggers clear operator guidance for appearance reindex/backfill.

### Commit 13 — Active learning prioritization

- Hard pos/neg, ambiguous top-2 margin, cross-camera disagreement queues.
- Builds on collector + review APIs.

### Commit 14 — Closed-loop docs + ops runbook

- How to collect → review → dataset → train → benchmark → export → promote.
- Explicit statement that training is optional and production does not require Torch.

---

## Appendix: Answers to Phase 0 checklist

### 5.1 Current ReID model

- Architecture: OSNet + LCT (`person-reidentification-retail-0286`)
- Files: `/models/person_reid_model/person-reidentification-retail-0286.{xml,bin}`
- Source: Intel OMZ 2023.0 FP16 binaries via `scripts/install-docker-models.sh`
- Framework: OpenVINO
- Input: 128×256 (W×H), BGR, float32
- Output: 256-D (OMZ; discovered dynamically)
- Init: `OpenVinoPersonReidentifier._load()` / isolated `reid` worker
- Device: AUTO→configured→CPU fallback
- Batching: 1 image per infer

### 5.2 Preprocessing

Tight bbox crop (no expansion) → size filter → `cv2.resize(..., INTER_AREA)` → float32 → optional BGR→RGB only for vehicle → NCHW transpose if needed → batch dim. No mean/std, no /255 in SurvNG code.

### 5.3 Embedding processing

Raw OpenVINO output → float32 flatten → L2 normalize in `embed()`. Tracker EMA blends then re-L2s. Index stores one **track-level** vector per `(event, track, model)`. Historical per-crop embeddings are **not** retained in the index (only recovery history similarities on the track summary).

### 5.4 Similarity / distance

Cosine **similarity** via `np.dot` / matrix multiply. Thresholds: person 0.70, vehicle 0.80; appearance index uses `max(anchor, candidate)` threshold. Lost-track ReID window: `reid_max_age_seconds` (30s). Brute-force scan of recent index rows (limit 10k), not ANN. Track appearance acts as a running prototype (EMA), not a gallery of named identities.

### 5.5–5.6 Detection → track

See §C. Tracker: `ByteTrackObjectTracker` / Hybrid. Track IDs are session-local integers. Lifetime: active until `lost_timeout_seconds` (3s), retained for ReID resume up to `reid_max_age_seconds`. ReID participates only after geometry fails. Tracks do not natively cross cameras; cross-camera is post-hoc similarity. Persistence: track fields in `events.objects_json` + optional `appearance_embeddings`.

### 5.7–5.8 Persistence / identity

See §E. Named identity exists for **faces** (`face_people`). Appearance ReID has no named identity. Extend carefully; do not create a competing People product without linkage.

### 5.9 Frontend

React + Vite; workspace paths in `workspaceNavigation.mjs`; Admin via `/admin` query subsections; People review at `/people`. APIs are FastAPI routers composed in `main.py`. Crop images via FileResponse + `image_cache`. Best reuse: face review queue + admin config ReID panel.

### 5.10 Runtime architecture for training

**Recommend option 2 + 3:** optional dependency group / separate venv (and optional container job), following MobileCLIP export. Not inside required production env. Not required for normal SurvNG operation.

---

## Verdict

SurvNG’s production person ReID is a mature **OpenVINO OSNet embedding + cosine-similarity** subsystem used for track recovery and forensic visual similarity. It is **not** a named-identity classifier. A closed-loop fine-tuning system should adapt to this contract, keep Torch optional, export OpenVINO IR, isolate training persistence, and treat face identity as a related but separate product surface.
