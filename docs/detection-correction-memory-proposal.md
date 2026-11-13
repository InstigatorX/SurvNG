# Detection correction memory for SurvNG — revised proposal

Status: design revision following [phase 0 findings](detection-correction-memory-phase0.md).
Target: `InstigatorX/SurvNG`, `v1.2`; code investigation at `98e0fde`.
This supersedes the supplied proposal's implementation plan. It authorizes no
runtime enforcement and does not claim that calibration is complete.

## Goal and boundaries

Let an administrator review a detection or annotate a missed object, retain that
review independently of event retention, and export reliable training labels.
Later, reuse reviewed examples to suggest or apply corrections to similar future
observations without retraining the detector.

The first implementation is **record-only**. It adds review storage, APIs, UI and
training export. It does not change detection, incident establishment, alert
eligibility, tracking, or historical admission decisions. Existing explicit scene
label edits retain their existing behavior.

Reuse existing dependencies and the Smart Search encoder. Do not change detector
weights, output parsing, class-index arrays, or candidate thresholds. Do not
require Smart Search to record a review. No retroactive incident rewriting and no
below-confidence-floor rescue are included.

## What changed after investigation

The findings note contains file-and-line evidence for all twelve investigation
questions. Four findings change the implementation:

- Scene acquisition can persist observations and establish activity before the
  previously proposed late handler hook. Replay recomputes admission, and tracking
  has its own persistence and alert path. Runtime corrections need a shared
  interpretation policy across these boundaries.
- Observation identity includes the detector label. Keep raw observations and
  their identities immutable; corrected labels are interpretations of them.
- Audited scene corrections and subject label overrides already exist. Extend the
  existing review experience and revision handling rather than introducing a
  conflicting relabel workflow.
- A bounded encoder-lock acquire does not bound inference. Runtime use requires
  an end-to-end deadline, bounded outstanding work and lifecycle-safe cancellation.

The live index has a SigLIP 2 generation, but the measured label-pair similarities
are unreviewed proxies. An isolated CPU batch of eight crops took about 710 ms
at the median. Neither result establishes a safe threshold or production deadline.
`match_threshold` and `min_margin` remain unset. Motion-region matching is deferred.

## Review semantics

| Review | Phase 1 result | Possible later runtime effect |
|---|---|---|
| `not_object` | Reviewed negative region; raw detection retained | Deny that matched object's contribution to admission/alerts |
| `wrong_label` | Reviewed replacement label with original label retained | Interpret the observation under the corrected label and re-evaluate policy |
| `confirmed` | Reviewed positive annotation | Counter-evidence against suppression/relabeling |
| `missed`, hidden candidate | Reviewed positive annotation on its actual source image | Counter-evidence; no below-floor promotion in initial enforcement |
| `missed`, drawn box | Reviewed positive annotation on a retained full image | Optional future motion-region advisory only |
| `verified_negative` | Reviewed whole-image negative | None |

`not_object` does not mean that independent camera or motion evidence must be
ignored. An unlabeled incident can still be warranted. UI wording must distinguish
object suppression from suppressing all activity on a camera.

Only the exact image/observation an operator reviewed becomes `human_reviewed`.
An existing subject-level label override does not prove that every associated
observation was reviewed. Propagated matches remain model-generated pseudo-labels.

## Phase 1: record reviews and preserve their evidence

### Build on existing scene corrections

Use the existing incident correction UI and revision-conflict experience as the
entry point. A subject label edit and an exemplar review are distinct operations:

- Preserve existing scene label-edit semantics and audit records.
- Offer “Use this reviewed example for future corrections” against an explicitly
  selected observation/image. Show the selected crop before saving.
- If one UI action requests both operations, commit both records atomically and
  cross-reference their audit IDs; neither operation should silently imply the other.
- Observation-only review is supported without changing the incident's displayed
  subject label. No action silently rewrites raw evidence or prior admission.

The implementation should reuse the current manager-generation lease and admin
scope conventions. Exact router placement follows existing incident routes; avoid
another overlapping `/incidents/{id}/corrections` contract.

### Durable evidence and review records

Use the event-store mixin/migration conventions. The following are logical records;
final table names may follow existing naming conventions.

| Record | Required content |
|---|---|
| Review asset | SurvNG-owned full image, content hash, dimensions, media type, capture reference, creation time; optional owned crop and crop recipe |
| Detection correction | ID, actor/time, source camera, source event/audit IDs, raw acquisition observation ID and projected ID if applicable, source revision/image hash, original label, verdict, corrected label, source kind, asset ID, normalized box and coordinate plane, scope/location rule, note, status, supersession link |
| Snapshot review | Asset ID/hash, source audit and revision, `verified_negative` verdict, actor/time, active/revoked state and audit history |
| Correction embedding | Correction ID, implementation, model/preprocessing fingerprints, dimensions, normalized vector, creation time and crop recipe/version |
| Review audit | Immutable create, supersede, revoke and unreview actions, actor/time, prior/new record references |

A correction must survive source event, audit, observation and recording deletion.
Source references must not cascade-delete reviews or block normal retention:
retain immutable source identifiers for audit, with nullable live links if needed.

Store assets in a dedicated `corrections/` area outside snapshot/motion-sample
expiry roots. A path to an ordinary snapshot is insufficient. Retain the full image
for drawn annotations and negatives; a crop alone cannot reproduce training
coordinates. Use existing storage permissions and media-serving security. Define
backup coverage, quota and ownership explicitly instead of adding an untracked
storage role implicitly.

Prepare image bytes and hashes outside the database write transaction. Publish
assets and review records with a recoverable protocol: write to a temporary file,
atomically publish the owned file, validate source revision again while committing,
and clean up unreferenced files after a grace period. A failed request must never
leave a visible review whose required image is missing. Coordinate source copying
with retention so a missing or changed image produces an explicit conflict/error.

Phase 1 may defer embedding entirely. If implemented, embedding is asynchronous;
missing vectors do not prevent a review or training export. Encoder changes create
new vector generations from the retained assets without asking for another review.

### Conflict, undo and revision rules

- Use stable observation/image identities. `object_index` may be accepted only
  together with the exact evidence revision, resolved server-side to its source.
- Audit-image reviews require an image hash/revision too: audit snapshots can be
  coalesced or replaced. Reject stale evidence with HTTP 409.
- Permit one active direct verdict per source observation/image annotation.
  Replacing it requires explicit supersession and an audit record. Distinct
  drawn boxes retain their own annotation identities.
- A verified-negative image cannot simultaneously have active positive annotations.
  Adding a positive requires explicitly replacing the negative review; marking
  negative requires explicitly revoking the positives, all in one transaction.
- Revoke/unreview is a new audited state transition. It affects active training
  export and future matching, not historical runtime decisions. Undoing a review
  does not silently undo a separate scene label edit.
- Support idempotency keys for creation so a retry cannot duplicate annotations.
- Revoke does not immediately delete owned evidence. Document a separate asset
  retention/deletion policy and only reclaim assets with no remaining owners.

### API and UI outcomes

Use admin writes and read-scoped listing/media access. Required operations are:

1. Create a review for an event/source observation or draw on a specific event image.
2. Create a drawn annotation or verified-negative review on a motion-audit image.
3. List/filter/page reviews; show original and corrected labels, source evidence,
   actor/time, status, and any existing scene-correction relationship.
4. Revoke, supersede, or unreview with revision checks.
5. Page an image review queue; prioritize existing probable-miss AI verdicts where
   available, without presenting them as human judgments.
6. Serve retained full images and crops through authenticated media routes.

Keep controls local to the incident card and owning admin views. A hidden-candidate
viewer must resolve the candidate's own image, dimensions and capture time.
Off-cover candidates cannot be drawn on the representative cover. Below-candidate-
floor detections are unavailable; drawing remains the fallback.

Render loading, missing-source, stale-revision, empty-list and failed-save states.
Reviews remain available after source retention through their owned assets.
Embedding status, if present, says pending/unavailable without suggesting the
review failed. Runtime mode remains off; the UI must not promise immediate learning.

### Label vocabulary

Allow reviewed labels outside detector classes, as strings with explicit length,
normalization and validation rules. Preserve display spelling while using a stable
normalized key for matching/policy. Reuse existing scene facet labels where useful.
Never append these labels to the detector's class-index array.

Phase 1 can export such labels without enabling runtime relabeling. Before runtime
use, provide a shared vocabulary for zone editors, incident facets and integrations,
and define class-specific confidence/temporal/speech/re-ID behavior. A score for
one detector class is not automatically calibrated confidence for a new class.

### Training manifest contract

Preserve the existing response when no relevant active reviews exist. Add optional
fields only for reviewed or propagated annotations; do not add null fields to every
legacy row if byte-for-byte compatibility is the acceptance criterion.

- Direct reviews: `annotation_state=human_reviewed`, `review_verdict`,
  `original_label`, `correction_id`, and explicit annotation source.
- `not_object` remains a returned negative region, never silently filtered out.
- Drawn boxes use `annotation_source=operator_drawn` and their retained full image.
- Verified negatives keep `sample_kind=negative_candidate`, become human-reviewed,
  set `assumed_negative=false` and `verified_negative=true`.
- A reviewed audit image with positive boxes becomes an annotated sample.
- A propagated match stays `model_generated`, with the applied correction reference.
- Add `review=any|reviewed`, default `any`. Define reviewed filtering at annotation
  and sample level so unrelated model-generated annotations cannot become reviewed.
- Sample IDs/revisions must cover the owned image and active review generation.
  Separate samples when annotations belong to different images. Source expiry must
  not remove review-owned samples from export or break pagination.

Document filter precedence: explicit negative regions must remain retrievable even
when their original detector label/eligibility would otherwise exclude them. Define
this alongside reviewed filtering and preserve legacy filter behavior for untouched
samples. Update `docs/training-api.md` and `CHANGELOG.md` with the implementation.

### Phase 1 acceptance tests

- Create/list/supersede/revoke reviews, including authorization, retry idempotency,
  stale revision rejection and concurrent contradictory submissions.
- Delete/expire source evidence: correction assets and training samples survive.
- Replace a cover or audit image: existing boxes stay on their original image.
- Export direct positives, negative regions, hidden candidates, drawn positives and
  verified negatives with correct provenance; unreview/revoke updates export.
- Prove existing manifest output is unchanged when no applicable reviews exist.
- Verify snapshot/recording/maintenance cleanup cannot delete correction-owned media;
  exercise failure between file publication and database commit.
- Verify phase 1 produces no changes to detection, admission, replay, tracking or
  notifications, apart from separately requested existing scene label edits.

## Calibration workstream: required before runtime matching

Use the [measurement harness](../scripts/measure-detection-corrections.py) and
follow the [phase 0 protocol](detection-correction-memory-phase0.md#measurement-tool-and-reproduction).
Recording reviews can proceed while this work remains incomplete.

Collect reviewed cross-event physical identities and verdicts on the same camera:
same object, different look-alikes with the same class, different classes, and
nuisance objects versus real threats. Include confirmed counterexamples, varied
lighting, crop sizes and capture sources. Do not infer ground truth from current
similarity, detector labels, or automatic tracking associations.

Use disjoint subjects/days for calibration and held-out evaluation. Compare
matching queries with their strongest relevant opposing verdict, applying the same
label, camera, location and scope gates intended for production. Measure operating
error rates as well as cosine distributions; report sample sizes and uncertainty.
An owner-agreed false-suppression target is needed before selecting thresholds.

Measure the actual deployment device, total request latency and live encoder-lock
waits under backfill. Controlled offline contention is not live lock telemetry.
Measure decision-time versus saved-image crop drift and verified spatial/temporal
motion-region pairs. Preserve source hashes and crop recipes in the case ledger.

Deliver a threshold/margin recommendation per encoder generation or an explicit
no-go. Same-frame motion-crop agreement alone cannot approve cross-event advisory
matching. Do not use the existing unreviewed proxy quantiles as config defaults.

## Later runtime architecture

### Immutable evidence, versioned interpretations

Introduce a shared interpretation operation over raw acquisition observations.
Inputs include raw source identity, immutable match decision, effective-label policy
and the relevant snapshotted admission policy. Outputs carry effective label,
object-level eligibility/reasons and correction provenance without mutating raw
label, confidence, box or identity.

Integrate the same operation into early scene establishment, normal decision
handling, refinement, tracking projection and replay. Enumerate these entry points
and test them before promising next-event coverage. The implementation must respect
the separation between physical activity, incident establishment and notifications.

Perform matching only while the correct source pixels are available. Persist the
result before any affected interpretation can be published. Replay consumes the
recorded result; it does not re-encode old images or apply today's correction set.
Record explicit no-match/skipped results where needed for deterministic replay.
New source observations may receive new matches; refinement of already interpreted
source observations must not silently replace their historical interpretation.

Record correction ID/revision, verdict, effective label, similarity, scope,
location rule, policy version, encoder identity, correction generation and mode.
Revocation affects newly admitted matching work. Each decision uses one immutable
correction generation; after revocation commits, newly started decisions must see
the new generation. Already-started decisions may finish against their recorded
generation, which must remain visible in audit.

### Pure matcher and asynchronous ownership

Keep matching itself free of I/O. Build immutable normalized vector matrices off
the decision path, isolated by model and preprocessing fingerprints. Store reloads,
asset reads, encoding and aggregated match-counter writes belong to separate owners.
Set bounded active-example and per-request crop limits; choose actual values during
implementation/calibration rather than treating suggestions as measured capacity.

Apply explicit detector-label gates to suppress/relabel examples. Positive
counterexamples are relevant only to the same candidate label/task. Camera-scoped
examples are the default; global scope is explicit. Location rules require the same
camera and valid normalized geometry. Define deterministic scope arbitration,
opposing-verdict groups, ties, top-k behavior and margin rules before implementation.
Camera priority must not hide a relevant confirmed counterexample. Conflicting or
ambiguous evidence produces no enforcement.

Resolve the encoder through lifecycle-safe access. Bound queue depth, outstanding
requests, lock wait and total completion deadline. Expired requests must not apply
late results or create an accumulating backlog. Reconfiguration cannot combine
vectors from different generations. Fail open on disabled/unavailable encoders,
contention, timeout and errors, with scoped/rate-limited diagnostics. Exclude Ollama
from enforcement until a separate deadline/availability assessment supports it.

### Modes and gates

| Phase | Behavior | Gate |
|---|---|---|
| 2: shadow | Record proposed matches; preserve labels, incident/alert eligibility and publication behavior | Shared source mapping, bounded inference, calibrated candidate thresholds, failure tests |
| 3: enforce | Suppress/relabel object contributions; re-evaluate all applicable policy | Held-out error target met; replay/tracking/refinement coverage; revocation and counterexample tests |
| 4: missed advisory | Match verified motion-region crops to missed exemplars; show advisory only | Separate cross-event motion-region calibration demonstrates utility |
| 5: optional | Retirement after model deployment and other separately designed features | Explicit follow-up scope and validation |

Shadow summaries must be sidecar diagnostics that do not cause incident creation,
evidence replacement or notification churn. Test off-versus-shadow equivalence
across all integration points, not just the final handler list.

Enforcement re-evaluates label-dependent zones, confidence policy and temporal
requirements while preserving depth, geometry, stationary, auxiliary and motion
restrictions. Confirmed exemplars block conflicting negative/relabel decisions;
they do not override the confidence floor. Global rules and new-label runtime
policy need explicit support before being enabled.

Motion-region matches never establish incidents or raise alerts. They are advisory
metadata on retained evidence and remain distinct from existing physical-motion
admission. Retirement must keep reviews/training labels and historical decision
provenance even when an exemplar is no longer active.

## Proposed defaults and unresolved release decisions

Use admin-only writes, camera scope, runtime off, no below-floor rescue, no
retroactive decisions and advisory-only motion matching. Default `not_object` to
appearance plus same-camera location if/when runtime matching is enabled. These
are design recommendations; no existing deployment setting changes with this document.

Before enforcement, settle the tolerated false-suppression rate, total latency
budget, global/camera arbitration, new-label confidence policy and scope of any
non-object incident suppression request. Before phase 1 release, settle review
asset quota/retention and confirm API/filter details in the implementation review.

The next implementation task is phase 1 only, using the acceptance tests above.
Calibration remains a separate prerequisite for runtime work. Stop and report at
each phase boundary; do not enable matching as part of recording reviews.
