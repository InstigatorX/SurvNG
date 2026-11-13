# Detection correction memory: phase 0 findings

Investigated 2026-11-12 at `v1.2` commit `98e0fde`, on
`investigation/detection-correction-memory-phase0`. The supplied proposal was
based on `ec8b13e`; references below describe the current checkout. No runtime,
configuration, database schema, API, or frontend behavior was changed.

The resulting design revision is in
[detection-correction-memory-proposal.md](detection-correction-memory-proposal.md).

**Decision: revise the design before phase 1. Recording reviewed examples is
feasible; enforcement and motion-region matching are not cleared.** The code
investigation and measurement harness are complete. Calibration is incomplete:
there is no reviewed physical-identity benchmark, no live encoder-lock timing,
and no verified motion-region/crop comparison. `match_threshold` and `min_margin`
remain **unset**, not guessed defaults. Do not advance to phase 1 automatically.

## Answers to the twelve investigation questions

1. **Replay consistency: a late object stamp is insufficient.**
   [`event_store/scene_admission.py:207`](../survng/app/event_store/scene_admission.py#L207)
   loads acquired samples and calls `evaluate_scene_establishment`, rather than
   accepting final `incident_eligible` as the establishment verdict.
   [`scene_zone_admission.py:224`](../survng/app/scene_zone_admission.py#L224)
   reconstructs spatial and confidence admission from label, box, depth, and
   snapshotted policy. More critically,
   [`decision_handler.py:513`](../survng/app/motion_pipeline/decision_handler.py#L513)
   copies scene observations **before** attribution/rescue; at line 571 it can
   acquire evidence, establish activity, and return before the proposed hook.
   [`scene_acquisition.py:24`](../survng/app/motion_pipeline/scene_acquisition.py#L24)
   owns that early persistence/evaluation path. Corrections need a versioned
   interpretation of immutable source observations, consumed by both initial
   establishment and later replay. Object suppression must not automatically
   erase independent physical activity: the evaluator explicitly separates
   activity from labels ([`scene_activity.py:1`](../survng/app/scene_activity.py#L1)).
   Current motion evidence can establish an **unlabeled** incident
   ([`scene_zone_admission.py:749`](../survng/app/scene_zone_admission.py#L749)).
   Define suppression as denying that object's admission/alert contribution,
   not a blanket promise that no incident can exist.

2. **Label-dependent policy is wider than an allowlist.**
   [`object_detection.py:2812`](../survng/app/motion_pipeline/object_detection.py#L2812)
   reads global/per-class confidence and candidate floors, then applies zones
   at line 2870. [`zones.py:12`](../survng/app/zones.py#L12) matches
   `zone.object_classes`; `apply_detection_zones` at line 86 applies foot-point
   geometry, ignore/incident behavior, class/zone confidence, and camera zone
   requirements. Depth filtering follows. No separate per-camera detector-label
   allowlist was found in this path. Per-label temporal confirmation counts
   also affect admission ([`object_detection.py:859`](../survng/app/motion_pipeline/object_detection.py#L859)).
   Relabeling must recompute these interpretations without losing temporal,
   geometry, auxiliary-face, stationary-context, and motion restrictions.
   Establishment deliberately omits notification preferences and zone confidence
   overrides ([`scene_zone_admission.py:61`](../survng/app/scene_zone_admission.py#L61));
   alert/notification policy is separate
   ([`manager.py:1783`](../survng/app/manager.py#L1783)). A dog detector score is
   not calibrated confidence for an operator-invented class; specify a policy
   for that rather than silently treating it as one.

3. **Process placement: reachable, but lifecycle ownership matters.**
   [`motion_decisions.py:180`](../survng/app/motion_decisions.py#L180) starts a
   thread with bound orchestrator callbacks, not a separate application process.
   [`manager.py:403`](../survng/app/manager.py#L403) owns the handler factory;
   [`inference_lifecycle.py:98`](../survng/app/inference_lifecycle.py#L98) constructs
   Smart Search in the manager process. `evidence_work.py:1` describes a
   cancellation context, not serialization of the decision handler to a worker.
   OpenVINO computation itself uses a child process via the encoder proxy
   ([`semantic_search.py:2443`](../survng/app/semantic_search.py#L2443)). Inject a
   lifecycle-safe accessor/lease, not a permanently cached service instance:
   [`inference_lifecycle.py:413`](../survng/app/inference_lifecycle.py#L413)
   replaces it on reconfigure. Index refresh already uses an accessor in
   [`manager.py:414`](../survng/app/manager.py#L414).

4. **Retention: a separate correction-owned directory works, with caveats.**
   [`database_retention.py:23`](../survng/app/database_retention.py#L23) explains
   audit aging and independent acquisition-image retention; its deletion owner
   registry is at line 62. Snapshot expiry is in
   [`store.py:1794`](../survng/app/event_store/store.py#L1794), explicit
   unreferenced deletion at line 1899. Maintenance scans snapshot and
   motion-sample roots ([`storage_maintenance.py:505`](../survng/app/storage_maintenance.py#L505)).
   A new sibling `corrections/` is outside these existing scans, but a pointer to
   an ordinary snapshot is not. Copy the crop **and the full annotation image**
   to dedicated owned storage; retain dimensions, content identity, capture
   reference, and source revision. Source event/observation foreign keys must
   not cascade-delete corrections or prevent ordinary source retention. Source
   IDs may need nullable links plus immutable audit values. Define orphan
   cleanup, quota/backup, atomic write+row publication, and revoke versus asset
   deletion explicitly; the existing media role enum has no corrections role
   ([`config.py:499`](../survng/app/config.py#L499)).

5. **Tracking bypasses the handler and can publish labels/alerts.**
   The old `object_tracking.py` is now a re-export facade. The substantive path
   is [`object_track/session.py:358`](../survng/app/object_track/session.py#L358):
   `_commit_tracking` calls `update_event`. At line 1336 tracking independently
   stamps `alert_eligible`. The callback is wired directly to
   `events.update_object_tracking`
   ([`inference_lifecycle.py:481`](../survng/app/inference_lifecycle.py#L481)).
   [`store.py:1945`](../survng/app/event_store/store.py#L1945) commits tracking
   and [`evidence.py:139`](../survng/app/event_store/evidence.py#L139) projects it
   through scene admission. Demand-only tracking suppresses activity/notify;
   ordinary tracking need not. Handler-only corrections therefore do **not**
   cover all incident decisions or notifications. Either share the interpretation
   policy at projection boundaries or explicitly label tracking as uncorrected;
   do not promise universal next-event behavior with the proposed hook alone.

6. **Ollama: exclude from enforce. Local encoding is not yet bounded either.**
   [`semantic_search.py:2707`](../survng/app/semantic_search.py#L2707) encodes
   images through HTTP; line 2778 uses the configured HTTP timeout. This is not
   an event-level admission deadline. The local proxy at line 2555 can wait
   **60 seconds per request**, restart, and retry; startup/cleanup add time.
   Lock acquisition bounds only queue entry. Introduce a genuinely bounded,
   capacity-limited optional inference request with cancellation/generation
   safety before enforcement. A timed-out Python future cannot cancel a native
   inference operation and must not leave unbounded jobs behind. “Never delayed”
   is incompatible with synchronous encoding; specify a small total budget and
   fail open when it expires. The current benchmark does not approve that budget.

7. **Observation identity must remain the raw source identity.**
   [`scene_identity.py:6`](../survng/app/scene_identity.py#L6) hashes label.
   [`scene_acquisition.py:284`](../survng/app/event_store/scene_acquisition.py#L284)
   uses it for immutable acquisition and sample membership.
   [`scenes.py:722`](../survng/app/event_store/scenes.py#L722) uses it for
   projected observations and `source_observation_id`; association, zone and
   decision tables reference those IDs. Semantic vectors also carry
   observation IDs ([`semantic_search.py:414`](../survng/app/semantic_search.py#L414)).
   Preserve the raw label and ID, attach an effective-label/verdict
   interpretation, and retain the acquisition-to-projection mapping. Re-hashing
   a changed label can duplicate evidence and break joins. Merely saving
   `detector_label` does not fix the existing identity function.

8. **Refinement shares the handler, but not every path reaches the hook.**
   [`decision_handler.py:383`](../survng/app/motion_pipeline/decision_handler.py#L383)
   calls `_handle_with_provider`, as initial `handle` does. A hook there covers
   ordinary successful refinement, but early acquisition/return paths above,
   preserved-evidence failures (line 597), and tracking remain separate. Match
   the actual source frame for each observation; later cover replacement must
   not reinterpret an old box against a different image. Version each match
   decision and retain it for replay; revocation should change future decisions,
   not silently rewrite historical decisions.

9. **Motion regions are normalized motion-blob history, not object boxes.**
   [`adaptive_stages.py:668`](../survng/app/motion_pipeline/adaptive_stages.py#L668)
   extracts connected components and stores normalized xyxy boxes at line 686.
   [`motion_qualification_service.py:696`](../survng/app/motion_qualification_service.py#L696)
   retains up to twelve observations from the dominant motion track. These can
   span time, fragment a subject, include changing background, or follow a
   subject's path. [`motion_correlation.py:41`](../survng/app/motion_correlation.py#L41)
   applies scale/offset between motion and detection planes and checks reliable
   alignment; intersection is deliberately permissive (0.035 margin, line 17).
   Qualification features persist with events; audit features are stored through
   [`decision_handler.py:1136`](../survng/app/motion_pipeline/decision_handler.py#L1136)
   and [`motion_intelligence.py:234`](../survng/app/event_store/motion_intelligence.py#L234).
   Regions are optional. They do not guarantee exact temporal alignment with a
   stored cover. Verify both spatial and capture alignment before calibration.

10. **Audit images vary, can disappear, and do not require the recording.**
    [`camera_media.py:128`](../survng/app/camera_media.py#L128) stores sampled
    rejected-motion frames; it prunes to **100 images per camera** (line 223).
    Other audits point at decision snapshots from the selected detection frame
    ([`decision_handler.py:750`](../survng/app/motion_pipeline/decision_handler.py#L750)).
    [`image_storage.py:47`](../survng/app/image_storage.py#L47) encodes the input
    frame without resizing, so resolution is producer-dependent, not uniformly
    native main-stream resolution. Ordinary snapshots have `snapshot_days`
    retention (default 1095); unlinked audit rows have `motion_audit_days`
    (default 30), plus rejected-sample pruning and storage cleanup. These are
    configuration defaults, not assertions about current effective retention.
    Stored images can be reviewed after recordings expire if the image still
    exists. `image_source=main_recording` cannot reconstruct an expired recording.
    The manifest skips missing stored files
    ([`training_routes.py:412`](../survng/app/training_routes.py#L412)). Pin/copy
    review evidence at annotation time; an audit ID alone is not an immutable
    image revision (audits can coalesce/replace snapshots in
    [`motion_intelligence.py:115`](../survng/app/event_store/motion_intelligence.py#L115)).

11. **Hidden candidates exist, but not for every path or on one image plane.**
    Detection runs at the candidate floor
    ([`object_detection.py:2824`](../survng/app/motion_pipeline/object_detection.py#L2824));
    discarded below-floor outputs cannot be recovered. Scene observation
    validation rejects invalid/under-candidate-floor boxes
    ([`scene_evidence.py:21`](../survng/app/motion_pipeline/scene_evidence.py#L21)).
    Multi-sample observations distinguish off-cover detections with
    `snapshot_visible=False`
    ([`object_detection.py:1140`](../survng/app/motion_pipeline/object_detection.py#L1140),
    line 2551). They may have different dimensions and capture references.
    [`training_routes.py:183`](../survng/app/training_routes.py#L183) exports only
    suitable top-level, snapshot-visible annotations with known dimensions;
    ineligible filtering does not imply complete candidate export. Some failed
    or empty decisions retain only acquisition/audit evidence, not an event
    candidate list. A hidden-candidate UI must resolve each observation's image
    and plane; never draw all retained boxes on the incident cover.

12. **New labels are already partly supported, but no unified managed vocabulary exists.**
    Contrary to the proposal, detector_labels.py loads model class metadata;
    it is not a general correction-label normalizer
    ([`detector_labels.py:49`](../survng/app/detector_labels.py#L49)). Detector
    tensor parsing relies on model class counts
    ([`detector.py:775`](../survng/app/detector.py#L775)); do not add operator
    labels to that array. Zone matching accepts strings (zones.py:12), config
    per-class thresholds normalize strings (config.py:1102), and training
    annotation labels are strings (training_routes.py:33). Existing scene
    `label_override` accepts any 1–100 character label and updates facet labels
    ([`scenes.py:1757`](../survng/app/event_store/scenes.py#L1757)). However,
    zone editors use model classes/configured detector labels
    ([`ConfigPage.jsx:1452`](../frontend/src/admin/ConfigPage.jsx#L1452)); manager
    zone publication defaults to detector labels (manager.py:1694,1718), and MQTT
    discovery assembles model/zone classes (mqtt.py:781). Tracking/re-ID has
    person/vehicle/face-specific handling (config.py:871,879); temporal
    confirmation and speech subject logic also depend on label. Use a separate
    effective-label vocabulary shared by editors, facets, zones and integrations;
    preserve class-specific restrictions. Existing facet labels are a useful
    precedent, not yet the proposed managed vocabulary.

## Corrections to the proposal and required design changes

- “No human-correction state anywhere” is false at this revision. There is
  `scene_corrections`, transactional revision checking, scene label overrides,
  and `POST /api/incidents/{incident_id}/corrections`
  ([`incident_queries.py:818`](../survng/app/incident_queries.py#L818),
  [`scenes.py:1695`](../survng/app/event_store/scenes.py#L1695)). This is not yet
  appearance memory or reviewed training export. Reuse its evidence/projection
  separation and settle how subject-level labels map to directly reviewed
  observations before building another conflicting correction UI.
- The single late handler stage cannot enforce the described behavior across
  acquisition, replay, refinement and tracking. Define a shared interpretation
  boundary while keeping original evidence/IDs immutable. Phase 1 can remain
  record-only, but its schema must support that boundary.
- Add an immutable full-image asset reference and revision/hash to corrections
  and snapshot reviews, not only `crop_path`. A crop alone cannot export a drawn
  annotation in its original full-image coordinate plane after retention.
  `verified_negative` needs revision checks, revocation/unreview, and a defined
  conflict rule when drawn positives exist on the same image.
- Separate create/review audit records from active appearance exemplars. Specify
  how duplicate or contradictory flags supersede each other, scope precedence,
  which verdicts oppose one another, and positive-label gates. Otherwise a
  “confirmed dog” could inadvertently block an unrelated person's correction.
  Persist applied verdict/policy data independently of the current active row.
- Keep matcher matrix reloads, crop I/O, embedding and audit-counter writes
  outside the pure cosine matcher. Publish immutable generation snapshots;
  revocation and reconfiguration need an atomic generation boundary. Do not
  retain a stale closed encoder through the handler factory.
- Keep `off` default, admin writes, camera default scope, no below-floor rescue,
  and advisory-only missed matching as recommendations, not owner-approved
  decisions. Explicitly decide independent-motion incident behavior. No phase 1
  runtime matching is justified by this investigation.

## Measurements obtained here

`./survngctl status` initially failed with ENOENT; a later retry succeeded and
reported Smart Search enabled. No service was restarted or reconfigured.
Persisted config was used only to locate the database/model assets. The database
was queried with SQLite `mode=ro`, `query_only`, and a read transaction; the
application's `SemanticIndex` constructor was deliberately avoided because it
initializes/migrates schema.

The index contained **222,290 object crops** (plus 64,767 full-frame rows), all
in one **SigLIP 2, 768-dimensional** generation:
`542add7f7d476199508c3bd3:c38b795c01361ecaaa411a8d`.
There was no MobileCLIP2 index generation to compare. The index-only run sampled
the latest 2,000 object rows (IDs 819727–823267) and 10,000 seeded random pair
attempts, retaining unique same-camera, cross-event pairs. Float16 blobs were
converted to float32 and renormalized. This is a recent-window convenience
sample, not independent, balanced, reviewed ground truth.

Representative results (camera aliases intentionally used):

| Camera | Unreviewed detector-label relationship | Pairs | p05 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| A | same label, physical identity unknown | 530 | .688 | .782 | .919 | 1.000 |
| A | different detector labels | 98 | .608 | .678 | .740 | .812 |
| B | same label, physical identity unknown | 197 | .728 | .853 | .928 | .964 |
| B | different detector labels | 5 | .785 | .827 | .935 | .961 |

These are **not** distributions of “same physical object” and “different
physical objects with the same label.” Model mistakes and shared source frames
across events can contaminate both proxies. No reviewed pairs or verdict
triplets were supplied. Consequently neither a safe threshold nor an opposing
neighbor margin can be estimated from this run. Do not interpret .961 as a
recommended cutoff or as a verified false match.

An isolated local CPU encoder benchmark used eight indexed crops from retained
2560×1920 images, batch size 8, five warm repetitions, and verified pinned
SigLIP 2 artifacts. It did not call the live service encoder or write its index.

| Measurement | Result |
|---|---:|
| First encode (after encoder construction) | 701.98 ms |
| Warm batch median / p95 / max | 710.05 / 777.48 / 793.98 ms |
| Controlled backfill lock timeout | 50 ms |
| Lock timeouts | 5/5 |
| Observed bounded waits, median / max | 50.084 / 50.093 ms |

Construction/model loading and image decoding are excluded from encode timing.
The competing worker held a separate benchmark RLock for one real encode batch
per trial. This is **controlled offline contention**, not live backfill lock
latency, fairness, GPU performance, or a production timeout recommendation.
The initial generic-manifest benchmark attempt failed because SigLIP 2 uses
pinned ONNX artifacts rather than the generic XML/BIN identity calculation;
the harness now validates the pinned package and uses its existing contract.
No model files were changed or downloaded.

The selected recent events had no motion regions suitable for verified paired
calibration. Motion-crop comparisons: **0**. Flag-time versus decision-time crop
drift is also unmeasured; the index contains persisted evidence, not guaranteed
copies of the exact live pixels. Phase 4 remains **no-go pending measurement**.
Even high same-frame detector/motion cosine would need held-out cross-event
retrieval evaluation before supporting an advisory feature.

## Measurement tool and reproduction

[`scripts/measure-detection-corrections.py`](../scripts/measure-detection-corrections.py)
uses existing NumPy/OpenCV/encoder dependencies and never builds or refreshes an
index. Reports have identity-isolated, per-camera distributions; invalid vectors,
missing reviewed IDs and incompatible generations fail explicitly. Existing output
files cannot be overwritten. No runtime hook or debug instrumentation is installed.

Index-only measurement:

```bash
.venv/bin/python scripts/measure-detection-corrections.py \
  --database /path/to/survng.sqlite3 --limit 2000 --pair-attempts 10000 \
  --seed 0 --output /tmp/correction-index-new.json
```

Optional reviewed cases JSON (IDs and paths below are placeholders):

```json
{
  "pairs": [
    {"a": 101, "b": 202, "category": "same_object", "reviewer": "operator"},
    {"a": 101, "b": 303, "category": "different_object_same_label", "reviewer": "operator"},
    {"a": 101, "b": 404, "category": "different_label", "reviewer": "operator"}
  ],
  "triplets": [
    {"query": 101, "positive": 202, "opposing": 303, "reviewer": "operator"}
  ],
  "crops": [
    {
      "event_id": 123,
      "image": "/path/to/exact-source-image.jpg",
      "coordinate_plane": [1920, 1080],
      "detector_box": [192, 108, 768, 864],
      "motion_box": [0.08, 0.08, 0.45, 0.85],
      "alignment_verified": true
    }
  ]
}
```

Pair and triplet IDs are **semantic embedding row IDs**, not event IDs. Review
actual image content; do not derive identity from cosine, class label, temporal
proximity, or automatically associated scene-object IDs. Pair categories describe
reviewed reality. Triplets identify the same-label-gated/scoped query's intended
winning verdict and its strongest opposing verdict, with the query from a
held-out event. Use disjoint physical subjects/days for calibration and evaluation,
include look-alike real threats versus nuisance objects, and balance cameras,
lighting and crop sizes. Keep an external reviewed case ledger with image hashes,
source IDs, encoder identity, capture time and split assignment. This harness
reports distributions/margins; it deliberately does not auto-approve thresholds.

For crop cases, `coordinate_plane` is the detector box's width/height, not
necessarily the stored image's dimensions. The tool uses Smart Search's scaled,
rounded, clipped crop geometry. `motion_box` is optional; when present it must
already be transformed into this image's normalized plane, overlap the detector
box, and have verified spatial **and temporal** alignment. Use source observation
images for off-cover detections. The tool measures paired cosine and loss from
identical-crop similarity (`1 - cosine`), not classifier accuracy.

```bash
.venv/bin/python scripts/measure-detection-corrections.py \
  --database /path/to/survng.sqlite3 --cases /tmp/reviewed-cases.json \
  --model-dir /path/to/existing/semantic/package --device CPU \
  --batch-size 8 --repeats 10 --lock-timeout-ms 50 \
  --output /tmp/correction-measurement-new.json
```

Local MobileCLIP/OpenVINO manifests and the existing pinned SigLIP 2 package are
supported. Run each encoder separately with its own indexed generation and
reviewed rows; the tool never compares incompatible vectors or downloads models.
Ollama is intentionally not benchmarked for enforcement. Run hardware benchmarks
in a quiet window and compare the actual deployment device before selecting a
budget. A separate process cannot access the service's RLock: actual under-backfill
wait telemetry remains an explicit follow-up requiring an approved diagnostic
surface or temporary in-process instrumentation, not credential extraction or an
unauthenticated endpoint.

To finish empirical phase 0: collect reviewed cross-event pair/triplet cohorts;
run exact-source detector/motion and decision/stored-image comparisons; obtain
actual-device total latency and live lock waits under backfill; set an owner-agreed
false-suppression target; sweep thresholds/margins on calibration data and validate
on the held-out split. Report sample sizes and error bounds before any enforce
recommendation. Until then, threshold and margin are null and phase 4 is deferred.

## Validation

Focused tests: `scripts/run-tests.sh -q tests/test_detection_correction_measurements.py`.
Coverage includes read-only database preservation, generation isolation, reviewed
margins, missing/self-pair evidence rejection, corrupt vectors, coordinate scaling,
and contention-worker error propagation. Real index sampling and isolated pinned
SigLIP 2 encoding were also exercised. No runtime behavior changed, so application
integration tests were not broadened.
