# Sparse Identity (production default)

`survng_sparse_identity` is SurvNG's promoted production tracker for identity
retention testing. Hybrid remains selectable. Compare and `tracking_evaluation`
still score both engines on the same saved detections and embeddings.

## Behavior

- Cascaded association: confirmed×high → contested appearance veto → buffered IoU
  → low-confidence geometry (no gallery update) → appearance (active before lost)
  → optional entity relink for long gaps
- Observation-centric velocity reset after gaps
- Quality-gated multi-prototype appearance gallery with top-two margin
- Co-occlusion freeze of gallery updates when boxes heavily overlap
- Completed-track prune beyond `reid_max_age_seconds`
- Track ID stays trajectory-local; `entity_id` can alias a new tracklet after a
  long gap when `entity_relink_enabled` / `tracking_profile=person_retention`

## OpenVINO ReID upgrade (after association gates pass)

1. Keep the association/memory pipeline fixed.
2. Install the torchreid OSNet-x1.0 MSMT17 IR with
   `scripts/install-person-reid-model.sh`.
3. Point `reid_model_path` at `models/person_reid_model/osnet_x1_0_msmt17.xml`;
   recalibrate `reid_match_threshold` on labeled same-camera pairs from the corpus.
4. Re-run held-out Compare profiles; accept only if false merges do not rise.
5. Do not swap backbones to chase benchmark mAP without crop-quality gates.

Recommended starting models for CPU NVRs: OSNet-x0.25 / x0.5 style OpenVINO
exports at ~256×128. Prefer Intel retail person ReID IR when already validated
on the fleet.

## Scene zones and cross-camera

Camera transition routes already constrain investigation linking. Detection
`zones` on tracks are used as soft compatibility during entity relink. Explicit
entrance/exit/occluder polygons remain future work; prefer SurvNG camera zones
named for those roles until then.

Cross-camera identity stays investigation-grade via the appearance index and
`camera_transition_routes`. Live track IDs are never merged across cameras.
