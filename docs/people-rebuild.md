# People: visits and recognition evidence

People → Visits reconstructs a person's visit from durable camera tracks and face
observations. It distinguishes confirmed faces, automatic face recognition,
sightings linked to an identified visit, and unresolved people. Linked sightings
never enter the trusted face gallery.

## Visit review

Select a UTC start and a 1, 6, or 24-hour window, or leave the start empty for
recent sightings. Open each sighting's incident/recording to inspect its body
evidence. Confirm suggested links, or select two sightings to mark them as the
same person/visit or keep them separate. Reset removes your relationship decision.
Separating a link also prevents the two sightings from reconnecting through a
third sighting. Existing face review controls continue to own names/enrollment.

Visits are a deterministic projection of retained evidence and persisted link
decisions, computed on demand. They are not another mutable copy of face names.
Face correction, deletion, model/evidence replacement, and retention are reflected
on the next read. Changed source evidence invalidates old relationship decisions;
renaming or correcting a face does not erase a rejection. Contradictory identities
block a previously accepted link. No visit-based alert or gallery learning runs.

The display is bounded to 400 sightings and 200 suggestions per query. Truncation
is reported and automatic association is suspended when sightings are incomplete.
Membership and the representative visit ID are scoped to the selected event-time
window; widening the window can reveal earlier anchors and change the visit ID.
These IDs must not be used as permanent person identifiers. The API supports any
positive window up to 24 hours. Existing event retention removes related decisions.

Body ReID and camera transition routes are required for automatic suggestions.
Without them, existing person detections and faces remain visible and can be
linked manually. Historical face tracks do not share body tracker IDs: automatic
ownership requires one body track, one face, one detected person and an overlapping
timestamp. Ambiguous multi-person events retain separate face/body sightings;
an operator can link the face to the correct body after inspecting the recording.

## Automatic association

The following `detector.tracking` settings control visits:

| Setting | Default | Meaning |
| --- | --- | --- |
| `visit_auto_link_enabled` | `false` | Review suggestions before enabling calibrated automatic links |
| `visit_match_threshold` | `0.85` | Minimum cosine similarity, also respecting each embedding's threshold |
| `visit_top_two_margin` | `0.08` | Minimum lead over competing candidates in each camera direction |
| `visit_min_quality` | `0.4` | Minimum producer-reported appearance quality (some producers use detector confidence) |
| `visit_max_seconds` | `1200` | Maximum whole-visit duration |

These are conservative initial settings, not calibrated accuracy guarantees.
Automatic edges require matching embedding versions/dimensions, at least two
observations, a configured directional route and plausible travel time. They
reject identity conflicts, same-camera overlap and distinct tracks in one event.
Every cross-group appearance pair must pass the threshold to prevent chained
appearance drift. Similarity is displayed as a score, never a probability.

## Enrollment and evaluation

Use confirmed, varied examples from the actual cameras and different days.
Keep a person's persistent identity separate from clothing during a visit.
Never promote inferred visit links into enrollment examples automatically.

Build a labeled corpus containing unknown visitors, lookalikes, clothing changes,
groups, crossings, night/IR and poor faces. Split by whole visit/day, including
people absent from enrollment. Tune on calibration data; evaluate on held-out
days. Measure wrong names, false visit merges, identified and unresolved visits,
review burden, latency and resource contention. Keep coverage alongside accuracy.
Start with pretrained models; fine-tune only if held-out evidence justifies it.

## Recorded face evidence

After recorded person confirmation, the pipeline ranks available person views by
crop area and visual quality. It searches up to three of those frames for faces,
then requests nearby frames at recording resolution. This runs inside the existing
decode reservation and event deadline; it does not change event qualification or
the selected incident cover. Missing recordings or optional decode failures keep
the original evidence. Cancellation remains attached to the existing evidence job.

| `detector` setting | Default | Meaning |
| --- | --- | --- |
| `face_evidence_enabled` | `true` | Enable refinement when face recognition is enabled |
| `face_evidence_max_extra_frames` | `4` | At most 0–8 extra decoded frames per sampling pass |
| `face_evidence_timeout_seconds` | `4` | Additional admission window, bounded by the existing event deadline |
| `face_embedding_profile` | `legacy_openvino` | Explicit embedding model input contract |

The time window stops new work; already admitted synchronous inference can finish
after it. Existing inference timeouts still apply. Timings include
`face_evidence_ms`, `face_evidence_samples`, `face_evidence_failed`, and decoder
counters. Inexact fallback timestamps are not accepted as extra independent frames.
Candidate collection retains at most four crops per face group and twelve per
pass, with temporal spacing. Face association stops across gaps greater than two
seconds. Source time/geometry keys prevent group renumbering from transferring an
identity between people. A changed first-frame anchor starts a separate group;
historical numbered groups remain intact and may appear separately for review.

Face consensus now weights supporting scores by image quality. Majority, minimum
quality, reference-count and margin guards still govern automatic identification.
The quality measure is a visual heuristic; it is not a calibrated face-quality
neural model or a probability of correct recognition.

## Pretrained model profiles

The legacy profile preserves existing unnormalized OpenVINO input behavior.
The `adaface` profile takes aligned 112×112 **BGR** pixels normalized by
`(pixel - 127.5) / 127.5`; `arcface` uses the same normalization with **RGB**.
Supply a local OpenVINO IR or compatible ONNX export with one image input and
one embedding output. Use an export without embedded pixel normalization for
the two normalized profiles. Multi-input or quality-conditioned research models
need their own adapters and are not supported by these profiles.

The existing five-point landmark model remains required. Invalid/out-of-crop,
collapsed or geometrically inconsistent landmarks reject the crop. Model/profile
fingerprints keep incompatible embeddings apart. Changing the profile reloads
the face engine; the existing reconciliation path refreshes embeddings and keeps
operator-confirmed labels. No model weights are bundled or downloaded, and no
new model is selected automatically. Check the chosen weights' usage terms.

Input references: [AdaFace inference](https://github.com/mk-minchul/AdaFace)
and [InsightFace ArcFace ONNX inference](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py).

## Offline enrollment and held-out evaluation

1. Review and confirm varied face crops for each enrolled person. Favor distinct
   visits, camera angles and lighting over many neighboring frames. Pin useful
   references using existing People review controls. Do not confirm a suggestion
   solely because another camera link inferred the name.
2. Build a separate, local corpus of face crops with manually verified labels.
   Reserve entire days/visits for calibration and other days/visits for held-out
   evaluation. Include unknown people in both probe splits. Use `null` for anyone
   absent from enrollment. Keep related frames within one labeled track.
3. Run each model against the same corpus. Compare wrong names, unknown false
   names and coverage together. Inspect disagreements and failure reasons before
   changing live thresholds. Benchmark resource/latency effects separately on
   the intended hardware.

The manifest has version `1` and a `samples` list. Each entry has these fields:

```json
{
  "id": "gate-alex-001",
  "path": "crops/gate-alex-001.jpg",
  "person": "Alex",
  "day": "2026-09-01",
  "visit": "visit-001",
  "track": "gate-person-1",
  "split": "enrollment"
}
```

Paths are relative to the manifest. `split` is `enrollment`, `calibration` or
`held_out`. Both probe splits must contain enrolled people and unknown visitors.
Supply tight, consistently prepared face crops, not full frames. The evaluator
uses the crop dimensions and assumes detector confidence of 1 for quality scoring.
Days, visits and duplicate decoded pixels cannot cross splits. Correct day/visit
labels and near-duplicate avoidance remain the dataset curator's responsibility.

Create a detector-only JSON file with local embedding/landmark paths,
`face_embedding_profile`, `face_recognition_device`, `face_min_size` and
`face_match_threshold`, then run from the repository:

```bash
.venv/bin/python -m survng.app.face_evaluation corpus/manifest.json \
  --detector-config corpus/adaface-model.json --output corpus/adaface-report.json
```

Reports record model, manifest and image fingerprints, failed samples, baseline
held-out results, calibration results and per-track held-out predictions. Threshold
selection uses calibration only: maximize correct names with zero observed wrong
names, declining to recommend anything if that cannot name at least one person.
This is a coarse 0.01-grid suggestion threshold experiment, not an automatic
identification calibration or a statistical guarantee of zero future errors.

The evaluator uses all supplied enrollment crops and top-three matching followed
by quality-weighted votes. Live gallery selection, reference limits, manual
rejections and automatic-identification guards are additional production policies;
offline results do not certify them. Failed probes remain in the coverage
denominator. The command never reads/writes the live gallery or changes settings.
Visit merge accuracy must also be checked on labeled camera sequences; this face
corpus tool does not measure visit association. No real-camera accuracy claim is
made until a suitable labeled corpus has been evaluated.
