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
| `face_embedding_profile` | `openvino_raw` | Explicit embedding model input contract |

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

The `openvino_raw` profile uses unnormalized OpenVINO input pixels. It replaces
the old `legacy_openvino` name without changing preprocessing or fingerprints.
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

## Face funnel report

To see where faces are lost on each camera, run this on the SurvNG host as the
service user, giving the directory that holds `survng.sqlite3`:

```bash
.venv/bin/python -m survng.app.face_funnel /path/to/database-dir --days 14
```

It opens the database read-only, loads no model and changes nothing. For each
camera it counts the recorded person events that reached each stage: a person
large enough for a face, a face found, kept, embedded, suggested and named. It
also prints person and face sizes in pixels and the stored faces by outcome
(`too_small`, `failed`, `embedded`). `--json` prints the same data as JSON.

The "large enough" stage is an estimate from the person's height, assuming a
face about 9% as wide as a standing person is tall. Stored faces are pruned at
`face_max_observations`, so the later stages undercount for older events; the
report says how far back stored faces reach.

Events recorded from this version on also carry face-search counters, shown as
"Search detail": people examined, people skipped over the per-frame cap, people
too small to crop, crops searched, crops where the detector found no face, and
faces found. A crop the detector could not search because inference was busy
or failed counts as "no face"; the two are not yet told apart.

## Every-frame face search comparison

To measure how much the frame choice costs, run this on the SurvNG host as the
service user. It is CPU-heavy, so keep it niced and bounded:

```bash
nice -n 19 ionice -c3 .venv/bin/python -m survng.app.face_frame_search \
    runtime/database --cameras boiler,gate --events-per-camera 10 --margin 3
```

For the most recent recorded person events on each camera it decodes every
frame of the event window from the main recording (software decode, full size,
through the shared recorded-frame decoder), runs the configured person detector
and the existing upper-body face search on each frame, and reports the best
face per event three ways:

- **old**: the refinement plan frames plus the face-evidence extras the old
  search would have added around its best anchors. Every plan frame counts as
  decoded and the face-evidence time budget is ignored, so this is an upper
  bound for the old search.
- **plan**: every frame in the window the old search could reach.
- **all**: every frame including `--margin` seconds either side.

A face is usable when its shorter side reaches `face_min_size`. Each result is
given twice: for any detected face, and for recognisable faces only. The 300 px
face detector also fires on the backs of heads, caps, torsos and leaves, and
the five-point landmark model places points on all of them, so neither
confidence nor landmark geometry separates them. Each face is instead aligned
exactly as the live path aligns it and scored with the AdaFace feature norm,
which on this host was 16-18 for real faces and 9-11 for those false hits. A
face counts as recognisable from `--min-norm` (default 13); one that fails
alignment scores 0. The norm comes from
`models/face_model/adaface-ir50-webface4m/adaface-ir50-webface4m-norm.xml`, an
OpenVINO export with both outputs made by `export_openvino_norm.py` in that
directory (the embedding-only export drops the norm); it matches PyTorch to a
cosine of 0.99999 and a norm within 0.1% on CPU at f32. `--sheets DIR` writes a contact sheet per camera with
each event's best face under both rules, for checking the cut by eye.

`--yunet` also runs the YuNet face detector (five built-in landmarks; model
and provenance under `models/face_detector/yunet/`) on the same upper-body crops
of the same frames, aligns its faces from its own landmarks, and reports and
draws it separately, so the two detectors differ in nothing else.
`--device GPU` runs the person and face detectors on the GPU and `--hw-decode`
decodes with the configured `hardware_acceleration`; the AdaFace norm and YuNet
stay on the CPU. FP16 GPU inference moves person boxes slightly, which can
change which frames the old search would have picked, so compare runs made on
the same device.

`--embeddings FILE.npz` also saves the AdaFace unit embedding of every face
that aligned, keyed by the face `id` in `--json` (which also records each face's
box and each event's epoch), for offline matching against confirmed faces.

`--embeddings FILE.npz` also writes each aligned face's AdaFace embedding; the
ids match the `id` of each face in `--json`, which also carries its box.

### Findings on this host (2026-11-24, 60 events on six cameras)

- Searching every frame instead of the old ~8 roughly doubles the events with
  a usable face; most of the gain is inside the window the old search already
  covered.
- The 300 px detector's false hits (heads, torsos, leaves) make "biggest
  face" the wrong selection rule. YuNet finds more real faces (front-door
  goes from none to five events) but also has false hits; neither is clean.
- Faces of one event embed close together: the median cosine of an event's
  recognisable faces to their mean is 0.79 (current detector) and 0.73
  (YuNet), so an event's faces can be grouped before anyone is named.
- Grouping events by face, with no labels (average linkage on each event's
  mean embedding, checked by eye): the most frequent person forms one pure
  cluster across four cameras (19 events with YuNet, 18 with the current
  detector), and two visitors at the gate each form their own. At a cosine
  threshold of 0.35 no cluster mixed two people; blurry and dim views of the
  frequent person split into small clusters of their own, and false hits that
  passed the norm cut grouped together. At 0.25 that junk cluster also took in
  another person. Grouping works for good faces; it needs a stronger junk
  filter and a way to rejoin split clusters (the visit or body track).
- Existing named people and confirmed faces in the face store came from the
  previous pipeline and are not used as ground truth for the rebuild.

The tool opens
the database and recording index read-only, loads its own CPU copies of the
models with the OpenVINO cache off (`--threads` sets the CPU threads), and does
not touch the service or its config. Zones are not evaluated offline, so anchor
ranking treats a person at the alert confidence as eligible. `--json` also
writes the per-event data, including every face's size, confidence and norm.
