# Face capture: picking the moment

Status: proposal, 2026-11-24. Branch `claude/people-rebuild`, built on PR 370
(timeline frame stepping). Evidence: [people-rebuild.md](people-rebuild.md).

## Problem

Face search today is a side effect of incident refinement. Inside the
refinement decode lease, `RecordedMotionObjectDetector._refine_face_evidence`
(`motion_pipeline/object_detection.py`) tries the plan frames plus up to four
extra frames around the two best person anchors, within 4 s, then hands
`face_candidates` to `decision_handler._persist_face_candidates`, which writes
each candidate's frame as a snapshot and calls the face store's
`ingest_candidates`. That stores observations and queues recognition in the
inference worker.

Measured on 60 recent events (six cameras):

- Searching every frame instead of the old ~8 roughly doubles the events with a
  usable face (shorter side >= 48 px, AdaFace norm >= 13). Most of the gain is
  inside the window the old search already covered; the best moment lands
  anywhere from -15 s to +16 s around the trigger.
- The old search swings on tiny changes: running the same detectors on GPU
  instead of CPU changed which few frames it picked and, for one event, its best
  face from 62 px to nothing.
- face-detection-retail-0004 returns heads, torsos and leaves as faces, so
  "biggest face" is the wrong selection rule. YuNet (five built-in landmarks)
  finds more real faces but also has false hits; an AdaFace norm cut removes
  most of both detectors' junk.
- Searching every main-stream frame costs about 40 s per event (4K decode and
  detection), too much for the live service.

## Goals

1. For each recorded person event, keep the few best faces of each person in
   it, chosen from the whole person track rather than from fixed offsets.
2. Never delay or compete with incident decisions.
3. Cost a small, bounded fraction of the every-frame search.
4. Store what grouping and naming need (embedding, quality, detector, person
   track) on a clean slate; the old face data was reset on 2026-11-24.

Out of scope here: grouping and naming (priority 2), carrying a name through a
visit (priority 4) and the People screen.

## Proposal

### 1. A deferred capture job, outside refinement

Face capture becomes its own job, queued when an event's incident decision is
persisted with a confirmed person, and run later at `InferenceWorkload.ENRICHMENT`
priority under the existing recorded-decode budget. It uses the existing leased
detection-job queue (`event_store/jobs.py`: enqueue, claim, checkpoint, retry)
with a new job kind, so it inherits leases, retries and restart recovery.
Inference and decoding run inside `evidence_work.cancellable_evidence_work`, so
incident work can preempt it.

`_refine_face_evidence` and the face part of `_detect_objects` are removed from
refinement once the job is live; refinement then spends nothing on faces.

Owner: a new package `survng/app/face_capture/` (job handler, moment selection,
face pass). The face store keeps ownership of persistence.

### 2. Pass one: find the moments on the sub-stream

Most cameras also record a `live` sub-stream (896x512 to 896x672 H.264, about
20 fps). The job decodes it over the event window plus a margin (default 3 s
each side, capped at 30 s) at about 5 fps, runs the configured person detector,
links boxes into short tracks by overlap, and scores each sample by the face
size the person would give on the main stream: the upper-body window height
scaled to main-stream pixels. It keeps the top K moments per track (default 4),
at least 0.5 s apart.

Fallbacks, in order: the event's `scene_observations` person boxes (exact
main-stream timestamps, but sparse and missing for some events), then the old
plan offsets.

### 3. Pass two: search the main stream only around those moments

For each moment, decode one frame run of the main stream with PR 370's
GOP-at-a-time decoder (`recording_frame_steps`: frame list from the packet
listing, `frame_run` from the keyframe, at most 30 frames), and search every
frame within +-0.5 s of the moment:

- face detector on the upper-body window of the person box (YuNet proposed;
  see decisions),
- alignment from the detector's own landmarks,
- AdaFace embedding and norm in one call
  (`adaface-ir50-webface4m-norm.xml`, CPU f32),
- keep a face only if its norm and size pass the cut; per track keep the top N
  by norm (default 3).

PR 370's `_decode_frame_run` lives in `recording_media_runtime` and returns JPEG
bytes for the timeline. Its decode core moves to a shared function that returns
`(pts, frame)` arrays; the timeline route keeps its JPEG encoding.

Expected cost per event: about 50 small sub-stream frames plus up to K x 30
main-stream frames, against about 330 main-stream frames today in the
every-frame search. To be measured before building (see verification).

### 4. What is stored

Per kept face: event id, person track key, offset, face box, detector and its
landmarks, norm, embedding with its model fingerprint, and a padded face crop
(not the full frame, which the old pipeline stored and later had to share-check
on deletion). The face tables are empty after the reset, so the schema can
change without migrating data; old columns that no longer have a writer are
dropped in the same change.

### 5. Lifecycle

- Idempotent per (event id, capture version): a rerun replaces the event's
  faces.
- A segment not yet finalised defers the job; a segment removed by retention
  completes it with nothing stored.
- Shutdown releases the lease; the job resumes on the next start.
- Reconfiguration (detector, cut, K, N) takes effect for jobs claimed after it;
  stored faces keep their capture version.
- Failures are counted per stage and rate-limited in the log.

### 6. Configuration

`face_capture_enabled` (separate from recognition), `face_capture_detector`,
`face_capture_min_norm`, `face_capture_min_size`, `face_capture_moments_per_track`,
`face_capture_faces_per_track`, `face_capture_margin_seconds`. Recognition
settings stay as they are until priority 2.

## Decisions needed

1. Face detector: YuNet with the norm cut (recommended), or both detectors.
2. Whether refinement stops searching for faces as soon as the job ships
   (recommended), or both run for a comparison period.
3. Cameras without a sub-stream use the fallbacks; confirm none need more.

## Verification before building

Extend `face_frame_search` with the two-pass selection and run it on the same
60 events. Pass condition: at least 90% of the events where the every-frame
search found a usable face also get one, at under 15% of its decode and
inference time.

## Verification of the build

- Unit: moment selection (tracks, spacing, scaling to main stream), the
  shared frame-run decode, the keep rule, job idempotency.
- Lifecycle: job claim, preemption by incident work, restart recovery,
  retention racing the job.
- Recording suite (`scripts/verify-architecture.sh recording`) for the decode
  change.
- On the host: `./survngctl status` decode and inference load while incidents
  run, and the two-pass result on new events against the offline tool.

## Migration order

1. Lift the frame-run decode core out of `recording_media_runtime`.
2. Load the AdaFace norm export in the recognizer (embedding and norm).
3. YuNet detection in the inference worker, with its landmarks.
4. Capture job, moment selection and storage, behind `face_capture_enabled`.
5. Remove face search from refinement.
