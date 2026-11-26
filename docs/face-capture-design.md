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

## Decisions (agreed 2026-11-24)

1. Face detector: YuNet, with the AdaFace norm cut.
2. Refinement stops searching for faces as soon as the capture job ships.
3. Cameras without a sub-stream use the fallbacks above.

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

## As built (2026-11-25)

Package `survng/app/face_capture/`: `moments` (track linking and moment
choice), `pipeline` (both passes; decoding and inference are passed in),
`engine` (YuNet, landmark alignment, AdaFace embedding and norm; loaded in the
isolated face worker), `events` (person events, timing, stored boxes), `store`
(`face-captures.sqlite3` and `face-captures/` beside the event store),
`service` (background loop) and `__main__` (offline run with contact sheets).
The frame-run decode is shared through `recording_frame_runs`.

Decoding: the proposal reused frame stepping's GOP runs, but runs suit a
stepping player, not a consumer that wants every frame of a window. Each run
restarts the decoder at its GOP's keyframe and holds the whole run in memory
(30 frames of 4K is 750 MB), so capture streams instead:
`recording_frame_runs.stream_frames` decodes one window per segment from its
keyframe through the window end, one frame held at a time (the same shared
decoder pieces; the timeline keeps its runs). The sub-stream uses it too,
thinned to 5 frames a second by time. Each event looks its recordings up once
per source (`rows_for_window`); an unbounded `start_epoch + duration > ?`
lookup in the recording index took 7 s per call and was most of the early
cost.

Deviations from the proposal, on purpose:

- The job is a dedicated loop with its own state table
  (`face_capture_events`: done or failed with an attempt count), not a new kind
  in the detection-job queue. It is restart-safe because the state is stored,
  and it keeps optional work out of the queue incident jobs depend on.
- Faces go to their own database instead of the old face tables, which keep
  their readers (People page, recognition) until priority 2 replaces them.
- The service does not take the recorded-decode budget. It runs one event at
  a time with one decoder process, at enrichment priority for inference; how
  much it competes with refinement on a busy evening is still to be measured.

Settings (all under `detector`, capture off by default): `face_capture_enabled`,
`face_capture_detector_model_path`, `face_capture_embedding_model_path`,
`face_capture_detection_threshold` (these four restart the face worker), and
`face_capture_min_norm`, `face_capture_min_size`,
`face_capture_moments_per_track`, `face_capture_faces_per_track`,
`face_capture_margin_seconds`, `face_capture_moment_radius_seconds`,
`face_capture_backfill_hours` (applied live).

### Trying it

Offline, without touching the service (reads the database and recordings,
writes only where told):

```bash
nice -n 19 .venv/bin/python -m survng.app.face_capture runtime/database \
    --cameras gate,foyer,downstairs --events-per-camera 5 --device GPU \
    --sheets /tmp/face-capture-sheets
```

In the service: set `detector.face_capture_enabled` to `true` and apply the
configuration. The face worker restarts with the capture models; the service
then works through the last `face_capture_backfill_hours` of recorded person
events, newest first, one at a time. Progress is in
`face-captures.sqlite3` (`face_capture_events`, `face_captures`) and the crops
in `face-captures/` under the database directory.

### Measured on this host (2026-11-25)

Offline run of the pipeline on the same 60 events as the every-frame report
(six cameras, YuNet, GPU person detection and decode, 2.5 GB memory cap):

| | Two-pass capture | Every frame |
|---|---|---|
| Events with a usable face (recognisable, >= 48 px) | 40 of 45 kept (89%) | 45 |
| Main-stream frames decoded | 5,599 (24%) | 22,954 |
| Time | 591 s (median 10.9 s an event, longest 20 s) | 2,207 s* |
| Peak memory (offline, models included) | 786 MB | |

\* The every-frame run paid about 7 s an event for the slow recording lookup,
so a fair time ratio is nearer a third. The pass condition set before
building (90% at under 15%) is not fully met: all misses are on the two
indoor cameras (downstairs 7 of 10, foyer 6 of 8), where faces show for one or
two frames, and person detection on every main-stream frame is now most of
the cost. Ranking moments by the face seen on the sub-stream instead of by
person height lifted the kept share from 87% and fixed several indoor misses.

## Binding faces to tracked people (2026-11-25)

A captured face is tied to the incident's scene object (tracked person) whose
box holds the face's centre on the same camera at the face's moment
(`face_capture/binding.py`). Scene observations are sparse, a median of 1.7 s
from a face, so the person's box is interpolated between two of its
observations at most 4 s apart, or taken from its nearest observation within
2 s; the same event is preferred. When the face cannot be placed (no
frame size, or no observation near its moment) and only one person was tracked
on that camera in the incident, the face is theirs; a face placed outside every
box stays unbound. Faces
store the size of their main-stream frame (faces stored before that use the
camera's newest main recording), because scene boxes are in the detector's
frame.

Binding is worked out when an incident is shown
(`GET /api/face-captures/by-incident/{id}`), not stored: observations keep
arriving while an incident is open. Each person card on the incident page shows
the name of the faces bound to it, across every camera that scene object
spans.

Measured on this host's captured faces (13 incidents, 11 of them with more
than one tracked person): 84% of 205 faces bound with these rules, against
59% of 173 for a one-second nearest-observation rule.

## Tuning after the first seed (2026-11-25)

Measured on 917 faces from the 1-day seed, with the groups named and ignored
on the People page as labels (two people, 46 junk tracks):

- **Duplicates.** 408 faces (44%) repeated a face another event of the same
  visit had stored: overlapping events of one scene episode cover the same
  frames (169 of 176 such event pairs shared an episode). Capture now skips
  moments another event of the episode already captured.
- **Grouping threshold.** Kept at 0.40. Lowering it joins a person's groups
  (0.30: 8 and 5 groups for the two people instead of 14 and 17) only by
  mixing more tracks into the wrong group (25 instead of 15). Faces of one
  person are only weakly alike here (median track-to-track cosine 0.24).
- **Automatic naming.** A track is compared with each named person's faces as
  a whole. At 0.45 with a 0.10 lead over the next person, 157 of 241 person
  tracks were named, none wrongly, and 1 of 46 junk tracks; at 0.40, 189
  named, 1 wrongly, 2 junk. 0.45 is used.
- **Quality cut-off.** Not changed: junk and real faces overlap in norm (a
  minimum of 15 would keep 69% of person tracks and still 36% of junk).

Names reach incidents, search, the person filter and scene identities
(which join one person's episodes across cameras) through
`AppManager.face_identity_observations`. A live event's names refresh its
incident and are announced on MQTT `events/identity`; a seeded one's are only
recorded.
