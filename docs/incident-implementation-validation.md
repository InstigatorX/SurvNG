# Canonical incident implementation validation

## Zone establishment revision (2026-09-28)

Zones now gate incident establishment and prolongation at the shared admission
boundary. They do not filter scene contents, confidence, or notifications.
`alert_eligible` is not the establishment gate. A restricting snapshot is an
enabled Ignore zone, or required incident zones. Ignore wins on overlap, including
class and depth-band rules already used for zone geometry. A depth band on an
incident zone does not reject objects outside that band. Full-frame behavior
remains when the snapshot does not restrict.

A camera or motion notice is not spatial evidence. When the snapshot restricts,
the notice establishes activity only if a retained box on that same camera is in
an eligible zone. Measured motion that names no observation cannot satisfy the
restriction. Cross-camera association requires zone-eligible activity on the
camera being admitted. The stored decision keeps the physical assessment and the
zone interpretation (`establishment_zones_v1`). Historical import does not apply
today's zones, and a rejected live decision is recorded so restart does not
admit it as historical camera activity. Tracking fills a missing snapshot and
does not replace one already stored with the event.

The inspected incident `incident-09302826569e4bec9fa4490c1eaaa95c` remains the
recorded policy-version-1 decision. It is complete, camera `back-right`, revision
851, last activity 2026-09-28T14:20:03.471Z. Its supporting observations are
`observation-289d39e78f44ef4660cf2c550421704a` (car, confidence 0.687, spatial
zone NoMotion; the stored alert reason is `outside_incident_zone` because the
confidence-gated match list was empty) and
`observation-2bcf851e361682a27b15dfac23f6e9ac` (car, confidence 0.7915, spatial
zone NoMotion, stored alert reason `ignored_zone`). Recomputed with the current
back-right geometry, both are establishment-ineligible Ignore (`NoMotion`).
Road's disabled notifications do not contain this activity and are not the reason
it fails the new rule. The record was not rewritten. Under the new rule that
movement would not have established or prolonged the incident; the observations
stay retained.

This correction was not covered by the earlier 2,948-test run. After it, the
backend suite passed **2,963 tests and 271 subtests** in 142.25 seconds. The
mixed-reason follow-up then passed the 16 zone-admission tests. All 69
frontend unit-test files passed, and the production frontend build completed.
The existing HTTPX deprecation and large frontend chunk warnings remain.

That code was deployed as PID 3824904, instance `0ebbd23e865a66684d1daae0`.
At about 59 seconds of uptime all 13 cameras were healthy, the detector had
completed 205 inferences with none failed, the journal had no error entries,
and `/api/health` returned HTTP 200. That window does not show the system is
free of later bugs. A later restart, described below, replaced that process.

`incident-09302826569e4bec9fa4490c1eaaa95c` was still revision 851, complete,
policy version 1, reason `video_verified_activity`. During the preceding process,
live decisions with policy `establishment_zones_v1` were persisted. They included
`eligible_zone_activity`, `ignored_zone`, and `insufficient_spatial_evidence`.
One back-right decision rejected physical movement whose witnesses were in
NoMotion or outside every incident zone; its observations stayed in the
acquisition ledger and it did not gain incident membership. Unlocalized notices
were stored as `insufficient_spatial_evidence` with an establishment row and no
membership. A later pass records a mix of Ignore and outside witnesses as
`ineligible_zone`. The decision already stored as `ignored_zone` was left as
recorded; its per-observation interpretation still lists both reasons.

## CPU access paths (2026-09-28)

The complete-observation model was spending multiple cores on repeated scans.
Semantic checks walked an entire model generation for each observation.
The live incident feed rebuilt camera and label facets from historical rows on
every refresh. Job expiry evaluated a JSON predicate across completed jobs.
Discovery scheduled recorded confirmation for every retained detection, including
frames whose boxes were all outside the establishing zones.

The lookup now uses `idx_semantic_observation`. An event's indexed observation
keys are read once. A current projection compares observation ids and snapshot
paths, and loads payloads only for observations that are not in the index yet.
Observations without a box, or with `snapshot_visible` false, are not treated as
missing embeddings. Unbounded live-feed facets read `scene_facet_cameras`,
`scene_facet_labels`, and `scene_facet_zones`. Those tables are filled once and
then maintained as episodes, labels, and zones are stored. Unconfirming an
incident drops a camera or label that no other confirmed incident still uses.
Job expiry restricts `camera_id` and `state in ('queued','running')` before the
JSON predicate. A discovery frame whose boxes are all outside the establishing
zones stays in the ledger and does not schedule recorded confirmation. Failed
analysis still requests confirmation. Tracking of an already admitted event is
unchanged, so an established incident still receives its retained observations.

These changes were not in the 2,963-test run. After them, the backend suite
passed **2,970 tests and 271 subtests** in 131.87 seconds. The HTTPX
deprecation warning remains. No frontend source changed.

The service was restarted onto this code. The running process is PID 4191316,
instance `8efa8625e8ebc6d9593599b1`, started 2026-09-28 15:00:42 EDT. At about
174 seconds of uptime all 13 cameras were healthy, the detector had completed
482 inferences with none failed, scene tracking was using one of three baseline
slots, and the journal had no error entries since start. Read-only timing of
the live facet tables was 0.02 ms for 6 labels, 0.01 ms for 13 cameras, and
0.02 ms for 23 zones. Before this change the same live feed spent about 1.2
seconds on labels and 1.6 seconds on cameras.

A 10-second sample at roughly five minutes of uptime measured the main process
at 239% CPU and the service cgroup at 5.40 cores of 16. An earlier sample in
the same window was 249% and 5.46 cores. The investigation's samples, before
these access-path fixes, were about 501–581% for the main process and 8.25–8.52
cgroup cores. This is not a controlled before/after: the workload and the
restart differ. A 15-second stack sample of the new process no longer showed
the incident-feed facet query. The remaining main-process time was per-camera
motion analysis, recording indexing, one camera's object tracking, and semantic
crop reads for observations that are not indexed yet. Those crop reads are the
backfill catching up, not the generation-wide lookup. About three of the 5.4
cgroup cores are in child processes, including capture. This window does not
show that steady-state CPU has reached a final floor.

## Activity-establishment revision (2026-09-28)

The three existing reviewers were reused for detection/physical activity,
storage/migration, and human review. Their agreed boundary is acquisition →
activity establishment → complete scene membership → separate alert decisions.
The primary agent completed integration and subsequent review after the delegated
reviewers reached their usage limits; their design agreement is not a claim that
they reviewed every final integration change.

The revision retains periodic detections without requiring an event or incident.
It verifies physical activity from recorded source frames, preserves original
low-resolution acquisitions, and presents main-stream images with their own
timestamps and geometry. Detector label/box changes alone cannot establish or
extend an incident, populate movement activity, or connect cameras. Slow physical
change accumulates against a bounded image baseline. Confirmation includes the
preceding discovery instant so an arrival followed by standing still can be
verified. Camera/motion notices remain explicitly identified evidence sources.

Review fixes cover admission/confirmation crash recovery, coalesced job
generations, stale leases, immutable raw evidence with later physical measurements,
retention, indexed candidate ownership, historical reclassification without
re-projecting objects, and startup query bounds. Restart after admission recovers
the committed event before inference or alert processing.

A full database copy contained 81,148 events and 373,820 scene observations.
Migration preserved the scene-observation count, imported 371,513 unique source
observations, and reclassified 112 historical discovery-only records without
discarding links or evidence. Its notification outbox remained empty. A consistent
copy passed SQLite `quick_check` (`ok`) and `foreign_key_check` (no violations).
Integrity validation used a temporary memory-backed file after the backup-storage
scan proved slow; that temporary file was removed afterward.

In the migration copy, `incident-09302826569e4bec9fa4490c1eaaa95c` resolved to
its preserved unconfirmed record and was excluded from the ordinary incident feed.
Subsequent live tracking of its retained recording supplied physical evidence of
a moving car at approximately 14:20:03 UTC. The live record therefore became
established from that later evidence, not from its original uncertain person
detection. The source recording was visually inspected and contains the car.
The historical window and earlier uncertain observations remain retained; this
does not assert continuous activity throughout that historical window.
Event 80913 retains
both original person observations, at 0.9414 and 0.751. No historical video
reanalysis or alerts were scheduled by migration.

Validation initially passed **2,941 tests and 271 subtests**. A later full run
during migration had two face-review background-refresh timeouts (2,940 other
tests passed); the complete face-review file passed on rerun, **7 tests**.
After the first live persistence fix, the full suite passed **2,945 tests and
271 subtests**. The timestamp and failed-retry fixes passed **108 focused tests**,
followed by the final full run: **2,948 tests and 271 subtests passed** in 136.63 seconds.
The frontend passed all **69 unit test files**, the production build, and the
Observations and canonical-incident browser suites (including errors, reconnect,
mobile layout, corrections, and playback transitions). The existing HTTPX
deprecation and large frontend chunk warnings remain.

Live validation found two source-identity failures that the earlier tests missed:
reanalysis could reference new detector geometry absent from the immutable
acquisition, and context projection could replace an observation's timestamp with
a capture timestamp differing by less than a microsecond. Analysis variants now
retain later observations without replacing originals; projection preserves the
original timestamp. Review also found that a failed retry could inherit earlier
successful observations; failed attempts now remain separate from both those
observations and successful empty frames. Regression tests reproduce these failures, verify replay
idempotence, preserve one supported track, and prevent repeated analysis from
counting as independent physical instants. A 72-row isolated copy of live event
81585 reproduced the timestamp foreign-key failure and passed after the fix.
The runtime database was opened read-only to obtain that replay evidence.

A fresh consistent pre-deployment backup is retained at
`/mnt/media1/SurvNG/.deployment-backups/survng-before-activity-ledger-20260928T155253Z.sqlite3`.
The final deployment runs PID 3561721, instance `c8311dd40ce0c24d0bb2b678`.
At 138 seconds of uptime, the owner-only snapshot showed all 13 cameras healthy,
recording matching each camera's configuration, 551 inferences with no inference
failures, and no logged errors in the untruncated runtime log window. The preceding
source-identity deployment also ran without persistence errors for more than four
minutes, and event 81585 resumed tracking successfully. The final deployment
retained native review images at 2560 × 1440, 4096 × 1800, and 4512 × 2512.
Original acquisition images remain unchanged.
The health endpoint and both frontend routes returned HTTP 200, and served
scripts matched the production build. Missing source frames and expired work
remain explicit incomplete coverage, rather than a negative activity finding.
The older sections record earlier implementation and deployments.

The implementation replaces event-gap reconstruction with the scene tables and
APIs described in [the evidence data path](incident-evidence-data-path.md).
Three delegated implementation/review scopes covered detection and tracking,
frontend review/playback, and consumer/migration architecture, using the inherited
session model. Primary integration covered canonical storage, corrections,
queries, media ranges, notification policy and retention.

## Evidence replay

A read-only selection of 1,000 retained events ending at 80913 was copied into a
temporary database, then migrated using the new EventStore. The import completed
in approximately 9.0 seconds after the review fixes and retained 18,325
observations, including objects from legacy tracking histories. No notifications were queued, including after
settling historical incidents.

Both people from 80913 were retained, at confidences 0.9414 and 0.751, with
supported observation certainty. Broader history also contains earlier sightings
in the same episode. When identity continuity between those sightings cannot be
established, the inventory and summary explicitly describe uncertain continuity;
they do not assert that each sighting is a different person. Historical coverage
remains incomplete rather than implying that discarded observations were recovered.

## Runtime state

At the user's request, the frontend was rebuilt into `survng/static` and
`survng.service` was restarted on 2026-09-28. The health endpoint returned HTTP
200 and the served static index matched the rebuilt file. The owner-only runtime
snapshot reported all 13 cameras healthy and the OpenVINO GPU detector ready,
with successful inference and no failed inferences during the deployment check.

One delegated playback test was initially invoked outside pytest isolation.
Accessing the lazily initialized application manager started the additive
historical migration against the configured runtime database before interruption.
Read-only inspection found 57,200 event memberships and 29,706 scene incidents,
with zero notification-outbox rows. The existing event tables were preserved.
The import is resumable on the next startup. No compensating deletion was made.
The owner-only observability socket remained responsive afterward.

A consistent SQLite read-snapshot backup passed `quick_check` before restart:
`/mnt/media1/SurvNG/.deployment-backups/survng-before-scene-cutover-snapshot-20260928T123659Z.sqlite3`.
The remaining import completed during startup: all 80,688 retained events had
canonical membership, the legacy tracking upgrade completed through event ID
81059, and the migration queued zero notifications. A read-only check confirmed
both 0.9414 and 0.751 person observations for event 80913 in the deployed database.
Startup took approximately eight minutes while migration ran. The old process
cancelled one task after its graceful-shutdown timeout.

Deployment follow-up exposed discovery retries caused by waiting for finalized
recordings, which trailed capture by roughly 25–60 seconds on sampled cameras.
The follow-up fix retains four scheduled capture frames per camera before
enqueueing discovery, analyzes the matching image through shared refinement
inference, and keeps recorded fallback for evicted or replayed jobs. Capture
provenance, observation thresholds, and alert confirmations remain separate.
A second issue expired queued discovery after 20 seconds even when an ordinary
25-second refinement was ahead of it; discovery now uses the existing 60-second
evidence-job window. Both changes were deployed and the service restarted.

Follow-up validation: **403 tests and 10 subtests passed**, covering acquisition,
recorded detection, decision handling, job recovery, camera capture, canonical
incidents, and event storage. This includes stale/invalid frame rejection,
capture copying, buffer eviction, inference failure, uncertainty and alert-zone
handling, and expiry behavior in both job stores. Live incident playback still
requires separate validation.

After the final restart, approximately 2.5 minutes of live verification recorded
**149 completed discovery jobs, all on their first attempt**, with no failed
discovery jobs or reported inference errors. All 13 cameras were healthy. At
the final check, all 11 detection-enabled cameras had discovery samples between
1.0 and 11.3 seconds old. Ordinary motion refinement temporarily delayed some
samples during the check; those queues caught up without the former early expiry.
Rollback can use the previous application with the preserved event tables;
retain scene tables to preserve corrections.

## Validation scope

Final backend run after the review/fix cycles: **2,892 passed, 271 subtests
passed**. The sole warning is the existing FastAPI/Starlette HTTPX test-client
deprecation. Named unittest execution also passed all **6 playback API tests**
under the shared temporary runtime configuration. The earlier frontend
validation passed all **69 unit test files**, the production build, and focused
browser tests; these review fixes did not change frontend code.
Whitespace/diff validation passed. The final review found no further actionable
issues in the reviewed paths; live codec and inference validation remains as
described below.

Backend regression tests cover scene membership, migration, replay and restart,
track association, corrections, observation-specific search, outbox delivery,
acquisition, recording analysis jobs, retention, and episode playback bounds.
Frontend validation includes the production build, unit suites, desktop/mobile
scene review and notification detail, activity, policy explanations, and
multi-camera playback including two-hour episodes.

Browser playback tests simulate media completion/error events and check source
URLs and transitions. Actual codec decoding and live inference load after
cutover remain operational validation; passing these tests does not claim those
checks have occurred.


## Review and fix cycles

Review used isolated reproductions and regression tests, followed by another
review of the changed paths. Confirmed defects fixed:

- Shared source frames disappearing from the second overlapping episode.
- Legacy track-only objects omitted from historical import, including rows
  already processed by an earlier partial import; known track IDs now also
  associate cover and tracking evidence with the same subject across restart.
- Duplicate acquisition batches preventing supporting-image attachment.
- Rare filters hydrating unmatched historical incident bodies before pagination.
- Recording discontinuities preventing analysis of later readable footage;
  processing cursors and actual analyzed times now remain distinct after restart.
- Decoder budget exhaustion advancing past unread video, and prior analysis
  progress skipping newly requested pre-roll. Known recording boundaries remain
  visible even when processing reaches its deadline.
- Slow cumulative movement missed by cross-camera continuity checks, and absent
  geometry incorrectly treated as measured movement.
- Named unittest imports bypassing pytest’s temporary runtime configuration.

The test package now establishes isolation before named unittest modules load;
pytest uses the same helper. Supported commands include `python -m pytest` and
`python -m unittest tests.test_scene_playback_api` (or discovery with the repository
as the top-level directory). A subprocess regression verifies that an inherited
runtime configuration is replaced before application import.

These review/fix cycles did not initialize the application against the runtime
database or restart the service. The migration replay copied source rows through
a read-only SQLite connection into a temporary database.
