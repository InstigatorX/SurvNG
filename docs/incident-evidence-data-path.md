# Incident evidence data path

An incident is a durable episode of observed activity, potentially spanning
multiple cameras. Scene membership, evidence certainty, and notification policy
are independent. An object outside an alert zone remains part of the scene.

## Acquisition and evidence

Camera and motion notices request detection. Periodic scene discovery also
requests one full-scene frame every ten seconds per detection-enabled camera
(`detector.scene_discovery_enabled`, `detector.scene_discovery_interval_seconds`).
Discovery uses the existing durable refinement worker and inference limits.
Scheduled capture frames are copied into a bounded four-frame buffer before
their jobs are queued. Workers analyze the matching retained frame without
waiting for recording finalization; its actual capture time and generation
remain attached to the observations. Evicted frames and jobs resumed after
restart use recorded evidence. Discovery uses the existing 60-second evidence
job window rather than the 20-second probe window, so an ordinary recorded
refinement ahead of it does not expire its queued sample prematurely.
Acquisition is independent of events. Every discovery result is retained in the
acquisition ledger, including successful empty samples and failed attempts.
Objects do not establish activity merely by appearing in a detector's output.
Candidates that can still establish an incident enter a durable confirmation
queue without requiring an incident. A discovery frame whose boxes are all
outside the establishing zones stays in the ledger and does not schedule that
recorded pass. Failed analysis still requests confirmation.
Unchanged observations enrich an existing episode without extending its
activity clock or reopening it.

Confirmation samples the recorded main stream, keeping each original image's
own time, resolution and geometry. Initially it requests ten seconds before
discovery and five seconds after, including the preceding discovery instant when
available within the bounded window. It waits for recording availability and retries within a
five-minute processing deadline. Overlapping work coalesces into windows of at
most 40 seconds, with at most eight frames per pass. These are processing limits;
they do not define when physical activity ended. Configured playback pre/post-roll
and subsequent recorded analysis remain separate.

The activity evaluator requires localized image change against a sufficiently
stable surrounding image, or an independently admitted camera/motion notice.
Label churn, repeated boxes, confidence and alert zones are not physical activity
evidence. Measurements retain their source sample and observation references and
policy version. Tracking accumulates slow image change against a bounded baseline;
associating observations alone never advances the activity clock. This is sampled
evidence, not a guarantee of detecting every action or eliminating every false positive.

Zones decide which of that physical evidence can establish or prolong an incident.
They do not remove observations from an incident that eligible activity already
established. The establishment policy is separate from alert eligibility:
`notifications_enabled` and confidence thresholds are not part of it, and
`alert_eligible` is not the gate. The snapshot records zone name, enabled state,
behavior, polygon, object classes, and depth band, plus `require_incident_zone`.
Replay uses that snapshot. Later zone edits do not rewrite a decision already
recorded for an episode, and historical import does not apply today's zones.

An enabled Ignore zone wins where it overlaps an incident zone, including by
object class and by a depth band on an Ignore zone. A depth band on an incident
zone does not reject objects outside that band. With no restricting zones, or
when incident zones are not required, the rest of the frame can still establish
activity. When an Ignore zone exists, or incident zones are required, activity
solely in an Ignore zone or outside every incident zone does not create an
incident and does not move the episode's activity clock. A mix of those two
is recorded as `ineligible_zone` rather than described as ignore-only. The configured
inactivity grace still closes the episode. Ignored observations remain in the
acquisition ledger and, once an episode exists, inside that episode's window.

A camera or motion notice is not spatial evidence. Under a restricting snapshot
it establishes activity only when a retained box on the same camera is in an
eligible zone (`verified_camera_notice`). A notice with no such box, and measured
motion that names no observation, cannot satisfy the restriction
(`insufficient_spatial_evidence`). A notice also cannot override a physical
witness that was measured and found ineligible. Cross-camera association uses the
same rule: the camera being admitted needs its own zone-eligible activity.
Decisions recorded before this policy remain as recorded.

Live detection stores the snapshot on the event. Tracking supplies the camera's
current snapshot only when the event does not already have one, and does not
keep that snapshot inside the tracking blob. A later ignore-only measurement
does not replace a supported decision or advance last activity. The stored
decision keeps the physical assessment and the zone interpretation
(`establishment_zones_v1` when a snapshot was applied).

The candidate acquisition floor is independent of alert confidence and defaults
to 0.25. Every usable acquired observation is retained before temporal consensus,
zone interpretation, or cover selection. Recorded temporal samples and tracking
frames carry camera/frame time, dimensions, confidence, geometry, source media,
and a session-scoped association key. Object certainty uses
`detector.tracking.confirmation_confidence_threshold` (default 0.45), independently
of alert confidence. Candidates outside alert zones and
stationary candidates are retained. Uncertain candidates remain uncertain.

Live frames retain receipt-time provenance and are not represented as exact
recorded timestamps. Recorded observations retain their actual frame time when
available. A missing or failed analysis pass cannot erase preceding observations.
Off-cover boxes never inherit the cover's image reference or coordinate plane.

## Durable model

The event database owns:

- `acquired_samples`, `acquired_observations` and their associations: source
  evidence before incident establishment, including failed and empty samples.
  Analyses with different observations of the same capture retain separate
  sample identities; replay of the same analysis remains idempotent. Activity
  verification counts distinct camera/capture timestamps, not analysis variants.
  Projection preserves each observation's original timestamp and geometry rather
  than substituting the enclosing capture's timestamp. Older persisted sample
  IDs remain readable, with later observations retained as additional analyses.
- `scene_activity_measurements` and `scene_activity_decisions`: append-only
  physical evidence and versioned establishment decisions. A zone-aware decision
  records the physical assessment separately from the zone interpretation and
  the observations that supported or failed establishment.
- `scene_candidate_jobs`, `scene_candidate_seeds`, `scene_candidate_admissions`:
  bounded confirmation work, indexed seed ownership and atomic admission per
  generation. Replayed jobs recover the committed event before inference or
  alerts; late coalesced seeds create a new generation of the same work item.
- `scene_incidents`: stable IDs, revisions, activity interval and lifecycle.
- `scene_episodes`: camera intervals, inactivity boundaries and analysis coverage.
- `scene_event_membership`: legacy event associations to authoritative episodes.
- `scene_objects` and `scene_observations`: object associations and retained model
  evidence, independent of current event covers.
- `scene_alert_decisions`: notification significance, separate from membership.
- `scene_corrections` and `scene_aliases`: operator history and preserved links.
- `scene_analysis_jobs`: leased, resumable recorded-analysis windows and cursors.
- `scene_notification_outbox`: durable revision snapshots awaiting delivery.

Object association uses supported track continuity or unambiguous nearby geometry.
Ambiguous associations remain separate possible objects. Operator corrections
can label objects, associate/separate observations, and merge/split incidents.
Corrections require the expected revision and execute atomically. Splitting an
association across episodes gives the resulting incidents independent subjects.
Original model labels remain on observations. Overlapping episode windows
retain their own associations to a shared source observation; merging those
episodes coalesces supported subjects without removing either episode’s evidence. Different sightings without
concurrent visibility or corrected identity are marked as uncertain continuity;
summaries count sightings rather than asserting a number of unique people.

The initial camera inactivity grace is 45 seconds. Repeated stationary discovery
does not reset it. Measured activity extends a recorded-analysis job; processing
budget limits produce resumable chunks rather than new incidents. Capacity or
recording failures retain cursors and report incomplete analysis. Readable
footage after a known discontinuity is analyzed with a new track context. Missing
historical intervals are recorded as gaps; the processing cursor is separate
from the last frame actually analyzed, including after restart. Jobs resume on
the existing refinement worker’s maintenance pass when tracking capacity becomes
available, including when the live camera is offline but recordings remain.

Confirmed identity evidence can connect temporally compatible moving subjects
across cameras, using configured camera transition windows or a 45-second
fallback. Both sides require recorded physical activity associated with the
person, and the camera being admitted must have zone-eligible activity.
Detector-box movement alone cannot connect them. An ignore-only camera does not
become a bridge. Appearance-only matches remain possible relationships. Operator split
boundaries prevent automatic reconnection.

## Presentation and consumers

`GET /api/incidents/{incident_id}` and
`GET /api/incidents/detail?incident_id=...` return the canonical incident.
Legacy `event_ids` links resolve full persisted membership. Multiple supplied
IDs that belong to different incidents return their canonical IDs in a 422
response rather than inventing a new grouping. Filtering and pagination select
incidents without changing their membership. Camera/object/zone/type filters and
pagination run in SQLite before full incident history is hydrated. Facets read
scalar metadata rather than loading every unmatched incident.

The `scene_objects` inventory covers the complete episode history. Top-level
`objects` and current event images remain source-frame presentation fields for
compatible image overlays; they do not define the scene inventory. A scene
object has supporting observations and visibility times. A selected cover is
only a view of the evidence. Activity describes observed position/zone changes
and the last retained sighting; it does not infer disappearance from missing footage.

Playback advances chronologically through camera episodes in 15-minute chunks.
The existing event MP4/HLS routes accept `episode_id`, `start_epoch`, and
`end_epoch`; the server verifies episode membership and bounds and uses a distinct
cache identity per chunk. Empty pre/post-roll remains inside the playback window.

`GET /api/incidents/observations/{observation_id}/snapshot` serves the retained
supporting image with its own source geometry. Missing or expired images return
404 while observation metadata remains inspectable. Retention clears evidence
references and advances the scene revision; cover replacement does not delete
another object's supporting image.

Incident detail includes an establishment explanation, separate from its alert
explanation. Observed objects and activity start collapsed. Larger retained
analyzed images can become the representative image; their own detections supply
the overlays. An additional main-stream image never inherits another frame's boxes.
Source dimensions and capture time remain visible in evidence details.

The Observations workspace uses `GET /api/observations` with a bounded day/window,
camera/status filters and pagination, plus `GET /api/observations/{record_id}`.
Pending, unsupported and incomplete acquisitions stay accessible without appearing
as newly established incidents. Capacity exhaustion or missing video is unresolved
coverage, never a finding that nothing happened. Observation thumbnails use cached
resizing; opening the original preserves its native resolution.

`POST /api/incidents/{incident_id}/corrections` accepts `expected_revision` and
one of `label`, `associate`, `separate`, `merge`, or `split`. Merge also requires
expected revisions for its source incidents. Conflicts return 409. The existing
API security boundary requires admin scope for these mutations.

Search, assistant investigation, recording views and notifications resolve the
same canonical scene. Semantic and appearance indexes can reference retained
observations independently of current covers. Image/model unavailability remains
explicit; metadata membership does not claim a vector exists for every object.

Notification schema 3 carries the complete object roster and separate alert
decisions. It contains representative evidence per object; full frame history is
available from canonical detail. The incident lifecycle publishes database
outbox revisions and acknowledges only after successful delivery. MQTT is a
transport; it no longer constructs incidents. Consumers deduplicate by incident
ID and revision. A crash after delivery but before acknowledgment may replay a
revision. Historical identity enrichment does not emit a new alert.

## Migration, retention and rollout

Startup imports retained event evidence in bounded transactions, including
previously excluded objects and retained legacy track histories. Track-level
confidence is explicitly identified as a summary when original per-frame scores
were not retained. A versioned, silent upgrade also repairs earlier partial imports.
Migration is resumable and emits no historical
notifications. Legacy coverage is marked historical because discarded frames
cannot be reconstructed. Historical semantic/appearance work follows the
existing worker budgets; migration does not bulk-reanalyze video.

Existing discovery-created incidents without retained establishment evidence are
marked unconfirmed and removed from the ordinary incident feed. Their IDs, links,
inventory and correction history remain available. Reclassification records an
auditable correction and creates a completed historical review record in
Observations; it schedules no video work and does not reconstruct object membership.

The schema additions preserve event records and old links. Deploy with a database
backup and validate migrations on a copy before service cutover. The previous
application can still read its event tables for rollback; preserve the new scene
tables so a later cutover does not lose operator corrections.

“All objects” means all retained detector observations in analyzed evidence.
Sampling, unavailable recordings and detector uncertainty remain visible; SurvNG
does not claim to recognize every physical object.
