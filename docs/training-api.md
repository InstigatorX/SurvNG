# Training samples API

SurvNG exposes original representative incident images and their matching
model-generated object boxes through a read-only, cursor-paginated manifest:

```http
GET /api/training/samples
```

The endpoint does not burn boxes or labels into the image. Each sample contains
an original snapshot URL plus pixel and normalized coordinates suitable for an
annotation or training-data importer.

## Example

```bash
curl --get 'http://survng.local:8088/api/training/samples' \
  --data-urlencode 'start_at=2026-08-01T00:00:00-04:00' \
  --data-urlencode 'end_at=2026-08-08T00:00:00-04:00' \
  --data-urlencode 'camera_ids=gate,front-door' \
  --data-urlencode 'object_labels=person,car' \
  --data-urlencode 'eligibility=eligible' \
  --data-urlencode 'minimum_confidence=0.50' \
  --data-urlencode 'limit=100'
```

When `base_path` is configured, returned image URLs include it. Unprefixed API
requests remain available to local clients.

## Query parameters

- `start_at` and `end_at` are required ISO 8601 timestamps with timezone
  offsets. The range is half-open (`start_at <= captured event < end_at`) and
  may span at most 366 days.
- `camera_ids` is an optional comma-separated camera ID filter.
- `object_labels` is an optional case-insensitive comma-separated class filter.
- `eligibility` is `eligible` (default), `ineligible`, or `all`. Incident
  eligibility is a SurvNG policy outcome, not proof that a box is correct.
- `minimum_confidence` ranges from 0 to 1.
- `include_empty=true` also returns snapshots with no annotations matching the
  selected filters. Treat these as review candidates rather than guaranteed
  negatives because the detector may have missed an object.
- `sample_kinds=negative_candidate` selects unreviewed negative candidates.
- `sources=motion_audit` uses clean motion-audit snapshots as that source.
- `image_source=stored` (default) serves the saved snapshot.
  `image_source=main_recording` selects native-resolution JPEGs extracted on
  demand from retained main recordings, for motion-audit negative candidates only.
- `limit` ranges from 1 to 500.
- `cursor` accepts the opaque `next_cursor` returned by the previous response.
  Keep the time range and filters unchanged while paging.

## Annotation coordinates

Every annotation includes:

- `bbox_xyxy`: pixel `[left, top, right, bottom]`;
- `bbox_xywh`: pixel `[left, top, width, height]`, compatible with COCO;
- `bbox_normalized_xyxy`: normalized `[left, top, right, bottom]`;
- `bbox_normalized_cxcywh`: normalized YOLO-style
  `[center_x, center_y, width, height]`;
- label, confidence, zones, temporal-consensus state, semantic tier, and
  incident eligibility.

`image.width` and `image.height` define the annotation coordinate plane. SurvNG
omits malformed boxes and objects whose stored coordinate plane does not match
the representative snapshot's other annotations.

`event_at` is the original trigger time. `captured_at` includes the temporal
sample offset for the representative image. `sample_id` remains stable for an
event, while `revision` changes if delayed refinement replaces its image or
object evidence. Importers should use both fields when synchronizing updates.
An event may begin with a provisional live/substream image and later promote a
compatible main-recording or tracked cover. The image dimensions, annotations,
`captured_at`, and `revision` move together so a consumer never needs to map an
old coordinate plane onto the new image. See [Incident evidence data
path](incident-evidence-data-path.md) for the source and promotion rules.

## Unreviewed negative candidates

Motion-audit snapshots where SurvNG did not confirm an object can be imported
as clean negative candidates:

```bash
curl --get 'http://survng.local:8088/api/training/samples' \
  --data-urlencode 'start_at=2026-08-01T00:00:00-04:00' \
  --data-urlencode 'end_at=2026-08-08T00:00:00-04:00' \
  --data-urlencode 'sample_kinds=negative_candidate' \
  --data-urlencode 'sources=motion_audit' \
  --data-urlencode 'limit=100'
```

These samples have `source=motion_audit`, `sample_kind=negative_candidate`,
`assumed_negative=true`, `annotation_state=unreviewed`, and an empty
`annotations` array. The training application must validate them before
treating them as ground truth. Audits where SurvNG already confirmed an object
are excluded. `sample_id` is the stable `motion_audit-{id}` identifier and
`revision` changes if its stored evidence changes.

### Native-resolution negative images

Add `--data-urlencode 'image_source=main_recording'` to the negative-candidate
query above. Fetch each returned `image.url` using the same authentication as
the manifest. Follow `next_cursor` with unchanged filters for further pages.

Listing candidates does not decode video or check recording availability. This
mode also includes audits without a saved snapshot. Downloading an image seeks
to the audit timestamp in an indexed main recording and preserves its native
dimensions, without upscaling or the preview width limit. Width and height in
the manifest remain null until the consumer reads the image. The revision
differs from the saved-snapshot variant; sample IDs remain the same.

Extraction runs in an API worker thread using the shared recording-preview
limiter (one extraction at a time), an eight-second FFmpeg timeout, and the
existing bounded disk cache. Repeat downloads reuse cached frames while the
source recording remains available and unchanged. The HTTP download waits for
its image; this is not a queued export job. Clients should download sequentially
and retry HTTP 429 according to `Retry-After`. Missing recording coverage/files
return 404; extraction timeouts return 504 and extraction failures return 500.
There is no silent fallback to a low-resolution snapshot.

Image responses include `X-SurvNG-Requested-Timestamp`,
`X-SurvNG-Timestamp-Source`, and, when source timing is available,
`X-SurvNG-Actual-Timestamp` (Unix seconds). Manifest `captured_at` is the requested
audit time; the decoded frame may differ slightly. Review the extracted image
itself before accepting it as a negative, including any extra area visible in
the main stream. Annotated event exports cannot use this option because their
boxes belong to the saved image's coordinate plane.

## Trust and access

All returned annotations are explicitly marked `model_generated`. They are
pseudo-labels and should be reviewed before being promoted to ground truth,
especially low-confidence, ineligible, or `include_empty` samples.

The endpoint follows SurvNG's existing HTTP security boundary and does not add
separate user authentication. Keep it on a trusted LAN/VPN or behind the same
authenticated reverse proxy as the rest of SurvNG. An external importer must
provide whatever proxy credentials that deployment requires.

## Human-reviewed samples

Administrators can record reviews using the [detection review APIs](detection-reviews.md).
These reviews do not change incident admission or enable appearance matching.

`review=any` is the default. With no applicable active reviews, the existing
manifest output is unchanged. With reviews, stored-image exports first page
through review-owned images, then continue through the original source manifest.
Treat cursors as opaque and keep all filters unchanged. A page can be empty with
a nonempty cursor when filtering or scanning reaches a boundary; continue paging.
Reviewed images retain their own bytes, dimensions and source identifiers after
source events, audit rows or recordings expire. Event/source IDs in these samples
are provenance, not a guarantee that the original route still exists.

`review=reviewed` returns only direct human annotations and verified negatives.
Use `sources=motion_audit&review=reviewed` for reviewed audit images, optionally
`sample_kinds=annotated` for drawn positives or `negative_candidate` for verified
negatives. Without an explicit kind, both kinds of reviewed audit images can be
returned. Human annotations require `image_source=stored`; they are never remapped
onto frames extracted with `image_source=main_recording`. The latter continues to
export the original unreviewed audit candidates.

Reviewed annotation fields:

- `annotation_state: human_reviewed` applies only to directly reviewed regions.
- `review_verdict`: `not_object`, `wrong_label`, `confirmed`, or `missed`.
- `original_label`: raw detector label, or null for a drawn box.
- `correction_id`: review ID, with audit details at
  `GET /api/detection-corrections/{id}`.
- `annotation_source`: `operator_reviewed` or `operator_drawn`.
- `confidence: null`: human judgment has no invented detector confidence score.

For directly reviewed detections, eligibility, zones and temporal metadata retain
the model evidence values; drawn annotations have no admission result and use
false/empty values. The review itself does not change admission.

Human annotations bypass the confidence and eligibility filters so a reviewed
hidden object or hard-negative region remains retrievable. `object_labels` matches
either its effective label or original label. `not_object` regions remain in the
response; importers must handle them as negative regions instead of positive
bounding-box labels. A corrected label is normalized by trimming/collapsing
whitespace and lowercasing; the original detector label remains unchanged.

With `review=any`, other retained model annotations on the same reviewed image
remain `model_generated` and still obey the model confidence/eligibility/label
filters. Only direct reviews gain review fields. A sample-level
`annotation_state=human_reviewed` means it contains reviewed evidence, **not** that
all its annotations are human-reviewed or that every object was exhaustively
annotated. Runtime-propagated labels are not produced in this phase.

A drawn positive makes a reviewed motion image an `annotated` sample. A verified
whole-image negative has `sample_kind=negative_candidate`,
`annotation_state=human_reviewed`, `assumed_negative=false`, and
`verified_negative=true`. Positive annotations and a verified negative on the same
source image cannot be active simultaneously. Revoke/unreview removes the human
verdict from future exports; it does not erase review history or rewrite source
model evidence. A surviving source image can consequently return to model-generated
or unreviewed export after revocation.

Reviewed ranges use the retained image's capture time where known (otherwise its
source event/audit timestamp). Image hashes and active review revisions determine
sample identity/revision. Multiple reviews of the same source image are grouped,
including when source metadata changed between reviews. A genuinely replaced image
is a separate sample with its original coordinate plane preserved.
