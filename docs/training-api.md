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

## Trust and access

All returned annotations are explicitly marked `model_generated`. They are
pseudo-labels and should be reviewed before being promoted to ground truth,
especially low-confidence, ineligible, or `include_empty` samples.

The endpoint follows SurvNG's existing HTTP security boundary and does not add
separate user authentication. Keep it on a trusted LAN/VPN or behind the same
authenticated reverse proxy as the rest of SurvNG. An external importer must
provide whatever proxy credentials that deployment requires.
