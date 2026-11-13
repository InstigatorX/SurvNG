# Detection reviews (record-only phase 1)

Use **Review detections for training** in incident details or a motion-audit image.
Select an existing box, reveal hidden candidates, or drag a box around a missed
object. Existing candidates are reviewed on their own evidence image; use the
observation selector for retained off-cover images. Unknown coordinate planes are
not offered as selectable candidates. Drawing remains available on a known image.

Choose Correct, Not a label, It is actually…, or confirm a hidden candidate as a
miss. A motion-audit image also supports **Nothing of interest in this image**,
which explicitly verifies the whole image as a training negative. Inspect the image
before selecting it. This is not an automated negative label.

**Appearance matching stays off.** Saving reviews does not change detector weights,
labels, incident/alert eligibility, tracking, scene admission or replay. Existing
scene subject-label edits are separate actions with their existing behavior. A
review does not silently perform that action, and Undo review does not undo a scene
label edit. Smart Search need not be enabled and no embedding requests run.

Admin → Detection → Object includes the review list and motion-image review queue.
The queue prioritizes likely-miss hints from the latest completed camera AI review;
those hints never save a human verdict.
Refresh after reviewing. A missing source image produces a visible error; annotations
cannot be guessed onto a replacement image. Administrators can revoke saved reviews;
viewers can read reviews and evidence but cannot create, supersede or revoke them.
API tokens use the same read/admin scopes as other APIs. When authentication is
explicitly disabled by the existing application configuration, writes are recorded
as `local:authentication-disabled`.

## API

Read endpoints require read scope; mutations require admin scope. All operations
hold a manager-generation lease and use the current store.

| Endpoint | Purpose |
|---|---|
| `GET /api/events/{id}/review-evidence` | Image revision, dimensions, selectable candidates, source identities and supporting observation IDs |
| `GET /api/motion-audits/{id}/review-evidence` | Exact audit image revision and dimensions |
| `GET /api/detection-corrections/source-image` | Revision-checked source image (`source`, `source_id`, `revision`, optional `observation_id`) |
| `POST /api/events/{id}/corrections` | Review a candidate or draw a missed object |
| `POST /api/motion-audits/{id}/corrections` | Draw a missed object on an audit image |
| `PUT /api/motion-audits/{id}/review` | Record `verified_negative` |
| `GET /api/detection-corrections` | Cursor-paginated review list |
| `GET /api/detection-corrections/{id}` | Review and immutable audit history |
| `DELETE /api/detection-corrections/{id}?revision=N` | Soft revoke/unreview using the current review revision |
| `GET /api/detection-corrections/assets/{id}` | Retained full image; `review_id` optionally selects a crop thumbnail |
| `GET /api/detection-corrections/status` | Counts, storage accounting/limits and mode `off`; embedding is deferred |
| `GET /api/detection-corrections/review-queue` | Pending motion-audit images, priority/ID cursor and optional camera filter |

List filters are `camera_id`, `status`, `verdict`, `source`, and `source_id`;
`limit` is 1–100 and `cursor` is the returned next review ID. Status is active,
revoked or superseded. The review queue uses its own opaque cursor and bounded
scanning; continue through empty pages with a cursor. Queue snapshots may expire
between listing and opening; they are copied only when a review is saved.

First fetch review evidence. For a supporting image pass its `observation_id` to
both evidence and create requests. Use the returned candidate `key` as `object_key`.
Do not infer an object index from displayed labels, which may omit hidden boxes.

Example detection review (replace the placeholders with returned values):

```json
{
  "evidence_revision": "<64-character revision from review-evidence>",
  "idempotency_key": "<unique client request UUID>",
  "object_key": "object:0",
  "verdict": "wrong_label",
  "corrected_label": "raccoon",
  "scope": "camera",
  "location_rule": true,
  "note": "Reviewed against the original image"
}
```

A drawn box uses `source=drawn`, `verdict=missed`, `corrected_label`, and `box` as
normalized `[left, top, right, bottom]` on that exact image. The returned width and
height define the pixel coordinate plane. `annotation_id` optionally identifies an
existing drawn annotation to supersede; otherwise the idempotency key identifies it.
No user-supplied file paths or image URLs are accepted.

For `verified_negative`, use the revision and idempotency key with that verdict.
The whole-image negative has no bounding box or runtime effect.

Reusing an idempotency key with the same actor and exact payload returns the saved
review even after source retention. Reusing it for a different request returns
409. A source revision/hash mismatch also returns 409. Refresh the evidence rather
than automatically retrying against a new revision.

Conflicting verdicts require explicit `supersedes: {"review_id": current_revision}`
for **all** conflicting active reviews. The replacement and their audit transitions
commit atomically. Metadata revisions on the same image do not bypass this check.
Positive boxes conflict with a whole-image verified negative and vice versa.
Undo/revoke changes active export state and appends audit history; it never rewrites
the source evidence. Revoking a replacement does not reactivate superseded reviews.

Future scope/location choices are retained for calibration and later design; they
have no matching effect now. Global scope with a location restriction still means
the recorded camera/location restriction would need to be respected. New labels
are allowed in reviews but are not inserted into detector class arrays or runtime
zone vocabularies.

## Storage, quotas and recovery

The main database holds `detection_review_assets`, `detection_reviews`, and
`detection_review_audit`. A verified-negative review is a snapshot-level row in the
same audit ledger, distinguished by its verdict and lack of a region. There are no
cascading links to source events/audits, so ordinary source retention remains free
to run. Original observation IDs, image hashes, source revisions and original labels
remain attached to the review. The original model evidence is not modified.

Owned original image bytes are copied into `<storage_dir>/corrections/`, outside
snapshot and motion-sample retention/maintenance roots. Crop thumbnails are derived
from those bytes when requested. Files are private, created atomically with fsync;
a filesystem lock serializes publication/orphan cleanup across store instances.
The database transaction rechecks source state before publishing the review. Failed
transactions can leave an unreferenced image, never a visible review that intentionally
points at an unpublished image. Unreferenced owned files older than 24 hours are
removed on a subsequent review creation. Referred assets, including revoked-review
assets, are never automatically reclaimed.

Initial limits are **64 MiB per source image**, **2 GiB of owned review files**, and
**10,000 review rows**, including revoked/superseded history. New writes fail visibly
at the cap; they do not evict reviewed evidence. Identical image bytes share a file.
These are code constants for this first release, not new live runtime settings.
The status endpoint reports unique referenced image bytes; the write quota also
includes orphan files on disk. A quota error
requires administrative storage-policy work rather than deleting files behind the
ledger. There is no hard-delete/purge API in this phase.

Back up the review directory together with the main database. Normal snapshot
retention is not a backup of these assets, and no automatic correction backup is
introduced. Keep storage mounted and available during review writes. Writes and retained-image
reads check the existing volume-identity gate and report unavailable storage
instead of creating data beneath an unavailable mount.

## Training and validation

See [Training samples API](training-api.md#human-reviewed-samples) for reviewed-only
export, hard-negative regions, mixed human/model annotations, and retained-image
pagination. Runtime propagation is deferred, so only directly reviewed examples
receive `human_reviewed` in this implementation.

Focused backend tests live in `tests/test_detection_corrections.py`; browser
interaction coverage lives in `frontend/tests/detection-reviews-browser.mjs`.
Phase 2 still requires the calibration and inference-deadline gates in the
[revised proposal](detection-correction-memory-proposal.md).
