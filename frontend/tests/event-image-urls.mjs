import assert from "node:assert/strict";

globalThis.window = { __SURVNG_BASE_PATH__: "/survng" };
globalThis.document = { documentElement: { dataset: {} } };
const { eventSnapshotUrl, eventSnapshotDownloadUrl, eventThumbnailUrl } = await import("../src/shared/mediaUrls.js");

const trigger = { id: 68967, snapshot_path: "available", evidence_revision: 1 };
const refined = { ...trigger, evidence_revision: 2 };
for (const makeUrl of [eventSnapshotUrl, eventThumbnailUrl]) {
  const before = makeUrl(trigger);
  const after = makeUrl(refined);
  assert.notEqual(before, after, "A refined snapshot must not reuse the cached trigger raster");
  assert.equal(after, makeUrl({ ...refined }), "Unchanged snapshots should retain their cache key");
  assert.equal(new URL(after, "http://localhost").searchParams.get("v"), String(refined.evidence_revision));
  assert.ok(after.startsWith("/survng/api/events/68967/"));
}
assert.equal(eventSnapshotUrl({ ...refined, id: 1, representative_event_id: 68967 }), eventSnapshotUrl(refined));
const download = new URL(eventSnapshotDownloadUrl(refined), "http://localhost");
assert.equal(download.searchParams.get("v"), String(refined.evidence_revision));
assert.equal(download.searchParams.get("download"), "true");
const focused = new URL(eventThumbnailUrl(refined, 960, 90, { objectFocus: true, incidentEligibleOnly: true }), "http://localhost");
assert.equal(focused.searchParams.get("v"), String(refined.evidence_revision));
assert.equal(focused.searchParams.get("object_focus"), "true");
assert.equal(focused.searchParams.get("width"), "960");
const evidence = { ...refined, snapshot_url: "/api/cameras/downstairs/recordings/preview.jpg?epoch=123&exact=true" };
assert.equal(eventSnapshotUrl(evidence), `/survng${evidence.snapshot_url}`);
assert.equal(eventThumbnailUrl(evidence), eventSnapshotUrl(evidence));
assert.equal(new URL(eventSnapshotDownloadUrl(evidence), "http://localhost").searchParams.get("exact"), "true");
assert.equal(eventSnapshotUrl({ id: 7 }), "/survng/api/events/7/snapshot.jpg");
assert.equal(eventSnapshotUrl({ id: 7, snapshot_path: "snapshots/legacy.webp" }), "/survng/api/events/7/snapshot.jpg");
// Tile pictures are asked for near their shown size, from a short list of widths.
const { THUMBNAIL_WIDTHS, thumbnailWidthFor } = await import("../src/shared/mediaUrls.js");
assert.equal(thumbnailWidthFor(282, 1), 320);
assert.equal(thumbnailWidthFor(282, 2), 640);
assert.equal(thumbnailWidthFor(282, 3), 640, "density is capped at 2");
assert.equal(thumbnailWidthFor(135, 1), 240);
assert.equal(thumbnailWidthFor(160, 2), 320);
assert.equal(thumbnailWidthFor(560, 1), 640);
assert.equal(thumbnailWidthFor(900, 2), 960, "never beyond the largest listed width");
for (const bad of [0, -5, Number.NaN, undefined]) assert.equal(thumbnailWidthFor(bad, 1), THUMBNAIL_WIDTHS[0]);
assert.equal(thumbnailWidthFor(282, "bad"), 320);
assert.match(eventThumbnailUrl({ id: 7, evidence_revision: 2 }, thumbnailWidthFor(282, 1)), /thumbnail\.jpg\?width=320&quality=82&v=2$/);

// A card that names its picture's version carries it, so the browser may keep the picture.
assert.match(eventThumbnailUrl({ id: 7, evidence_revision: 2, image_version: "ab12" }, 320), /thumbnail\.jpg\?width=320&quality=82&v=2&s=ab12$/);
assert.match(eventThumbnailUrl({ snapshot_url: "/api/incidents/observations/obs-1/snapshot", snapshot_observation_id: "obs-1", image_version: "cd34" }, 320),
  /observations\/obs-1\/snapshot\?width=320&quality=82&s=cd34$/);
assert.doesNotMatch(eventThumbnailUrl({ id: 7, evidence_revision: 2 }, 320), /[?&]s=/);

console.log("event image revision URL tests passed");
