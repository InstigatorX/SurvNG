import assert from "node:assert/strict";

globalThis.window = { __SURVNG_BASE_PATH__: "/survng" };
globalThis.document = { documentElement: { dataset: {} } };
const { eventSnapshotUrl, eventSnapshotDownloadUrl, eventThumbnailUrl } = await import("../src/shared/mediaUrls.js");

const trigger = { id: 68967, snapshot_path: "snapshots/downstairs/trigger.webp" };
const refined = { ...trigger, snapshot_path: "snapshots/downstairs/selected.webp" };
for (const makeUrl of [eventSnapshotUrl, eventThumbnailUrl]) {
  const before = makeUrl(trigger);
  const after = makeUrl(refined);
  assert.notEqual(before, after, "A refined snapshot must not reuse the cached trigger raster");
  assert.equal(after, makeUrl({ ...refined }), "Unchanged snapshots should retain their cache key");
  assert.equal(new URL(after, "http://localhost").searchParams.get("v"), refined.snapshot_path);
  assert.ok(after.startsWith("/survng/api/events/68967/"));
}
assert.equal(eventSnapshotUrl({ ...refined, id: 1, representative_event_id: 68967 }), eventSnapshotUrl(refined));
const download = new URL(eventSnapshotDownloadUrl(refined), "http://localhost");
assert.equal(download.searchParams.get("v"), refined.snapshot_path);
assert.equal(download.searchParams.get("download"), "true");
const focused = new URL(eventThumbnailUrl(refined, 960, 90, { objectFocus: true, incidentEligibleOnly: true }), "http://localhost");
assert.equal(focused.searchParams.get("v"), refined.snapshot_path);
assert.equal(focused.searchParams.get("object_focus"), "true");
assert.equal(focused.searchParams.get("width"), "960");
const evidence = { ...refined, snapshot_url: "/api/cameras/downstairs/recordings/preview.jpg?epoch=123&exact=true" };
assert.equal(eventSnapshotUrl(evidence), `/survng${evidence.snapshot_url}`);
assert.equal(eventThumbnailUrl(evidence), eventSnapshotUrl(evidence));
assert.equal(new URL(eventSnapshotDownloadUrl(evidence), "http://localhost").searchParams.get("exact"), "true");
assert.equal(eventSnapshotUrl({ id: 7 }), "/survng/api/events/7/snapshot.jpg");
console.log("event image revision URL tests passed");
