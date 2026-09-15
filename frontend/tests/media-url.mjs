import assert from "node:assert/strict";
import { safeMediaUrl } from "../src/mediaUrl.mjs";

assert.equal(safeMediaUrl("/api/events/1/image", "/survng", "https://nvr.test"), "/survng/api/events/1/image");
assert.equal(safeMediaUrl("/survng/api/events/1/image", "/survng", "https://nvr.test"), "/survng/api/events/1/image");
assert.equal(safeMediaUrl("blob:https://nvr.test/id", "/survng", "https://nvr.test"), "blob:https://nvr.test/id");
assert.equal(safeMediaUrl("data:image/webp;base64,AA", "/survng", "https://nvr.test"), "data:image/webp;base64,AA");
assert.equal(safeMediaUrl("https://nvr.test/api/image", "/survng", "https://nvr.test"), "https://nvr.test/api/image");
assert.equal(safeMediaUrl("https://other.test/image", "/survng", "https://nvr.test"), "");
assert.equal(safeMediaUrl("//other.test/image", "/survng", "https://nvr.test"), "");

globalThis.window = { __SURVNG_BASE_PATH__: "/survng" };
globalThis.document = { documentElement: { dataset: {} } };
const { eventSnapshotUrl, eventThumbnailUrl } = await import("../src/shared/mediaUrls.js");
const oldCover = { id: 7, snapshot_path: "available", evidence_revision: 1 };
const newCover = { ...oldCover, evidence_revision: 2 };
assert.equal(eventSnapshotUrl(oldCover), "/survng/api/events/7/snapshot.jpg?v=1");
assert.equal(eventSnapshotUrl(newCover), "/survng/api/events/7/snapshot.jpg?v=2");
assert.notEqual(eventThumbnailUrl(oldCover), eventThumbnailUrl(newCover));
assert.equal(eventSnapshotUrl({ id: 7, snapshot_path: "available" }), "/survng/api/events/7/snapshot.jpg");
assert.equal(eventSnapshotUrl({ ...newCover, snapshot_url: "/api/recording/preview?epoch=1" }), "/survng/api/recording/preview?epoch=1");

console.log("media URL tests passed");
