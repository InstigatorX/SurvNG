import assert from "node:assert/strict";
import {
  normalizeTrailEventIds,
  parseTrailEventIds,
  readVisualSearchTrail,
  serializeTrailEventIds,
  trailHitForEvent,
  trailPosition,
  writeVisualSearchTrail,
} from "../src/visualSearchTrail.mjs";
import { parseTimelineView } from "../src/timelineWorkspace.mjs";
import { timelineHref } from "../src/workspaceNavigation.mjs";

assert.deepEqual(normalizeTrailEventIds([1, "1", 2, 0, -3, 2, "x", 3]), [1, 2, 3]);
assert.equal(serializeTrailEventIds([9, 8, 9]), "9,8");
assert.deepEqual(parseTrailEventIds("9,8,8,bad"), [9, 8]);
assert.deepEqual(trailPosition([9, 8, 7], 8), {
  index: 1,
  count: 3,
  previousId: 9,
  nextId: 7,
});

const storage = {
  value: "",
  setItem(_key, value) { this.value = String(value); },
  getItem() { return this.value; },
  removeItem() { this.value = ""; },
};
const written = writeVisualSearchTrail(storage, {
  eventIds: [12, 84],
  hits: [
    { query_mode: "appearance", event: { id: 12, camera_id: "gate", created_at: "2026-01-01T00:00:00Z" } },
    { query_mode: "visual", event: { id: 84, camera_id: "drive", created_at: "2026-01-01T01:00:00Z" } },
  ],
});
assert.equal(written.eventIds.length, 2);
const read = readVisualSearchTrail(storage);
assert.equal(trailHitForEvent(read, 84)?.query_mode, "visual");
assert.equal(trailHitForEvent(read, 12)?.event?.snapshot_path, "available");

assert.equal(
  timelineHref({
    cameraId: "gate",
    epoch: 100,
    eventId: 84,
    trailEventIds: [12, 84, 91],
  }),
  "/timeline?camera=gate&at=100&event=84&trail=12%2C84%2C91",
);

const view = parseTimelineView("?camera=gate&at=100&event=84&trail=12,84,91&query_mode=appearance", "2026-08-27");
assert.deepEqual(view.trailEventIds, [12, 84, 91]);
assert.equal(view.trailQueryMode, "appearance");
assert.equal(view.eventId, 84);

console.log("visual search trail tests passed");
