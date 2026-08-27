import assert from "node:assert/strict";
import {
  appearanceMatchesPath,
  visualMatchLabel,
  visualSearchRequest,
} from "../src/visualSearch.mjs";

assert.deepEqual(visualSearchRequest({
  eventId: 42,
  objectIndex: 1,
  cameraFilter: "gate",
  objectFilter: "person",
  startAt: "start",
  endAt: "end",
  sourceKinds: ["object_crop"],
}), {
  event_id: 42,
  object_index: 1,
  camera_ids: ["gate"],
  object_labels: ["person"],
  start_at: "start",
  end_at: "end",
  limit: 50,
  source_kinds: ["object_crop"],
  exclude_anchor: true,
});

assert.equal(visualMatchLabel("strong_match"), "Strong match");
assert.equal(visualMatchLabel("possible_match"), "Possible match");
assert.equal(visualMatchLabel(""), "Visually similar");

assert.equal(
  appearanceMatchesPath(9, { hours: 12, limit: 8, trackId: 3, crossCameraOnly: false }),
  "/api/events/9/appearance-matches?hours=12&limit=8&cross_camera_only=false&track_id=3",
);

console.log("visual search helper tests passed");
