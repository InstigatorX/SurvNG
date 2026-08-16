import assert from "node:assert/strict";
import { focusedLiveCameraId, liveActivityEventId, liveActivityIncidentHref, liveActivityQuickFilter, liveActivityQuickSelection, liveDensityPage, normalizedLiveDensity, orderedLiveCamerasForFocus } from "../src/liveWorkspace.mjs";

const cameras = [{ id: "gate" }, { id: "front-door" }];
assert.equal(focusedLiveCameraId(cameras, "front-door"), "front-door");
assert.equal(focusedLiveCameraId(cameras, "missing"), "gate");
assert.equal(focusedLiveCameraId([], "gate"), "");
assert.deepEqual(orderedLiveCamerasForFocus(cameras, "front-door", true).map((camera) => camera.id), ["front-door", "gate"]);
assert.deepEqual(orderedLiveCamerasForFocus(cameras, "front-door", false).map((camera) => camera.id), ["gate", "front-door"]);
assert.equal(orderedLiveCamerasForFocus(cameras, "front-door", true)[0], cameras[1]);
assert.equal(liveActivityEventId({ representative_event_id: 42, id: 9 }), 42);
assert.equal(liveActivityEventId({ events: [{ id: 17 }], id: 9 }), 17);
assert.equal(liveActivityIncidentHref({ representative_event_id: 42 }), "/incidents?event_ids=42");
assert.equal(liveActivityIncidentHref({}), "/incidents");
assert.equal(normalizedLiveDensity("6"), "6");
assert.equal(normalizedLiveDensity("bogus"), "fit");
assert.deepEqual(liveDensityPage([{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }], "4", 1), {
  cameras: [{ id: 5 }],
  page: 1,
  pageCount: 2,
});
assert.deepEqual(liveDensityPage(cameras, "fit", 9), { cameras, page: 0, pageCount: 1 });
assert.equal(liveActivityQuickFilter("all", "all"), "all");
assert.equal(liveActivityQuickFilter("object", "person"), "person");
assert.equal(liveActivityQuickFilter("object", "car"), "vehicle");
assert.equal(liveActivityQuickFilter("motion", "all"), "custom");
assert.deepEqual(liveActivityQuickSelection("vehicle"), { eventType: "object", objectFilter: "car" });

console.log("live workspace tests passed");
