import assert from "node:assert/strict";
import { focusedLiveCameraId, liveActivityEventId, liveActivityIncidentHref, orderedLiveCamerasForFocus } from "../src/liveWorkspace.mjs";

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

console.log("live workspace tests passed");
