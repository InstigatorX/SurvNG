import assert from "node:assert/strict";
import { filteredTimelineCameras, normalizedTimelinePlaybackRate, TIMELINE_PLAYBACK_RATES } from "../src/timelineWorkspace.mjs";

assert.deepEqual(TIMELINE_PLAYBACK_RATES, [0.5, 1, 2, 4]);
assert.equal(normalizedTimelinePlaybackRate("2"), 2);
assert.equal(normalizedTimelinePlaybackRate(3), 1);

const cameras = [
  { id: "front-door", name: "Front Door" },
  { id: "garage", name: "Lower Garage" },
];
assert.deepEqual(filteredTimelineCameras(cameras, "front"), [cameras[0]]);
assert.deepEqual(filteredTimelineCameras(cameras, "GARAGE"), [cameras[1]]);
assert.equal(filteredTimelineCameras(cameras, "missing").length, 0);
assert.equal(filteredTimelineCameras(cameras, "").length, 2);

console.log("timeline workspace tests passed");
