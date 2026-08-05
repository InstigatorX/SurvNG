import assert from "node:assert/strict";
import {
  recordingCameraAspect,
  recordingGridBestEpoch,
  recordingGridLayout,
} from "../src/recordingGrid.mjs";

const cameras = [
  { id: "wide-1", stream_dimensions: { live: { width: 1920, height: 1080 } } },
  { id: "portrait", stream_dimensions: { live: { width: 720, height: 1280 } } },
  { id: "wide-2", stream_dimensions: { live: { width: 1920, height: 1080 } } },
  { id: "four-three", stream_dimensions: { live: { width: 640, height: 480 } } },
];

assert.equal(recordingCameraAspect(cameras[0], "live"), 16 / 9);
assert.equal(recordingCameraAspect({ stream_dimensions: {} }, "live"), 16 / 9);

const layout = recordingGridLayout(cameras, "live", 1000, 500, 6);
assert.equal(layout.length, cameras.length);
assert.deepEqual(layout.map((item) => item.camera.id).sort(), cameras.map((camera) => camera.id).sort());
layout.forEach((item) => {
  assert.ok(item.x >= 0);
  assert.ok(item.y >= 0);
  assert.ok(item.x + item.width <= 1000.001);
  assert.ok(item.y + item.height <= 500.001);
  assert.ok(Math.abs(item.width / item.height - recordingCameraAspect(item.camera, "live")) < 0.001);
});
const overridden = recordingGridLayout(cameras, "live", 1000, 500, 6, { portrait: 1 });
const overriddenPortrait = overridden.find((item) => item.camera.id === "portrait");
assert.ok(Math.abs(overriddenPortrait.width / overriddenPortrait.height - 1) < 0.001);

const ranges = [
  { camera_id: "ahead", start_epoch: 980, end_epoch: 1000 },
  { camera_id: "one", start_epoch: 900, end_epoch: 990 },
  { camera_id: "two", start_epoch: 910, end_epoch: 990 },
  { camera_id: "three", start_epoch: 920, end_epoch: 988 },
];
assert.equal(recordingGridBestEpoch(ranges, 999), 986);
assert.equal(recordingGridBestEpoch([], 999), null);
assert.equal(recordingGridBestEpoch(ranges, Number.NaN), null);

console.log("recording grid tests passed");
