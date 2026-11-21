import assert from "node:assert/strict";
import {
  ALL_CAMERAS_WINDOW_SECONDS,
  allCamerasGridColumns,
  allCamerasWindowStart,
  allCamerasWithRecording,
  recordingGridBestEpoch,
} from "../src/recordingGrid.mjs";

assert.equal(ALL_CAMERAS_WINDOW_SECONDS, 900);
assert.equal(allCamerasWindowStart(1800 + 899.9), 1800);
assert.equal(allCamerasWindowStart(2700), 2700);
{
  const cameras = [{ id: "gate" }, { id: "yard" }, { id: "drive" }, { id: "barn" }];
  const ranges = [
    { camera_id: "gate", source: "live", start_epoch: 1700, end_epoch: 1810 },
    { camera_id: "yard", source: "main", start_epoch: 2600, end_epoch: 2700 },
    { camera_id: "drive", source: "live", start_epoch: 2700, end_epoch: 2800 },
  ];
  assert.deepEqual(allCamerasWithRecording(cameras, ranges, 2000).map((camera) => camera.id), ["gate", "yard"],
    "a camera shows for the whole window it has any recording in, from either stream; touching the edge does not count");
  assert.deepEqual(allCamerasWithRecording(cameras, ranges, 2750).map((camera) => camera.id), ["drive"]);
  assert.deepEqual(allCamerasWithRecording(cameras, [], 2000), []);
}
assert.deepEqual([0, 1, 2, 3, 4, 5, 9, 10, 16].map(allCamerasGridColumns), [1, 1, 2, 2, 2, 3, 3, 4, 4], "the grid stays roughly square");

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
