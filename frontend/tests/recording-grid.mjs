import assert from "node:assert/strict";
import {
  ALL_CAMERAS_WINDOW_SECONDS,
  allCamerasCustomIds,
  allCamerasGridSelection,
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
{
  // Custom set: stored ids that still exist, in camera order; none means all.
  const cameras = [{ id: "a" }, { id: "b" }, { id: "c" }];
  assert.deepEqual(allCamerasCustomIds('["c","a","gone"]', cameras), ["a", "c"]);
  assert.deepEqual(allCamerasCustomIds('["gone"]', cameras), ["a", "b", "c"]);
  assert.deepEqual(allCamerasCustomIds("", cameras), ["a", "b", "c"]);
  assert.deepEqual(allCamerasCustomIds("not json", cameras), ["a", "b", "c"]);
}
{
  const cameras = ["a", "b", "c", "d", "e", "f"].map((id) => ({ id }));
  const ranges = cameras.filter((camera) => camera.id !== "f").map((camera) => ({ camera_id: camera.id, start_epoch: 0, end_epoch: 900 }));
  const base = { cameras, ranges, epoch: 100, mode: "auto", density: "4", page: 1, customIds: ["b", "f"], focusCameraId: "c" };
  const auto = allCamerasGridSelection(base);
  assert.deepEqual([auto.layoutMode, auto.density, auto.page, auto.pageCount], ["auto", "4", 1, 2], "Automatic pages the recorded cameras by density");
  assert.deepEqual(auto.shown.map((camera) => camera.id), ["e"]);
  assert.deepEqual(auto.unrecorded.map((camera) => camera.id), ["f"]);
  const custom = allCamerasGridSelection({ ...base, mode: "custom" });
  assert.deepEqual([custom.density, custom.pageCount], ["fit", 1], "Custom shows its whole set, like Live");
  assert.deepEqual(custom.shown.map((camera) => camera.id), ["b"]);
  assert.deepEqual(custom.unrecorded.map((camera) => camera.id), ["f"], "a chosen camera without recording is named, not shown");
  const focus = allCamerasGridSelection({ ...base, mode: "focus" });
  assert.deepEqual([focus.layoutMode, focus.density, focus.focusCameraId, focus.shown.length], ["focus", "fit", "c", 5]);
  assert.equal(allCamerasGridSelection({ ...base, mode: "focus", focusCameraId: "f" }).focusCameraId, "a", "a focus camera without recording falls to the first shown");
  assert.deepEqual(allCamerasGridSelection({ ...base, mode: "focus", focusScope: "custom", focusCameraId: "b" }).shown.map((camera) => camera.id), ["b"], "Focus entered from Custom keeps the chosen set");
  const narrow = allCamerasGridSelection({ ...base, mode: "custom", layoutsAvailable: false });
  assert.deepEqual([narrow.layoutMode, narrow.density, narrow.pageCount], ["auto", "fit", 1], "below desktop width, Automatic at fit applies (Live hides density there)");
  assert.equal(allCamerasGridSelection({ ...base, mode: "bogus" }).layoutMode, "auto");
  assert.equal(allCamerasGridSelection({ ...base, page: 9 }).page, 1, "the page is clamped");
}

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
