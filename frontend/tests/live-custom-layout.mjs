import assert from "node:assert/strict";
import {
  liveCustomGridMetrics,
  moveLiveCamera,
  readLiveCustomLayout,
  resizeLiveCamera,
} from "../src/liveCustomLayout.mjs";

const cameras = [{ id: "gate" }, { id: "door" }, { id: "portrait" }];
const layout = readLiveCustomLayout(
  JSON.stringify({ order: ["door", "missing", "door"], sizes: { door: { columns: 99, rows: 0 } } }),
  cameras,
  { portrait: 0.75 },
);
assert.deepEqual(layout.order, ["door", "gate", "portrait"]);
assert.deepEqual(layout.sizes.door, { columns: 12, rows: 1 });
assert.deepEqual(layout.sizes.portrait, { columns: 3, rows: 2 });
assert.deepEqual(moveLiveCamera(layout.order, "portrait", "door"), ["portrait", "door", "gate"]);
assert.deepEqual(resizeLiveCamera({ columns: 3, rows: 1 }, 2, 1), { columns: 5, rows: 2 });
assert.deepEqual(resizeLiveCamera({ columns: 12, rows: 4 }, 4, 3), { columns: 12, rows: 4 });
assert.equal(liveCustomGridMetrics(1200, 800).columnWidth > 90, true);
assert.equal(liveCustomGridMetrics(1200, 800).rowHeight, 194);

console.log("live custom layout tests passed");
