import assert from "node:assert/strict";
import {
  liveCustomGridMetrics,
  liveCustomDropTarget,
  liveCustomTilePlacement,
  moveLiveCamera,
  readLiveCustomLayout,
  resizeLiveCamera,
  resizeLiveCameraToAspect,
} from "../src/liveCustomLayout.mjs";

const cameras = [{ id: "gate" }, { id: "door" }, { id: "portrait" }];
const layout = readLiveCustomLayout(
  JSON.stringify({ order: ["door", "missing", "door"], sizes: { door: { columns: 99, rows: 0 } } }),
  cameras,
  { portrait: 0.75 },
);
assert.deepEqual(layout.order, ["door", "gate", "portrait"]);
assert.deepEqual(layout.sizes.door, { columns: 12, rows: 1, aspectLocked: false });
assert.deepEqual(layout.sizes.portrait, { columns: 3, rows: 2, aspectLocked: false });
assert.deepEqual(moveLiveCamera(layout.order, "portrait", "door"), ["portrait", "door", "gate"]);
assert.deepEqual(moveLiveCamera(layout.order, "door", "gate", "after"), ["gate", "door", "portrait"]);
assert.deepEqual(moveLiveCamera(["door", "gate", "portrait"], "door", "gate", "after"), ["gate", "door", "portrait"]);
assert.deepEqual(moveLiveCamera(["door", "gate", "portrait"], "door", "portrait", "swap"), ["portrait", "gate", "door"]);
assert.deepEqual(moveLiveCamera(layout.order, "door", "door", "after"), layout.order);
const dragSlots = [
  { id: "door", left: 0, top: 0, width: 100, height: 100 },
  { id: "gate", left: 110, top: 0, width: 100, height: 100 },
  { id: "portrait", left: 220, top: 0, width: 100, height: 210 },
];
assert.deepEqual(liveCustomDropTarget(dragSlots, 50, 50, "door"), { targetId: "door", position: "original" });
assert.deepEqual(liveCustomDropTarget(dragSlots, 130, 50, "door"), { targetId: "gate", position: "swap" });
assert.deepEqual(liveCustomDropTarget(dragSlots, 180, 50, "door"), { targetId: "gate", position: "swap" });
assert.deepEqual(liveCustomDropTarget(dragSlots, 270, 160, "door"), { targetId: "portrait", position: "swap" });
assert.deepEqual(liveCustomDropTarget(dragSlots, 112, 50, "door"), { targetId: "gate", position: "before" });
assert.deepEqual(liveCustomDropTarget(dragSlots, 208, 50, "door"), { targetId: "gate", position: "after" });
assert.equal(liveCustomDropTarget([], 50, 50, "door"), null);
assert.deepEqual(resizeLiveCamera({ columns: 3, rows: 1 }, 2, 1), { columns: 5, rows: 2, aspectLocked: false });
assert.deepEqual(resizeLiveCamera({ columns: 12, rows: 4 }, 4, 3), { columns: 12, rows: 4, aspectLocked: false });
assert.equal(liveCustomGridMetrics(1200, 800).columnWidth > 90, true);
assert.equal(liveCustomGridMetrics(1200, 800).rowHeight, 197);
assert.equal(liveCustomGridMetrics(1200, 800).packRowHeight, 2);
assert.equal(liveCustomGridMetrics(1200, 800).gap, 4);

const snapped = resizeLiveCameraToAspect(
  { columns: 3, rows: 1 },
  95,
  0,
  liveCustomGridMetrics(1200, 800),
  4 / 3,
);
assert.equal(snapped.aspectLocked, true);
assert.equal(snapped.columns >= 2, true);
assert.equal(snapped.rows >= 1, true);

assert.deepEqual(
  resizeLiveCameraToAspect(
    { columns: 3, rows: 1 },
    100,
    194,
    liveCustomGridMetrics(1200, 800),
    0,
  ),
  { columns: 4, rows: 2, aspectLocked: false },
);

const packedLandscape = liveCustomTilePlacement(
  { columns: 3, rows: 2, aspectLocked: true },
  liveCustomGridMetrics(1200, 800),
  16 / 9,
);
assert.equal(packedLandscape.columns, 3);
assert.equal(Math.abs(packedLandscape.height - 167.0625) < 0.001, true);
assert.equal(packedLandscape.packedRows, 29);

const packedFreeform = liveCustomTilePlacement(
  { columns: 4, rows: 2, aspectLocked: false },
  liveCustomGridMetrics(1200, 800),
  16 / 9,
);
assert.equal(packedFreeform.height, 398);
assert.equal(packedFreeform.packedRows, 67);

console.log("live custom layout tests passed");
