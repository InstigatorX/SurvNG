import assert from "node:assert/strict";
import { focusLiveMosaicLayout, focusedLiveCameraId } from "../src/liveFocusLayout.mjs";
import { UNIFORM_LIVE_TILE_ASPECT, uniformLiveGridLayout } from "../src/liveWorkspace.mjs";

const cameras = [{ id: "front" }, { id: "drive" }, { id: "gate" }];

assert.equal(focusedLiveCameraId(cameras, "drive"), "drive");
assert.equal(focusedLiveCameraId(cameras, "missing"), "front");
assert.equal(focusedLiveCameraId([], "front"), "");
assert.deepEqual(focusLiveMosaicLayout([], 1000, 600, "front"), []);
assert.deepEqual(focusLiveMosaicLayout(cameras, Infinity, 600, "front"), []);
assert.deepEqual(focusLiveMosaicLayout(cameras, 1000, NaN, "front"), []);

const one = focusLiveMosaicLayout(cameras.slice(0, 1), 1000, 600, "front");
assert.equal(one[0].width / one[0].height, 16 / 9);
assert.equal(one[0].y, (600 - one[0].height) / 2);

const layout = focusLiveMosaicLayout(cameras, 952, 766, "drive");
assert.equal(layout.length, 3);
assert.equal(layout[0].camera.id, "drive");
assert.equal(layout[0].primary, true);
assert.ok(layout[0].width > layout[1].width);
assert.ok(layout[0].width / layout[0].height > 1.7);
assert.ok(layout[1].width / layout[1].height > 1.7);

const five = focusLiveMosaicLayout([...cameras, { id: "garage" }, { id: "yard" }], 1000, 600, "front");
assert.equal(five[1].width, five[2].width);
const automatic = uniformLiveGridLayout(cameras, 952, 766);
const childGeometry = (tiles) => tiles.slice(1).map(({ x, y, width, height }) => ({ x, y, width, height }));
for (const primaryAspect of [9 / 16, 1, 4 / 3, 32 / 9]) {
  const mixed = focusLiveMosaicLayout(cameras, 952, 766, "drive", 4, primaryAspect);
  assert.deepEqual(childGeometry(mixed), childGeometry(layout), "changing the primary aspect must not reshape or move the children");
  assert.ok(Math.abs(mixed[0].width / mixed[0].height - primaryAspect) < 1e-10, "primary retains its own aspect");
  for (const child of mixed.slice(1)) {
    assert.ok(Math.abs(child.width / child.height - UNIFORM_LIVE_TILE_ASPECT) < 1e-10);
    assert.ok(Math.abs(child.width / child.height - automatic[0].width / automatic[0].height) < 1e-10, "children use Automatic's crop viewport");
    const primary = mixed[0];
    assert.equal(primary.x < child.x + child.width && primary.x + primary.width > child.x && primary.y < child.y + child.height && primary.y + primary.height > child.y, false);
  }
}
for (const candidate of [layout, five]) {
  const bounds = candidate === layout ? { width: 952, height: 766 } : { width: 1000, height: 600 };
  for (const tile of candidate) {
    assert.ok(tile.x >= 0 && tile.y >= 0);
    assert.ok(tile.x + tile.width <= bounds.width);
    assert.ok(tile.y + tile.height <= bounds.height);
  }
  for (let left = 0; left < candidate.length; left += 1) {
    for (let right = left + 1; right < candidate.length; right += 1) {
      const a = candidate[left];
      const b = candidate[right];
      const overlaps = a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
      assert.equal(overlaps, false, `${a.camera.id} and ${b.camera.id} should not overlap`);
    }
  }
}

console.log("live focus layout tests passed");
