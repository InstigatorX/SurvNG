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
  assert.notDeepEqual(childGeometry(mixed), childGeometry(layout), "children should repack around the primary aspect");
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

const many = Array.from({ length: 64 }, (_, index) => ({ id: String(index), live_view: { live: { focal_x: index, zoom: 1 + index / 100 } } }));
const portrait = focusLiveMosaicLayout(many.slice(0, 13), 1618, 914, "0", 4, 3 / 4);
assert.ok(portrait[0].height > 600, "the screenshot's portrait primary must grow beyond its old four-cell slot");
assert.ok(portrait[0].width > 450);
assert.ok(Math.abs(portrait[1].x - portrait[0].x - portrait[0].width - 4) < 1e-8, "children start beside the actual primary edge, with no reserved blank slot");
for (const count of [2, 3, 6, 13, 25, 64]) {
  for (const [width, height] of [[952, 766], [1618, 914], [1024, 400], [400, 1000]]) {
    for (const aspect of [9 / 16, 3 / 4, 1, 4 / 3, 16 / 9, 32 / 9]) {
      const tiles = focusLiveMosaicLayout(many.slice(0, count), width, height, "1", 4, aspect);
      assert.equal(tiles.length, count);
      assert.equal(tiles[0].camera.id, "1");
      assert.ok(Math.abs(tiles[0].width / tiles[0].height - aspect) < 1e-8);
      for (const tile of tiles) {
        assert.equal(tile.camera, many[Number(tile.camera.id)], "repacking preserves each camera and its saved framing");
        assert.ok(tile.x >= -1e-8 && tile.y >= -1e-8 && tile.width > 0 && tile.height > 0);
        assert.ok(tile.x + tile.width <= width + 1e-8 && tile.y + tile.height <= height + 1e-8);
        if (!tile.primary) {
          assert.ok(Math.abs(tile.width / tile.height - UNIFORM_LIVE_TILE_ASPECT) < 1e-8);
          assert.equal(tile.width, tiles[1].width);
          assert.equal(tile.height, tiles[1].height);
        }
      }
      for (let i = 0; i < tiles.length; i += 1) {
        for (let j = i + 1; j < tiles.length; j += 1) {
          const a = tiles[i], b = tiles[j];
          assert.equal(a.x < b.x + b.width - 1e-8 && a.x + a.width > b.x + 1e-8 && a.y < b.y + b.height - 1e-8 && a.y + a.height > b.y + 1e-8, false, "no overlapping panes");
        }
      }
    }
  }
}

console.log("live focus layout tests passed");
