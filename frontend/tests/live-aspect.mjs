import assert from "node:assert/strict";
import {
  DEFAULT_LIVE_ASPECT,
  aspectFromDimensions,
  cameraSourceAspect,
  initialCameraAspect,
  liveAspectStorageKey,
  storedCameraAspect,
  validLiveAspect,
} from "../src/liveAspect.mjs";

class MemoryStorage {
  constructor(values = []) {
    this.values = new Map(values);
  }

  getItem(key) {
    return this.values.get(key) ?? null;
  }
}

const camera = {
  id: "front-door",
  stream_dimensions: {
    live: { width: 672, height: 896 },
    main: { width: 2160, height: 3840 },
  },
};
const storage = new MemoryStorage([
  [liveAspectStorageKey("front-door", "live"), "16 / 9"],
  [liveAspectStorageKey("cached", "main"), "4 / 3"],
]);

assert.equal(aspectFromDimensions(672, 896), "672 / 896");
assert.equal(aspectFromDimensions(0, 896), null);
assert.equal(validLiveAspect(" 4 / 3 "), "4 / 3");
assert.equal(validLiveAspect("auto"), null);
assert.equal(cameraSourceAspect(camera, "live"), "672 / 896");
assert.equal(cameraSourceAspect(camera, "main"), "2160 / 3840");
assert.equal(initialCameraAspect(camera, "live", storage), "672 / 896");
assert.equal(initialCameraAspect({ id: "cached" }, "main", storage), "4 / 3");
assert.equal(initialCameraAspect({ id: "unknown" }, "live", storage), DEFAULT_LIVE_ASPECT);
assert.equal(storedCameraAspect(storage, "front-door", "sub"), "16 / 9");

console.log("live aspect tests passed");
