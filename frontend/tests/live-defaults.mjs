import assert from "node:assert/strict";
import {
  LIVE_DEFAULTS_INSTANCE_KEY,
  resetLiveDefaultsForServer,
} from "../src/liveDefaults.mjs";

class MemoryStorage {
  constructor(values = []) {
    this.values = new Map(values);
  }

  get length() { return this.values.size; }
  key(index) { return [...this.values.keys()][index] ?? null; }
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, value); }
  removeItem(key) { this.values.delete(key); }
}

const storage = new MemoryStorage([
  [LIVE_DEFAULTS_INSTANCE_KEY, "old-server"],
  ["survng.streamMode.v3.gate", "webrtc"],
  ["survng.sourceMode.gate", "main"],
  ["survng.liveOverlaySource.gate", "main"],
  ["survng.liveCameraOrder.v1", "[\"gate\"]"],
  ["survng.liveAspect.gate.live", "4 / 3"],
]);

assert.equal(resetLiveDefaultsForServer(storage, "new-server"), true);
assert.equal(storage.getItem("survng.streamMode.v3.gate"), null);
assert.equal(storage.getItem("survng.sourceMode.gate"), null);
assert.equal(storage.getItem("survng.liveOverlaySource.gate"), null);
assert.equal(storage.getItem("survng.liveCameraOrder.v1"), "[\"gate\"]");
assert.equal(storage.getItem("survng.liveAspect.gate.live"), "4 / 3");
assert.equal(storage.getItem(LIVE_DEFAULTS_INSTANCE_KEY), "new-server");
assert.equal(resetLiveDefaultsForServer(storage, "new-server"), false);
assert.equal(resetLiveDefaultsForServer(storage, ""), false);

console.log("live restart defaults tests passed");
