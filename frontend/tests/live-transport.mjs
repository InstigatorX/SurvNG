import assert from "node:assert/strict";
import {
  WEBRTC_FAILURE_COOLDOWN_MS,
  clearWebRtcFailure,
  initialLiveTransport,
  nextNativeFallbackSource,
  rememberWebRtcFailure,
  webRtcFailureKey,
  webRtcRetryDelay,
} from "../src/liveTransport.mjs";

class MemoryStorage {
  constructor() {
    this.values = new Map();
  }

  getItem(key) {
    return this.values.get(key) ?? null;
  }

  setItem(key, value) {
    this.values.set(key, value);
  }

  removeItem(key) {
    this.values.delete(key);
  }
}

const storage = new MemoryStorage();
const now = 1_800_000_000_000;
assert.equal(initialLiveTransport("gate", "live", storage, now), "webrtc");
assert.equal(nextNativeFallbackSource("main", "main"), "live");
assert.equal(nextNativeFallbackSource("main", "live"), null);
assert.equal(nextNativeFallbackSource("live", "live"), null);

rememberWebRtcFailure("gate", "live", storage, now);
assert.equal(initialLiveTransport("gate", "live", storage, now + 1_000), "mse");
assert.equal(initialLiveTransport("gate", "live", storage, now - 1_000), "webrtc");
assert.equal(initialLiveTransport("gate", "main", storage, now + 1_000), "webrtc");
assert.equal(
  webRtcRetryDelay("gate", "live", storage, now + 1_000),
  WEBRTC_FAILURE_COOLDOWN_MS - 1_000,
);
assert.equal(
  initialLiveTransport("gate", "live", storage, now + WEBRTC_FAILURE_COOLDOWN_MS),
  "webrtc",
);

clearWebRtcFailure("gate", "live", storage);
assert.equal(storage.getItem(webRtcFailureKey("gate", "live")), null);
assert.equal(initialLiveTransport("gate", "live", storage, now), "webrtc");

console.log("live transport tests passed");
