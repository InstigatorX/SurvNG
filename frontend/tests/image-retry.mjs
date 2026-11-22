import assert from "node:assert/strict";
import { imageFinished, imageVisible, retryImageUrl } from "../src/shared/imageRetry.mjs";

assert.equal(retryImageUrl("/api/events/7/thumbnail.jpg?width=720", 0), "/api/events/7/thumbnail.jpg?width=720");
assert.equal(retryImageUrl("/api/events/7/thumbnail.jpg?width=720", 1), "/api/events/7/thumbnail.jpg?width=720&retry=1");
assert.equal(retryImageUrl("/api/events/7/snapshot.jpg", 2), "/api/events/7/snapshot.jpg?retry=2");
assert.equal(retryImageUrl("/a.jpg#crop", 1), "/a.jpg?retry=1#crop");
assert.equal(retryImageUrl("", 1), "");

// A stalled download is not complete; a broken one has no dimensions.
assert.equal(imageFinished({ complete: false, naturalWidth: 720 }), false);
assert.equal(imageFinished({ complete: true, naturalWidth: 0 }), false);
assert.equal(imageFinished({ complete: true, naturalWidth: 720 }), true);
assert.equal(imageFinished(null), false);

const viewport = { innerWidth: 390, innerHeight: 844 };
const at = (top, height = 200) => ({ isConnected: true, getBoundingClientRect: () => ({ top, bottom: top + height, left: 0, right: 360, width: 360, height }) });
assert.equal(imageVisible(at(100), viewport), true);
assert.equal(imageVisible(at(900), viewport), false, "a lazy image below the fold is not checked");
assert.equal(imageVisible(at(-300), viewport), false);
assert.equal(imageVisible({ ...at(100), isConnected: false }, viewport), false);

console.log("image retry tests passed");
