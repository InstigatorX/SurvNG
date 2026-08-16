import assert from "node:assert/strict";
import { safeMediaUrl } from "../src/mediaUrl.mjs";

assert.equal(safeMediaUrl("/api/events/1/image", "/survng", "https://nvr.test"), "/survng/api/events/1/image");
assert.equal(safeMediaUrl("/survng/api/events/1/image", "/survng", "https://nvr.test"), "/survng/api/events/1/image");
assert.equal(safeMediaUrl("blob:https://nvr.test/id", "/survng", "https://nvr.test"), "blob:https://nvr.test/id");
assert.equal(safeMediaUrl("data:image/webp;base64,AA", "/survng", "https://nvr.test"), "data:image/webp;base64,AA");
assert.equal(safeMediaUrl("https://nvr.test/api/image", "/survng", "https://nvr.test"), "https://nvr.test/api/image");
assert.equal(safeMediaUrl("https://other.test/image", "/survng", "https://nvr.test"), "");
assert.equal(safeMediaUrl("//other.test/image", "/survng", "https://nvr.test"), "");

console.log("media URL tests passed");
