import assert from "node:assert/strict";
import { browserStorage, readStoredValue, writeStoredValue } from "../src/storage.mjs";

const values = new Map();
const storage = {
  getItem(key) { return values.get(key) ?? null; },
  setItem(key, value) { values.set(key, value); },
};

assert.equal(readStoredValue(storage, "missing", "fallback"), "fallback");
assert.equal(writeStoredValue(storage, "mode", "dark"), true);
assert.equal(readStoredValue(storage, "mode", "fallback"), "dark");
assert.equal(readStoredValue({ getItem() { throw new DOMException("denied"); } }, "mode", "auto"), "auto");
assert.equal(writeStoredValue({ setItem() { throw new DOMException("full"); } }, "mode", "dark"), false);
assert.equal(readStoredValue(null, "mode", "auto"), "auto");
assert.equal(browserStorage({ localStorage: storage }), storage);
assert.equal(browserStorage(Object.defineProperty({}, "localStorage", { get() { throw new DOMException("denied"); } })), null);

console.log("storage tests passed");
