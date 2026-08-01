import assert from "node:assert/strict";
import {
  ASSISTANT_HISTORY_TTL_MS,
  readAssistantHistory,
  writeAssistantHistory,
} from "../src/assistantStorage.mjs";

const values = new Map();
const storage = {
  getItem: (key) => values.get(key) || null,
  setItem: (key, value) => values.set(key, value),
  removeItem: (key) => values.delete(key),
};

assert.equal(writeAssistantHistory(storage, "history", [{ role: "user", content: "hello" }], 1000), true);
assert.deepEqual(readAssistantHistory(storage, "history", 1001), [{ role: "user", content: "hello" }]);
assert.deepEqual(readAssistantHistory(storage, "history", 1000 + ASSISTANT_HISTORY_TTL_MS), []);
assert.equal(values.has("history"), false);

values.set("history", JSON.stringify([{ role: "user", content: "legacy never-expiring data" }]));
assert.deepEqual(readAssistantHistory(storage, "history", 2000), []);

console.log("assistant storage tests passed");
