import assert from "node:assert/strict";
import {
  addSemanticSearchHistory,
  clearSemanticSearchSession,
  readSemanticSearchHistory,
  readSemanticSearchSession,
  SEMANTIC_SEARCH_HISTORY_KEY,
  SEMANTIC_SEARCH_SESSION_KEY,
  semanticSearchResultsForCamera,
  writeSemanticSearchHistory,
  writeSemanticSearchSession,
} from "../src/semanticSearchState.mjs";

class MemoryStorage {
  constructor() { this.values = new Map(); }
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { this.values.set(key, String(value)); }
  removeItem(key) { this.values.delete(key); }
}

const storage = new MemoryStorage();
let history = [];
for (let index = 1; index <= 6; index += 1) {
  history = addSemanticSearchHistory(history, { query: `query ${index}`, cameraId: index % 2 ? "gate" : "" });
}
assert.equal(history.length, 5);
assert.equal(history[0].query, "query 6");
assert.equal(history[4].query, "query 2");
history = addSemanticSearchHistory(history, { query: "QUERY 4", cameraId: "" });
assert.equal(history.length, 5);
assert.equal(history[0].query, "QUERY 4");
assert.equal(history.filter((entry) => entry.query.toLowerCase() === "query 4" && !entry.cameraId).length, 1);
assert.equal(writeSemanticSearchHistory(storage, history), true);
assert.deepEqual(readSemanticSearchHistory(storage), history);
assert.ok(storage.getItem(SEMANTIC_SEARCH_HISTORY_KEY));

const results = [{ score: 0.82, event: { id: 17, camera_id: "gate" } }];
const mixedResults = [...results, { score: 0.76, event: { id: 18, camera_id: "foyer" } }];
assert.deepEqual(semanticSearchResultsForCamera(mixedResults, "gate"), results);
assert.deepEqual(semanticSearchResultsForCamera(mixedResults, ""), mixedResults);
assert.equal(writeSemanticSearchSession(storage, { query: "white truck", cameraId: "gate", results }), true);
assert.deepEqual(readSemanticSearchSession(storage, "white truck", "gate")?.results, results);
assert.equal(readSemanticSearchSession(storage, "white truck", ""), null);
assert.equal(readSemanticSearchSession(storage, "red truck", "gate"), null);
assert.ok(storage.getItem(SEMANTIC_SEARCH_SESSION_KEY));
assert.equal(clearSemanticSearchSession(storage), true);
assert.equal(readSemanticSearchSession(storage, "white truck", "gate"), null);

storage.setItem(SEMANTIC_SEARCH_HISTORY_KEY, "invalid");
storage.setItem(SEMANTIC_SEARCH_SESSION_KEY, "invalid");
assert.deepEqual(readSemanticSearchHistory(storage), []);
assert.equal(readSemanticSearchSession(storage, "anything", ""), null);

console.log("semantic search state tests passed");
