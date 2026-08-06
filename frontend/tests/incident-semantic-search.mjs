import assert from "node:assert/strict";
import { mapWithConcurrency, rankSemanticIncidentDetails, semanticIncidentRequest } from "../src/incidentSemanticSearch.mjs";

assert.deepEqual(semanticIncidentRequest({
  query: " white delivery truck ",
  cameraFilter: "gate",
  objectFilter: "truck",
  startAt: "start",
  endAt: "end",
}), {
  query: "white delivery truck",
  camera_ids: ["gate"],
  object_labels: ["truck"],
  start_at: "start",
  end_at: "end",
  limit: 60,
});

const ranked = rankSemanticIncidentDetails([
  { id: "a", zones: ["Road"], semantic_search: { score: 0.4 } },
  { id: "b", zones: ["Road"], semantic_search: { score: 0.8 } },
  { id: "a", zones: ["Road"], semantic_search: { score: 0.6 } },
  { id: "c", zones: ["Porch"], semantic_search: { score: 0.9 } },
], "Road");
assert.deepEqual(ranked.map((item) => [item.id, item.semantic_search.score]), [["b", 0.8], ["a", 0.6]]);

let active = 0;
let maximumActive = 0;
const mapped = await mapWithConcurrency([1, 2, 3, 4, 5], 2, async (value) => {
  active += 1;
  maximumActive = Math.max(maximumActive, active);
  await new Promise((resolve) => setTimeout(resolve, 2));
  active -= 1;
  return value * 2;
});
assert.deepEqual(mapped, [2, 4, 6, 8, 10]);
assert.equal(maximumActive, 2);

console.log("incident semantic search tests passed");
