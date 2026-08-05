import assert from "node:assert/strict";
import { rankSemanticIncidentDetails, semanticIncidentRequest } from "../src/incidentSemanticSearch.mjs";

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

console.log("incident semantic search tests passed");
