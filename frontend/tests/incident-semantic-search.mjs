import assert from "node:assert/strict";
import { incidentPeakConfidence, keepListedIncidents, mapWithConcurrency, rankSemanticIncidentDetails, semanticIncidentRequest } from "../src/incidentSemanticSearch.mjs";

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
  { id: "a", zones: ["Road"], semantic_search: { score: 0.9, rank_score: 0.4 } },
  { id: "b", zones: ["Road"], semantic_search: { score: 0.8, rank_score: 0.8 } },
  { id: "a", zones: ["Road"], semantic_search: { score: 0.6, rank_score: 0.6 } },
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


// The confidence threshold applies to search matches by the list's rule.
const sure = { id: "a", zones: [], semantic_search: { score: .5 }, objects: [{ label: "person", confidence: .9 }, { label: "car", confidence: .4 }] };
const unsure = { id: "b", zones: [], semantic_search: { score: .9 }, scene_objects: [{ label: "person", confidence: .5, observations: [{ confidence: .62 }] }] };
const unknown = { id: "c", zones: [], semantic_search: { score: .7 }, objects: [{ label: "person" }] };
assert.equal(incidentPeakConfidence(sure), .9);
assert.equal(incidentPeakConfidence(sure, ["car"]), .4);
assert.equal(incidentPeakConfidence(unsure), .62, "an observation counts under its object's label");
assert.equal(incidentPeakConfidence(unsure, ["car"]), null);
assert.equal(incidentPeakConfidence(unknown), null);
assert.deepEqual(rankSemanticIncidentDetails([sure, unsure, unknown]).map((item) => item.id), ["b", "c", "a"]);
assert.deepEqual(rankSemanticIncidentDetails([sure, unsure, unknown], "all", { minConfidence: .6 }).map((item) => item.id), ["b", "a"]);
assert.deepEqual(rankSemanticIncidentDetails([sure, unsure, unknown], "all", { minConfidence: .8 }).map((item) => item.id), ["a"]);
assert.deepEqual(rankSemanticIncidentDetails([sure, unsure], "all", { minConfidence: .8, labels: ["car"] }).map((item) => item.id), []);


// Matches follow the list's review and people filters and carry its review marks.
const listed = new Map([["incident-a", "reviewed"], ["incident-b", null]]);
const kept = keepListedIncidents([{ id: "x", incident_id: "incident-a" }, { id: "incident-b" }, { id: "incident-c" }, null, {}], listed);
assert.deepEqual(kept.map((item) => [item.incident_id || item.id, item.review_verdict]), [["incident-a", "reviewed"], ["incident-b", null]]);
assert.deepEqual(keepListedIncidents(undefined, listed), []);
// With no filter to apply, a match the list did not reach is kept, without a mark.
assert.deepEqual(keepListedIncidents([{ id: "incident-a" }, { id: "incident-z" }, {}], listed, { keepUnlisted: true }).map((item) => item.id), ["incident-a", "incident-z"]);

console.log("incident semantic search tests passed");
