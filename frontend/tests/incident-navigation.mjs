import assert from "node:assert/strict";
import { adjacentIncident, createIncidentPageCache, incidentIndexForEvent, incidentThumbnailPageSize, incidentsNewestFirst, showIncidentCardAnnotations } from "../src/incidentNavigation.mjs";

const incidents = [
  { id: 100, events: [{ id: 101 }, { id: 102 }] },
  { id: 200, events: [{ id: 201 }] },
  { id: "300", events: [{ id: 301 }] },
];

assert.equal(incidentIndexForEvent(incidents, incidents[0]), 0);
assert.equal(incidentIndexForEvent(incidents, { id: 102 }), 0);
assert.equal(incidentIndexForEvent(incidents, { id: 999 }), -1);
assert.equal(adjacentIncident(incidents, incidents[0], 1), incidents[1]);
assert.equal(adjacentIncident(incidents, { id: 101 }, 1), incidents[1]);
assert.equal(adjacentIncident(incidents, { id: 201 }, -1), incidents[0]);
assert.equal(adjacentIncident(incidents, { id: 301 }, 1), incidents[0]);
assert.equal(adjacentIncident([incidents[0]], incidents[0], 1), null);

assert.equal(showIncidentCardAnnotations(false, true), true);
assert.equal(showIncidentCardAnnotations(false, false), false);
assert.equal(showIncidentCardAnnotations(true, true), false);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 500, density: "compact" }), 8);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 720, density: "compact" }), 12);
assert.equal(incidentThumbnailPageSize({ width: 334, height: 500, density: "comfortable" }), 2);
assert.equal(incidentThumbnailPageSize({ width: 400, height: 500, density: "compact", columns: 2, gap: 10, horizontalPadding: 24 }), 8);
assert.equal(incidentThumbnailPageSize({ width: 0, height: 0, density: "compact" }), 16);

const unorderedIncidents = [
  { id: "old", last_epoch: 100 },
  { id: "new", end_at: "2026-07-30T18:00:00Z" },
  { id: "middle", start_epoch: 200 },
  { id: "same-a", last_epoch: 150 },
  { id: "same-b", last_epoch: 150 },
];
const orderedIncidents = incidentsNewestFirst(unorderedIncidents);
assert.deepEqual(orderedIncidents.map((incident) => incident.id), ["new", "middle", "same-a", "same-b", "old"]);
assert.notEqual(orderedIncidents, unorderedIncidents);
assert.deepEqual(incidentsNewestFirst(null), []);

const loadedPages = [];
const pageCache = createIncidentPageCache(async (key) => {
  loadedPages.push(key);
  return { key };
});
const firstPending = pageCache.load("page-1");
assert.equal(pageCache.load("page-1"), firstPending);
assert.deepEqual(await firstPending, { key: "page-1" });
assert.deepEqual(pageCache.peek("page-1"), { key: "page-1" });
assert.deepEqual(loadedPages, ["page-1"]);
await pageCache.load("page-2");
await pageCache.load("page-3");
pageCache.retain(["page-2", "page-3"]);
assert.equal(pageCache.size(), 2);
await pageCache.load("page-1");
assert.deepEqual(loadedPages, ["page-1", "page-2", "page-3", "page-1"]);
pageCache.clear();
assert.equal(pageCache.size(), 0);
assert.equal(pageCache.peek("page-1"), undefined);

let retryAttempts = 0;
const retryingCache = createIncidentPageCache(async () => {
  retryAttempts += 1;
  if (retryAttempts === 1) throw new Error("temporary failure");
  return "recovered";
});
await assert.rejects(retryingCache.load("page"), /temporary failure/);
assert.equal(await retryingCache.load("page"), "recovered");
assert.equal(retryAttempts, 2);

console.log("incident navigation tests passed");
