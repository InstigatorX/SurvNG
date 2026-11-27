import assert from "node:assert/strict";
import { INCIDENT_GALLERY_CHUNK, incidentGalleryColumns, incidentGalleryWanted, incidentFilterValues, keepIncidentIdentity, mergeIncidentHead, toggleIncidentFilterValue } from "../src/incidentNavigation.mjs";

// Preferences read from browser storage and range inputs arrive as strings.
for (const columns of [2, 3, 4, 5, 6, 7, 8]) {
  assert.equal(incidentGalleryColumns(columns), columns);
  assert.equal(incidentGalleryColumns(String(columns)), columns);
}
for (const invalid of [undefined, null, "", "bad", 0, 1, 9, -4, 100]) {
  assert.equal(incidentGalleryColumns(invalid), 4);
}

// Returning to the gallery loads far enough to show the open incident, and never shrinks.
assert.equal(incidentGalleryWanted(0), INCIDENT_GALLERY_CHUNK);
assert.equal(incidentGalleryWanted(INCIDENT_GALLERY_CHUNK - 1), INCIDENT_GALLERY_CHUNK);
assert.equal(incidentGalleryWanted(INCIDENT_GALLERY_CHUNK), 2 * INCIDENT_GALLERY_CHUNK);
assert.equal(incidentGalleryWanted(5, 3 * INCIDENT_GALLERY_CHUNK), 3 * INCIDENT_GALLERY_CHUNK);
assert.equal(incidentGalleryWanted(-1), INCIDENT_GALLERY_CHUNK);
assert.equal(incidentGalleryWanted("bad"), INCIDENT_GALLERY_CHUNK);
assert.ok(INCIDENT_GALLERY_CHUNK <= 100, "one request per chunk: the search endpoint caps a page at 100");

// Multi-value filters are kept as the comma-joined form the search endpoint reads.
assert.deepEqual(incidentFilterValues("all"), []);
assert.deepEqual(incidentFilterValues(""), []);
assert.deepEqual(incidentFilterValues("person"), ["person"]);
assert.deepEqual(incidentFilterValues("car, person,,car"), ["car", "person"]);
assert.equal(toggleIncidentFilterValue("all", "person"), "person");
assert.equal(toggleIncidentFilterValue("person", "car"), "car,person");
assert.equal(toggleIncidentFilterValue("car,person", "car"), "person");
assert.equal(toggleIncidentFilterValue("person", "person"), "all", "nothing checked means every object");

assert.equal(toggleIncidentFilterValue("all", 7), "7", "person ids arrive as numbers");
assert.equal(toggleIncidentFilterValue("7", 7), "all");

// A refresh re-reads only the newest incidents and folds them into what is shown.
const card = (id, revision = 1) => ({ id: `incident-${id}`, revision, labels: ["person"] });
const shown = [5, 4, 3, 2, 1].map((id) => card(id));
const sameAgain = keepIncidentIdentity(shown, [card(5), card(4, 2)]);
assert.equal(sameAgain[0], shown[0], "an unchanged incident keeps its object");
assert.notEqual(sameAgain[1], shown[1], "a changed incident is replaced");
const merged = mergeIncidentHead(shown, [card(7), card(6), card(5), card(4, 2)]);
assert.deepEqual(merged.map((item) => item.id), [7, 6, 5, 4, 3, 2, 1].map((id) => `incident-${id}`));
assert.equal(merged[2], shown[0]);
assert.equal(merged[3].revision, 2);
assert.equal(merged[4], shown[2], "incidents past the fresh read are untouched");
// An incident that left the fresh range (merged away, or filtered out) goes.
assert.deepEqual(mergeIncidentHead(shown, [card(5), card(3)]).map((item) => item.id), ["incident-5", "incident-3", "incident-2", "incident-1"]);
assert.equal(mergeIncidentHead(shown, [card(9), card(8)]), null, "no overlap: read everything again");
assert.deepEqual(mergeIncidentHead([], [card(2)]).map((item) => item.id), ["incident-2"]);
assert.deepEqual(mergeIncidentHead(shown, shown.map((item) => ({ ...item }))), shown);

console.log("incident gallery tests passed");
