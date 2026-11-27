import assert from "node:assert/strict";
import { INCIDENT_GALLERY_CHUNK, incidentGalleryColumns, incidentGalleryWanted } from "../src/incidentNavigation.mjs";

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

console.log("incident gallery tests passed");
