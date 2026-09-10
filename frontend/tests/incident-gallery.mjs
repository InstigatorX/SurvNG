import assert from "node:assert/strict";
import { incidentGalleryPageSize } from "../src/incidentNavigation.mjs";

// Preferences read from browser storage and native selects arrive as strings.
for (const size of [25, 50, 100]) {
  assert.equal(incidentGalleryPageSize(size), size);
  assert.equal(incidentGalleryPageSize(String(size)), size);
}
for (const invalid of [undefined, null, "", "bad", 0, -25, 60, 1000, 50.5]) {
  assert.equal(incidentGalleryPageSize(invalid), 25);
}

console.log("incident gallery page-size tests passed");
