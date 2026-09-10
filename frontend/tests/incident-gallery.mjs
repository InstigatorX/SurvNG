import assert from "node:assert/strict";
import { incidentGalleryPageSize } from "../src/incidentNavigation.mjs";

// Gallery capacity fits complete thumbnail rows and stays bounded. These
// dimensions include the results area only, below the toolbar and rail header.
assert.equal(incidentGalleryPageSize({ width: 1440, height: 720 }), 15);
assert.equal(incidentGalleryPageSize({ width: 1024, height: 640 }), 12);
assert.equal(incidentGalleryPageSize({ width: 768, height: 580 }), 6);
assert.equal(incidentGalleryPageSize({ width: 240, height: 100 }), 1);
assert.equal(incidentGalleryPageSize({ width: 7680, height: 4320 }), 60);
assert.equal(incidentGalleryPageSize({ width: 0, height: 0 }), 12);
assert.equal(incidentGalleryPageSize({}), 12);

console.log("incident gallery capacity tests passed");
