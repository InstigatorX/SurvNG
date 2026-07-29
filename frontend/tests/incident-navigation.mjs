import assert from "node:assert/strict";
import { adjacentIncident, incidentIndexForEvent } from "../src/incidentNavigation.mjs";

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

console.log("incident navigation tests passed");
