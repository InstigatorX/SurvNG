import assert from "node:assert/strict";
import { assistantContextLabel, assistantContextPrompts, canonicalAssistantPage, snapshotAssistantContext } from "../src/assistantContext.mjs";

assert.equal(canonicalAssistantPage("recordings"), "timeline");
assert.equal(canonicalAssistantPage("faces"), "people");
assert.equal(canonicalAssistantPage("config"), "admin");
assert.equal(canonicalAssistantPage("unknown"), "live");

const incident = snapshotAssistantContext({
  page: "incidents",
  camera_id: " front-door ",
  incident_event_id: "42",
  filters: { camera_name: "Front Door", ignored: null },
}, "America/New_York");
assert.equal(incident.camera_id, "front-door");
assert.equal(incident.incident_event_id, 42);
assert.equal(incident.time_zone, "America/New_York");
assert.deepEqual(incident.filters, { camera_name: "Front Door" });
assert.equal(assistantContextLabel(incident), "Incidents · Front Door · Event #42");
assert.deepEqual(assistantContextPrompts(incident).slice(0, 2), ["Analyze this incident", "Trace this incident across cameras"]);

assert.equal(assistantContextLabel({ page: "search", filters: { query: "white truck" } }), "Search · “white truck”");
assert.match(assistantContextPrompts({ page: "admin" })[0], /settings/i);

console.log("assistant context tests passed");
