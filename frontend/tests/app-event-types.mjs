import assert from "node:assert/strict";

import { appEventTypesKey } from "../src/shared/appEventTypes.mjs";

// Incidents: the top bar's runtime state plus incident changes, no motion/object.
assert.equal(appEventTypesKey([["cameras_state", "camera_state", "system_state"], ["incident"]]), "camera_state,cameras_state,incident,system_state");
// Any listener that reads everything (Live, Timeline, Property) widens to all.
assert.equal(appEventTypesKey([["incident"], null]), "");
assert.equal(appEventTypesKey([]), "");
assert.equal(appEventTypesKey([["incident"], ["incident"]]), "incident");
console.log("app event type tests passed");
