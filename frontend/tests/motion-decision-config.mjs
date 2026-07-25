import assert from "node:assert/strict";
import {
  buildMotionDecisionFusion,
  readMotionDecisionFusion,
} from "../src/motionDecisionConfig.mjs";

const defaults = readMotionDecisionFusion(undefined);
assert.equal(defaults.custom, false);
assert.equal(defaults.usesDefaults, true);
assert.equal(defaults.settings.policy, "audit");

const graph = buildMotionDecisionFusion({
  ...defaults.settings,
  policy: "all",
  activationFrames: 2,
});
assert.equal(graph[0].implementation, "buffered_evidence_fusion");
assert.equal(graph[0].options.policy, "all");
assert.equal(graph[1].options.activation_frames, 2);
assert.equal(graph[2].implementation, "score_trigger");

const roundTrip = readMotionDecisionFusion(graph);
assert.equal(roundTrip.custom, false);
assert.equal(roundTrip.settings.policy, "all");
assert.equal(roundTrip.settings.activationFrames, 2);

const custom = readMotionDecisionFusion([
  { stage_id: "custom", implementation: "site_specific_fusion" },
]);
assert.equal(custom.custom, true);

console.log("motion decision configuration tests passed");
