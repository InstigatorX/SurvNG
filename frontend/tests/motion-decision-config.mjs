import assert from "node:assert/strict";
import {
  buildMotionDecisionFusion,
  MOTION_MODE_OPTIONS,
  motionModeInfo,
  readMotionDecisionFusion,
} from "../src/motionDecisionConfig.mjs";

assert.deepEqual(MOTION_MODE_OPTIONS.map((option) => option.value), ["enforce", "audit", "off"]);
assert.match(motionModeInfo("enforce").description, /start object detection/);
assert.match(motionModeInfo("audit").description, /visual analysis alone cannot create an event/);
assert.match(motionModeInfo("off").description, /Only camera ONVIF or manual notices/);
assert.equal(motionModeInfo("unknown").value, "audit");

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

const parallel = structuredClone(graph);
parallel[0].parallel_group = "signals";
assert.equal(readMotionDecisionFusion(parallel).custom, true);

const extended = structuredClone(graph);
extended[0].options.future_fusion_control = true;
assert.equal(readMotionDecisionFusion(extended).custom, true);

const alternateMinimum = structuredClone(graph);
alternateMinimum[0].options.minimum_sources = 2;
assert.equal(readMotionDecisionFusion(alternateMinimum).custom, true);

console.log("motion decision configuration tests passed");
