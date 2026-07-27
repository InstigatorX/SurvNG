import assert from "node:assert/strict";
import {
  buildMotionDecisionFusion,
  MOTION_MODE_OPTIONS,
  motionModeInfo,
  motionValidatorSettings,
  readMotionDecisionFusion,
} from "../src/motionDecisionConfig.mjs";

assert.deepEqual(MOTION_MODE_OPTIONS.map((option) => option.value), ["camera", "adaptive"]);
assert.match(motionModeInfo("camera").description, /Only camera ONVIF notices/);
assert.match(motionModeInfo("adaptive").description, /ONVIF notices.*cannot trigger detection/);
assert.equal(motionModeInfo("unknown").value, "camera");
assert.equal(motionModeInfo("audit").value, "audit");
assert.match(motionModeInfo("enforce").description, /ambiguous hybrid/);

const defaults = readMotionDecisionFusion(undefined);
assert.equal(defaults.custom, false);
assert.equal(defaults.usesDefaults, true);
assert.equal(defaults.settings.policy, "audit");
assert.equal(defaults.settings.includePrimary, true);
assert.equal(defaults.settings.failOpen, true);

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
assert.equal(roundTrip.settings.includePrimary, true);
assert.equal(roundTrip.settings.failOpen, true);

const cameraEither = motionValidatorSettings(defaults.settings, {
  mode: "camera",
  adaptiveEnabled: true,
  mog2Enabled: true,
  agreement: "any",
});
assert.equal(cameraEither.policy, "any");
assert.deepEqual(cameraEither.sources, ["mog2"]);

const visualConfirmation = motionValidatorSettings(cameraEither, {
  mode: "adaptive",
  adaptiveEnabled: true,
  mog2Enabled: true,
  agreement: "any",
});
assert.equal(visualConfirmation.policy, "all");
assert.equal(visualConfirmation.includePrimary, true);

const cameraUnvalidated = motionValidatorSettings(defaults.settings, {
  mode: "camera",
  adaptiveEnabled: false,
  mog2Enabled: false,
});
assert.equal(cameraUnvalidated.policy, "bypass");
assert.equal(cameraUnvalidated.includePrimary, false);

const scalarGraph = buildMotionDecisionFusion({
  ...defaults.settings,
  policy: "all",
  sources: ["mog2"],
});
scalarGraph[0].options.sources = " MOG2 ";
scalarGraph[0].options.policy = " ALL ";
const scalarSource = readMotionDecisionFusion(scalarGraph);
assert.deepEqual(scalarSource.settings.sources, ["mog2"]);
assert.equal(scalarSource.settings.policy, "all");

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
