import assert from "node:assert/strict";
import {
  buildMotionDecisionFusion,
  MOTION_BEHAVIOR_OPTIONS,
  MOTION_MODE_OPTIONS,
  motionBehaviorOption,
  motionBehaviorSettings,
  motionBehaviorValue,
  motionModeInfo,
  motionValidatorSettings,
  readMotionDecisionFusion,
} from "../src/motionDecisionConfig.mjs";

assert.deepEqual(MOTION_MODE_OPTIONS.map((option) => option.value), ["camera", "camera_rescue", "adaptive"]);
assert.deepEqual(MOTION_BEHAVIOR_OPTIONS.map((option) => option.value), ["camera", "camera_validation", "camera_rescue", "adaptive"]);
assert.match(motionModeInfo("camera").description, /Only camera ONVIF notices/);
assert.match(motionModeInfo("camera_rescue").description, /Pays extra CPU for recall/);
assert.match(motionBehaviorOption("camera").description, /minimum extra CPU/);
assert.match(motionBehaviorOption("camera_validation").description, /cut noisy false notices/);
assert.match(motionBehaviorOption("camera_rescue").description, /does not filter ONVIF false positives/);
assert.match(motionModeInfo("adaptive").description, /ONVIF notices.*cannot trigger detection/);
assert.equal(motionModeInfo("unknown").value, "camera_rescue");
assert.equal(motionBehaviorOption("unknown").value, "camera_rescue");
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
});
assert.equal(graph[0].implementation, "buffered_evidence_fusion");
assert.equal(graph[0].options.policy, "all");
assert.equal(graph.length, 1);

const roundTrip = readMotionDecisionFusion(graph);
assert.equal(roundTrip.custom, false);
assert.equal(roundTrip.settings.policy, "all");
assert.equal(roundTrip.settings.includePrimary, true);
assert.equal(roundTrip.settings.failOpen, true);

const visualConfirmation = motionValidatorSettings(defaults.settings, {
  mode: "adaptive",
  adaptiveEnabled: true,
});
assert.equal(visualConfirmation.policy, "audit");
assert.equal(visualConfirmation.includePrimary, true);
assert.deepEqual(visualConfirmation.sources, []);

const cameraUnvalidated = motionValidatorSettings(defaults.settings, {
  mode: "camera",
  adaptiveEnabled: false,
});
assert.equal(cameraUnvalidated.policy, "bypass");
assert.equal(cameraUnvalidated.includePrimary, false);
assert.equal(motionBehaviorValue("camera", cameraUnvalidated), "camera");

const cameraValidated = motionBehaviorSettings(cameraUnvalidated, "camera_validation");
assert.equal(cameraValidated.mode, "camera");
assert.equal(cameraValidated.settings.policy, "audit");
assert.equal(cameraValidated.settings.includePrimary, true);
assert.equal(motionBehaviorValue(cameraValidated.mode, cameraValidated.settings), "camera_validation");

const rescue = motionValidatorSettings(cameraUnvalidated, {
  mode: "camera_rescue",
  adaptiveEnabled: false,
});
assert.equal(rescue.policy, "audit");
assert.equal(rescue.includePrimary, true);
assert.equal(motionBehaviorSettings(defaults.settings, "camera_rescue").mode, "camera_rescue");
assert.equal(motionBehaviorSettings(defaults.settings, "adaptive").mode, "adaptive");

const scalarGraph = buildMotionDecisionFusion({
  ...defaults.settings,
  policy: "all",
  sources: ["onvif"],
});
scalarGraph[0].options.sources = " ONVIF ";
scalarGraph[0].options.policy = " ALL ";
const scalarSource = readMotionDecisionFusion(scalarGraph);
assert.deepEqual(scalarSource.settings.sources, ["onvif"]);
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

const legacyThreeStage = [
  graph[0],
  { stage_id: "event_state", implementation: "score_event_state", options: {} },
  { stage_id: "trigger", implementation: "score_trigger", options: {} },
];
assert.equal(readMotionDecisionFusion(legacyThreeStage).custom, false);

console.log("motion decision configuration tests passed");
