import assert from "node:assert/strict";
import {
  availableQualificationPresets,
  motionAnalysisPresetSelectionUseful,
  presetQualificationGraph,
  readMotionAnalysisPreset,
} from "../src/motionAnalysisConfig.mjs";

const catalog = {
  presets: [
    {
      id: "modular",
      graph: "qualification",
      recommended: true,
      available: true,
      stages: [{ stage_id: "preprocess", implementation: "gray_blur", options: {} }],
    },
    {
      id: "classic",
      graph: "qualification",
      available: true,
      stages: [{ stage_id: "qualification", implementation: "legacy_qualifier", options: {} }],
    },
  ],
};

assert.equal(availableQualificationPresets(catalog).length, 2);
assert.equal(motionAnalysisPresetSelectionUseful(catalog), true);
assert.equal(motionAnalysisPresetSelectionUseful({ presets: [catalog.presets[0]] }), false);
assert.equal(motionAnalysisPresetSelectionUseful({ presets: [] }), false);
assert.equal(readMotionAnalysisPreset([], catalog).preset.id, "modular");
assert.equal(readMotionAnalysisPreset(null, catalog).inherited, true);
assert.equal(readMotionAnalysisPreset(catalog.presets[1].stages, catalog).preset.id, "classic");
assert.equal(readMotionAnalysisPreset([{ stage_id: "custom", implementation: "custom" }], catalog).custom, true);
assert.equal(readMotionAnalysisPreset([
  { ...catalog.presets[0].stages[0], parallel_group: "sources" },
], catalog).custom, true);

const copied = presetQualificationGraph(catalog.presets[0]);
copied[0].stage_id = "changed";
assert.equal(catalog.presets[0].stages[0].stage_id, "preprocess");

console.log("motion analysis configuration tests passed");
