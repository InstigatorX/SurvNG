import assert from "node:assert/strict";
import {
  tuneupHistoryTitle,
  tuneupOutcome,
  tuneupRecommendationGroup,
  tuneupValue,
} from "../src/detectionTuneup.mjs";

assert.equal(tuneupValue("inherit"), "Use system default");
assert.equal(tuneupValue("high"), "High");
assert.equal(tuneupRecommendationGroup({ subsystem: "tracking" }), "Improve identification overlays");
assert.equal(tuneupRecommendationGroup({ setting: "motion.frame_width", current: 640, proposed: 480 }), "Reduce processing load");
assert.equal(tuneupRecommendationGroup({ setting: "motion.visual_backup_min_score", current: 0.6, proposed: 0.7 }), "Reduce unwanted motion");
assert.equal(tuneupRecommendationGroup({ setting: "motion.visual_backup_min_score", current: 0.7, proposed: 0.6 }), "Catch more important activity");
assert.deepEqual(tuneupOutcome({ evaluation: { outcome: "regressed" } }), ["Performance declined", "bad"]);
assert.equal(tuneupHistoryTitle({ mode: "standard", camera_ids: ["gate", "yard"] }, [{ id: "gate" }, { id: "yard" }]), "7 days review · All cameras");
assert.equal(tuneupHistoryTitle({ mode: "quick", camera_ids: ["gate"] }, [{ id: "gate", name: "Front Gate" }, { id: "yard" }]), "24 hours review · Front Gate");

console.log("detection tune-up workflow tests passed");
