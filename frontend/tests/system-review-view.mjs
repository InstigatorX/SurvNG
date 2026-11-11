import assert from "node:assert/strict";
import {
  autoStatusLabel,
  findingsByCamera,
  formatReviewValue,
  effectText,
  glanceSentence,
  outcomeLabel,
  phaseLabel,
  plainNote,
  verdictLabel,
} from "../src/systemReviewView.mjs";

assert.equal(formatReviewValue(0.76, "percent"), "76%");
assert.equal(formatReviewValue(1, "seconds"), "1 second");
assert.equal(formatReviewValue(1.5, "seconds"), "1.5 seconds");
assert.equal(formatReviewValue(2.5, "fps"), "2.5 per second");
assert.equal(formatReviewValue(640, "px"), "640 px wide");
assert.equal(formatReviewValue("inherit", "percent"), "Site setting");
assert.equal(formatReviewValue(false, "choice"), "Off");
assert.equal(formatReviewValue("high", "choice"), "High");
assert.equal(verdictLabel("likely_miss"), "Likely missed");
assert.equal(verdictLabel("likely_misclassification"), "Wrong label");
assert.equal(phaseLabel("watching"), "Watching");
assert.equal(phaseLabel("rolled_back"), "Rolled back");
assert.equal(outcomeLabel("worsened"), "Got worse");
assert.equal(autoStatusLabel("automatic"), "Applies automatically on the next weekly run");
assert.equal(autoStatusLabel("disabled"), "Applying is turned off in Server settings");
assert.equal(
  effectText({ current: 0.7, proposed: 0.76 }, { higher: "Fewer alerts.", lower: "More alerts." }),
  "Fewer alerts.",
);
assert.equal(
  effectText({ current: "balanced", proposed: "low" }, { higher: "More sensitive.", lower: "Less sensitive." }),
  "Less sensitive.",
);

assert.equal(
  glanceSentence({ analyzed: 12, cameras: 3, suggestions: 1 }),
  "Looked at 12 pictures across 3 cameras. 1 suggestion.",
);
assert.equal(
  glanceSentence({ analyzed: 0, cameras: 0, pass: "daily", summary: "Checked stored activity." }),
  "Checked stored activity.",
);

assert.equal(
  plainNote("motion.visual_backup_min_score stays manual.", {
    "motion.visual_backup_min_score": { label: "Backup alert confidence" },
    "motion.visual_backup_min": { label: "Should not match a shorter key first" },
  }),
  "Backup alert confidence stays manual.",
);

const groups = findingsByCamera({
  camera_summaries: [{ camera_id: "gate", camera_name: "Front Door" }],
  camera_reports: {
    gate: {
      samples: [
        { verdict: "consistent", summary: "Fine", image_url: "/api/a.jpg" },
        { verdict: "likely_false_alarm", summary: "Empty driveway", image_url: "/api/b.jpg" },
      ],
    },
  },
});
assert.equal(groups.length, 1);
assert.equal(groups[0].cameraName, "Front Door");
assert.equal(groups[0].notable, 1);
assert.equal(groups[0].samples[0].verdict, "likely_false_alarm");

console.log("system review view tests passed");
