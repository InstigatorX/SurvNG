import assert from "node:assert/strict";
import {
  assistantComposerPlaceholder,
  assistantCoachSeen,
  assistantEvidenceLabel,
  assistantThinkingStages,
  assistantWelcomeCopy,
  markAssistantCoachSeen,
  splitAssistantCitations,
  stripAssistantCitationMarkers,
} from "../src/assistantMessage.mjs";

assert.deepEqual(splitAssistantCitations(""), []);
assert.deepEqual(splitAssistantCitations("No citations here."), [
  { type: "text", value: "No citations here." },
]);
assert.deepEqual(splitAssistantCitations("Gate was busy [E1] and later quiet [E-activity]."), [
  { type: "text", value: "Gate was busy " },
  { type: "citation", value: "[E1]", evidenceId: "E1" },
  { type: "text", value: " and later quiet " },
  { type: "citation", value: "[E-activity]", evidenceId: "E-activity" },
  { type: "text", value: "." },
]);
assert.equal(
  stripAssistantCitationMarkers("All cameras are online [E-system]. Four need attention [E-system]."),
  "All cameras are online. Four need attention.",
);

assert.equal(assistantEvidenceLabel({ id: "E1", title: "Front Door" }), "Front Door");
assert.equal(assistantEvidenceLabel({ id: "E1" }), "E1");

const stages = assistantThinkingStages("Live · Front Door");
assert.equal(stages[0], "Looking at Front Door…");
assert.ok(stages.length >= 3);

const welcome = assistantWelcomeCopy("Incidents · Gate · Event #42");
assert.match(welcome.title, /look into/i);
assert.match(welcome.body, /Incidents · Gate/);
assert.match(assistantComposerPlaceholder("Live · Gate"), /Ask about Live · Gate/);

const storage = {
  data: {},
  getItem(key) { return this.data[key] ?? null; },
  setItem(key, value) { this.data[key] = String(value); },
};
assert.equal(assistantCoachSeen(storage), false);
markAssistantCoachSeen(storage);
assert.equal(assistantCoachSeen(storage), true);

console.log("assistant message tests passed");
