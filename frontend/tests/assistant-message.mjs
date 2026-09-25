import assert from "node:assert/strict";
import {
  assistantComposerPlaceholder,
  assistantCoachSeen,
  assistantThinkingStages,
  assistantWelcomeCopy,
  markAssistantCoachSeen,
  stripAssistantCitationMarkers,
} from "../src/assistantMessage.mjs";

assert.equal(
  stripAssistantCitationMarkers("All cameras are online [E-system]. Four need attention [E-system]."),
  "All cameras are online. Four need attention.",
);

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
