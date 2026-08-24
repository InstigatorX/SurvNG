import assert from "node:assert/strict";
import {
  crossCameraMatchCameraLabel,
  crossCameraMatchLabel,
  crossCameraTracePath,
} from "../src/crossCameraTrace.mjs";

assert.equal(
  crossCameraTracePath(42),
  "/api/incidents/by-event/42/cross-camera-trace",
);
assert.equal(
  crossCameraTracePath(42, { time_zone: "America/Los_Angeles", limit: 8 }),
  "/api/incidents/by-event/42/cross-camera-trace?time_zone=America%2FLos_Angeles&limit=8",
);
assert.equal(
  crossCameraMatchLabel({ match_strength: "confirmed_identity" }),
  "Confirmed face",
);
assert.equal(
  crossCameraMatchLabel({ match_strength: "appearance_similarity", appearance_similarity: 0.91 }),
  "Visually similar 91%",
);
assert.equal(
  crossCameraMatchCameraLabel({ camera_id: "gate" }, new Map([["gate", "Front Gate"]])),
  "Front Gate",
);

console.log("cross-camera-trace tests passed");
