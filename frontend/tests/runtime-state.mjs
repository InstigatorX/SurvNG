import assert from "node:assert/strict";
import { RUNTIME_STATE_PATHS } from "../src/shared/runtimeState.mjs";

assert.deepEqual(RUNTIME_STATE_PATHS, [
  "/api/cameras",
  "/api/config",
  "/api/system/status",
]);
assert.equal(new Set(RUNTIME_STATE_PATHS).size, RUNTIME_STATE_PATHS.length);

console.log("runtime state request contract tests passed");
