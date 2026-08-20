import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadRuntimeState, RUNTIME_STATE_PATHS } from "../src/shared/runtimeState.mjs";

assert.deepEqual(RUNTIME_STATE_PATHS, [
  "/api/cameras",
  "/api/config",
  "/api/system/status",
]);
assert.equal(new Set(RUNTIME_STATE_PATHS).size, RUNTIME_STATE_PATHS.length);

const requested = [];
const partial = await loadRuntimeState(async (path) => {
  requested.push(path);
  if (path === "/api/config") throw new Error("configuration temporarily unavailable");
  if (path === "/api/cameras") {
    return { ok: true, json: async () => [{ id: "gate", running: true }] };
  }
  return { ok: true, json: async () => ({ lifecycle: "running" }) };
});
assert.deepEqual(requested, RUNTIME_STATE_PATHS);
assert.deepEqual(partial, {
  cameras: [{ id: "gate", running: true }],
  appConfig: null,
  system: { lifecycle: "running" },
});

const unavailable = await loadRuntimeState(async () => ({ ok: false, json: async () => ({}) }));
assert.deepEqual(unavailable, { cameras: null, appConfig: null, system: null });

const eventsSource = readFileSync(new URL("../src/shared/events.js", import.meta.url), "utf8");
const pollingSource = readFileSync(new URL("../src/shared/polling.js", import.meta.url), "utf8");
assert.match(eventsSource, /export function useAppEvents\(handler, enabled = true\)/);
assert.match(eventsSource, /enabled \? subscribeAppEvents/);
assert.match(pollingSource, /useAppEvents\([\s\S]*?}, enabled\);/);

console.log("runtime state request contract tests passed");
