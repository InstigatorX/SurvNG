import assert from "node:assert/strict";

import { mutationNeedsCsrf, requestApiPath } from "../src/shared/requestPath.mjs";

const base = "/survng";

assert.equal(requestApiPath("/api/exports", base), "/api/exports");
assert.equal(requestApiPath("/survng/api/exports", base), "/api/exports");
assert.equal(requestApiPath("/survng/api/exports?limit=100", base), "/api/exports?limit=100");
assert.equal(requestApiPath("/survng-other/api/exports", base), "/survng-other/api/exports");
assert.equal(requestApiPath("https://nvr.test/api/exports", base), "");

assert.equal(mutationNeedsCsrf("/api/exports", "POST", base), true);
assert.equal(mutationNeedsCsrf("/survng/api/exports", "POST", base), true);
assert.equal(mutationNeedsCsrf("/survng/api/semantic-search/visual", "POST", base), true);
assert.equal(mutationNeedsCsrf("/survng/api/exports/job-1", "DELETE", base), true);
assert.equal(mutationNeedsCsrf("/survng/api/onvif-inspector/clear", "POST", base), true);
assert.equal(mutationNeedsCsrf("/survng/api/exports", "GET", base), false);
assert.equal(mutationNeedsCsrf("/survng/api/auth/login", "POST", base), false);
assert.equal(mutationNeedsCsrf("/survng/api/auth/bootstrap", "POST", base), false);
assert.equal(mutationNeedsCsrf("/api/auth/login?next=/", "POST", ""), false);
assert.equal(mutationNeedsCsrf("/survng/static/app.js", "POST", base), false);

const calls = [];
globalThis.window = {
  __SURVNG_BASE_PATH__: "/survng",
  location: { origin: "https://nvr.test", pathname: "/survng/timeline" },
  fetch: async (resource, options) => {
    calls.push({
      resource,
      method: options.method,
      csrf: options.headers.get("X-SurvNG-CSRF"),
    });
    return { status: 200 };
  },
};
window.self = window;
window.top = window;
globalThis.document = {
  documentElement: { dataset: {} },
  cookie: "survng_csrf=token-value",
};
const { fetch: apiFetch } = await import("../src/shared/api.js");

await apiFetch("/survng/api/exports", { method: "POST" });
await apiFetch("/api/exports/job-1", { method: "DELETE" });
await apiFetch("/api/auth/login", { method: "POST" });
await apiFetch("/survng/api/exports?limit=100");

assert.deepEqual(calls, [
  { resource: "/survng/api/exports", method: "POST", csrf: "token-value" },
  { resource: "/survng/api/exports/job-1", method: "DELETE", csrf: "token-value" },
  { resource: "/survng/api/auth/login", method: "POST", csrf: null },
  { resource: "/survng/api/exports?limit=100", method: undefined, csrf: null },
]);

console.log("csrf request path tests passed");
