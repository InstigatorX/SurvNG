import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../src/admin/AccessSettings.jsx", import.meta.url), "utf8");
const body = source.slice(source.indexOf("  async function saveTls("), source.indexOf("  async function generateCert("));
assert.match(source, /onClick=\{\(\) => void saveTls\(Boolean\(tls\?\.enabled\)\)\}/);
assert.match(source, /await saveTls\(Boolean\(tls\?\.enabled\), true\)/);

for (const mode of ["save", "restart", "reject-save", "reject-restart"]) {
  const requests = [], commits = [], errors = [];
  const context = vm.createContext({
    tlsHostname: "camera.example", tlsPort: 9443,
    setBusy() {}, setTls() {}, setError(error) { if (error) errors.push(error); },
    commitImmediateConfig(...args) { commits.push(args); },
    apiDetail: (payload, fallback) => payload.detail || fallback,
    fetch: async (url, options) => {
      requests.push({ url, ...options });
      const ok = !(mode === "reject-save" && url === "/api/tls") && !(mode === "reject-restart" && url === "/api/tls/apply");
      return { ok, json: async () => ok ? { enabled: true, hostname: "camera.example", port: 9443 } : { detail: "Rejected" } };
    },
  });
  vm.runInContext(body, context);
  await context.saveTls(true, mode !== "save");
  assert.deepEqual(JSON.parse(requests[0].body), { enabled: true, hostname: "camera.example", port: 9443 });
  assert.equal(requests[0].method, "PUT");
  if (mode === "reject-save") {
    assert.equal(requests.length, 1); // failure must never restart
    assert.equal(commits.length, 0);
  } else {
    assert.equal(commits[0][1].port, 9443);
    if (mode !== "save") assert.equal(requests[1].url, "/api/tls/apply");
  }
  assert.equal(errors.length, mode.startsWith("reject-") ? 1 : 0);
}
console.log("HTTPS settings persistence and restart ordering passed");
