import assert from "node:assert/strict";
import { createConditionalJsonClient } from "../src/conditionalJson.mjs";

const requests = [];
const responses = [
  {
    status: 200,
    ok: true,
    headers: new Headers({ etag: '"people-1"' }),
    json: async () => [{ id: 1, name: "Ada" }],
  },
  {
    status: 304,
    ok: false,
    headers: new Headers({ etag: '"people-1"' }),
    json: async () => { throw new Error("304 responses have no body"); },
  },
  {
    status: 200,
    ok: true,
    headers: new Headers({ etag: '"people-2"' }),
    json: async () => [{ id: 1, name: "Ada" }, { id: 2, name: "Grace" }],
  },
];
const client = createConditionalJsonClient(async (url, options) => {
  requests.push({ url, options });
  return responses.shift();
});

const first = await client.get("/api/faces/people");
const notModified = await client.get("/api/faces/people");
const changed = await client.get("/api/faces/people");

assert.deepEqual(first, [{ id: 1, name: "Ada" }]);
assert.strictEqual(notModified, first);
assert.deepEqual(changed, [{ id: 1, name: "Ada" }, { id: 2, name: "Grace" }]);
assert.equal(requests[0].options.headers.has("If-None-Match"), false);
assert.equal(requests[1].options.headers.get("If-None-Match"), '"people-1"');
assert.equal(requests[2].options.headers.get("If-None-Match"), '"people-1"');

const emptyClient = createConditionalJsonClient(async () => ({ status: 304, ok: false }));
await assert.rejects(
  emptyClient.get("/api/faces/unknown-clusters", "Unable to load unknown clusters"),
  /cached response unavailable/,
);

console.log("conditional JSON tests passed");
