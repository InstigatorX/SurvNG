import assert from "node:assert/strict";
import {
  assistantEvidenceHref,
  assistantIncidentHref,
} from "../src/assistantNavigation.mjs";

assert.equal(assistantIncidentHref(42), "/incidents?event_ids=42");
assert.equal(assistantIncidentHref("43"), "/incidents?event_ids=43");
assert.equal(assistantIncidentHref(0), "");
assert.equal(assistantIncidentHref("invalid"), "");

assert.equal(
  assistantEvidenceHref({
    href: "/incidents?event_ids=42,43",
    image_url: "/api/events/42/thumbnail.jpg?width=960",
  }),
  "/incidents?event_ids=42,43",
);
assert.equal(
  assistantEvidenceHref({ image_url: "/api/events/44/thumbnail.jpg?width=960" }),
  "/incidents?event_ids=44",
);
assert.equal(
  assistantEvidenceHref({ image_url: "/api/motion-audit/5/snapshot.jpg" }),
  "/api/motion-audit/5/snapshot.jpg",
);
assert.equal(assistantEvidenceHref({ href: "https://untrusted.example" }), "");

console.log("assistant navigation tests passed");
