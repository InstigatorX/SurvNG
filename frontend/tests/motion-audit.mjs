import assert from "node:assert/strict";
import { motionAuditRegions } from "../src/motionAudit.mjs";

assert.deepEqual(motionAuditRegions({ motion_regions: [[0.1, 0.2, 0.4, 0.7]] }), [[0.1, 0.2, 0.4, 0.7]]);
assert.deepEqual(motionAuditRegions({ motion_regions: [[-0.1, 0.2, 0.4, 0.7], [0.1, 0.2, 1.1, 0.7], [0.4, 0.2, 0.1, 0.7], ["bad", 0, 1, 1]] }), []);
assert.deepEqual(motionAuditRegions({}), []);

console.log("motion audit helpers passed");
