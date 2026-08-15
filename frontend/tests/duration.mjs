import assert from "node:assert/strict";
import { formatServerUptime } from "../src/duration.mjs";

assert.equal(formatServerUptime(0), "Less than 1 min");
assert.equal(formatServerUptime(60), "1 min");
assert.equal(formatServerUptime(3660), "1 hour, 1 min");
assert.equal(formatServerUptime(8 * 86400 + 2 * 3600 + 3 * 60), "1 week, 1 day, 2 hours, 3 mins");
assert.equal(formatServerUptime(46 * 86400 + 4 * 3600 + 5 * 60), "1 month, 2 weeks, 2 days, 4 hours, 5 mins");

console.log("server uptime formatting tests passed");
