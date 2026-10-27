import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  LIVE_STREAM_IDLE_MS,
  livePresenceActive,
  liveStreamIdleLabel,
  nextLivePresence,
} from "../src/liveStreamPresence.mjs";

const now = 1_700_000_000_000;

assert.equal(LIVE_STREAM_IDLE_MS, 15 * 60 * 1000);
assert.equal(liveStreamIdleLabel(), "15 minutes");
assert.equal(liveStreamIdleLabel(60_000), "1 minute");

assert.equal(livePresenceActive({
  lastActivityAt: now - 1000,
  now,
  documentVisible: true,
}), true);
assert.equal(livePresenceActive({
  lastActivityAt: now - LIVE_STREAM_IDLE_MS + 1,
  now,
  documentVisible: true,
}), true);
assert.equal(livePresenceActive({
  lastActivityAt: now - LIVE_STREAM_IDLE_MS,
  now,
  documentVisible: true,
}), false);
assert.equal(livePresenceActive({
  lastActivityAt: now,
  now,
  documentVisible: false,
}), false);
assert.equal(livePresenceActive({
  lastActivityAt: Number.NaN,
  now,
  documentVisible: true,
}), false);

const fresh = nextLivePresence({
  lastActivityAt: now - 5000,
  now,
  documentVisible: true,
});
assert.equal(fresh.active, true);
assert.equal(fresh.documentVisible, true);
assert.equal(fresh.nextCheckMs, LIVE_STREAM_IDLE_MS - 5000);

const idle = nextLivePresence({
  lastActivityAt: now - LIVE_STREAM_IDLE_MS,
  now,
  documentVisible: true,
});
assert.equal(idle.active, false);
assert.equal(idle.nextCheckMs, null);

const hidden = nextLivePresence({
  lastActivityAt: now,
  now,
  documentVisible: false,
});
assert.equal(hidden.active, false);
assert.equal(hidden.nextCheckMs, null);

const directory = dirname(fileURLToPath(import.meta.url));
const livePage = readFileSync(join(directory, "../src/live/LivePage.jsx"), "utf8");
const presenceHook = readFileSync(join(directory, "../src/useLiveStreamPresence.mjs"), "utf8");

assert.match(livePage, /useLiveStreamPresence\(\)/);
assert.match(livePage, /userActive: presence\.active/);
assert.match(livePage, /data-live-paused=\{!presence\.active && presence\.documentVisible \? "true" : "false"\}/);
assert.match(livePage, /liveStreamActive \? \(/);
assert.match(livePage, /noteLiveStreamActivity/);
assert.match(livePage, /Live stream paused after \{liveStreamIdleLabel\(\)\}/);
assert.match(livePage, /Live view paused after \{liveStreamIdleLabel\(\)\}/);
assert.match(livePage, /\{liveStreamActive \? \([\s\S]*?<WebRtcLive/);
assert.match(presenceHook, /visibilitychange/);
assert.match(presenceHook, /pointermove/);
assert.match(presenceHook, /nextLivePresence/);

console.log("live stream presence tests passed");
