import assert from "node:assert/strict";
import { liveSnapshotRefreshMs, logPayloadSignature } from "../src/pollingPolicy.mjs";

const base = { running: true, visible: true, documentVisible: true, streamReady: true, transport: "snapshot" };
assert.equal(liveSnapshotRefreshMs({ ...base, mobile: false, primary: false }), 2000);
assert.equal(liveSnapshotRefreshMs({ ...base, mobile: true, primary: true }), 2000);
assert.equal(liveSnapshotRefreshMs({ ...base, mobile: true, primary: false }), 12000);
assert.equal(liveSnapshotRefreshMs({ ...base, visible: false, mobile: true, primary: false }), null);
assert.equal(liveSnapshotRefreshMs({ ...base, documentVisible: false, mobile: false, primary: true }), null);
assert.equal(liveSnapshotRefreshMs({ ...base, transport: "webrtc", mobile: false, primary: true }), null);
assert.equal(logPayloadSignature([{ id: 1 }, { id: 2 }]), logPayloadSignature([{ id: 1 }, { id: 2 }]));
assert.notEqual(logPayloadSignature([{ id: 1 }]), logPayloadSignature([{ id: 2 }]));
assert.notEqual(
  logPayloadSignature([{ id: 1 }, { id: 2 }, { id: 3 }]),
  logPayloadSignature([{ id: 1 }, { id: 9 }, { id: 3 }]),
);

console.log("polling policy tests passed");
