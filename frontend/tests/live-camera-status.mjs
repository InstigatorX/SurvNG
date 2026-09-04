import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { detectionStatus, recordingStatus } from "../src/liveCameraStatus.mjs";

const enabled = { recording_configured: true, recording_enabled: true, detection_enabled: true };
const disabled = { recording_configured: true, recording_enabled: false, detection_enabled: false };

assert.deepEqual(recordingStatus(enabled), {
  kind: "recording", state: "on", shortLabel: "REC", symbol: "✓", label: "Recording on", title: "Recording is on",
});
assert.equal(recordingStatus(disabled).state, "off");
assert.equal(recordingStatus({ recording_configured: false }).state, "unavailable");
assert.equal(recordingStatus(enabled, { busy: true }).state, "busy");
assert.equal(recordingStatus(enabled, { error: "Request failed" }).state, "error");
assert.equal(detectionStatus(enabled).state, "on");
assert.equal(detectionStatus(disabled).state, "off");
assert.equal(detectionStatus(disabled, { busy: true }).label, "Detection updating");

const livePage = readFileSync(new URL("../src/live/LivePage.jsx", import.meta.url), "utf8");
const styles = readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const mobileStyles = readFileSync(new URL("../src/shell/mobile.css", import.meta.url), "utf8");

assert.match(livePage, /<CameraTileStatus status=\{currentRecordingStatus\}/);
assert.match(livePage, /<CameraTileStatus status=\{currentDetectionStatus\}/);
assert.match(livePage, /aria-label=\{status\.label\}/);
assert.match(livePage, /title=\{status\.title\}/);
assert.match(styles, /\.camera-tile-control-menu \.tile-control-button,[\s\S]*?color:\s*#e8f0f2/);
assert.match(styles, /\.camera-tile-status\.error/);
assert.match(styles, /\.camera-tile-status\s*\{[\s\S]*?pointer-events:\s*auto/);
assert.match(mobileStyles, /\.live-camera-grid \.camera-tile-status\s*\{[\s\S]*?width:\s*24px/);
assert.match(mobileStyles, /\.live-camera-grid \.camera-tile-status-label[\s\S]*?display:\s*none/);

console.log("live camera status tests passed");
