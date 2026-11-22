import assert from "node:assert/strict";
import {
  cameraCaptureConnectivity,
  cameraConnectivityClass,
  cameraConnectivityLabel,
  cameraTileLiveState,
} from "../src/cameraConnectivity.mjs";

assert.equal(cameraCaptureConnectivity({ running: true, connected: true, capture_running: true }), "healthy");
assert.equal(cameraCaptureConnectivity({ running: true, connected: false, capture_running: true }), "reconnecting");
assert.equal(cameraCaptureConnectivity({ running: true, connected: false, capture_running: false }), "offline");
assert.equal(cameraCaptureConnectivity({ running: false, connected: false, capture_running: false }), "paused");
assert.equal(cameraCaptureConnectivity({ capture_connectivity: "reconnecting" }), "reconnecting");
assert.equal(cameraConnectivityLabel("reconnecting"), "Reconnecting");
assert.equal(cameraConnectivityClass("reconnecting"), "attention");
assert.equal(cameraTileLiveState({ running: true, connected: false, capture_running: true }), "RECON");
assert.equal(cameraCaptureConnectivity({ capture_connectivity: "asleep" }), "asleep");
assert.equal(cameraCaptureConnectivity({ capture_connectivity: "wake_failed" }), "wake_failed");
assert.equal(cameraConnectivityLabel("asleep"), "Asleep");
assert.equal(cameraConnectivityLabel("wake_failed"), "Wake failed");
assert.equal(cameraConnectivityClass("asleep"), "asleep");
assert.equal(cameraConnectivityClass("wake_failed"), "attention");
assert.equal(cameraTileLiveState({ capture_connectivity: "asleep" }), "ASLEEP");
assert.equal(cameraTileLiveState({ capture_connectivity: "wake_failed" }), "WAKE");

console.log("camera-connectivity tests passed");
