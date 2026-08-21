import assert from "node:assert/strict";
import {
  LIVE_CAMERA_HOLD_MOVE_PX,
  LIVE_CAMERA_HOLD_PREVIEW_MS,
  LIVE_CAMERA_OVERLAY_MOTION_MS,
  liveCameraHoldExceededMove,
  shouldArmLiveCameraHoldPreview,
  shouldSuppressLiveCameraOpenClick,
} from "../src/liveCameraHoldPreview.mjs";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

assert.equal(LIVE_CAMERA_HOLD_PREVIEW_MS, 220);
assert.equal(LIVE_CAMERA_HOLD_MOVE_PX, 12);
assert.equal(LIVE_CAMERA_OVERLAY_MOTION_MS, 240);

assert.equal(shouldArmLiveCameraHoldPreview({ mobileView: true, pointerType: "touch" }), true);
assert.equal(shouldArmLiveCameraHoldPreview({ mobileView: true, pointerType: "pen" }), true);
assert.equal(shouldArmLiveCameraHoldPreview({ mobileView: true, pointerType: "mouse" }), false);
assert.equal(shouldArmLiveCameraHoldPreview({ mobileView: false, pointerType: "touch" }), false);

assert.equal(liveCameraHoldExceededMove(0, 0, 5, 5), false);
assert.equal(liveCameraHoldExceededMove(0, 0, 13, 0), true);
assert.equal(liveCameraHoldExceededMove(10, 10, 10, 10 + LIVE_CAMERA_HOLD_MOVE_PX), false);
assert.equal(liveCameraHoldExceededMove(10, 10, 10, 10 + LIVE_CAMERA_HOLD_MOVE_PX + 1), true);

assert.equal(shouldSuppressLiveCameraOpenClick({ holdOpened: true }), true);
assert.equal(shouldSuppressLiveCameraOpenClick({ suppressClick: true }), true);
assert.equal(shouldSuppressLiveCameraOpenClick({ holdOpened: false, suppressClick: false }), false);

const directory = dirname(fileURLToPath(import.meta.url));
const livePageSource = readFileSync(join(directory, "../src/live/LivePage.jsx"), "utf8");
const styles = readFileSync(join(directory, "../src/styles.css"), "utf8");
const contract = readFileSync(join(directory, "../../docs/live-view-visual-contract.md"), "utf8");

assert.match(livePageSource, /shouldArmLiveCameraHoldPreview/);
assert.match(livePageSource, /onPreviewOpen/);
assert.match(livePageSource, /onPreviewClose/);
assert.match(livePageSource, /overlayMode/);
assert.match(styles, /\.live-overlay\[data-phase="open"\]/);
assert.match(styles, /\.live-overlay\[data-phase="closing"\]/);
assert.match(styles, /@keyframes live-overlay-panel-in/);
assert.match(styles, /-webkit-touch-callout:\s*none/);
assert.match(livePageSource, /onContextMenu=\{blockBrowserHoldMenu\}/);
assert.match(livePageSource, /selectstart/);
assert.match(contract, /Press-and-hold/);

console.log("live camera hold preview tests passed");
