export const LIVE_CAMERA_HOLD_PREVIEW_MS = 220;
export const LIVE_CAMERA_HOLD_MOVE_PX = 12;
export const LIVE_CAMERA_OVERLAY_MOTION_MS = 240;

export function shouldArmLiveCameraHoldPreview({ mobileView = false, pointerType = "" } = {}) {
  if (!mobileView) return false;
  const type = String(pointerType || "");
  return type === "touch" || type === "pen";
}

export function liveCameraHoldExceededMove(startX, startY, clientX, clientY, thresholdPx = LIVE_CAMERA_HOLD_MOVE_PX) {
  const dx = Number(clientX) - Number(startX);
  const dy = Number(clientY) - Number(startY);
  const limit = Math.max(0, Number(thresholdPx) || 0);
  return (dx * dx) + (dy * dy) > (limit * limit);
}

export function shouldSuppressLiveCameraOpenClick({ holdOpened = false, suppressClick = false } = {}) {
  return Boolean(holdOpened || suppressClick);
}
