// Recovery for image downloads that stall partway. Mobile Safari, especially
// behind a reverse proxy, can leave a request hanging with only the top of a
// JPEG drawn until the app is backgrounded and resumed.

/** How often a visible, unfinished image is checked. */
export const IMAGE_STALL_CHECK_MS = 4000;
/** Consecutive checks an image must stay visible and unfinished before it is re-requested. */
export const IMAGE_STALL_CHECKS = 2;
export const IMAGE_MAX_RETRIES = 2;

/** The same image URL with a retry marker so the browser starts a new request. */
export function retryImageUrl(src, attempt) {
  if (!src || !(attempt > 0)) return src || "";
  const [path, hash] = String(src).split("#", 2);
  const marked = `${path}${path.includes("?") ? "&" : "?"}retry=${attempt}`;
  return hash == null ? marked : `${marked}#${hash}`;
}

/** True once an image element has fully downloaded and decoded dimensions. */
export function imageFinished(image) {
  return Boolean(image?.complete && image.naturalWidth > 0);
}

/** True when any part of the element is inside the viewport. */
export function imageVisible(image, viewport = globalThis.window) {
  if (!image?.isConnected || !viewport) return false;
  const rect = image.getBoundingClientRect();
  return rect.bottom > 0 && rect.right > 0 && rect.top < viewport.innerHeight && rect.left < viewport.innerWidth && rect.width > 0 && rect.height > 0;
}
