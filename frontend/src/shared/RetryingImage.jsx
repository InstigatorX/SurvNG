import React, { useCallback, useEffect, useRef, useState } from "react";
import { IMAGE_MAX_RETRIES, IMAGE_STALL_CHECK_MS, IMAGE_STALL_CHECKS, imageFinished, imageVisible, retryImageUrl } from "./imageRetry.mjs";

/**
 * An <img> that stays hidden until it has fully loaded, so a partial download
 * never shows as a half-drawn picture, and re-requests the image when it
 * errors, stays unfinished while visible, or is unfinished when the page
 * becomes visible again.
 */
export function RetryingImage({ src, className = "", onLoad, onError, maxRetries = IMAGE_MAX_RETRIES, ...props }) {
  const imageRef = useRef(null);
  const [state, setState] = useState({ src, attempt: 0, loaded: false });
  const current = state.src === src ? state : { src, attempt: 0, loaded: false };
  if (current !== state) setState(current);
  const { attempt, loaded } = current;

  const retry = useCallback(() => {
    setState((value) => (value.src === src && !value.loaded && value.attempt < maxRetries
      ? { ...value, attempt: value.attempt + 1 }
      : value));
  }, [maxRetries, src]);

  const markLoaded = useCallback(() => {
    setState((value) => (value.src === src && !value.loaded ? { ...value, loaded: true } : value));
  }, [src]);

  const attachImage = useCallback((image) => {
    imageRef.current = image;
    // A cached image can finish before React attaches onLoad.
    if (imageFinished(image)) markLoaded();
  }, [markLoaded]);

  useEffect(() => {
    if (loaded || attempt >= maxRetries) return undefined;
    let stalledChecks = 0;
    const check = () => {
      const image = imageRef.current;
      if (imageFinished(image)) { markLoaded(); return; }
      stalledChecks = imageVisible(image) ? stalledChecks + 1 : 0;
      if (stalledChecks >= IMAGE_STALL_CHECKS) retry();
    };
    const onVisibility = () => {
      if (document.visibilityState !== "visible") return;
      const image = imageRef.current;
      if (imageFinished(image)) markLoaded();
      else if (imageVisible(image)) retry();
    };
    const timer = window.setInterval(check, IMAGE_STALL_CHECK_MS);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [attempt, loaded, markLoaded, maxRetries, retry]);

  return (
    <img
      {...props}
      ref={attachImage}
      src={retryImageUrl(src, attempt)}
      className={`${className} ${loaded ? "image-loaded" : "image-loading"}`.trim()}
      onLoad={(event) => { markLoaded(); onLoad?.(event); }}
      onError={(event) => {
        if (attempt < maxRetries) retry();
        else onError?.(event);
      }}
    />
  );
}
