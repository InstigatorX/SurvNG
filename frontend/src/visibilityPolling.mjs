import { useEffect, useRef } from "react";

export function useVisiblePolling(callback, delayMs, enabled = true, { immediate = true, restartKey = "" } = {}) {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;
  useEffect(() => {
    if (!enabled || !Number.isFinite(Number(delayMs)) || Number(delayMs) <= 0) return undefined;
    let disposed = false;
    let inFlight = false;
    let controller = null;
    const run = () => {
      if (disposed || document.hidden || inFlight) return;
      inFlight = true;
      controller = new AbortController();
      Promise.resolve(callbackRef.current?.(controller.signal)).catch(() => {
        // Polling surfaces retain their last useful state and retry on the next interval.
      }).finally(() => {
        inFlight = false;
        controller = null;
      });
    };
    if (immediate) run();
    const timer = window.setInterval(run, Number(delayMs));
    const onVisibility = () => { if (!document.hidden) run(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      disposed = true;
      controller?.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [delayMs, enabled, immediate, restartKey]);
}
